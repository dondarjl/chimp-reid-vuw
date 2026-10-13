"""
Fine-tune an ImageNet ResNet-50 on the TRAIN split of a manifest CSV and evaluate it
with the same closed-set protocol used for MegaDescriptor and the zero-shot ResNet-50:

    embed train images (gallery) + test images -> L2-normalise -> cosine similarity
    -> identities ranked by their best-matching gallery image -> top-1 / top-5 accuracy

Design rules (so the comparison is fair and the test split stays clean):
  * Only rows with split == "train" are used for training.
  * Hyperparameters are fixed in advance. There is NO model selection on the test split:
    the model after the last epoch is the one evaluated.
  * Evaluation preprocessing (resize to 224x224 + ImageNet normalisation) is identical to
    the zero-shot ResNet-50, so `--epochs 0` should reproduce the zero-shot numbers of
    Table VII (CZoo 57.92 / 85.82, CTai 51.94 / 80.02), up to small resize differences.
    Run that first as a sanity check of this script.
  * No embedding cache is used (avoids the cross-dataset cache collision of finding #4).

Assumes pre-cropped face images (CZoo / CTai). Manifest needs a path, identity and split
column (names configurable below).

Usage (PowerShell, from the chimp_reid folder):
  python finetune_resnet50.py --manifest manifests/czoo.csv --dataset czoo --out-dir results/resnet50_ft
  python finetune_resnet50.py --manifest manifests/ctai.csv --dataset ctai --out-dir results/resnet50_ft
  python finetune_resnet50.py --manifest manifests/czoo.csv --dataset czoo --epochs 0   # sanity check
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T


# ----------------------------------------------------------------------------- matching
def rank_identities(gallery, gallery_labels, queries):
    """Rank identities for each query by the cosine similarity of their best gallery image.

    gallery, queries: L2-normalised arrays (n, d). gallery_labels: array of identity names.
    Returns (classes, order) where order[i] lists class indices from best to worst for query i.
    Mirrors the fixed ClosedSetMatcher ranking (min distance per identity, independent of k).
    """
    classes, codes = np.unique(np.asarray(gallery_labels), return_inverse=True)
    sort_idx = np.argsort(codes, kind="stable")
    sims = queries @ gallery[sort_idx].T                      # (n_query, n_gallery_sorted)
    sorted_codes = codes[sort_idx]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    best_per_class = np.maximum.reduceat(sims, starts, axis=1)  # (n_query, n_classes)
    order = np.argsort(-best_per_class, axis=1, kind="stable")
    return classes, order


def evaluate(gallery, gallery_labels, queries, query_labels, topk=5):
    classes, order = rank_identities(gallery, gallery_labels, queries)
    query_labels = np.asarray(query_labels)
    ranked_names = classes[order]                              # (n_query, n_classes)
    k = min(topk, ranked_names.shape[1])
    top1 = ranked_names[:, 0] == query_labels
    topk_hit = (ranked_names[:, :k] == query_labels[:, None]).any(axis=1)
    return top1, topk_hit, ranked_names[:, 0]


# ----------------------------------------------------------------------------- training
def batch_hard_triplet_loss(emb, labels, margin):
    d = torch.cdist(emb, emb)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos_mask = same & ~eye
    neg_mask = ~same
    hardest_pos = (d * pos_mask).max(dim=1).values
    hardest_neg = d.masked_fill(~neg_mask, float("inf")).min(dim=1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


class PKBatchSampler:
    """Each batch: P identities x K images (images drawn with replacement if an identity has < K)."""

    def __init__(self, labels, p, k, n_batches, seed):
        self.by_id = defaultdict(list)
        for i, y in enumerate(labels):
            self.by_id[int(y)].append(i)
        self.p = min(p, len(self.by_id))
        self.k, self.n_batches = k, n_batches
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        ids, order = list(self.by_id), []
        for _ in range(self.n_batches):
            if len(order) < self.p:
                perm = [i for i in ids if i not in order]
                self.rng.shuffle(perm)
                order += perm
            chosen, order = order[: self.p], order[self.p:]
            batch = []
            for c in chosen:
                pool = self.by_id[c]
                batch += self.rng.sample(pool, self.k) if len(pool) >= self.k else self.rng.choices(pool, k=self.k)
            yield batch


class ImgDS(Dataset):
    """Module-level (not nested in main) so DataLoader workers can pickle it on Windows."""

    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = list(paths), list(labels), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True, help="name used for the output folder, e.g. czoo / ctai")
    ap.add_argument("--out-dir", default="results/resnet50_ft")
    ap.add_argument("--image-root", default="", help="prefix for relative image paths, if needed")
    ap.add_argument("--col-path", default="filepath")
    ap.add_argument("--col-id", default="identity")
    ap.add_argument("--col-split", default="split")
    ap.add_argument("--epochs", type=int, default=30, help="0 = zero-shot sanity check")
    ap.add_argument("--p", type=int, default=16, help="identities per batch")
    ap.add_argument("--k", type=int, default=4, help="images per identity per batch")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4, help="use 0 if DataLoader workers cause trouble on Windows")
    ap.add_argument("--save-ckpt", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data
    df = pd.read_csv(args.manifest)
    root = Path(args.image_root) if args.image_root else None

    def resolve(p):
        p = Path(p)
        return str(p if (p.exists() or root is None) else root / p)

    df["_path"] = df[args.col_path].map(resolve)
    train = df[df[args.col_split] == "train"].reset_index(drop=True)
    test = df[df[args.col_split] == "test"].reset_index(drop=True)
    id_to_code = {n: i for i, n in enumerate(sorted(train[args.col_id].unique()))}
    missing = set(test[args.col_id]) - set(id_to_code)
    if missing:
        raise SystemExit(f"Test identities with no training images (fix the split): {sorted(missing)}")
    print(f"{args.dataset}: {len(train)} train / {len(test)} test, {len(id_to_code)} identities, device={device}")

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    eval_tf = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor(), norm])
    train_tf = T.Compose([
        T.RandomResizedCrop(args.img_size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(), norm,
    ])

    # ---- model: ImageNet ResNet-50, head removed -> 2048-d embedding (same as the zero-shot baseline)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.to(device)

    history = []
    if args.epochs > 0:
        y_train = train[args.col_id].map(id_to_code).to_numpy()
        n_batches = math.ceil(len(train) / (args.p * args.k))
        sampler = PKBatchSampler(y_train, args.p, args.k, n_batches, args.seed)
        loader = DataLoader(ImgDS(train["_path"], y_train, train_tf), batch_sampler=sampler,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * n_batches)
        for ep in range(args.epochs):
            model.train()
            losses = []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                emb = F.normalize(model(x), dim=1)
                loss = batch_hard_triplet_loss(emb, y, args.margin)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
            history.append(float(np.mean(losses)))
            print(f"epoch {ep + 1}/{args.epochs}  triplet loss {history[-1]:.4f}")

    # ---- evaluation (test split touched only here, once)
    @torch.no_grad()
    def embed(paths):
        model.eval()
        loader = DataLoader(ImgDS(paths, [0] * len(paths), eval_tf), batch_size=128,
                            num_workers=args.num_workers)
        out = [F.normalize(model(x.to(device)), dim=1).cpu() for x, _ in loader]
        return torch.cat(out).numpy()

    gal = embed(train["_path"])
    qry = embed(test["_path"])
    top1, top5, pred = evaluate(gal, train[args.col_id].to_numpy(), qry, test[args.col_id].to_numpy())

    result = {
        "dataset": args.dataset, "model": "resnet50_finetuned" if args.epochs > 0 else "resnet50_zeroshot",
        "top1": float(top1.mean()), "top5": float(top5.mean()),
        "n_train": len(train), "n_test": len(test), "n_identities": len(id_to_code),
        "n_test_errors": int((~top1).sum()), "hyperparameters": vars(args), "train_loss_per_epoch": history,
    }
    out = Path(args.out_dir) / f"{args.dataset}_ep{args.epochs}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    per_id = (pd.DataFrame({"identity": test[args.col_id], "correct": top1})
              .groupby("identity")["correct"].agg(n_test="size", n_correct="sum").reset_index())
    per_id["accuracy"] = per_id["n_correct"] / per_id["n_test"]
    per_id.sort_values("accuracy").to_csv(out / "per_identity_accuracy.csv", index=False)
    pd.DataFrame({"filepath": test[args.col_path], "true": test[args.col_id], "pred": pred,
                  "correct": top1}).to_csv(out / "predictions.csv", index=False)
    if args.epochs > 0 and args.save_ckpt:
        torch.save(model.state_dict(), out / "resnet50_ft.pt")
    print(f"\n{args.dataset} [{result['model']}]  top-1 {100 * result['top1']:.2f}%  top-5 {100 * result['top5']:.2f}%"
          f"  ({result['n_test_errors']}/{len(test)} errors)\nSaved to {out}")


if __name__ == "__main__":
    main()