"""
diagnostics/error_analysis.py — visualizes misclassified test images.

For every test image the model got wrong, saves a side-by-side panel:
    [query]  |  [nearest train image of the WRONG predicted identity]  |  [nearest train image of the TRUE identity]

This answers "what did the model confuse this photo with, and how close was
the correct answer?" instead of just "it was wrong" — the qualitative-error
step already flagged as pending in the report (Lome, CZoo), generalized to
run over an entire test split.

Usage:
    python diagnostics/error_analysis.py --config config/ctai.yaml --out diagnostics/errors/ctai
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

import numpy as np
import yaml
import shutil
from PIL import Ima ge, ImageDraw
from sklearn.metrics import pairwise_distances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.manifest import load_manifest
from pipeline.dataset import ManifestFaceDataset
from pipeline.cache import EmbeddingCache
from pipeline.matching import ClosedSetMatcher


def build_real_transform(img_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # rgb channel normalization for ImageNet-pretrained models
    ])


def extract_split_with_paths(dataset: ManifestFaceDataset, cache: EmbeddingCache, extractor):
    """Same as run_pipeline.extract_split, but also returns filepaths so the
    actual images can be reloaded afterwards for the error panels."""
    from pipeline.embedding import extract_with_cache
    embeddings, identities, filepaths = [], [], []
    for idx in range(len(dataset)):
        image_tensor, identity, filepath = dataset[idx]
        emb = extract_with_cache(extractor, cache, filepath, image_tensor)
        embeddings.append(emb)
        identities.append(identity)
        filepaths.append(filepath)
    return np.array(embeddings), np.array(identities), np.array(filepaths)


def make_panel(image_root: str, query_path: str, wrong_path: str, correct_path: str,
               query_id: str, wrong_id: str, wrong_dist: float, correct_dist: float,
               out_path: str, thumb_size: int = 260) -> None:
    imgs = [Image.open(os.path.join(image_root, p)).convert("RGB").resize((thumb_size, thumb_size))
            for p in (query_path, wrong_path, correct_path)]
    labels = [f"Query (true: {query_id})",
              f"Predicted (wrong): {wrong_id}  d={wrong_dist:.3f}",
              f"True identity, nearest match  d={correct_dist:.3f}"]

    label_h = 34
    panel = Image.new("RGB", (thumb_size * 3, thumb_size + label_h), "white")
    draw = ImageDraw.Draw(panel)
    for i, (img, label) in enumerate(zip(imgs, labels)):
        panel.paste(img, (i * thumb_size, label_h))
        draw.text((i * thumb_size + 8, 8), label, fill="black")
    panel.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="diagnostics/errors")
    ap.add_argument("--max-errors", type=int, default=None,
                     help="Cap the number of panels generated (omit for all).")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    df = load_manifest(config["dataset"]["manifest_path"], image_root=config["dataset"]["image_root"])
    train_df, test_df = df[df.split == "train"], df[df.split == "test"]

    transform = build_real_transform(config["model"]["img_size"])
    from pipeline.embedding import MegaDescriptorExtractor
    extractor = MegaDescriptorExtractor(config["model"]["name"], device=config["model"]["device"])
    # Reuses the SAME cache dir as run_pipeline.py's real runs, so if you already
    # ran the real evaluation for this config, every embedding here is a cache
    # hit — no recomputation, no GPU time spent twice.
    cache = EmbeddingCache(config["cache"]["dir"], config["model"]["name"],
                            config["model"]["img_size"], enabled=config["cache"]["enabled"])

    train_ds = ManifestFaceDataset(train_df, config["dataset"]["image_root"], transform=transform)
    test_ds = ManifestFaceDataset(test_df, config["dataset"]["image_root"], transform=transform)

    train_emb, train_ids, train_paths = extract_split_with_paths(train_ds, cache, extractor)
    test_emb, test_ids, test_paths = extract_split_with_paths(test_ds, cache, extractor)

    matcher = ClosedSetMatcher(k_neighbors=config["evaluation"]["k_neighbors"],
                                metric=config["evaluation"]["metric"])
    matcher.fit(train_emb, train_ids)
    preds = matcher.predict(test_emb)

    wrong_idx = np.where(preds != test_ids)[0]
    print(f"{len(wrong_idx)} / {len(test_ids)} test images misclassified.")
    if args.max_errors:
        wrong_idx = wrong_idx[:args.max_errors]

    dist_matrix = pairwise_distances(test_emb, train_emb, metric=config["evaluation"]["metric"])

    rows = []
    for i in wrong_idx:
        true_id, wrong_id = test_ids[i], preds[i]

        wrong_mask = train_ids == wrong_id
        wrong_local = dist_matrix[i, wrong_mask].argmin()
        wrong_path = train_paths[wrong_mask][wrong_local]
        wrong_dist = dist_matrix[i, wrong_mask][wrong_local]

        correct_mask = train_ids == true_id
        correct_local = dist_matrix[i, correct_mask].argmin()
        correct_path = train_paths[correct_mask][correct_local]
        correct_dist = dist_matrix[i, correct_mask][correct_local]

        out_name = f"{true_id}_as_{wrong_id}_{i}.jpg".replace(" ", "_")
        make_panel(config["dataset"]["image_root"], test_paths[i], wrong_path, correct_path,
                   true_id, wrong_id, wrong_dist, correct_dist, os.path.join(args.out, out_name))

        rows.append({"true_identity": true_id, "predicted_identity": wrong_id,
                     "dist_to_wrong": float(wrong_dist), "dist_to_correct": float(correct_dist),
                     "panel": out_name})

    with open(os.path.join(args.out, "_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} error panels + summary.csv to {args.out}")


if __name__ == "__main__":
    main()