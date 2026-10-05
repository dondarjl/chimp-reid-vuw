"""
run_pipeline.py — single entrypoint that ties every module together.

    python run_pipeline.py --config config/czoo.yaml            # real run (needs GPU/Colab)
    python run_pipeline.py --config config/czoo.yaml --dry-run  # architecture sanity check,
                                                                 # runs anywhere, no torch needed

--dry-run swaps MegaDescriptorExtractor for a seeded random-vector stand-in but
otherwise runs the REAL manifest loading, REAL image loading + detection/crop,
REAL kNN matching, and REAL logging — so a green dry-run means the only
untested part left is the model's forward pass itself, not the plumbing
around it. Use it after writing adapters/wellington_to_manifest.py, before
spending Colab GPU time on the real thing.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from pipeline.manifest import load_manifest, summarize
from pipeline.dataset import ManifestFaceDataset
from pipeline.cache import EmbeddingCache
from pipeline.matching import ClosedSetMatcher
from pipeline.evaluation import summarize_run, per_identity_accuracy


class _FakeExtractor:
    """Used only under --dry-run. Deterministic per-identity latent + noise,
    like the integration check already run manually — NOT a real accuracy
    estimate, purely a plumbing check. See docstring above.
    """
    def __init__(self, seed: int = 42, dim: int = 128):
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self._latent: dict[str, np.ndarray] = {}

    def extract_one_for_identity(self, identity: str) -> np.ndarray:
        if identity not in self._latent:
            self._latent[identity] = self.rng.normal(size=self.dim)
        return self._latent[identity] + self.rng.normal(scale=0.5, size=self.dim)


def build_real_transform(img_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_split(dataset: ManifestFaceDataset, cache: EmbeddingCache, dry_run: bool,
                   extractor=None, fake_extractor: _FakeExtractor | None = None):
    embeddings, identities = [], []
    n = len(dataset)
    for idx in range(n):
        if dry_run:
            _, identity, filepath = dataset.get_pil_crop(idx)
            cached = cache.get(filepath)
            if cached is not None:
                emb = cached
            else:
                emb = fake_extractor.extract_one_for_identity(identity)
                cache.set(filepath, emb)
        else:
            from pipeline.embedding import extract_with_cache
            image_tensor, identity, filepath = dataset[idx]
            emb = extract_with_cache(extractor, cache, filepath, image_tensor)
        embeddings.append(emb)
        identities.append(identity)
        if (idx + 1) % 200 == 0 or idx + 1 == n:
            print(f"  {idx + 1}/{n} embeddings extracted", flush=True)
    return np.array(embeddings), np.array(identities)


def log_result(results_log: str, run_name: str, config: dict, metrics: dict, n_train: int,
                n_test: int, n_identities: int, dry_run: bool) -> None:
    os.makedirs(os.path.dirname(results_log), exist_ok=True)
    is_new = not os.path.exists(results_log)
    with open(results_log, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "run_name", "dataset", "model", "dry_run",
                              "n_train", "n_test", "n_identities", "top_1_accuracy",
                              "top_5_accuracy", "notes"])
        writer.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            run_name, config["dataset"]["name"], config["model"]["name"], dry_run,
            n_train, n_test, n_identities,
            f"{metrics.get('top_1_accuracy', ''):.4f}" if 'top_1_accuracy' in metrics else "",
            f"{metrics.get('top_5_accuracy', ''):.4f}" if 'top_5_accuracy' in metrics else "",
            config["output"].get("notes", ""),
        ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true",
                     help="Validate the full pipeline with fake embeddings, no torch/GPU needed.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"=== {config['output']['run_name']} (dry_run={args.dry_run}) ===\n")

    df = load_manifest(config["dataset"]["manifest_path"], image_root=config["dataset"]["image_root"])
    print(summarize(df), "\n")

    train_df = df[df.split == "train"]
    test_df = df[df.split == "test"]

    # Dry-run gets its OWN cache subfolder, never the one real runs use — this
    # is the fix for a real bug found during development: since a fresh
    # EmbeddingCache only learns its expected dimension from the first vector
    # it sees, a dry-run's fake embeddings sitting in the same folder as a
    # real run's cache would be accepted as valid on the very first cache
    # hit, silently short-circuiting the real model entirely. Separate
    # folders make this collision structurally impossible instead of
    # relying on a dimension check to catch it after the fact.
    cache_dir = config["cache"]["dir"]
    if args.dry_run:
        cache_dir = os.path.join(cache_dir, "_dryrun_fake_embeddings")
        print("⚠️  DRY RUN: using fake embeddings in a separate cache folder — "
              "these results are a plumbing check, NOT real accuracy.\n")
    cache = EmbeddingCache(cache_dir, config["model"]["name"], config["model"]["img_size"],
                        dataset_name=config["dataset"]["name"], enabled=config["cache"]["enabled"])
    print("Cache status:", cache.stats(df["filepath"].tolist()), "\n")

    if args.dry_run:
        transform = None
        extractor = None
        fake_extractor = _FakeExtractor()
    else:
        transform = build_real_transform(config["model"]["img_size"])
        model_type = config["model"].get("type", "megadescriptor")
        if model_type == "resnet50":
            from pipeline.embedding import ResNet50Extractor
            extractor = ResNet50Extractor(config["model"]["name"], device=config["model"]["device"])
        else:
            from pipeline.embedding import MegaDescriptorExtractor
            extractor = MegaDescriptorExtractor(config["model"]["name"], device=config["model"]["device"])
        fake_extractor = None

    train_ds = ManifestFaceDataset(train_df, config["dataset"]["image_root"], transform=transform)
    test_ds = ManifestFaceDataset(test_df, config["dataset"]["image_root"], transform=transform)

    train_emb, train_ids = extract_split(train_ds, cache, args.dry_run, extractor, fake_extractor)
    test_emb, test_ids = extract_split(test_ds, cache, args.dry_run, extractor, fake_extractor)

    matcher = ClosedSetMatcher(k_neighbors=config["evaluation"]["k_neighbors"],
                                metric=config["evaluation"]["metric"])
    matcher.fit(train_emb, train_ids)
    preds = matcher.predict(test_emb)
    ranked_ids, _ = matcher.predict_proba_ranked(test_emb)

    metrics = summarize_run(test_ids, preds, ranked_ids, k_values=tuple(config["evaluation"]["top_k_values"]))
    print("Metrics:", metrics, "\n")
    print("Per-identity accuracy (worst first):")
    print(per_identity_accuracy(test_ids, preds).head(10), "\n")

    log_result(config["output"]["results_log"], config["output"]["run_name"], config,
               metrics, len(train_df), len(test_df), df["identity"].nunique(), args.dry_run)
    print(f"Logged to {config['output']['results_log']}")


if __name__ == "__main__":
    main()
