"""
diagnostics/error_analysis.py — visualizes misclassified test images, and
optionally a sample of correct ones too.

For every test image the model got wrong, saves a side-by-side panel:
    [query]  |  [nearest train image of the WRONG predicted identity]  |  [nearest train image of the TRUE identity]

This answers "what did the model confuse this photo with, and how close was
the correct answer?" instead of just "it was wrong" — the qualitative-error
step already flagged as pending in the report (Lome, CZoo), generalized to
run over an entire test split.

With --correct-examples N, also saves N randomly-sampled CORRECT-prediction
panels (query | nearest train image of that same identity), to show the
method working, not just failing — useful for a report that wants both.

Usage:
    python diagnostics/error_analysis.py --config config/ctai.yaml --out diagnostics/errors/ctai --correct-examples 6
"""
from __future__ import annotations
import argparse
import csv
import os
import shutil
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet RGB stats
    ])


def extract_split_with_paths(dataset: ManifestFaceDataset, cache: EmbeddingCache, extractor):
    """Same as run_pipeline.extract_split, but also returns filepaths so the
    actual images can be reloaded afterwards for the panels."""
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


def make_correct_panel(image_root: str, query_path: str, match_path: str, identity: str,
                        dist: float, out_path: str, thumb_size: int = 260) -> None:
    """Same visual format as make_panel, but for a CORRECT prediction: shows
    the query next to the nearest train image of that same (correctly
    predicted) identity, so the panel demonstrates a hit, not a miss.
    """
    imgs = [Image.open(os.path.join(image_root, p)).convert("RGB").resize((thumb_size, thumb_size))
            for p in (query_path, match_path)]
    labels = [f"Query (true: {identity})",
              f"Predicted (correct): {identity}  d={dist:.3f}"]

    label_h = 34
    panel = Image.new("RGB", (thumb_size * 2, thumb_size + label_h), "white")
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
                     help="Cap the number of error panels generated (omit for all).")
    ap.add_argument("--correct-examples", type=int, default=0,
                     help="Also save this many CORRECT-prediction panels (randomly sampled), "
                          "to show successful cases alongside errors.")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Start from a clean output directory every run, so stale panels from a
    # previous (possibly differently-configured, e.g. pre-cache-fix) run
    # never linger alongside new ones.
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    errors_dir = os.path.join(args.out, "errors")
    correct_dir = os.path.join(args.out, "correct")
    os.makedirs(errors_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)

    df = load_manifest(config["dataset"]["manifest_path"], image_root=config["dataset"]["image_root"])
    train_df, test_df = df[df.split == "train"], df[df.split == "test"]

    transform = build_real_transform(config["model"]["img_size"])
    model_type = config["model"].get("type", "megadescriptor")
    if model_type == "resnet50":
        from pipeline.embedding import ResNet50Extractor
        extractor = ResNet50Extractor(config["model"]["name"], device=config["model"]["device"])
    else:
        from pipeline.embedding import MegaDescriptorExtractor
        extractor = MegaDescriptorExtractor(config["model"]["name"], device=config["model"]["device"])
    
    # Reuses the SAME cache dir as run_pipeline.py's real runs, so if you already
    # ran the real evaluation for this config, every embedding here is a cache
    # hit — no recomputation, no GPU time spent twice.
    cache = EmbeddingCache(config["cache"]["dir"], config["model"]["name"], config["model"]["img_size"],
                            dataset_name=config["dataset"]["name"], enabled=config["cache"]["enabled"])

    train_ds = ManifestFaceDataset(train_df, config["dataset"]["image_root"], transform=transform)
    test_ds = ManifestFaceDataset(test_df, config["dataset"]["image_root"], transform=transform)

    train_emb, train_ids, train_paths = extract_split_with_paths(train_ds, cache, extractor)
    test_emb, test_ids, test_paths = extract_split_with_paths(test_ds, cache, extractor)

    matcher = ClosedSetMatcher(k_neighbors=config["evaluation"]["k_neighbors"],
                                metric=config["evaluation"]["metric"])
    matcher.fit(train_emb, train_ids)
    preds = matcher.predict(test_emb)

    dist_matrix = pairwise_distances(test_emb, train_emb, metric=config["evaluation"]["metric"])

    # --- Errors ---
    wrong_idx = np.where(preds != test_ids)[0]
    print(f"{len(wrong_idx)} / {len(test_ids)} test images misclassified.")
    if args.max_errors:
        wrong_idx = wrong_idx[:args.max_errors]

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
                   true_id, wrong_id, wrong_dist, correct_dist, os.path.join(errors_dir, out_name))

        rows.append({"true_identity": true_id, "predicted_identity": wrong_id,
                     "dist_to_wrong": float(wrong_dist), "dist_to_correct": float(correct_dist),
                     "panel": out_name})

    if rows:
        with open(os.path.join(errors_dir, "_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} error panels + summary.csv to {errors_dir}")

    # --- Correct examples (optional) ---
    if args.correct_examples > 0:
        correct_idx = np.where(preds == test_ids)[0]
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(correct_idx, size=min(args.correct_examples, len(correct_idx)), replace=False)
        print(f"Saving {len(sample_idx)} correct-prediction examples (out of {len(correct_idx)} correct)...")

        correct_rows = []
        for i in sample_idx:
            identity = test_ids[i]
            same_mask = train_ids == identity
            local = dist_matrix[i, same_mask].argmin()
            match_path = train_paths[same_mask][local]
            dist = dist_matrix[i, same_mask][local]

            out_name = f"CORRECT_{identity}_{i}.jpg".replace(" ", "_")
            make_correct_panel(config["dataset"]["image_root"], test_paths[i], match_path,
                                identity, dist, os.path.join(correct_dir, out_name))
            correct_rows.append({"identity": identity, "dist_to_match": float(dist), "panel": out_name})

        if correct_rows:
            with open(os.path.join(correct_dir, "_correct_summary.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=correct_rows[0].keys())
                writer.writeheader()
                writer.writerows(correct_rows)
        print(f"Wrote {len(correct_rows)} correct-prediction panels + _correct_summary.csv to {correct_dir}")


if __name__ == "__main__":
    main()