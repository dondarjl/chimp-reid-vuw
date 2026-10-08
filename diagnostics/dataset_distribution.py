"""
diagnostics/dataset_distribution.py — per-identity image-count distribution,
rank-ordered, for each dataset, saved as one PNG per dataset per mode.

Two modes:
  --raw (default)   Distribution BEFORE any cleaning decision: parses the
                     Freytag annotation file directly via the UNFILTERED
                     parser, bypassing every adapter-level cleaning choice
                     (including the "Adult" non-identity filter) entirely,
                     so it reflects the data exactly as it arrives — e.g.
                     CTai shows 78 identities here, matching Section III-C
                     and the threshold discussion in Section V-B, with
                     "Adult" appearing as its own (anomalously tall) bar.
                     This is the exploratory view meant to justify the
                     min-images-per-identity threshold and the later
                     "Adult" exclusion (Sections V-B/V-C), not a result
                     computed after those decisions were already made.
  --cleaned          Distribution AFTER the manifest's exclusions, read from
                     manifests/*.csv — useful for a before/after comparison,
                     not for justifying the cleaning itself.

Identity names are intentionally not shown (axis is rank, not name): the
distribution's SHAPE is what matters here, not which individual is which.

Usage:
    python diagnostics/dataset_distribution.py                 # raw, default, one PNG per dataset
    python diagnostics/dataset_distribution.py --cleaned
    python diagnostics/dataset_distribution.py --raw --cleaned  # both modes, one PNG each
"""
from __future__ import annotations
import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.manifest import load_manifest
from adapters._freytag_common import parse_freytag_annotations_unfiltered

RAW_ANNOTATIONS = {
    "ctai": "notebooks/chimpanzee_faces/datasets_cropped_chimpanzee_faces/data_CTai/annotations_ctai.txt",
    "czoo": "notebooks/chimpanzee_faces/datasets_cropped_chimpanzee_faces/data_CZoo/annotations_czoo.txt",
}
CLEANED_MANIFESTS = {
    "ctai": "manifests/ctai_manifest.csv",
    "czoo": "manifests/czoo_manifest.csv",
}
COLORS = {"ctai": "#4C72B0", "czoo": "#DD8452"}


def raw_counts(name: str):
    df = parse_freytag_annotations_unfiltered(RAW_ANNOTATIONS[name])  # no cleaning applied at all
    return df["identity"].value_counts().sort_values(ascending=False).reset_index(drop=True)


def cleaned_counts(name: str):
    df = load_manifest(CLEANED_MANIFESTS[name], image_root=None)
    return df["identity"].value_counts().sort_values(ascending=False).reset_index(drop=True)


def plot_distribution(counts, dataset: str, mode: str, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ranks = range(1, len(counts) + 1)
    ax.bar(ranks, counts, color=COLORS[dataset], width=0.9)
    ax.set_title(f"{dataset.upper()} ({mode}) — {len(counts)} identities, {counts.sum()} images")
    ax.set_xlabel("Identity rank (most to fewest images)")
    ax.set_ylabel("Images")
    ax.axhline(counts.mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean = {counts.mean():.1f}")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{dataset}_distribution_{mode}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--raw", action="store_true", help="Pre-cleaning distribution (unfiltered Freytag parser).")
    ap.add_argument("--cleaned", action="store_true", help="Post-cleaning distribution (from manifests/*.csv).")
    args = ap.parse_args()
    if not args.raw and not args.cleaned:
        args.raw = True  # default: exploration happens BEFORE cleaning decisions
    os.makedirs(args.out, exist_ok=True)

    modes = [m for m, flag in [("raw", args.raw), ("cleaned", args.cleaned)] if flag]
    for mode in modes:
        get_counts = raw_counts if mode == "raw" else cleaned_counts
        for name in ["ctai", "czoo"]:
            counts = get_counts(name)
            print(f"[{mode}] {name}: {len(counts)} identities, {counts.sum()} images, "
                  f"min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
            out_path = plot_distribution(counts, name, mode, args.out)
            print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()