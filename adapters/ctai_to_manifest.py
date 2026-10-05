"""
Converts the raw CTai annotation file (Freytag et al. 2016 format) into this
project's standard manifest.csv.

Usage:
    python adapters/ctai_to_manifest.py \
        --data-root chimpanzee_faces/datasets_cropped_chimpanzee_faces/data_CTai \
        --out manifests/ctai_manifest.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.manifest import make_row, save_manifest, load_manifest, summarize
from adapters._freytag_common import parse_freytag_annotations, stratified_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="chimpanzee_faces/datasets_cropped_chimpanzee_faces/data_CTai")
    ap.add_argument("--out", default="manifests/ctai_manifest.csv")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann_path = os.path.join(args.data_root, "annotations_ctai.txt")
    df = parse_freytag_annotations(ann_path)
    df = stratified_split(df, args.test_fraction, args.seed)

    rows = [
        make_row(filepath=r.filepath, identity=r.identity, split=r.split,
                  source="ctai", bbox=None)  # None = already a tight face crop
        for r in df.itertuples()
    ]
    save_manifest(rows, args.out)

    check = load_manifest(args.out, image_root=args.data_root)
    print(f"Wrote {args.out}\n")
    print(summarize(check))


if __name__ == "__main__":
    main()
