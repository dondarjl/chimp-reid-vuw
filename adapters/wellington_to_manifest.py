"""
TEMPLATE — fill this in once Wellington Zoo images arrive. This is the ONLY
file in the whole project that needs writing from scratch for the real
dataset. Nothing in pipeline/ or run_pipeline.py should need to change.

Two example implementations are sketched below for the two most likely ways
the data will actually show up. Delete whichever one you don't need, adjust
column/folder names to match reality, and keep the parts that write the
manifest (identical in both).

IMPORTANT — bbox handling:
Unlike CZoo/CTai (pre-cropped faces), Wellington images will very likely be
full photos of the enclosure, not tight face crops (see the report, Section 6:
this is the pipeline's most important open gap). That means:
  - If you crop faces yourself before this step (semi-manual crop, recommended
    for the first data batch — see report Section 7), pass bbox=None per row,
    exactly like CZoo/CTai: the image IS the face.
  - If you keep full photos and record a bounding box per face (e.g. from a
    detector or manual annotation), pass the real bbox=(x, y, w, h) instead.
    dataset.py will crop it at load time using pipeline/detection.py.
  - Do not mix the two within one manifest column silently — manifest.py's
    validator will reject a manifest where only SOME of x/y/w/h are set per
    row, precisely to catch this kind of half-done annotation early.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.manifest import make_row, save_manifest, load_manifest, summarize
from adapters._freytag_common import stratified_split
import pandas as pd


# --------------------------------------------------------------------------
# OPTION A — folder-per-individual, pre-cropped faces (Bristol-style)
#   wellington_data/
#     ├── kambiri/
#     │     ├── kambiri_001.jpg
#     │     └── kambiri_002.jpg
#     ├── zula/
#     │     └── ...
# This is the recommended structure for the first data-collection batch —
# see the report's data-collection protocol note (Section 7). It's the
# easiest to get right on a phone/camera at the zoo: one sub-folder per
# named individual, keeper-verified.
# --------------------------------------------------------------------------
def load_folder_per_individual(data_root: str) -> pd.DataFrame:
    rows = []
    for identity in sorted(os.listdir(data_root)):
        identity_dir = os.path.join(data_root, identity)
        if not os.path.isdir(identity_dir):
            continue
        for fname in sorted(os.listdir(identity_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "filepath": os.path.join(identity, fname),
                    "identity": identity,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# OPTION B — flat folder + a spreadsheet/CSV of keeper-verified labels
#   wellington_data/images/IMG_0001.jpg, IMG_0002.jpg, ...
#   wellington_data/labels.csv  with columns: filename, identity[, x,y,w,h]
# Use this if keepers log sightings in a spreadsheet rather than you sorting
# files into folders — often more realistic once video/burst capture starts.
# --------------------------------------------------------------------------
def load_flat_plus_csv(data_root: str, labels_csv: str) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    df = pd.DataFrame({
        "filepath": labels["filename"].apply(lambda f: os.path.join("images", f)),
        "identity": labels["identity"],
    })
    # If the CSV has bbox columns, carry them through instead of None below.
    for col in ("x", "y", "w", "h"):
        if col in labels.columns:
            df[f"bbox_{col}"] = labels[col]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["folder", "flat_csv"], default="folder",
                     help="Pick the layout that matches how your data actually arrives.")
    ap.add_argument("--data-root", default="wellington_data")
    ap.add_argument("--labels-csv", default="wellington_data/labels.csv",
                     help="Only used when --mode=flat_csv")
    ap.add_argument("--out", default="manifests/wellington_manifest.csv")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "folder":
        df = load_folder_per_individual(args.data_root)
        has_bbox = False
    else:
        df = load_flat_plus_csv(args.data_root, args.labels_csv)
        has_bbox = all(c in df.columns for c in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"])

    df = stratified_split(df, args.test_fraction, args.seed)

    rows = []
    for r in df.itertuples():
        bbox = (r.bbox_x, r.bbox_y, r.bbox_w, r.bbox_h) if has_bbox else None
        rows.append(make_row(filepath=r.filepath, identity=r.identity, split=r.split,
                              source="wellington", bbox=bbox))
    save_manifest(rows, args.out)

    check = load_manifest(args.out, image_root=args.data_root)
    print(f"Wrote {args.out}\n")
    print(summarize(check))
    print("\nSanity-check this output before training on it: identity counts per\n"
          "individual, in particular, since sampling bias toward subordinate\n"
          "individuals was flagged as a known risk in the literature review.")


if __name__ == "__main__":
    main()
