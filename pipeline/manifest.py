"""
manifest.py — the single data contract for the whole pipeline.

Every dataset (CZoo, CTai, Bristol, Wellington-future) gets converted ONCE by
a small adapter script (see adapters/) into a manifest CSV with this exact
schema. Every other module in this package (dataset.py, embedding.py,
matching.py, run_pipeline.py) reads ONLY the manifest — never a
dataset-specific annotation format directly.

This is the key reproducibility decision: adding a new dataset means writing
one new adapter, never touching the pipeline itself.

Schema
------
filepath   : str   — path to the image file, relative to `image_root` in the
                     dataset's config YAML (kept relative so the manifest is
                     portable across machines/Colab sessions).
identity   : str   — individual's name/ID. Must match the closed-set roster.
split      : str   — one of {"train", "test"}.
bbox_x     : float — top-left x of the face bounding box, in pixels. Empty/NaN
                     if the image is already a tight face crop (CZoo/CTai).
bbox_y     : float — top-left y of the face bounding box. Empty/NaN if pre-cropped.
bbox_w     : float — bounding box width. Empty/NaN if pre-cropped.
bbox_h     : float — bounding box height. Empty/NaN if pre-cropped.
source     : str   — which dataset this row came from (e.g. "czoo", "ctai",
                     "wellington"). Lets you mix datasets in one manifest and
                     still report accuracy broken down by source (important —
                     see the report's note on CZoo vs CTai image quality).

A missing bbox is a *signal*, not a data gap: it tells dataset.py "use the
full image, no cropping needed" (precropped case) instead of an error.
"""

from __future__ import annotations
import os
import pandas as pd

REQUIRED_COLUMNS = [
    "filepath", "identity", "split", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "source",
]

VALID_SPLITS = {"train", "test"}


class ManifestError(ValueError):
    pass


def new_manifest_rows() -> list[dict]:
    """Adapters build a list of these dicts, then pass to save_manifest()."""
    return []


def make_row(filepath: str, identity: str, split: str, source: str,
             bbox: tuple[float, float, float, float] | None = None) -> dict:
    """Helper so every adapter constructs rows the same way.
    bbox=None means 'already a tight face crop, use the full image'.
    """
    bx, by, bw, bh = bbox if bbox is not None else (None, None, None, None)
    return {
        "filepath": filepath, "identity": identity, "split": split,
        "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh, "source": source,
    }


def save_manifest(rows: list[dict], path: str) -> None:
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    validate_manifest(df, image_root=None)  # structural check only, no file-existence check yet
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_manifest(path: str, image_root: str | None = None) -> pd.DataFrame:
    """Loads and validates a manifest. If image_root is given, also checks
    that every file actually exists on disk — catches a stale/moved dataset
    immediately instead of failing deep inside a DataLoader worker.
    """
    if not os.path.exists(path):
        raise ManifestError(
            f"No manifest at {path}. Run the matching adapter in adapters/ "
            f"first (e.g. adapters/czoo_to_manifest.py)."
        )
    df = pd.read_csv(path)
    validate_manifest(df, image_root=image_root)
    return df


def validate_manifest(df: pd.DataFrame, image_root: str | None) -> None:
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ManifestError(f"Manifest missing required columns: {missing_cols}")

    if df["filepath"].isna().any():
        raise ManifestError("Manifest has rows with empty filepath.")
    if df["identity"].isna().any():
        raise ManifestError("Manifest has rows with empty identity.")

    bad_splits = set(df["split"].unique()) - VALID_SPLITS
    if bad_splits:
        raise ManifestError(f"Manifest has invalid split values: {bad_splits}. "
                             f"Only {VALID_SPLITS} are allowed.")

    # bbox columns must be all-present or all-absent PER ROW, not a mix of two values
    bbox_cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    bbox_present = df[bbox_cols].notna()
    partial = bbox_present.any(axis=1) & ~bbox_present.all(axis=1)
    if partial.any():
        bad_rows = df[partial]["filepath"].tolist()[:5]
        raise ManifestError(
            f"{partial.sum()} row(s) have a partial bbox (some of x/y/w/h set, "
            f"not all four). A row must have all four or none. "
            f"First offending files: {bad_rows}"
        )

    if image_root is not None:
        missing_files = [
            fp for fp in df["filepath"]
            if not os.path.exists(os.path.join(image_root, fp))
        ]
        if missing_files:
            raise ManifestError(
                f"{len(missing_files)} file(s) listed in the manifest don't exist "
                f"under image_root={image_root}. First few: {missing_files[:5]}"
            )


def summarize(df: pd.DataFrame) -> str:
    lines = [
        f"Rows: {len(df)}",
        f"Identities: {df['identity'].nunique()}",
        f"Sources: {dict(df['source'].value_counts())}",
        f"Split sizes: {dict(df['split'].value_counts())}",
        f"Pre-cropped rows (no bbox): {df['bbox_x'].isna().sum()}",
        f"Raw-image rows (bbox given): {df['bbox_x'].notna().sum()}",
    ]
    return "\n".join(lines)
