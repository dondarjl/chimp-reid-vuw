"""
dataset.py — torch Dataset that reads ONLY from a manifest DataFrame.

Requires torch + torchvision for the tensor conversion in __getitem__. The
non-tensor logic here (row -> image -> crop, in get_pil_crop) is written to
be testable without torch — no automated test exists yet, but it is the
natural place to add one.
"""

from __future__ import annotations
import os
import pandas as pd
from PIL import Image

from pipeline.detection import FaceDetector, PassthroughDetector, ManifestBBoxDetector, BBox, crop


class ManifestFaceDataset:
    """torch.utils.data.Dataset-compatible (duck-typed to avoid a hard torch
    import at module load time, so non-Colab environments can still import
    and unit-test the row/crop logic below).

    image_root : directory the manifest's relative filepaths are joined to.
    transform  : any callable image -> tensor (torchvision.transforms.Compose
                 in practice). Passed in rather than hardcoded so switching
                 model/IMG_SIZE only touches config.yaml + run_pipeline.py.
    detector   : FaceDetector instance. If None, chosen automatically per row
                 based on whether the manifest row has a bbox — this is what
                 lets CZoo/CTai (no bbox) and a future raw-photo Wellington
                 batch (bbox present) share one Dataset class.
    """
    def __init__(self, df: pd.DataFrame, image_root: str, transform=None,
                 detector: FaceDetector | None = None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self._fixed_detector = detector

    def __len__(self):
        return len(self.df)

    def _resolve_detector(self, row) -> FaceDetector:
        if self._fixed_detector is not None:
            return self._fixed_detector
        if pd.isna(row.bbox_x):
            return PassthroughDetector()
        return ManifestBBoxDetector(BBox(row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h))

    def get_pil_crop(self, idx: int) -> tuple[Image.Image, str, str]:
        """Row -> cropped PIL image, without touching torch. This is the part
        covered by the sandbox tests; __getitem__ below just adds the tensor
        conversion on top for actual training/inference use in Colab.
        """
        row = self.df.iloc[idx]
        path = os.path.join(self.image_root, row.filepath)
        image = Image.open(path).convert("RGB")
        detector = self._resolve_detector(row)
        bbox = detector.detect(image)
        return crop(image, bbox), row.identity, row.filepath

    def __getitem__(self, idx: int):
        cropped, identity, filepath = self.get_pil_crop(idx)
        if self.transform is not None:
            cropped = self.transform(cropped)
        return cropped, identity, filepath
