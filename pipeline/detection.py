"""
detection.py — pluggable face-cropping stage.

This module exists because of the single biggest structural gap identified
so far: CZoo/CTai are already tight face crops, but Wellington Zoo photos
almost certainly won't be — they'll be full shots of an enclosure. Rather
than bake "assume a pre-cropped face" into dataset.py, cropping is isolated
behind one interface, so the SAME pipeline runs whether or not a real
detector exists yet.

Today, only PassthroughDetector is implemented (used by CZoo/CTai, and by
Wellington IF you semi-manually crop faces before ingestion — see the report,
Section 7, and adapters/wellington_to_manifest.py). YoloFaceDetector is a
documented stub for later, once there's enough Wellington footage to justify
training or reusing a real detector (GorillaWatch/Bristol's YOLOv8 recipe is
the natural starting point per the literature review).
"""
from __future__ import annotations
from dataclasses import dataclass
from PIL import Image


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float


class FaceDetector:
    """Interface: every detector takes a PIL image and returns exactly one
    BBox — the face crop region to feed to the embedding model. Closed-set
    re-ID with one face per image doesn't need multi-face detection; if a
    future frame has more than one chimpanzee in it, that's a tracking
    problem (see the report's tracklet-voting stage), not this module's job.
    """
    def detect(self, image: Image.Image) -> BBox:
        raise NotImplementedError


class PassthroughDetector(FaceDetector):
    """For manifest rows with no bbox (bbox_x is NaN): the image IS the face
    already. Used for CZoo, CTai, and any Wellington batch that was
    semi-manually cropped before being added to the manifest.
    """
    def detect(self, image: Image.Image) -> BBox:
        w, h = image.size
        return BBox(x=0, y=0, w=w, h=h)


class ManifestBBoxDetector(FaceDetector):
    """For manifest rows that DO have a bbox already recorded (e.g. from
    manual annotation or a future automated detector's output stored ahead
    of time). Just replays the stored box — this class exists so
    dataset.py has one consistent code path regardless of where the bbox
    came from.
    """
    def __init__(self, bbox: BBox):
        self._bbox = bbox

    def detect(self, image: Image.Image) -> BBox:
        return self._bbox


class YoloFaceDetector(FaceDetector):
    """STUB — not implemented yet.

    When there's enough Wellington footage to justify it, this is where a
    trained face detector plugs in (e.g. Ultralytics YOLOv8, fine-tuned on
    chimpanzee faces the way Bristol/GorillaWatch did for gorillas). It only
    needs to satisfy the same detect() -> BBox interface; nothing else in
    the pipeline needs to change to start using it — swap PassthroughDetector
    for YoloFaceDetector in run_pipeline.py's detector selection and re-run.
    """
    def __init__(self, weights_path: str):
        raise NotImplementedError(
            "No trained chimpanzee face detector yet. Use PassthroughDetector "
            "with semi-manually cropped images for now (see report, Section 7)."
        )

    def detect(self, image: Image.Image) -> BBox:
        raise NotImplementedError


def crop(image: Image.Image, bbox: BBox) -> Image.Image:
    return image.crop((int(bbox.x), int(bbox.y), int(bbox.x + bbox.w), int(bbox.y + bbox.h)))
