"""
cache.py — on-disk embedding cache.

Extracting embeddings is the expensive step (a full forward pass through
MegaDescriptor for every image). Without caching, every re-run of the
pipeline — e.g. trying a different k in kNN, or re-running after adding five
new Wellington photos — silently redoes ALL of that GPU work from scratch.

Cache key = hash(filepath + model_name + image_size), so:
  - Two different models never collide in the cache.
  - Changing IMG_SIZE in config invalidates old entries automatically instead
    of silently mixing embeddings computed at different resolutions.
  - Re-running the exact same (image, model, size) combo is a disk read, not
    a GPU forward pass.
"""
from __future__ import annotations
import hashlib
import os
import numpy as np


class CacheDimensionMismatch(RuntimeError):
    """Raised when a cached embedding's dimension doesn't match what this
    session has been producing. This exact bug was hit once already during
    development here: a stray cache entry from an unrelated manual test sat
    in the same directory a real run used, silently mixing a 768-dim vector
    into a batch of 128-dim ones. numpy's error for that (three steps later,
    inside np.array(embeddings)) is confusing; this one tells you exactly
    which file and which two dimensions collided.
    """
    pass


# cache.py — cambios en EmbeddingCache

class EmbeddingCache:
    def __init__(self, cache_dir: str, model_name: str, img_size: int,
                 dataset_name: str, enabled: bool = True):
        self.enabled = enabled
        self.img_size = img_size
        model_slug = model_name.replace("/", "_").replace(":", "_")
        # dataset_name is now part of the namespace, not just model+size —
        # two datasets that happen to share a relative filepath scheme
        # (e.g. CTai and CZoo, both Freytag-format) must never collide.
        self.dir = os.path.join(cache_dir, f"{model_slug}_{img_size}", dataset_name)
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)
        self._dim: int | None = None

    def _key(self, filepath: str) -> str:
        digest = hashlib.sha1(filepath.encode("utf-8")).hexdigest()
        return os.path.join(self.dir, f"{digest}.npy")

    def _check_dim(self, filepath: str, embedding: np.ndarray) -> None:
        dim = embedding.shape[-1]
        if self._dim is None:
            self._dim = dim
        elif dim != self._dim:
            raise CacheDimensionMismatch(
                f"Embedding for {filepath!r} has dimension {dim}, but this cache "
                f"session has been using dimension {self._dim}. This almost always "
                f"means stale/foreign entries are sitting in {self.dir} — e.g. from "
                f"a different model, a manual test, or an interrupted run with a "
                f"different embedding size. Clear that directory and re-run."
            )

    def get(self, filepath: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        key_path = self._key(filepath)
        if os.path.exists(key_path):
            embedding = np.load(key_path)
            self._check_dim(filepath, embedding)
            return embedding
        return None

    def set(self, filepath: str, embedding: np.ndarray) -> None:
        if not self.enabled:
            return
        self._check_dim(filepath, embedding)
        np.save(self._key(filepath), embedding)

    def stats(self, manifest_filepaths: list[str]) -> dict:
        n_cached = sum(1 for fp in manifest_filepaths if os.path.exists(self._key(fp)))
        return {"total": len(manifest_filepaths), "cached": n_cached,
                "to_compute": len(manifest_filepaths) - n_cached}
