"""
embedding.py — wraps MegaDescriptor (via timm's HF-hub integration) to
produce L2-normalized embedding vectors for cosine-similarity matching in
matching.py.

Mirrors the rest of the pipeline: given an already-preprocessed image tensor
(from dataset.py's transform) and the identifying filepath, look for a
cached embedding first (cache.py) and only run the forward pass on a miss.
MegaDescriptorExtractor never touches the cache directly — every call site
in run_pipeline.py goes through extract_with_cache().
"""
from __future__ import annotations

import numpy as np
import torch

from pipeline.cache import EmbeddingCache


class MegaDescriptorExtractor:
    """Thin wrapper around a MegaDescriptor backbone loaded via timm's HF-hub
    integration. model_name is an HF-hub timm identifier, e.g.
    "hf-hub:BVRA/MegaDescriptor-L-384" (see config/*.yaml).
    """

    def __init__(self, model_name: str, device: str = "auto"):
        import timm

        self.model_name = model_name
        self.device = self._resolve_device(device)

        # num_classes=0 -> pooled feature vector, no classification head
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> np.ndarray:
        """image_tensor: a single already-transformed CHW tensor, as produced
        by ManifestFaceDataset.__getitem__ via build_real_transform() in
        run_pipeline.py. Returns an L2-normalized 1D numpy embedding.
        """
        batch = image_tensor.unsqueeze(0).to(self.device)
        features = self.model(batch)  # (1, dim)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu().numpy()


class ResNet50Extractor:
    """Zero-shot CNN baseline: ImageNet-pretrained ResNet-50 with its
    classification head removed, used purely as a fixed feature extractor
    (no fine-tuning). Exists to quantify what MegaDescriptor's wildlife-
    specific pretraining gains over a generic, off-the-shelf CNN backbone
    on the same closed-set matching pipeline. Mirrors MegaDescriptorExtractor's
    interface exactly so extract_with_cache() and run_pipeline.py need no
    changes beyond picking which extractor to instantiate.
    """

    def __init__(self, model_name: str = "resnet50", device: str = "auto"):
        import torchvision.models as tv_models

        self.model_name = model_name
        self.device = MegaDescriptorExtractor._resolve_device(device)

        # Identical pretraining source (ImageNet) and normalization stats as
        # MegaDescriptor's backbone, so build_real_transform() in
        # run_pipeline.py needs no changes — only img_size differs (224,
        # ResNet-50's native resolution, set via config).
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        backbone = tv_models.resnet50(weights=weights)
        backbone.fc = torch.nn.Identity()  # drop classification head -> pooled 2048-d features
        self.model = backbone
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Same contract as MegaDescriptorExtractor.extract: single CHW
        tensor in, L2-normalized 1D numpy embedding out.
        """
        batch = image_tensor.unsqueeze(0).to(self.device)
        features = self.model(batch)  # (1, 2048)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.squeeze(0).cpu().numpy()
    
def extract_with_cache(extractor: MegaDescriptorExtractor, cache: EmbeddingCache,
                        filepath: str, image_tensor: torch.Tensor) -> np.ndarray:
    """Single call site used by run_pipeline.py: cache-first, compute-on-miss.
    filepath is the cache key (see cache.py); image_tensor is only touched
    if there's a cache miss.
    """
    cached = cache.get(filepath)
    if cached is not None:
        return cached
    embedding = extractor.extract(image_tensor)
    cache.set(filepath, embedding)
    return embedding