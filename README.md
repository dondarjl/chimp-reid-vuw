# Chimpanzee Re-ID Pipeline — Wellington Zoo

Closed-set re-identification pipeline: a small, known, named roster of
individuals — not open-set wildlife discovery. Built and validated against
CZoo/CTai (Freytag et al. 2016) as stand-in benchmarks while waiting for
real Wellington Zoo photos, with a single, deliberately narrow adapter point
for swapping the real data in later.

## Why it's organised this way

Every module exists to answer one question: **"what's the one thing that
changes when Wellington data replaces CZoo/CTai?"** The answer should always
be "one adapter file and one config file" — never the pipeline logic itself.
See `adapters/wellington_to_manifest.py` and `config/wellington.yaml.template`
for that swap-in point.

## Structure

```
config/ YAML per dataset — paths, model, eval settings
czoo.yaml
ctai.yaml
ctai_resnet50.yaml / czoo_resnet50.yaml ResNet-50 baseline variants
wellington.yaml.template ← copy to wellington.yaml when data arrives

adapters/ One file per dataset source → manifests/*.csv
czoo_to_manifest.py
ctai_to_manifest.py
wellington_to_manifest.py ← the file to actually write when data arrives
_freytag_common.py shared parser, not a dataset itself

pipeline/ Dataset-agnostic — never touches raw formats
manifest.py the data contract (see docstring for schema)
detection.py face-cropping interface (Passthrough today)
dataset.py torch Dataset reading only from a manifest
cache.py on-disk embedding cache, namespaced by
(dataset, model, image size) — never shared
across datasets or models
embedding.py MegaDescriptor + ResNet-50 extractors, both
behind the same cache-integrated interface
matching.py closed-set kNN (not open-set clustering);
full-gallery distance ranking, not predict_proba
evaluation.py top-1/top-k accuracy, per-identity breakdown

diagnostics/
error_analysis.py generates a query/wrong-match/true-match panel
and a summary CSV for every misclassified test
image — the tool that surfaced two of the four
findings below

manifests/ generated CSVs (git-ignored, regenerate via adapters/)
cache/embeddings/ generated embedding cache (git-ignored)
results/experiment_log.csv one row appended per run — the running experiment log

run_pipeline.py single entrypoint tying everything together
```

## Quickstart

**Dry run (works anywhere, no GPU, no torch required)** — validates the
entire architecture with seeded fake embeddings instead of a real model.
Useful after writing a new adapter, before spending GPU time:

```bash
pip install -r requirements.txt   # only the top section is needed for this
python adapters/czoo_to_manifest.py
python adapters/ctai_to_manifest.py
python run_pipeline.py --config config/czoo.yaml --dry-run
python run_pipeline.py --config config/ctai.yaml --dry-run
```

Dry-run accuracy numbers are meaningless as accuracy — they come from a
synthetic per-identity latent vector plus noise, not a real model. All a
passing dry-run tells you is that manifests load, images open and crop
correctly, and matching/logging work.

**Real run**:

```bash
python run_pipeline.py --config config/czoo.yaml
python run_pipeline.py --config config/ctai.yaml
python run_pipeline.py --config config/ctai_resnet50.yaml   # CNN baseline comparison
```

## Current results

| Dataset | Model | Top-1 | Top-5 |
|---|---|---|---|
| CZoo | MegaDescriptor (zero-shot) | 99.76% | 99.76% |
| CZoo | ResNet-50 (zero-shot) | 36.41% | 63.12% |
| CTai | MegaDescriptor (zero-shot) | 73.76% | 90.50% |
| CTai | ResNet-50 (zero-shot) | 51.94% | 80.02% |

Full breakdown, per-identity numbers, and discussion in the report.

## Qualitative error analysis

```bash
python diagnostics/error_analysis.py --config config/ctai.yaml --out diagnostics/errors/ctai
```

Generates one panel image per misclassified test image (query | nearest
wrong-class match | nearest true-class match) plus `_summary.csv` with
per-error distances — used to find the Fredy/Victor confusion pattern
reported as the main qualitative finding (see report, Section V-C).

## Methodological findings (all caught before being reported)

This project's diagnostic-first workflow (dry-run first, then real run;
qualitative error review; suspicious-result verification) caught four
silent defects during development, each documented in the report:

1. **Unstratified train/test split** left several CTai identities with zero
   training images (`adapters/_freytag_common.py::stratified_split`, fixed
   with a minimum-images-per-identity threshold).
2. **Degenerate k=1 ranking**: `KNeighborsClassifier.predict_proba` collapses
   to one-hot at k=1, making any rank beyond #1 an alphabetical artefact, not
   a similarity ranking. Fixed in `matching.py` by ranking on real distance
   to the gallery instead.
3. **"Adult" is not an identity** — a Freytag `Age_Group` placeholder value
   sitting in 416 CTai rows' `Name` field, not a real individual. Found via
   qualitative error review, fixed with an explicit non-identity filter in
   `_freytag_common.py`.
4. **Cross-dataset cache collision**: `EmbeddingCache` originally keyed
   entries by `filepath` alone; since CTai and CZoo share Freytag's filename
   scheme, ~85% of CZoo images were silently served CTai embeddings when
   both datasets were evaluated with the same model. Fixed by namespacing
   the cache directory by `(dataset, model, image_size)`. **If re-running an
   older clone of this repo, clear `cache/embeddings/` first.**

## Adding Wellington Zoo data, when it arrives

1. Decide a layout — folder-per-individual (recommended for the first,
   semi-manually-cropped batch) or flat-folder-plus-CSV. Both are sketched
   in `adapters/wellington_to_manifest.py`.
2. Run that adapter to produce `manifests/wellington_manifest.csv`. **Check
   the per-identity image counts it prints** — the troop is small, and the
   shared `stratified_split()` default (minimum 4 images/identity) may
   exclude more individuals than expected on a small first batch; adjust if so.
3. Copy `config/wellington.yaml.template` to `config/wellington.yaml`,
   fill in `image_root`, and set `precropped` correctly (see the template's
   comments — this is the raw-photo-vs-pre-cropped-face decision).
4. `python run_pipeline.py --config config/wellington.yaml --dry-run` first,
   to catch data issues without spending GPU time, then drop `--dry-run`.

Nothing in `pipeline/` or `run_pipeline.py` should need to change.

## Known gaps

- **No trained face detector.** `detection.py`'s `YoloFaceDetector` is a
  documented stub. The planned approach (MegaDetector for coarse
  localisation, then a YOLOv8 fine-tuned on a small annotated Wellington
  subset for the face region — see report, Section III-C) has not been
  built or validated, since no Wellington imagery is available yet.
- **CNN baseline is zero-shot only.** `ResNet50Extractor` in `embedding.py`
  is an untrained, ImageNet-pretrained feature extractor used as-is. A
  fine-tuned (triplet-loss) version was scoped but not implemented.
- **Evaluation metrics are top-1/top-k and per-identity accuracy only.**
  mAP, full CMC curves, and a confusion matrix were scoped but not
  implemented; the per-identity breakdown was enough to surface the
  sampling-bias pattern discussed in the report, but is coarser than a
  full image-level mAP.