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
config/                    YAML per dataset — paths, model, eval settings
  czoo.yaml
  ctai.yaml
  czoo_resnet50.yaml / ctai_resnet50.yaml   ResNet-50 zero-shot baseline variants
  wellington.yaml.template ← copy to wellington.yaml when data arrives

adapters/                  One file per dataset source → manifests/*.csv
  czoo_to_manifest.py
  ctai_to_manifest.py
  wellington_to_manifest.py ← the file to actually write when data arrives
  _freytag_common.py       shared parser, not a dataset itself

pipeline/                  Dataset-agnostic — never touches raw formats
  manifest.py              the data contract (see docstring for schema)
  detection.py             face-cropping interface (Passthrough today)
  dataset.py               torch Dataset reading only from a manifest
  cache.py                 on-disk embedding cache, namespaced by
                           (dataset, model, image size) — never shared
                           across datasets or models
  embedding.py             MegaDescriptor + ResNet-50 (zero-shot) extractors,
                           both behind the same cache-integrated interface
  matching.py              closed-set kNN (not open-set clustering);
                           full-gallery distance ranking, not predict_proba
  evaluation.py            top-1/top-k accuracy, per-identity breakdown

diagnostics/
  error_analysis.py        generates a query/wrong-match/true-match panel
                           and a summary CSV for every misclassified test
                           image — the tool that surfaced two of the four
                           findings below
  dataset_distribution.py  per-dataset image-count distribution plots
                           (figures land in results/figures/)
  errors/                  generated output, one folder per config
                           (ctai, ctai_resnet50, czoo, czoo_resnet50),
                           each with correct/ and errors/

manifests/                 generated CSVs (git-ignored, regenerate via adapters/):
                           czoo_manifest.csv, ctai_manifest.csv
cache/embeddings/          generated embedding cache (git-ignored)
results/
  experiment_log.csv       one row appended per run — the running experiment log
  czoo_real_run_log.txt    console log of the CZoo real run
  figures/                 distribution figures used in the report
  resnet50_ft/             output of finetune_resnet50.py, one folder per run
                           (czoo_ep0, czoo_ep30, ctai_ep0, ctai_ep30), each with
                           metrics.json, per_identity_accuracy.csv, predictions.csv

notebooks/
  colab_pipeline.ipynb     Colab version of the pipeline
  chimpanzee_faces/        raw Freytag datasets (git-ignored); config image_root
                           points into datasets_cropped_chimpanzee_faces/
                           (data_CZoo/face_images, data_CTai/face_images)

run_pipeline.py            single entrypoint tying everything together
finetune_resnet50.py       STANDALONE script for the fine-tuned ResNet-50 baseline.
                           Not part of pipeline/: it reads a manifest, trains and
                           evaluates on its own, and bypasses the embedding cache
requirements.txt
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
python run_pipeline.py --config config/ctai_resnet50.yaml   # zero-shot CNN baseline
python run_pipeline.py --config config/czoo_resnet50.yaml
```

## Fine-tuned ResNet-50 baseline

Run separately from the pipeline, from the project root, once per dataset.
`--image-root` is the `image_root` folder written in the dataset's config YAML
(not the YAML itself):

```powershell
# 1) Sanity check: with --epochs 0 it must reproduce the zero-shot ResNet-50 numbers below
python finetune_resnet50.py --manifest manifests/czoo_manifest.csv --dataset czoo --image-root <image_root> --epochs 0
# 2) Fine-tuning (30 epochs, batch-hard triplet loss, fixed hyperparameters)
python finetune_resnet50.py --manifest manifests/czoo_manifest.csv --dataset czoo --image-root <image_root> --save-ckpt
```

Protocol: trained on the `train` split only, hyperparameters fixed in advance,
no model selection on the test split (the last-epoch model is evaluated).
Evaluation uses the same gallery/1-NN protocol as the other models. On a CPU
it takes hours; a GPU (or Colab) is strongly recommended.

## Current results

All numbers are on the cleaned manifests (CZoo: 1,686 train / 423 test, 24
identities; CTai: 3,715 train / 926 test, 65 identities).

| Dataset | Model | Top-1 | Top-5 |
|---|---|---|---|
| CZoo | MegaDescriptor (zero-shot) | 99.76% | 100.00% |
| CZoo | ResNet-50 (zero-shot) | 57.92% | 85.82% |
| CZoo | ResNet-50 (fine-tuned) | 75.18% | 95.04% |
| CTai | MegaDescriptor (zero-shot) | 97.95% | 99.35% |
| CTai | ResNet-50 (zero-shot) | 51.94% | 80.02% |
| CTai | ResNet-50 (fine-tuned) | 51.30% | 77.32% |

Earlier figures that appeared in previous versions of this README (CZoo
ResNet-50 36.41%, CTai MegaDescriptor 73.76%) came from runs affected by the
cache collision and the unstratified split described below; they are
superseded and must not be quoted.

Caveats: the fine-tuned ResNet-50 was trained once, with no hyperparameter
tuning, and its training loss had not converged, so it likely understates what
a tuned CNN can reach. MegaDescriptor's pretraining data appears to include
CZoo and CTai, so its "zero-shot" accuracy here is probably optimistic for
unseen individuals. Full breakdown, per-identity numbers, and discussion in the
report.

## Qualitative error analysis

```bash
python diagnostics/error_analysis.py --config config/ctai.yaml --out diagnostics/errors/ctai
```

Generates one panel image per misclassified test image (query | nearest
wrong-class match | nearest true-class match) plus `_summary.csv` with
per-error distances. On the clean CTai run MegaDescriptor misclassifies 19 of
926 test images, and the only identity pair that repeats is Fredy predicted as
Victor (twice); see the report, Section V-C. The earlier claim that this was a
dominant error pattern came from the contaminated run and no longer holds.

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

The roster is not fixed over time (births, deaths). With the zero-shot models,
removing a departed individual or adding a new one only means editing the
gallery, with no retraining; see the report's Discussion.

## Known gaps

- **No trained face detector.** `detection.py`'s `YoloFaceDetector` is a
  documented stub. The planned approach (MegaDetector for coarse
  localisation, then a YOLOv8 fine-tuned on a small annotated Wellington
  subset for the face region — see report, Section III-C) has not been
  built or validated, since no Wellington imagery is available yet.
- **Fine-tuned ResNet-50 is outside the pipeline and untuned.**
  `finetune_resnet50.py` is a standalone script: its model is not available
  through `run_pipeline.py` configs, it does not use `cache.py`, and
  `diagnostics/error_analysis.py` cannot produce panels for it. It was run
  once (30 epochs, fixed hyperparameters, no validation split), improved CZoo
  but not CTai, and its training loss had not converged. A proper comparison
  would need a validation split inside `train` to choose between
  configurations (e.g. a cross-entropy objective), never the test split.
- **Evaluation metrics are top-1/top-k and per-identity accuracy only.**
  mAP, full CMC curves, and a confusion matrix were scoped but not
  implemented; the per-identity breakdown was enough to surface the
  sampling-bias pattern discussed in the report, but is coarser than a
  full image-level mAP.