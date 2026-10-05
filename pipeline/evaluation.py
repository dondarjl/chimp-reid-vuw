"""
evaluation.py — just enough evaluation for run_pipeline.py to report a
meaningful result and log it to results/experiment_log.csv.

Deliberately NOT the full diagnostics suite discussed alongside the pipeline
architecture (confusion-matrix heatmaps, t-SNE/UMAP embedding plots, the
shortcut-learning background probe) — that's the next bucket of work, kept
separate on purpose so this one stays focused on architecture. What's here
is the minimum needed to know whether the architecture actually works
end-to-end: top-1/top-k accuracy and a per-identity breakdown.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def top_k_accuracy(true_identities: np.ndarray, ranked_identities: np.ndarray, k: int) -> float:
    """ranked_identities: (n_samples, n_classes), best match first (from
    ClosedSetMatcher.predict_proba_ranked). True positive if the true label
    appears anywhere in the top k.
    """
    top_k = ranked_identities[:, :k]
    hits = [true in row for true, row in zip(true_identities, top_k)]
    return float(np.mean(hits))


def per_identity_accuracy(true_identities: np.ndarray, predicted_identities: np.ndarray) -> pd.DataFrame:
    """Breaks accuracy down per individual — this is what would surface the
    sampling-bias / juvenile-drift risks flagged in the report as concrete
    numbers rather than a hunch, once real diagnostics work starts.
    """
    df = pd.DataFrame({"true": true_identities, "pred": predicted_identities})
    df["correct"] = df["true"] == df["pred"]
    summary = df.groupby("true")["correct"].agg(["mean", "count"]).rename(
        columns={"mean": "accuracy", "count": "n_test_images"}
    )
    return summary.sort_values("accuracy")


def summarize_run(true_identities: np.ndarray, predicted_identities: np.ndarray,
                   ranked_identities: np.ndarray | None = None,
                   k_values: tuple[int, ...] = (1, 5)) -> dict:
    result = {"top_1_accuracy": float(np.mean(true_identities == predicted_identities))}
    if ranked_identities is not None:
        for k in k_values:
            if k <= ranked_identities.shape[1]:
                result[f"top_{k}_accuracy"] = top_k_accuracy(true_identities, ranked_identities, k)
    return result
