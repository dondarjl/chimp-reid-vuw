"""
matching.py — closed-set kNN matching.

Deliberately NOT clustering/open-set discovery: the roster is known in
advance (see report, Section 4), so this is prototype/kNN matching against
labelled training embeddings, full stop.
"""
# matching.py
from __future__ import annotations
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances


class ClosedSetMatcher:
    def __init__(self, k_neighbors: int = 1, metric: str = "cosine"):
        self.k_neighbors = k_neighbors
        self.metric = metric
        self._clf: KNeighborsClassifier | None = None
        self._train_embeddings: np.ndarray | None = None
        self._train_identities: np.ndarray | None = None
        self._classes: np.ndarray | None = None

    def fit(self, train_embeddings: np.ndarray, train_identities: np.ndarray) -> None:
        self._train_embeddings = train_embeddings
        self._train_identities = np.asarray(train_identities)
        self._classes = np.unique(self._train_identities)
        # kept only so predict() still reflects the configured k_neighbors
        self._clf = KNeighborsClassifier(n_neighbors=self.k_neighbors, metric=self.metric)
        self._clf.fit(train_embeddings, train_identities)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Call fit() before predict().")
        return self._clf.predict(embeddings)

    def predict_proba_ranked(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ranks ALL identities per query by real distance to the training
        gallery, independent of k_neighbors. For each candidate identity, the
        score is the distance to that identity's CLOSEST training image
        (min-distance-per-class, standard closed-set protocol) — NOT via
        KNeighborsClassifier.predict_proba, which at k_neighbors=1 collapses
        to a one-hot vector and makes every rank beyond #1 meaningless
        (ties broken alphabetically, not by similarity). See evaluation
        notes / report for why this replaced the original implementation.
        """
        if self._train_embeddings is None:
            raise RuntimeError("Call fit() before predict_proba_ranked().")
        dist = pairwise_distances(embeddings, self._train_embeddings, metric=self.metric)
        n = embeddings.shape[0]
        ranked_identities = np.empty((n, len(self._classes)), dtype=self._classes.dtype)
        ranked_scores = np.empty((n, len(self._classes)))
        for i in range(n):
            per_class_min = {
                c: dist[i, self._train_identities == c].min() for c in self._classes
            }
            order = sorted(per_class_min, key=per_class_min.get)  # ascending distance = closest first
            ranked_identities[i] = order
            ranked_scores[i] = [per_class_min[c] for c in order]
        return ranked_identities, ranked_scores