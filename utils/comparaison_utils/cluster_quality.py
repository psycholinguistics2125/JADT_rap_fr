#!/usr/bin/env python3
"""
Cross-embedding cluster-quality metrics (semantic D5).
======================================================

Supporting numbers for the semantic evaluation: how well does each partition
(Iramuteq / LDA / BERTopic) separate in a given sentence-embedding space? These
are standard internal cluster-validity indices computed on verse embeddings:

- **Calinski-Harabasz** — variance-ratio criterion, O(n), on the full matrix.
- **Silhouette (cosine)** — on a stratified sample (full silhouette is O(n²)).
- **Davies-Bouldin** — cheap, one more index for triangulation.

Non-circularity is the whole point: evaluate a BERTopic partition in embedding
spaces it did *not* cluster on. The caller passes ``clustered_space`` so the
aligned (circular) space can be flagged in the output rather than silently
mixed in with the independent evidence.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _stratified_sample_indices(labels: np.ndarray, sample_size: int,
                               seed: int = 42) -> np.ndarray:
    """Deterministic class-stratified sample of row indices.

    Draws (roughly) proportional counts per class with a fixed RNG so the
    silhouette sample is reproducible. Returns sorted indices into ``labels``.
    """
    n = len(labels)
    if sample_size >= n:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    chosen: List[np.ndarray] = []
    for c in classes:
        c_idx = np.where(labels == c)[0]
        take = max(1, int(round(len(c_idx) * sample_size / n)))
        take = min(take, len(c_idx))
        chosen.append(rng.choice(c_idx, size=take, replace=False))

    idx = np.concatenate(chosen)
    if len(idx) > sample_size:
        idx = rng.choice(idx, size=sample_size, replace=False)
    return np.sort(idx)


def compute_cluster_quality(
    embeddings: np.ndarray,
    labels: List[int],
    sample_size: int = 10000,
    seed: int = 42,
    clustered_space: Optional[str] = None,
    eval_space: Optional[str] = None,
) -> Dict[str, object]:
    """Internal cluster-validity indices for one partition in one embedding space.

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) verse embeddings (L2-normalized; cosine silhouette assumes this).
    labels : List[int]
        Partition label per verse. Outliers (label == -1) are dropped.
    sample_size : int, default=10000
        Size of the stratified sample for the (O(n²)) silhouette.
    seed : int, default=42
        RNG seed for the silhouette sample.
    clustered_space / eval_space : str, optional
        Embedding-space keys. If both given and equal, ``circular`` is set True
        in the result to flag that this score is aligned-by-construction and is
        *not* independent evidence.

    Returns
    -------
    dict
        ``{calinski_harabasz, silhouette_cosine, davies_bouldin, n_docs,
        n_sample, seed, eval_space, circular}``.
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    mask = labels != -1
    X = embeddings[mask]
    y = labels[mask]

    n_classes = len(np.unique(y))
    result: Dict[str, object] = {
        'eval_space': eval_space,
        'n_docs': int(len(y)),
        'seed': int(seed),
        'circular': bool(clustered_space is not None and eval_space is not None
                         and clustered_space == eval_space),
    }

    if n_classes < 2 or len(y) < 2:
        result.update({'calinski_harabasz': 0.0, 'silhouette_cosine': 0.0,
                       'davies_bouldin': 0.0, 'n_sample': 0})
        return result

    # Full-matrix indices (both O(n)).
    result['calinski_harabasz'] = float(calinski_harabasz_score(X, y))
    result['davies_bouldin'] = float(davies_bouldin_score(X, y))

    # Stratified-sample cosine silhouette (full silhouette is O(n²)).
    sample_idx = _stratified_sample_indices(y, sample_size, seed)
    Xs, ys = X[sample_idx], y[sample_idx]
    if len(np.unique(ys)) >= 2:
        result['silhouette_cosine'] = float(silhouette_score(Xs, ys, metric='cosine'))
    else:
        result['silhouette_cosine'] = 0.0
    result['n_sample'] = int(len(sample_idx))

    return result
