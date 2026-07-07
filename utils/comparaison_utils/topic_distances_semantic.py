#!/usr/bin/env python3
"""
Semantic (embedding-space) counterpart to the lexical Q5 topic-distance suite.
==============================================================================

This module mirrors :mod:`topic_distances` exactly, swapping bag-of-words
counts for sentence embeddings and the Labbé/Jensen-Shannon distances for
**cosine distance**. It exists so the lexical Q5 separation analysis has a
symmetric embedding-space version: the same intra/inter aggregation logic,
the same data-driven aggregation grid, the same ``seed=42`` pair sampling,
and the same nested result shape consumed by
:func:`utils.comparaison_utils.visualization.create_aggregation_curve_plot`
(the metric key is ``'cosine'`` instead of ``'labbe'``).

Framing note (carried from the plan): these embedding metrics are
aligned-by-construction with embedding methods, exactly as χ²/n is aligned
with Reinert's criterion. They are reported as the semantic-space counterpart
**for transparency**, not as an overall arbiter of model quality.
Non-circularity: a BERTopic partition clustered on one embedding space should
be evaluated here in the *other* French embedding spaces.

Design mirror of :mod:`topic_distances`:
- ``aggregate_embeddings``      ~ ``aggregate_documents``
- ``evaluate_semantic_topic_distances``    ~ ``evaluate_topic_distances``
- ``evaluate_semantic_multi_aggregation``  ~ ``evaluate_multi_aggregation``
- ``compute_semantic_centroid_distances``  ~ ``compute_topic_centroid_distances``

Inputs are assumed to be **L2-normalized** verse embeddings (the BERTopic
pipeline caches them normalized), so cosine distance = ``1 - u·v``.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the exact unique-pair sampler from the lexical suite so that the
# semantic and lexical curves draw pairs identically (comparability).
from .topic_distances import _sample_pairs


# =============================================================================
# EMBEDDING CACHE LOADING (reuses the BERTopic verse-embedding caches)
# =============================================================================

# models/embeddings/verses_iramuteq_filter_{key}.npy — same cache written by
# build_and_evaluate_bertopic.compute_embeddings. We load it here with a
# lightweight path helper to avoid importing the heavy BERTopic dependencies
# (bertopic/umap/sentence-transformers) into the comparison pipeline.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = PROJECT_ROOT / "models" / "embeddings"


def get_embedding_path(embedding_key: str) -> Path:
    """Path of the cached verse embeddings for a given backend key.

    Mirrors build_and_evaluate_bertopic.get_embedding_path.
    """
    return EMBEDDINGS_DIR / f"verses_iramuteq_filter_{embedding_key}.npy"


def load_verse_embeddings(embedding_key: str) -> np.ndarray:
    """Load the cached, corpus-ordered, L2-normalized verse embeddings.

    The cache is indexed by corpus row order, i.e. by the ``original_index``
    used to align the three partitions, so ``embeddings[original_index]`` is
    the vector for that verse.
    """
    path = get_embedding_path(embedding_key)
    if not path.exists():
        raise FileNotFoundError(
            f"Verse embeddings not found at {path}. Encode them first with "
            f"build_and_evaluate_bertopic (compute_embeddings) for key "
            f"'{embedding_key}'."
        )
    return np.load(path)


# =============================================================================
# AGGREGATION (mirror of aggregate_documents, vectors instead of Counters)
# =============================================================================

def aggregate_embeddings(
    embeddings: np.ndarray,
    indices: List[int],
    aggregation_size: int = 20,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Aggregate verse embeddings into larger units by averaging.

    Mirrors :func:`topic_distances.aggregate_documents`: shuffles indices
    (seeded), batches into groups of ``aggregation_size``, and skips trailing
    batches smaller than ``aggregation_size // 2``. Each aggregated unit is the
    **L2-normalized mean** of its member verse embeddings, so cosine distance
    between units is ``1 - u·v``.

    Parameters
    ----------
    embeddings : np.ndarray
        (N, D) L2-normalized verse embeddings.
    indices : List[int]
        Row indices (into ``embeddings``) of documents to aggregate.
    aggregation_size : int, default=20
        Number of verses merged into each unit.
    random_seed : int, optional
        Seed for the shuffle (reseeded per call, matching the lexical code).

    Returns
    -------
    np.ndarray
        (n_units, D) array of normalized aggregated unit vectors.
    """
    if random_seed is not None:
        random.seed(random_seed)

    shuffled_indices = list(indices)
    random.shuffle(shuffled_indices)

    units = []
    for i in range(0, len(shuffled_indices), aggregation_size):
        batch = shuffled_indices[i:i + aggregation_size]
        if len(batch) < aggregation_size // 2:
            # Skip very small trailing batches (matches aggregate_documents)
            continue
        vec = embeddings[batch].mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            units.append(vec / norm)

    if not units:
        return np.zeros((0, embeddings.shape[1]), dtype=embeddings.dtype)
    return np.asarray(units)


def _mean_pairwise_cosine(units: np.ndarray, sample_size: int) -> Tuple[float, int]:
    """Mean cosine distance over sampled unique pairs of unit vectors."""
    n = len(units)
    if n < 2:
        return 0.0, 0
    n_total_pairs = n * (n - 1) // 2
    if sample_size > 0 and n_total_pairs > sample_size:
        pairs = _sample_pairs(n, sample_size)
    else:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return 0.0, 0
    i_idx = np.fromiter((p[0] for p in pairs), dtype=int, count=len(pairs))
    j_idx = np.fromiter((p[1] for p in pairs), dtype=int, count=len(pairs))
    dots = np.sum(units[i_idx] * units[j_idx], axis=1)
    return float(np.mean(1.0 - dots)), len(pairs)


def _mean_cross_cosine(inside: np.ndarray, outside: np.ndarray,
                       sample_size: int) -> Tuple[float, int]:
    """Mean cosine distance over sampled inside×outside unit pairs."""
    n_in, n_out = len(inside), len(outside)
    if n_in == 0 or n_out == 0:
        return 0.0, 0
    n_total_pairs = n_in * n_out
    if sample_size > 0 and n_total_pairs > sample_size:
        # Mirror the inside×outside sampler in _compute_inter_aggregated.
        pairs = []
        seen = set()
        max_attempts = sample_size * 3
        attempts = 0
        while len(pairs) < sample_size and attempts < max_attempts:
            i = random.randint(0, n_in - 1)
            j = random.randint(0, n_out - 1)
            if (i, j) not in seen:
                seen.add((i, j))
                pairs.append((i, j))
            attempts += 1
    else:
        pairs = [(i, j) for i in range(n_in) for j in range(n_out)]
    if not pairs:
        return 0.0, 0
    i_idx = np.fromiter((p[0] for p in pairs), dtype=int, count=len(pairs))
    j_idx = np.fromiter((p[1] for p in pairs), dtype=int, count=len(pairs))
    dots = np.sum(inside[i_idx] * outside[j_idx], axis=1)
    return float(np.mean(1.0 - dots)), len(pairs)


# =============================================================================
# SEMANTIC TOPIC-DISTANCE EVALUATION (mirror of evaluate_topic_distances)
# =============================================================================

def evaluate_semantic_topic_distances(
    embeddings: np.ndarray,
    topic_assignments: List[int],
    mode: str = 'intra_aggregated',
    aggregation_size: int = 20,
    sample_size: int = 5000,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, dict]:
    """Semantic analogue of :func:`topic_distances.evaluate_topic_distances`.

    Supports the same four modes:
    - ``intra_all_paired``  : cosine distance between verse pairs within a topic
    - ``inter_all_paired``  : cosine distance inside-vs-outside a topic
    - ``intra_aggregated``  : intra-topic on aggregated units (homogeneity)
    - ``inter_aggregated``  : inter-topic on aggregated units (separation)

    Outliers (topic == -1) are excluded, exactly as in the lexical suite.

    Returns
    -------
    dict
        ``{'cosine': {'mean', 'std', 'per_topic', 'n_topics',
        'total_documents', 'mode'}}`` — mean/std computed across topics, so the
        shape matches the lexical result and reuses the same figure helper.
    """
    embeddings = np.asarray(embeddings)
    if len(embeddings) != len(topic_assignments):
        raise ValueError(
            f"Length mismatch: {len(embeddings)} embeddings vs "
            f"{len(topic_assignments)} topic assignments"
        )

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Group indices by topic (exclude outliers).
    topic_doc_indices: Dict[int, List[int]] = {}
    all_non_outlier: List[int] = []
    for idx, topic in enumerate(topic_assignments):
        if topic == -1:
            continue
        topic_doc_indices.setdefault(topic, []).append(idx)
        all_non_outlier.append(idx)

    if not topic_doc_indices:
        return {'cosine': {'mean': 0.0, 'std': 0.0, 'per_topic': {},
                           'n_topics': 0, 'total_documents': 0, 'mode': mode}}

    topic_ids = sorted(topic_doc_indices.keys())
    aggregated = mode in ('intra_aggregated', 'inter_aggregated')

    # For inter modes, pre-build the per-topic units once (mirrors the lexical
    # optimization in _compute_inter_aggregated).
    per_topic_units: Dict[int, np.ndarray] = {}
    if aggregated:
        for tid in topic_ids:
            per_topic_units[tid] = aggregate_embeddings(
                embeddings, topic_doc_indices[tid], aggregation_size, random_seed
            )

    per_topic_results: Dict[int, dict] = {}
    topic_means: List[float] = []

    for tid in topic_ids:
        doc_idx = topic_doc_indices[tid]

        if mode == 'intra_all_paired':
            units = embeddings[doc_idx]
            mean_dist, n_pairs = _mean_pairwise_cosine(units, sample_size)
            per_topic_results[tid] = {
                'mean_distance': mean_dist, 'n_documents': len(doc_idx),
                'n_pairs_sampled': n_pairs}

        elif mode == 'intra_aggregated':
            units = per_topic_units[tid]
            mean_dist, n_pairs = _mean_pairwise_cosine(units, sample_size)
            per_topic_results[tid] = {
                'mean_distance': mean_dist, 'n_documents': len(doc_idx),
                'n_aggregated_units': len(units), 'n_pairs_sampled': n_pairs}

        elif mode == 'inter_all_paired':
            inside = embeddings[doc_idx]
            outside_idx = [i for i in all_non_outlier if topic_assignments[i] != tid]
            outside = embeddings[outside_idx]
            mean_dist, n_pairs = _mean_cross_cosine(inside, outside, sample_size)
            per_topic_results[tid] = {
                'mean_distance': mean_dist, 'n_inside_docs': len(doc_idx),
                'n_outside_docs': len(outside_idx), 'n_pairs_sampled': n_pairs}

        elif mode == 'inter_aggregated':
            inside = per_topic_units[tid]
            outside = (np.concatenate([per_topic_units[t] for t in topic_ids
                                       if t != tid and len(per_topic_units[t])])
                       if len(topic_ids) > 1 else np.zeros((0, embeddings.shape[1])))
            mean_dist, n_pairs = _mean_cross_cosine(inside, outside, sample_size)
            per_topic_results[tid] = {
                'mean_distance': mean_dist, 'n_inside_units': len(inside),
                'n_outside_units': len(outside), 'n_pairs_sampled': n_pairs}
        else:
            raise ValueError(f"Unknown mode: {mode}")

        topic_means.append(per_topic_results[tid]['mean_distance'])
        if verbose:
            print(f"    [cosine] Topic {tid}: mean={topic_means[-1]:.4f}")

    return {'cosine': {
        'mean': float(np.mean(topic_means)) if topic_means else 0.0,
        'std': float(np.std(topic_means)) if topic_means else 0.0,
        'per_topic': per_topic_results,
        'n_topics': len(topic_ids),
        'total_documents': len(all_non_outlier),
        'mode': mode,
    }}


def evaluate_semantic_multi_aggregation(
    embeddings: np.ndarray,
    topic_assignments: List[int],
    aggregation_sizes: List[int],
    modes: Optional[List[str]] = None,
    sample_size: int = 1000,
    random_seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[int, dict]:
    """Semantic analogue of :func:`topic_distances.evaluate_multi_aggregation`.

    Returns ``{agg_size: {mode: {'cosine': {'mean', 'std', ...}}}}`` — the same
    structure the lexical version returns (with ``'cosine'`` where the lexical
    one has ``'labbe'``), so ``create_aggregation_curve_plot`` renders it with
    ``metric_key='cosine'``.
    """
    if modes is None:
        modes = ['intra_aggregated', 'inter_aggregated']

    results: Dict[int, dict] = {}
    for agg_size in aggregation_sizes:
        if verbose:
            print(f"    Aggregation size = {agg_size}")
        results[agg_size] = {}
        for mode in modes:
            results[agg_size][mode] = evaluate_semantic_topic_distances(
                embeddings, topic_assignments, mode=mode,
                aggregation_size=agg_size, sample_size=sample_size,
                random_seed=random_seed, verbose=False)
    return results


def compute_semantic_centroid_distances(
    embeddings: np.ndarray,
    topic_assignments: List[int],
) -> Dict[str, dict]:
    """Semantic analogue of :func:`topic_distances.compute_topic_centroid_distances`.

    Per topic, the centroid is the L2-normalized mean of its verse vectors and
    the 'rest' is the normalized mean of all other verses; the reported
    distance is the cosine distance between them (one-vs-rest separation, no
    sampling). Returns ``{'cosine': {'mean', 'std', 'per_topic', ...}}``.
    """
    embeddings = np.asarray(embeddings)
    topic_doc_indices: Dict[int, List[int]] = {}
    for idx, topic in enumerate(topic_assignments):
        if topic == -1:
            continue
        topic_doc_indices.setdefault(topic, []).append(idx)

    topic_ids = sorted(topic_doc_indices.keys())
    if not topic_ids:
        return {'cosine': {'mean': 0.0, 'std': 0.0, 'per_topic': {}, 'n_topics': 0}}

    # Global sum for cheap "rest" centroid computation.
    all_idx = [i for tid in topic_ids for i in topic_doc_indices[tid]]
    total_sum = embeddings[all_idx].sum(axis=0)
    n_total = len(all_idx)

    per_topic: Dict[int, dict] = {}
    means: List[float] = []
    for tid in topic_ids:
        idx = topic_doc_indices[tid]
        c_sum = embeddings[idx].sum(axis=0)
        centroid = c_sum / max(len(idx), 1)
        rest = (total_sum - c_sum) / max(n_total - len(idx), 1)
        cn = np.linalg.norm(centroid)
        rn = np.linalg.norm(rest)
        dist = float(1.0 - np.dot(centroid / cn, rest / rn)) if cn > 0 and rn > 0 else 0.0
        per_topic[tid] = {'mean_distance': dist, 'n_documents': len(idx)}
        means.append(dist)

    return {'cosine': {
        'mean': float(np.mean(means)) if means else 0.0,
        'std': float(np.std(means)) if means else 0.0,
        'per_topic': per_topic,
        'n_topics': len(topic_ids),
    }}


# =============================================================================
# SEPARATION RATIO — SR = inter / intra (applies to lexical AND semantic)
# =============================================================================

def compute_separation_ratios(
    multi_agg_results: Dict[int, dict],
    metric_key: str = 'labbe',
) -> Dict[int, float]:
    """Per-aggregation-size separation ratio SR = inter_mean / intra_mean.

    Works on any ``multi_agg_results`` from either
    :func:`topic_distances.evaluate_multi_aggregation` (``metric_key='labbe'``
    or ``'js'``) or :func:`evaluate_semantic_multi_aggregation`
    (``metric_key='cosine'``). SR > 1 means between-class distances exceed
    within-class distances (better separation).

    Returns ``{aggregation_size: SR}`` (SR = 0.0 when intra is 0).
    """
    ratios: Dict[int, float] = {}
    for agg_size, modes in multi_agg_results.items():
        intra = modes.get('intra_aggregated', {}).get(metric_key, {}).get('mean', 0.0)
        inter = modes.get('inter_aggregated', {}).get(metric_key, {}).get('mean', 0.0)
        ratios[agg_size] = float(inter / intra) if intra else 0.0
    return ratios
