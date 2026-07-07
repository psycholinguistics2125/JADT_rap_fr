#!/usr/bin/env python3
"""
Sanity checks for the semantic (embedding-space) Q5 counterpart and the
cross-embedding cluster-quality metrics.

Runnable directly (``python test_semantic_evaluation.py``) or via pytest.
Uses synthetic embeddings so it has no dependency on the corpus/caches.
"""

import numpy as np

from utils.comparaison_utils import (
    aggregate_embeddings,
    evaluate_semantic_topic_distances,
    evaluate_semantic_multi_aggregation,
    compute_semantic_centroid_distances,
    compute_separation_ratios,
    compute_cluster_quality,
)


def _make_clusters(n_per=400, k=4, dim=32, spread=0.15, seed=0):
    """K well-separated, L2-normalized clusters (mirrors the real caches)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X, y = [], []
    for c in range(k):
        X.append(centers[c] + spread * rng.normal(size=(n_per, dim)))
        y += [c] * n_per
    X = np.vstack(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, np.array(y)


def test_aggregate_embeddings_normalized():
    X, y = _make_clusters()
    units = aggregate_embeddings(X, list(np.where(y == 0)[0]), aggregation_size=20,
                                 random_seed=42)
    assert units.shape[1] == X.shape[1]
    assert np.allclose(np.linalg.norm(units, axis=1), 1.0, atol=1e-5)


def test_separation_ratio_gt_one_for_separated_clusters():
    X, y = _make_clusters()
    res = evaluate_semantic_multi_aggregation(
        X, y, aggregation_sizes=[5, 20], sample_size=1000, random_seed=42)
    sr = compute_separation_ratios(res, metric_key='cosine')
    # Between-class distances must exceed within-class distances.
    assert all(v > 1.0 for v in sr.values()), sr


def test_modes_and_centroid_run():
    X, y = _make_clusters()
    for mode in ('intra_all_paired', 'inter_all_paired',
                 'intra_aggregated', 'inter_aggregated'):
        r = evaluate_semantic_topic_distances(X, y, mode=mode, aggregation_size=20,
                                              sample_size=500, random_seed=42)
        assert 'cosine' in r and r['cosine']['n_topics'] == 4
    cen = compute_semantic_centroid_distances(X, y)
    assert cen['cosine']['mean'] > 0.5  # separated clusters -> large centroid gap


def test_cluster_quality_real_vs_shuffled():
    X, y = _make_clusters()
    real = compute_cluster_quality(X, y, sample_size=500, seed=42,
                                   clustered_space='e5', eval_space='solon')
    y_shuf = y.copy()
    np.random.default_rng(1).shuffle(y_shuf)
    shuf = compute_cluster_quality(X, y_shuf, sample_size=500, seed=42)

    assert real['silhouette_cosine'] > 0.3
    assert abs(shuf['silhouette_cosine']) < 0.1          # ~0 for random labels
    assert real['calinski_harabasz'] > 50 * shuf['calinski_harabasz']
    assert real['circular'] is False                     # e5 != solon


def test_circular_flag():
    X, y = _make_clusters()
    r = compute_cluster_quality(X, y, sample_size=500, seed=42,
                                clustered_space='camembert', eval_space='camembert')
    assert r['circular'] is True


def test_determinism():
    X, y = _make_clusters()
    a = evaluate_semantic_multi_aggregation(X, y, [10], sample_size=500, random_seed=42)
    b = evaluate_semantic_multi_aggregation(X, y, [10], sample_size=500, random_seed=42)
    assert (a[10]['intra_aggregated']['cosine']['mean']
            == b[10]['intra_aggregated']['cosine']['mean'])


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} semantic-evaluation sanity checks passed.")
