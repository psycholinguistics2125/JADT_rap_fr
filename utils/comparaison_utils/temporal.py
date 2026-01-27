#!/usr/bin/env python3
"""
Temporal Analysis (Q3)
======================
Functions for analyzing temporal evolution of topics.
"""

import pandas as pd

from scipy.spatial.distance import jensenshannon


def compute_temporal_comparison(bertopic_evolution: pd.DataFrame,
                                 lda_evolution: pd.DataFrame,
                                 iramuteq_evolution: pd.DataFrame) -> dict:
    """
    Compare temporal dynamics across models.
    """
    results = {}

    for name, evolution in [('bertopic', bertopic_evolution),
                            ('lda', lda_evolution),
                            ('iramuteq', iramuteq_evolution)]:
        if evolution.empty:
            results[f'{name}_mean_variance'] = 0
            results[f'{name}_topic_variances'] = []
            continue

        # Compute variance per topic
        variances = evolution.var()
        results[f'{name}_mean_variance'] = float(variances.mean())
        results[f'{name}_topic_variances'] = variances.to_dict()

        # Most variable topic
        max_var_topic = variances.idxmax()
        results[f'{name}_most_variable_topic'] = str(max_var_topic)
        results[f'{name}_max_variance'] = float(variances.max())

    return results


def compute_decade_js_divergence(evolution: pd.DataFrame) -> dict:
    """
    Compute JS distance between decades.
    """
    if evolution.empty:
        return {}

    decades = {}
    for year in evolution.index:
        try:
            y = int(year)
            decade = f"{(y // 10) * 10}s"
            if decade not in decades:
                decades[decade] = []
            decades[decade].append(evolution.loc[year])
        except (ValueError, TypeError):
            continue

    # Average distribution per decade
    decade_dists = {}
    for decade, dists in sorted(decades.items()):
        decade_dists[decade] = pd.concat(dists, axis=1).mean(axis=1)

    # Compute JS distance between consecutive decades
    js_results = {}
    decade_list = list(decade_dists.keys())
    for i in range(len(decade_list) - 1):
        d1, d2 = decade_list[i], decade_list[i + 1]
        dist1 = decade_dists[d1].values
        dist2 = decade_dists[d2].values

        # Normalize to ensure they sum to 1
        dist1 = dist1 / dist1.sum() if dist1.sum() > 0 else dist1
        dist2 = dist2 / dist2.sum() if dist2.sum() > 0 else dist2

        js = jensenshannon(dist1, dist2)
        js_results[f"{d1}->{d2}"] = float(js)

    return js_results
