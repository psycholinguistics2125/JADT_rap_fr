#!/usr/bin/env python3
"""
Artist Separation Analysis (Q2)
================================
Functions for analyzing how well topic models separate artists.
"""

import numpy as np
import pandas as pd
from typing import List

from scipy.stats import chi2_contingency


def compute_cramers_v(topics: np.ndarray, artists: np.ndarray) -> float:
    """
    Compute Cramer's V statistic for association between topics and artists.

    V = sqrt(chi2 / (n * min(k-1, r-1)))

    Higher V = stronger association = topics better distinguish artists.
    """
    # Create contingency table
    contingency = pd.crosstab(artists, topics)

    # Compute chi-square
    chi2, p, dof, expected = chi2_contingency(contingency)

    n = contingency.sum().sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)

    if min_dim == 0:
        return 0.0

    return np.sqrt(chi2 / (n * min_dim))


def compute_standardized_residuals(topics: np.ndarray, artists: np.ndarray,
                                    min_docs: int = 10) -> pd.DataFrame:
    """
    Compute standardized Pearson residuals for artist-topic cross-tabulation.

    residual = (observed - expected) / sqrt(expected)

    Interpretation:
    - |r| > 1.96: significant at p < 0.05
    - |r| > 2.58: significant at p < 0.01

    Returns DataFrame of residuals (artists x topics).
    """
    # Filter artists with minimum documents
    artist_counts = pd.Series(artists).value_counts()
    valid_artists = artist_counts[artist_counts >= min_docs].index

    mask = np.isin(artists, valid_artists)
    filtered_artists = artists[mask]
    filtered_topics = topics[mask]

    # Create contingency table
    contingency = pd.crosstab(filtered_artists, filtered_topics)

    # Compute expected frequencies
    row_totals = contingency.sum(axis=1)
    col_totals = contingency.sum(axis=0)
    total = contingency.sum().sum()

    expected = np.outer(row_totals, col_totals) / total
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

    # Compute standardized residuals
    residuals = (contingency - expected_df) / np.sqrt(expected_df + 1e-10)

    return residuals


def compute_artist_separation_comparison(bertopic_data: dict, lda_data: dict,
                                          iramuteq_data: dict,
                                          min_docs_per_artist: int = 10) -> dict:
    """
    Compare artist separation metrics across all three models.
    """
    results = {}

    for name, data in [('bertopic', bertopic_data), ('lda', lda_data), ('iramuteq', iramuteq_data)]:
        df = data['doc_assignments']
        topics = df['topic'].values
        artists = df['artist'].values

        # Cramer's V
        results[f'{name}_cramers_v'] = compute_cramers_v(topics, artists)

        # Residuals
        results[f'{name}_residuals'] = compute_standardized_residuals(
            topics, artists, min_docs_per_artist
        )

        # Extract metrics from loaded data
        metrics = data.get('metrics', {})
        artist_metrics = metrics.get('artist_metrics', {})

        results[f'{name}_pct_specialists'] = artist_metrics.get('pct_specialists', 0)
        results[f'{name}_pct_moderate'] = artist_metrics.get('pct_moderate', 0)
        results[f'{name}_pct_generalists'] = artist_metrics.get('pct_generalists', 0)
        results[f'{name}_mean_entropy'] = artist_metrics.get('mean_artist_entropy', 0)
        results[f'{name}_mean_js_divergence'] = artist_metrics.get('mean_js_divergence', 0)

    return results


def get_top_residual_pairs(residuals: pd.DataFrame, top_n: int = 20) -> List[dict]:
    """
    Get top over-represented artist-topic pairs.
    """
    pairs = []
    for artist in residuals.index:
        for topic in residuals.columns:
            val = residuals.loc[artist, topic]
            if val > 1.96:  # Significant at p<0.05
                pairs.append({
                    'artist': artist,
                    'topic': topic,
                    'residual': round(val, 2)
                })

    # Sort by residual (descending)
    pairs.sort(key=lambda x: x['residual'], reverse=True)
    return pairs[:top_n]
