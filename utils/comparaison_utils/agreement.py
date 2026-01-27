#!/usr/bin/env python3
"""
Model Agreement Metrics (Q1)
============================
Functions for computing agreement metrics between topic model classifications.
"""

import numpy as np
import pandas as pd
from typing import List

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)


def compute_pairwise_agreement(topics1: np.ndarray, topics2: np.ndarray) -> dict:
    """
    Compute ARI, NMI, and AMI between two topic assignments.

    Args:
        topics1, topics2: Arrays of topic assignments (same length)

    Returns:
        Dictionary with 'ari', 'nmi', 'ami' scores.
    """
    return {
        'ari': adjusted_rand_score(topics1, topics2),
        'nmi': normalized_mutual_info_score(topics1, topics2, average_method='arithmetic'),
        'ami': adjusted_mutual_info_score(topics1, topics2),
    }


def compute_contingency_analysis(topics1: np.ndarray, topics2: np.ndarray,
                                  name1: str = "Model1", name2: str = "Model2") -> dict:
    """
    Build contingency table and identify topic correspondences.

    Returns:
        Dictionary with contingency_table, correspondences, fragmented_topics.
    """
    unique1 = sorted(set(topics1))
    unique2 = sorted(set(topics2))

    # Build contingency table
    contingency = pd.DataFrame(
        np.zeros((len(unique1), len(unique2)), dtype=int),
        index=[f"{name1}_T{t}" for t in unique1],
        columns=[f"{name2}_T{t}" for t in unique2]
    )

    for t1, t2 in zip(topics1, topics2):
        contingency.loc[f"{name1}_T{t1}", f"{name2}_T{t2}"] += 1

    # Find correspondences (best matching topic for each topic in model1)
    correspondences = []
    for i, t1 in enumerate(unique1):
        row = contingency.iloc[i]
        best_match_idx = row.argmax()
        best_match = unique2[best_match_idx]
        overlap = row.iloc[best_match_idx]
        total_t1 = row.sum()
        overlap_pct = overlap / total_t1 * 100 if total_t1 > 0 else 0
        correspondences.append({
            f'{name1}_topic': t1,
            f'{name2}_topic': best_match,
            'overlap_count': int(overlap),
            'overlap_pct': round(overlap_pct, 1),
            'total_in_source': int(total_t1)
        })

    # Identify fragmented topics (topics that split across multiple targets)
    fragmented = []
    for i, t1 in enumerate(unique1):
        row = contingency.iloc[i]
        total = row.sum()
        if total > 0:
            significant_targets = (row / total > 0.2).sum()  # >20% of documents
            if significant_targets > 1:
                fragmented.append({
                    'topic': t1,
                    'n_targets': int(significant_targets),
                    'distribution': {unique2[j]: int(row.iloc[j]) for j in range(len(unique2)) if row.iloc[j] > 0}
                })

    return {
        'contingency_table': contingency,
        'correspondences': correspondences,
        'fragmented_topics': fragmented,
        'n_one_to_one': len([c for c in correspondences if c['overlap_pct'] > 50])
    }


def compute_all_pairwise_agreements(bertopic_topics: np.ndarray,
                                     lda_topics: np.ndarray,
                                     iramuteq_topics: np.ndarray) -> dict:
    """
    Compute agreement metrics for all three model pairs.
    """
    results = {}

    # BERTopic vs LDA
    results['bertopic_vs_lda'] = {
        'agreement': compute_pairwise_agreement(bertopic_topics, lda_topics),
        'contingency': compute_contingency_analysis(bertopic_topics, lda_topics, "BERTopic", "LDA")
    }

    # BERTopic vs IRAMUTEQ
    results['bertopic_vs_iramuteq'] = {
        'agreement': compute_pairwise_agreement(bertopic_topics, iramuteq_topics),
        'contingency': compute_contingency_analysis(bertopic_topics, iramuteq_topics, "BERTopic", "IRAMUTEQ")
    }

    # LDA vs IRAMUTEQ
    results['lda_vs_iramuteq'] = {
        'agreement': compute_pairwise_agreement(lda_topics, iramuteq_topics),
        'contingency': compute_contingency_analysis(lda_topics, iramuteq_topics, "LDA", "IRAMUTEQ")
    }

    return results
