#!/usr/bin/env python3
"""
Shared Evaluation Utilities for Topic Modeling
===============================================
Common evaluation functions for LDA, BERTopic, and IRAMUTEQ clustering.

Why Jensen-Shannon Distance?
----------------------------
We use Jensen-Shannon (JS) distance throughout this evaluation framework for
comparing probability distributions (topic profiles between artists, temporal
evolution between time periods, etc.) for the following reasons:

NOTE: scipy.spatial.distance.jensenshannon() returns the JS **distance** (square
root of JS divergence), not the divergence itself. This is the value we use.

1. **Symmetry**: Unlike KL divergence, JS(P||Q) = JS(Q||P). This is important
   when comparing artists or time periods where there's no "reference" distribution.

2. **Bounded**: JS distance is always bounded between 0 and 1:
   - 0 = identical distributions
   - 1 = maximally different distributions

3. **Always defined**: JS distance is defined even when one distribution has
   zeros where the other doesn't (unlike KL divergence which goes to infinity).
   This is crucial for topic modeling where an artist may have zero documents
   in some topics.

4. **Metric property**: JS distance is a proper metric, satisfying the triangle
   inequality. This allows meaningful comparisons.

5. **Information-theoretic interpretation**: The squared JS distance (JS divergence)
   measures how much information is lost when using the average distribution
   M = (P+Q)/2 instead of P or Q individually.

Alternative measures considered:
- **Chi-square test**: Sensitive to sample size, not suitable for comparing
  distributions of different sizes.
- **Cosine similarity**: Doesn't account for the probabilistic nature of
  topic distributions.
- **Total Variation Distance**: Less sensitive to small differences in
  distribution tails.
- **Wasserstein distance**: Requires a metric on the topic space, which
  isn't naturally available for discrete topics.

References:
- Lin, J. (1991). Divergence measures based on the Shannon entropy.
  IEEE Transactions on Information Theory, 37(1), 145-151.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon


def compute_artist_separation(topics: np.ndarray, df: pd.DataFrame,
                              min_docs_per_artist: int = 10,
                              top_artists_per_topic: int = 20,
                              doc_topics: np.ndarray = None) -> dict:
    """
    Compute how well topics separate/distinguish artists.

    Key question: Do topics capture artist-specific styles?
    - If yes: artists should be "specialists" (concentrated in few topics)
    - If no: artists are "generalists" (spread across many topics)

    Args:
        topics: Array of topic assignments (dominant topic per document).
                For LDA, this should be argmax(doc_topics, axis=1).
        df: DataFrame with 'artist' column
        min_docs_per_artist: Minimum documents to include an artist
        top_artists_per_topic: Number of top artists to save per topic
        doc_topics: Optional array of topic probabilities per document (for LDA).
                    Shape: (n_docs, n_topics). If provided, artist profiles are
                    computed by averaging probabilities. If None, profiles are
                    computed by counting topic assignments.

    Returns:
        Dictionary with per-artist metrics, general metrics, and topic_top_artists
    """
    print("\nComputing artist separation metrics...")
    print("  Question: Do topics distinguish artists?")

    metrics = {}

    # Handle topic -1 (outliers) for BERTopic
    valid_topic_mask = topics >= 0
    unique_topics = np.unique(topics[valid_topic_mask])
    n_topics = len(unique_topics)
    max_entropy = np.log(n_topics)  # Maximum possible entropy

    # Create topic-to-index mapping (handles non-zero-indexed topics like IRAMUTEQ 1-20)
    topic_to_idx = {t: i for i, t in enumerate(sorted(unique_topics))}
    idx_to_topic = {i: t for t, i in topic_to_idx.items()}

    print(f"  Number of topics: {n_topics}")

    # Get artists with enough documents
    artist_counts = df['artist'].value_counts()
    valid_artists = artist_counts[artist_counts >= min_docs_per_artist].index.tolist()
    print(f"  Artists with >= {min_docs_per_artist} docs: {len(valid_artists)}")

    # =========================================================================
    # PER-ARTIST METRICS
    # =========================================================================
    artist_metrics = []

    for artist in valid_artists:
        artist_mask = (df['artist'] == artist).values
        artist_topics_all = topics[artist_mask]
        # Filter out outliers (topic -1)
        artist_topics = artist_topics_all[artist_topics_all >= 0]
        n_docs = len(artist_topics)

        if n_docs == 0:
            continue

        # Map topics to indices and compute distribution
        artist_topics_idx = np.array([topic_to_idx[t] for t in artist_topics])
        topic_counts = np.bincount(artist_topics_idx, minlength=n_topics)
        topic_probs = topic_counts / topic_counts.sum()

        # Entropy: low = specialist, high = generalist
        # Filter zeros to avoid numerical issues (0 * log(0) = 0 mathematically)
        entropy = stats.entropy(topic_probs[topic_probs > 0])
        normalized_entropy = entropy / max_entropy  # 0-1 scale

        # Dominant topic info (convert index back to original topic ID)
        dominant_idx = int(topic_probs.argmax())
        dominant_topic = int(idx_to_topic[dominant_idx])
        dominant_ratio = float(topic_probs.max())

        # Number of "significant" topics (>5% of artist's docs)
        n_significant_topics = int((topic_probs > 0.05).sum())

        # Top 3 topics for this artist (convert indices back to original topic IDs)
        top_3_idx = np.argsort(topic_probs)[::-1][:3].tolist()
        top_3_topics = [int(idx_to_topic[idx]) for idx in top_3_idx]
        top_3_ratios = [float(topic_probs[idx]) for idx in top_3_idx]

        # Classification
        if dominant_ratio >= 0.5:
            classification = 'specialist'
        elif dominant_ratio >= 0.3:
            classification = 'moderate'
        else:
            classification = 'generalist'

        # Gini coefficient (inequality measure)
        sorted_probs = np.sort(topic_probs)
        n = len(sorted_probs)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_probs) - (n + 1) * np.sum(sorted_probs)) / (n * np.sum(sorted_probs) + 1e-10)

        artist_metrics.append({
            'artist': artist,
            'n_docs': n_docs,
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'dominant_topic': dominant_topic,
            'dominant_ratio': dominant_ratio,
            'n_significant_topics': n_significant_topics,
            'top_3_topics': top_3_topics,
            'top_3_ratios': top_3_ratios,
            'gini_coefficient': float(gini),
            'classification': classification,
        })

    metrics['per_artist_metrics'] = artist_metrics

    # =========================================================================
    # GENERAL METRICS
    # =========================================================================
    if not artist_metrics:
        print("  WARNING: No valid artists found")
        return metrics

    entropies = [a['entropy'] for a in artist_metrics]
    dominant_ratios = [a['dominant_ratio'] for a in artist_metrics]
    n_sig_topics = [a['n_significant_topics'] for a in artist_metrics]
    classifications = [a['classification'] for a in artist_metrics]

    # 1. Entropy statistics
    metrics['mean_artist_entropy'] = float(np.mean(entropies))
    metrics['std_artist_entropy'] = float(np.std(entropies))
    metrics['median_artist_entropy'] = float(np.median(entropies))

    # 2. Specialization score (1 - normalized mean entropy)
    metrics['artist_specialization'] = float(1 - (np.mean(entropies) / max_entropy))

    # 3. Distribution of artist types
    n_specialists = classifications.count('specialist')
    n_moderate = classifications.count('moderate')
    n_generalists = classifications.count('generalist')
    total = len(classifications)

    metrics['pct_specialists'] = float(n_specialists / total * 100)
    metrics['pct_moderate'] = float(n_moderate / total * 100)
    metrics['pct_generalists'] = float(n_generalists / total * 100)

    print(f"\n  [ARTIST TYPE DISTRIBUTION]")
    print(f"    Specialists (>50% in 1 topic): {n_specialists} ({metrics['pct_specialists']:.1f}%)")
    print(f"    Moderate (30-50% in 1 topic):  {n_moderate} ({metrics['pct_moderate']:.1f}%)")
    print(f"    Generalists (<30% in 1 topic): {n_generalists} ({metrics['pct_generalists']:.1f}%)")

    # 4. Average dominant topic ratio
    metrics['mean_dominant_ratio'] = float(np.mean(dominant_ratios))
    metrics['median_dominant_ratio'] = float(np.median(dominant_ratios))
    print(f"\n  [TOPIC CONCENTRATION]")
    print(f"    Mean dominant topic ratio: {metrics['mean_dominant_ratio']:.2%}")
    print(f"    Median dominant topic ratio: {metrics['median_dominant_ratio']:.2%}")

    # 5. Average number of significant topics per artist
    metrics['mean_significant_topics'] = float(np.mean(n_sig_topics))
    print(f"    Mean significant topics per artist: {metrics['mean_significant_topics']:.1f}")

    # 6. Jensen-Shannon divergence between artists
    artist_topic_profiles = {}
    for a in artist_metrics:
        artist = a['artist']
        artist_mask = (df['artist'] == artist).values

        if doc_topics is not None:
            # LDA case: use probability matrix (average probabilities)
            artist_docs = doc_topics[artist_mask]
            if len(artist_docs) > 0:
                artist_topic_profiles[artist] = artist_docs.mean(axis=0)
        else:
            # BERTopic/IRAMUTEQ case: use topic counts
            artist_topics_all = topics[artist_mask]
            artist_topics = artist_topics_all[artist_topics_all >= 0]
            if len(artist_topics) > 0:
                # Map topics to indices for consistent bincount
                artist_topics_idx = np.array([topic_to_idx[t] for t in artist_topics])
                topic_counts = np.bincount(artist_topics_idx, minlength=n_topics)
                if topic_counts.sum() > 0:
                    artist_topic_profiles[artist] = topic_counts / topic_counts.sum()

    js_distances = []
    valid_profile_artists = list(artist_topic_profiles.keys())
    for i, artist1 in enumerate(valid_profile_artists):
        for artist2 in valid_profile_artists[i+1:]:
            js_dist = jensenshannon(
                artist_topic_profiles[artist1],
                artist_topic_profiles[artist2]
            )
            if not np.isnan(js_dist):
                js_distances.append(js_dist)

    if js_distances:
        # Note: variable named 'divergence' for backward compatibility, but value is JS distance
        metrics['mean_js_divergence'] = float(np.mean(js_distances))
        metrics['std_js_divergence'] = float(np.std(js_distances))
        print(f"\n  [INTER-ARTIST JS DISTANCE]")
        print(f"    Mean JS distance: {metrics['mean_js_divergence']:.4f} (higher = more different)")

    # 7. Topic purity: for each topic, how concentrated is it among few artists?
    topic_concentration = []
    for topic_id in unique_topics:
        topic_mask = topics == topic_id
        if topic_mask.sum() == 0:
            continue
        topic_artists = df.loc[topic_mask, 'artist'].value_counts(normalize=True)
        # Top 5 artists share what % of this topic?
        top5_share = topic_artists.head(5).sum()
        topic_concentration.append(float(top5_share))

    if topic_concentration:
        metrics['mean_topic_concentration'] = float(np.mean(topic_concentration))
        print(f"    Mean topic concentration (top 5 artists): {metrics['mean_topic_concentration']:.2%}")

    # 8. Overall interpretation
    print(f"\n  [INTERPRETATION]")
    if metrics['pct_specialists'] > 50:
        print(f"    ✓ Topics DISTINGUISH artists well ({metrics['pct_specialists']:.0f}% are specialists)")
    elif metrics['pct_specialists'] > 30:
        print(f"    ~ Topics MODERATELY distinguish artists ({metrics['pct_specialists']:.0f}% are specialists)")
    else:
        print(f"    ✗ Topics do NOT distinguish artists well ({metrics['pct_specialists']:.0f}% are specialists)")

    # =========================================================================
    # TOP ARTISTS PER TOPIC
    # =========================================================================
    topic_top_artists = {}
    for topic_id in unique_topics:
        topic_mask = topics == topic_id
        total_docs_in_topic = topic_mask.sum()

        if total_docs_in_topic == 0:
            topic_top_artists[int(topic_id)] = []
            continue

        topic_scores = {}
        for artist in valid_artists:
            artist_mask = (df['artist'] == artist).values
            n_docs_in_topic = (topic_mask & artist_mask).sum()
            if n_docs_in_topic > 0:
                pct_of_topic = (n_docs_in_topic / total_docs_in_topic) * 100
                topic_scores[artist] = {
                    'n_docs': int(n_docs_in_topic),
                    'pct_of_topic': float(pct_of_topic),
                    'total_topic_docs': int(total_docs_in_topic)
                }

        sorted_artists = sorted(topic_scores.items(), key=lambda x: x[1]['n_docs'], reverse=True)[:top_artists_per_topic]
        topic_top_artists[int(topic_id)] = sorted_artists

    metrics['topic_top_artists'] = topic_top_artists

    return metrics


def compute_temporal_separation(topics: np.ndarray, df: pd.DataFrame,
                                year_column: str = 'year',
                                doc_topics: np.ndarray = None) -> dict:
    """
    Compute how topics evolve over time.

    Args:
        topics: Array of topic assignments (dominant topic per document).
                For LDA, this should be argmax(doc_topics, axis=1).
        df: DataFrame with year column
        year_column: Name of the year column
        doc_topics: Optional array of topic probabilities per document (for LDA).
                    Shape: (n_docs, n_topics). If provided, temporal profiles are
                    computed by averaging probabilities. If None, profiles are
                    computed by counting topic assignments.

    Returns:
        Dictionary with temporal metrics
    """
    print("\nComputing temporal separation metrics...")

    metrics = {}

    # Determine n_topics based on input
    if doc_topics is not None:
        # LDA case: n_topics from probability matrix shape
        n_topics = doc_topics.shape[1]
        topic_to_idx = None  # Not needed for LDA
        # Filter valid years only (LDA has no outliers)
        valid_mask = df[year_column].notna()
        doc_topics_valid = doc_topics[valid_mask.values]
    else:
        # BERTopic/IRAMUTEQ case: determine from topic assignments
        valid_topic_mask = topics >= 0
        unique_topics = np.unique(topics[valid_topic_mask])
        n_topics = len(unique_topics)
        # Create topic-to-index mapping (handles non-zero-indexed topics like IRAMUTEQ 1-20)
        topic_to_idx = {t: i for i, t in enumerate(sorted(unique_topics))}
        # Filter valid years and non-outlier topics
        valid_mask = df[year_column].notna() & (topics >= 0)
        doc_topics_valid = None

    years = df.loc[valid_mask, year_column].astype(int).values
    topics_valid = topics[valid_mask.values]

    unique_years = sorted(np.unique(years))
    print(f"  Years covered: {min(unique_years)} - {max(unique_years)}")

    # 1. Topic distribution by year
    topic_by_year = {}
    for year in unique_years:
        year_mask = years == year
        if year_mask.sum() > 0:
            if doc_topics is not None:
                # LDA case: use probability matrix (average probabilities)
                topic_by_year[year] = doc_topics_valid[year_mask].mean(axis=0)
            else:
                # BERTopic/IRAMUTEQ case: use topic counts
                year_topics = topics_valid[year_mask]
                year_topics_idx = np.array([topic_to_idx[t] for t in year_topics])
                topic_counts = np.bincount(year_topics_idx, minlength=n_topics)
                topic_by_year[year] = topic_counts / topic_counts.sum()

    topic_evolution_df = pd.DataFrame(topic_by_year).T
    topic_evolution_df.index.name = 'year'

    # 2. Dominant topic distribution per year (always count-based for classification)
    dominant_by_year = {}
    for year in unique_years:
        year_mask = years == year
        if year_mask.sum() > 0:
            year_topics = topics_valid[year_mask]
            if doc_topics is not None:
                # LDA: topics are already 0-indexed
                topic_counts = np.bincount(year_topics, minlength=n_topics)
            else:
                year_topics_idx = np.array([topic_to_idx[t] for t in year_topics])
                topic_counts = np.bincount(year_topics_idx, minlength=n_topics)
            dominant_by_year[year] = topic_counts / topic_counts.sum()

    # 3. Trend correlations per topic
    trend_correlations = {}
    years_array = np.array(list(topic_by_year.keys()))

    for topic_id in range(n_topics):
        topic_values = [topic_by_year[y][topic_id] for y in years_array]
        if len(set(topic_values)) > 1:  # Need variance for correlation
            corr, p_value = stats.pearsonr(years_array, topic_values)
            trend_correlations[topic_id] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'trend': 'increasing' if corr > 0.3 else ('decreasing' if corr < -0.3 else 'stable')
            }
        else:
            trend_correlations[topic_id] = {'correlation': 0.0, 'p_value': 1.0, 'trend': 'stable'}

    metrics['trend_correlations'] = trend_correlations

    # 4. Temporal variance per topic
    temporal_variance = []
    for topic_id in range(n_topics):
        topic_values = [topic_by_year[y][topic_id] for y in years_array]
        temporal_variance.append(np.var(topic_values))

    metrics['mean_temporal_variance'] = float(np.mean(temporal_variance))
    metrics['topic_temporal_variance'] = [float(v) for v in temporal_variance]
    print(f"  Mean temporal variance: {metrics['mean_temporal_variance']:.6f}")

    # 5. Decade comparison
    if max(unique_years) - min(unique_years) >= 20:
        decades = {}
        for year in unique_years:
            decade = (year // 10) * 10
            if decade not in decades:
                decades[decade] = []
            decades[decade].append(year)

        decade_profiles = {}
        for decade, decade_years in decades.items():
            decade_mask = np.isin(years, decade_years)
            if decade_mask.sum() > 0:
                if doc_topics is not None:
                    # LDA case: use probability matrix
                    decade_profiles[decade] = doc_topics_valid[decade_mask].mean(axis=0)
                else:
                    # BERTopic/IRAMUTEQ case: use topic counts
                    decade_topics = topics_valid[decade_mask]
                    decade_topics_idx = np.array([topic_to_idx[t] for t in decade_topics])
                    topic_counts = np.bincount(decade_topics_idx, minlength=n_topics)
                    decade_profiles[decade] = topic_counts / topic_counts.sum()

        sorted_decades = sorted(decade_profiles.keys())
        decade_changes = {}
        for i in range(len(sorted_decades) - 1):
            d1, d2 = sorted_decades[i], sorted_decades[i+1]
            js_dist = jensenshannon(decade_profiles[d1], decade_profiles[d2])
            decade_changes[f"{d1}s->{d2}s"] = float(js_dist)

        metrics['decade_changes'] = decade_changes
        print(f"  Decade changes (JS distance): {decade_changes}")

    # 6. 2-year window JS distance
    if len(unique_years) >= 4:
        # Create 2-year windows
        min_year = min(unique_years)
        max_year = max(unique_years)

        # Align to even years for consistent windows
        start_year = min_year if min_year % 2 == 0 else min_year + 1

        window_profiles = {}
        for window_start in range(start_year, max_year - 1, 2):
            window_years = [window_start, window_start + 1]
            window_mask = np.isin(years, window_years)
            if window_mask.sum() >= 10:  # Need minimum docs
                if doc_topics is not None:
                    # LDA case: use probability matrix
                    window_profiles[window_start] = doc_topics_valid[window_mask].mean(axis=0)
                else:
                    # BERTopic/IRAMUTEQ case: use topic counts
                    window_topics = topics_valid[window_mask]
                    window_topics_idx = np.array([topic_to_idx[t] for t in window_topics])
                    topic_counts = np.bincount(window_topics_idx, minlength=n_topics)
                    if topic_counts.sum() > 0:
                        window_profiles[window_start] = topic_counts / topic_counts.sum()

        # Compute JS distance between consecutive windows
        sorted_windows = sorted(window_profiles.keys())
        window_changes = {}
        for i in range(len(sorted_windows) - 1):
            w1, w2 = sorted_windows[i], sorted_windows[i + 1]
            js_dist = jensenshannon(window_profiles[w1], window_profiles[w2])
            if not np.isnan(js_dist):
                window_changes[f"{w1}-{w1+1}->{w2}-{w2+1}"] = float(js_dist)

        metrics['biannual_changes'] = window_changes
        metrics['mean_biannual_js'] = float(np.mean(list(window_changes.values()))) if window_changes else 0.0

        print(f"  2-year window changes: {len(window_changes)} transitions")
        print(f"  Mean 2-year JS distance: {metrics['mean_biannual_js']:.4f}")

    metrics['topic_evolution'] = topic_evolution_df.to_dict()
    metrics['dominant_by_year'] = {int(k): v.tolist() for k, v in dominant_by_year.items()}

    return metrics


def compute_cluster_metrics(topics: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Compute basic cluster/topic metrics.

    Args:
        topics: Array of topic assignments
        df: DataFrame

    Returns:
        Dictionary with cluster metrics
    """
    print("\nComputing cluster metrics...")

    metrics = {}

    # Handle outliers
    valid_mask = topics >= 0
    unique_topics = np.unique(topics[valid_mask])
    n_topics = len(unique_topics)

    print(f"  Number of clusters/topics: {n_topics}")

    # Topic distribution
    topic_counts = {}
    for topic_id in unique_topics:
        count = (topics == topic_id).sum()
        topic_counts[int(topic_id)] = int(count)

    metrics['n_topics'] = n_topics
    metrics['topic_counts'] = topic_counts
    metrics['total_documents'] = len(topics)
    metrics['outliers'] = int((topics < 0).sum())

    # Distribution statistics
    counts = list(topic_counts.values())
    metrics['min_topic_size'] = min(counts)
    metrics['max_topic_size'] = max(counts)
    metrics['mean_topic_size'] = float(np.mean(counts))
    metrics['std_topic_size'] = float(np.std(counts))

    # Imbalance ratio (max/min)
    metrics['imbalance_ratio'] = max(counts) / min(counts) if min(counts) > 0 else float('inf')

    # Entropy of topic distribution (measure of balance)
    total = sum(counts)
    probs = [c / total for c in counts]
    metrics['distribution_entropy'] = float(stats.entropy(probs))
    metrics['max_possible_entropy'] = float(np.log(n_topics))
    metrics['normalized_entropy'] = metrics['distribution_entropy'] / metrics['max_possible_entropy']

    print(f"  Topic size range: {metrics['min_topic_size']} - {metrics['max_topic_size']}")
    print(f"  Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
    print(f"  Distribution entropy: {metrics['normalized_entropy']:.3f} (1.0 = perfectly balanced)")

    return metrics


def save_artist_metrics(artist_separation: dict, run_dir: str):
    """Save artist-related metrics to files."""
    # Save top artists per topic
    artists_path = os.path.join(run_dir, "topic_top_artists.csv")
    artists_data = []
    if 'topic_top_artists' in artist_separation:
        for topic_id, artists in artist_separation['topic_top_artists'].items():
            for rank, (artist, stats) in enumerate(artists, 1):
                artists_data.append({
                    'topic': topic_id,
                    'rank': rank,
                    'artist': artist,
                    'n_docs': stats['n_docs'],
                    'pct_of_topic': stats['pct_of_topic'],
                    'total_topic_docs': stats['total_topic_docs']
                })
    pd.DataFrame(artists_data).to_csv(artists_path, index=False)
    print(f"  Top artists per topic saved to: {artists_path}")

    # Save per-artist metrics
    if 'per_artist_metrics' in artist_separation and artist_separation['per_artist_metrics']:
        artist_metrics_path = os.path.join(run_dir, "artist_topic_metrics.csv")
        artist_metrics_df = pd.DataFrame(artist_separation['per_artist_metrics'])
        # Convert list columns to strings for CSV
        if 'top_3_topics' in artist_metrics_df.columns:
            artist_metrics_df['top_3_topics'] = artist_metrics_df['top_3_topics'].apply(lambda x: ','.join(map(str, x)))
        if 'top_3_ratios' in artist_metrics_df.columns:
            artist_metrics_df['top_3_ratios'] = artist_metrics_df['top_3_ratios'].apply(lambda x: ','.join(f'{r:.3f}' for r in x))
        artist_metrics_df.to_csv(artist_metrics_path, index=False)
        print(f"  Per-artist metrics saved to: {artist_metrics_path}")


def save_temporal_metrics(temporal_separation: dict, run_dir: str):
    """Save temporal metrics to files."""
    evolution_path = os.path.join(run_dir, "topic_evolution.csv")
    evolution_df = pd.DataFrame(temporal_separation['topic_evolution'])
    evolution_df.to_csv(evolution_path)
    print(f"  Topic evolution saved to: {evolution_path}")

    # Save 2-year window JS distance
    if 'biannual_changes' in temporal_separation and temporal_separation['biannual_changes']:
        biannual_path = os.path.join(run_dir, "biannual_js_divergence.csv")
        biannual_data = [
            {'window_transition': k, 'js_divergence': v}
            for k, v in temporal_separation['biannual_changes'].items()
        ]
        pd.DataFrame(biannual_data).to_csv(biannual_path, index=False)
        print(f"  2-year JS distance saved to: {biannual_path}")


def create_topic_distribution_plot(topics: np.ndarray, run_dir: str, title: str = "Topic Distribution"):
    """Create and save topic distribution bar plot."""
    valid_topics = topics[topics >= 0]
    unique_topics = sorted(np.unique(valid_topics))
    n_topics = len(unique_topics)

    # Count documents per topic (handles non-zero-indexed topics)
    topic_counts = [int((topics == t).sum()) for t in unique_topics]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(n_topics), topic_counts)
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels([str(t) for t in unique_topics])
    ax.set_xlabel('Topic/Cluster ID')
    ax.set_ylabel('Number of Documents')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'topic_distribution.png'), dpi=150)
    plt.close()
    print("  Saved topic distribution plot")


def create_artist_topics_heatmap(topics: np.ndarray, df: pd.DataFrame, run_dir: str,
                                  top_n_artists: int = 50, title: str = None):
    """Create and save artist topic profiles heatmap."""
    valid_topics = topics[topics >= 0]
    unique_topics = sorted(np.unique(valid_topics))
    n_topics = len(unique_topics)

    # Create topic-to-index mapping (handles non-zero-indexed topics)
    topic_to_idx = {t: i for i, t in enumerate(unique_topics)}

    valid_artists = df['artist'].value_counts().head(top_n_artists).index.tolist()
    artist_profiles = []

    for artist in valid_artists:
        mask = df['artist'] == artist
        artist_topics = topics[mask.values]
        valid_artist_topics = artist_topics[artist_topics >= 0]
        if len(valid_artist_topics) > 0:
            # Map topics to indices for consistent bincount
            artist_topics_idx = np.array([topic_to_idx[t] for t in valid_artist_topics])
            topic_counts = np.bincount(artist_topics_idx, minlength=n_topics)
            profile = topic_counts / topic_counts.sum()
        else:
            profile = np.zeros(n_topics)
        artist_profiles.append(profile)

    artist_profiles = np.array(artist_profiles)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(artist_profiles, cmap='YlOrRd', ax=ax,
                yticklabels=valid_artists, xticklabels=unique_topics)
    ax.set_xlabel('Topic/Cluster ID')
    ax.set_ylabel('Artist')
    ax.set_title(title or f'Top {top_n_artists} Artists - Topic Profiles')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'artist_topics_heatmap.png'), dpi=150)
    plt.close()
    print(f"  Saved artist topic profiles (top {top_n_artists})")


def print_evaluation_summary(results: dict, method_name: str = "Topic Model"):
    """Print a summary of evaluation metrics."""
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY - {method_name}")
    print("="*60)

    # Cluster metrics
    if 'cluster_metrics' in results:
        print("\n[CLUSTER METRICS]")
        cm = results['cluster_metrics']
        print(f"   Number of topics: {cm.get('n_topics')}")
        print(f"   Topic size range: {cm.get('min_topic_size')} - {cm.get('max_topic_size')}")
        print(f"   Imbalance ratio: {cm.get('imbalance_ratio', 0):.2f}")
        print(f"   Distribution entropy: {cm.get('normalized_entropy', 0):.3f}")

    # Artist separation
    if 'artist_separation' in results:
        print("\n[ARTIST SEPARATION]")
        asep = results['artist_separation']
        if asep.get('pct_specialists') is not None:
            print(f"   Specialists: {asep['pct_specialists']:.1f}%")
            print(f"   Moderate: {asep.get('pct_moderate', 0):.1f}%")
            print(f"   Generalists: {asep.get('pct_generalists', 0):.1f}%")
        if asep.get('mean_js_divergence') is not None:
            print(f"   Mean JS Distance: {asep['mean_js_divergence']:.4f}")
        if asep.get('artist_specialization') is not None:
            print(f"   Artist Specialization: {asep['artist_specialization']:.4f}")

    # Temporal evolution
    if 'temporal_separation' in results:
        print("\n[TEMPORAL EVOLUTION]")
        tsep = results['temporal_separation']
        print(f"   Mean Temporal Variance: {tsep.get('mean_temporal_variance', 0):.6f}")

        if 'mean_biannual_js' in tsep:
            print(f"   Mean 2-year JS Divergence: {tsep['mean_biannual_js']:.4f}")

        if 'decade_changes' in tsep:
            print("   Decade Changes (JS distance):")
            for period, change in tsep['decade_changes'].items():
                print(f"      {period}: {change:.4f}")

        if 'biannual_changes' in tsep and len(tsep['biannual_changes']) > 0:
            # Show summary stats for 2-year changes
            biannual_values = list(tsep['biannual_changes'].values())
            print(f"   2-year Window Changes: {len(biannual_values)} transitions")
            print(f"      Min: {min(biannual_values):.4f}, Max: {max(biannual_values):.4f}")

        if 'trend_correlations' in tsep:
            increasing = [tid for tid, t in tsep['trend_correlations'].items() if t['trend'] == 'increasing']
            decreasing = [tid for tid, t in tsep['trend_correlations'].items() if t['trend'] == 'decreasing']
            print(f"\n   Increasing topics: {increasing}")
            print(f"   Decreasing topics: {decreasing}")


def create_artist_specialization_plot(artist_separation: dict, run_dir: str):
    """
    Create plot showing distribution of artist specialization types.

    Creates a two-panel figure:
    - Left: Pie chart of artist classifications (specialist/moderate/generalist)
    - Right: Histogram of dominant topic ratios with threshold lines
    """
    if 'per_artist_metrics' not in artist_separation or not artist_separation['per_artist_metrics']:
        print("  Skipping artist specialization plot (no artist data)")
        return

    per_artist = artist_separation['per_artist_metrics']

    # Count classifications
    classifications = [a['classification'] for a in per_artist]
    class_counts = pd.Series(classifications).value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart of classifications
    colors = {'specialist': '#2ecc71', 'moderate': '#f39c12', 'generalist': '#e74c3c'}
    ax1 = axes[0]
    ax1.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
            colors=[colors.get(c, 'gray') for c in class_counts.index])
    ax1.set_title('Artist Classification Distribution')

    # Histogram of dominant ratios
    ax2 = axes[1]
    dominant_ratios = [a['dominant_ratio'] for a in per_artist]
    ax2.hist(dominant_ratios, bins=30, edgecolor='black', color='steelblue')
    ax2.axvline(x=0.5, color='green', linestyle='--', label='Specialist threshold (50%)')
    ax2.axvline(x=0.3, color='orange', linestyle='--', label='Moderate threshold (30%)')
    ax2.set_xlabel('Dominant Topic Ratio')
    ax2.set_ylabel('Number of Artists')
    ax2.set_title('Distribution of Artist Dominant Topic Ratios')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'artist_specialization.png'), dpi=150)
    plt.close()
    print("  Saved artist specialization plot")


def create_biannual_js_plot(temporal_separation: dict, run_dir: str,
                            title: str = "2-Year Window JS Divergence Over Time"):
    """
    Create line plot showing JS distance between consecutive 2-year windows.

    This visualizes how topic distributions change over time at a finer
    granularity than decade comparisons.
    """
    if 'biannual_changes' not in temporal_separation or not temporal_separation['biannual_changes']:
        print("  Skipping biannual JS plot (no biannual data)")
        return

    biannual = temporal_separation['biannual_changes']

    # Parse the window transitions to get start years
    transitions = list(biannual.keys())
    js_values = list(biannual.values())

    # Extract start year from transition string (e.g., "1992-1993->1994-1995" -> 1994)
    years = []
    for t in transitions:
        # Get the second window's start year
        second_window = t.split('->')[1]
        year = int(second_window.split('-')[0])
        years.append(year)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(years, js_values, marker='o', linewidth=2, markersize=6, color='steelblue')
    ax.fill_between(years, js_values, alpha=0.3, color='steelblue')

    # Add mean line
    mean_js = np.mean(js_values)
    ax.axhline(y=mean_js, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean JS: {mean_js:.4f}')

    ax.set_xlabel('Year (start of 2-year window)')
    ax.set_ylabel('JS Divergence from Previous Window')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotate x labels if many years
    if len(years) > 15:
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'biannual_js_divergence.png'), dpi=150)
    plt.close()
    print("  Saved biannual JS distance plot")


def create_year_topic_heatmap(topics: np.ndarray, df: pd.DataFrame, run_dir: str,
                               year_column: str = 'year',
                               title: str = "Topic Distribution by Year"):
    """
    Create heatmap of topic distribution by year (normalized per year).
    """
    # Handle outliers
    valid_mask = (topics >= 0) & df[year_column].notna()
    valid_topics = topics[valid_mask]
    valid_years = df.loc[valid_mask, year_column].astype(int)

    # Create cross-tabulation
    df_temp = pd.DataFrame({'year': valid_years, 'topic': valid_topics})
    year_topic = pd.crosstab(df_temp['year'], df_temp['topic'], normalize='index')

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(year_topic, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Topic/Cluster ID')
    ax.set_ylabel('Year')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'year_topic_heatmap.png'), dpi=150)
    plt.close()
    print("  Saved year-topic heatmap")


def create_all_standard_visualizations(results: dict, topics: np.ndarray, df: pd.DataFrame,
                                        run_dir: str, method_name: str = "Topic Model",
                                        top_n_artists: int = 50):
    """
    Create all standard visualizations for topic model evaluation.

    This function ensures consistent visualizations across LDA, BERTopic, and IRAMUTEQ.

    Creates:
    1. Topic distribution bar plot
    2. Year-topic heatmap (topic distribution over time)
    3. Artist topic profiles heatmap
    4. Artist specialization plot
    5. Biannual JS distance plot

    Args:
        results: Dictionary containing 'temporal_separation' and 'artist_separation'
        topics: Array of topic assignments
        df: DataFrame with 'artist' and 'year' columns
        run_dir: Directory to save visualizations
        method_name: Name of the method for plot titles
        top_n_artists: Number of top artists to show in heatmap
    """
    print("\n" + "="*60)
    print("CREATING STANDARD VISUALIZATIONS")
    print("="*60)

    # 1. Topic distribution
    create_topic_distribution_plot(topics, run_dir,
                                    title=f"{method_name} Topic Distribution")

    # 2. Year-topic heatmap (shows topic distribution over time)
    create_year_topic_heatmap(topics, df, run_dir,
                               title=f"{method_name} Topic Distribution by Year")

    # 3. Artist topic profiles heatmap
    create_artist_topics_heatmap(topics, df, run_dir, top_n_artists=top_n_artists,
                                  title=f"Top {top_n_artists} Artists - {method_name} Topic Profiles")

    # 4. Artist specialization plot
    if 'artist_separation' in results:
        create_artist_specialization_plot(results['artist_separation'], run_dir)

    # 5. Biannual JS distance plot
    if 'temporal_separation' in results:
        create_biannual_js_plot(results['temporal_separation'], run_dir,
                                 title=f"{method_name} - 2-Year Window JS Divergence")

    print("  All standard visualizations saved!")
