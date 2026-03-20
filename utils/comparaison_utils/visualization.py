#!/usr/bin/env python3
"""
Visualization Functions
=======================
Functions for creating plots and visualizations for topic model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Optional: plotly for Sankey diagrams
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_sankey_diagram(topics1: np.ndarray, topics2: np.ndarray,
                           name1: str, name2: str, output_path: str,
                           min_flow: int = 100):
    """
    Create Sankey diagram showing document flow between classifications.
    """
    if not PLOTLY_AVAILABLE:
        print("  Plotly not available, skipping Sankey diagram")
        return

    unique1 = sorted(set(topics1))
    unique2 = sorted(set(topics2))

    # Node labels
    node_labels = [f"{name1} T{t}" for t in unique1] + [f"{name2} T{t}" for t in unique2]

    # Generate colors
    colors1 = plt.cm.tab20(np.linspace(0, 1, len(unique1)))
    colors2 = plt.cm.tab20(np.linspace(0, 1, len(unique2)))
    node_colors = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.8)' for c in colors1]
    node_colors += [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.8)' for c in colors2]

    # Links
    source, target, value, link_colors = [], [], [], []
    for i, t1 in enumerate(unique1):
        for j, t2 in enumerate(unique2):
            count = int(((topics1 == t1) & (topics2 == t2)).sum())
            if count >= min_flow:
                source.append(i)
                target.append(len(unique1) + j)
                value.append(count)
                # Link color matches source
                c = colors1[i]
                link_colors.append(f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.4)')

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])

    fig.update_layout(
        title=f"Document Flow: {name1} → {name2}",
        font_size=10,
        width=1200,
        height=800
    )

    # Save HTML and PNG
    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path)
    try:
        fig.write_image(output_path)
    except Exception as e:
        print(f"  Could not save PNG (kaleido may be missing): {e}")


def create_agreement_heatmap(contingency: pd.DataFrame, title: str, output_path: str):
    """
    Create heatmap of topic contingency table.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Normalize by row for better visualization
    contingency_norm = contingency.div(contingency.sum(axis=1), axis=0) * 100

    sns.heatmap(
        contingency_norm,
        cmap='YlOrRd',
        annot=False,
        fmt='.0f',
        ax=ax,
        cbar_kws={'label': 'Percentage of source topic'}
    )

    ax.set_title(title)
    ax.set_xlabel('Target Topics')
    ax.set_ylabel('Source Topics')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_artist_specificity_heatmap(residuals: pd.DataFrame, model_name: str,
                                       output_path: str, top_n: int = 30):
    """
    Create heatmap of standardized residuals.
    """
    # Select top artists by max absolute residual
    max_abs = residuals.abs().max(axis=1)
    top_artists = max_abs.nlargest(top_n).index
    subset = residuals.loc[top_artists]

    fig, ax = plt.subplots(figsize=(14, 12))

    # Clip extreme values for better visualization
    clipped = subset.clip(-5, 5)

    sns.heatmap(
        clipped,
        cmap='RdBu_r',
        center=0,
        vmin=-5, vmax=5,
        ax=ax,
        cbar_kws={'label': 'Standardized Residual'}
    )

    ax.set_title(f"{model_name} - Artist-Topic Standardized Residuals\n(Red=over-represented, Blue=under-represented)")
    ax.set_xlabel('Topic')
    ax.set_ylabel('Artist')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_temporal_comparison_plot(bertopic_evo: pd.DataFrame, lda_evo: pd.DataFrame,
                                     iramuteq_evo: pd.DataFrame, output_path: str):
    """
    Create multi-panel figure comparing temporal evolution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    evolutions = [
        (bertopic_evo, 'BERTopic', axes[0, 0]),
        (lda_evo, 'LDA', axes[0, 1]),
        (iramuteq_evo, 'IRAMUTEQ', axes[1, 0])
    ]

    for evo, name, ax in evolutions:
        if evo.empty:
            ax.text(0.5, 0.5, f'{name}: No data', ha='center', va='center')
            continue

        # Stack plot
        evo.plot.area(ax=ax, alpha=0.7, legend=False)
        ax.set_title(f'{name} Topic Prevalence Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Proportion')

    # Bar chart of mean variance
    variances = {}
    for evo, name, _ in evolutions:
        if not evo.empty:
            variances[name] = evo.var().mean()

    ax = axes[1, 1]
    if variances:
        ax.bar(variances.keys(), variances.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_title('Mean Temporal Variance')
        ax.set_ylabel('Variance')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_vocabulary_comparison_plot(vocab_results: dict, output_path: str):
    """
    Create visualization of vocabulary overlap between corresponding topics.
    """
    comparisons = vocab_results.get('topic_comparisons', [])
    if not comparisons:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Jaccard similarity bar chart
    topics = [f"B{c['bertopic_topic']}-L{c['lda_topic']}" for c in comparisons]
    jaccards = [c['jaccard'] for c in comparisons]

    ax = axes[0]
    bars = ax.bar(range(len(topics)), jaccards, color='steelblue')
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha='right')
    ax.set_xlabel('Topic Pair (BERTopic-LDA)')
    ax.set_ylabel('Jaccard Similarity')
    ax.set_title('Vocabulary Overlap Between Corresponding Topics')
    ax.axhline(y=vocab_results['mean_jaccard'], color='red', linestyle='--', label=f"Mean: {vocab_results['mean_jaccard']:.3f}")
    ax.legend()

    # Histogram of Jaccard values
    ax = axes[1]
    ax.hist(jaccards, bins=10, color='steelblue', edgecolor='white')
    ax.axvline(x=vocab_results['mean_jaccard'], color='red', linestyle='--', label=f"Mean: {vocab_results['mean_jaccard']:.3f}")
    ax.set_xlabel('Jaccard Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Vocabulary Overlap')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_corpus_year_distribution(df: pd.DataFrame, output_path: str):
    """Create year distribution histogram for the corpus."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Year distribution
    year_col = df['year'].dropna().astype(int)
    ax = axes[0]
    year_counts = year_col.value_counts().sort_index()
    ax.bar(year_counts.index, year_counts.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Documents')
    ax.set_title('Document Distribution by Year')
    ax.tick_params(axis='x', rotation=45)

    # Artist distribution (log scale)
    ax = axes[1]
    artist_counts = df['artist'].value_counts()
    ax.hist(artist_counts.values, bins=50, color='coral', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Documents per Artist')
    ax.set_ylabel('Number of Artists (log scale)')
    ax.set_title('Distribution of Artist Productivity')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_decade_breakdown_plot(df: pd.DataFrame, output_path: str):
    """Create decade breakdown visualization."""
    year_col = df['year'].dropna().astype(int)
    decades = (year_col // 10 * 10).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(decades)))
    bars = ax.bar(decades.index.astype(str) + 's', decades.values, color=colors)

    ax.set_xlabel('Decade')
    ax.set_ylabel('Number of Documents')
    ax.set_title('Corpus Distribution by Decade')

    # Add value labels on bars
    for bar, val in zip(bars, decades.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{val:,}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Q5: AGGREGATION STABILIZATION CURVE
# =============================================================================

MODEL_COLORS = {'bertopic': '#1f77b4', 'lda': '#ff7f0e', 'iramuteq': '#2ca02c'}
MODEL_MARKERS = {'bertopic': 'o', 'lda': 's', 'iramuteq': '^'}


def create_aggregation_curve_plot(
    multi_agg_results: dict,
    output_path: str,
    model_names: list = None,
):
    """
    Plot aggregation size vs mean Labbé distance (intra and inter).

    Creates a 2-panel figure:
    - Left: intra-topic Labbé distance (homogeneity) — lower is better
    - Right: inter-topic Labbé distance (separation) — higher is better

    One line per model, all on the same axes for direct comparison.

    Parameters
    ----------
    multi_agg_results : dict
        {model_name: {agg_size: {mode: {'labbe': {'mean': ..., 'std': ...}}}}}
        Output from evaluate_multi_aggregation() for each model.
    output_path : str
        Path to save PNG.
    model_names : list, optional
        Model names in display order. Default: sorted keys.
    """
    if model_names is None:
        model_names = sorted(multi_agg_results.keys())

    fig, (ax_intra, ax_inter) = plt.subplots(1, 2, figsize=(14, 6))

    for model in model_names:
        model_data = multi_agg_results.get(model, {})
        if not model_data:
            continue

        agg_sizes = sorted(model_data.keys())
        color = MODEL_COLORS.get(model, '#333333')
        marker = MODEL_MARKERS.get(model, 'o')

        # Intra-aggregated Labbé
        intra_means = []
        intra_stds = []
        for agg in agg_sizes:
            intra = model_data[agg].get('intra_aggregated', {}).get('labbe', {})
            intra_means.append(intra.get('mean', 0))
            intra_stds.append(intra.get('std', 0))

        intra_means = np.array(intra_means)
        intra_stds = np.array(intra_stds)

        ax_intra.plot(agg_sizes, intra_means, color=color, marker=marker,
                      label=model.upper(), linewidth=2, markersize=6)
        ax_intra.fill_between(agg_sizes,
                              intra_means - intra_stds,
                              intra_means + intra_stds,
                              color=color, alpha=0.15)

        # Inter-aggregated Labbé
        inter_means = []
        inter_stds = []
        for agg in agg_sizes:
            inter = model_data[agg].get('inter_aggregated', {}).get('labbe', {})
            inter_means.append(inter.get('mean', 0))
            inter_stds.append(inter.get('std', 0))

        inter_means = np.array(inter_means)
        inter_stds = np.array(inter_stds)

        ax_inter.plot(agg_sizes, inter_means, color=color, marker=marker,
                      label=model.upper(), linewidth=2, markersize=6)
        ax_inter.fill_between(agg_sizes,
                              inter_means - inter_stds,
                              inter_means + inter_stds,
                              color=color, alpha=0.15)

    ax_intra.set_xlabel('Aggregation size (documents)')
    ax_intra.set_ylabel('Mean Labbé distance')
    ax_intra.set_title('Intra-topic (homogeneity)')
    ax_intra.legend()
    ax_intra.grid(True, alpha=0.3)

    ax_inter.set_xlabel('Aggregation size (documents)')
    ax_inter.set_ylabel('Mean Labbé distance')
    ax_inter.set_title('Inter-topic (separation)')
    ax_inter.legend()
    ax_inter.grid(True, alpha=0.3)

    fig.suptitle('Labbé Distance Stabilization by Aggregation Size',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Q5: INTER-TOPIC SEPARATION RANKING
# =============================================================================

def create_inter_topic_ranking_plot(
    inter_results: dict,
    model_name: str,
    output_path: str,
    topic_labels: dict = None,
):
    """
    Horizontal bar chart of per-topic inter-distance (one-vs-rest), sorted.

    Creates a 2-panel figure: left=Labbé, right=JS.
    Topics are sorted by mean Labbé inter-distance (same order for both).

    Parameters
    ----------
    inter_results : dict
        The model's inter_all_paired results. Expected structure:
        {'labbe': {'per_topic': {topic_id: {'mean_distance': float}}},
         'js': {'per_topic': {topic_id: {'mean_distance': float}}}}
    model_name : str
        Model display name (e.g., 'BERTOPIC').
    output_path : str
        Path to save PNG.
    topic_labels : dict, optional
        Mapping from topic_id to human-readable label.
    """
    labbe_per_topic = inter_results.get('labbe', {}).get('per_topic', {})
    js_per_topic = inter_results.get('js', {}).get('per_topic', {})

    if not labbe_per_topic:
        return

    # Sort topics by Labbé inter-distance ascending
    sorted_topics = sorted(
        labbe_per_topic.items(),
        key=lambda x: x[1].get('mean_distance', 0)
    )

    topic_ids = [t[0] for t in sorted_topics]
    labbe_dists = [t[1].get('mean_distance', 0) for t in sorted_topics]

    # Get JS distances in the same topic order
    js_dists = [
        js_per_topic.get(tid, {}).get('mean_distance', 0)
        for tid in topic_ids
    ]

    # Topic labels
    labels = []
    for tid in topic_ids:
        if topic_labels and tid in topic_labels:
            labels.append(f"T{tid}: {topic_labels[tid]}")
        else:
            labels.append(f"Topic {tid}")

    n_topics = len(topic_ids)
    fig_height = max(6, n_topics * 0.4)
    fig, (ax_labbe, ax_js) = plt.subplots(1, 2, figsize=(14, fig_height))

    y_pos = np.arange(n_topics)

    # Labbé panel
    ax_labbe.barh(y_pos, labbe_dists, color='steelblue', height=0.7)
    ax_labbe.set_yticks(y_pos)
    ax_labbe.set_yticklabels(labels)
    ax_labbe.set_xlabel('Mean Labbé inter-distance')
    ax_labbe.set_title(f'{model_name} — Labbé')
    ax_labbe.grid(True, axis='x', alpha=0.3)

    # JS panel
    ax_js.barh(y_pos, js_dists, color='coral', height=0.7)
    ax_js.set_yticks(y_pos)
    ax_js.set_yticklabels([])  # Labels already on left panel
    ax_js.set_xlabel('Mean JS inter-distance')
    ax_js.set_title(f'{model_name} — Jensen-Shannon')
    ax_js.grid(True, axis='x', alpha=0.3)

    fig.suptitle(f'{model_name} — Inter-Topic Separation Ranking (one-vs-rest)',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
