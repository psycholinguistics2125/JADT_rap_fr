#!/usr/bin/env python3
"""
Evaluate IRAMUTEQ Classification for French Rap Verses
======================================================
This script evaluates the IRAMUTEQ_CLASSES clustering already computed
on the French rap corpus, using the same metrics as LDA and BERTopic.
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import shared evaluation utilities
from utils.utils_evaluation import (
    compute_artist_separation,
    compute_temporal_separation,
    compute_cluster_metrics,
    save_artist_metrics,
    save_temporal_metrics,
    create_topic_distribution_plot,
    create_artist_topics_heatmap,
    create_artist_specialization_plot,
    create_biannual_js_plot,
    create_year_topic_heatmap,
    print_evaluation_summary,
)

warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = "/home/robin/Code_repo/psycholinguistic2125/JADT_rap_fr/results/IRAMUTEQ"
DATA_PATH = "/home/robin/Code_repo/psycholinguistic2125/JADT_rap_fr/data/20260123_filter_verses_lrfaf_corpus.csv"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(path: str, sample_size: int = None) -> pd.DataFrame:
    """Load the dataset with IRAMUTEQ classes, optionally sampling."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} verses")

    # Check for IRAMUTEQ_CLASSES column
    if 'IRAMUTEQ_CLASSES' not in df.columns:
        raise ValueError("IRAMUTEQ_CLASSES column not found in data!")

    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} documents for testing...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"Years range: {df['year'].min()} - {df['year'].max()}")
    print(f"Number of unique artists: {df['artist'].nunique()}")
    print(f"Number of IRAMUTEQ classes: {df['IRAMUTEQ_CLASSES'].nunique()}")

    return df


def analyze_class_distribution(topics: np.ndarray, df: pd.DataFrame) -> dict:
    """Analyze the distribution of IRAMUTEQ classes."""
    print("\n" + "="*60)
    print("IRAMUTEQ CLASS DISTRIBUTION")
    print("="*60)

    unique_classes = sorted(df['IRAMUTEQ_CLASSES'].unique())
    n_classes = len(unique_classes)

    print(f"\nNumber of classes: {n_classes}")
    print(f"Class IDs: {unique_classes}")

    # Size distribution
    class_counts = df['IRAMUTEQ_CLASSES'].value_counts().sort_index()
    print(f"\nClass sizes:")
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"  Class {cls:2d}: {count:6d} docs ({pct:5.2f}%)")

    return {
        'n_classes': int(n_classes),
        'class_ids': [int(c) for c in unique_classes],
        'class_counts': {int(k): int(v) for k, v in class_counts.to_dict().items()},
    }


def create_class_size_barplot(topics: np.ndarray, run_dir: str):
    """Create bar plot of class sizes."""
    unique_topics = sorted(np.unique(topics))

    class_counts = []
    class_labels = []
    for cls in unique_topics:
        class_counts.append((topics == cls).sum())
        class_labels.append(str(cls))

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(class_labels, class_counts, color='steelblue')

    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                str(count), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('IRAMUTEQ Class')
    ax.set_ylabel('Number of Documents')
    ax.set_title('IRAMUTEQ Class Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    print("  Saved class distribution plot")


def save_results(results: dict, df: pd.DataFrame, topics: np.ndarray, run_dir: str):
    """Save all results to the run directory."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save metrics as JSON
    metrics_to_save = {
        'cluster_metrics': results['cluster_metrics'],
        'artist_metrics': {
            'js_divergence': results['artist_separation'].get('mean_js_divergence'),
            'specialization': results['artist_separation'].get('artist_specialization'),
            'pct_specialists': results['artist_separation'].get('pct_specialists'),
            'pct_moderate': results['artist_separation'].get('pct_moderate'),
            'pct_generalists': results['artist_separation'].get('pct_generalists'),
            'mean_dominant_ratio': results['artist_separation'].get('mean_dominant_ratio'),
            'mean_significant_topics': results['artist_separation'].get('mean_significant_topics'),
            'mean_topic_concentration': results['artist_separation'].get('mean_topic_concentration'),
        },
        'temporal_metrics': {
            'mean_variance': results['temporal_separation']['mean_temporal_variance'],
            'decade_changes': results['temporal_separation'].get('decade_changes', {}),
            'biannual_changes': results['temporal_separation'].get('biannual_changes', {}),
            'mean_biannual_js': results['temporal_separation'].get('mean_biannual_js'),
        },
        'parameters': results['parameters'],
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved to: {metrics_path}")

    # Save class distribution
    class_dist_path = os.path.join(run_dir, "class_distribution.json")
    with open(class_dist_path, 'w', encoding='utf-8') as f:
        json.dump(results['class_distribution'], f, indent=2, ensure_ascii=False)
    print(f"  Class distribution saved to: {class_dist_path}")

    # Save artist metrics using shared utility
    save_artist_metrics(results['artist_separation'], run_dir)

    # Save temporal metrics using shared utility
    save_temporal_metrics(results['temporal_separation'], run_dir)

    # Save document assignments
    doc_assign_path = os.path.join(run_dir, "doc_assignments.csv")
    doc_df = df.copy()
    doc_df['iramuteq_class'] = topics
    doc_df[['artist', 'title', 'year', 'iramuteq_class']].to_csv(doc_assign_path, index=False)
    print(f"  Document assignments saved to: {doc_assign_path}")

    print(f"\n  All results saved to: {run_dir}")


def create_visualizations(results: dict, topics: np.ndarray, df: pd.DataFrame,
                          run_dir: str, top_n_artists: int = 50):
    """Create and save all visualizations using shared utilities."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # 1. Class size distribution (IRAMUTEQ-specific with value labels)
    create_class_size_barplot(topics, run_dir)

    # 2. Topic distribution (using shared utility)
    create_topic_distribution_plot(topics, run_dir, title="IRAMUTEQ Class Distribution")

    # 3. Year-class heatmap (shows class distribution over time, using shared utility)
    create_year_topic_heatmap(topics, df, run_dir,
                               title="IRAMUTEQ Class Distribution by Year")

    # 5. Artist topic profiles heatmap (using shared utility)
    create_artist_topics_heatmap(topics, df, run_dir, top_n_artists=top_n_artists,
                                  title=f"Top {top_n_artists} Artists - IRAMUTEQ Class Profiles")

    # 6. Artist specialization distribution (using shared utility)
    create_artist_specialization_plot(results['artist_separation'], run_dir)

    # 7. Biannual JS divergence plot (using shared utility)
    create_biannual_js_plot(results['temporal_separation'], run_dir,
                            title="IRAMUTEQ - 2-Year Window JS Divergence")

    print("  All visualizations saved!")


def run_evaluation(sample_size: int = None,
                   min_docs_per_artist: int = 10,
                   top_artists_per_topic: int = 20,
                   top_n_artists_heatmap: int = 50):
    """
    Run complete evaluation of IRAMUTEQ classification.
    """
    print("\n" + "="*60)
    print("IRAMUTEQ EVALUATION FOR FRENCH RAP CORPUS")
    print("="*60)

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"evaluation_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    # Load data
    df = load_data(DATA_PATH, sample_size=sample_size)

    # Get IRAMUTEQ classes as numpy array
    topics = df['IRAMUTEQ_CLASSES'].values

    # Analyze class distribution
    class_distribution = analyze_class_distribution(topics, df)

    # Compute all metrics using shared utilities
    cluster_metrics = compute_cluster_metrics(topics, df)
    artist_metrics = compute_artist_separation(
        topics, df,
        min_docs_per_artist=min_docs_per_artist,
        top_artists_per_topic=top_artists_per_topic
    )
    temporal_metrics = compute_temporal_separation(topics, df)

    # Compile results
    results = {
        'cluster_metrics': cluster_metrics,
        'class_distribution': class_distribution,
        'artist_separation': artist_metrics,
        'temporal_separation': temporal_metrics,
        'parameters': {
            'method': 'IRAMUTEQ',
            'n_classes': cluster_metrics['n_topics'],
            'n_documents': len(df),
            'min_docs_per_artist': min_docs_per_artist,
            'top_artists_per_topic': top_artists_per_topic,
            'timestamp': timestamp,
            'run_dir': run_dir,
        }
    }

    # Print summary
    print_evaluation_summary(results, method_name="IRAMUTEQ")

    # Save everything
    save_results(results, df, topics, run_dir)

    # Create visualizations
    create_visualizations(results, topics, df, run_dir, top_n_artists=top_n_artists_heatmap)

    return results, df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate IRAMUTEQ classification for French rap corpus')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for testing')
    parser.add_argument('--min-docs-artist', type=int, default=10, help='Min docs per artist')
    parser.add_argument('--top-artists-topic', type=int, default=20, help='Top N artists per topic')
    parser.add_argument('--top-artists-heatmap', type=int, default=50, help='Top N artists in heatmap')

    args = parser.parse_args()

    results, df = run_evaluation(
        sample_size=args.sample,
        min_docs_per_artist=args.min_docs_artist,
        top_artists_per_topic=args.top_artists_topic,
        top_n_artists_heatmap=args.top_artists_heatmap,
    )
