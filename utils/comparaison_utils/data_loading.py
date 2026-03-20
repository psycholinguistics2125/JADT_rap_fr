#!/usr/bin/env python3
"""
Data Loading Functions for Topic Model Comparison
==================================================
Functions for loading and aligning data from LDA, BERTopic, and IRAMUTEQ runs.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_run_data(run_dir: str, model_type: str,
                  iramuteq_original_dir: str = None) -> dict:
    """
    Load all data from a topic model run directory.

    Args:
        run_dir: Path to the run folder
        model_type: One of 'bertopic', 'lda', 'iramuteq'
        iramuteq_original_dir: Optional path to original IRAMUTEQ run with profiles.csv

    Returns:
        Dictionary containing doc_assignments, metrics, topics, topic_evolution, etc.
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    data = {
        'model_type': model_type,
        'run_dir': str(run_dir),
    }

    # Load doc_assignments.csv
    doc_assignments_path = run_dir / 'doc_assignments.csv'
    if doc_assignments_path.exists():
        data['doc_assignments'] = pd.read_csv(doc_assignments_path)
        data['doc_assignments'] = normalize_topic_column(data['doc_assignments'], model_type)

        # Add original_index if not present (needed for text merging in Q5)
        # IRAMUTEQ typically doesn't have this column, so we add it based on row order
        if 'original_index' not in data['doc_assignments'].columns:
            data['doc_assignments']['original_index'] = data['doc_assignments'].index
    else:
        raise FileNotFoundError(f"doc_assignments.csv not found in {run_dir}")

    # Load metrics.json
    metrics_path = run_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            data['metrics'] = json.load(f)
    else:
        data['metrics'] = {}

    # Load topics.json (BERTopic and LDA only)
    topics_path = run_dir / 'topics.json'
    if topics_path.exists():
        with open(topics_path, 'r') as f:
            data['topics'] = json.load(f)
    else:
        data['topics'] = {}

    # Load topic_evolution.csv
    evolution_path = run_dir / 'topic_evolution.csv'
    if evolution_path.exists():
        data['topic_evolution'] = pd.read_csv(evolution_path, index_col=0)
    else:
        data['topic_evolution'] = pd.DataFrame()

    # Load artist_topic_metrics.csv
    artist_metrics_path = run_dir / 'artist_topic_metrics.csv'
    if artist_metrics_path.exists():
        data['artist_topic_metrics'] = pd.read_csv(artist_metrics_path)
    else:
        data['artist_topic_metrics'] = pd.DataFrame()

    # Load annual/biannual JS divergence (try annual first, fall back to biannual)
    annual_js_path = run_dir / 'annual_js_divergence.csv'
    biannual_js_path = run_dir / 'biannual_js_divergence.csv'
    if annual_js_path.exists():
        data['annual_js_divergence'] = pd.read_csv(annual_js_path)
    elif biannual_js_path.exists():
        data['annual_js_divergence'] = pd.read_csv(biannual_js_path)
    else:
        data['annual_js_divergence'] = pd.DataFrame()

    # For IRAMUTEQ, try to load vocabulary from original run
    if model_type == 'iramuteq' and iramuteq_original_dir:
        original_dir = Path(iramuteq_original_dir)
        profiles_path = original_dir / 'profiles.csv'
        if profiles_path.exists():
            data['topics'] = load_iramuteq_vocabulary(profiles_path)

    return data


def load_iramuteq_vocabulary(profiles_path: str) -> dict:
    """
    Load IRAMUTEQ vocabulary from profiles.csv.

    The profiles.csv format (semicolon separated):
    - Lines with "classe" indicate start of a new class
    - Following lines contain: count_in_class, total_count, pct, chi2, word, pvalue

    Returns:
        Dictionary mapping class IDs to word lists with chi-square scores.
    """
    profiles_path = Path(profiles_path)
    topics = {}

    current_class = None

    with open(profiles_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse semicolon-separated values
            parts = [p.strip().strip('"') for p in line.split(';')]

            # Check if this is a class header line
            if len(parts) >= 3 and parts[1] == 'classe':
                try:
                    current_class = int(parts[2])
                    topics[str(current_class)] = {
                        'words': [],
                        'chi2_scores': [],
                        'top_words': ''
                    }
                except ValueError:
                    continue
            elif current_class is not None and len(parts) >= 6:
                # Data line: count_in_class, total_count, pct, chi2, word, pvalue
                try:
                    word = parts[4].strip()
                    chi2_score = float(parts[3])
                    if word and not word.startswith('*'):
                        topics[str(current_class)]['words'].append(word)
                        topics[str(current_class)]['chi2_scores'].append(chi2_score)
                except (ValueError, IndexError):
                    continue

    # Create top_words summary for each class
    for class_id, topic_data in topics.items():
        if topic_data['words']:
            topics[class_id]['top_words'] = ', '.join(topic_data['words'][:30])

    return topics


def normalize_topic_column(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """
    Normalize topic column names across models to 'topic'.

    - BERTopic: 'topic' -> 'topic' (no change)
    - LDA: 'dominant_topic' -> 'topic'
    - IRAMUTEQ: 'iramuteq_class' -> 'topic' (keeps original 1-indexed values)

    Note: Each model may use different indexing schemes (LDA: 0-indexed,
    IRAMUTEQ: 1-indexed). The comparison metrics (ARI, NMI, etc.) are
    invariant to relabeling, so this does not affect results.
    """
    df = df.copy()

    if model_type == 'lda' and 'dominant_topic' in df.columns:
        df['topic'] = df['dominant_topic'].astype(int)
    elif model_type == 'iramuteq' and 'iramuteq_class' in df.columns:
        # IRAMUTEQ uses 1-indexed classes, keep original indexing
        df['topic'] = df['iramuteq_class'].astype(int)
    elif 'topic' in df.columns:
        df['topic'] = df['topic'].astype(int)
    else:
        # Try to find any topic-like column
        for col in ['class', 'cluster', 'label']:
            if col in df.columns:
                df['topic'] = df[col].astype(int)
                break

    return df


def align_documents(bertopic_df: pd.DataFrame, lda_df: pd.DataFrame,
                    iramuteq_df: pd.DataFrame,
                    original_csv_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align documents across three model outputs.

    The key challenge is that LDA may filter some documents (e.g., those with < 3 tokens
    after stopword removal), while BERTopic and IRAMUTEQ keep all documents.

    Strategy:
    1. If 'original_index' column exists in doc_assignments (preferred), use it directly
    2. Otherwise, fall back to heuristic matching (less accurate for multi-verse songs)

    Args:
        bertopic_df: BERTopic doc_assignments
        lda_df: LDA doc_assignments (may be shorter due to filtering)
        iramuteq_df: IRAMUTEQ doc_assignments
        original_csv_path: Path to original CSV (optional, for verification)

    Returns:
        Three DataFrames with identical row order for comparison.
    """
    bertopic_df = bertopic_df.copy()
    lda_df = lda_df.copy()
    iramuteq_df = iramuteq_df.copy()

    print(f"  Documents in BERTopic: {len(bertopic_df)}")
    print(f"  Documents in LDA: {len(lda_df)}")
    print(f"  Documents in IRAMUTEQ: {len(iramuteq_df)}")

    # Check if original_index column exists (preferred method)
    has_orig_idx_bert = 'original_index' in bertopic_df.columns
    has_orig_idx_lda = 'original_index' in lda_df.columns

    if has_orig_idx_bert and has_orig_idx_lda:
        print("  Using original_index columns for precise alignment")

        # Use original_index directly
        bertopic_df['_orig_idx'] = bertopic_df['original_index']
        lda_df['_orig_idx'] = lda_df['original_index']
        iramuteq_df['_orig_idx'] = iramuteq_df.get('original_index', range(len(iramuteq_df)))

        # Find common indices (intersection of all three)
        bert_indices = set(bertopic_df['_orig_idx'])
        lda_indices = set(lda_df['_orig_idx'])
        ira_indices = set(iramuteq_df['_orig_idx']) if 'original_index' in iramuteq_df.columns else bert_indices

        common_indices = bert_indices & lda_indices & ira_indices

        print(f"  Common indices (all 3 models): {len(common_indices)}")
        print(f"  Indices only in BERTopic: {len(bert_indices - common_indices)}")
        print(f"  Indices only in LDA: {len(lda_indices - common_indices)}")

        # Filter to common indices
        bertopic_aligned = bertopic_df[bertopic_df['_orig_idx'].isin(common_indices)].copy()
        lda_aligned = lda_df[lda_df['_orig_idx'].isin(common_indices)].copy()
        iramuteq_aligned = iramuteq_df[iramuteq_df['_orig_idx'].isin(common_indices)].copy()

    else:
        # Fallback: heuristic matching (less accurate for multi-verse songs)
        print("  WARNING: original_index not found, using heuristic alignment")
        print("  NOTE: Re-run LDA and BERTopic scripts to get original_index columns for precise alignment")

        # BERTopic and IRAMUTEQ should be perfectly aligned (same order as original CSV)
        bertopic_df['_orig_idx'] = range(len(bertopic_df))
        iramuteq_df['_orig_idx'] = range(len(iramuteq_df))

        # For LDA, use heuristic: match by (artist, title, year) and verse position
        def add_verse_counter(df):
            """Add verse number within each song (0-indexed)."""
            base_key = df['artist'].astype(str) + '|' + df['title'].astype(str) + '|' + df['year'].astype(str)
            df['_verse_num'] = df.groupby(base_key).cumcount()
            df['_base_key'] = base_key
            return df

        bertopic_df = add_verse_counter(bertopic_df)
        lda_df = add_verse_counter(lda_df)

        # Group by song and match
        bert_by_song = bertopic_df.groupby('_base_key')
        lda_by_song = lda_df.groupby('_base_key')

        lda_to_bert_mapping = []
        misaligned_songs = 0

        for song_key, lda_group in lda_by_song:
            if song_key not in bert_by_song.groups:
                continue

            bert_group = bert_by_song.get_group(song_key)
            bert_indices = bert_group['_orig_idx'].tolist()
            lda_verse_count = len(lda_group)
            bert_verse_count = len(bert_indices)

            if lda_verse_count != bert_verse_count:
                misaligned_songs += 1

            # Match verses in order (may be incorrect if middle verse was filtered)
            for i, (lda_idx, _) in enumerate(lda_group.iterrows()):
                if i < len(bert_indices):
                    lda_to_bert_mapping.append((lda_idx, bert_indices[i]))

        if misaligned_songs > 0:
            print(f"  WARNING: {misaligned_songs} songs have different verse counts (alignment may be inaccurate)")

        # Create mapping
        mapping_df = pd.DataFrame(lda_to_bert_mapping, columns=['lda_row', 'bert_orig_idx'])
        lda_df = lda_df.reset_index(drop=True)
        lda_df['_orig_idx'] = mapping_df['bert_orig_idx'].values

        # Find common indices
        common_indices = set(lda_df['_orig_idx'].values)

        bertopic_aligned = bertopic_df[bertopic_df['_orig_idx'].isin(common_indices)].copy()
        lda_aligned = lda_df.copy()
        iramuteq_aligned = iramuteq_df[iramuteq_df['_orig_idx'].isin(common_indices)].copy()

    # Sort all by original index to ensure alignment
    bertopic_aligned = bertopic_aligned.sort_values('_orig_idx').reset_index(drop=True)
    lda_aligned = lda_aligned.sort_values('_orig_idx').reset_index(drop=True)
    iramuteq_aligned = iramuteq_aligned.sort_values('_orig_idx').reset_index(drop=True)

    filtered_count = len(bertopic_df) - len(bertopic_aligned)
    print(f"  Documents filtered (not in all models): {filtered_count}")
    print(f"  Aligned documents: {len(bertopic_aligned)}")

    # Verify alignment
    assert len(bertopic_aligned) == len(lda_aligned) == len(iramuteq_aligned), \
        f"Alignment failed! Lengths: {len(bertopic_aligned)}, {len(lda_aligned)}, {len(iramuteq_aligned)}"

    # Verify original indices match
    if not all(bertopic_aligned['_orig_idx'] == lda_aligned['_orig_idx']):
        mismatches = (bertopic_aligned['_orig_idx'] != lda_aligned['_orig_idx']).sum()
        print(f"  ERROR: {mismatches} index mismatches between BERTopic and LDA!")

    # Cleanup: drop temporary columns (keep original_index for text merging in Q5)
    cols_to_drop = ['_orig_idx', '_verse_num', '_base_key']
    for df in [bertopic_aligned, lda_aligned, iramuteq_aligned]:
        for col in cols_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    return bertopic_aligned, lda_aligned, iramuteq_aligned
