#!/usr/bin/env python3
"""
Compare Topic Models: LDA vs BERTopic vs IRAMUTEQ
=================================================
This script generates a comprehensive comparison report between three topic
modeling approaches, answering key research questions about model agreement,
artist separation, temporal dynamics, and vocabulary distinctiveness.

Usage:
    python main_comparate_run.py \
        --lda-folder results/LDA/run_20260123_204347_both \
        --bertopic-folder results/BERTopic/run_20260123_210852_e5 \
        --iramuteq-folder results/IRAMUTEQ/evaluation_20260123_202525 \
        --iramuteq-original results/IRAMUTEQ/original_run_from_pierre_corpus_rap2_alceste_1
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Import comparison utilities
from utils.comparaison_utils import (
    # Data loading
    load_run_data,
    align_documents,
    # Q1: Agreement
    compute_all_pairwise_agreements,
    # Q2: Artist separation
    compute_artist_separation_comparison,
    get_top_residual_pairs,
    # Q3: Temporal
    compute_temporal_comparison,
    compute_decade_js_divergence,
    # Q4: Vocabulary
    compute_vocabulary_overlap,
    compute_vocabulary_distinctiveness,
    compare_topic_vocabularies,
    # Q5: Intra-topic distances (tokenization)
    SpaCyTokenizer,
    NLTKTokenizer,
    SimpleSpaceTokenizer,
    evaluate_topic_distances,
    compute_aggregation_range,
    evaluate_multi_aggregation,
    compute_topic_centroid_distances,
    compute_word_topic_chi2,
    compute_full_vocab_jaccard,
    compute_cross_model_full_vocab_jaccard,
    build_topic_labels,
    # Visualizations
    create_sankey_diagram,
    create_agreement_heatmap,
    create_artist_specificity_heatmap,
    create_temporal_comparison_plot,
    create_vocabulary_comparison_plot,
    create_corpus_year_distribution,
    create_decade_breakdown_plot,
    create_aggregation_curve_plot,
    create_inter_topic_ranking_plot,
    copy_run_figures,
    # Report
    generate_comparison_report,
    generate_pdf_report,
)

warnings.filterwarnings('ignore')

# Default directories
RESULTS_DIR = Path(__file__).parent / "results"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare topic models: LDA vs BERTopic vs IRAMUTEQ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main_comparate_run.py \\
      --lda-folder results/LDA/run_20260123_204347_both \\
      --bertopic-folder results/BERTopic/run_20260123_210852_e5 \\
      --iramuteq-folder results/IRAMUTEQ/evaluation_20260123_202525

  # With IRAMUTEQ vocabulary from original run
  python main_comparate_run.py \\
      --lda-folder results/LDA/run_20260123_204347_both \\
      --bertopic-folder results/BERTopic/run_20260123_210852_e5 \\
      --iramuteq-folder results/IRAMUTEQ/evaluation_20260123_202525 \\
      --iramuteq-original results/IRAMUTEQ/original_run_from_pierre_corpus_rap2_alceste_1
        """
    )

    parser.add_argument(
        '--lda-folder', type=str, required=True,
        help='Path to LDA run folder (e.g., results/LDA/run_YYYYMMDD_HHMMSS_*)'
    )
    parser.add_argument(
        '--bertopic-folder', type=str, required=True,
        help='Path to BERTopic run folder (e.g., results/BERTopic/run_YYYYMMDD_HHMMSS_*)'
    )
    parser.add_argument(
        '--iramuteq-folder', type=str, required=True,
        help='Path to IRAMUTEQ evaluation folder (e.g., results/IRAMUTEQ/evaluation_YYYYMMDD_HHMMSS)'
    )
    parser.add_argument(
        '--iramuteq-original', type=str, default=None,
        help='Path to original IRAMUTEQ run with profiles.csv for vocabulary analysis'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: results/comparisons/comparison_YYYYMMDD_HHMMSS)'
    )
    parser.add_argument(
        '--top-words', type=int, default=30,
        help='Number of top words per topic for vocabulary analysis (default: 30)'
    )
    parser.add_argument(
        '--min-docs-artist', type=int, default=10,
        help='Minimum documents per artist for analysis (default: 10)'
    )
    parser.add_argument(
        '--no-sankey', action='store_true',
        help='Skip Sankey diagram generation (requires plotly)'
    )
    parser.add_argument(
        '--no-figures', action='store_true',
        help='Skip figure generation (faster, report only)'
    )
    parser.add_argument(
        '--corpus', type=str, default='data/20260123_filter_verses_lrfaf_corpus.csv',
        help='Path to corpus CSV with text column (default: data/20260123_filter_verses_lrfaf_corpus.csv)'
    )
    parser.add_argument(
        '--text-column', type=str, default='lyrics_cleaned',
        help='Column name containing document text (default: lyrics_cleaned)'
    )
    parser.add_argument(
        '--lang', type=str, default='fr', choices=['fr', 'en'],
        help='Report language: fr (French, default) or en (English)'
    )
    parser.add_argument(
        '--pdf-engine', type=str, default='latex', choices=['latex', 'markdown'],
        dest='pdf_engine',
        help='PDF generation engine: latex (default, better equations) or markdown (pypandoc)'
    )
    parser.add_argument(
        '--tokenizer', type=str, default='spacy',
        choices=['spacy', 'nltk', 'space'],
        help='Tokenizer for distance metrics: spacy (default, accurate), nltk (faster), or space (fastest, for testing)'
    )
    parser.add_argument(
        '--spacy-model', type=str, default='fr_core_news_lg',
        choices=['fr_core_news_sm', 'fr_core_news_md', 'fr_core_news_lg'],
        dest='spacy_model',
        help='SpaCy model to use when --tokenizer=spacy (default: fr_core_news_lg)'
    )
    parser.add_argument(
        '--aggregation-size', type=int, default=20,
        dest='aggregation_size',
        help='Number of verses to aggregate for aggregated distance modes (default: 20)'
    )

    return parser.parse_args()


def validate_folders(args):
    """Validate that all input folders exist and contain required files."""
    folders = [
        ('LDA', args.lda_folder),
        ('BERTopic', args.bertopic_folder),
        ('IRAMUTEQ', args.iramuteq_folder),
    ]

    for name, folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"{name} folder not found: {folder}")

        doc_assignments = folder_path / 'doc_assignments.csv'
        if not doc_assignments.exists():
            raise FileNotFoundError(f"doc_assignments.csv not found in {name} folder: {folder}")

    if args.iramuteq_original:
        original_path = Path(args.iramuteq_original)
        if not original_path.exists():
            raise FileNotFoundError(f"IRAMUTEQ original folder not found: {args.iramuteq_original}")


def main():
    """Main execution flow."""
    args = parse_arguments()

    print("=" * 70)
    print("TOPIC MODEL COMPARISON")
    print("=" * 70)

    # Validate input folders
    print("\n[1/9] Validating input folders...")
    validate_folders(args)
    print("  All folders validated successfully.")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = COMPARISONS_DIR / f"comparison_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"\n[2/9] Output directory: {output_dir}")

    # Load data from all three runs
    print("\n[3/9] Loading model data...")

    print("  Loading BERTopic data...")
    bertopic_data = load_run_data(args.bertopic_folder, 'bertopic')

    print("  Loading LDA data...")
    lda_data = load_run_data(args.lda_folder, 'lda')

    print("  Loading IRAMUTEQ data...")
    iramuteq_data = load_run_data(
        args.iramuteq_folder, 'iramuteq',
        iramuteq_original_dir=args.iramuteq_original
    )

    # Align documents across models
    print("\n[4/9] Aligning documents across models...")
    bertopic_df, lda_df, iramuteq_df = align_documents(
        bertopic_data['doc_assignments'],
        lda_data['doc_assignments'],
        iramuteq_data['doc_assignments']
    )

    # Update data with aligned dataframes
    bertopic_data['doc_assignments'] = bertopic_df
    lda_data['doc_assignments'] = lda_df
    iramuteq_data['doc_assignments'] = iramuteq_df

    # Save aligned assignments
    aligned_df = bertopic_df[['artist', 'title', 'year']].copy()
    aligned_df['bertopic_topic'] = bertopic_df['topic']
    aligned_df['lda_topic'] = lda_df['topic']
    aligned_df['iramuteq_topic'] = iramuteq_df['topic']
    aligned_df.to_csv(data_dir / 'aligned_assignments.csv', index=False)
    print(f"  Saved aligned assignments to {data_dir / 'aligned_assignments.csv'}")

    # Extract topic arrays
    bertopic_topics = bertopic_df['topic'].values
    lda_topics = lda_df['topic'].values
    iramuteq_topics = iramuteq_df['topic'].values

    # Build human-readable topic labels per model
    topic_labels_per_model = {}
    for model_name, model_data, model_type in [
        ('bertopic', bertopic_data, 'bertopic'),
        ('lda', lda_data, 'lda'),
        ('iramuteq', iramuteq_data, 'iramuteq'),
    ]:
        topics = model_data.get('topics', {})
        if topics:
            topic_labels_per_model[model_name] = build_topic_labels(topics, model_type)

    # Q1: Model Agreement
    print("\n[5/9] Computing Q1: Model Agreement...")
    print("  Computing ARI, NMI, AMI for all pairs...")
    agreement_results = compute_all_pairwise_agreements(
        bertopic_topics, lda_topics, iramuteq_topics
    )

    # Print summary
    for pair_name, pair_data in agreement_results.items():
        agr = pair_data['agreement']
        print(f"    {pair_name}: ARI={agr['ari']:.4f}, NMI={agr['nmi']:.4f}")

    # Save contingency tables
    for pair_name, pair_data in agreement_results.items():
        cont = pair_data['contingency']['contingency_table']
        cont.to_csv(data_dir / f'contingency_{pair_name}.csv')

    # Q2: Artist Separation
    print("\n[6/9] Computing Q2: Artist Separation...")
    artist_results = compute_artist_separation_comparison(
        bertopic_data, lda_data, iramuteq_data,
        min_docs_per_artist=args.min_docs_artist
    )

    # Print summary
    for model in ['bertopic', 'lda', 'iramuteq']:
        v = artist_results.get(f'{model}_cramers_v', 0)
        spec = artist_results.get(f'{model}_pct_specialists', 0)
        print(f"    {model.upper()}: Cramer's V={v:.4f}, Specialists={spec:.1f}%")

    # Save residuals
    for model in ['bertopic', 'lda', 'iramuteq']:
        residuals = artist_results.get(f'{model}_residuals')
        if residuals is not None and not residuals.empty:
            residuals.to_csv(data_dir / f'residuals_{model}.csv')

    # Q3: Temporal Analysis
    print("\n[7/9] Computing Q3: Temporal Analysis...")
    temporal_results = compute_temporal_comparison(
        bertopic_data.get('topic_evolution', pd.DataFrame()),
        lda_data.get('topic_evolution', pd.DataFrame()),
        iramuteq_data.get('topic_evolution', pd.DataFrame())
    )

    # Add decade JS divergence
    for model, data in [('bertopic', bertopic_data), ('lda', lda_data), ('iramuteq', iramuteq_data)]:
        evo = data.get('topic_evolution', pd.DataFrame())
        if not evo.empty:
            temporal_results[f'{model}_decade_js'] = compute_decade_js_divergence(evo)

    # Print summary
    for model in ['bertopic', 'lda', 'iramuteq']:
        var = temporal_results.get(f'{model}_mean_variance', 0)
        print(f"    {model.upper()}: Mean variance={var:.6f}")

    # Q4: Vocabulary Analysis
    print("\n[8/9] Computing Q4: Vocabulary Analysis...")

    vocabulary_results = {}

    # Compute distinctiveness for each model
    for model, data in [('bertopic', bertopic_data), ('lda', lda_data), ('iramuteq', iramuteq_data)]:
        topics = data.get('topics', {})
        if topics:
            vocabulary_results[f'{model}_distinctiveness'] = compute_vocabulary_distinctiveness(topics)
            print(f"    {model.upper()}: Distinctiveness={vocabulary_results[f'{model}_distinctiveness']:.4f}")
        else:
            vocabulary_results[f'{model}_distinctiveness'] = 0
            print(f"    {model.upper()}: No vocabulary data available")

    # Cross-model vocabulary comparison (BERTopic vs LDA) — top-N words
    if bertopic_data.get('topics') and lda_data.get('topics'):
        correspondences = agreement_results['bertopic_vs_lda']['contingency']['correspondences']
        vocabulary_results['bertopic_vs_lda'] = compare_topic_vocabularies(
            bertopic_data['topics'],
            lda_data['topics'],
            correspondences,
            top_n=args.top_words
        )
        print(f"    BERTopic vs LDA vocabulary (top-{args.top_words}): Mean Jaccard={vocabulary_results['bertopic_vs_lda']['mean_jaccard']:.4f}")

    # Q5: Topic Distance Analysis (4 configurations)
    # Configurations:
    # - intra_all_paired: pairwise distances within topics (homogeneity)
    # - inter_all_paired: distances between inside and outside topic (separation)
    # - intra_aggregated: intra-topic with aggregated documents
    # - inter_aggregated: inter-topic with aggregated documents
    print("\n[9/9] Computing Q5: Topic Distances (4 configurations)...")
    topic_distance_results = {}

    # Load corpus with text for distance computation
    corpus_path = Path(args.corpus)
    corpus_tokens = None  # Pre-tokenized corpus (list of token lists)
    corpus_tokens_lemma = None  # Lemmatized variant (SpaCy only)

    if corpus_path.exists():
        print(f"  Loading corpus text from {corpus_path}...")
        corpus_df = pd.read_csv(corpus_path)
        text_col = args.text_column

        if text_col not in corpus_df.columns:
            print(f"    Warning: Text column '{text_col}' not found in corpus. Available: {list(corpus_df.columns)}")
            corpus_df = None
        else:
            # Get all documents for tokenization
            documents_for_tokenization = corpus_df[text_col].fillna('').tolist()
            print(f"  Loaded {len(documents_for_tokenization)} documents with text column '{text_col}'")

            # Tokenize ALL documents ONCE (done once, reused for all 3 models)
            tokenizer_type = getattr(args, 'tokenizer', 'spacy')

            if tokenizer_type == 'space':
                # Simple space tokenizer (fastest, for testing)
                print(f"\n  Tokenizing corpus with simple space tokenizer (fastest) - done ONCE for all models...")
                tokenizer = SimpleSpaceTokenizer(
                    lowercase=True,
                    min_word_length=2,
                    remove_stopwords=True,
                )
                corpus_tokens = tokenizer.batch_tokenize(documents_for_tokenization, verbose=True)
            elif tokenizer_type == 'nltk':
                # NLTK fallback: no lemmatization, simpler tokenization
                print(f"\n  Tokenizing corpus with NLTK (no lemmatization) - done ONCE for all models...")
                tokenizer = NLTKTokenizer(
                    lowercase=True,
                    min_word_length=2,
                    remove_stopwords=True,
                )
                corpus_tokens = tokenizer.batch_tokenize(documents_for_tokenization, verbose=True)
            else:
                # SpaCy: dual tokenization (surface + lemma) in a single pass
                spacy_model = getattr(args, 'spacy_model', 'fr_core_news_lg')
                print(f"\n  Tokenizing corpus with SpaCy ({spacy_model}) - dual mode (surface + lemma)...")
                tokenizer = SpaCyTokenizer(
                    model_name=spacy_model,
                    lowercase=True,
                    min_word_length=2,
                    remove_stopwords=True,
                    lemmatize=False,
                    batch_size=1000,
                    n_process=12
                )
                corpus_tokens, corpus_tokens_lemma = tokenizer.batch_tokenize_dual(
                    documents_for_tokenization, verbose=True)

            # Create index mapping for merging
            corpus_df = corpus_df[[text_col]].copy()
            corpus_df['corpus_index'] = corpus_df.index
    else:
        print(f"  Warning: Corpus file not found: {corpus_path}")
        corpus_df = None

    # Define distance configurations
    aggregation_size = getattr(args, 'aggregation_size', 20)
    distance_configs = [
        ('intra_all_paired', 'intra_all_paired'),
        ('inter_all_paired', 'inter_all_paired'),
        (f'intra_aggregated_{aggregation_size}', 'intra_aggregated'),
        (f'inter_aggregated_{aggregation_size}', 'inter_aggregated'),
    ]

    # Compute distances for each model using the PRE-TOKENIZED corpus
    # Also cache model tokens for multi-aggregation evaluation
    model_token_cache = {}

    for model_name, model_data, model_df in [
        ('bertopic', bertopic_data, bertopic_df),
        ('lda', lda_data, lda_df),
        ('iramuteq', iramuteq_data, iramuteq_df)
    ]:
        topic_distance_results[model_name] = {}

        if corpus_tokens is None:
            print(f"    {model_name.upper()}: No tokenized corpus available, skipping distance computation")
            for config_name, _ in distance_configs:
                topic_distance_results[model_name][config_name] = {
                    'js': {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}},
                    'labbe': {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}}
                }
            continue

        # Map model documents to tokenized corpus using original_index
        if 'original_index' in model_df.columns:
            # Get the original indices and topics from the model
            original_indices = model_df['original_index'].tolist()
            topics = model_df['topic'].tolist()

            # Extract token lists for these documents (using original_index as corpus index)
            model_tokens = []
            valid_topics = []
            for idx, topic in zip(original_indices, topics):
                if 0 <= idx < len(corpus_tokens):
                    model_tokens.append(corpus_tokens[idx])
                    valid_topics.append(topic)
                else:
                    # Document index out of range, add empty tokens
                    model_tokens.append([])
                    valid_topics.append(topic)

            print(f"    {model_name.upper()}: Using {len(model_tokens)} pre-tokenized documents")

            # Cache for multi-aggregation
            model_token_cache[model_name] = (model_tokens, valid_topics)
        else:
            print(f"    {model_name.upper()}: No original_index column, skipping distance computation")
            for config_name, _ in distance_configs:
                topic_distance_results[model_name][config_name] = {
                    'js': {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}},
                    'labbe': {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}}
                }
            continue

        # Compute all 4 distance configurations
        print(f"    {model_name.upper()}: Computing 4 distance configurations...")
        for config_name, mode in distance_configs:
            print(f"      Computing {config_name}...")
            distance_results = evaluate_topic_distances(
                model_tokens,
                valid_topics,
                mode=mode,
                distance_type='both',
                aggregation_size=aggregation_size,
                sample_size=5000,
                random_seed=42,
                verbose=False  # Suppress per-topic output
            )
            topic_distance_results[model_name][config_name] = distance_results

            js_mean = distance_results.get('js', {}).get('mean', 0)
            labbe_mean = distance_results.get('labbe', {}).get('mean', 0)
            print(f"        JS={js_mean:.4f}, Labbé={labbe_mean:.4f}")

    # Q5 Feature: Multi-aggregation stabilization curve
    multi_agg_results = {}
    agg_metadata = None

    if model_token_cache:
        print("\n[*] Computing aggregation stabilization curve...")

        # Find min_topic_size across ALL models for a shared range
        global_min_topic_size = float('inf')
        for model_name, (tokens, topics) in model_token_cache.items():
            _, meta = compute_aggregation_range(tokens, topics)
            model_min = meta.get('min_topic_size', float('inf'))
            if model_min < global_min_topic_size:
                global_min_topic_size = model_min

        # Compute shared range using first model's tokens for mean_doc_length
        # (all use same corpus tokens, so mean_doc_length is the same)
        first_tokens, first_topics = next(iter(model_token_cache.values()))
        agg_sizes, agg_metadata = compute_aggregation_range(
            first_tokens, first_topics,
            n_points=5, min_words_per_unit=500, min_units_per_topic=5,
            min_step=10,
            override_min_topic_size=int(global_min_topic_size)
        )
        agg_metadata['min_words_per_unit'] = 500
        agg_metadata['min_units_per_topic'] = 5
        agg_metadata['n_points'] = len(agg_sizes)

        print(f"    Shared aggregation range: {agg_sizes[0]}-{agg_sizes[-1]} ({len(agg_sizes)} points)")
        print(f"    Sizes: {agg_sizes}")
        print(f"    Min topic size (global): {global_min_topic_size}")

        for model_name, (tokens, topics) in model_token_cache.items():
            print(f"    {model_name.upper()}: Computing multi-aggregation...")
            multi_agg_results[model_name] = evaluate_multi_aggregation(
                tokens, topics,
                aggregation_sizes=agg_sizes,
                distance_type='labbe',
                sample_size=1000,
                random_seed=42,
                verbose=True
            )

    # Create backward-compatible intra_topic_results for legacy report support
    # Uses intra_all_paired as the default (same as before)
    intra_topic_results = {}
    for model in ['bertopic', 'lda', 'iramuteq']:
        default_config = topic_distance_results.get(model, {}).get('intra_all_paired', {})
        intra_topic_results[f'{model}_js'] = default_config.get('js', {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}})
        intra_topic_results[f'{model}_labbe'] = default_config.get('labbe', {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}})

    # Q5 Feature: Topic centroid distances (one merged doc per topic vs rest)
    centroid_results = {}
    if model_token_cache:
        print("\n[*] Computing topic centroid distances (one-vs-rest)...")
        for model_name, (tokens, topics) in model_token_cache.items():
            centroid_results[model_name] = compute_topic_centroid_distances(
                tokens, topics, distance_type='both')
            labbe_mean = centroid_results[model_name].get('labbe', {}).get('mean', 0)
            print(f"    {model_name.upper()}: mean centroid Labbé = {labbe_mean:.4f}")

    # Q5 Feature: Full vocabulary Jaccard (within-model pairwise)
    full_vocab_jaccard = {}
    if model_token_cache:
        print("\n[*] Computing full vocabulary Jaccard (min_freq=5)...")
        for model_name, (tokens, topics) in model_token_cache.items():
            result = compute_full_vocab_jaccard(tokens, topics, min_freq=5)
            full_vocab_jaccard[model_name] = result
            print(f"    {model_name.upper()}: mean Jaccard={result['mean_jaccard']:.4f}, "
                  f"vocab sizes: {', '.join(f'T{t}={s}' for t, s in list(result['vocab_sizes'].items())[:5])}...")

    # Q5 Feature: Cross-model full vocabulary Jaccard (BERTopic vs LDA)
    cross_model_jaccard = {}
    if model_token_cache and 'bertopic' in model_token_cache and 'lda' in model_token_cache:
        print("\n[*] Computing cross-model full vocabulary Jaccard...")
        correspondences = agreement_results['bertopic_vs_lda']['contingency']['correspondences']
        bert_tokens, bert_topics = model_token_cache['bertopic']
        lda_tokens, lda_topics = model_token_cache['lda']
        cross_model_jaccard['bertopic_vs_lda'] = compute_cross_model_full_vocab_jaccard(
            bert_tokens, bert_topics, lda_topics, correspondences,
            model_a_name='bertopic', model_b_name='lda',
            min_freq_thresholds=[1, 5, 20]
        )
        for thresh, data in cross_model_jaccard['bertopic_vs_lda']['per_threshold'].items():
            print(f"    BERTopic vs LDA (min_freq={thresh}): mean Jaccard={data['mean_jaccard']:.4f}")

    # Q5 Feature: χ²/n word × topic independence test
    chi2_results = {'non_lemmatized': {}, 'lemmatized': {}}

    if model_token_cache:
        print("\n[*] Computing χ²/n word × topic independence test...")

        # Non-lemmatized (surface forms) — uses existing cached tokens
        for model_name, (tokens, topics) in model_token_cache.items():
            result = compute_word_topic_chi2(tokens, topics, min_word_freq=5)
            chi2_results['non_lemmatized'][model_name] = result
            print(f"    {model_name.upper()} (surface): χ²/n={result['chi2_over_n']:.4f}, "
                  f"vocab={result['vocab_size']:,}, N={result['n']:,}")

        # Lemmatized — build lemmatized token lists from corpus_tokens_lemma
        if corpus_tokens_lemma is not None:
            print("    Computing lemmatized variant...")
            for model_name, model_data, model_df in [
                ('bertopic', bertopic_data, bertopic_df),
                ('lda', lda_data, lda_df),
                ('iramuteq', iramuteq_data, iramuteq_df)
            ]:
                if model_name not in model_token_cache:
                    continue
                _, valid_topics = model_token_cache[model_name]

                # Build lemmatized token list using same original_index mapping
                if 'original_index' in model_df.columns:
                    original_indices = model_df['original_index'].tolist()
                    topics = model_df['topic'].tolist()
                    lemma_tokens = []
                    lemma_topics = []
                    for idx, topic in zip(original_indices, topics):
                        if 0 <= idx < len(corpus_tokens_lemma):
                            lemma_tokens.append(corpus_tokens_lemma[idx])
                            lemma_topics.append(topic)
                        else:
                            lemma_tokens.append([])
                            lemma_topics.append(topic)

                    result = compute_word_topic_chi2(lemma_tokens, lemma_topics, min_word_freq=5)
                    chi2_results['lemmatized'][model_name] = result
                    print(f"    {model_name.upper()} (lemma):   χ²/n={result['chi2_over_n']:.4f}, "
                          f"vocab={result['vocab_size']:,}, N={result['n']:,}")
        else:
            print("    Lemmatized variant not available (requires SpaCy tokenizer)")

    # Generate visualizations
    if not args.no_figures:
        print("\n[*] Generating visualizations...")

        # Corpus visualizations
        print("    Creating corpus visualizations...")
        create_corpus_year_distribution(aligned_df, str(figures_dir / 'corpus_year_distribution.png'))
        create_decade_breakdown_plot(aligned_df, str(figures_dir / 'corpus_decade_breakdown.png'))
        print("    Saved: corpus_year_distribution.png, corpus_decade_breakdown.png")

        # Copy figures from run directories
        print("    Copying figures from run directories...")
        for model_name, model_data in [('bertopic', bertopic_data), ('lda', lda_data), ('iramuteq', iramuteq_data)]:
            run_dir = model_data.get('run_dir', '')
            if run_dir:
                copied = copy_run_figures(run_dir, str(figures_dir), model_name)
                if copied:
                    print(f"      {model_name.upper()}: copied {len(copied)} figures")

        # Contingency heatmaps
        for pair_name, pair_data in agreement_results.items():
            cont = pair_data['contingency']['contingency_table']
            title = f"Topic Contingency: {pair_name.replace('_', ' ').title()}"
            output_path = figures_dir / f'contingency_{pair_name}.png'
            create_agreement_heatmap(cont, title, str(output_path))
            print(f"    Saved: {output_path.name}")

        # Sankey diagrams
        if not args.no_sankey:
            print("    Generating Sankey diagrams...")
            create_sankey_diagram(
                bertopic_topics, lda_topics,
                "BERTopic", "LDA",
                str(figures_dir / 'sankey_bertopic_lda.png')
            )
            create_sankey_diagram(
                bertopic_topics, iramuteq_topics,
                "BERTopic", "IRAMUTEQ",
                str(figures_dir / 'sankey_bertopic_iramuteq.png')
            )
            create_sankey_diagram(
                lda_topics, iramuteq_topics,
                "LDA", "IRAMUTEQ",
                str(figures_dir / 'sankey_lda_iramuteq.png')
            )
            print("    Saved: sankey_*.html/png")

        # Artist specificity heatmaps
        for model in ['bertopic', 'lda', 'iramuteq']:
            residuals = artist_results.get(f'{model}_residuals')
            if residuals is not None and not residuals.empty:
                output_path = figures_dir / f'artist_specificity_{model}.png'
                create_artist_specificity_heatmap(residuals, model.upper(), str(output_path))
                print(f"    Saved: {output_path.name}")

        # Temporal comparison
        create_temporal_comparison_plot(
            bertopic_data.get('topic_evolution', pd.DataFrame()),
            lda_data.get('topic_evolution', pd.DataFrame()),
            iramuteq_data.get('topic_evolution', pd.DataFrame()),
            str(figures_dir / 'temporal_comparison.png')
        )
        print("    Saved: temporal_comparison.png")

        # Vocabulary comparison
        if 'bertopic_vs_lda' in vocabulary_results:
            create_vocabulary_comparison_plot(
                vocabulary_results['bertopic_vs_lda'],
                str(figures_dir / 'vocabulary_comparison.png')
            )
            print("    Saved: vocabulary_comparison.png")

        # Q5 Feature: Aggregation stabilization curve
        if multi_agg_results:
            create_aggregation_curve_plot(
                multi_agg_results,
                str(figures_dir / 'aggregation_curve.png')
            )
            print("    Saved: aggregation_curve.png")

        # Q5 Feature: Inter-topic separation ranking bar charts
        for model in ['bertopic', 'lda', 'iramuteq']:
            centroid_data = centroid_results.get(model, {})
            if centroid_data and centroid_data.get('labbe', {}).get('per_topic'):
                create_inter_topic_ranking_plot(
                    centroid_data,
                    model.upper(),
                    str(figures_dir / f'inter_topic_ranking_{model}.png'),
                    topic_labels=topic_labels_per_model.get(model)
                )
                print(f"    Saved: inter_topic_ranking_{model}.png")

    # Compile all results
    all_results = {
        'bertopic': bertopic_data,
        'lda': lda_data,
        'iramuteq': iramuteq_data,
        'agreement': agreement_results,
        'artist_separation': artist_results,
        'temporal': temporal_results,
        'vocabulary': vocabulary_results,
        'intra_topic_distances': intra_topic_results,  # Legacy format for backward compatibility
        'topic_distance_results': topic_distance_results,  # New: all 4 configurations
        'aggregation_size': aggregation_size,  # For report generation
        'multi_agg_results': multi_agg_results,  # Aggregation stabilization curve data
        'chi2_results': chi2_results,  # χ²/n word × topic independence test
        'topic_labels_per_model': topic_labels_per_model,  # Human-readable topic labels
        'centroid_results': centroid_results,  # Topic centroid distances
        'full_vocab_jaccard': full_vocab_jaccard,  # Full vocabulary Jaccard per model
        'cross_model_jaccard': cross_model_jaccard,  # Cross-model full-vocab Jaccard
        'agg_metadata': agg_metadata,  # Aggregation range metadata
    }

    # Generate markdown report
    print(f"\n[*] Generating enhanced markdown report (lang={args.lang})...")
    report_content = generate_comparison_report(all_results, str(output_dir), str(figures_dir), lang=args.lang)
    report_path = output_dir / 'comparison_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"    Saved: {report_path}")

    # Generate PDF report using LaTeX engine (better equation rendering)
    # Set pdf_engine='markdown' to use pypandoc instead, or 'weasyprint' for legacy HTML-based PDF
    pdf_engine = getattr(args, 'pdf_engine', 'latex')
    print(f"\n[*] Generating PDF report (engine={pdf_engine})...")
    pdf_path = output_dir / 'comparison_report.pdf'

    success = generate_pdf_report(
        markdown_content=report_content,
        output_path=pdf_path,
        lang=args.lang,
        pdf_engine=pdf_engine,
        results=all_results,
        output_dir=str(output_dir),
        figures_dir=str(figures_dir)
    )

    if success:
        print(f"    Saved: {pdf_path}")
        # Also note the .tex file for LaTeX engine
        if pdf_engine == 'latex':
            tex_path = pdf_path.with_suffix('.tex')
            if tex_path.exists():
                print(f"    LaTeX source: {tex_path}")
    else:
        print("    Warning: PDF generation failed")
        print("    For LaTeX engine: ensure xelatex or pdflatex is installed")
        print("    For markdown engine: pip install pypandoc")
        pdf_path = None

    # Save metrics JSON
    print("\n[*] Saving metrics JSON...")
    metrics_to_save = {
        'timestamp': timestamp,
        'inputs': {
            'lda_folder': args.lda_folder,
            'bertopic_folder': args.bertopic_folder,
            'iramuteq_folder': args.iramuteq_folder,
            'iramuteq_original': args.iramuteq_original,
        },
        'n_aligned_documents': len(aligned_df),
        'agreement': {
            pair_name: {
                'ari': pair_data['agreement']['ari'],
                'nmi': pair_data['agreement']['nmi'],
                'ami': pair_data['agreement']['ami'],
                'n_one_to_one': pair_data['contingency']['n_one_to_one'],
            }
            for pair_name, pair_data in agreement_results.items()
        },
        'artist_separation': {
            model: {
                'cramers_v': artist_results.get(f'{model}_cramers_v', 0),
                'pct_specialists': artist_results.get(f'{model}_pct_specialists', 0),
                'pct_moderate': artist_results.get(f'{model}_pct_moderate', 0),
                'pct_generalists': artist_results.get(f'{model}_pct_generalists', 0),
                'mean_entropy': artist_results.get(f'{model}_mean_entropy', 0),
            }
            for model in ['bertopic', 'lda', 'iramuteq']
        },
        'temporal': {
            model: {
                'mean_variance': temporal_results.get(f'{model}_mean_variance', 0),
                'most_variable_topic': temporal_results.get(f'{model}_most_variable_topic', ''),
                'max_variance': temporal_results.get(f'{model}_max_variance', 0),
            }
            for model in ['bertopic', 'lda', 'iramuteq']
        },
        'vocabulary': {
            'distinctiveness': {
                model: vocabulary_results.get(f'{model}_distinctiveness', 0)
                for model in ['bertopic', 'lda', 'iramuteq']
            },
            'bertopic_vs_lda': {
                'mean_jaccard': vocabulary_results.get('bertopic_vs_lda', {}).get('mean_jaccard', 0),
                'mean_overlap_coef': vocabulary_results.get('bertopic_vs_lda', {}).get('mean_overlap_coef', 0),
            }
        },
        'intra_topic_distances': {
            model: {
                'js_mean': intra_topic_results.get(f'{model}_js', {}).get('mean', 0),
                'js_std': intra_topic_results.get(f'{model}_js', {}).get('std', 0),
                'labbe_mean': intra_topic_results.get(f'{model}_labbe', {}).get('mean', 0),
                'labbe_std': intra_topic_results.get(f'{model}_labbe', {}).get('std', 0),
                'n_topics': intra_topic_results.get(f'{model}_js', {}).get('n_topics', 0),
            }
            for model in ['bertopic', 'lda', 'iramuteq']
        },
        # All 4 distance configurations
        'topic_distances': {
            model: {
                config_name: {
                    'js_mean': topic_distance_results.get(model, {}).get(config_name, {}).get('js', {}).get('mean', 0),
                    'js_std': topic_distance_results.get(model, {}).get(config_name, {}).get('js', {}).get('std', 0),
                    'labbe_mean': topic_distance_results.get(model, {}).get(config_name, {}).get('labbe', {}).get('mean', 0),
                    'labbe_std': topic_distance_results.get(model, {}).get(config_name, {}).get('labbe', {}).get('std', 0),
                }
                for config_name in ['intra_all_paired', 'inter_all_paired',
                                    f'intra_aggregated_{aggregation_size}', f'inter_aggregated_{aggregation_size}']
            }
            for model in ['bertopic', 'lda', 'iramuteq']
        },
        'aggregation_size': aggregation_size,
        # Per-topic inter distances (for ranking chart reproducibility)
        'inter_topic_per_topic': {
            model: {
                str(topic_id): stats.get('mean_distance', 0)
                for topic_id, stats in topic_distance_results.get(model, {})
                    .get('inter_all_paired', {})
                    .get('labbe', {})
                    .get('per_topic', {}).items()
            }
            for model in ['bertopic', 'lda', 'iramuteq']
        },
        # Multi-aggregation curve data
        'multi_aggregation': {
            model: {
                'aggregation_sizes': sorted(data.keys()),
                'intra_labbe_means': [
                    data[agg].get('intra_aggregated', {}).get('labbe', {}).get('mean', 0)
                    for agg in sorted(data.keys())
                ],
                'inter_labbe_means': [
                    data[agg].get('inter_aggregated', {}).get('labbe', {}).get('mean', 0)
                    for agg in sorted(data.keys())
                ],
            }
            for model, data in multi_agg_results.items()
        } if multi_agg_results else {},
        'agg_metadata': {
            k: v for k, v in (agg_metadata or {}).items()
            if k != 'topic_sizes'  # Exclude verbose per-topic sizes
        },
        # χ²/n word × topic independence test
        'chi2_results': {
            variant: {
                model: {
                    'chi2': r.get('chi2', 0),
                    'n': r.get('n', 0),
                    'chi2_over_n': r.get('chi2_over_n', 0),
                    'vocab_size': r.get('vocab_size', 0),
                    'n_topics': r.get('n_topics', 0),
                }
                for model, r in variant_data.items()
            }
            for variant, variant_data in chi2_results.items()
            if variant_data
        },
    }

    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, default=str)
    print(f"    Saved: {metrics_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - Report (Markdown): {report_path}")
    if pdf_path:
        print(f"  - Report (PDF): {pdf_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Figures: {figures_dir}/")
    print(f"  - Data: {data_dir}/")

    # Summary stats
    print(f"\nSummary:")
    print(f"  - Documents compared: {len(aligned_df):,}")

    best_agreement = max(agreement_results.items(), key=lambda x: x[1]['agreement']['nmi'])
    print(f"  - Best agreement: {best_agreement[0]} (NMI={best_agreement[1]['agreement']['nmi']:.4f})")

    best_artist = max(['bertopic', 'lda', 'iramuteq'],
                      key=lambda x: artist_results.get(f'{x}_cramers_v', 0))
    print(f"  - Best artist separation: {best_artist.upper()} (V={artist_results.get(f'{best_artist}_cramers_v', 0):.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
