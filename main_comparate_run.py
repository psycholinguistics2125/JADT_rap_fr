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
    evaluate_topic_coherence_from_tokens,
    # Visualizations
    create_sankey_diagram,
    create_agreement_heatmap,
    create_artist_specificity_heatmap,
    create_temporal_comparison_plot,
    create_vocabulary_comparison_plot,
    create_corpus_year_distribution,
    create_decade_breakdown_plot,
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
        '--include-sankey', action='store_true',
        help='Generate Sankey diagrams (requires plotly)'
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
        '--spacy-model', type=str, default='fr_core_news_lg',
        choices=['fr_core_news_sm', 'fr_core_news_md', 'fr_core_news_lg', 'none'],
        dest='spacy_model',
        help='Tokenizer for distance metrics: spaCy model sm/md/lg (default: lg), or "none" for NLTK fallback (no lemmatization)'
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

    # Cross-model vocabulary comparison (BERTopic vs LDA)
    if bertopic_data.get('topics') and lda_data.get('topics'):
        correspondences = agreement_results['bertopic_vs_lda']['contingency']['correspondences']
        vocabulary_results['bertopic_vs_lda'] = compare_topic_vocabularies(
            bertopic_data['topics'],
            lda_data['topics'],
            correspondences,
            top_n=args.top_words
        )
        print(f"    BERTopic vs LDA vocabulary: Mean Jaccard={vocabulary_results['bertopic_vs_lda']['mean_jaccard']:.4f}")

    # Q5: Intra-topic Distance Analysis (SpaCy tokenization, done ONCE for all models)
    print("\n[9/9] Computing Q5: Intra-topic Distances...")
    intra_topic_results = {}

    # Load corpus with text for distance computation
    corpus_path = Path(args.corpus)
    corpus_tokens = None  # Pre-tokenized corpus (list of token lists)

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
            spacy_model = getattr(args, 'spacy_model', 'fr_core_news_lg')

            if spacy_model == 'none':
                # NLTK fallback: no lemmatization, simpler tokenization
                print(f"\n  Tokenizing corpus with NLTK (no lemmatization) - done ONCE for all models...")
                tokenizer = NLTKTokenizer(
                    lowercase=True,
                    min_word_length=2,
                    remove_stopwords=True,
                )
            else:
                # SpaCy: POS-aware tokenization, no lemmatization
                print(f"\n  Tokenizing corpus with SpaCy ({spacy_model}) - done ONCE for all models...")
                tokenizer = SpaCyTokenizer(
                    model_name=spacy_model,
                    lowercase=True,
                    min_word_length=2,
                    remove_stopwords=True,
                    lemmatize=False,
                    batch_size=1000,
                    n_process=-1  # Use all CPUs
                )

            corpus_tokens = tokenizer.batch_tokenize(documents_for_tokenization, verbose=True)

            # Create index mapping for merging
            corpus_df = corpus_df[[text_col]].copy()
            corpus_df['corpus_index'] = corpus_df.index
    else:
        print(f"  Warning: Corpus file not found: {corpus_path}")
        corpus_df = None

    # Compute distances for each model using the PRE-TOKENIZED corpus
    for model_name, model_data, model_df in [
        ('bertopic', bertopic_data, bertopic_df),
        ('lda', lda_data, lda_df),
        ('iramuteq', iramuteq_data, iramuteq_df)
    ]:
        if corpus_tokens is None:
            print(f"    {model_name.upper()}: No tokenized corpus available, skipping distance computation")
            intra_topic_results[f'{model_name}_js'] = {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}}
            intra_topic_results[f'{model_name}_labbe'] = {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}}
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
        else:
            print(f"    {model_name.upper()}: No original_index column, skipping distance computation")
            intra_topic_results[f'{model_name}_js'] = {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}}
            intra_topic_results[f'{model_name}_labbe'] = {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}}
            continue

        # Compute both JS and Labbé distances using pre-tokenized documents
        print(f"    {model_name.upper()}: Computing distances from pre-tokenized documents...")
        distance_results = evaluate_topic_coherence_from_tokens(
            model_tokens,
            valid_topics,
            distance_type='both',
            sample_size=50,
            random_seed=42,
            verbose=False
        )

        intra_topic_results[f'{model_name}_js'] = distance_results.get('js', {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}})
        intra_topic_results[f'{model_name}_labbe'] = distance_results.get('labbe', {'mean': 0, 'std': 0, 'n_topics': 0, 'per_topic': {}})
        print(f"      JS mean: {intra_topic_results[f'{model_name}_js']['mean']:.4f}, Labbé mean: {intra_topic_results[f'{model_name}_labbe']['mean']:.4f}")

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
        if args.include_sankey:
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

    # Compile all results
    all_results = {
        'bertopic': bertopic_data,
        'lda': lda_data,
        'iramuteq': iramuteq_data,
        'agreement': agreement_results,
        'artist_separation': artist_results,
        'temporal': temporal_results,
        'vocabulary': vocabulary_results,
        'intra_topic_distances': intra_topic_results,
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
        }
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
