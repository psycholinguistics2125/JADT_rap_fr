#!/usr/bin/env python3
"""
Comparison Utilities for Topic Model Analysis
==============================================

This package provides utilities for comparing LDA, BERTopic, and IRAMUTEQ topic models.

Modules:
--------
- constants: Scientific references and metric definitions
- data_loading: Functions for loading and aligning model outputs
- agreement: Model agreement metrics (ARI, NMI, AMI)
- artist_separation: Artist separation analysis (Cramer's V, residuals)
- temporal: Temporal evolution analysis
- vocabulary: Vocabulary overlap and distinctiveness
- topic_distances: Intra-topic distance metrics (Labbé, Jensen-Shannon)
- visualization: Plotting and visualization functions
- report: Markdown report generation

Usage:
------
    from utils.comparaison_utils import (
        load_run_data,
        align_documents,
        compute_all_pairwise_agreements,
        generate_comparison_report,
        # Topic distances
        LabbeDistance,
        JensenShannonDistance,
        evaluate_topic_coherence,
    )
"""

# Constants
from .constants import METRIC_REFERENCES

# Data loading
from .data_loading import (
    load_run_data,
    load_iramuteq_vocabulary,
    normalize_topic_column,
    align_documents,
)

# Q1: Model Agreement
from .agreement import (
    compute_pairwise_agreement,
    compute_contingency_analysis,
    compute_all_pairwise_agreements,
)

# Q2: Artist Separation
from .artist_separation import (
    compute_cramers_v,
    compute_standardized_residuals,
    compute_artist_separation_comparison,
    get_top_residual_pairs,
)

# Q3: Temporal Analysis
from .temporal import (
    compute_temporal_comparison,
    compute_decade_js_divergence,
)

# Q4: Vocabulary Analysis
from .vocabulary import (
    extract_topic_words,
    compute_vocabulary_overlap,
    compute_vocabulary_distinctiveness,
    compare_topic_vocabularies,
)

# Q5: Topic Distance Metrics
from .topic_distances import (
    BaseDistance,
    LabbeDistance,
    JensenShannonDistance,
    WMDDistance,
    evaluate_topic_coherence,
    # Batch tokenization (SpaCy or NLTK)
    SpaCyTokenizer,
    NLTKTokenizer,
    batch_tokenize_documents,
    evaluate_topic_coherence_from_tokens,
)

# Visualization
from .visualization import (
    create_sankey_diagram,
    create_agreement_heatmap,
    create_artist_specificity_heatmap,
    create_temporal_comparison_plot,
    create_vocabulary_comparison_plot,
    create_corpus_year_distribution,
    create_decade_breakdown_plot,
)

# Report Generation
from .report import (
    copy_run_figures,
    generate_corpus_description,
    generate_run_description,
    generate_comparison_report,
    generate_intra_topic_distance_section,
    generate_pdf_report,
    generate_latex_report,
)

__all__ = [
    # Constants
    'METRIC_REFERENCES',
    # Data loading
    'load_run_data',
    'load_iramuteq_vocabulary',
    'normalize_topic_column',
    'align_documents',
    # Agreement
    'compute_pairwise_agreement',
    'compute_contingency_analysis',
    'compute_all_pairwise_agreements',
    # Artist separation
    'compute_cramers_v',
    'compute_standardized_residuals',
    'compute_artist_separation_comparison',
    'get_top_residual_pairs',
    # Temporal
    'compute_temporal_comparison',
    'compute_decade_js_divergence',
    # Vocabulary
    'extract_topic_words',
    'compute_vocabulary_overlap',
    'compute_vocabulary_distinctiveness',
    'compare_topic_vocabularies',
    # Topic distances
    'BaseDistance',
    'LabbeDistance',
    'JensenShannonDistance',
    'WMDDistance',
    'evaluate_topic_coherence',
    'SpaCyTokenizer',
    'NLTKTokenizer',
    'batch_tokenize_documents',
    'evaluate_topic_coherence_from_tokens',
    # Visualization
    'create_sankey_diagram',
    'create_agreement_heatmap',
    'create_artist_specificity_heatmap',
    'create_temporal_comparison_plot',
    'create_vocabulary_comparison_plot',
    'create_corpus_year_distribution',
    'create_decade_breakdown_plot',
    # Report
    'copy_run_figures',
    'generate_corpus_description',
    'generate_run_description',
    'generate_comparison_report',
    'generate_intra_topic_distance_section',
    'generate_pdf_report',
    'generate_latex_report',
]
