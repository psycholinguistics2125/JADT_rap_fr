#!/usr/bin/env python3
"""
Report generation package for topic model comparison.

Provides markdown, LaTeX, and PDF report generation with
French/English language support.
"""

from .sections import (
    copy_run_figures,
    generate_corpus_description,
    generate_run_description,
    generate_intra_topic_distance_section,
    generate_topic_distance_4configs_section,
    generate_aggregation_curve_section,
    generate_inter_topic_ranking_section,
    generate_word_topic_chi2_section,
)
from .markdown_report import generate_comparison_report
from .latex_report import generate_latex_report
from .pdf_compiler import generate_pdf_report

__all__ = [
    'copy_run_figures',
    'generate_corpus_description',
    'generate_run_description',
    'generate_comparison_report',
    'generate_intra_topic_distance_section',
    'generate_topic_distance_4configs_section',
    'generate_aggregation_curve_section',
    'generate_inter_topic_ranking_section',
    'generate_word_topic_chi2_section',
    'generate_pdf_report',
    'generate_latex_report',
]
