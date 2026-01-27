#!/usr/bin/env python3
"""
LaTeX helper functions, templates, and table/figure generators.
"""

from pathlib import Path


# =============================================================================
# LATEX ESCAPE & FORMATTING
# =============================================================================

def latex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    if text is None:
        return ''
    text = str(text)
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def latex_safe_number(value, fmt='.4f') -> str:
    """Format number for LaTeX, handling N/A and special values."""
    if value is None or value == 'N/A':
        return 'N/A'
    if isinstance(value, str):
        return latex_escape(value)
    if isinstance(value, float):
        if value == float('inf'):
            return r'$\infty$'
        return f'{value:{fmt}}'
    return str(value)


# =============================================================================
# LATEX DOCUMENT TEMPLATE
# =============================================================================

LATEX_PREAMBLE = r'''
\documentclass[11pt,a4paper]{article}

% Encoding and fonts
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}

% Language
\usepackage[{babel_lang}]{babel}

% Math packages
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}

% Tables
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}

% Graphics
\usepackage{graphicx}
\usepackage{float}
\usepackage[export]{adjustbox}

% Colors
\usepackage{xcolor}
\definecolor{linkblue}{RGB}{0,102,204}

% Hyperlinks
\usepackage[colorlinks=true,linkcolor=linkblue,urlcolor=linkblue,citecolor=linkblue]{hyperref}

% Page geometry
\usepackage[margin=2.5cm]{geometry}

% Headers/footers
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\rhead{\thepage}
\lhead{\leftmark}

% Code listings
\usepackage{listings}
\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10}
}

% Better paragraphs
\usepackage{parskip}
\setlength{\parindent}{0pt}

% Custom commands for metrics
\newcommand{\metric}[1]{\textbf{#1}}
\newcommand{\interpretation}[1]{\textit{#1}}

\begin{document}
'''

LATEX_END = r'''
\end{document}
'''


# =============================================================================
# TABLE & FIGURE GENERATION
# =============================================================================

def generate_latex_table(headers: list, rows: list, caption: str = None, label: str = None) -> str:
    """Generate a LaTeX table from headers and rows."""
    n_cols = len(headers)
    col_spec = '|' + 'l|' * n_cols

    tex = r'\begin{table}[H]' + '\n'
    tex += r'\centering' + '\n'
    tex += r'\begin{tabular}{' + col_spec + r'}' + '\n'
    tex += r'\hline' + '\n'

    # Headers
    tex += ' & '.join([r'\textbf{' + latex_escape(h) + '}' for h in headers])
    tex += r' \\' + '\n'
    tex += r'\hline' + '\n'

    # Rows
    for row in rows:
        tex += ' & '.join([str(cell) for cell in row])
        tex += r' \\' + '\n'

    tex += r'\hline' + '\n'
    tex += r'\end{tabular}' + '\n'

    if caption:
        tex += r'\caption{' + latex_escape(caption) + '}' + '\n'
    if label:
        tex += r'\label{' + label + '}' + '\n'

    tex += r'\end{table}' + '\n\n'
    return tex


def generate_latex_figure(image_path: str, caption: str = None, label: str = None, width: str = '0.8') -> str:
    """Generate LaTeX figure inclusion."""
    # Use absolute path for the image
    abs_path = str(Path(image_path).resolve())
    tex = r'\begin{figure}[H]' + '\n'
    tex += r'\centering' + '\n'
    tex += r'\includegraphics[width=' + width + r'\textwidth]{' + abs_path + '}' + '\n'
    if caption:
        tex += r'\caption{' + latex_escape(caption) + '}' + '\n'
    if label:
        tex += r'\label{' + label + '}' + '\n'
    tex += r'\end{figure}' + '\n\n'
    return tex
