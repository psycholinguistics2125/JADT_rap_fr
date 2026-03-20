#!/usr/bin/env python3
"""
LaTeX helper functions, templates, and table/figure generators.
"""

import re
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


def markdown_to_latex(text: str) -> str:
    """Convert markdown-formatted text to LaTeX.

    Handles: **bold**, *italic*, ### headings, - bullet lists,
    1. numbered lists, and inline `code`.
    """
    if text is None:
        return ''
    text = str(text)

    # Split into paragraphs (double newline)
    paragraphs = re.split(r'\n\n+', text)
    result_parts = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        lines = para.split('\n')
        non_empty = [l for l in lines if l.strip()]

        # Check if this paragraph is a list (all lines start with - or N.)
        is_bullet = all(re.match(r'\s*-\s', l) for l in non_empty)
        # Numbered list: first line starts with N., rest can be N. or indented bullets
        is_numbered = (bool(non_empty) and
                       bool(re.match(r'\s*\d+\.\s', non_empty[0])) and
                       all(re.match(r'\s*(\d+\.|-)\s', l) for l in non_empty))

        if is_bullet:
            items = []
            for line in lines:
                if not line.strip():
                    continue
                item_text = re.sub(r'^\s*-\s+', '', line)
                items.append(r'\item ' + _convert_inline_markdown(item_text))
            result_parts.append(
                r'\begin{itemize}' + '\n'
                + '\n'.join(items) + '\n'
                + r'\end{itemize}'
            )
        elif is_numbered:
            # Handle numbered lists, possibly with nested bullet sub-items
            result = []
            in_nested = False
            for line in lines:
                if not line.strip():
                    continue
                if re.match(r'\s*\d+\.\s', line):
                    # Close nested itemize if open
                    if in_nested:
                        result.append(r'  \end{itemize}')
                        in_nested = False
                    item_text = re.sub(r'^\s*\d+\.\s+', '', line)
                    result.append(r'\item ' + _convert_inline_markdown(item_text))
                elif re.match(r'\s+-\s', line):
                    # Nested bullet within numbered list
                    sub_text = re.sub(r'^\s+-\s+', '', line)
                    if not in_nested:
                        result.append(r'  \begin{itemize}')
                        in_nested = True
                    result.append(r'  \item ' + _convert_inline_markdown(sub_text))
                else:
                    if in_nested:
                        result.append(r'  \end{itemize}')
                        in_nested = False
                    result.append(_convert_inline_markdown(line))

            if in_nested:
                result.append(r'  \end{itemize}')

            result_parts.append(
                r'\begin{enumerate}' + '\n'
                + '\n'.join(result) + '\n'
                + r'\end{enumerate}'
            )
        elif lines[0].startswith('#'):
            # Heading line
            match = re.match(r'^(#{1,5})\s+(.*)', lines[0])
            if match:
                level = len(match.group(1))
                title = _convert_inline_markdown(match.group(2))
                cmds = {1: 'section', 2: 'subsection', 3: 'subsubsection',
                        4: 'paragraph', 5: 'subparagraph'}
                cmd = cmds.get(level, 'paragraph')
                heading = '\\' + cmd + '{' + title + '}'
                # If there's more text after the heading, add it
                rest = '\n'.join(lines[1:]).strip()
                if rest:
                    heading += '\n\n' + _convert_inline_markdown(rest)
                result_parts.append(heading)
            else:
                result_parts.append(_convert_inline_markdown(para))
        else:
            # Regular paragraph — check for mixed content with embedded bullets
            has_bullets = any(re.match(r'\s+-\s', l) for l in lines)
            if has_bullets:
                # Split into text before bullets, the bullets, and text after
                text_parts = []
                bullet_items = []
                in_bullets = False
                for line in lines:
                    if re.match(r'\s+-\s', line):
                        in_bullets = True
                        sub_text = re.sub(r'^\s+-\s+', '', line)
                        bullet_items.append(r'\item ' + _convert_inline_markdown(sub_text))
                    else:
                        if in_bullets and bullet_items:
                            text_parts.append(
                                r'\begin{itemize}' + '\n'
                                + '\n'.join(bullet_items) + '\n'
                                + r'\end{itemize}')
                            bullet_items = []
                            in_bullets = False
                        text_parts.append(_convert_inline_markdown(line))
                if bullet_items:
                    text_parts.append(
                        r'\begin{itemize}' + '\n'
                        + '\n'.join(bullet_items) + '\n'
                        + r'\end{itemize}')
                result_parts.append('\n'.join(text_parts))
            else:
                # Regular paragraph — convert inline formatting
                result_parts.append(_convert_inline_markdown(para))

    return '\n\n'.join(result_parts)


def _convert_inline_markdown(text: str) -> str:
    """Convert inline markdown (**bold**, *italic*, `code`) to LaTeX.

    Applies latex_escape first, then converts markdown patterns.
    """
    # First escape LaTeX special chars in the raw text,
    # but we need to be careful: we must handle markdown *before* escaping
    # because escaping will turn * into something else.

    # Strategy: extract markdown patterns first, then escape the rest.

    # Handle **bold** (must come before *italic*)
    parts = []
    pos = 0
    for m in re.finditer(r'\*\*(.+?)\*\*', text):
        parts.append(latex_escape(text[pos:m.start()]))
        parts.append(r'\textbf{' + latex_escape(m.group(1)) + '}')
        pos = m.end()
    parts.append(latex_escape(text[pos:]))
    text = ''.join(parts)

    # Handle *italic* (avoid matching already-converted \textbf)
    parts = []
    pos = 0
    for m in re.finditer(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', text):
        parts.append(text[pos:m.start()])
        # The content inside was already latex_escaped above
        parts.append(r'\textit{' + m.group(1) + '}')
        pos = m.end()
    parts.append(text[pos:])
    text = ''.join(parts)

    # Handle `code`
    parts = []
    pos = 0
    for m in re.finditer(r'`([^`]+)`', text):
        parts.append(text[pos:m.start()])
        parts.append(r'\texttt{' + m.group(1) + '}')
        pos = m.end()
    parts.append(text[pos:])
    text = ''.join(parts)

    return text


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
\usepackage{placeins}
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

def generate_latex_table(headers: list, rows: list, caption: str = None,
                         label: str = None, col_widths: list = None) -> str:
    """Generate a LaTeX table from headers and rows.

    Parameters
    ----------
    col_widths : list, optional
        Per-column widths. Use a string like '5cm' for wrapped columns (p{5cm}),
        or None for auto-width (l). Length must match len(headers).
    """
    n_cols = len(headers)

    # Build column spec
    if col_widths and len(col_widths) == n_cols:
        specs = []
        for w in col_widths:
            specs.append(f'p{{{w}}}' if w else 'l')
        col_spec = '|' + '|'.join(specs) + '|'
    else:
        col_spec = '|' + 'l|' * n_cols

    tex = r'\begin{table}[H]' + '\n'
    tex += r'\centering' + '\n'
    # Use \small for wide tables (>4 cols) to prevent overflow
    if n_cols > 4:
        tex += r'\small' + '\n'
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
