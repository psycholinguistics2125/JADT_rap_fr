# Report Generation: Architecture & Usage

This document explains how the comparison report generation code works and how to use it, both from the CLI and programmatically.

---

## Table of Contents

1. [Overview](#overview)
2. [CLI Usage](#cli-usage)
3. [Code Architecture](#code-architecture)
4. [Report Package Structure](#report-package-structure)
5. [How the Report Pipeline Works](#how-the-report-pipeline-works)
6. [Translation System](#translation-system)
7. [PDF Generation Engines](#pdf-generation-engines)
8. [Programmatic Usage](#programmatic-usage)
9. [Customization](#customization)

---

## Overview

The report generation system produces a comprehensive statistical comparison report between three topic models (LDA, BERTopic, IRAMUTEQ). It supports:

- **Markdown** output with tables, metric interpretations, and figure references.
- **LaTeX** output with proper mathematical equations.
- **PDF** compilation via xelatex/pdflatex or pypandoc.
- **Bilingual** reports (French and English).

The code lives in `utils/comparaison_utils/report/`, a Python package with 7 modules.

---

## CLI Usage

Reports are generated automatically by `main_comparate_run.py`. The relevant CLI parameters are:

```bash
python main_comparate_run.py \
    --lda-folder results/LDA/run_YYYYMMDD_HHMMSS_both \
    --bertopic-folder results/BERTopic/run_YYYYMMDD_HHMMSS_camembert \
    --iramuteq-folder results/IRAMUTEQ/evaluation_YYYYMMDD_HHMMSS \
    --lang fr \
    --pdf-engine latex
```

### Report-specific parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lang` | str | `fr` | Report language: `fr` (French) or `en` (English) |
| `--pdf-engine` | str | `latex` | PDF generation engine: `latex` or `markdown` |
| `--no-figures` | flag | `False` | Skip figure generation (faster, text-only report) |
| `--include-sankey` | flag | `False` | Include Sankey flow diagrams (requires plotly) |

### Output files

The comparison pipeline creates a timestamped directory under `results/comparisons/` with:

| File | Description |
|------|-------------|
| `comparison_report.md` | Full Markdown report |
| `comparison_report.pdf` | PDF report |
| `comparison_report.tex` | LaTeX source (when using `--pdf-engine latex`) |
| `metrics.json` | All computed metrics as JSON |
| `figures/` | All visualization figures (PNGs, HTMLs) |
| `data/` | Aligned assignments, contingency tables, residuals |

---

## Code Architecture

The report generation is split into a dedicated Python package under `utils/comparaison_utils/report/`:

```
utils/comparaison_utils/report/
├── __init__.py            # Re-exports 7 public functions
├── translations.py        # Translation loading + lookup (FR/EN)
├── translations.json      # All translation strings
├── latex_helpers.py       # LaTeX escaping, table/figure generators
├── sections.py            # Reusable report sections (corpus, model, distance)
├── markdown_report.py     # Full Markdown report generator
├── latex_report.py        # Full LaTeX report generator
└── pdf_compiler.py        # PDF compilation (xelatex/pdflatex/pypandoc)
```

### Internal import chain

```
translations.py      ← imports from ..constants
latex_helpers.py     ← no internal deps
sections.py          ← imports from translations, latex_helpers, ..constants, ..vocabulary, ..visualization
markdown_report.py   ← imports from translations, sections, ..constants
latex_report.py      ← imports from translations, latex_helpers, sections
pdf_compiler.py      ← imports from latex_report
```

### Public API

The package re-exports 7 public functions through `__init__.py`:

```python
from utils.comparaison_utils.report import (
    copy_run_figures,                  # Copy model figures to comparison output
    generate_corpus_description,       # Markdown section for corpus stats
    generate_run_description,          # Markdown section for a single model
    generate_intra_topic_distance_section,  # Distance metrics section
    generate_comparison_report,        # Full Markdown report
    generate_latex_report,             # Full LaTeX report
    generate_pdf_report,               # PDF compilation
)
```

These same functions are also available from the parent package:

```python
from utils.comparaison_utils import generate_comparison_report, generate_pdf_report
```

---

## Report Package Structure

### `translations.py`

Loads translation strings from `translations.json` (cached in memory). Provides two functions:

- `get_text(key, lang)` -- returns the translated string for a key in the given language.
- `get_metric_description(metric_key, lang)` -- returns a formatted description for a statistical metric.

### `translations.json`

JSON file containing all report strings in French and English. Structure:

```json
{
  "fr": {
    "title": "Rapport de Comparaison des Modeles de Topics",
    "abstract_title": "Resume",
    "q1_title": "### 3.1 Question 1 : Accord entre modeles",
    ...
  },
  "en": {
    "title": "Topic Model Comparison Report",
    "abstract_title": "Abstract",
    "q1_title": "### 3.1 Question 1: Model Agreement",
    ...
  }
}
```

### `latex_helpers.py`

Utility functions for LaTeX output:

- `latex_escape(text)` -- escapes special LaTeX characters (`&`, `%`, `$`, `#`, `_`, `{`, `}`, `~`, `^`).
- `latex_safe_number(value, fmt)` -- formats numbers safely for LaTeX.
- `LATEX_PREAMBLE` -- complete LaTeX document preamble (packages, fonts, colors).
- `LATEX_END` -- document closing.
- `generate_latex_table(headers, rows, caption, label)` -- generates a `longtable` environment.
- `generate_latex_figure(path, caption, label, width)` -- generates an `includegraphics` block.

### `sections.py`

Reusable report sections (~700 lines):

- `compute_topic_distribution_metrics(doc_assignments)` -- computes Gini, entropy, imbalance for topic distribution.
- `copy_run_figures(run_dir, figures_dir, model_name)` -- copies model PNG files to the comparison figures directory.
- `generate_corpus_description(df, figures_dir, lang)` -- corpus statistics section (document count, year range, artist count, decade breakdown).
- `generate_run_description(data, title, model_type, ...)` -- individual model section (parameters, topic distribution, top words, metrics).
- `generate_intra_topic_distance_section(distance_results, lang)` -- Q5 intra-topic distance section with JS and Labbe tables.
- `generate_distance_appendix(lang)` -- mathematical appendix with distance formula definitions.

### `markdown_report.py`

The main report generator (~450 lines). `generate_comparison_report(results, output_dir, figures_dir, lang)` produces a complete Markdown document with:

1. **Abstract** -- report summary.
2. **Section 1: Corpus description** -- dataset statistics with figures.
3. **Section 2: Individual models** -- BERTopic, LDA, IRAMUTEQ descriptions.
4. **Section 3: Comparative analysis**
   - Q1: Model agreement (ARI, NMI tables)
   - Q2: Artist separation (Cramer's V)
   - Q3: Temporal dynamics (variance, decade JS)
   - Q4: Vocabulary (Jaccard, distinctiveness)
   - Q5: Intra-topic distance (JS, Labbe)
5. **Section 4: Summary** -- key findings and recommendations.
6. **Section 5: References** -- full academic citations.
7. **Appendix** -- run details and mathematical formulas.

### `latex_report.py`

Full LaTeX report generator (~660 lines). `generate_latex_report(results, output_dir, figures_dir, lang)` produces a complete `.tex` document with:

- Proper mathematical equations (`\begin{equation}...\end{equation}`)
- Professional tables using `longtable`
- Figures with `\includegraphics`
- Table of contents
- Bibliography section

Internal helper functions:
- `_generate_latex_corpus_section()` -- corpus statistics in LaTeX.
- `_generate_latex_model_section()` -- individual model description.
- `_generate_latex_comparison_section()` -- Q1-Q4 comparative analysis.
- `_generate_latex_distance_section()` -- Q5 intra-topic distances.
- `_generate_latex_distance_appendix()` -- mathematical formula appendix.
- `_get_imbalance_interp()`, `_get_entropy_interp()` -- threshold interpreters.

### `pdf_compiler.py`

PDF compilation from LaTeX or Markdown:

- `compile_latex(tex_path, output_dir)` -- compiles `.tex` to PDF using xelatex (preferred) or pdflatex (fallback). Runs twice for table of contents.
- `generate_pdf_report(markdown_content, output_path, lang, pdf_engine, results, output_dir, figures_dir)` -- main entry point:
  - If `pdf_engine='latex'`: generates LaTeX via `generate_latex_report()`, writes `.tex`, compiles to PDF.
  - If `pdf_engine='markdown'`: converts Markdown to PDF via pypandoc with xelatex backend, falls back to pdflatex.

---

## How the Report Pipeline Works

When you run `main_comparate_run.py`, the report generation happens in the final steps:

```
main_comparate_run.py
  │
  ├── Steps 1-4: Validate folders, load data, align documents
  ├── Steps 5-9: Compute Q1-Q5 metrics
  │
  ├── [*] Generate visualizations
  │     └── Copies model figures + creates comparison plots
  │
  ├── [*] Generate Markdown report
  │     └── generate_comparison_report(all_results, output_dir, figures_dir, lang)
  │           ├── generate_corpus_description()     → Section 1
  │           ├── generate_run_description() × 3    → Section 2
  │           ├── Q1-Q5 tables and interpretations   → Section 3
  │           ├── Summary + references               → Section 4-5
  │           └── generate_distance_appendix()       → Appendix
  │
  ├── [*] Generate PDF report
  │     └── generate_pdf_report(markdown, output_path, lang, engine, results, ...)
  │           ├── engine='latex':
  │           │     ├── generate_latex_report(results, output_dir, ...)  → .tex
  │           │     └── compile_latex(tex_path) → .pdf
  │           └── engine='markdown':
  │                 └── pypandoc.convert_text(markdown, 'pdf', ...)  → .pdf
  │
  └── [*] Save metrics JSON
```

### Data Flow

The `results` dictionary passed to report generators has this structure:

```python
results = {
    'bertopic': {                    # BERTopic model data
        'doc_assignments': pd.DataFrame,  # Document-topic assignments
        'topics': dict,              # Topic word distributions
        'metrics': dict,             # Model metrics
        'topic_evolution': pd.DataFrame,  # Topic proportions by year
        'run_dir': str,              # Source run directory
    },
    'lda': { ... },                  # Same structure
    'iramuteq': { ... },             # Same structure
    'agreement': {                   # Q1 results
        'bertopic_vs_lda': {'agreement': {'ari': float, 'nmi': float}, 'contingency': {...}},
        'bertopic_vs_iramuteq': { ... },
        'lda_vs_iramuteq': { ... },
    },
    'artist_separation': {           # Q2 results
        'bertopic_cramers_v': float,
        'lda_cramers_v': float,
        'iramuteq_cramers_v': float,
        ...
    },
    'temporal': {                    # Q3 results
        'bertopic_mean_variance': float,
        'bertopic_decade_js': dict,
        ...
    },
    'vocabulary': {                  # Q4 results
        'bertopic_distinctiveness': float,
        'bertopic_vs_lda': {'mean_jaccard': float, ...},
        ...
    },
    'intra_topic_distances': {       # Q5 results
        'bertopic_js': {'mean': float, 'std': float, 'per_topic': dict},
        'bertopic_labbe': {'mean': float, 'std': float, 'per_topic': dict},
        ...
    },
}
```

---

## Translation System

All report text is externalized in `translations.json`. To add or modify translations:

1. Edit `utils/comparaison_utils/report/translations.json`.
2. Add/modify keys under the `"fr"` and/or `"en"` sections.
3. Use `{variable}` placeholders for dynamic values (e.g., `"q1_obs1": "La paire {pair} montre le meilleur accord (NMI={nmi:.4f})"`).

The translation system uses simple Python string formatting:

```python
from utils.comparaison_utils.report.translations import get_text

# Basic lookup
title = get_text('title', lang='fr')
# → "Rapport de Comparaison des Modeles de Topics"

# With formatting (done by the report generator)
text = get_text('q1_obs1', lang='fr').format(pair='BERTopic vs LDA', nmi=0.35)
```

---

## PDF Generation Engines

### LaTeX engine (`--pdf-engine latex`)

Produces a professional document with proper mathematical equations. Requires xelatex or pdflatex.

**How it works:**
1. `generate_latex_report()` builds a complete `.tex` document using `LATEX_PREAMBLE`, content sections, and `LATEX_END`.
2. `compile_latex()` runs the LaTeX compiler twice (for TOC resolution).
3. Tries xelatex first (better Unicode support for French), falls back to pdflatex.

**Generated files:** `.tex` (source) + `.pdf` (compiled)

**Requirements:**
```bash
# Ubuntu/Debian
sudo apt install texlive-xetex texlive-latex-extra texlive-fonts-recommended

# macOS
brew install --cask mactex
```

### Markdown engine (`--pdf-engine markdown`)

Uses pypandoc to convert Markdown to PDF. Simpler setup, but equations render as text.

**How it works:**
1. Extracts the `# Title` from Markdown and converts it to YAML metadata (so it appears before the TOC).
2. Calls `pypandoc.convert_text()` with xelatex backend.
3. Falls back to pdflatex without custom fonts if xelatex fails.

**Generated files:** `.pdf` only (no `.tex` source)

**Requirements:**
```bash
pip install pypandoc
# Plus a LaTeX distribution (xelatex or pdflatex)
```

### Fallback chain

```
latex engine → xelatex → pdflatex → markdown engine → xelatex → pdflatex → failure
```

If all engines fail, the Markdown report is still saved and usable.

---

## Programmatic Usage

You can generate reports from Python code without using the CLI:

```python
from utils.comparaison_utils import (
    generate_comparison_report,
    generate_pdf_report,
    generate_latex_report,
)

# Generate Markdown report
markdown_content = generate_comparison_report(
    results=all_results,       # dict with all model data and metrics
    output_dir="./output",
    figures_dir="./output/figures",
    lang='fr'                  # 'fr' or 'en'
)

# Save Markdown
with open("./output/report.md", "w") as f:
    f.write(markdown_content)

# Generate PDF (latex engine)
success = generate_pdf_report(
    markdown_content=markdown_content,
    output_path="./output/report.pdf",
    lang='fr',
    pdf_engine='latex',
    results=all_results,
    output_dir="./output",
    figures_dir="./output/figures"
)

# Or generate LaTeX source directly
latex_content = generate_latex_report(
    results=all_results,
    output_dir="./output",
    figures_dir="./output/figures",
    lang='fr'
)
with open("./output/report.tex", "w") as f:
    f.write(latex_content)
```

### Using individual sections

You can also use individual section generators:

```python
from utils.comparaison_utils.report import (
    generate_corpus_description,
    generate_run_description,
    generate_intra_topic_distance_section,
    copy_run_figures,
)

# Generate just the corpus description section
corpus_md = generate_corpus_description(
    df=doc_assignments_df,
    figures_dir="./figures",
    lang='fr'
)

# Generate a single model description
model_md = generate_run_description(
    data=bertopic_data,
    title="2.1 BERTopic",
    model_type="bertopic",
    figures_dir="./figures",
    comparison_figures_dir="./figures",
    lang='fr'
)

# Copy figures from a run directory
copied_files = copy_run_figures(
    run_dir="results/BERTopic/run_20260126_130645_camembert",
    figures_dir="./output/figures",
    model_name="bertopic"
)
```

---

## Customization

### Adding a new report section

1. Add the section generator function to `sections.py`.
2. Call it from `markdown_report.py` (for Markdown) and `latex_report.py` (for LaTeX).
3. Add any new translation keys to `translations.json`.

### Adding a new language

1. Add a new top-level key in `translations.json` (e.g., `"es"` for Spanish).
2. Translate all existing keys.
3. The `--lang` parameter in `main_comparate_run.py` accepts any language code present in the JSON.

### Modifying the LaTeX template

Edit `LATEX_PREAMBLE` and `LATEX_END` in `latex_helpers.py` to change:
- Page margins, fonts, colors
- Header/footer configuration
- Package imports
