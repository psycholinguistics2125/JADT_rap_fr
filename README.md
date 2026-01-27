# JADT_rap_fr

Analyse thematique d'un corpus de rap francais (Thematic analysis of a French rap corpus).

**Paper abstract:** [Google Docs](https://docs.google.com/document/d/1CtiysCRxDHNxa3nZk1hDG6pVS5jVDXoWI75YgkcPBM0/edit?usp=sharing)

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Code Logic](#code-logic)
5. [Topic Modeling Scripts](#topic-modeling-scripts)
6. [Quick Start](#quick-start)
7. [Batch Experiments](#batch-experiments)
8. [Output Files](#output-files)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Data Format](#data-format)
11. [Documentation](#documentation)

---

## Overview

This project compares three topic modeling approaches applied to a corpus of French rap lyrics:

- **LDA** (Latent Dirichlet Allocation) -- probabilistic bag-of-words topic modeling with n-gram support (Gensim).
- **BERTopic** -- neural topic modeling using pre-trained sentence embeddings (CamemBERT, E5, MPNet) with UMAP dimensionality reduction and configurable clustering.
- **IRAMUTEQ** (Reinert method) -- evaluation of a pre-computed hierarchical descending classification.

All three methods produce standardized evaluation metrics (artist separation, temporal evolution, vocabulary distinctiveness) enabling rigorous cross-model comparison. A dedicated comparison pipeline generates a statistical report (Markdown + PDF) answering five research questions about model agreement, artist separation, temporal dynamics, vocabulary overlap, and intra-topic coherence.

---

## Project Structure

```
JADT_rap_fr/
├── data/                                  # Corpus CSV files
├── models/
│   └── embeddings/                        # Cached sentence embeddings (.npy)
├── results/
│   ├── LDA/                               # LDA run outputs (timestamped)
│   ├── BERTopic/                          # BERTopic run outputs (timestamped)
│   ├── IRAMUTEQ/                          # IRAMUTEQ evaluation outputs
│   └── comparisons/                       # Comparison report outputs
├── logs/                                  # Batch run logs
├── notebooks/                             # Jupyter notebooks (exploration)
│   ├── bertopic_songs.ipynb
│   ├── bertopic_verses.ipynb
│   ├── bertopic_verses_with_seed.ipynb
│   ├── bunka.ipynb
│   └── lda_songs.ipynb
├── docs/                                  # Documentation
│   ├── TOPIC_MODELING_SCRIPTS.md          # Full CLI reference for all scripts
│   ├── REPORT_GENERATION_CLI.md           # Report generation code & usage
│   └── Instruction_to_test.md             # Step-by-step testing guide
├── utils/
│   ├── utils_evaluation.py                # Shared evaluation (metrics, plots)
│   ├── utils_visualization_html.py        # Interactive HTML visualization
│   ├── clean_lyrics.py                    # Lyrics cleaning functions
│   ├── clean_songs.py                     # Song-level cleaning
│   ├── clean_data.ipynb                   # Data cleaning notebook
│   ├── api_year_finder.py                 # Year lookup via API
│   └── comparaison_utils/                 # Comparison pipeline package
│       ├── __init__.py                    # Public API (re-exports all modules)
│       ├── constants.py                   # Scientific references, metric defs
│       ├── data_loading.py                # Load & align model outputs
│       ├── agreement.py                   # Q1: ARI, NMI, AMI, contingency
│       ├── artist_separation.py           # Q2: Cramer's V, residuals
│       ├── temporal.py                    # Q3: temporal variance, JS divergence
│       ├── vocabulary.py                  # Q4: Jaccard, distinctiveness
│       ├── topic_distances.py             # Q5: Labbe, JS, WMD distances
│       ├── visualization.py              # Comparison plots (Sankey, heatmaps)
│       └── report/                        # Report generation package
│           ├── __init__.py                # Re-exports public report functions
│           ├── translations.py            #  (French/English)
│           ├── translations.json          # Translation strings
│           ├── sections.py                # Reusable report sections
│           ├── markdown_report.py         # Markdown report generator
│           ├── latex_report.py            # LaTeX report generator
│           ├── latex_helpers.py           # LaTeX escaping, tables, figures
│           └── pdf_compiler.py            # PDF compilation (xelatex/pypandoc)
│
├── build_and_evaluate_bertopic.py         # BERTopic training + evaluation
├── build_and_evaluate_LDA.py              # LDA training + evaluation
├── evaluate_iramuteq.py                   # IRAMUTEQ evaluation
├── main_comparate_run.py                  # Cross-model comparison pipeline
├── main_run_multiple_evaluation.py        # Batch experiment runner
├── script_main_clean_data.py              # Data preprocessing
├── script_main_create_verses_df.py        # Verse extraction
├── script_test_bertopic_parameter.py      # BERTopic parameter search
├── test_with_sample.py                    # Quick validation test
├── config.yaml                            # Configuration file
├── requirements.txt                       # Python dependencies
└── .env                                   # API keys (not committed)
```

---

## Installation

### 1. Create a virtual environment (recommended)

It is strongly recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Create the virtual environment
python3 -m venv rap_fr

# Activate it
source rap_fr/bin/activate        # Linux / macOS
# or
rap_fr\Scripts\activate           # Windows
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install spaCy French model (for comparison reports)

The comparison pipeline (and LDA) uses spaCy for tokenization. Install the model you need:

```bash
# Large model (recommended, best accuracy)
python -m spacy download fr_core_news_lg

# Or medium model (faster, slightly less accurate)
python -m spacy download fr_core_news_md

# Or small model (fastest)
python -m spacy download fr_core_news_sm
```

You can also skip spaCy entirely by passing `--spacy-model none` to use NLTK tokenization instead.

### 4. (Optional) Install LaTeX for PDF reports

If you want PDF reports with proper equation rendering:

```bash
# Ubuntu / Debian
sudo apt install texlive-xetex texlive-latex-extra texlive-fonts-recommended

# macOS
brew install --cask mactex
```

Alternatively, use `--pdf-engine markdown` to generate PDFs via pypandoc without LaTeX.

### 5. (Optional) OpenAI API key

For BERTopic automatic topic labeling, create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

## Code Logic

The project follows a three-stage pipeline:

### Stage 1: Individual Model Training & Evaluation

Each script (`build_and_evaluate_bertopic.py`, `build_and_evaluate_LDA.py`, `evaluate_iramuteq.py`) independently:

1. **Loads** the corpus CSV (`data/20260123_filter_verses_lrfaf_corpus.csv`).
2. **Trains** the topic model (or loads pre-computed IRAMUTEQ classes).
3. **Evaluates** using shared metrics from `utils/utils_evaluation.py`:
   - Cluster distribution metrics
   - Artist separation (specialization score, JS divergence between artist profiles)
   - Temporal evolution (topic variance over years, decade JS divergence)
4. **Generates** standardized outputs: `metrics.json`, `doc_assignments.csv`, visualization PNGs.

All outputs go to timestamped directories under `results/{METHOD}/`.

### Stage 2: Cross-Model Comparison

`main_comparate_run.py` takes one run folder from each method and computes:

| Step | Research Question | Metrics |
|------|-------------------|---------|
| Q1 | Do models agree on document clustering? | ARI, NMI, AMI, contingency tables |
| Q2 | Do models separate artists differently? | Cramer's V, standardized residuals |
| Q3 | Do models capture the same temporal dynamics? | Temporal variance, decade JS divergence |
| Q4 | Do models use different vocabularies? | Jaccard similarity, vocabulary distinctiveness |
| Q5 | Are topics internally coherent? | Intra-topic JS distance, Labbe distance |

### Stage 3: Report Generation

The comparison pipeline generates:
- A **Markdown report** with tables, metric interpretations, and figure references.
- A **PDF report** via LaTeX (proper equations) or pypandoc (simpler, no LaTeX needed).
- A **metrics JSON** file with all computed values.
- **Visualization figures** (heatmaps, Sankey diagrams, temporal plots).

The report generation code lives in `utils/comparaison_utils/report/` as a dedicated package with translation support (French/English).

### Key Utilities

| Module | Purpose |
|--------|---------|
| `utils/utils_evaluation.py` | Shared evaluation metrics and plot functions used by all three model scripts |
| `utils/comparaison_utils/` | Full comparison pipeline: data loading, statistical metrics, visualization, report generation |
| `utils/comparaison_utils/topic_distances.py` | SpaCy/NLTK tokenizers + Labbe/JS/WMD distance implementations |
| `utils/comparaison_utils/report/` | Report generation package (Markdown, LaTeX, PDF) with French/English i18n |

---

## Topic Modeling Scripts

| Method | Script | Description |
|--------|--------|-------------|
| **LDA** | `build_and_evaluate_LDA.py` | Latent Dirichlet Allocation with n-gram support |
| **BERTopic** | `build_and_evaluate_bertopic.py` | Neural topic modeling with sentence embeddings |
| **IRAMUTEQ** | `evaluate_iramuteq.py` | Evaluation of pre-computed Reinert classification |
| **Comparison** | `main_comparate_run.py` | Cross-model statistical comparison & report |

**For the full CLI reference with all parameters, see: [docs/TOPIC_MODELING_SCRIPTS.md](docs/TOPIC_MODELING_SCRIPTS.md)**

---

## Quick Start

```bash
# Activate your virtual environment
source rap_fr/bin/activate

# Run LDA (20 topics, with bigrams and trigrams)
python build_and_evaluate_LDA.py --topics 20 --ngrams both

# Run BERTopic with French CamemBERT embeddings
python build_and_evaluate_bertopic.py --embedding camembert --compute-embeddings

# Evaluate IRAMUTEQ classification
python evaluate_iramuteq.py

# Compare all three models
python main_comparate_run.py \
    --lda-folder results/LDA/run_YYYYMMDD_HHMMSS_both \
    --bertopic-folder results/BERTopic/run_YYYYMMDD_HHMMSS_camembert \
    --iramuteq-folder results/IRAMUTEQ/evaluation_YYYYMMDD_HHMMSS
```

### Available Parameters (summary)

#### LDA

```bash
python build_and_evaluate_LDA.py [OPTIONS]

Options:
  --topics N          Number of topics (default: 20)
  --passes N          Training passes (default: 15)
  --iterations N      Max iterations (default: 400)
  --ngrams MODE       N-gram mode: unigrams, bigrams, trigrams, both,
                      ngrams_only, bigram_only, trigram_only (default: both)
  --sample N          Sample size for testing
  --no-pyldavis       Skip pyLDAvis HTML generation
```

#### BERTopic

```bash
python build_and_evaluate_bertopic.py [OPTIONS]

Options:
  --embedding MODEL     Embedding model: camembert, e5, mpnet (default: camembert)
  --compute-embeddings  Compute and save embeddings (only needed with --no-keybert)
  --clustering ALG      Algorithm: kmeans, hdbscan, agglomerative (default: kmeans)
  --clusters N          Number of clusters (default: 20)
  --sample N            Sample size for testing
  --no-openai           Disable OpenAI topic labeling
  --no-keybert          Disable KeyBERTInspired representation
  --no-interactive-html Disable interactive HTML visualization
```

#### IRAMUTEQ

```bash
python evaluate_iramuteq.py [OPTIONS]

Options:
  --sample N            Sample size for testing
  --min-docs-artist N   Min documents per artist (default: 10)
```

#### Comparison

```bash
python main_comparate_run.py [OPTIONS]

Options:
  --lda-folder PATH       Path to LDA results folder (required)
  --bertopic-folder PATH  Path to BERTopic results folder (required)
  --iramuteq-folder PATH  Path to IRAMUTEQ results folder (required)
  --lang LANG             Report language: fr or en (default: fr)
  --pdf-engine ENGINE     PDF engine: latex or markdown (default: latex)
  --spacy-model MODEL     Tokenizer: fr_core_news_sm/md/lg or none (default: lg)
```

---

## Batch Experiments

Run multiple configurations in a tmux session:

```bash
# Start tmux session
tmux new -s topic_models

# Run batch experiments (3 BERTopic + 4 LDA + 1 IRAMUTEQ)
python main_run_multiple_evaluation.py

# Detach: Ctrl+B, D
# Reattach: tmux attach -t topic_models
```

This runs:
- **BERTopic**: camembert, e5, mpnet embeddings
- **LDA**: bigram_only, trigram_only, ngrams_only (bi+tri), both (uni+bi+tri)
- **IRAMUTEQ**: evaluation of pre-computed classes

---

## Output Files

Each evaluation creates a timestamped directory with:

| File | Description |
|------|-------------|
| `metrics.json` | All computed metrics |
| `doc_assignments.csv` | Document-to-topic assignments |
| `topic_distribution.png` | Topic size distribution |
| `topic_evolution_heatmap.png` | Topic prevalence over time |
| `year_topic_heatmap.png` | Topic distribution by year |
| `artist_topics_heatmap.png` | Artist topic profiles |
| `artist_specialization.png` | Artist classification distribution |
| `biannual_js_divergence.png` | 2-year temporal changes |
| `interactive_bertopic.html` | Interactive topic explorer (BERTopic only) |

---

## Evaluation Metrics

All methods produce comparable metrics:

**Artist Separation** (only artists with >= `min_docs_artist` verses):
- Artist specialization score (mean dominant topic ratio)
- Jensen-Shannon divergence between artist profiles
- Classification: specialists (>50%), moderate (30-50%), generalists (<30%)

**Temporal Evolution:**
- Mean temporal variance
- Decade-to-decade JS divergence
- 2-year window JS divergence (finer granularity)

**Why Jensen-Shannon Divergence?**
- Symmetric: JS(P||Q) = JS(Q||P)
- Bounded: 0 (identical) to 1 (maximally different)
- Always defined (handles zeros in distributions)
- Information-theoretic interpretation

See [utils/utils_evaluation.py](utils/utils_evaluation.py) for implementation details.

---

## Data Format

The corpus should be a CSV file with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `lyrics_cleaned` | Yes | Preprocessed text |
| `artist` | Yes | Artist name |
| `year` | Yes | Year of release |
| `title` | Yes | Song title |
| `IRAMUTEQ_CLASSES` | For IRAMUTEQ only | Pre-computed Reinert classification |

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/TOPIC_MODELING_SCRIPTS.md](docs/TOPIC_MODELING_SCRIPTS.md) | Full CLI reference for all scripts (LDA, BERTopic, IRAMUTEQ, Comparison) |
| [docs/REPORT_GENERATION_CLI.md](docs/REPORT_GENERATION_CLI.md) | Report generation code architecture and usage |
| [docs/Instruction_to_test.md](docs/Instruction_to_test.md) | Step-by-step guide to run a complete test from scratch |

---

## License

[Add license information]

## Citation

[Add citation information for JADT conference]
