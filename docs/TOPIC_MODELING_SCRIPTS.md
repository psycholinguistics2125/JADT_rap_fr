# Topic Modeling Evaluation Scripts Documentation

This documentation covers the three topic modeling/clustering evaluation scripts used for analyzing the French Rap corpus:

1. **LDA (Latent Dirichlet Allocation)** - `build_and_evaluate_LDA.py`
2. **BERTopic** - `build_and_evaluate_bertopic.py`
3. **IRAMUTEQ** - `evaluate_iramuteq.py`

All three scripts produce standardized evaluation metrics and visualizations for comparison.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [LDA Script](#lda-script)
3. [BERTopic Script](#bertopic-script)
4. [IRAMUTEQ Script](#iramuteq-script)
5. [Shared Evaluation Metrics](#shared-evaluation-metrics)
6. [Output Files](#output-files)
7. [Running Multiple Experiments](#running-multiple-experiments)
8. [Model Comparison Script](#model-comparison-script)
9. [Report Utilities](#report-utilities)

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install gensim scikit-learn bertopic sentence-transformers umap-learn hdbscan
pip install matplotlib seaborn pandas numpy scipy
pip install pyLDAvis  # For LDA visualization
pip install python-dotenv openai  # For BERTopic OpenAI labeling (optional)
```

### Basic Usage

```bash
# Run LDA with default parameters (20 topics)
python build_and_evaluate_LDA.py

# Run BERTopic with CamemBERT embeddings
python build_and_evaluate_bertopic.py --embedding camembert

# Evaluate pre-computed IRAMUTEQ classes
python evaluate_iramuteq.py
```

### Sample Mode (for testing)

```bash
# Test with a small sample
python build_and_evaluate_LDA.py --sample 1000
python build_and_evaluate_bertopic.py --sample 1000  # auto-computes embeddings
python evaluate_iramuteq.py --sample 1000
```

---

## LDA Script

### Description

`build_and_evaluate_LDA.py` builds a Latent Dirichlet Allocation topic model using Gensim. It supports n-gram detection (bigrams, trigrams) and provides comprehensive evaluation metrics including coherence scores.

### Command Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--topics` | int | 20 | Number of topics to extract |
| `--passes` | int | 15 | Number of passes through the corpus |
| `--iterations` | int | 400 | Maximum iterations per pass |
| `--alpha` | str/float | 'symmetric' | Document-topic density. Options: 'symmetric', 'asymmetric', or a float value |
| `--eta` | str/float | 'auto' | Topic-word density. Options: 'auto' or a float value |
| `--sample` | int | None | Sample size for testing (uses full dataset if None) |
| `--load-corpus` | str | None | Path to pre-computed corpus (pickle file) |
| `--no-save-corpus` | flag | False | Do not save the corpus to disk |
| `--no-pyldavis` | flag | False | Do not create pyLDAvis HTML visualization |
| `--num-words` | int | 30 | Number of top words per topic to save |
| `--top-artists-topic` | int | 20 | Number of top artists to show per topic |
| `--top-artists-heatmap` | int | 50 | Number of top artists in heatmap visualizations |

#### N-gram Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--ngrams` | str | 'both' | N-gram mode (see options below) |
| `--ngram-min-count` | int | 10 | Minimum frequency for n-gram detection |
| `--ngram-threshold` | int | 50 | Threshold for n-gram detection (higher = stricter) |

**N-gram Mode Options:**
- `unigrams`: Only single words (no n-gram detection)
- `bigrams`: Unigrams + bigrams
- `trigrams`: Unigrams + trigrams
- `both`: Unigrams + bigrams + trigrams (default)
- `ngrams_only`: Only bigrams + trigrams (no unigrams)
- `bigram_only`: Only bigrams (no unigrams, no trigrams)
- `trigram_only`: Only trigrams (no unigrams, no bigrams)

### Examples

```bash
# Standard run with 25 topics
python build_and_evaluate_LDA.py --topics 25

# High-quality run with more passes
python build_and_evaluate_LDA.py --topics 20 --passes 30 --iterations 600

# Only bigrams, stricter threshold
python build_and_evaluate_LDA.py --ngrams bigram_only --ngram-threshold 100

# Asymmetric alpha for varied topic sizes
python build_and_evaluate_LDA.py --alpha asymmetric

# Load pre-computed corpus
python build_and_evaluate_LDA.py --load-corpus results/LDA/corpus.pkl
```

### LDA-Specific Outputs

- `coherence_plot.png`: Bar plot of per-topic C_V coherence scores
- `topic_pca.png`: PCA visualization of documents colored by dominant topic
- `pyldavis.html`: Interactive pyLDAvis visualization
- `lda_model/`: Saved Gensim LDA model (can be reloaded)
- `dictionary.pkl`: Gensim dictionary
- `corpus.pkl`: Bag-of-words corpus

---

## BERTopic Script

### Description

`build_and_evaluate_bertopic.py` builds a BERTopic model using pre-trained sentence embeddings. It supports multiple embedding models optimized for French text and uses UMAP for dimensionality reduction with configurable clustering (KMeans by default, HDBSCAN optional).

### Command Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--embedding` | str | 'camembert' | Embedding model to use |
| `--compute-embeddings` | flag | False | Compute and save embeddings (only needed with `--no-keybert`) |
| `--clustering` | str | 'kmeans' | Clustering algorithm: 'kmeans', 'hdbscan', or 'agglomerative' |
| `--clusters` | int | 20 | Number of clusters (for kmeans and agglomerative) |
| `--sample` | int | None | Sample size (auto-computes embeddings) |
| `--no-openai` | flag | False | Disable OpenAI topic labeling |
| `--num-words` | int | 30 | Number of top words per topic |
| `--top-artists-topic` | int | 20 | Top N artists per topic |
| `--top-artists-heatmap` | int | 50 | Top N artists in heatmap |
| `--no-keybert` | flag | False | Disable KeyBERTInspired representation (enabled by default) |
| `--no-interactive-html` | flag | False | Disable interactive HTML visualization (enabled by default) |

#### Clustering Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--clustering` | str | 'kmeans' | Algorithm: 'kmeans', 'hdbscan', or 'agglomerative' |
| `--clusters` | int | 20 | Number of clusters (KMeans and Agglomerative) |

**HDBSCAN Parameters** (only used if `--clustering=hdbscan`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--hdbscan-min-cluster-size` | int | 15 | Min cluster size (larger = fewer clusters) |
| `--hdbscan-min-samples` | int | 10 | Min samples for core points |
| `--hdbscan-selection` | str | 'eom' | Cluster selection: 'eom' (coarse) or 'leaf' (fine) |

**Agglomerative Parameters** (only used if `--clustering=agglomerative`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--agglomerative-linkage` | str | 'ward' | Linkage: 'ward', 'complete', 'average', 'single' |
| `--agglomerative-metric` | str | 'euclidean' | Distance metric (euclidean required for ward) |

#### UMAP Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--umap-neighbors` | int | 15 | UMAP n_neighbors (local vs global structure) |
| `--umap-components` | int | 5 | UMAP n_components (dimensionality) |
| `--umap-min-dist` | float | 0.0 | UMAP min_dist (cluster tightness) |

**Available Embedding Models:**
- `camembert`: `dangvantuan/sentence-camembert-base` - French-optimized (recommended)
- `e5`: `intfloat/multilingual-e5-base` - Multilingual model
- `mpnet`: `sentence-transformers/all-mpnet-base-v2` - General English model

### Examples

```bash
# Standard run with CamemBERT
python build_and_evaluate_bertopic.py --embedding camembert

# Use multilingual E5 embeddings
python build_and_evaluate_bertopic.py --embedding e5 --compute-embeddings

# More clusters with tighter UMAP
python build_and_evaluate_bertopic.py --clusters 30 --umap-neighbors 10 --umap-min-dist 0.1

# Disable OpenAI labeling (faster)
python build_and_evaluate_bertopic.py --no-openai

# Test with sample (must compute embeddings for sample)
python build_and_evaluate_bertopic.py --sample 1000  # auto-computes embeddings

# Without KeyBERTInspired representation (faster)
python build_and_evaluate_bertopic.py --embedding camembert --no-keybert

# Without interactive HTML visualization
python build_and_evaluate_bertopic.py --embedding camembert --no-interactive-html

# Minimal run (no KeyBERT, no HTML, no OpenAI)
python build_and_evaluate_bertopic.py --embedding camembert --no-keybert --no-interactive-html --no-openai
```

### BERTopic-Specific Outputs

- `silhouette_plot.png`: Per-cluster silhouette scores
- `umap_topics.png`: 2D UMAP visualization colored by topic
- `bertopic_model/`: Saved BERTopic model
- `interactive_bertopic.html`: Interactive visualization (with `--interactive-html`)
- Embeddings cached in `models/embeddings/`

### KeyBERTInspired Representation

KeyBERTInspired is **enabled by default**. Use `--no-keybert` to disable it.

When enabled, the script loads the SentenceTransformer model and computes embeddings automatically (no need for `--compute-embeddings`). This representation method:

- Uses the embedding model to find keywords that are semantically similar to the topic
- Produces different keywords than c-TF-IDF (statistical) or MMR (diversity-focused)
- Requires the embedding model to be loaded (increases memory usage)
- Results are saved in `topics.json` under the "keybert" key

**Note:** When KeyBERT is enabled, embeddings are computed on-the-fly and not saved (to avoid overwriting cached full embeddings). Use `--no-keybert --compute-embeddings` to save embeddings for future runs without KeyBERT.

### Interactive HTML Visualization

Interactive HTML visualization is **enabled by default**. Use `--no-interactive-html` to disable it.

The script generates a pyLDAvis-style interactive visualization:

- **Left panel**: UMAP scatter plot of all documents colored by topic
- **Right panel**: Topic details (updates when you click on a cluster)
  - Keywords from all representations (c-TF-IDF, MMR, KeyBERT)
  - Top artists for the topic
  - Year distribution mini-chart
  - Example documents

The HTML file is self-contained (uses CDN for Plotly.js) and can be opened in any modern browser.

### Environment Variables

For OpenAI topic labeling, set in `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

---

## IRAMUTEQ Script

### Description

`evaluate_iramuteq.py` evaluates pre-computed IRAMUTEQ classification (Reinert method). Unlike LDA and BERTopic, this script does not build a model - it evaluates existing cluster assignments stored in the `IRAMUTEQ_CLASSES` column of the dataset.

### Command Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sample` | int | None | Sample size for testing |
| `--min-docs-artist` | int | 10 | Minimum documents per artist for metrics |
| `--top-artists-topic` | int | 20 | Top N artists per topic |
| `--top-artists-heatmap` | int | 50 | Top N artists in heatmap |

### Examples

```bash
# Standard evaluation
python evaluate_iramuteq.py

# Test with sample
python evaluate_iramuteq.py --sample 1000

# Stricter artist filtering
python evaluate_iramuteq.py --min-docs-artist 20
```

### IRAMUTEQ-Specific Outputs

- `class_distribution.png`: Bar plot with document counts per class

---

## Shared Evaluation Metrics

All three scripts compute the same standardized metrics using `utils/utils_evaluation.py`:

### Cluster Metrics

- **Number of topics/clusters**: Count of distinct clusters
- **Documents per topic**: Distribution of documents across topics

### Artist Separation Metrics

These metrics measure how artists specialize across topics:

- **Artist Specialization Score**: Mean proportion of artist's documents in their dominant topic
- **Mean JS Divergence**: Average Jensen-Shannon divergence between artist topic profiles
- **Classification Distribution**:
  - **Specialists**: >50% of documents in dominant topic
  - **Moderate**: 30-50% in dominant topic
  - **Generalists**: <30% in dominant topic
- **Mean Dominant Ratio**: Average ratio of dominant topic across artists
- **Mean Significant Topics**: Average number of topics with >10% of artist's documents

**Note on `--min-docs-artist` parameter:**

This parameter (default: 10) sets the minimum number of documents (verses) an artist must have to be included in artist separation metrics. Artists with very few verses give unreliable statistics - for example, an artist with only 2 verses who happens to have both in topic 5 would appear as a "100% specialist" when this is just noise, not meaningful specialization.

- **Lower value (e.g., 5)**: More artists included, but noisier statistics
- **Higher value (e.g., 20)**: Fewer artists, but more reliable per-artist metrics

The global metrics (mean specialization, % specialists, etc.) are computed only over "valid" artists who meet this threshold.

### Temporal Separation Metrics

These metrics measure how topics evolve over time:

- **Mean Temporal Variance**: Variance in topic proportions across years
- **Decade Changes (JS)**: Jensen-Shannon divergence between consecutive decades
- **Biannual Changes (JS)**: Jensen-Shannon divergence between consecutive 2-year windows
- **Trend Correlations**: Topics with increasing/decreasing prevalence over time

### Why Jensen-Shannon Divergence?

We use Jensen-Shannon (JS) divergence for comparing probability distributions because:

1. **Symmetry**: JS(P||Q) = JS(Q||P), important when no "reference" distribution exists
2. **Bounded**: Values between 0 (identical) and 1 (maximally different)
3. **Always defined**: Works even when distributions have zeros
4. **Metric property**: Square root of JS is a proper metric
5. **Information-theoretic**: Measures information lost when using average distribution

---

## Output Files

Each run creates a timestamped directory under `results/{METHOD}/evaluation_YYYYMMDD_HHMMSS/`:

### Common Outputs (All Methods)

| File | Description |
|------|-------------|
| `metrics.json` | All computed metrics in JSON format |
| `doc_assignments.csv` | Document-to-topic assignments |
| `topic_distribution.png` | Bar plot of documents per topic |
| `topic_evolution_heatmap.png` | Heatmap of topic prevalence over time |
| `year_topic_heatmap.png` | Heatmap of topic distribution by year |
| `artist_topics_heatmap.png` | Top N artists' topic profiles |
| `artist_specialization.png` | Artist classification distribution |
| `biannual_js_divergence.png` | 2-year window JS divergence plot |
| `topic_evolution.csv` | Topic proportions by year |
| `biannual_js_divergence.csv` | JS values between 2-year windows |
| `topic_top_artists.csv` | Top artists per topic |
| `artist_topic_metrics.csv` | Per-artist metrics |

### Method-Specific Outputs

**LDA:**
- `topics.json`: Topic word distributions
- `coherence_plot.png`: Per-topic coherence
- `topic_pca.png`: PCA visualization
- `pyldavis.html`: Interactive visualization
- `lda_model/`, `dictionary.pkl`, `corpus.pkl`

**BERTopic:**
- `topics.json`: Topic representations (c-TF-IDF, MMR, KeyBERT if enabled)
- `silhouette_plot.png`: Silhouette scores
- `umap_topics.png`: UMAP visualization
- `interactive_bertopic.html`: Interactive visualization (with `--interactive-html`)
- `bertopic_model/`

**IRAMUTEQ:**
- `class_distribution.json`: Class counts
- `class_distribution.png`: Class size bar plot

---

## Running Multiple Experiments

### Using tmux for Long-Running Jobs

```bash
# Start a new tmux session
tmux new -s topic_models

# Run the batch script
python main_run_multiple_evaluation.py

# Detach with Ctrl+B, then D
# Reattach later with: tmux attach -t topic_models
```

### Batch Experiment Script

Use `main_run_multiple_evaluation.py` to run multiple configurations:

```bash
# Run all experiments (3 BERTopic + 4 LDA configurations)
python main_run_multiple_evaluation.py

# This will run:
# - BERTopic with camembert, e5, mpnet embeddings
# - LDA with bigram_only, trigram_only, ngrams_only, both
```

See the script for customization options.

---

## Tips and Best Practices

### Memory Considerations

- BERTopic embeddings require significant RAM (~8GB for full corpus)
- Use `--sample` for testing before full runs
- Pre-compute embeddings once, then reuse

### Choosing Parameters

**LDA:**
- Start with 20 topics, adjust based on coherence scores
- Higher passes (20-30) improve quality but increase time
- 'symmetric' alpha gives balanced topics

**BERTopic:**
- CamemBERT recommended for French text
- Lower UMAP neighbors (10-15) preserves local structure
- UMAP components of 5 balances speed/quality

### Comparing Results

- All methods produce comparable `metrics.json` files
- Use the shared metrics (artist separation, temporal evolution) for comparison
- Method-specific metrics (coherence for LDA, silhouette for BERTopic) are not directly comparable

---

## Troubleshooting

### Common Issues

**"IRAMUTEQ_CLASSES column not found"**
- Ensure your dataset has the IRAMUTEQ classification column

**"Embeddings don't match documents"**
- For sample runs: embeddings are now auto-computed (no action needed)
- For full runs: use `--compute-embeddings` to recompute

**Division by zero in artist metrics**
- Happens when no artists have enough documents (increase sample or decrease `--min-docs-artist`)

**CUDA out of memory**
- Reduce batch size in embedding computation
- Use CPU with smaller sample first

### Getting Help

Check the source code comments for detailed explanations of each metric and function.

---

## Model Comparison Script

### Description

`main_comparate_run.py` compares the three topic modeling approaches (LDA, BERTopic, IRAMUTEQ) and generates a comprehensive comparison report with statistical analyses.

### Command Line Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--lda-folder` | str | (required) | Path to LDA results folder |
| `--bertopic-folder` | str | (required) | Path to BERTopic results folder |
| `--iramuteq-folder` | str | (required) | Path to IRAMUTEQ results folder |
| `--iramuteq-original` | str | None | Path to original IRAMUTEQ folder with profiles.csv |
| `--output-dir` | str | (auto-generate) | Output directory for comparison results |
| `--corpus` | str | (default path) | Path to corpus CSV file |
| `--text-column` | str | 'lyrics_cleaned' | Column name containing document text |
| `--lang` | str | 'fr' | Report language: 'fr' (French) or 'en' (English) |
| `--pdf-engine` | str | 'latex' | PDF generation engine (see below) |
| `--tokenizer` | str | 'spacy' | Tokenizer backend: 'spacy', 'nltk', or 'space' (see below) |
| `--spacy-model` | str | 'fr_core_news_lg' | SpaCy model (only when `--tokenizer spacy`) |
| `--aggregation-size` | int | 20 | Number of verses to aggregate for aggregated distance modes |
| `--top-words` | int | 30 | Number of top words per topic for vocabulary analysis |
| `--min-docs-artist` | int | 10 | Minimum documents per artist for analysis |
| `--no-sankey` | flag | False | Skip Sankey diagram generation (requires plotly) |
| `--no-figures` | flag | False | Skip figure generation (faster, report only) |

### Tokenizer for Distance Metrics

The script tokenizes all corpus documents **once** and reuses the tokens for each model's Labbé and Jensen-Shannon distance computations. Three tokenizer backends are available, selected via `--tokenizer`:

| `--tokenizer` | Backend | Lemmatization | Requirements | Best For |
|----------------|---------|---------------|--------------|----------|
| `spacy` (default) | SpaCy | No (disabled) | `python -m spacy download fr_core_news_lg` | Accurate POS-aware tokenization |
| `nltk` | NLTK | No | `nltk` (punkt_tab, stopwords) | No spaCy dependency, simpler tokenization |
| `space` | SimpleSpace | No | None | Fastest, for testing (simple whitespace split) |

When using `--tokenizer spacy`, you can additionally choose the model size via `--spacy-model`:

| `--spacy-model` | Description |
|------------------|-------------|
| `fr_core_news_lg` (default) | Large model, best accuracy |
| `fr_core_news_md` | Medium model, balanced speed/accuracy |
| `fr_core_news_sm` | Small model, fastest |

**SpaCy tokenizer features:**
- No lemmatization (surface forms preserved, as in the LDA pipeline)
- POS-aware filtering (punctuation, numbers, spaces filtered out)
- Parallelized batch processing via `nlp.pipe()`
- Disabled unnecessary components (`parser`, `ner`) for speed
- Stopword removal with extended French + rap filler words

**NLTK tokenizer features:**
- `word_tokenize` with French language support
- Extended French stopword list (including rap filler words)
- No lemmatization (surface forms only)
- No spaCy dependency required

**SimpleSpace tokenizer features:**
- Fast whitespace-based tokenization with minimal regex for punctuation removal
- No external dependency required
- Best for quick testing runs

```bash
# Install SpaCy model (recommended)
python -m spacy download fr_core_news_lg

# Or use NLTK fallback (no spaCy needed)
python main_comparate_run.py --tokenizer nltk

# Or use simple space tokenizer (fastest, for testing)
python main_comparate_run.py --tokenizer space
```

### PDF Generation Engines

The script supports two PDF generation engines:

| Engine | Description | Requirements | Best For |
|--------|-------------|--------------|----------|
| `latex` (default) | Pure LaTeX with proper math equations | xelatex or pdflatex | Academic reports, proper equation rendering |
| `markdown` | Pypandoc-based conversion | pypandoc, pandoc | Quick reports, when LaTeX not available |

**LaTeX Engine Features:**
- Properly rendered mathematical equations using `\begin{equation}...\end{equation}`
- Professional document structure with TOC, headers, page numbers
- Automatic fallback to markdown if LaTeX compilation fails
- Generates both `.tex` source and `.pdf` output

**Installation for LaTeX:**
```bash
# Ubuntu/Debian
sudo apt install texlive-xetex texlive-latex-extra texlive-fonts-recommended

# macOS (MacTeX)
brew install --cask mactex

# Verify installation
xelatex --version
```

### Examples

```bash
# Standard comparison with default LaTeX PDF
python main_comparate_run.py

# Specify folders explicitly
python main_comparate_run.py \
    --lda-folder results/LDA/run_20260126_124051_bigram_only \
    --bertopic-folder results/BERTopic/run_20260126_130645_camembert \
    --iramuteq-folder results/IRAMUTEQ/evaluation_20260126_124001

# Generate English report
python main_comparate_run.py --lang en

# Use markdown engine (when LaTeX not available)
python main_comparate_run.py --pdf-engine markdown

# Use NLTK tokenizer (no spaCy dependency)
python main_comparate_run.py --tokenizer nltk

# Use small spaCy model (faster)
python main_comparate_run.py --tokenizer spacy --spacy-model fr_core_news_sm

# Full example with all options
python main_comparate_run.py \
    --lda-folder results/LDA/run_YYYYMMDD_HHMMSS_both \
    --bertopic-folder results/BERTopic/run_YYYYMMDD_HHMMSS_camembert \
    --iramuteq-folder results/IRAMUTEQ/evaluation_YYYYMMDD_HHMMSS \
    --lang fr \
    --pdf-engine latex \
    --tokenizer spacy \
    --spacy-model fr_core_news_lg \
    --aggregation-size 20 \
    --output-dir results/comparisons/my_comparison
```

### Comparison Outputs

The comparison script generates results in `results/comparisons/comparison_YYYYMMDD_HHMMSS/`:

| File | Description |
|------|-------------|
| `comparison_report.md` | Full markdown report |
| `comparison_report.pdf` | PDF report (with equations) |
| `comparison_report.tex` | LaTeX source (when using latex engine) |
| `comparison_metrics.json` | All metrics in JSON format |
| `figures/` | All visualization figures |

### Comparison Metrics

The script computes several comparison metrics:

**Q1: Model Agreement**
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Contingency tables

**Q2: Artist Separation**
- Cramér's V for artist-topic association
- Standardized residuals analysis

**Q3: Temporal Dynamics**
- Topic evolution over time
- Jensen-Shannon divergence between periods

**Q4: Vocabulary Overlap**
- Jaccard similarity of topic vocabularies
- Shared vs unique terms

**Q5: Topic Distance Analysis**
- Intra/inter-topic distances in 4 configurations (intra/inter x paired/aggregated)
- Multi-aggregation stabilization curves (distance vs aggregation size)
- Centroid distances (one-vs-rest topic ranking)
- Chi2/n word-topic independence test
- Jensen-Shannon and Labbé distance metrics

### Report Sections

The generated report includes:

1. **Corpus Description** - Dataset statistics and temporal coverage
2. **Individual Model Descriptions** - Parameters and metrics for each model
3. **Comparative Analysis** - Cross-model comparisons (Q1-Q4) and topic distance analysis (Q5: 4 configurations, aggregation curves, centroid ranking, chi2/n)
4. **Summary and Recommendations** - Key findings and dynamic conclusions
5. **References** - Full academic citations
6. **Mathematical Appendix** - Formula definitions and methodological notes

---

## Report Utilities

### Translation System

Reports support French (default) and English. Translations are stored in `utils/comparaison_utils/report/translations.json`.

To add or modify translations:
```json
{
  "fr": {
    "title": "Rapport de Comparaison des Modèles de Topics",
    ...
  },
  "en": {
    "title": "Topic Model Comparison Report",
    ...
  }
}
```

### Programmatic Report Generation

You can generate reports programmatically:

```python
from utils.comparaison_utils import (
    generate_comparison_report,
    generate_pdf_report,
    generate_latex_report,
)

# Generate markdown report
markdown_content = generate_comparison_report(
    results=all_results,
    output_dir="./output",
    figures_dir="./output/figures",
    lang='fr'
)

# Generate PDF with LaTeX engine
success = generate_pdf_report(
    markdown_content=markdown_content,
    output_path="./output/report.pdf",
    lang='fr',
    pdf_engine='latex',  # or 'markdown'
    results=all_results,
    output_dir="./output",
    figures_dir="./output/figures"
)

# Or generate LaTeX directly
latex_content = generate_latex_report(
    results=all_results,
    output_dir="./output",
    figures_dir="./output/figures",
    lang='fr'
)
```

---
