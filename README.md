# JADT_rap_fr

Analyse thematique d'un corpus de rap francais (Thematic analysis of a French rap corpus)

## Project Structure

```
JADT_rap_fr/
├── data/                          # Corpus data
├── models/                        # Saved models and embeddings
├── results/                       # Evaluation results
│   ├── LDA/                       # LDA results
│   ├── BERTopic/                  # BERTopic results
│   └── IRAMUTEQ/                  # IRAMUTEQ results
├── logs/                          # Batch run logs
├── utils/
│   ├── utils_evaluation.py        # Shared evaluation utilities
│   └── utils_visualization_html.py # Interactive HTML visualization
├── docs/
│   └── TOPIC_MODELING_SCRIPTS.md  # Detailed documentation
├── build_and_evaluate_LDA.py      # LDA topic modeling
├── build_and_evaluate_bertopic.py # BERTopic topic modeling
├── evaluate_iramuteq.py           # IRAMUTEQ evaluation
└── main_run_multiple_evaluation.py # Batch experiment runner
```

## Topic Modeling Scripts

This project includes three topic modeling/clustering approaches for analyzing French rap lyrics:

| Method | Script | Description |
|--------|--------|-------------|
| **LDA** | `build_and_evaluate_LDA.py` | Latent Dirichlet Allocation with n-gram support |
| **BERTopic** | `build_and_evaluate_bertopic.py` | Neural topic modeling with sentence embeddings |
| **IRAMUTEQ** | `evaluate_iramuteq.py` | Evaluation of pre-computed Reinert classification |

**For detailed documentation, see: [docs/TOPIC_MODELING_SCRIPTS.md](docs/TOPIC_MODELING_SCRIPTS.md)**

### Quick Start

```bash
# Install dependencies
pip install gensim scikit-learn bertopic sentence-transformers umap-learn hdbscan
pip install matplotlib seaborn pandas numpy scipy pyLDAvis python-dotenv

# Run LDA (20 topics, with bigrams and trigrams)
python build_and_evaluate_LDA.py --topics 20 --ngrams both

# Run BERTopic with French CamemBERT embeddings
python build_and_evaluate_bertopic.py --embedding camembert --compute-embeddings

# Evaluate IRAMUTEQ classification
python evaluate_iramuteq.py
```

### Available Parameters

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
  --clusters N          Number of clusters (default: 20)
  --umap-neighbors N    UMAP n_neighbors (default: 15)
  --sample N            Sample size for testing
  --no-openai           Disable OpenAI topic labeling
  --no-keybert          Disable KeyBERTInspired representation (enabled by default)
  --no-interactive-html Disable interactive HTML visualization (enabled by default)
```

#### IRAMUTEQ

```bash
python evaluate_iramuteq.py [OPTIONS]

Options:
  --sample N            Sample size for testing
  --min-docs-artist N   Min documents per artist (default: 10)
```

### Batch Experiments

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

### Output Files

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
| `interactive_bertopic.html` | Interactive topic explorer (BERTopic only, enabled by default) |

### Evaluation Metrics

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

## Data

The corpus should be a CSV file with the following columns:
- `lyrics_cleaned`: Preprocessed text
- `artist`: Artist name
- `year`: Year of release
- `title`: Song title
- `IRAMUTEQ_CLASSES`: (Optional) Pre-computed IRAMUTEQ classification

## License

[Add license information]

## Citation

[Add citation information for JADT conference]
