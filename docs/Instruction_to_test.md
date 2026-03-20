# Instructions: Complete Test from Scratch

This guide walks you through running a complete test of the pipeline from scratch: training BERTopic (agglomerative clustering), LDA (unigrams only), evaluating IRAMUTEQ, and generating a comparison report using the Markdown PDF engine (no LaTeX installation required).

---

## Prerequisites

### 1. Clone and enter the project

```bash
cd /path/to/JADT_rap_fr
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv rap_fr
source rap_fr/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the spaCy French model

The comparison script needs a spaCy model for tokenization. We use `fr_core_news_lg` (large model, best accuracy):

```bash
python -m spacy download fr_core_news_lg
```

### 5. Verify the corpus file exists

```bash
ls data/20260123_filter_verses_lrfaf_corpus.csv
```

This CSV must contain at minimum: `lyrics_cleaned`, `artist`, `year`, `title`, `IRAMUTEQ_CLASSES`.

---

## Step 1: Run BERTopic with Agglomerative Clustering

We use CamemBERT embeddings with agglomerative clustering (20 clusters). We disable OpenAI labeling to avoid needing an API key.

```bash
python build_and_evaluate_bertopic.py \
    --embedding camembert \
    --clustering agglomerative \
    --clusters 20 \
    --no-openai
```

**What this does:**
1. Loads the corpus CSV.
2. Loads (or computes) CamemBERT sentence embeddings.
3. Applies UMAP dimensionality reduction (15 neighbors, 5 components).
4. Clusters with agglomerative clustering (Ward linkage, 20 clusters).
5. Fits BERTopic with c-TF-IDF and KeyBERTInspired representations.
6. Computes evaluation metrics (artist separation, temporal evolution).
7. Generates visualizations (topic distribution, heatmaps, UMAP plot, silhouette scores).

**Output:** A timestamped directory under `results/BERTopic/`, for example:
```
results/BERTopic/run_20260127_143012_camembert/
```

Note down this folder path -- you will need it for the comparison step.

---

## Step 2: Run LDA with Unigrams Only

We train an LDA model with 20 topics using only unigrams (no n-gram detection).

```bash
python build_and_evaluate_LDA.py \
    --topics 20 \
    --ngrams unigrams \
    --passes 15
```

**What this does:**
1. Loads the corpus CSV.
2. Tokenizes and filters text (stopword removal, minimum word length).
3. Builds a Gensim dictionary and bag-of-words corpus (unigrams only).
4. Trains LDA with 20 topics, 15 passes, 400 iterations.
5. Computes coherence scores (C_V) per topic.
6. Computes evaluation metrics (artist separation, temporal evolution).
7. Generates visualizations (coherence plot, PCA, pyLDAvis, heatmaps).

**Output:** A timestamped directory under `results/LDA/`, for example:
```
results/LDA/run_20260127_144523_unigrams/
```

Note down this folder path.

---

## Step 3: Evaluate IRAMUTEQ Classification

This evaluates the pre-computed IRAMUTEQ classes stored in the `IRAMUTEQ_CLASSES` column of the dataset.

```bash
python evaluate_iramuteq.py
```

**What this does:**
1. Loads the corpus CSV.
2. Reads the `IRAMUTEQ_CLASSES` column (pre-computed Reinert classification).
3. Computes the same evaluation metrics as LDA and BERTopic.
4. Generates visualizations (class distribution, heatmaps).

**Output:** A timestamped directory under `results/IRAMUTEQ/`, for example:
```
results/IRAMUTEQ/evaluation_20260127_150034/
```

Note down this folder path.

---

## Step 4: Generate the Comparison Report

Now we compare all three models. We use:
- `--tokenizer spacy --spacy-model fr_core_news_lg` for tokenization (installed in prerequisites).
- `--pdf-engine markdown` to avoid needing a LaTeX installation.
- `--lang fr` for a French report.

Replace the folder paths below with the actual paths from steps 1-3:

```bash
python main_comparate_run.py \
    --bertopic-folder results/BERTopic/run_YYYYMMDD_HHMMSS_camembert \
    --lda-folder results/LDA/run_YYYYMMDD_HHMMSS_unigrams \
    --iramuteq-folder results/IRAMUTEQ/evaluation_YYYYMMDD_HHMMSS \
    --tokenizer spacy \
    --spacy-model fr_core_news_lg \
    --pdf-engine markdown \
    --lang fr
```

**What this does:**
1. Validates that all three input folders contain `doc_assignments.csv`.
2. Loads and aligns documents across the three models.
3. Computes five research questions:
   - **Q1:** Model agreement (ARI, NMI, AMI, contingency tables).
   - **Q2:** Artist separation (Cramer's V, standardized residuals).
   - **Q3:** Temporal dynamics (variance, decade JS divergence).
   - **Q4:** Vocabulary overlap (Jaccard similarity, distinctiveness).
   - **Q5:** Topic distances (4 configurations, multi-aggregation, centroid distances, chi2/n).
4. Generates comparison visualizations (contingency heatmaps, temporal comparison, vocabulary comparison).
5. Copies relevant figures from each model's run directory.
6. Generates a Markdown report (`comparison_report.md`).
7. Generates a PDF report (`comparison_report.pdf`) using pypandoc (Markdown engine).
8. Saves all metrics to `metrics.json`.

**Output:** A timestamped directory under `results/comparisons/`, for example:
```
results/comparisons/comparison_20260127_151200/
├── comparison_report.md       # Full Markdown report
├── comparison_report.pdf      # PDF report
├── metrics.json               # All metrics as JSON
├── figures/                   # All visualizations
│   ├── corpus_year_distribution.png
│   ├── corpus_decade_breakdown.png
│   ├── contingency_bertopic_vs_lda.png
│   ├── contingency_bertopic_vs_iramuteq.png
│   ├── contingency_lda_vs_iramuteq.png
│   ├── temporal_comparison.png
│   ├── vocabulary_comparison.png
│   ├── bertopic_topic_distribution.png
│   ├── lda_topic_distribution.png
│   ├── iramuteq_topic_distribution.png
│   └── ...
└── data/                      # Raw data
    ├── aligned_assignments.csv
    ├── contingency_bertopic_vs_lda.csv
    ├── contingency_bertopic_vs_iramuteq.csv
    └── contingency_lda_vs_iramuteq.csv
```

---

## Step 5: Build the Interactive Website

Once all three models and the comparison report are generated, you can build an interactive Dash website to explore the results:

```bash
python build_website.py \
    --lda-folder "$LDA_DIR" \
    --bertopic-folder "$BERTOPIC_DIR" \
    --iramuteq-folder "$IRAMUTEQ_DIR" \
    --comparison-folder "$COMPARISON_DIR" \
    --output-dir website_output
```

This generates a self-contained `website_output/` directory with a pre-processed data bundle (`site_data.json`) and a multi-page Dash app. It includes individual model exploration pages (topic keywords, artist profiles, temporal evolution), cross-model comparisons (Sankey diagrams, contingency heatmaps, distance curves), and a FR/EN language toggle.

To run the website locally:

```bash
cd website_output
pip install -r requirements.txt
python app.py
# Open http://localhost:8050
```

Or with Docker:

```bash
cd website_output
docker compose up --build
```

---

## Complete Script (copy-paste)

Here is the entire sequence as a single script you can copy and run. Replace `YYYYMMDD_HHMMSS` placeholders with actual timestamps after each step, or use the version below that captures them automatically:

```bash
#!/bin/bash
set -e

# Activate virtual environment
source venv/bin/activate

echo "=== Step 1: BERTopic (agglomerative, camembert) ==="
python build_and_evaluate_bertopic.py \
    --embedding camembert \
    --clustering agglomerative \
    --clusters 20 \
    --no-openai

# Find the latest BERTopic run
BERTOPIC_DIR=$(ls -td results/BERTopic/run_*_camembert 2>/dev/null | head -1)
echo "BERTopic output: $BERTOPIC_DIR"

echo "=== Step 2: LDA (unigrams, 20 topics) ==="
python build_and_evaluate_LDA.py \
    --topics 20 \
    --ngrams unigrams \
    --passes 15

# Find the latest LDA run
LDA_DIR=$(ls -td results/LDA/run_*_unigrams 2>/dev/null | head -1)
echo "LDA output: $LDA_DIR"

echo "=== Step 3: IRAMUTEQ evaluation ==="
python evaluate_iramuteq.py

# Find the latest IRAMUTEQ evaluation
IRAMUTEQ_DIR=$(ls -td results/IRAMUTEQ/evaluation_* 2>/dev/null | head -1)
echo "IRAMUTEQ output: $IRAMUTEQ_DIR"

echo "=== Step 4: Comparison report ==="
python main_comparate_run.py \
    --bertopic-folder "$BERTOPIC_DIR" \
    --lda-folder "$LDA_DIR" \
    --iramuteq-folder "$IRAMUTEQ_DIR" \
    --tokenizer spacy \
    --spacy-model fr_core_news_lg \
    --pdf-engine markdown \
    --lang fr

# Find the latest comparison
COMPARISON_DIR=$(ls -td results/comparisons/comparison_* 2>/dev/null | head -1)
echo ""
echo "=== Step 5: Interactive website ==="
python build_website.py \
    --lda-folder "$LDA_DIR" \
    --bertopic-folder "$BERTOPIC_DIR" \
    --iramuteq-folder "$IRAMUTEQ_DIR" \
    --comparison-folder "$COMPARISON_DIR" \
    --output-dir website_output

echo ""
echo "=== Done! ==="
echo "Comparison report: $COMPARISON_DIR/comparison_report.md"
echo "PDF report:        $COMPARISON_DIR/comparison_report.pdf"
echo "Metrics:           $COMPARISON_DIR/metrics.json"
echo "Website:           cd website_output && python app.py"
```

Save this as `run_test.sh` and execute with:

```bash
chmod +x run_test.sh
./run_test.sh
```

---

## Troubleshooting

### "IRAMUTEQ_CLASSES column not found"

Your corpus CSV does not have the `IRAMUTEQ_CLASSES` column. This column must contain pre-computed IRAMUTEQ class assignments (integers). Ensure the data file includes this column before running `evaluate_iramuteq.py`.

### "Embeddings don't match documents"

If you change the dataset or use `--sample`, embeddings are auto-computed. For full runs, pre-computed embeddings are loaded from `models/embeddings/`. If they are stale, delete them and re-run with `--compute-embeddings`.

### PDF generation fails

If `--pdf-engine markdown` fails, ensure pypandoc and a LaTeX distribution are installed:

```bash
pip install pypandoc
sudo apt install texlive-xetex texlive-fonts-recommended  # Ubuntu/Debian
```

If you cannot install LaTeX at all, the Markdown report (`comparison_report.md`) is still fully usable and can be opened in any text editor or rendered by GitHub/GitLab.

### spaCy model not found

```bash
python -m spacy download fr_core_news_lg
```

Or use NLTK tokenization instead (no spaCy needed):

```bash
python main_comparate_run.py ... --tokenizer nltk
```

### Memory issues with BERTopic

For large corpora, CamemBERT embeddings and agglomerative clustering require significant RAM (~80 GB estimated). Test with a sample first:

```bash
python build_and_evaluate_bertopic.py --sample 2000 --clustering agglomerative --clusters 20 --no-openai
```
