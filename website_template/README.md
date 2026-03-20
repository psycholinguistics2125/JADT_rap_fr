# Topic Modeling Comparison — French Rap (JADT 2026)

Interactive website for exploring topic modeling results comparing BERTopic, LDA, and IRAMUTEQ on 115,805 French rap verses from 605 artists (1992-2021).

## Quick Start

### Run locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8050
```

### Run with Docker

```bash
docker compose up --build
# Open http://localhost:8050
```

## Structure

| Path | Description |
|------|-------------|
| `app.py` | Dash entry point |
| `data/site_data.json` | Pre-processed model + comparison data |
| `pages/home.py` | Home page: corpus summary + model overview |
| `pages/model_*.py` | Individual model exploration (Topics, Artists, Temporal) |
| `pages/compare_*.py` | Cross-model comparisons (Agreement, Artists, Temporal, Vocabulary, Distances) |
| `components/` | Shared components (navbar, cards, colors, translations) |
| `assets/style.css` | Custom styling |

## Bilingual Support

Toggle FR/EN using the button in the navigation bar. Language preference is stored in the browser.

## Building from Source

From the project root:

```bash
python build_website.py \
  --lda-folder results/LDA/run_20260224_094852_bigram_only \
  --bertopic-folder results/BERTopic/run_20260126_141647_mpnet \
  --iramuteq-folder results/IRAMUTEQ/evaluation_20260126_124001 \
  --comparison-folder results/comparisons/comparison_20260318_175850 \
  --output-dir website_output
```
