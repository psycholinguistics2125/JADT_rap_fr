#!/usr/bin/env python3
"""
Build Website Data Pipeline
============================
CLI script that loads all model data (LDA, BERTopic, IRAMUTEQ) and comparison
results, then generates a unified site_data.json for the Dash website.

Usage:
    python build_website.py \
        --lda-folder results/LDA/run_20260224_094852_bigram_only \
        --bertopic-folder results/BERTopic/run_20260126_141647_mpnet \
        --iramuteq-folder results/IRAMUTEQ/evaluation_20260126_124001 \
        --comparison-folder results/comparisons/comparison_20260318_175850
"""

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(obj):
    """
    Recursively convert numpy / pandas types to JSON-safe Python builtins.
    NaN and Inf become None.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _df_to_records(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of dicts with NaN replaced by None."""
    records = df.where(df.notna(), other=None).to_dict(orient="records")
    return _sanitize(records)


def _validate_folder(path: Path, label: str, required_files: list[str]):
    """Raise if the folder does not exist or lacks required files."""
    if not path.is_dir():
        print(f"ERROR: {label} folder does not exist: {path}", file=sys.stderr)
        sys.exit(1)
    missing = [f for f in required_files if not (path / f).exists()]
    if missing:
        print(
            f"WARNING: {label} folder {path} is missing: {', '.join(missing)}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------------

def load_corpus_stats(csv_path: Path) -> dict:
    """Load corpus CSV and compute summary statistics."""
    print(f"  Loading corpus from {csv_path} ...")
    df = pd.read_csv(csv_path)
    years = df["year"].dropna().astype(int)
    year_counts = Counter(years)
    decade_counts = Counter((y // 10) * 10 for y in years)

    return _sanitize({
        "n_documents": len(df),
        "n_artists": df["artist"].nunique(),
        "year_range": [int(years.min()), int(years.max())],
        "year_distribution": dict(sorted(year_counts.items())),
        "decade_distribution": {
            str(k): v for k, v in sorted(decade_counts.items())
        },
    })


def load_corpus_samples(csv_path: Path, n=200, excerpt_len=200) -> list:
    """Load a random sample of corpus documents for the 'Explore Corpus' section."""
    df = pd.read_csv(csv_path)
    sample = df.sample(n=min(n, len(df)), random_state=42)
    records = []
    for _, row in sample.iterrows():
        lyrics = str(row.get("lyrics", row.get("lyrics_cleaned", "")))
        records.append({
            "artist": row.get("artist", ""),
            "title": row.get("title", ""),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "excerpt": lyrics[:excerpt_len] + ("..." if len(lyrics) > excerpt_len else ""),
        })
    return _sanitize(records)


def load_all_docs(csv_path: Path, max_per_artist: int = 100, excerpt_len: int = 200) -> list:
    """Load up to *max_per_artist* docs per artist with lyrics excerpts."""
    df = pd.read_csv(csv_path)
    sampled = pd.concat(
        grp.sample(n=min(max_per_artist, len(grp)), random_state=42)
        for _, grp in df.groupby("artist")
    )
    records = []
    for _, row in sampled.iterrows():
        lyrics = str(row.get("lyrics", row.get("lyrics_cleaned", "")))
        records.append({
            "artist": row.get("artist", ""),
            "title": row.get("title", ""),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "excerpt": lyrics[:excerpt_len] + ("..." if len(lyrics) > excerpt_len else ""),
        })
    return _sanitize(records)


# ---------------------------------------------------------------------------
# Topic label generation
# ---------------------------------------------------------------------------

def _bertopic_label(topic_id: str, topic_data: dict) -> str:
    """
    BERTopic: 'T{id}: {openai_label}' (fallback: top 5 keybert words).
    """
    openai = topic_data.get("openai", [])
    if openai and openai[0]:
        label_text = openai[0].strip().strip('"')
        return f"T{topic_id}: {label_text}"
    keybert = topic_data.get("keybert", [])
    if keybert:
        return f"T{topic_id}: {', '.join(keybert[:5])}"
    top_words = topic_data.get("top_words", "")
    if isinstance(top_words, str) and top_words:
        words = [w.strip() for w in top_words.split(",")]
        return f"T{topic_id}: {', '.join(words[:5])}"
    return f"T{topic_id}"


def _lda_label(topic_id: str, topic_data: dict) -> str:
    """
    LDA: 'T{id}: {top5_words}' from words list.
    """
    words = topic_data.get("words", [])
    if words:
        return f"T{topic_id}: {', '.join(words[:5])}"
    top_words = topic_data.get("top_words", "")
    if isinstance(top_words, str) and top_words:
        word_list = [w.strip() for w in top_words.split(",")]
        return f"T{topic_id}: {', '.join(word_list[:5])}"
    return f"T{topic_id}"


def _iramuteq_label(topic_id: str, topic_data: dict) -> str:
    """
    IRAMUTEQ: 'C{id}: {top5_words}'.
    """
    words = topic_data.get("words", [])
    if words:
        return f"C{topic_id}: {', '.join(words[:5])}"
    top_words = topic_data.get("top_words", "")
    if isinstance(top_words, str) and top_words:
        word_list = [w.strip() for w in top_words.split(",")]
        return f"C{topic_id}: {', '.join(word_list[:5])}"
    return f"C{topic_id}"


# ---------------------------------------------------------------------------
# Per-model data loading
# ---------------------------------------------------------------------------

def load_model_data(folder: Path, model_type: str) -> dict:
    """
    Load a single model's data from its run folder.

    Returns a dict with: params, metrics, topics, topic_labels,
    topic_evolution, artist_metrics, top_artists, annual_js, doc_samples.
    """
    print(f"  Loading {model_type} data from {folder} ...")

    # ---- metrics.json ----
    metrics_path = folder / "metrics.json"
    metrics_raw = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_raw = json.load(f)

    params = metrics_raw.get("parameters", {})
    # Flatten relevant metric blocks
    metrics = {}
    for key in ("coherence_metrics", "silhouette_metrics", "cluster_metrics",
                "artist_metrics", "temporal_metrics"):
        if key in metrics_raw:
            metrics[key] = metrics_raw[key]

    # ---- topics.json ----
    topics_path = folder / "topics.json"
    topics_raw = {}
    if topics_path.exists():
        with open(topics_path, "r", encoding="utf-8") as f:
            topics_raw = json.load(f)

    # For IRAMUTEQ without topics.json, build empty placeholders
    if model_type == "iramuteq" and not topics_raw:
        class_dist_path = folder / "class_distribution.json"
        if class_dist_path.exists():
            with open(class_dist_path, "r", encoding="utf-8") as f:
                class_dist = json.load(f)
            class_ids = class_dist.get("class_ids", list(range(1, 21)))
        else:
            class_ids = list(range(1, 21))
        topics_raw = {
            str(cid): {"words": [], "top_words": ""} for cid in class_ids
        }

    # ---- Generate labels ----
    label_fn = {
        "bertopic": _bertopic_label,
        "lda": _lda_label,
        "iramuteq": _iramuteq_label,
    }[model_type]

    topic_labels = {tid: label_fn(tid, tdata) for tid, tdata in topics_raw.items()}

    # ---- topic_evolution.csv ----
    evolution_path = folder / "topic_evolution.csv"
    topic_evolution = []
    if evolution_path.exists():
        evo_df = pd.read_csv(evolution_path, index_col=0)
        evo_df.index.name = "year"
        evo_df = evo_df.reset_index()
        topic_evolution = _df_to_records(evo_df)

    # ---- artist_topic_metrics.csv ----
    artist_metrics_list = []
    artist_metrics_path = folder / "artist_topic_metrics.csv"
    if artist_metrics_path.exists():
        am_df = pd.read_csv(artist_metrics_path)
        artist_metrics_list = _df_to_records(am_df)

    # ---- topic_top_artists.csv ----
    top_artists = []
    top_artists_path = folder / "topic_top_artists.csv"
    if top_artists_path.exists():
        ta_df = pd.read_csv(top_artists_path)
        top_artists = _df_to_records(ta_df)

    # ---- annual / biannual JS divergence ----
    annual_js = []
    for fname in ("annual_js_divergence.csv", "biannual_js_divergence.csv"):
        js_path = folder / fname
        if js_path.exists():
            js_df = pd.read_csv(js_path)
            annual_js = _df_to_records(js_df)
            break

    # ---- doc_assignments.csv  ->  doc_samples (20 per topic) ----
    doc_samples = {}
    doc_assign_path = folder / "doc_assignments.csv"
    if doc_assign_path.exists():
        da_df = pd.read_csv(doc_assign_path)
        # Normalise the topic column name
        topic_col = _detect_topic_col(da_df, model_type)
        prob_col = _detect_prob_col(da_df, model_type)

        if topic_col:
            for topic_val, grp in da_df.groupby(topic_col):
                if prob_col and prob_col in grp.columns:
                    grp = grp.sort_values(prob_col, ascending=False)
                sampled = grp.head(20)
                doc_samples[str(int(topic_val))] = _df_to_records(sampled)

    # ---- Friendly model name ----
    name_map = {"bertopic": "BERTopic", "lda": "LDA", "iramuteq": "IRAMUTEQ"}

    return _sanitize({
        "name": name_map[model_type],
        "params": params,
        "metrics": metrics,
        "topics": topics_raw,
        "topic_labels": topic_labels,
        "topic_evolution": topic_evolution,
        "artist_metrics": artist_metrics_list,
        "top_artists": top_artists,
        "annual_js": annual_js,
        "doc_samples": doc_samples,
    })


def _detect_topic_col(df: pd.DataFrame, model_type: str) -> str | None:
    """Find the dominant-topic column in a doc_assignments DataFrame."""
    candidates = {
        "bertopic": ["topic", "dominant_topic"],
        "lda": ["dominant_topic", "topic"],
        "iramuteq": ["iramuteq_class", "topic", "dominant_topic"],
    }
    for col in candidates.get(model_type, []):
        if col in df.columns:
            return col
    return None


def _detect_prob_col(df: pd.DataFrame, model_type: str) -> str | None:
    """Find the probability / confidence column."""
    for col in ("dominant_topic_prob", "probability", "prob", "confidence"):
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Comparison data
# ---------------------------------------------------------------------------

def load_comparison_data(folder: Path) -> dict:
    """Load comparison results including metrics, contingency tables, residuals, and Sankey data."""
    print(f"  Loading comparison data from {folder} ...")

    # ---- metrics.json ----
    metrics_path = folder / "metrics.json"
    if not metrics_path.exists():
        print("  WARNING: comparison metrics.json not found", file=sys.stderr)
        return {}

    with open(metrics_path, "r", encoding="utf-8") as f:
        cmp_raw = json.load(f)

    result = {
        "agreement": cmp_raw.get("agreement", {}),
        "artist_separation": cmp_raw.get("artist_separation", {}),
        "temporal": cmp_raw.get("temporal", {}),
        "vocabulary": cmp_raw.get("vocabulary", {}),
        "distances": {
            "intra_topic_distances": cmp_raw.get("intra_topic_distances", {}),
            "topic_distances": cmp_raw.get("topic_distances", {}),
            "inter_topic_per_topic": cmp_raw.get("inter_topic_per_topic", {}),
        },
        "multi_aggregation": cmp_raw.get("multi_aggregation", {}),
        "chi2_results": cmp_raw.get("chi2_results", {}),
    }

    data_dir = folder / "data"

    # ---- Contingency tables ----
    contingency = {}
    contingency_labels = {}
    pairs = [
        ("bertopic_vs_lda", "bertopic_vs_lda"),
        ("bertopic_vs_iramuteq", "bertopic_vs_iramuteq"),
        ("lda_vs_iramuteq", "lda_vs_iramuteq"),
    ]
    for pair_key, file_suffix in pairs:
        ct_path = data_dir / f"contingency_{file_suffix}.csv"
        if ct_path.exists():
            ct_df = pd.read_csv(ct_path, index_col=0)
            contingency[pair_key] = ct_df.values.tolist()
            contingency_labels[pair_key] = {
                "rows": list(ct_df.index.astype(str)),
                "cols": list(ct_df.columns.astype(str)),
            }
    result["contingency"] = contingency
    result["contingency_labels"] = contingency_labels

    # ---- Residuals ----
    residuals = {}
    residuals_labels = {}
    for model_name in ("bertopic", "lda", "iramuteq"):
        res_path = data_dir / f"residuals_{model_name}.csv"
        if res_path.exists():
            res_df = pd.read_csv(res_path, index_col=0)
            residuals[model_name] = res_df.values.tolist()
            residuals_labels[model_name] = {
                "rows": list(res_df.index.astype(str)),
                "cols": list(res_df.columns.astype(str)),
            }
    result["residuals"] = residuals
    result["residuals_labels"] = residuals_labels

    # ---- Sankey data from aligned_assignments.csv ----
    aligned_path = data_dir / "aligned_assignments.csv"
    if aligned_path.exists():
        result["sankey"] = _build_sankey_data(aligned_path)
    else:
        result["sankey"] = {}

    return _sanitize(result)


def _build_sankey_data(aligned_path: Path) -> dict:
    """
    Compute Sankey diagram data (source, target, value, labels) for each
    pair of models from the aligned_assignments CSV.

    Columns expected: artist, title, year, bertopic_topic, lda_topic, iramuteq_topic
    """
    df = pd.read_csv(aligned_path)

    model_pairs = [
        ("bertopic_vs_lda", "bertopic_topic", "lda_topic"),
        ("bertopic_vs_iramuteq", "bertopic_topic", "iramuteq_topic"),
        ("lda_vs_iramuteq", "lda_topic", "iramuteq_topic"),
    ]

    sankey = {}
    for pair_key, col_a, col_b in model_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue

        sub = df[[col_a, col_b]].dropna()
        ct = pd.crosstab(sub[col_a], sub[col_b])

        source_labels = [str(s) for s in ct.index]
        target_labels = [str(t) for t in ct.columns]

        source_list = []
        target_list = []
        value_list = []

        n_sources = len(source_labels)

        for i, src in enumerate(ct.index):
            for j, tgt in enumerate(ct.columns):
                val = int(ct.loc[src, tgt])
                if val > 0:
                    source_list.append(i)
                    target_list.append(n_sources + j)
                    value_list.append(val)

        sankey[pair_key] = {
            "source": source_list,
            "target": target_list,
            "value": value_list,
            "source_labels": source_labels,
            "target_labels": target_labels,
        }

    return sankey


# ---------------------------------------------------------------------------
# Output directory setup  --  copy template files and HTML artefacts
# ---------------------------------------------------------------------------

def setup_output_dir(output_dir: Path,
                     bertopic_folder: Path,
                     lda_folder: Path,
                     iramuteq_folder: Path):
    """
    Create the output directory structure, copy template files and HTML artefacts.
    """
    # Copy entire website_template into output_dir
    template_dir = PROJECT_ROOT / "website_template"
    if template_dir.is_dir():
        for item in template_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(template_dir)
                dest = output_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
        print(f"  Copied website template files from {template_dir}")
    else:
        # Ensure directories exist even without template
        for subdir in ("pages", "components", "assets", "data"):
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Ensure data/ directory exists for JSON and HTML artefacts
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    # Move site_data.json into data/ if it's at the root
    root_json = output_dir / "site_data.json"
    data_json = output_dir / "data" / "site_data.json"
    if root_json.exists() and not data_json.exists():
        shutil.move(str(root_json), str(data_json))

    # Copy interactive HTML files from model folders into assets/ (Dash auto-serves)
    (output_dir / "assets").mkdir(parents=True, exist_ok=True)
    html_files = {
        "pyldavis.html": lda_folder / "pyldavis.html",
        "interactive_bertopic.html": bertopic_folder / "interactive_bertopic.html",
    }
    for dest_name, src_path in html_files.items():
        if src_path.exists():
            dest = output_dir / "assets" / dest_name
            shutil.copy2(src_path, dest)
            print(f"  Copied {src_path.name} -> {dest}")

    # Copy IRAMUTEQ static images
    iramuteq_images = {
        "dendrogramme_iramuteq.png": PROJECT_ROOT / "data" / "dendrogramme_1.png",
        "chrono_prop_iramuteq.png": PROJECT_ROOT / "data" / "chrono_prop.png",
    }
    for dest_name, src_path in iramuteq_images.items():
        if src_path.exists():
            dest = output_dir / "assets" / dest_name
            shutil.copy2(src_path, dest)
            print(f"  Copied {src_path.name} -> {dest}")


def copy_comparison_html(output_dir: Path, comparison_folder: Path):
    """Copy any Sankey or other HTML files from the comparison figures directory."""
    figures_dir = comparison_folder / "figures"
    if not figures_dir.is_dir():
        return
    for html_file in figures_dir.glob("*.html"):
        dest = output_dir / "data" / html_file.name
        shutil.copy2(html_file, dest)
        print(f"  Copied comparison HTML: {html_file.name} -> {dest}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load all model data and generate site_data.json for the Dash website.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python build_website.py \\
      --lda-folder results/LDA/run_20260224_094852_bigram_only \\
      --bertopic-folder results/BERTopic/run_20260126_141647_mpnet \\
      --iramuteq-folder results/IRAMUTEQ/evaluation_20260126_124001 \\
      --comparison-folder results/comparisons/comparison_20260318_175850
""",
    )
    parser.add_argument(
        "--lda-folder", type=str, required=True,
        help="Path to LDA run folder",
    )
    parser.add_argument(
        "--bertopic-folder", type=str, required=True,
        help="Path to BERTopic run folder",
    )
    parser.add_argument(
        "--iramuteq-folder", type=str, required=True,
        help="Path to IRAMUTEQ evaluation folder",
    )
    parser.add_argument(
        "--comparison-folder", type=str, required=True,
        help="Path to comparison results folder",
    )
    parser.add_argument(
        "--corpus-csv", type=str,
        default=str(PROJECT_ROOT / "data" / "20260123_filter_verses_lrfaf_corpus.csv"),
        help="Path to corpus CSV (default: data/20260123_filter_verses_lrfaf_corpus.csv)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="website_output",
        help="Output directory (default: website_output)",
    )
    parser.add_argument(
        "--lang", type=str, choices=["fr", "en"], default="fr",
        help="Default language fr/en (default: fr)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve all paths relative to PROJECT_ROOT if not absolute
    def resolve(p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    lda_folder = resolve(args.lda_folder)
    bertopic_folder = resolve(args.bertopic_folder)
    iramuteq_folder = resolve(args.iramuteq_folder)
    comparison_folder = resolve(args.comparison_folder)
    corpus_csv = resolve(args.corpus_csv)
    output_dir = resolve(args.output_dir)

    print("=" * 60)
    print("BUILD WEBSITE DATA PIPELINE")
    print("=" * 60)
    print(f"  LDA folder:        {lda_folder}")
    print(f"  BERTopic folder:   {bertopic_folder}")
    print(f"  IRAMUTEQ folder:   {iramuteq_folder}")
    print(f"  Comparison folder: {comparison_folder}")
    print(f"  Corpus CSV:        {corpus_csv}")
    print(f"  Output directory:  {output_dir}")
    print(f"  Language:          {args.lang}")
    print()

    # ---- 1. Validate folders ----
    print("[1/7] Validating input folders ...")
    _validate_folder(lda_folder, "LDA", [
        "metrics.json", "topics.json", "topic_evolution.csv",
        "artist_topic_metrics.csv", "topic_top_artists.csv", "doc_assignments.csv",
    ])
    _validate_folder(bertopic_folder, "BERTopic", [
        "metrics.json", "topics.json", "topic_evolution.csv",
        "artist_topic_metrics.csv", "topic_top_artists.csv", "doc_assignments.csv",
    ])
    _validate_folder(iramuteq_folder, "IRAMUTEQ", [
        "metrics.json", "topic_evolution.csv",
        "artist_topic_metrics.csv", "topic_top_artists.csv", "doc_assignments.csv",
    ])
    _validate_folder(comparison_folder, "Comparison", ["metrics.json"])
    if not corpus_csv.is_file():
        print(f"ERROR: Corpus CSV not found: {corpus_csv}", file=sys.stderr)
        sys.exit(1)
    print("  All folders validated.")

    # ---- 2. Load corpus stats + samples ----
    print("\n[2/7] Loading corpus statistics ...")
    corpus_stats = load_corpus_stats(corpus_csv)
    corpus_stats["doc_samples"] = load_corpus_samples(corpus_csv)
    corpus_stats["all_docs"] = load_all_docs(corpus_csv)
    print(f"  {corpus_stats['n_documents']} documents, "
          f"{corpus_stats['n_artists']} artists, "
          f"years {corpus_stats['year_range'][0]}-{corpus_stats['year_range'][1]}, "
          f"{len(corpus_stats['doc_samples'])} sample docs, "
          f"{len(corpus_stats['all_docs'])} all docs")

    # ---- 3. Load each model's data ----
    print("\n[3/7] Loading model data ...")
    models = {
        "bertopic": load_model_data(bertopic_folder, "bertopic"),
        "lda": load_model_data(lda_folder, "lda"),
        "iramuteq": load_model_data(iramuteq_folder, "iramuteq"),
    }

    for name, mdata in models.items():
        n_topics = len(mdata.get("topics", {}))
        n_evo = len(mdata.get("topic_evolution", []))
        n_artist = len(mdata.get("artist_metrics", []))
        n_samples = sum(len(v) for v in mdata.get("doc_samples", {}).values())
        print(f"  {name}: {n_topics} topics, {n_evo} evolution rows, "
              f"{n_artist} artist profiles, {n_samples} doc samples")

    # ---- 4. Load comparison data ----
    print("\n[4/7] Loading comparison data ...")
    comparison = load_comparison_data(comparison_folder)
    n_sankey = len(comparison.get("sankey", {}))
    n_contingency = len(comparison.get("contingency", {}))
    print(f"  {n_contingency} contingency tables, {n_sankey} Sankey diagrams")

    # ---- 5. Assemble and write site_data.json ----
    print("\n[5/7] Assembling site_data.json ...")
    site_data = {
        "corpus": corpus_stats,
        "models": models,
        "comparison": comparison,
        "default_lang": args.lang,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data" / "site_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(site_data, f, indent=2, ensure_ascii=False)

    json_size_mb = json_path.stat().st_size / (1024 * 1024)
    print(f"  Written {json_path}  ({json_size_mb:.1f} MB)")

    # ---- 6. Set up output directory and copy artefacts ----
    print("\n[6/7] Setting up output directory ...")
    setup_output_dir(output_dir, bertopic_folder, lda_folder, iramuteq_folder)
    copy_comparison_html(output_dir, comparison_folder)
    print("  Output directory ready.")

    # ---- Done ----
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nFiles written to: {output_dir}/")
    print(f"  site_data.json   ({json_size_mb:.1f} MB)")
    print()
    print("To run the Dash app:")
    print(f"  cd {output_dir}")
    print("  python app.py")
    print()


if __name__ == "__main__":
    main()
