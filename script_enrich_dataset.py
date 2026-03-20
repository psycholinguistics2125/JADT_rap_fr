"""
Enrich the main corpus CSV with class/topic assignments from model results.

Usage:
    python enrich_dataset.py <model_folder> [<model_folder> ...]

Examples:
    # Add a single model's assignments
    python enrich_dataset.py LDA/run_20260224_094852_bigram_only

    # Add multiple models at once
    python enrich_dataset.py LDA/run_20260224_094852_bigram_only BERTopic/run_20260126_141647_mpnet

    # Use just the run folder name (auto-detected under results/)
    python enrich_dataset.py run_20260224_094852_bigram_only

    # Optionally specify a custom column name
    python enrich_dataset.py LDA/run_20260224_094852_bigram_only --column-name LDA_bigram

The script reads doc_assignments.csv from each model folder and adds
a column to the main dataset with the class/topic assignments.

Exemple : python enrich_dataset.py LDA/run_20260224_094852_bigram_only LDA/run_20260224_095305_trigram_only LDA/run_20260224_100615_both LDA/run_20260224_100155_ngrams_only BERTopic/run_20260126_130645_camembert BERTopic/run_20260126_133046_e5 BERTopic/run_20260126_141647_mpnet IRAMUTEQ/evaluation_20260126_124001 -o data/20260225_enrich_dataset.csv

"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "20260123_filter_verses_lrfaf_corpus.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# Mapping of model type -> column name in doc_assignments.csv that holds the class
CLASS_COLUMNS = {
    "LDA": "dominant_topic",
    "BERTopic": "topic",
    "IRAMUTEQ": "iramuteq_class",
}


def find_model_folder(folder_arg: str) -> Path:
    """Resolve a folder argument to a full path under results/.

    Accepts either:
      - A relative path like 'LDA/run_20260224_094852_bigram_only'
      - Just the run folder name like 'run_20260224_094852_bigram_only'
    """
    # Try as a direct relative path under results/
    candidate = RESULTS_DIR / folder_arg
    if candidate.is_dir():
        return candidate

    # Search recursively for a matching folder name
    for path in RESULTS_DIR.rglob(folder_arg):
        if path.is_dir():
            return path

    # Try partial match on folder name
    matches = list(RESULTS_DIR.rglob(f"*{folder_arg}*"))
    dirs = [m for m in matches if m.is_dir()]
    if len(dirs) == 1:
        return dirs[0]
    elif len(dirs) > 1:
        print(f"Error: Ambiguous folder name '{folder_arg}'. Multiple matches found:")
        for d in dirs:
            print(f"  - {d.relative_to(RESULTS_DIR)}")
        sys.exit(1)

    print(f"Error: Could not find folder '{folder_arg}' under {RESULTS_DIR}")
    sys.exit(1)


def detect_class_column(model_folder: Path) -> str:
    """Detect which column holds the class assignment based on the model type."""
    # Determine model type from parent folder name
    relative = model_folder.relative_to(RESULTS_DIR)
    model_type = relative.parts[0]  # e.g., 'LDA', 'BERTopic', 'IRAMUTEQ'

    if model_type in CLASS_COLUMNS:
        return CLASS_COLUMNS[model_type]

    # Fallback: read the CSV header and pick the likely class column
    assignments_path = model_folder / "doc_assignments.csv"
    if assignments_path.exists():
        df = pd.read_csv(assignments_path, nrows=0)
        for col in ["dominant_topic", "topic", "iramuteq_class", "class", "cluster"]:
            if col in df.columns:
                return col

    print(f"Error: Cannot detect class column for model type '{model_type}'.")
    print(f"  Known types: {', '.join(CLASS_COLUMNS.keys())}")
    sys.exit(1)


def derive_column_name(model_folder: Path) -> str:
    """Derive a column name from the folder path.

    Examples:
        LDA/run_20260224_094852_bigram_only -> LDA_bigram_only
        BERTopic/run_20260126_141647_mpnet  -> BERTopic_mpnet
        IRAMUTEQ/evaluation_20260126_124001 -> IRAMUTEQ_evaluation
    """
    relative = model_folder.relative_to(RESULTS_DIR)
    model_type = relative.parts[0]
    run_name = relative.parts[-1]

    # Extract the suffix after the timestamp (e.g., 'bigram_only' from 'run_20260224_094852_bigram_only')
    parts = run_name.split("_")
    # Find timestamp pattern: YYYYMMDD_HHMMSS (positions after 'run' or 'evaluation')
    suffix_parts = []
    skip_count = 0
    for i, part in enumerate(parts):
        if skip_count > 0:
            skip_count -= 1
            continue
        # Skip prefix like 'run' or 'evaluation' and timestamp digits
        if i == 0 and part in ("run", "evaluation"):
            continue
        if part.isdigit() and len(part) in (6, 8, 14):
            continue
        suffix_parts.append(part)

    suffix = "_".join(suffix_parts) if suffix_parts else run_name
    return f"{model_type}_{suffix}"


def enrich(model_folder: Path, column_name: str, df_main: pd.DataFrame) -> pd.DataFrame:
    """Add class assignments from a model to the main dataframe."""
    assignments_path = model_folder / "doc_assignments.csv"
    if not assignments_path.exists():
        print(f"Error: No doc_assignments.csv found in {model_folder}")
        sys.exit(1)

    class_col = detect_class_column(model_folder)
    df_assign = pd.read_csv(assignments_path)

    if class_col not in df_assign.columns:
        print(f"Error: Column '{class_col}' not found in {assignments_path}")
        print(f"  Available columns: {', '.join(df_assign.columns)}")
        sys.exit(1)

    classes = df_assign[class_col].values

    if len(classes) != len(df_main):
        print(f"Warning: Row count mismatch - main CSV has {len(df_main)} rows, "
              f"doc_assignments has {len(classes)} rows.")
        print("  Merging by original_index if available, otherwise by position.")

    # Merge strategy: positional if row counts match, else by original_index
    if len(classes) == len(df_main):
        df_main[column_name] = classes
    elif "original_index" in df_assign.columns:
        mapping = df_assign.set_index("original_index")[class_col]
        df_main[column_name] = df_main.index.map(mapping)
    else:
        print(f"Error: Cannot align — row counts differ and no original_index column.")
        sys.exit(1)

    n_unique = df_main[column_name].nunique()
    n_null = df_main[column_name].isna().sum()
    print(f"  Added column '{column_name}' ({n_unique} unique classes, {n_null} missing values)")
    return df_main


def main():
    parser = argparse.ArgumentParser(
        description="Enrich the main corpus CSV with model class/topic assignments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="Model result folder(s), e.g. 'LDA/run_20260224_094852_bigram_only' "
             "or just 'run_20260224_094852_bigram_only'",
    )
    parser.add_argument(
        "--column-name",
        nargs="*",
        default=None,
        help="Custom column name(s) for each folder. If not provided, names are auto-derived.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path. Defaults to overwriting the main dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying any file.",
    )
    args = parser.parse_args()

    # Validate column-name count if provided
    if args.column_name is not None and len(args.column_name) != len(args.folders):
        print(f"Error: Got {len(args.column_name)} --column-name(s) but {len(args.folders)} folder(s).")
        sys.exit(1)

    # Resolve all folders first
    resolved = []
    for folder_arg in args.folders:
        model_folder = find_model_folder(folder_arg)
        resolved.append(model_folder)
        print(f"Found: {model_folder.relative_to(RESULTS_DIR)}")

    # Derive column names
    if args.column_name:
        col_names = args.column_name
    else:
        col_names = [derive_column_name(f) for f in resolved]

    # Check for duplicate column names
    if len(set(col_names)) != len(col_names):
        print(f"Error: Duplicate column names detected: {col_names}")
        sys.exit(1)

    if args.dry_run:
        print("\n[Dry run] Would add the following columns:")
        for folder, col_name in zip(resolved, col_names):
            class_col = detect_class_column(folder)
            print(f"  {col_name} <- {folder.relative_to(RESULTS_DIR)}/doc_assignments.csv [{class_col}]")
        return

    # Load main dataset
    print(f"\nLoading main dataset: {DATA_PATH}")
    df_main = pd.read_csv(DATA_PATH, index_col=0)
    print(f"  Shape: {df_main.shape}")
    print(f"  Existing columns: {', '.join(df_main.columns)}")

    # Enrich with each model
    for model_folder, col_name in zip(resolved, col_names):
        if col_name in df_main.columns:
            print(f"\n  Column '{col_name}' already exists — overwriting.")
        print(f"\nProcessing: {model_folder.relative_to(RESULTS_DIR)}")
        df_main = enrich(model_folder, col_name, df_main)

    # Save
    output_path = Path(args.output) if args.output else DATA_PATH
    df_main.to_csv(output_path)
    print(f"\nSaved enriched dataset to: {output_path}")
    print(f"  Final shape: {df_main.shape}")
    print(f"  Columns: {', '.join(df_main.columns)}")


if __name__ == "__main__":
    main()
