import json
from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_site_data():
    """Load the pre-processed site data JSON."""
    data_path = DATA_DIR / "site_data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_corpus():
    """Get corpus-level metadata."""
    return load_site_data()["corpus"]


def get_model(model_key):
    """Get model data. model_key is 'bertopic', 'lda', or 'iramuteq'."""
    return load_site_data()["models"].get(model_key, {})


def get_comparison():
    """Get cross-model comparison data."""
    return load_site_data()["comparison"]
