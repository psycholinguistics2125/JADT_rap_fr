

"""
Grid-search BERTopic parameters on precomputed embeddings.

Assumes you already have:
- docs: list[str] of length N (here: 120 000 verses)
- embeddings: np.ndarray shape (N, d)

You can load them however you want (pickle, npy, etc.)
"""

import time
import itertools
import numpy as np
import pandas as pd

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN


#

n_neighbors_list = [15, 30, 50, 80, 120]
min_cluster_size_list = [10,20, 30, 40, 60, 80]
cluster_selection_methods = ["eom", "leaf"] 

# You can change these if you want to experiment more later
N_COMPONENTS = 15
MIN_DIST = 0.0
UMAP_METRIC = "cosine"

MIN_SAMPLES = 1                    # fixed for this grid


# Output file
RESULTS_CSV = "mpet_n_15_bertopic_param_search.csv"


# ---------------------------------------
# 2. HELPER: RUN ONE CONFIG
# ---------------------------------------
def run_single_config(docs, embeddings,
                      config_id,
                      n_neighbors,
                      min_cluster_size,
                      min_samples=MIN_SAMPLES,
                      cluster_selection_method="eom"):
    """
    Build UMAP + HDBSCAN + BERTopic with given params,
    fit on docs + embeddings, and return metrics.
    """
    print(f"\n=== Running config {config_id} ===")
    print(f"n_neighbors={n_neighbors}, min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}, cluster_selection_method={cluster_selection_method}")

    start_time = time.time()

    # UMAP
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=N_COMPONENTS,
        min_dist=MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
    )

    # HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
    )

    # BERTopic
    topic_model = BERTopic(
        embedding_model=None,   # IMPORTANT: we pass embeddings directly
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="french",
        nr_topics=None,         # or "auto" if you want topic merging
        low_memory=True,
    )

    # Fit model with precomputed embeddings
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # Compute metrics
    topics_np = np.array(topics)
    noise_mask = topics_np == -1
    noise_ratio = noise_mask.mean()

    unique_topics = set(topics_np.tolist())
    n_topics = len(unique_topics - {-1})  # exclude noise topic

    duration_minutes = (time.time() - start_time) / 60.0

    print(f"Finished config {config_id}: "
          f"noise_ratio={noise_ratio:.3f}, n_topics={n_topics}, "
          f"duration={duration_minutes:.1f} min")

    result = {
        "config_id": config_id,
        "n_neighbors": n_neighbors,
        "n_components": N_COMPONENTS,
        "min_dist": MIN_DIST,
        "umap_metric": UMAP_METRIC,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": cluster_selection_method,
        "noise_ratio": noise_ratio,
        "n_topics": n_topics,
        "duration_minutes": duration_minutes,
        "n_docs": len(docs),
    }

    return result


# ---------------------------------------
# 3. MAIN LOOP OVER GRID
# ---------------------------------------
def main(docs, embeddings):
    results = []

    # Load existing results if you re-run / resume
    try:
        existing = pd.read_csv(RESULTS_CSV)
        done_ids = set(existing["config_id"].tolist())
        results = existing.to_dict(orient="records")
        print(f"Loaded {len(existing)} existing results from {RESULTS_CSV}")
    except FileNotFoundError:
        done_ids = set()
        print("No existing results file found, starting fresh.")

    # Build the grid
    grid = list(itertools.product(n_neighbors_list, min_cluster_size_list,cluster_selection_methods))

    for idx, (n_neighbors, min_cluster_size,cluster_selection_method) in enumerate(grid, start=1):
        config_id = f"cfg_{idx:02d}"

        if config_id in done_ids:
            print(f"Skipping {config_id}, already done.")
            continue

        try:
            result = run_single_config(docs, embeddings,
                config_id=config_id,
                n_neighbors=n_neighbors,
                min_cluster_size=min_cluster_size,
                min_samples=MIN_SAMPLES,
                cluster_selection_method=cluster_selection_method,
            )
            results.append(result)

            # Save intermediate results every iteration
            df_results = pd.DataFrame(results)
            df_results.to_csv(RESULTS_CSV, index=False)
            print(f"Saved results to {RESULTS_CSV}")

        except Exception as e:
            print(f"ERROR in config {config_id}: {e}")
            # Log failure as well, so you see what crashed
            results.append({
                "config_id": config_id,
                "n_neighbors": n_neighbors,
                "n_components": N_COMPONENTS,
                "min_dist": MIN_DIST,
                "umap_metric": UMAP_METRIC,
                "min_cluster_size": min_cluster_size,
                "min_samples": MIN_SAMPLES,
                "cluster_selection_method": cluster_selection_method,
                "noise_ratio": np.nan,
                "n_topics": np.nan,
                "duration_minutes": np.nan,
                "n_docs": len(docs),
                "error": str(e),
            })
            df_results = pd.DataFrame(results)
            df_results.to_csv(RESULTS_CSV, index=False)
            print(f"Saved partial results to {RESULTS_CSV}")

    print("\nAll configs processed.")
    final_df = pd.DataFrame(results)
    print(final_df.sort_values("noise_ratio").head(50))
    print(len(final_df))


if __name__ == "__main__":


    

    df = pd.read_csv("/home/robin/Code_repo/psycholinguistic2125/JADT_rap_fr/data/20251126_cleaned_verses_lrfaf_corpus.csv")
    df = df[(df.year > 1991) & (df.year < 2024)]

    docs = df['lyrics_cleaned'].astype(str).tolist()
    embeddings = np.load("models/verses_mpnet_base_embeddings.npy")

    main(docs, embeddings)
