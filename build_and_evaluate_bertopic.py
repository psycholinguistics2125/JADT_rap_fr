#!/usr/bin/env python3
"""
Build and Evaluate BERTopic Model for French Rap Verses
========================================================
This script trains BERTopic models on French rap lyrics using:
- Pre-computed embeddings (3 models: CamemBERT, E5, MPNet)
- UMAP dimensionality reduction
- KMeans clustering (20 clusters for comparison with LDA)
- Evaluation metrics: silhouette score, artist/year separation
"""

import os
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from dotenv import load_dotenv

# ML imports
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from umap import UMAP

# BERTopic
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance

# Sentence Transformers
from sentence_transformers import SentenceTransformer

import openai

# Import shared evaluation utilities
from utils.utils_evaluation import (
    save_artist_metrics,
    save_temporal_metrics,
    create_topic_distribution_plot,
    create_topic_evolution_heatmap,
    create_artist_topics_heatmap,
    create_artist_specialization_plot,
    create_biannual_js_plot,
    create_year_topic_heatmap,
    print_evaluation_summary,
)

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
RESULTS_DIR = "/home/robin/Code_repo/psycholinguistic2125/JADT_rap_fr/results/BERTopic"
EMBEDDINGS_DIR = "/home/robin/Code_repo/psycholinguistic2125/JADT_rap_fr/models/embeddings"
DATA_PATH = "/home/robin/Code_repo/psycholinguistic2125/JADT_rap_fr/data/20260123_filter_verses_lrfaf_corpus.csv"

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Embedding models to test
EMBEDDING_MODELS = {
    'camembert': 'dangvantuan/sentence-camembert-base',
    'e5': 'intfloat/multilingual-e5-base',
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
}

# Default UMAP parameters
DEFAULT_UMAP_PARAMS = {
    'n_neighbors': 15,
    'n_components': 5,
    'min_dist': 0.0,
    'metric': 'cosine',
    'random_state': 42,
}


def load_data(path: str, sample_size: int = None) -> pd.DataFrame:
    """Load the dataset, optionally sampling."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} verses")

    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} documents for testing...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"Years range: {df['year'].min()} - {df['year'].max()}")
    print(f"Number of unique artists: {df['artist'].nunique()}")
    return df


def get_embedding_path(embedding_name: str) -> str:
    """Get the path for storing/loading embeddings."""
    return os.path.join(EMBEDDINGS_DIR, f"verses_iramuteq_filter_{embedding_name}.npy")


def compute_embeddings(docs: list, model_name: str, embedding_key: str,
                       batch_size: int = 64, save: bool = True,
                       force: bool = False) -> np.ndarray:
    """
    Compute embeddings for documents using a sentence transformer model.
    """
    embedding_path = get_embedding_path(embedding_key)

    # Check if embeddings already exist and match document count
    if os.path.exists(embedding_path) and not force:
        existing = np.load(embedding_path)
        if len(existing) == len(docs):
            print(f"Loading existing embeddings from {embedding_path}...")
            print(f"Loaded embeddings shape: {existing.shape}")
            return existing
        else:
            print(f"Existing embeddings ({len(existing)}) don't match documents ({len(docs)}), recomputing...")

    print(f"\nComputing embeddings with {model_name}...")
    print(f"This may take a while for {len(docs)} documents...")

    # Load model
    device = 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)

    # For E5 models, prepend "query: " to documents
    if 'e5' in model_name.lower():
        print("Adding 'query: ' prefix for E5 model...")
        docs_to_encode = [f"query: {doc}" for doc in docs]
    else:
        docs_to_encode = docs

    # Compute embeddings
    embeddings = model.encode(
        docs_to_encode,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings
    if save:
        print(f"Saving embeddings to {embedding_path}...")
        np.save(embedding_path, embeddings)

    return embeddings


def load_embeddings(embedding_key: str) -> np.ndarray:
    """Load pre-computed embeddings."""
    embedding_path = get_embedding_path(embedding_key)
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embeddings not found at {embedding_path}. Run with --compute-embeddings first.")

    print(f"Loading embeddings from {embedding_path}...")
    embeddings = np.load(embedding_path)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    return embeddings


def create_umap_model(n_neighbors: int = 15, n_components: int = 5,
                      min_dist: float = 0.0, metric: str = 'cosine',
                      random_state: int = 42) -> UMAP:
    """Create UMAP model with specified parameters."""
    return UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=True,
    )


def create_kmeans_model(n_clusters: int = 20, random_state: int = 42) -> KMeans:
    """Create KMeans clustering model."""
    return KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )


def create_bertopic_model(docs: list, embeddings: np.ndarray,
                          umap_model: UMAP, cluster_model: KMeans,
                          use_openai: bool = True,
                          openai_client=None) -> tuple:
    """
    Create and fit BERTopic model with pre-computed embeddings.

    Returns:
        topic_model: Fitted BERTopic model
        topics: Topic assignments for each document
        probs: Topic probabilities (None for KMeans)
    """
    print("\n" + "="*60)
    print("CREATING BERTOPIC MODEL")
    print("="*60)

    # Representation models
    representation_models = {}

    # MMR for diverse keywords
    mmr_model = MaximalMarginalRelevance(diversity=0.5)
    representation_models["MMR"] = mmr_model

    # OpenAI representation (optional)
    if use_openai and openai_client:
        try:
            import tiktoken
            from bertopic.representation import OpenAI as OpenAIRep

            openai_representation = OpenAIRep(
                openai_client,
                model="gpt-4o-mini",
                chat=True,
                nr_docs=15,
                delay_in_seconds=2,
                doc_length=150,
                diversity=0.3,
                tokenizer=tiktoken.encoding_for_model("gpt-4o-mini"),
                prompt="""Voici des paroles de rap français appartenant à un même thème :
[DOCUMENTS]

Mots-clés du thème : [KEYWORDS]

Analyse ces paroles et identifie le thème principal en tenant compte :
- Du vocabulaire et de l'argot utilisé
- Des références culturelles hip-hop
- Des émotions et messages véhiculés
- Du contexte social ou personnel

Donne un titre de thème en 3-7 mots, précis et évocateur.
Exemples : "Violence Urbaine et Survie", "Critique du Système Policier", "Nostalgie du Quartier", "Ambition et Réussite Matérielle", "Amour Toxique"

Titre du thème :"""
            )
            representation_models["OpenAI"] = openai_representation
            print("  OpenAI representation enabled")
        except Exception as e:
            print(f"  Could not enable OpenAI representation: {e}")

    # Create BERTopic model
    print("Creating BERTopic model...")
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=cluster_model,  # Using KMeans instead of HDBSCAN
        embedding_model=None,  # Pre-computed embeddings
        language="french",
        low_memory=True,
        representation_model=representation_models if representation_models else None,
        calculate_probabilities=False,  # KMeans doesn't give probabilities
    )

    # Fit the model
    print("Fitting BERTopic...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    print(f"Number of topics found: {len(set(topics)) - (1 if -1 in topics else 0)}")
    print(f"Documents in outlier topic (-1): {(np.array(topics) == -1).sum()}")

    return topic_model, np.array(topics), probs


def compute_silhouette_metrics(umap_embeddings: np.ndarray, topics: np.ndarray) -> dict:
    """
    Compute silhouette scores in UMAP-reduced space only.
    (Computing on original high-dimensional embeddings is not meaningful for clustering evaluation)
    """
    print("\nComputing silhouette metrics (UMAP space)...")

    metrics = {}

    # Filter out outliers (topic -1) for silhouette computation
    valid_mask = topics != -1
    topics_valid = topics[valid_mask]

    if len(np.unique(topics_valid)) < 2:
        print("  Not enough clusters for silhouette score")
        return {'silhouette_umap': None, 'per_cluster_silhouette_umap': None}

    # Silhouette on UMAP-reduced embeddings
    umap_valid = umap_embeddings[valid_mask]
    sil_umap = silhouette_score(umap_valid, topics_valid, metric='euclidean')
    metrics['silhouette_umap'] = float(sil_umap)
    print(f"  Silhouette (UMAP space): {sil_umap:.4f}")

    # Per-cluster silhouette on UMAP
    sil_samples_umap = silhouette_samples(umap_valid, topics_valid, metric='euclidean')
    per_cluster_sil_umap = {}
    for cluster_id in np.unique(topics_valid):
        cluster_mask = topics_valid == cluster_id
        per_cluster_sil_umap[int(cluster_id)] = float(sil_samples_umap[cluster_mask].mean())
    metrics['per_cluster_silhouette_umap'] = per_cluster_sil_umap

    return metrics


def compute_artist_separation(embeddings: np.ndarray, topics: np.ndarray,
                              df: pd.DataFrame, min_docs_per_artist: int = 10,
                              top_artists_per_topic: int = 20) -> dict:
    """
    Compute how well topics separate/distinguish artists.

    Key question: Do topics capture artist-specific styles?
    - If yes: artists should be "specialists" (concentrated in few topics)
    - If no: artists are "generalists" (spread across many topics)
    """
    print("\nComputing artist separation metrics...")
    print("  Question: Do topics distinguish artists?")

    metrics = {}
    n_topics = len(np.unique(topics[topics >= 0]))
    max_entropy = np.log(n_topics)  # Maximum possible entropy

    # Get artists with enough documents
    artist_counts = df['artist'].value_counts()
    valid_artists = artist_counts[artist_counts >= min_docs_per_artist].index.tolist()
    print(f"  Artists with >= {min_docs_per_artist} docs: {len(valid_artists)}")

    # Filter to valid artists
    mask = df['artist'].isin(valid_artists)
    embeddings_filtered = embeddings[mask.values]
    topics_filtered = topics[mask.values]
    artists_filtered = df.loc[mask, 'artist'].values

    # =========================================================================
    # PER-ARTIST METRICS
    # =========================================================================
    artist_metrics = []

    for artist in valid_artists:
        artist_mask = (df['artist'] == artist).values
        artist_topics_all = topics[artist_mask]
        # Filter out outliers (topic -1)
        artist_topics = artist_topics_all[artist_topics_all >= 0]
        n_docs = len(artist_topics)

        if n_docs == 0:
            continue

        # Topic distribution for this artist
        topic_counts = np.bincount(artist_topics, minlength=n_topics)
        topic_probs = topic_counts / topic_counts.sum()

        # Entropy: low = specialist, high = generalist
        entropy = stats.entropy(topic_probs + 1e-10)
        normalized_entropy = entropy / max_entropy  # 0-1 scale

        # Dominant topic info
        dominant_topic = int(topic_probs.argmax())
        dominant_ratio = float(topic_probs.max())

        # Number of "significant" topics (>5% of artist's docs)
        n_significant_topics = int((topic_probs > 0.05).sum())

        # Top 3 topics for this artist
        top_3_topics = np.argsort(topic_probs)[::-1][:3].tolist()
        top_3_ratios = [float(topic_probs[t]) for t in top_3_topics]

        # Classification
        if dominant_ratio >= 0.5:
            classification = 'specialist'
        elif dominant_ratio >= 0.3:
            classification = 'moderate'
        else:
            classification = 'generalist'

        # Gini coefficient (inequality measure)
        sorted_probs = np.sort(topic_probs)
        n = len(sorted_probs)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_probs) - (n + 1) * np.sum(sorted_probs)) / (n * np.sum(sorted_probs) + 1e-10)

        artist_metrics.append({
            'artist': artist,
            'n_docs': n_docs,
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy),
            'dominant_topic': dominant_topic,
            'dominant_ratio': dominant_ratio,
            'n_significant_topics': n_significant_topics,
            'top_3_topics': top_3_topics,
            'top_3_ratios': top_3_ratios,
            'gini_coefficient': float(gini),
            'classification': classification,
        })

    metrics['per_artist_metrics'] = artist_metrics

    # =========================================================================
    # GENERAL METRICS
    # =========================================================================
    if not artist_metrics:
        print("  WARNING: No valid artists found (not enough docs per artist)")
        metrics['pct_specialists'] = 0.0
        metrics['pct_moderate'] = 0.0
        metrics['pct_generalists'] = 0.0
        metrics['mean_dominant_ratio'] = 0.0
        metrics['mean_js_divergence'] = 0.0
        metrics['mean_topic_concentration'] = 0.0
        metrics['artist_specialization'] = 0.0
        metrics['topic_top_artists'] = {}
        return metrics

    entropies = [a['entropy'] for a in artist_metrics]
    dominant_ratios = [a['dominant_ratio'] for a in artist_metrics]
    n_sig_topics = [a['n_significant_topics'] for a in artist_metrics]
    classifications = [a['classification'] for a in artist_metrics]

    # 1. Entropy statistics
    metrics['mean_artist_entropy'] = float(np.mean(entropies))
    metrics['std_artist_entropy'] = float(np.std(entropies))
    metrics['median_artist_entropy'] = float(np.median(entropies))

    # 2. Specialization score (1 - normalized mean entropy)
    metrics['artist_specialization'] = float(1 - (np.mean(entropies) / max_entropy))

    # 3. Distribution of artist types
    n_specialists = classifications.count('specialist')
    n_moderate = classifications.count('moderate')
    n_generalists = classifications.count('generalist')
    total = len(classifications)

    metrics['pct_specialists'] = float(n_specialists / total * 100)
    metrics['pct_moderate'] = float(n_moderate / total * 100)
    metrics['pct_generalists'] = float(n_generalists / total * 100)

    print(f"\n  [ARTIST TYPE DISTRIBUTION]")
    print(f"    Specialists (>50% in 1 topic): {n_specialists} ({metrics['pct_specialists']:.1f}%)")
    print(f"    Moderate (30-50% in 1 topic):  {n_moderate} ({metrics['pct_moderate']:.1f}%)")
    print(f"    Generalists (<30% in 1 topic): {n_generalists} ({metrics['pct_generalists']:.1f}%)")

    # 4. Average dominant topic ratio
    metrics['mean_dominant_ratio'] = float(np.mean(dominant_ratios))
    metrics['median_dominant_ratio'] = float(np.median(dominant_ratios))
    print(f"\n  [TOPIC CONCENTRATION]")
    print(f"    Mean dominant topic ratio: {metrics['mean_dominant_ratio']:.2%}")
    print(f"    Median dominant topic ratio: {metrics['median_dominant_ratio']:.2%}")

    # 5. Average number of significant topics per artist
    metrics['mean_significant_topics'] = float(np.mean(n_sig_topics))
    print(f"    Mean significant topics per artist: {metrics['mean_significant_topics']:.1f}")

    # 6. Jensen-Shannon divergence between artists
    artist_topic_profiles = {}
    for a in artist_metrics:
        artist = a['artist']
        artist_mask = df['artist'] == artist
        artist_topics_all = topics[artist_mask.values]
        artist_topics = artist_topics_all[artist_topics_all >= 0]
        topic_counts = np.bincount(artist_topics, minlength=n_topics)
        if topic_counts.sum() > 0:
            artist_topic_profiles[artist] = topic_counts / topic_counts.sum()

    js_distances = []
    valid_profile_artists = list(artist_topic_profiles.keys())
    for i, artist1 in enumerate(valid_profile_artists):
        for artist2 in valid_profile_artists[i+1:]:
            js_dist = jensenshannon(
                artist_topic_profiles[artist1],
                artist_topic_profiles[artist2]
            )
            if not np.isnan(js_dist):
                js_distances.append(js_dist)

    if js_distances:
        metrics['mean_js_divergence'] = float(np.mean(js_distances))
        metrics['std_js_divergence'] = float(np.std(js_distances))
        print(f"\n  [INTER-ARTIST DIVERGENCE]")
        print(f"    Mean JS divergence: {metrics['mean_js_divergence']:.4f} (higher = more different)")

    # 7. Topic purity: for each topic, how concentrated is it among few artists?
    topic_concentration = []
    for topic_id in range(n_topics):
        topic_mask = topics == topic_id
        if topic_mask.sum() == 0:
            continue
        topic_artists = df.loc[topic_mask, 'artist'].value_counts(normalize=True)
        # Top 5 artists share what % of this topic?
        top5_share = topic_artists.head(5).sum()
        topic_concentration.append(float(top5_share))

    if topic_concentration:
        metrics['mean_topic_concentration'] = float(np.mean(topic_concentration))
        print(f"    Mean topic concentration (top 5 artists): {metrics['mean_topic_concentration']:.2%}")

    # 8. Overall interpretation
    print(f"\n  [INTERPRETATION]")
    if metrics['pct_specialists'] > 50:
        print(f"    ✓ Topics DISTINGUISH artists well ({metrics['pct_specialists']:.0f}% are specialists)")
    elif metrics['pct_specialists'] > 30:
        print(f"    ~ Topics MODERATELY distinguish artists ({metrics['pct_specialists']:.0f}% are specialists)")
    else:
        print(f"    ✗ Topics do NOT distinguish artists well ({metrics['pct_specialists']:.0f}% are specialists)")

    # Silhouette score (clustering quality based on artist labels)
    if len(valid_artists) > 1:
        try:
            artist_labels = pd.Categorical(artists_filtered).codes
            sil_score = silhouette_score(embeddings_filtered, artist_labels, metric='cosine')
            metrics['silhouette_by_artist'] = float(sil_score)
        except Exception as e:
            metrics['silhouette_by_artist'] = None

    # =========================================================================
    # TOP ARTISTS PER TOPIC
    # =========================================================================
    topic_top_artists = {}
    for topic_id in range(n_topics):
        topic_mask = topics == topic_id
        total_docs_in_topic = topic_mask.sum()

        if total_docs_in_topic == 0:
            topic_top_artists[topic_id] = []
            continue

        topic_scores = {}
        for artist in valid_artists:
            artist_mask = (df['artist'] == artist).values
            n_docs_in_topic = (topic_mask & artist_mask).sum()
            if n_docs_in_topic > 0:
                pct_of_topic = (n_docs_in_topic / total_docs_in_topic) * 100
                topic_scores[artist] = {
                    'n_docs': int(n_docs_in_topic),
                    'pct_of_topic': float(pct_of_topic),
                    'total_topic_docs': int(total_docs_in_topic)
                }

        sorted_artists = sorted(topic_scores.items(), key=lambda x: x[1]['n_docs'], reverse=True)[:top_artists_per_topic]
        topic_top_artists[topic_id] = sorted_artists

    metrics['topic_top_artists'] = topic_top_artists

    return metrics


def compute_temporal_separation(embeddings: np.ndarray, topics: np.ndarray,
                                df: pd.DataFrame, year_column: str = 'year') -> dict:
    """Compute how topics evolve over time."""
    print("\nComputing temporal separation metrics...")

    metrics = {}
    n_topics = len(np.unique(topics[topics >= 0]))

    # Filter valid years and non-outlier topics
    valid_mask = df[year_column].notna() & (topics >= 0)
    years = df.loc[valid_mask, year_column].astype(int).values
    topics_valid = topics[valid_mask.values]

    unique_years = sorted(np.unique(years))
    print(f"  Years covered: {min(unique_years)} - {max(unique_years)}")

    # 1. Topic distribution by year
    topic_by_year = {}
    for year in unique_years:
        year_mask = years == year
        if year_mask.sum() > 0:
            year_topics = topics_valid[year_mask]
            topic_counts = np.bincount(year_topics, minlength=n_topics)
            topic_by_year[year] = topic_counts / topic_counts.sum()

    topic_evolution_df = pd.DataFrame(topic_by_year).T
    topic_evolution_df.index.name = 'year'

    # 2. Trend correlations per topic
    trend_correlations = {}
    years_array = np.array(list(topic_by_year.keys()))

    for topic_id in range(n_topics):
        topic_values = [topic_by_year[y][topic_id] for y in years_array]
        if len(set(topic_values)) > 1:  # Need variance for correlation
            corr, p_value = stats.pearsonr(years_array, topic_values)
            trend_correlations[topic_id] = {
                'correlation': float(corr),
                'p_value': float(p_value),
                'trend': 'increasing' if corr > 0.3 else ('decreasing' if corr < -0.3 else 'stable')
            }
        else:
            trend_correlations[topic_id] = {'correlation': 0.0, 'p_value': 1.0, 'trend': 'stable'}

    metrics['trend_correlations'] = trend_correlations

    # 3. Temporal variance per topic
    temporal_variance = []
    for topic_id in range(n_topics):
        topic_values = [topic_by_year[y][topic_id] for y in years_array]
        temporal_variance.append(np.var(topic_values))

    metrics['mean_temporal_variance'] = float(np.mean(temporal_variance))
    metrics['topic_temporal_variance'] = [float(v) for v in temporal_variance]
    print(f"  Mean temporal variance: {metrics['mean_temporal_variance']:.6f}")

    # 4. Decade comparison
    if max(unique_years) - min(unique_years) >= 20:
        decades = {}
        for year in unique_years:
            decade = (year // 10) * 10
            if decade not in decades:
                decades[decade] = []
            decades[decade].append(year)

        decade_profiles = {}
        for decade, decade_years in decades.items():
            decade_mask = np.isin(years, decade_years)
            if decade_mask.sum() > 0:
                decade_topics = topics_valid[decade_mask]
                topic_counts = np.bincount(decade_topics, minlength=n_topics)
                decade_profiles[decade] = topic_counts / topic_counts.sum()

        sorted_decades = sorted(decade_profiles.keys())
        decade_changes = {}
        for i in range(len(sorted_decades) - 1):
            d1, d2 = sorted_decades[i], sorted_decades[i+1]
            js_dist = jensenshannon(decade_profiles[d1], decade_profiles[d2])
            decade_changes[f"{d1}s->{d2}s"] = float(js_dist)

        metrics['decade_changes'] = decade_changes
        print(f"  Decade changes (JS divergence): {decade_changes}")

    metrics['topic_evolution'] = topic_evolution_df.to_dict()

    return metrics


def display_topics(topic_model: BERTopic, num_words: int = 30) -> dict:
    """
    Display and return topic descriptions with ALL representation methods.

    Includes c-TF-IDF (default), MMR, OpenAI, and any other configured representations.
    """
    print("\n" + "="*60)
    print("TOPIC DESCRIPTIONS")
    print("="*60)

    topics_info = topic_model.get_topic_info()
    topics = {}

    # Get all available representation columns
    representation_cols = [col for col in topics_info.columns
                          if col not in ['Topic', 'Count', 'Name', 'Representative_Docs', 'Representation']]
    print(f"Available representations: {representation_cols if representation_cols else ['c-TF-IDF (default)']}")

    for idx, row in topics_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            continue

        # Get c-TF-IDF representation (default)
        topic_words = topic_model.get_topic(topic_id)
        if not topic_words:
            continue

        words_ctfidf = [word for word, score in topic_words[:num_words]]
        scores_ctfidf = [float(score) for word, score in topic_words[:num_words]]

        # Build topic data with all representations
        topic_data = {
            'count': int(row['Count']),
            'top_words': ', '.join(words_ctfidf[:15]),
            # c-TF-IDF representation (the default/main one)
            'ctfidf': {
                'words': words_ctfidf,
                'scores': scores_ctfidf,
            },
        }

        # Add all other representations from topic_info columns
        for col in representation_cols:
            if col not in row:
                continue
            value = row[col]
            # Handle array/list for notna check
            if isinstance(value, (list, np.ndarray)):
                if len(value) == 0:
                    continue
            elif pd.isna(value):
                continue
                # Some representations are lists of tuples, some are strings
                if isinstance(value, list):
                    # It's a list of (word, score) tuples
                    if value and isinstance(value[0], tuple):
                        topic_data[col.lower()] = {
                            'words': [w for w, s in value[:num_words]],
                            'scores': [float(s) for w, s in value[:num_words]],
                        }
                    else:
                        topic_data[col.lower()] = value
                elif isinstance(value, str):
                    topic_data[col.lower()] = value

        # Also try to get representations directly from the model if available
        if hasattr(topic_model, 'topic_representations_'):
            for rep_name, rep_data in topic_model.topic_representations_.items():
                if topic_id in rep_data and rep_name.lower() not in topic_data:
                    rep_value = rep_data[topic_id]
                    if isinstance(rep_value, list) and rep_value:
                        if isinstance(rep_value[0], tuple):
                            topic_data[rep_name.lower()] = {
                                'words': [w for w, s in rep_value[:num_words]],
                                'scores': [float(s) if isinstance(s, (int, float)) else 0.0 for w, s in rep_value[:num_words]],
                            }
                        else:
                            topic_data[rep_name.lower()] = rep_value

        topics[topic_id] = topic_data

        # Display with OpenAI label if available
        openai_label = topic_data.get('openai', None)
        label_str = f" [{openai_label}]" if openai_label else ""
        print(f"\nTopic {topic_id}{label_str} (n={row['Count']}): {', '.join(words_ctfidf[:15])}")

    return topics


def save_results(results: dict, topic_model: BERTopic, embeddings: np.ndarray,
                 umap_embeddings: np.ndarray, topics: np.ndarray, df: pd.DataFrame,
                 run_dir: str):
    """Save all results and model artifacts."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save BERTopic model
    model_path = os.path.join(run_dir, "bertopic_model")
    topic_model.save(model_path, serialization="safetensors", save_ctfidf=True)
    print(f"  Model saved to: {model_path}")

    # Save UMAP embeddings
    umap_path = os.path.join(run_dir, "umap_embeddings.npy")
    np.save(umap_path, umap_embeddings)
    print(f"  UMAP embeddings saved to: {umap_path}")

    # Save topic assignments
    topics_path = os.path.join(run_dir, "topics.npy")
    np.save(topics_path, topics)
    print(f"  Topic assignments saved to: {topics_path}")

    # Save metrics as JSON
    metrics_to_save = {
        'silhouette_metrics': results['silhouette'],
        'artist_metrics': {
            'silhouette_by_artist': results['artist_separation'].get('silhouette_by_artist'),
            'js_divergence': results['artist_separation'].get('mean_js_divergence'),
            'specialization': results['artist_separation'].get('artist_specialization'),
            'pct_specialists': results['artist_separation'].get('pct_specialists'),
            'pct_moderate': results['artist_separation'].get('pct_moderate'),
            'pct_generalists': results['artist_separation'].get('pct_generalists'),
            'mean_dominant_ratio': results['artist_separation'].get('mean_dominant_ratio'),
            'mean_significant_topics': results['artist_separation'].get('mean_significant_topics'),
            'mean_topic_concentration': results['artist_separation'].get('mean_topic_concentration'),
        },
        'temporal_metrics': {
            'mean_variance': results['temporal_separation']['mean_temporal_variance'],
            'decade_changes': results['temporal_separation'].get('decade_changes', {}),
            'biannual_changes': results['temporal_separation'].get('biannual_changes', {}),
            'mean_biannual_js': results['temporal_separation'].get('mean_biannual_js'),
        },
        'parameters': results['parameters'],
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    print(f"  Metrics saved to: {metrics_path}")

    # Save topic descriptions
    topics_desc_path = os.path.join(run_dir, "topics.json")
    topics_to_save = {}
    for tid, tdata in results['topics'].items():
        topics_to_save[str(tid)] = tdata
    with open(topics_desc_path, 'w', encoding='utf-8') as f:
        json.dump(topics_to_save, f, indent=2, ensure_ascii=False)
    print(f"  Topics saved to: {topics_desc_path}")

    # Save topic evolution and artist metrics using shared utilities
    save_temporal_metrics(results['temporal_separation'], run_dir)
    save_artist_metrics(results['artist_separation'], run_dir)

    # Save document assignments
    doc_assign_path = os.path.join(run_dir, "doc_assignments.csv")
    doc_df = df.copy()
    doc_df['topic'] = topics
    doc_df[['artist', 'title', 'year', 'topic']].to_csv(doc_assign_path, index=False)
    print(f"  Document assignments saved to: {doc_assign_path}")

    print(f"\n  All results saved to: {run_dir}")


def create_visualizations(results: dict, embeddings: np.ndarray, umap_embeddings: np.ndarray,
                          topics: np.ndarray, df: pd.DataFrame, run_dir: str,
                          top_n_artists: int = 50):
    """Create and save visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    n_topics = len(np.unique(topics[topics >= 0]))

    # 1. Per-cluster silhouette bar plot (BERTopic-specific)
    if 'per_cluster_silhouette_umap' in results['silhouette']:
        fig, ax = plt.subplots(figsize=(12, 6))
        sil_scores = results['silhouette']['per_cluster_silhouette_umap']
        topic_ids = sorted(sil_scores.keys())
        scores = [sil_scores[tid] for tid in topic_ids]
        colors = ['green' if s > 0.3 else 'orange' if s > 0.1 else 'red' for s in scores]
        ax.bar(topic_ids, scores, color=colors)
        ax.axhline(y=0.3, color='green', linestyle='--', label='Good threshold')
        ax.axhline(y=0.1, color='orange', linestyle='--', label='Acceptable threshold')
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Per-Cluster Silhouette Scores (UMAP space)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'silhouette_plot.png'), dpi=150)
        plt.close()
        print("  Saved silhouette plot")

    # 2. Topic evolution heatmap (using shared utility)
    create_topic_evolution_heatmap(results['temporal_separation'], run_dir,
                                    title="BERTopic Topic Prevalence Over Time")

    # 3. Topic distribution (using shared utility)
    create_topic_distribution_plot(topics, run_dir, title="BERTopic Topic Distribution")

    # 4. Year-topic heatmap (using shared utility)
    create_year_topic_heatmap(topics, df, run_dir,
                               title="BERTopic Topic Distribution by Year")

    # 5. UMAP visualization colored by topic (BERTopic-specific)
    if umap_embeddings.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(12, 10))

        # Use first 2 UMAP dimensions or project to 2D if more
        if umap_embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            plot_coords = pca.fit_transform(umap_embeddings)
        else:
            plot_coords = umap_embeddings

        scatter = ax.scatter(plot_coords[:, 0], plot_coords[:, 1],
                           c=topics, cmap='tab20', alpha=0.5, s=5)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Documents in UMAP Space - Colored by Topic')
        plt.colorbar(scatter, ax=ax, label='Topic')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'umap_topics.png'), dpi=150)
        plt.close()
        print("  Saved UMAP visualization")

    # 6. Artist topic profiles heatmap (using shared utility)
    create_artist_topics_heatmap(topics, df, run_dir, top_n_artists=top_n_artists,
                                  title=f"Top {top_n_artists} Artists - BERTopic Profiles")

    # 7. Artist specialization plot (using shared utility)
    create_artist_specialization_plot(results['artist_separation'], run_dir)

    # 8. Biannual JS divergence plot (using shared utility)
    create_biannual_js_plot(results['temporal_separation'], run_dir,
                            title="BERTopic - 2-Year Window JS Divergence")

    print("  All visualizations saved!")


def print_summary(results: dict):
    """Print a summary of all metrics."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    print("\n[SILHOUETTE METRICS]")
    if results['silhouette'].get('silhouette_umap'):
        print(f"   Silhouette (UMAP space): {results['silhouette']['silhouette_umap']:.4f}")

    print("\n[ARTIST SEPARATION]")
    if results['artist_separation'].get('silhouette_by_artist'):
        print(f"   Silhouette by Artist: {results['artist_separation']['silhouette_by_artist']:.4f}")
    if results['artist_separation'].get('mean_js_divergence'):
        print(f"   Mean JS Divergence: {results['artist_separation']['mean_js_divergence']:.4f}")
    if results['artist_separation'].get('artist_specialization'):
        print(f"   Artist Specialization: {results['artist_separation']['artist_specialization']:.4f}")

    print("\n[TEMPORAL EVOLUTION]")
    print(f"   Mean Temporal Variance: {results['temporal_separation']['mean_temporal_variance']:.6f}")
    if 'decade_changes' in results['temporal_separation']:
        print("   Decade Changes (JS divergence):")
        for period, change in results['temporal_separation']['decade_changes'].items():
            print(f"      {period}: {change:.4f}")

    print("\n[TOPIC TRENDS]")
    increasing = []
    decreasing = []
    for tid, trend_data in results['temporal_separation']['trend_correlations'].items():
        if trend_data['trend'] == 'increasing':
            increasing.append(tid)
        elif trend_data['trend'] == 'decreasing':
            decreasing.append(tid)
    print(f"   Increasing topics: {increasing}")
    print(f"   Decreasing topics: {decreasing}")


def run_experiment(embedding_key: str = 'camembert',
                   n_clusters: int = 20,
                   umap_params: dict = None,
                   sample_size: int = None,
                   compute_embeddings_flag: bool = False,
                   use_openai: bool = True,
                   num_words_per_topic: int = 30,
                   top_artists_per_topic: int = 20,
                   top_n_artists_heatmap: int = 50):
    """Run a complete BERTopic experiment."""

    print("\n" + "="*60)
    print("BERTOPIC EXPERIMENT FOR FRENCH RAP CORPUS")
    print("="*60)

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}_{embedding_key}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    # UMAP parameters
    if umap_params is None:
        umap_params = DEFAULT_UMAP_PARAMS.copy()

    print(f"\nEmbedding model: {EMBEDDING_MODELS[embedding_key]}")
    print(f"UMAP parameters: {umap_params}")
    print(f"Number of clusters: {n_clusters}")

    # Load data
    df = load_data(DATA_PATH, sample_size=sample_size)
    docs = df['lyrics_cleaned'].fillna("").astype(str).tolist()

    # Get or compute embeddings
    if compute_embeddings_flag:
        # Don't save if sampling (would overwrite full embeddings)
        should_save = sample_size is None
        embeddings = compute_embeddings(
            docs,
            EMBEDDING_MODELS[embedding_key],
            embedding_key,
            batch_size=64,
            save=should_save,
            force=True  # Force recompute when flag is set
        )
    else:
        embeddings = load_embeddings(embedding_key)

    # Verify embeddings match documents
    if len(embeddings) != len(docs):
        print(f"WARNING: Embeddings ({len(embeddings)}) don't match documents ({len(docs)})")
        print("This may happen if using pre-computed embeddings with different sample size.")
        print("Consider recomputing embeddings with --compute-embeddings flag.")

    # Create UMAP model and transform embeddings
    print("\nApplying UMAP dimensionality reduction...")
    umap_model = create_umap_model(**umap_params)
    umap_embeddings = umap_model.fit_transform(embeddings)
    print(f"UMAP embeddings shape: {umap_embeddings.shape}")

    # Create KMeans model
    kmeans_model = create_kmeans_model(n_clusters=n_clusters)

    # Initialize OpenAI client if needed
    openai_client = None
    if use_openai:
        try:
            openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print("OpenAI client initialized")
        except Exception as e:
            print(f"Could not initialize OpenAI client: {e}")
            use_openai = False

    # Create and fit BERTopic model
    topic_model, topics, probs = create_bertopic_model(
        docs, embeddings, umap_model, kmeans_model,
        use_openai=use_openai, openai_client=openai_client
    )

    # Compute metrics
    silhouette_metrics = compute_silhouette_metrics(umap_embeddings, topics)
    artist_metrics = compute_artist_separation(embeddings, topics, df,
                                               top_artists_per_topic=top_artists_per_topic)
    temporal_metrics = compute_temporal_separation(embeddings, topics, df)

    # Display topics
    topics_desc = display_topics(topic_model, num_words=num_words_per_topic)

    # Compile results
    results = {
        'silhouette': silhouette_metrics,
        'artist_separation': artist_metrics,
        'temporal_separation': temporal_metrics,
        'topics': topics_desc,
        'parameters': {
            'embedding_model': EMBEDDING_MODELS[embedding_key],
            'embedding_key': embedding_key,
            'n_clusters': n_clusters,
            'umap_params': umap_params,
            'num_documents': len(docs),
            'num_words_per_topic': num_words_per_topic,
            'timestamp': timestamp,
            'run_dir': run_dir,
            'use_openai': use_openai,
        }
    }

    # Print summary
    print_summary(results)

    # Save everything
    save_results(results, topic_model, embeddings, umap_embeddings, topics, df, run_dir)

    # Create visualizations
    create_visualizations(results, embeddings, umap_embeddings, topics, df, run_dir,
                         top_n_artists=top_n_artists_heatmap)

    return results, topic_model, topics, df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build and evaluate BERTopic model for French rap corpus')

    # Model selection
    parser.add_argument('--embedding', type=str, default='camembert',
                        choices=['camembert', 'e5', 'mpnet'],
                        help='Embedding model to use')
    parser.add_argument('--compute-embeddings', action='store_true',
                        help='Compute embeddings (otherwise load from disk)')

    # Clustering parameters
    parser.add_argument('--clusters', type=int, default=20, help='Number of clusters')

    # UMAP parameters
    parser.add_argument('--umap-neighbors', type=int, default=15, help='UMAP n_neighbors')
    parser.add_argument('--umap-components', type=int, default=5, help='UMAP n_components')
    parser.add_argument('--umap-min-dist', type=float, default=0.0, help='UMAP min_dist')

    # Other parameters
    parser.add_argument('--sample', type=int, default=None, help='Sample size for testing')
    parser.add_argument('--no-openai', action='store_true', help='Disable OpenAI labeling')
    parser.add_argument('--num-words', type=int, default=30, help='Number of words per topic')
    parser.add_argument('--top-artists-topic', type=int, default=20, help='Top N artists per topic')
    parser.add_argument('--top-artists-heatmap', type=int, default=50, help='Top N artists in heatmap')

    args = parser.parse_args()

    # Build UMAP params
    umap_params = {
        'n_neighbors': args.umap_neighbors,
        'n_components': args.umap_components,
        'min_dist': args.umap_min_dist,
        'metric': 'cosine',
        'random_state': 42,
    }

    results, model, topics, df = run_experiment(
        embedding_key=args.embedding,
        n_clusters=args.clusters,
        umap_params=umap_params,
        sample_size=args.sample,
        compute_embeddings_flag=args.compute_embeddings,
        use_openai=not args.no_openai,
        num_words_per_topic=args.num_words,
        top_artists_per_topic=args.top_artists_topic,
        top_n_artists_heatmap=args.top_artists_heatmap,
    )
