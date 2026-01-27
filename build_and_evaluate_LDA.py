#!/usr/bin/env python3
"""
Build and Evaluate LDA Model for French Rap Verses
===================================================
This script trains LDA models on French rap lyrics and evaluates them using:
- Topic coherence metrics (C_V, U_Mass)
- Artist separation (how well topics distinguish artists)
- Year separation (temporal topic evolution)
"""

import os
import re
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP and LDA
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser

# Sklearn for additional metrics
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Import shared evaluation utilities
from utils.utils_evaluation import (
    compute_artist_separation,
    compute_temporal_separation,
    save_artist_metrics,
    save_temporal_metrics,
    create_topic_distribution_plot,
    create_artist_topics_heatmap,
    create_artist_specialization_plot,
    create_biannual_js_plot,
    create_year_topic_heatmap,
    print_evaluation_summary,
)

warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = str(PROJECT_ROOT / "results" / "LDA")
DATA_PATH = str(PROJECT_ROOT / "data" / "20260123_filter_verses_lrfaf_corpus.csv")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# French stopwords extended for rap context
FRENCH_STOPWORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'à', 'au', 'aux',
    'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'que', 'qui', 'quoi',
    'dont', 'où', 'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta',
    'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
    'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'me', 'te',
    'se', 'lui', 'y', 'en', 'ne', 'pas', 'plus', 'moins', 'très', 'trop',
    'bien', 'mal', 'peu', 'beaucoup', 'tout', 'tous', 'toute', 'toutes',
    'rien', 'personne', 'quelque', 'quelques', 'chaque', 'même', 'autre',
    'autres', 'dans', 'sur', 'sous', 'avec', 'sans', 'pour', 'par', 'entre',
    'vers', 'chez', 'avant', 'après', 'depuis', 'pendant', 'comme', 'si',
    'quand', 'lorsque', 'parce', 'puisque', 'ainsi', 'alors', 'donc',
    'être', 'avoir', 'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir',
    'vouloir', 'devoir', 'falloir', 'venir', 'prendre', 'mettre', 'partir',
    'est', 'sont', 'était', 'été', 'ai', 'as', 'a', 'avons', 'avez', 'ont',
    'suis', 'es', 'sommes', 'êtes', 'fait', 'fais', 'font', 'va', 'vais',
    'vont', 'dit', 'dis', 'peut', 'peux', 'peuvent', 'veut', 'veux', 'veulent',
    'doit', 'dois', 'doivent', 'faut', 'vient', 'viens', 'viennent',
    'c', 'd', 'j', 'l', 'm', 'n', 's', 't', 'qu', 'jusqu', 'lorsqu',
    'aujourd', 'hui', 'là', 'ça', 'cela', 'ceci', 'celui', 'celle', 'ceux',
    'oh', 'ah', 'eh', 'hé', 'ouais', 'yeah', 'yo', 'hey', 'ok', 'okay',
    'nan', 'non', 'oui', 'bah', 'ben', 'hein', 'quoi', 'genre',
    'moi', 'toi', 'soi', 'eux',
    'j\'ai', 'j\'suis', 't\'as', 't\'es', 'c\'est', 'qu\'est', 'n\'est',
    'y\'a', 'd\'un', 'd\'une', 'l\'on', 's\'en', 'm\'en', 't\'en',
}

ADDITIONAL_STOPWORDS = {
    # Filler words and interjections
    'wesh', 'gros', 'frère', 'mec', 'poto', 'igo', 'izi', 'baby',
    'wouh', 'aïe', 'pah', 'pam', 'boum', 'clic', 'clac',
    'la', 'ra', 'ta', 'da', 'na', 'pa', 'ma', 'wa', 'ya', 'za',
    'uh', 'huh', 'mm', 'hmm', 'ooh', 'aah', 'ah', 'oh', 'han', 'han_han',
    'ouai', 'ouais', 'ouai_ouai', 'yeah', 'yeh', 'yah', 'woh', 'wow', 'mmh',
    # Common contractions/abbreviations
    'y\'', 'y', 'ca', 'ça', 'ter', 'tise', 'lala', 'vien',
    # Lemmatization artifacts (from analysis of bad topics)
    'fai', 'taire', 'laisse', 'attend', 'entend', 'ouer', 'luire',
    '-ce', 'pler', 'ider', 'b\'_soin', 'v\'_nu', 'p\'_tit', 'sai',
    # Too common verbs (extended)
    'aller', 'venir', 'mettre', 'prendre', 'donner', 'laisser',
    'passer', 'rester', 'tenir', 'sortir', 'partir', 'arriver',
    'tomber', 'chercher', 'trouver', 'croire', 'penser', 'sentir',
    'aimer', 'parler', 'regarder', 'appeler', 'attendre', 'devenir',
    'finir', 'perdre', 'oublier', 'mourir', 'vivre', 'entrer',
    'rendre', 'changer',
    # Too generic nouns (from catch-all topics 9 and 16)
    'vie', 'temps', 'monde', 'jour', 'seul', 'cœur', 'mort', 'rêve',
    'gens', 'amour', 'reste', 'loin', 'homme', 'chose', 'peur',
    'fond', 'histoire', 'terre', 'tant', 'peine', 'mieux',
    'jamais', 'toujours', 'petit', 'fois', 'juste', 'encore',
    'heure', 'grand', 'soir', 'nuit', 'main', 'maintenant',
    'personn', 'déjà', 'devant', 'fort', 'bel', 'tour', 'porte', 'femme',
    # English words (extended - very common in French rap)
    'the', 'of', 'it', 'to', 'my', 'is', 'and', 'you', 'me', 'we',
    'in', 'on', 'for', 'with', 'that', 'this', 'be', 'are', 'was',
    'bitch', 'money', 'no', 'all', 'girl', 'get', 'real', 'what',
    'boy', 'big', 'weed', 'do', 'one', 'bad', 'go', 'up', 'so',
    'life', 'love', 'high', 'fuck', 'nigga', 'game', 'shit',
    'out', 'got', 'from', 'way', 'new', 'like', 'let', 'know',
    'back', 'man', 'hood', 'gang', 'cash', 'boss', 'gun', 'smoke',
    'flow', 'beat', 'rap', 'mic', 'dj', 'club', 'party',
}

ALL_STOPWORDS = FRENCH_STOPWORDS | ADDITIONAL_STOPWORDS


def load_data(path: str, sample_size: int = None) -> pd.DataFrame:
    """Load the dataset, optionally sampling."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} verses")

    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} documents for testing...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"Columns: {df.columns.tolist()}")
    print(f"Years range: {df['year'].min()} - {df['year'].max()}")
    print(f"Number of unique artists: {df['artist'].nunique()}")
    return df


def tokenize_simple(text: str, min_word_len: int = 2) -> list:
    """
    Simple tokenization without lemmatization.
    Better for slang/verlan in French rap.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []

    # Lowercase and clean
    text = text.lower()
    # Keep letters, accents, apostrophes, hyphens
    text = re.sub(r'[^\w\sàâäéèêëïîôùûüÿœæç\'-]', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into tokens
    tokens = text.split()

    # Filter tokens
    filtered = []
    for token in tokens:
        # Remove leading/trailing punctuation
        token = token.strip("'-")
        if (len(token) >= min_word_len and
            token not in ALL_STOPWORDS and
            not token.isdigit()):
            filtered.append(token)

    return filtered


def get_corpus_path(base_dir: str, ngram_mode: str) -> str:
    """Get the corpus path for a specific n-gram mode."""
    return os.path.join(base_dir, f'corpus_{ngram_mode}.pkl')


def create_corpus(df: pd.DataFrame, text_column: str = 'lyrics_cleaned',
                  min_word_len: int = 2, min_doc_freq: int = 5,
                  max_doc_freq_ratio: float = 0.3,
                  use_ngrams: str = 'both',
                  ngram_min_count: int = 10,
                  ngram_threshold: int = 50,
                  save_path: str = None) -> tuple:
    """
    Create corpus for LDA from dataframe.

    Args:
        use_ngrams: N-gram mode:
            - 'unigrams': no n-grams, just single words
            - 'bigrams': unigrams + bigrams
            - 'trigrams': unigrams + trigrams
            - 'both': unigrams + bigrams + trigrams
            - 'ngrams_only': bigrams + trigrams (no unigrams)
            - 'bigram_only': only bigrams (no unigrams, no trigrams)
            - 'trigram_only': only trigrams (no unigrams, no bigrams)

    Returns: (texts, dictionary, corpus, df_filtered)
    """
    print("\n" + "="*60)
    print("PREPROCESSING CORPUS")
    print("="*60)
    print(f"N-gram mode: {use_ngrams}")

    # Simple tokenization (no spaCy, no lemmatization)
    print(f"Tokenizing {len(df)} documents (no lemmatization, preserving slang)...")
    raw_texts = df[text_column].fillna("").tolist()

    texts = []
    valid_indices = []

    for idx, text in enumerate(raw_texts):
        tokens = tokenize_simple(text, min_word_len)
        if len(tokens) >= 3:
            texts.append(tokens)
            valid_indices.append(idx)
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(raw_texts)} documents...")

    df_filtered = df.iloc[valid_indices].copy()
    # Store original indices before resetting
    df_filtered['original_index'] = valid_indices
    df_filtered = df_filtered.reset_index(drop=True)
    print(f"Documents after filtering: {len(texts)}")

    # Create n-grams based on mode
    # Modes that need bigrams: bigrams, both, ngrams_only, bigram_only
    if use_ngrams in ['bigrams', 'both', 'ngrams_only', 'bigram_only']:
        print(f"Creating bigrams (min_count={ngram_min_count}, threshold={ngram_threshold})...")
        bigram = Phrases(texts, min_count=ngram_min_count, threshold=ngram_threshold)
        bigram_mod = Phraser(bigram)
        texts = [bigram_mod[doc] for doc in texts]

    # Modes that need trigrams: trigrams, both, ngrams_only, trigram_only
    # Note: trigram_only needs bigrams first to build trigrams
    if use_ngrams in ['trigrams', 'both', 'ngrams_only', 'trigram_only']:
        # For trigram_only, we need bigrams first (to chain into trigrams)
        if use_ngrams == 'trigram_only':
            print(f"Creating bigrams first (for trigram chaining)...")
            bigram = Phrases(texts, min_count=ngram_min_count, threshold=ngram_threshold)
            bigram_mod = Phraser(bigram)
            texts = [bigram_mod[doc] for doc in texts]

        print(f"Creating trigrams (min_count={ngram_min_count}, threshold={ngram_threshold})...")
        trigram = Phrases(texts, min_count=ngram_min_count, threshold=ngram_threshold)
        trigram_mod = Phraser(trigram)
        texts = [trigram_mod[doc] for doc in texts]

    # Filter to only n-grams if requested
    if use_ngrams in ['ngrams_only', 'bigram_only', 'trigram_only']:
        print(f"Filtering to n-grams only (mode: {use_ngrams})...")
        texts_ngrams = []
        for doc in texts:
            if use_ngrams == 'bigram_only':
                # Keep only bigrams (exactly one '_')
                ngrams = [token for token in doc if token.count('_') == 1]
            elif use_ngrams == 'trigram_only':
                # Keep only trigrams (exactly two '_')
                ngrams = [token for token in doc if token.count('_') == 2]
            else:  # ngrams_only: both bigrams and trigrams
                ngrams = [token for token in doc if '_' in token]

            if len(ngrams) >= 2:  # Need at least 2 n-grams
                texts_ngrams.append(ngrams)
            else:
                texts_ngrams.append(doc)  # Keep original if not enough n-grams
        texts = texts_ngrams

    # Create dictionary
    print("Creating dictionary...")
    dictionary = corpora.Dictionary(texts)
    print(f"Initial vocabulary size: {len(dictionary)}")

    # Filter extremes
    dictionary.filter_extremes(no_below=min_doc_freq, no_above=max_doc_freq_ratio)
    print(f"Vocabulary after filtering: {len(dictionary)}")

    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Save if path provided
    if save_path:
        print(f"Saving corpus data to {save_path}...")
        corpus_data = {
            'texts': texts,
            'corpus': corpus,
            'df_filtered': df_filtered,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(corpus_data, f)
        dictionary.save(save_path.replace('.pkl', '.dict'))
        print("Corpus saved!")

    return texts, dictionary, corpus, df_filtered


def load_corpus(corpus_path: str) -> tuple:
    """Load pre-computed corpus from disk."""
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, 'rb') as f:
        corpus_data = pickle.load(f)

    dict_path = corpus_path.replace('.pkl', '.dict')
    dictionary = corpora.Dictionary.load(dict_path)

    print(f"Loaded {len(corpus_data['corpus'])} documents, vocabulary size: {len(dictionary)}")
    return corpus_data['texts'], dictionary, corpus_data['corpus'], corpus_data['df_filtered']


def train_lda(corpus, dictionary, num_topics: int = 20,
              alpha: str = 'symmetric', eta: str = 'auto',
              passes: int = 15, iterations: int = 400,
              random_state: int = 42, chunksize: int = 2000) -> LdaModel:
    """Train LDA model with given parameters."""
    print(f"\nTraining LDA with {num_topics} topics...")
    print(f"Parameters: alpha={alpha}, eta={eta}, passes={passes}, iterations={iterations}")

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha=alpha,
        eta=eta,
        passes=passes,
        iterations=iterations,
        random_state=random_state,
        chunksize=chunksize,
        per_word_topics=True
    )

    return lda_model


def compute_coherence_metrics(lda_model, texts, dictionary, corpus) -> dict:
    """Compute various coherence metrics."""
    print("\nComputing coherence metrics...")

    metrics = {}

    # C_V coherence (best for human interpretability)
    coherence_cv = CoherenceModel(
        model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v'
    )
    metrics['coherence_cv'] = coherence_cv.get_coherence()
    print(f"  C_V Coherence: {metrics['coherence_cv']:.4f}")

    # U_Mass coherence (faster, corpus-based)
    coherence_umass = CoherenceModel(
        model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass'
    )
    metrics['coherence_umass'] = coherence_umass.get_coherence()
    print(f"  U_Mass Coherence: {metrics['coherence_umass']:.4f}")

    # Per-topic coherence
    metrics['per_topic_cv'] = coherence_cv.get_coherence_per_topic()

    return metrics


def get_document_topics(lda_model, corpus) -> tuple:
    """
    Get topic distribution and dominant topic for each document.
    Returns: (doc_topics array, dominant_topics array)
    """
    num_topics = lda_model.num_topics
    doc_topics = np.zeros((len(corpus), num_topics))

    for i, doc in enumerate(corpus):
        topic_dist = lda_model.get_document_topics(doc, minimum_probability=0)
        for topic_id, prob in topic_dist:
            doc_topics[i, topic_id] = prob

    # Get dominant topic for each document (most probable)
    dominant_topics = doc_topics.argmax(axis=1)

    return doc_topics, dominant_topics


def display_topics(lda_model, num_words: int = 30) -> dict:
    """Display and return topic descriptions."""
    print("\n" + "="*60)
    print("TOPIC DESCRIPTIONS")
    print("="*60)

    topics = {}
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, num_words)
        words = [word for word, prob in topic_words]
        probs = [float(prob) for word, prob in topic_words]

        topics[topic_id] = {
            'words': words,
            'probabilities': probs,
            'top_words': ', '.join(words[:15])
        }

        print(f"\nTopic {topic_id}: {', '.join(words[:15])}")

    return topics


def create_pyldavis(lda_model, corpus, dictionary, run_dir: str):
    """Create pyLDAvis HTML visualization."""
    print("\nCreating pyLDAvis visualization...")
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models

        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        vis_path = os.path.join(run_dir, 'pyldavis.html')
        pyLDAvis.save_html(vis_data, vis_path)
        print(f"  pyLDAvis saved to: {vis_path}")
        return vis_path
    except Exception as e:
        print(f"  Could not create pyLDAvis: {e}")
        return None


def save_results(results: dict, lda_model, dictionary, corpus, df, doc_topics, dominant_topics,
                 run_dir: str, create_pyldavis_html: bool = True):
    """Save all results and model artifacts to run directory."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save model
    model_path = os.path.join(run_dir, "lda_model")
    lda_model.save(model_path)
    print(f"  Model saved to: {model_path}")

    # Save dictionary
    dict_path = os.path.join(run_dir, "dictionary.dict")
    dictionary.save(dict_path)
    print(f"  Dictionary saved to: {dict_path}")

    # Save document-topic matrix and dominant topics
    doc_topics_path = os.path.join(run_dir, "doc_topics.npy")
    np.save(doc_topics_path, doc_topics)
    np.save(os.path.join(run_dir, "dominant_topics.npy"), dominant_topics)
    print(f"  Document-topics saved to: {doc_topics_path}")

    # Save metrics as JSON
    metrics_to_save = {
        'coherence_metrics': {
            'cv': results['coherence']['coherence_cv'],
            'umass': results['coherence']['coherence_umass'],
            'per_topic_cv': [float(x) for x in results['coherence']['per_topic_cv']],
        },
        'artist_metrics': {
            'silhouette': results['artist_separation'].get('silhouette_score'),
            'js_divergence': results['artist_separation']['mean_js_divergence'],
            'specialization': results['artist_separation']['artist_specialization'],
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

    # Save topic descriptions (with 30 words)
    topics_path = os.path.join(run_dir, "topics.json")
    topics_to_save = {}
    for tid, tdata in results['topics'].items():
        topics_to_save[str(tid)] = {
            'words': tdata['words'],
            'probabilities': tdata['probabilities'],
            'top_words': tdata['top_words'],
        }
    with open(topics_path, 'w', encoding='utf-8') as f:
        json.dump(topics_to_save, f, indent=2, ensure_ascii=False)
    print(f"  Topics saved to: {topics_path}")

    # Save topic evolution and artist metrics using shared utilities
    save_temporal_metrics(results['temporal_separation'], run_dir)
    save_artist_metrics(results['artist_separation'], run_dir)

    # Save document assignments (including original_index for alignment with other models)
    doc_assign_path = os.path.join(run_dir, "doc_assignments.csv")
    doc_df = df.copy()
    doc_df['dominant_topic'] = dominant_topics
    doc_df['dominant_topic_prob'] = doc_topics.max(axis=1)
    # Include original_index if available (for proper alignment with BERTopic/IRAMUTEQ)
    cols_to_save = ['artist', 'title', 'year', 'dominant_topic', 'dominant_topic_prob']
    if 'original_index' in doc_df.columns:
        cols_to_save.insert(0, 'original_index')
    doc_df[cols_to_save].to_csv(doc_assign_path, index=False)
    print(f"  Document assignments saved to: {doc_assign_path}")

    # Create pyLDAvis if requested
    if create_pyldavis_html:
        create_pyldavis(lda_model, corpus, dictionary, run_dir)

    print(f"\n  All results saved to: {run_dir}")


def create_visualizations(results: dict, doc_topics: np.ndarray, dominant_topics: np.ndarray,
                          df: pd.DataFrame, run_dir: str, top_n_artists: int = 50):
    """Create and save visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # 1. Topic coherence bar plot (LDA-specific)
    fig, ax = plt.subplots(figsize=(12, 6))
    topic_ids = range(len(results['coherence']['per_topic_cv']))
    coherences = results['coherence']['per_topic_cv']
    colors = ['green' if c > 0.4 else 'orange' if c > 0.3 else 'red' for c in coherences]
    ax.bar(topic_ids, coherences, color=colors)
    ax.axhline(y=0.4, color='green', linestyle='--', label='Good threshold')
    ax.axhline(y=0.3, color='orange', linestyle='--', label='Acceptable threshold')
    ax.set_xlabel('Topic ID')
    ax.set_ylabel('C_V Coherence')
    ax.set_title('Topic Coherence Scores')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'coherence_plot.png'), dpi=150)
    plt.close()
    print("  Saved coherence plot")

    # 2. Dominant topic distribution (using shared utility)
    create_topic_distribution_plot(dominant_topics, run_dir, title="LDA Dominant Topic Distribution")

    # 3. Year-topic heatmap (shows topic distribution over time, using shared utility)
    create_year_topic_heatmap(dominant_topics, df, run_dir,
                               title="LDA Topic Distribution by Year")

    # 5. Topic distribution in PCA space (LDA-specific)
    if doc_topics.shape[0] > 100:
        pca = PCA(n_components=2)
        doc_pca = pca.fit_transform(doc_topics)

        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(doc_pca[:, 0], doc_pca[:, 1],
                           c=dominant_topics, cmap='tab20', alpha=0.5, s=5)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.set_title('Documents in Topic Space (PCA) - Colored by Dominant Topic')
        plt.colorbar(scatter, ax=ax, label='Dominant Topic')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'topic_pca.png'), dpi=150)
        plt.close()
        print("  Saved PCA visualization")

    # 6. Artist topic profiles heatmap (LDA-specific: uses probability distributions)
    valid_artists = df['artist'].value_counts().head(top_n_artists).index.tolist()
    artist_profiles = []
    for artist in valid_artists:
        mask = df['artist'] == artist
        profile = doc_topics[mask.values].mean(axis=0)
        artist_profiles.append(profile)

    artist_profiles = np.array(artist_profiles)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(artist_profiles, cmap='YlOrRd', ax=ax,
                yticklabels=valid_artists, xticklabels=range(doc_topics.shape[1]))
    ax.set_xlabel('Topic ID')
    ax.set_ylabel('Artist')
    ax.set_title(f'Top {top_n_artists} Artists - LDA Topic Profiles (Probability)')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'artist_topics_heatmap.png'), dpi=150)
    plt.close()
    print(f"  Saved artist topic profiles (top {top_n_artists})")

    # 7. Artist specialization plot (using shared utility)
    create_artist_specialization_plot(results['artist_separation'], run_dir)

    # 8. Biannual JS divergence plot (using shared utility)
    create_biannual_js_plot(results['temporal_separation'], run_dir,
                            title="LDA - 2-Year Window JS Divergence")

    print("  All visualizations saved!")


def print_summary(results: dict):
    """Print a summary of all metrics."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    print("\n[COHERENCE METRICS]")
    print(f"   C_V Coherence: {results['coherence']['coherence_cv']:.4f}")
    print(f"   U_Mass Coherence: {results['coherence']['coherence_umass']:.4f}")

    print("\n[ARTIST SEPARATION]")
    if results['artist_separation'].get('silhouette_score'):
        print(f"   Silhouette Score: {results['artist_separation']['silhouette_score']:.4f}")
    print(f"   Mean JS Divergence: {results['artist_separation']['mean_js_divergence']:.4f}")
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


def run_experiment(num_topics: int = 20, alpha: str = 'symmetric', eta: str = 'auto',
                   passes: int = 15, iterations: int = 400,
                   min_word_len: int = 2, min_doc_freq: int = 5,
                   max_doc_freq_ratio: float = 0.3,
                   use_ngrams: str = 'both',
                   ngram_min_count: int = 10, ngram_threshold: int = 50,
                   sample_size: int = None, corpus_path: str = None,
                   save_corpus: bool = True, create_pyldavis_html: bool = True,
                   num_words_per_topic: int = 30, top_artists_per_topic: int = 20,
                   top_n_artists_heatmap: int = 50):
    """
    Run a complete LDA experiment with given parameters.

    Args:
        use_ngrams: N-gram mode - 'unigrams', 'bigrams', 'trigrams', 'both',
                    'ngrams_only', 'bigram_only', 'trigram_only'
    """

    print("\n" + "="*60)
    print("LDA EXPERIMENT FOR FRENCH RAP CORPUS")
    print("="*60)

    # Create timestamped run directory with n-gram mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}_{use_ngrams}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")
    print(f"N-gram mode: {use_ngrams}")

    # Determine corpus path based on n-gram mode
    mode_corpus_path = get_corpus_path(RESULTS_DIR, use_ngrams)

    # Load or create corpus
    if corpus_path and os.path.exists(corpus_path):
        # User specified a specific corpus path
        print(f"Loading user-specified corpus: {corpus_path}")
        texts, dictionary, corpus, df_filtered = load_corpus(corpus_path)
    elif os.path.exists(mode_corpus_path) and sample_size is None:
        # Auto-detect: corpus for this mode already exists (full dataset only)
        print(f"Found existing corpus for mode '{use_ngrams}': {mode_corpus_path}")
        texts, dictionary, corpus, df_filtered = load_corpus(mode_corpus_path)
    else:
        # Create new corpus
        df = load_data(DATA_PATH, sample_size=sample_size)
        # Only save corpus if not sampling (full dataset)
        corpus_save_path = mode_corpus_path if (save_corpus and sample_size is None) else None
        texts, dictionary, corpus, df_filtered = create_corpus(
            df,
            min_word_len=min_word_len,
            min_doc_freq=min_doc_freq,
            max_doc_freq_ratio=max_doc_freq_ratio,
            use_ngrams=use_ngrams,
            ngram_min_count=ngram_min_count,
            ngram_threshold=ngram_threshold,
            save_path=corpus_save_path
        )

    # Train LDA
    lda_model = train_lda(
        corpus, dictionary,
        num_topics=num_topics,
        alpha=alpha,
        eta=eta,
        passes=passes,
        iterations=iterations
    )

    # Get document-topic distributions and dominant topics
    doc_topics, dominant_topics = get_document_topics(lda_model, corpus)

    # Compute all metrics
    coherence_metrics = compute_coherence_metrics(lda_model, texts, dictionary, corpus)
    # Use shared functions with doc_topics for probability-based profiles
    artist_metrics = compute_artist_separation(dominant_topics, df_filtered,
                                               top_artists_per_topic=top_artists_per_topic,
                                               doc_topics=doc_topics)
    temporal_metrics = compute_temporal_separation(dominant_topics, df_filtered,
                                                   doc_topics=doc_topics)

    # Display topics
    topics = display_topics(lda_model, num_words=num_words_per_topic)

    # Compile results
    results = {
        'coherence': coherence_metrics,
        'artist_separation': artist_metrics,
        'temporal_separation': temporal_metrics,
        'topics': topics,
        'parameters': {
            'num_topics': num_topics,
            'alpha': str(alpha),
            'eta': str(eta),
            'passes': passes,
            'iterations': iterations,
            'min_word_len': min_word_len,
            'min_doc_freq': min_doc_freq,
            'max_doc_freq_ratio': max_doc_freq_ratio,
            'use_ngrams': use_ngrams,
            'ngram_min_count': ngram_min_count,
            'ngram_threshold': ngram_threshold,
            'vocabulary_size': len(dictionary),
            'num_documents': len(corpus),
            'num_words_per_topic': num_words_per_topic,
            'timestamp': timestamp,
            'run_dir': run_dir,
        }
    }

    # Print summary
    print_summary(results)

    # Save everything
    save_results(results, lda_model, dictionary, corpus, df_filtered, doc_topics, dominant_topics,
                 run_dir=run_dir, create_pyldavis_html=create_pyldavis_html)

    # Create visualizations
    create_visualizations(results, doc_topics, dominant_topics, df_filtered,
                          run_dir=run_dir, top_n_artists=top_n_artists_heatmap)

    return results, lda_model, doc_topics, dominant_topics, df_filtered


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build and evaluate LDA model for French rap corpus')
    parser.add_argument('--topics', type=int, default=20, help='Number of topics')
    parser.add_argument('--passes', type=int, default=15, help='Number of passes')
    parser.add_argument('--iterations', type=int, default=400, help='Number of iterations')
    parser.add_argument('--alpha', type=str, default='symmetric', help='Alpha parameter (symmetric for balanced topics)')
    parser.add_argument('--eta', type=str, default='auto', help='Eta parameter')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for testing')
    parser.add_argument('--load-corpus', type=str, default=None, help='Path to load pre-computed corpus')
    parser.add_argument('--no-save-corpus', action='store_true', help='Do not save corpus')
    parser.add_argument('--no-pyldavis', action='store_true', help='Do not create pyLDAvis visualization')
    parser.add_argument('--num-words', type=int, default=30, help='Number of words per topic')
    parser.add_argument('--top-artists-topic', type=int, default=20, help='Top N artists per topic')
    parser.add_argument('--top-artists-heatmap', type=int, default=50, help='Top N artists in heatmap')

    # N-gram options
    parser.add_argument('--ngrams', type=str, default='both',
                        choices=['unigrams', 'bigrams', 'trigrams', 'both', 'ngrams_only', 'bigram_only', 'trigram_only'],
                        help='N-gram mode: unigrams, bigrams (uni+bi), trigrams (uni+tri), both (uni+bi+tri), ngrams_only (bi+tri), bigram_only (only bigrams), trigram_only (only trigrams)')
    parser.add_argument('--ngram-min-count', type=int, default=10, help='Min count for n-gram detection')
    parser.add_argument('--ngram-threshold', type=int, default=50, help='Threshold for n-gram detection')

    args = parser.parse_args()

    # Convert alpha/eta if numeric
    alpha = float(args.alpha) if args.alpha.replace('.', '').isdigit() else args.alpha
    eta = float(args.eta) if args.eta.replace('.', '').isdigit() else args.eta

    results, model, doc_topics, dominant_topics, df = run_experiment(
        num_topics=args.topics,
        alpha=alpha,
        eta=eta,
        passes=args.passes,
        iterations=args.iterations,
        sample_size=args.sample,
        corpus_path=args.load_corpus,
        save_corpus=not args.no_save_corpus,
        create_pyldavis_html=not args.no_pyldavis,
        num_words_per_topic=args.num_words,
        top_artists_per_topic=args.top_artists_topic,
        top_n_artists_heatmap=args.top_artists_heatmap,
        use_ngrams=args.ngrams,
        ngram_min_count=args.ngram_min_count,
        ngram_threshold=args.ngram_threshold,
    )
