#!/usr/bin/env python3
"""
Vocabulary Analysis (Q4)
========================
Functions for analyzing vocabulary overlap and distinctiveness.
"""

import numpy as np
from typing import List


def build_topic_labels(topics: dict, model_type: str) -> dict:
    """
    Build human-readable topic labels from topics.json data.

    Naming convention:
    - BERTopic: "T{id}: {openai_label}" (fallback: top 5 keybert words)
    - LDA: "T{id}: {top5_words}"
    - IRAMUTEQ: "C{id}: {top5_words}"

    Parameters
    ----------
    topics : dict
        Topics data loaded from topics.json ({topic_id_str: topic_data}).
    model_type : str
        'bertopic', 'lda', or 'iramuteq'.

    Returns
    -------
    dict
        {topic_id (int or str): label_string}
    """
    labels = {}
    prefix = 'C' if model_type == 'iramuteq' else 'T'

    for tid, topic_data in topics.items():
        if not isinstance(topic_data, dict):
            labels[tid] = f"{prefix}{tid}"
            continue

        if model_type == 'bertopic':
            # Prefer OpenAI label
            openai = topic_data.get('openai', [])
            if openai and isinstance(openai, list) and openai[0]:
                label = openai[0].strip('"').strip()
                labels[tid] = f"T{tid}: {label}"
                continue
            # Fallback to keybert top 5
            keybert = topic_data.get('keybert', [])
            if keybert:
                words = ', '.join(keybert[:5])
                labels[tid] = f"T{tid}: {words}"
                continue

        # LDA / IRAMUTEQ: top 5 words
        words = extract_topic_words(topic_data, top_n=5)
        if words:
            labels[tid] = f"{prefix}{tid}: {', '.join(words)}"
        else:
            labels[tid] = f"{prefix}{tid}"

    return labels


def extract_topic_words(topic_data: dict, top_n: int = 30) -> List[str]:
    """
    Extract words from a topic data structure.

    Handles different formats:
    - BERTopic: Combines ctfidf.words, mmr, keybert for more complete coverage
    - LDA: words list or top_words (space-separated string)
    - IRAMUTEQ: words list

    Returns unique words, preserving order from primary source.
    """
    if not isinstance(topic_data, dict):
        return []

    words = []
    seen = set()

    def add_words(word_list):
        for w in word_list:
            if w not in seen:
                words.append(w)
                seen.add(w)

    # For BERTopic: combine multiple word sources
    if 'ctfidf' in topic_data:
        ctfidf = topic_data['ctfidf']
        if isinstance(ctfidf, dict) and 'words' in ctfidf:
            add_words(ctfidf['words'])

        # Also add mmr and keybert for more coverage
        if 'mmr' in topic_data and isinstance(topic_data['mmr'], list):
            add_words(topic_data['mmr'])
        if 'keybert' in topic_data and isinstance(topic_data['keybert'], list):
            add_words(topic_data['keybert'])

        if words:
            return words[:top_n]

    # Try words list (LDA or IRAMUTEQ)
    if 'words' in topic_data and isinstance(topic_data['words'], list):
        return topic_data['words'][:top_n]

    # Try top_words string (LDA or fallback)
    if 'top_words' in topic_data:
        top_words = topic_data['top_words']
        if isinstance(top_words, str):
            # Could be comma-separated or space-separated
            if ',' in top_words:
                return [w.strip() for w in top_words.split(',')][:top_n]
            else:
                return top_words.split()[:top_n]
        elif isinstance(top_words, list):
            return top_words[:top_n]

    return []


def compute_full_vocab_jaccard(
    doc_tokens: List[List[str]],
    topic_assignments: List[int],
    min_freq: int = 5,
) -> dict:
    """
    Compute pairwise Jaccard similarity between topics using full vocabulary.

    For each topic, builds the set of words appearing ≥ min_freq times
    across its documents, then computes pairwise Jaccard between all topics.

    Parameters
    ----------
    doc_tokens : List[List[str]]
        Pre-tokenized documents.
    topic_assignments : List[int]
        Topic assignment per document (-1 = outliers, excluded).
    min_freq : int, default=5
        Minimum word frequency within a topic to be included.

    Returns
    -------
    dict
        {
            'mean_jaccard': float,
            'per_pair': {(tid1, tid2): jaccard},
            'vocab_sizes': {tid: vocab_size},
        }
    """
    from collections import Counter

    # Build per-topic word counters
    topic_counters: Dict[int, Counter] = {}
    for tokens, topic in zip(doc_tokens, topic_assignments):
        if topic == -1:
            continue
        if topic not in topic_counters:
            topic_counters[topic] = Counter()
        topic_counters[topic].update(tokens)

    # Filter by min_freq to get per-topic vocabulary sets
    topic_vocabs: Dict[int, set] = {}
    for tid, counter in topic_counters.items():
        topic_vocabs[tid] = {w for w, c in counter.items() if c >= min_freq}

    topic_ids = sorted(topic_vocabs.keys())
    vocab_sizes = {tid: len(topic_vocabs[tid]) for tid in topic_ids}

    # Pairwise Jaccard
    per_pair = {}
    jaccard_values = []
    for i, tid1 in enumerate(topic_ids):
        for tid2 in topic_ids[i + 1:]:
            inter = len(topic_vocabs[tid1] & topic_vocabs[tid2])
            union = len(topic_vocabs[tid1] | topic_vocabs[tid2])
            jacc = inter / union if union > 0 else 0.0
            per_pair[(tid1, tid2)] = jacc
            jaccard_values.append(jacc)

    import numpy as np
    return {
        'mean_jaccard': float(np.mean(jaccard_values)) if jaccard_values else 0.0,
        'std_jaccard': float(np.std(jaccard_values)) if jaccard_values else 0.0,
        'per_pair': per_pair,
        'vocab_sizes': vocab_sizes,
        'min_freq': min_freq,
        'n_topics': len(topic_ids),
    }


def compute_cross_model_full_vocab_jaccard(
    doc_tokens: List[List[str]],
    topics_a: List[int],
    topics_b: List[int],
    correspondences: List[dict],
    model_a_name: str = 'bertopic',
    model_b_name: str = 'lda',
    min_freq_thresholds: List[int] = None,
) -> dict:
    """
    Compute full-vocabulary Jaccard between corresponding topics of two models.

    For each correspondence pair (topic_a, topic_b), builds the vocabulary
    from all documents assigned to that topic, then computes Jaccard.

    Parameters
    ----------
    doc_tokens : List[List[str]]
        Pre-tokenized documents (same corpus for both models).
    topics_a : List[int]
        Topic assignments from model A.
    topics_b : List[int]
        Topic assignments from model B.
    correspondences : List[dict]
        Topic correspondences from contingency table.
    model_a_name, model_b_name : str
        Model names for key lookup in correspondences.
    min_freq_thresholds : List[int], optional
        Frequency thresholds to compute Jaccard at. Default: [1, 5, 20].

    Returns
    -------
    dict
        {
            'per_threshold': {min_freq: {'mean_jaccard': float, 'per_pair': [...]}},
        }
    """
    from collections import Counter
    import numpy as np

    if min_freq_thresholds is None:
        min_freq_thresholds = [1, 5, 20]

    # Build per-topic word counters for each model
    def _build_counters(assignments):
        counters = {}
        for tokens, topic in zip(doc_tokens, assignments):
            if topic == -1:
                continue
            if topic not in counters:
                counters[topic] = Counter()
            counters[topic].update(tokens)
        return counters

    counters_a = _build_counters(topics_a)
    counters_b = _build_counters(topics_b)

    # Build correspondence pairs
    key_a = f'{model_a_name.capitalize()}_topic'
    key_a_lower = f'{model_a_name}_topic'
    key_b = f'{model_b_name.capitalize()}_topic'
    key_b_lower = f'{model_b_name}_topic'

    pairs = []
    for corr in correspondences:
        tid_a = corr.get(key_a, corr.get(key_a_lower, corr.get('BERTopic_topic', '')))
        tid_b = corr.get(key_b, corr.get(key_b_lower, corr.get('LDA_topic', '')))
        # Convert to int if possible
        try:
            tid_a = int(tid_a)
        except (ValueError, TypeError):
            pass
        try:
            tid_b = int(tid_b)
        except (ValueError, TypeError):
            pass
        if tid_a in counters_a and tid_b in counters_b:
            pairs.append((tid_a, tid_b))

    results = {'per_threshold': {}}

    for min_freq in min_freq_thresholds:
        per_pair = []
        jaccard_values = []

        for tid_a, tid_b in pairs:
            vocab_a = {w for w, c in counters_a[tid_a].items() if c >= min_freq}
            vocab_b = {w for w, c in counters_b[tid_b].items() if c >= min_freq}
            inter = len(vocab_a & vocab_b)
            union = len(vocab_a | vocab_b)
            jacc = inter / union if union > 0 else 0.0
            per_pair.append({
                f'{model_a_name}_topic': tid_a,
                f'{model_b_name}_topic': tid_b,
                'jaccard': jacc,
                'vocab_a_size': len(vocab_a),
                'vocab_b_size': len(vocab_b),
                'intersection': inter,
            })
            jaccard_values.append(jacc)

        results['per_threshold'][min_freq] = {
            'mean_jaccard': float(np.mean(jaccard_values)) if jaccard_values else 0.0,
            'per_pair': per_pair,
            'n_pairs': len(pairs),
        }

    return results


def compute_vocabulary_overlap(vocab1: List[str], vocab2: List[str]) -> dict:
    """
    Compute vocabulary overlap between two word lists.
    """
    set1 = set(vocab1)
    set2 = set(vocab2)

    intersection = set1 & set2
    union = set1 | set2

    jaccard = len(intersection) / len(union) if union else 0
    overlap_coef = len(intersection) / min(len(set1), len(set2)) if set1 and set2 else 0

    return {
        'jaccard': round(jaccard, 4),
        'overlap_coefficient': round(overlap_coef, 4),
        'n_common': len(intersection),
        'common_words': list(intersection)[:20],
        'unique_to_first': list(set1 - set2)[:10],
        'unique_to_second': list(set2 - set1)[:10]
    }


def compute_vocabulary_distinctiveness(topics_vocab: dict) -> float:
    """
    Measure how distinct topics are from each other lexically.

    Computes mean pairwise Jaccard distance between topic vocabularies.
    """
    topic_ids = list(topics_vocab.keys())
    if len(topic_ids) < 2:
        return 0.0

    distances = []
    for i, t1 in enumerate(topic_ids):
        for t2 in topic_ids[i + 1:]:
            words1 = extract_topic_words(topics_vocab[t1], top_n=30)
            words2 = extract_topic_words(topics_vocab[t2], top_n=30)

            if not words1 or not words2:
                continue

            set1, set2 = set(words1), set(words2)
            jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
            distances.append(1 - jaccard)  # Distance = 1 - similarity

    return np.mean(distances) if distances else 0.0


def compare_topic_vocabularies(bertopic_topics: dict, lda_topics: dict,
                                correspondences: List[dict], top_n: int = 30) -> dict:
    """
    Compare vocabulary between corresponding topics across models.
    """
    results = []

    for corr in correspondences:
        bert_topic = str(corr.get('BERTopic_topic', corr.get('bertopic_topic', '')))
        lda_topic = str(corr.get('LDA_topic', corr.get('lda_topic', '')))

        bert_words = []
        lda_words = []

        # Get BERTopic words
        if bert_topic in bertopic_topics:
            bert_words = extract_topic_words(bertopic_topics[bert_topic], top_n=top_n)

        # Get LDA words
        if lda_topic in lda_topics:
            lda_words = extract_topic_words(lda_topics[lda_topic], top_n=top_n)

        if bert_words and lda_words:
            overlap = compute_vocabulary_overlap(bert_words, lda_words)
            results.append({
                'bertopic_topic': bert_topic,
                'lda_topic': lda_topic,
                **overlap
            })

    return {
        'topic_comparisons': results,
        'mean_jaccard': np.mean([r['jaccard'] for r in results]) if results else 0,
        'mean_overlap_coef': np.mean([r['overlap_coefficient'] for r in results]) if results else 0
    }
