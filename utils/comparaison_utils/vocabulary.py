#!/usr/bin/env python3
"""
Vocabulary Analysis (Q4)
========================
Functions for analyzing vocabulary overlap and distinctiveness.
"""

import numpy as np
from typing import List


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
