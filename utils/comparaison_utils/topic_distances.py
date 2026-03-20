#!/usr/bin/env python3
"""
Topic Distance Metrics (Q5)
===========================
Module for computing intra-topic distances to evaluate cluster homogeneity.

This module provides distance metrics to assess how lexically homogeneous
documents within the same topic are. Lower intra-topic distances indicate
more coherent topic assignments.

Classes:
--------
- BaseDistance: Abstract base class for distance metrics
- LabbeDistance: Labbé intertextual distance (relative frequencies)
- JensenShannonDistance: Jensen-Shannon divergence (distributional)

Functions:
----------
- evaluate_topic_distances: Comprehensive topic distance evaluation (4 modes)
- aggregate_documents: Merge multiple documents into larger units

References:
-----------
- Labbé, C., & Labbé, D. (2001). Inter-textual distance and authorship attribution.
  Journal of Quantitative Linguistics, 8(3), 213-231.
- Lin, J. (1991). Divergence measures based on the Shannon entropy.
  IEEE Transactions on Information Theory, 37(1), 145-151.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple, Callable
import random
import itertools

import numpy as np
from scipy.spatial.distance import jensenshannon


class BaseDistance(ABC):
    """
    Abstract base class for text distance metrics.

    All distance implementations must inherit from this class and implement
    the `compute` method.

    A distance metric should return:
    - 0 for identical texts
    - Positive values for dissimilar texts
    - Higher values indicate greater dissimilarity
    """

    @abstractmethod
    def compute(self, text1: str, text2: str) -> float:
        """
        Compute distance between two texts.

        Parameters
        ----------
        text1 : str
            First text document.
        text2 : str
            Second text document.

        Returns
        -------
        float
            Distance value >= 0. Lower values indicate more similar texts.
        """
        pass

    def compute_from_counts(self, counts1: Counter, counts2: Counter) -> float:
        """
        Compute distance from pre-computed word counts (optimized).

        Parameters
        ----------
        counts1 : Counter
            Word counts for first document.
        counts2 : Counter
            Word counts for second document.

        Returns
        -------
        float
            Distance value >= 0.
        """
        # Default implementation: fall back to text-based if not overridden
        raise NotImplementedError("Subclass should implement compute_from_counts for optimization")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LabbeDistance(BaseDistance):
    """
    Labbé intertextual distance.

    Measures lexical similarity between two texts based on word frequencies,
    using the original algorithm from IRAMUTEQ/ALCESTE. Standard metric in
    French stylometry and JADT community.

    Algorithm (from IRAMUTEQ R implementation):
        1. Identify the smaller text (N_small) and larger text (N_large)
        2. Scale the larger text's counts by U = N_small / N_large
        3. Compute sum of absolute differences on scaled counts
        4. Normalize: D = sum_diff / (N_small + sum(scaled_large where >= 1))

    The distance is bounded [0, 1]:
    - 0: Identical word distributions
    - 1: No vocabulary overlap

    References
    ----------
    - Labbé, C., & Labbé, D. (2001). Inter-textual distance and authorship
      attribution. Journal of Quantitative Linguistics, 8(3), 213-231.
    - Labbé, D., & Monière, D. (2003). Le vocabulaire gouvernemental:
      Canada, Québec, France (1945-2000). Champion.
    - IRAMUTEQ implementation: gitlab.huma-num.fr/pratinaud/iramuteq

    Parameters
    ----------
    lowercase : bool, default=True
        Convert texts to lowercase before tokenization.
    min_word_length : int, default=1
        Minimum word length to consider.

    Examples
    --------
    >>> dist = LabbeDistance()
    >>> dist.compute("le chat noir", "le chien noir")
    0.333...
    """

    def __init__(self, lowercase: bool = True, min_word_length: int = 1):
        self.lowercase = lowercase
        self.min_word_length = min_word_length

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()
        # Simple whitespace tokenization, filter by length
        words = text.split()
        return [w for w in words if len(w) >= self.min_word_length]

    def get_counts(self, text: str) -> Counter:
        """Get word counts for a text."""
        return Counter(self.tokenize(text))

    def compute_from_counts(self, counts1: Counter, counts2: Counter) -> float:
        """
        Compute Labbé distance from pre-computed word counts.

        Implements the original IRAMUTEQ algorithm:
        1. Scale larger text to match smaller text size
        2. Sum absolute differences on scaled counts
        3. Normalize by (N_small + adjusted_N_large)

        Parameters
        ----------
        counts1 : Counter
            Word counts for first document.
        counts2 : Counter
            Word counts for second document.

        Returns
        -------
        float
            Labbé distance in [0, 1].
        """
        N1 = sum(counts1.values())
        N2 = sum(counts2.values())

        if N1 == 0 or N2 == 0:
            return 1.0  # Maximum distance for empty texts

        # Identify smaller and larger texts
        if N1 > N2:
            counts_large, counts_small = counts1, counts2
            N_large, N_small = N1, N2
        else:
            counts_large, counts_small = counts2, counts1
            N_large, N_small = N2, N1

        # Scale factor to normalize larger text to smaller
        U = N_small / N_large

        # Scale the larger text's counts
        scaled_large = {word: count * U for word, count in counts_large.items()}

        # Compute sum of scaled counts where scaled >= 1
        # (this is used for normalization denominator)
        sum_scaled_large_ge1 = sum(
            scaled_count for scaled_count in scaled_large.values()
            if scaled_count >= 1
        )

        # Three word groups (matching R's commun/deA/deB logic):
        # - commun: words present in both texts
        # - deA: words only in the smaller text
        # - deB: words only in the larger text WITH scaled count >= 1
        # Words exclusive to the larger text with scaled count < 1 are
        # excluded from both numerator and denominator (R behavior).
        total_diff = 0.0
        for word in counts_small:
            # commun + deA: all words in the smaller text
            val_small = counts_small[word]
            val_large_scaled = scaled_large.get(word, 0)
            total_diff += abs(val_small - val_large_scaled)

        for word in counts_large:
            if word not in counts_small:
                # deB: words only in larger text, but only if scaled >= 1
                val_large_scaled = scaled_large[word]
                if val_large_scaled >= 1:
                    total_diff += val_large_scaled

        # Normalize by (N_small + sum of scaled large counts >= 1)
        denominator = N_small + sum_scaled_large_ge1
        if denominator == 0:
            return 1.0

        return total_diff / denominator

    def compute(self, text1: str, text2: str) -> float:
        """
        Compute Labbé distance between two texts.

        Parameters
        ----------
        text1 : str
            First text document.
        text2 : str
            Second text document.

        Returns
        -------
        float
            Labbé distance in [0, 1].
        """
        counts1 = self.get_counts(text1)
        counts2 = self.get_counts(text2)
        return self.compute_from_counts(counts1, counts2)


class JensenShannonDistance(BaseDistance):
    """
    Jensen-Shannon divergence for text comparison.

    Converts texts to word frequency distributions and computes the
    Jensen-Shannon divergence, a symmetric and bounded measure based
    on KL divergence.

    Formula:
        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5 * (P + Q)

    The square root of JSD is a proper metric (satisfies triangle inequality).

    Properties:
    - Bounded [0, 1] (when using sqrt, which scipy returns by default)
    - Symmetric: JSD(P || Q) = JSD(Q || P)
    - 0: Identical distributions
    - 1: Completely different distributions

    References
    ----------
    - Lin, J. (1991). Divergence measures based on the Shannon entropy.
      IEEE Transactions on Information Theory, 37(1), 145-151.
    - Endres, D. M., & Schindelin, J. E. (2003). A new metric for probability
      distributions. IEEE Transactions on Information Theory, 49(7), 1858-1860.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert texts to lowercase before tokenization.
    min_word_length : int, default=1
        Minimum word length to consider.
    use_sqrt : bool, default=True
        Return sqrt(JSD) which is a proper metric. If False, returns raw JSD.

    Examples
    --------
    >>> dist = JensenShannonDistance()
    >>> dist.compute("le chat dort", "le chien dort")
    0.408...  # Moderate distance (1 word different out of 3)
    """

    def __init__(self, lowercase: bool = True, min_word_length: int = 1,
                 use_sqrt: bool = True):
        self.lowercase = lowercase
        self.min_word_length = min_word_length
        self.use_sqrt = use_sqrt

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.lowercase:
            text = text.lower()
        words = text.split()
        return [w for w in words if len(w) >= self.min_word_length]

    def get_counts(self, text: str) -> Counter:
        """Get word counts for a text."""
        return Counter(self.tokenize(text))

    def compute_from_counts(self, counts1: Counter, counts2: Counter) -> float:
        """
        Compute Jensen-Shannon distance from pre-computed word counts (optimized).

        Parameters
        ----------
        counts1 : Counter
            Word counts for first document.
        counts2 : Counter
            Word counts for second document.

        Returns
        -------
        float
            Jensen-Shannon distance in [0, 1].
        """
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())

        if total1 == 0 or total2 == 0:
            return 1.0  # Maximum distance for empty texts

        # Joint vocabulary (use list for ordering)
        all_words = list(set(counts1.keys()) | set(counts2.keys()))

        # Create aligned frequency vectors directly as numpy arrays
        vec1 = np.array([counts1.get(w, 0) for w in all_words], dtype=np.float64)
        vec2 = np.array([counts2.get(w, 0) for w in all_words], dtype=np.float64)

        # Normalize to probability distributions
        vec1 /= total1
        vec2 /= total2

        # scipy.jensenshannon returns sqrt(JSD) by default
        js_dist = jensenshannon(vec1, vec2)

        if not self.use_sqrt:
            # Return raw JSD (square of the metric)
            return js_dist ** 2

        return float(js_dist)

    def compute(self, text1: str, text2: str) -> float:
        """
        Compute Jensen-Shannon distance between two texts.

        Parameters
        ----------
        text1 : str
            First text document.
        text2 : str
            Second text document.

        Returns
        -------
        float
            Jensen-Shannon distance in [0, 1].
        """
        counts1 = self.get_counts(text1)
        counts2 = self.get_counts(text2)
        return self.compute_from_counts(counts1, counts2)


# =============================================================================
# MODULE-LEVEL DISTANCE HELPERS (used by evaluate_topic_distances)
# =============================================================================

def _labbe_from_counts(c1: Counter, c2: Counter) -> float:
    """
    Labbé distance from token counts (IRAMUTEQ R algorithm).

    Matches the R compute.labbe() implementation:
    - Scale the larger text down to match the smaller text's size
    - Exclude words exclusive to the larger text with scaled count < 1
      from both numerator and denominator
    - Result is bounded [0, 1]
    """
    N1 = sum(c1.values())
    N2 = sum(c2.values())
    if N1 == 0 or N2 == 0:
        return 1.0

    if N1 > N2:
        counts_large, counts_small = c1, c2
        N_large, N_small = N1, N2
    else:
        counts_large, counts_small = c2, c1
        N_large, N_small = N2, N1

    U = N_small / N_large
    scaled_large = {w: c * U for w, c in counts_large.items()}
    sum_scaled_ge1 = sum(sc for sc in scaled_large.values() if sc >= 1)

    # Words in smaller text (commun + deA)
    total_diff = sum(
        abs(counts_small[w] - scaled_large.get(w, 0))
        for w in counts_small
    )
    # Words only in larger text with scaled count >= 1 (deB)
    total_diff += sum(
        scaled_large[w] for w in counts_large
        if w not in counts_small and scaled_large[w] >= 1
    )

    denominator = N_small + sum_scaled_ge1
    return total_diff / denominator if denominator > 0 else 1.0


def _js_from_counts(c1: Counter, c2: Counter) -> float:
    """Jensen-Shannon distance from token counts."""
    total1 = sum(c1.values())
    total2 = sum(c2.values())
    if total1 == 0 or total2 == 0:
        return 1.0
    all_words = list(set(c1.keys()) | set(c2.keys()))
    vec1 = np.array([c1.get(w, 0) for w in all_words], dtype=np.float64)
    vec2 = np.array([c2.get(w, 0) for w in all_words], dtype=np.float64)
    vec1 /= total1
    vec2 /= total2
    return float(jensenshannon(vec1, vec2))


# =============================================================================
# AGGREGATION UTILITIES
# =============================================================================

def aggregate_documents(
    doc_tokens: List[List[str]],
    indices: List[int],
    aggregation_size: int = 20,
    random_seed: Optional[int] = None
) -> List[Counter]:
    """
    Aggregate multiple documents into larger units by merging tokens.

    This helps with Labbé distance which is sensitive to text length.
    By aggregating n verses together, we get more comparable text sizes.

    Parameters
    ----------
    doc_tokens : List[List[str]]
        Pre-tokenized documents.
    indices : List[int]
        Indices of documents to aggregate.
    aggregation_size : int, default=20
        Number of documents to merge into each unit.
    random_seed : int, optional
        Random seed for shuffling before aggregation.

    Returns
    -------
    List[Counter]
        List of aggregated token counts.
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Shuffle indices to avoid bias from document ordering
    shuffled_indices = indices.copy()
    random.shuffle(shuffled_indices)

    aggregated = []
    for i in range(0, len(shuffled_indices), aggregation_size):
        batch_indices = shuffled_indices[i:i + aggregation_size]
        if len(batch_indices) < aggregation_size // 2:
            # Skip very small batches at the end
            continue

        # Merge all tokens from the batch
        merged_counts = Counter()
        for idx in batch_indices:
            merged_counts.update(doc_tokens[idx])

        if sum(merged_counts.values()) > 0:
            aggregated.append(merged_counts)

    return aggregated


# =============================================================================
# COMPREHENSIVE TOPIC DISTANCE EVALUATION
# =============================================================================

def evaluate_topic_distances(
    doc_tokens: List[List[str]],
    topic_assignments: List[int],
    mode: str = 'intra_all_paired',
    distance_type: str = 'both',
    aggregation_size: int = 20,
    sample_size: int = 5000,
    random_seed: Optional[int] = None,
    max_topics: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, dict]:
    """
    Comprehensive topic distance evaluation with multiple modes.

    Parameters
    ----------
    doc_tokens : List[List[str]]
        Pre-tokenized documents (from SpaCyTokenizer.batch_tokenize).
    topic_assignments : List[int]
        Topic assignment for each document. Topic -1 = outliers (excluded).
    mode : str, default='intra_all_paired'
        Distance computation mode:
        - 'intra_all_paired': Pairwise distances within topics (homogeneity)
        - 'inter_all_paired': Distances between inside and outside topic (separation)
        - 'intra_aggregated': Intra-topic with aggregated documents
        - 'inter_aggregated': Inter-topic with aggregated documents
    distance_type : str, default='both'
        Which distances to compute: 'js', 'labbe', or 'both'
    aggregation_size : int, default=20
        Number of documents to aggregate (for aggregated modes).
    sample_size : int, default=5000
        Maximum pairs to sample per topic.
    random_seed : int, optional
        Random seed for reproducibility.
    max_topics : int, optional
        Maximum number of topics to analyze.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    dict
        Results containing 'js' and/or 'labbe' sub-dicts with:
        - 'mean': Overall mean distance
        - 'std': Standard deviation across topics
        - 'per_topic': Per-topic statistics
        - 'mode': The computation mode used
        - 'aggregation_size': Aggregation size (if applicable)
    """
    if len(doc_tokens) != len(topic_assignments):
        raise ValueError(
            f"Length mismatch: {len(doc_tokens)} token lists vs "
            f"{len(topic_assignments)} topic assignments"
        )

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Group document indices by topic (exclude outliers: topic -1)
    topic_doc_indices: Dict[int, List[int]] = {}
    all_non_outlier_indices: List[int] = []
    for idx, topic in enumerate(topic_assignments):
        if topic == -1:
            continue
        if topic not in topic_doc_indices:
            topic_doc_indices[topic] = []
        topic_doc_indices[topic].append(idx)
        all_non_outlier_indices.append(idx)

    results = {}

    if not topic_doc_indices:
        empty_result = {
            'mean': 0.0, 'std': 0.0, 'per_topic': {}, 'n_topics': 0,
            'total_documents': 0, 'mode': mode
        }
        if distance_type in ('js', 'both'):
            results['js'] = empty_result.copy()
        if distance_type in ('labbe', 'both'):
            results['labbe'] = empty_result.copy()
        return results

    topic_ids = sorted(topic_doc_indices.keys())
    if max_topics is not None and len(topic_ids) > max_topics:
        topic_ids = topic_ids[:max_topics]

    if verbose:
        print(f"  Mode: {mode}, Distance: {distance_type}")
        print(f"  Topics: {len(topic_ids)}, Documents: {len(all_non_outlier_indices)}")

    # Pre-compute token counts
    if verbose:
        print("  Pre-computing token counts...")

    doc_counts: Dict[int, Counter] = {}
    for idx in all_non_outlier_indices:
        doc_counts[idx] = Counter(doc_tokens[idx])

    # Dispatch to appropriate computation mode
    if mode == 'intra_all_paired':
        results = _compute_intra_paired(
            topic_ids, topic_doc_indices, doc_counts,
            _labbe_from_counts, _js_from_counts, distance_type,
            sample_size, verbose
        )
    elif mode == 'inter_all_paired':
        results = _compute_inter_paired(
            topic_ids, topic_doc_indices, all_non_outlier_indices, doc_counts,
            _labbe_from_counts, _js_from_counts, distance_type,
            sample_size, verbose
        )
    elif mode == 'intra_aggregated':
        results = _compute_intra_aggregated(
            topic_ids, topic_doc_indices, doc_tokens,
            _labbe_from_counts, _js_from_counts, distance_type,
            aggregation_size, sample_size, random_seed, verbose
        )
    elif mode == 'inter_aggregated':
        results = _compute_inter_aggregated(
            topic_ids, topic_doc_indices, all_non_outlier_indices, doc_tokens,
            _labbe_from_counts, _js_from_counts, distance_type,
            aggregation_size, sample_size, random_seed, verbose
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'intra_all_paired', "
                        f"'inter_all_paired', 'intra_aggregated', or 'inter_aggregated'")

    # Add mode info to results
    for key in results:
        results[key]['mode'] = mode
        if 'aggregated' in mode:
            results[key]['aggregation_size'] = aggregation_size

    return results


def _compute_intra_paired(
    topic_ids: List[int],
    topic_doc_indices: Dict[int, List[int]],
    doc_counts: Dict[int, Counter],
    labbe_fn: Callable,
    js_fn: Callable,
    distance_type: str,
    sample_size: int,
    verbose: bool
) -> Dict[str, dict]:
    """Compute intra-topic pairwise distances (homogeneity)."""

    def compute_for_metric(compute_fn: Callable, metric_name: str) -> dict:
        per_topic_results = {}
        all_topic_means = []

        for topic_idx, topic_id in enumerate(topic_ids):
            doc_indices = topic_doc_indices[topic_id]
            n_docs = len(doc_indices)

            if n_docs < 2:
                per_topic_results[topic_id] = {
                    'mean_distance': 0.0, 'n_documents': n_docs, 'n_pairs_sampled': 0
                }
                all_topic_means.append(0.0)
                continue

            # Sample pairs
            n_total_pairs = n_docs * (n_docs - 1) // 2
            if sample_size > 0 and n_total_pairs > sample_size:
                sampled_pairs = _sample_pairs(n_docs, sample_size)
            else:
                sampled_pairs = list(itertools.combinations(range(n_docs), 2))

            # Compute distances
            distances = []
            for i, j in sampled_pairs:
                dist = compute_fn(doc_counts[doc_indices[i]], doc_counts[doc_indices[j]])
                distances.append(dist)

            mean_dist = np.mean(distances) if distances else 0.0
            per_topic_results[topic_id] = {
                'mean_distance': float(mean_dist),
                'n_documents': n_docs,
                'n_pairs_sampled': len(sampled_pairs)
            }
            all_topic_means.append(mean_dist)

            if verbose:
                print(f"    [{metric_name}] Topic {topic_id}: {n_docs} docs, "
                      f"mean={mean_dist:.4f}")

        total_docs = sum(len(topic_doc_indices[t]) for t in topic_ids)
        return {
            'mean': float(np.mean(all_topic_means)) if all_topic_means else 0.0,
            'std': float(np.std(all_topic_means)) if all_topic_means else 0.0,
            'per_topic': per_topic_results,
            'n_topics': len(topic_ids),
            'total_documents': total_docs
        }

    results = {}
    if distance_type in ('labbe', 'both'):
        if verbose:
            print("  Computing intra-topic Labbé distances...")
        results['labbe'] = compute_for_metric(labbe_fn, 'Labbé')
    if distance_type in ('js', 'both'):
        if verbose:
            print("  Computing intra-topic Jensen-Shannon distances...")
        results['js'] = compute_for_metric(js_fn, 'JS')

    return results


def _compute_inter_paired(
    topic_ids: List[int],
    topic_doc_indices: Dict[int, List[int]],
    all_indices: List[int],
    doc_counts: Dict[int, Counter],
    labbe_fn: Callable,
    js_fn: Callable,
    distance_type: str,
    sample_size: int,
    verbose: bool
) -> Dict[str, dict]:
    """Compute inter-topic distances (inside vs outside topic - separation)."""

    # Build set of indices per topic for fast lookup
    topic_index_sets = {t: set(topic_doc_indices[t]) for t in topic_ids}

    def compute_for_metric(compute_fn: Callable, metric_name: str) -> dict:
        per_topic_results = {}
        all_topic_means = []

        for topic_idx, topic_id in enumerate(topic_ids):
            inside_indices = topic_doc_indices[topic_id]
            outside_indices = [i for i in all_indices if i not in topic_index_sets[topic_id]]

            n_inside = len(inside_indices)
            n_outside = len(outside_indices)

            if n_inside == 0 or n_outside == 0:
                per_topic_results[topic_id] = {
                    'mean_distance': 0.0, 'n_inside': n_inside,
                    'n_outside': n_outside, 'n_pairs_sampled': 0
                }
                all_topic_means.append(0.0)
                continue

            # Total possible pairs = n_inside * n_outside
            n_total_pairs = n_inside * n_outside
            if sample_size > 0 and n_total_pairs > sample_size:
                # Sample random pairs (inside, outside)
                sampled_pairs = []
                seen = set()
                max_attempts = sample_size * 3
                attempts = 0
                while len(sampled_pairs) < sample_size and attempts < max_attempts:
                    i = random.randint(0, n_inside - 1)
                    j = random.randint(0, n_outside - 1)
                    if (i, j) not in seen:
                        seen.add((i, j))
                        sampled_pairs.append((inside_indices[i], outside_indices[j]))
                    attempts += 1
            else:
                sampled_pairs = [
                    (inside_indices[i], outside_indices[j])
                    for i in range(n_inside) for j in range(n_outside)
                ]

            # Compute distances
            distances = []
            for idx_in, idx_out in sampled_pairs:
                dist = compute_fn(doc_counts[idx_in], doc_counts[idx_out])
                distances.append(dist)

            mean_dist = np.mean(distances) if distances else 0.0
            per_topic_results[topic_id] = {
                'mean_distance': float(mean_dist),
                'n_inside': n_inside,
                'n_outside': n_outside,
                'n_pairs_sampled': len(sampled_pairs)
            }
            all_topic_means.append(mean_dist)

            if verbose:
                print(f"    [{metric_name}] Topic {topic_id}: {n_inside} in, "
                      f"{n_outside} out, mean={mean_dist:.4f}")

        total_docs = len(all_indices)
        return {
            'mean': float(np.mean(all_topic_means)) if all_topic_means else 0.0,
            'std': float(np.std(all_topic_means)) if all_topic_means else 0.0,
            'per_topic': per_topic_results,
            'n_topics': len(topic_ids),
            'total_documents': total_docs
        }

    results = {}
    if distance_type in ('labbe', 'both'):
        if verbose:
            print("  Computing inter-topic Labbé distances...")
        results['labbe'] = compute_for_metric(labbe_fn, 'Labbé')
    if distance_type in ('js', 'both'):
        if verbose:
            print("  Computing inter-topic Jensen-Shannon distances...")
        results['js'] = compute_for_metric(js_fn, 'JS')

    return results


def _compute_intra_aggregated(
    topic_ids: List[int],
    topic_doc_indices: Dict[int, List[int]],
    doc_tokens: List[List[str]],
    labbe_fn: Callable,
    js_fn: Callable,
    distance_type: str,
    aggregation_size: int,
    sample_size: int,
    random_seed: Optional[int],
    verbose: bool
) -> Dict[str, dict]:
    """Compute intra-topic distances with aggregated documents."""

    def compute_for_metric(compute_fn: Callable, metric_name: str) -> dict:
        per_topic_results = {}
        all_topic_means = []

        for topic_idx, topic_id in enumerate(topic_ids):
            doc_indices = topic_doc_indices[topic_id]

            # Aggregate documents
            aggregated_counts = aggregate_documents(
                doc_tokens, doc_indices, aggregation_size, random_seed
            )
            n_units = len(aggregated_counts)

            if n_units < 2:
                per_topic_results[topic_id] = {
                    'mean_distance': 0.0, 'n_documents': len(doc_indices),
                    'n_aggregated_units': n_units, 'n_pairs_sampled': 0
                }
                all_topic_means.append(0.0)
                continue

            # Sample pairs of aggregated units
            n_total_pairs = n_units * (n_units - 1) // 2
            if sample_size > 0 and n_total_pairs > sample_size:
                sampled_pairs = _sample_pairs(n_units, sample_size)
            else:
                sampled_pairs = list(itertools.combinations(range(n_units), 2))

            # Compute distances
            distances = []
            for i, j in sampled_pairs:
                dist = compute_fn(aggregated_counts[i], aggregated_counts[j])
                distances.append(dist)

            mean_dist = np.mean(distances) if distances else 0.0
            per_topic_results[topic_id] = {
                'mean_distance': float(mean_dist),
                'n_documents': len(doc_indices),
                'n_aggregated_units': n_units,
                'n_pairs_sampled': len(sampled_pairs)
            }
            all_topic_means.append(mean_dist)

            if verbose:
                print(f"    [{metric_name}] Topic {topic_id}: {len(doc_indices)} docs -> "
                      f"{n_units} units, mean={mean_dist:.4f}")

        total_docs = sum(len(topic_doc_indices[t]) for t in topic_ids)
        return {
            'mean': float(np.mean(all_topic_means)) if all_topic_means else 0.0,
            'std': float(np.std(all_topic_means)) if all_topic_means else 0.0,
            'per_topic': per_topic_results,
            'n_topics': len(topic_ids),
            'total_documents': total_docs
        }

    results = {}
    if distance_type in ('labbe', 'both'):
        if verbose:
            print(f"  Computing intra-topic Labbé distances (aggregated, n={aggregation_size})...")
        results['labbe'] = compute_for_metric(labbe_fn, 'Labbé')
    if distance_type in ('js', 'both'):
        if verbose:
            print(f"  Computing intra-topic Jensen-Shannon distances (aggregated, n={aggregation_size})...")
        results['js'] = compute_for_metric(js_fn, 'JS')

    return results


def _compute_inter_aggregated(
    topic_ids: List[int],
    topic_doc_indices: Dict[int, List[int]],
    all_indices: List[int],
    doc_tokens: List[List[str]],
    labbe_fn: Callable,
    js_fn: Callable,
    distance_type: str,
    aggregation_size: int,
    sample_size: int,
    random_seed: Optional[int],
    verbose: bool
) -> Dict[str, dict]:
    """Compute inter-topic distances with aggregated documents.

    Optimization: pre-aggregates each topic's docs ONCE, then composes
    the 'outside' set from other topics' pre-aggregated units. This avoids
    redundant aggregation of ~N_total docs for each of the K topics.
    """

    # Pre-aggregate each topic's docs once
    per_topic_units = {}
    for topic_id in topic_ids:
        per_topic_units[topic_id] = aggregate_documents(
            doc_tokens, topic_doc_indices[topic_id], aggregation_size, random_seed
        )

    def compute_for_metric(compute_fn: Callable, metric_name: str) -> dict:
        per_topic_results = {}
        all_topic_means = []

        for topic_idx, topic_id in enumerate(topic_ids):
            inside_aggregated = per_topic_units[topic_id]

            # Outside = union of all other topics' pre-aggregated units
            outside_aggregated = [
                unit for tid, units in per_topic_units.items()
                if tid != topic_id for unit in units
            ]

            n_inside = len(inside_aggregated)
            n_outside = len(outside_aggregated)

            if n_inside == 0 or n_outside == 0:
                per_topic_results[topic_id] = {
                    'mean_distance': 0.0, 'n_inside_units': n_inside,
                    'n_outside_units': n_outside, 'n_pairs_sampled': 0
                }
                all_topic_means.append(0.0)
                continue

            # Sample pairs
            n_total_pairs = n_inside * n_outside
            if sample_size > 0 and n_total_pairs > sample_size:
                sampled_pairs = []
                seen = set()
                max_attempts = sample_size * 3
                attempts = 0
                while len(sampled_pairs) < sample_size and attempts < max_attempts:
                    i = random.randint(0, n_inside - 1)
                    j = random.randint(0, n_outside - 1)
                    if (i, j) not in seen:
                        seen.add((i, j))
                        sampled_pairs.append((i, j))
                    attempts += 1
            else:
                sampled_pairs = [(i, j) for i in range(n_inside) for j in range(n_outside)]

            # Compute distances
            distances = []
            for i, j in sampled_pairs:
                dist = compute_fn(inside_aggregated[i], outside_aggregated[j])
                distances.append(dist)

            mean_dist = np.mean(distances) if distances else 0.0
            n_inside_docs = len(topic_doc_indices[topic_id])
            n_outside_docs = sum(len(topic_doc_indices[t]) for t in topic_ids if t != topic_id)
            per_topic_results[topic_id] = {
                'mean_distance': float(mean_dist),
                'n_inside_docs': n_inside_docs,
                'n_outside_docs': n_outside_docs,
                'n_inside_units': n_inside,
                'n_outside_units': n_outside,
                'n_pairs_sampled': len(sampled_pairs)
            }
            all_topic_means.append(mean_dist)

            if verbose:
                print(f"    [{metric_name}] Topic {topic_id}: {n_inside} in-units, "
                      f"{n_outside} out-units, mean={mean_dist:.4f}")

        total_docs = len(all_indices)
        return {
            'mean': float(np.mean(all_topic_means)) if all_topic_means else 0.0,
            'std': float(np.std(all_topic_means)) if all_topic_means else 0.0,
            'per_topic': per_topic_results,
            'n_topics': len(topic_ids),
            'total_documents': total_docs
        }

    results = {}
    if distance_type in ('labbe', 'both'):
        if verbose:
            print(f"  Computing inter-topic Labbé distances (aggregated, n={aggregation_size})...")
        results['labbe'] = compute_for_metric(labbe_fn, 'Labbé')
    if distance_type in ('js', 'both'):
        if verbose:
            print(f"  Computing inter-topic Jensen-Shannon distances (aggregated, n={aggregation_size})...")
        results['js'] = compute_for_metric(js_fn, 'JS')

    return results


def _sample_pairs(n: int, sample_size: int) -> List[Tuple[int, int]]:
    """Sample unique pairs from range(n)."""
    sampled = set()
    max_attempts = sample_size * 3
    attempts = 0
    while len(sampled) < sample_size and attempts < max_attempts:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            pair = (min(i, j), max(i, j))
            sampled.add(pair)
        attempts += 1
    return list(sampled)


# =============================================================================
# DYNAMIC AGGREGATION RANGE
# =============================================================================

def compute_aggregation_range(
    doc_tokens: List[List[str]],
    topic_assignments: List[int],
    n_points: int = 5,
    min_words_per_unit: int = 500,
    min_units_per_topic: int = 5,
    min_step: int = 10,
    override_min_topic_size: Optional[int] = None,
) -> Tuple[List[int], dict]:
    """
    Compute a data-driven range of aggregation sizes.

    The range is determined by two constraints:
    - Minimum: smallest batch size guaranteeing >min_words_per_unit tokens
      per aggregated unit (based on mean document length).
    - Maximum: largest batch size <= 1/min_units_per_topic of the smallest
      topic size (so every topic produces enough aggregated units).

    After generating linearly-spaced points, filters out any with gap
    < min_step from the previous point, ensuring meaningful increments.

    Parameters
    ----------
    doc_tokens : List[List[str]]
        Pre-tokenized documents.
    topic_assignments : List[int]
        Topic assignment per document (-1 = outliers, excluded).
    n_points : int, default=5
        Target number of linearly-spaced aggregation sizes.
    min_words_per_unit : int, default=500
        Minimum total tokens required per aggregated unit.
    min_units_per_topic : int, default=5
        Minimum aggregated units per topic (determines upper bound).
    min_step : int, default=10
        Minimum gap between consecutive aggregation sizes.
    override_min_topic_size : int, optional
        If provided, use this as the minimum topic size instead of
        computing it from topic_assignments. Useful for enforcing a
        shared range across multiple models.

    Returns
    -------
    Tuple[List[int], dict]
        - Sorted list of aggregation sizes (integers).
        - Metadata dict with 'mean_doc_length', 'min_topic_size',
          'agg_min', 'agg_max', 'topic_sizes'.
    """
    import math

    # Mean document length (in tokens)
    doc_lengths = [len(tokens) for tokens in doc_tokens if tokens]
    mean_doc_length = float(np.mean(doc_lengths)) if doc_lengths else 1.0

    # Topic sizes (excluding outliers)
    topic_sizes: Dict[int, int] = {}
    for topic in topic_assignments:
        if topic == -1:
            continue
        topic_sizes[topic] = topic_sizes.get(topic, 0) + 1

    if not topic_sizes:
        return [20], {'mean_doc_length': mean_doc_length, 'min_topic_size': 0,
                       'agg_min': 20, 'agg_max': 20, 'topic_sizes': {}}

    min_topic_size = override_min_topic_size if override_min_topic_size is not None else min(topic_sizes.values())

    # Minimum aggregation: ceil(min_words / mean_doc_length)
    agg_min = max(2, math.ceil(min_words_per_unit / mean_doc_length))

    # Maximum aggregation: floor(min_topic_size / min_units_per_topic)
    agg_max = max(agg_min, min_topic_size // min_units_per_topic)

    if agg_min >= agg_max:
        sizes = [agg_min]
    else:
        # Generate candidate points
        raw_sizes = sorted(set(
            int(round(v))
            for v in np.linspace(agg_min, agg_max, n_points)
        ))

        # Filter by min_step: keep points with gap >= min_step from previous
        sizes = [raw_sizes[0]]
        for s in raw_sizes[1:]:
            if s - sizes[-1] >= min_step:
                sizes.append(s)
        # Always include the max
        if sizes[-1] != raw_sizes[-1]:
            sizes.append(raw_sizes[-1])

        # Fallback if too few points after filtering
        if len(sizes) < 3 and agg_max - agg_min >= 2 * min_step:
            mid = (agg_min + agg_max) // 2
            sizes = sorted(set([agg_min, mid, agg_max]))

    metadata = {
        'mean_doc_length': float(mean_doc_length),
        'min_topic_size': int(min_topic_size),
        'agg_min': int(agg_min),
        'agg_max': int(agg_max),
        'topic_sizes': {int(k): int(v) for k, v in topic_sizes.items()},
    }

    return sizes, metadata


# =============================================================================
# MULTI-AGGREGATION EVALUATION
# =============================================================================

def evaluate_multi_aggregation(
    doc_tokens: List[List[str]],
    topic_assignments: List[int],
    aggregation_sizes: List[int],
    modes: Optional[List[str]] = None,
    distance_type: str = 'both',
    sample_size: int = 5000,
    random_seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, dict]:
    """
    Run topic distance evaluation across multiple aggregation sizes.

    For each aggregation size, computes intra and/or inter aggregated
    distances. Returns per-size and per-topic results suitable for
    plotting the stabilization curve and inter-topic ranking tables.

    Parameters
    ----------
    doc_tokens : List[List[str]]
        Pre-tokenized documents.
    topic_assignments : List[int]
        Topic assignment per document (-1 = outliers, excluded).
    aggregation_sizes : List[int]
        List of aggregation sizes to evaluate.
    modes : List[str], optional
        Modes to evaluate. Default: ['intra_aggregated', 'inter_aggregated'].
    distance_type : str, default='both'
        Distance metric: 'js', 'labbe', or 'both'.
    sample_size : int, default=5000
        Maximum pairs to sample per topic.
    random_seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    dict
        Structure: {
            aggregation_size (int): {
                mode (str): {
                    'js': {'mean': ..., 'std': ..., 'per_topic': {...}},
                    'labbe': {'mean': ..., 'std': ..., 'per_topic': {...}}
                }
            }
        }
    """
    if modes is None:
        modes = ['intra_aggregated', 'inter_aggregated']

    results = {}
    for agg_size in aggregation_sizes:
        if verbose:
            print(f"    Aggregation size = {agg_size}")
        results[agg_size] = {}

        for mode in modes:
            result = evaluate_topic_distances(
                doc_tokens,
                topic_assignments,
                mode=mode,
                distance_type=distance_type,
                aggregation_size=agg_size,
                sample_size=sample_size,
                random_seed=random_seed,
                verbose=False
            )
            results[agg_size][mode] = result

    return results


# =============================================================================
# TOPIC CENTROID DISTANCES (one-vs-rest, no sampling)
# =============================================================================

def compute_topic_centroid_distances(
    doc_tokens: List[List[str]],
    topic_assignments: List[int],
    distance_type: str = 'both',
) -> Dict[str, dict]:
    """
    Compute inter-topic distances by merging all docs per topic into one centroid.

    For each topic, merges all its documents' tokens into a single Counter
    (centroid), then merges all other topics' tokens into a rest Counter.
    Computes Labbé and/or JS between (centroid, rest).

    This gives maximum discrimination between topics — no sampling noise,
    no aggregation size dependency.

    Parameters
    ----------
    doc_tokens : List[List[str]]
        Pre-tokenized documents.
    topic_assignments : List[int]
        Topic assignment per document (-1 = outliers, excluded).
    distance_type : str, default='both'
        'labbe', 'js', or 'both'.

    Returns
    -------
    dict
        Same structure as evaluate_topic_distances inter results:
        {'labbe': {'mean': ..., 'std': ..., 'per_topic': {tid: {'mean_distance': d}}},
         'js': {...}}
    """
    # Build per-topic centroids
    topic_centroids: Dict[int, Counter] = {}
    for tokens, topic in zip(doc_tokens, topic_assignments):
        if topic == -1:
            continue
        if topic not in topic_centroids:
            topic_centroids[topic] = Counter()
        topic_centroids[topic].update(tokens)

    if len(topic_centroids) < 2:
        empty = {'mean': 0.0, 'std': 0.0, 'per_topic': {}, 'n_topics': len(topic_centroids)}
        results = {}
        if distance_type in ('labbe', 'both'):
            results['labbe'] = empty.copy()
        if distance_type in ('js', 'both'):
            results['js'] = empty.copy()
        return results

    topic_ids = sorted(topic_centroids.keys())

    # Total corpus centroid (for building rest)
    total_centroid = Counter()
    for c in topic_centroids.values():
        total_centroid.update(c)

    results = {}

    for metric_name, compute_fn in [('labbe', _labbe_from_counts), ('js', _js_from_counts)]:
        if distance_type not in (metric_name, 'both'):
            continue

        per_topic = {}
        distances = []

        for tid in topic_ids:
            # rest = total - this topic
            rest = total_centroid.copy()
            rest.subtract(topic_centroids[tid])
            # Remove zero/negative counts
            rest = Counter({w: c for w, c in rest.items() if c > 0})

            dist = compute_fn(topic_centroids[tid], rest)
            per_topic[tid] = {'mean_distance': float(dist)}
            distances.append(dist)

        results[metric_name] = {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'per_topic': per_topic,
            'n_topics': len(topic_ids),
        }

    return results


# =============================================================================
# χ²/n WORD × TOPIC INDEPENDENCE TEST
# =============================================================================

def compute_word_topic_chi2(
    doc_tokens: List[List[str]],
    topic_assignments: List[int],
    min_word_freq: int = 5,
) -> dict:
    """
    Compute χ²/n on a word × topic contingency table.

    Measures how strongly word frequencies depend on topic assignment.
    Higher χ²/n = topics capture more distinctive vocabulary.

    Also computes per-topic contribution to χ², showing which topics
    drive the departure from word-topic independence the most.

    Parameters
    ----------
    doc_tokens : List[List[str]]
        Pre-tokenized documents.
    topic_assignments : List[int]
        Topic assignment per document (-1 = outliers, excluded).
    min_word_freq : int, default=5
        Minimum total frequency for a word to be included in the table
        (avoids sparse cells inflating χ²).

    Returns
    -------
    dict
        {
            'chi2': float,
            'n': int (total token count),
            'chi2_over_n': float,
            'p_value': float,
            'dof': int,
            'vocab_size': int (after filtering),
            'n_topics': int,
            'min_word_freq': int,
            'per_topic_chi2': {topic_id: float},
            'per_topic_chi2_pct': {topic_id: float},
        }
    """
    from scipy.stats import chi2_contingency

    # Build word counts per topic
    topic_counters: Dict[int, Counter] = {}
    for tokens, topic in zip(doc_tokens, topic_assignments):
        if topic == -1:
            continue
        if topic not in topic_counters:
            topic_counters[topic] = Counter()
        topic_counters[topic].update(tokens)

    if not topic_counters:
        return {
            'chi2': 0.0, 'n': 0, 'chi2_over_n': 0.0, 'p_value': 1.0,
            'dof': 0, 'vocab_size': 0, 'n_topics': 0,
            'min_word_freq': min_word_freq,
            'per_topic_chi2': {}, 'per_topic_chi2_pct': {},
        }

    # Global vocabulary with minimum frequency filter
    global_counts = Counter()
    for counter in topic_counters.values():
        global_counts.update(counter)

    vocab = sorted(w for w, c in global_counts.items() if c >= min_word_freq)
    vocab_index = {w: i for i, w in enumerate(vocab)}

    topic_ids = sorted(topic_counters.keys())
    n_vocab = len(vocab)
    n_topics = len(topic_ids)

    if n_vocab == 0 or n_topics < 2:
        return {
            'chi2': 0.0, 'n': 0, 'chi2_over_n': 0.0, 'p_value': 1.0,
            'dof': 0, 'vocab_size': n_vocab, 'n_topics': n_topics,
            'min_word_freq': min_word_freq,
            'per_topic_chi2': {}, 'per_topic_chi2_pct': {},
        }

    # Build contingency table: rows = vocabulary, cols = topics
    table = np.zeros((n_vocab, n_topics), dtype=np.float64)
    for j, tid in enumerate(topic_ids):
        counter = topic_counters[tid]
        for word, idx in vocab_index.items():
            table[idx, j] = counter.get(word, 0)

    n_total = table.sum()

    # Chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(table)

    chi2_over_n = chi2 / n_total if n_total > 0 else 0.0

    # Per-topic contribution: sum of (observed - expected)² / expected per column
    cell_chi2 = (table - expected) ** 2 / np.where(expected > 0, expected, 1.0)

    per_topic_chi2 = {}
    per_topic_chi2_pct = {}
    for j, tid in enumerate(topic_ids):
        contribution = float(cell_chi2[:, j].sum())
        per_topic_chi2[tid] = contribution
        per_topic_chi2_pct[tid] = (contribution / chi2 * 100) if chi2 > 0 else 0.0

    return {
        'chi2': float(chi2),
        'n': int(n_total),
        'chi2_over_n': float(chi2_over_n),
        'p_value': float(p_value),
        'dof': int(dof),
        'vocab_size': n_vocab,
        'n_topics': n_topics,
        'min_word_freq': min_word_freq,
        'per_topic_chi2': per_topic_chi2,
        'per_topic_chi2_pct': per_topic_chi2_pct,
    }
