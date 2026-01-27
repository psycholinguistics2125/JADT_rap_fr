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
- WMDDistance: Word Mover's Distance stub (to be implemented with FastText)

Functions:
----------
- evaluate_topic_coherence: Compute mean intra-topic distance for all topics

References:
-----------
- Labbé, C., & Labbé, D. (2001). Inter-textual distance and authorship attribution.
  Journal of Quantitative Linguistics, 8(3), 213-231.
- Lin, J. (1991). Divergence measures based on the Shannon entropy.
  IEEE Transactions on Information Theory, 37(1), 145-151.
- Lu, Y., et al. (2020). Topic modeling for text analysis. In Encyclopedia of
  Big Data Technologies.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation.
  Journal of Machine Learning Research, 3, 993-1022.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, Callable
import random
import itertools

import numpy as np
from scipy.spatial.distance import jensenshannon


# =============================================================================
# SPACY TOKENIZER FOR EFFICIENT BATCH PROCESSING
# =============================================================================

class SpaCyTokenizer:
    """
    SpaCy-based tokenizer optimized for batch processing.

    Uses spaCy's pipe() method for efficient parallel tokenization of large
    document collections. Disables unnecessary pipeline components for speed.

    Parameters
    ----------
    model_name : str, default='fr_core_news_lg'
        SpaCy model to use. Options:
        - 'fr_core_news_sm': Small, fast (12MB)
        - 'fr_core_news_md': Medium, balanced (44MB)
        - 'fr_core_news_lg': Large, accurate (540MB) - recommended
    lowercase : bool, default=True
        Convert tokens to lowercase.
    min_word_length : int, default=2
        Minimum token length to keep.
    remove_stopwords : bool, default=True
        Remove French stopwords.
    lemmatize : bool, default=False
        Use lemmas instead of surface forms.
    batch_size : int, default=1000
        Batch size for spaCy's pipe() method.
    n_process : int, default=-1
        Number of processes for parallel processing.
        -1 uses all available CPUs.

    Examples
    --------
    >>> tokenizer = SpaCyTokenizer(model_name='fr_core_news_lg')
    >>> documents = ["Le chat dort sur le canapé", "Les chiens jouent dehors"]
    >>> token_counts = tokenizer.batch_tokenize(documents)
    >>> token_counts[0]
    ['chat', 'dort', 'canapé']
    """

    def __init__(
        self,
        model_name: str = 'fr_core_news_lg',
        lowercase: bool = True,
        min_word_length: int = 2,
        remove_stopwords: bool = True,
        lemmatize: bool = False,
        batch_size: int = 1000,
        n_process: int = -1
    ):
        self.model_name = model_name
        self.lowercase = lowercase
        self.min_word_length = min_word_length
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.batch_size = batch_size
        self.n_process = n_process
        self._nlp = None
        self._stopwords = None

    def _load_model(self):
        """Lazy load spaCy model with optimized settings."""
        if self._nlp is None:
            import spacy

            # Disable components we don't need for tokenization
            # Keep: tok2vec, tagger (for lemma/POS), morphologizer
            # Disable: parser, ner, textcat, etc.
            disable_components = ['parser', 'ner', 'textcat', 'custom']

            try:
                self._nlp = spacy.load(
                    self.model_name,
                    disable=disable_components
                )
                print(f"  Loaded spaCy model: {self.model_name}")
            except OSError:
                # Fallback to small model if large not available
                fallback = 'fr_core_news_sm'
                print(f"  Warning: {self.model_name} not found, using {fallback}")
                self._nlp = spacy.load(fallback, disable=disable_components)

            # Load stopwords
            if self.remove_stopwords:
                self._stopwords = self._nlp.Defaults.stop_words
                # Add common French filler words in rap
                self._stopwords.update({
                    'ouais', 'wesh', 'han', 'hein', 'euh', 'bah', 'ben',
                    'genre', 'style', 'truc', 'machin', 'nan', 'yo', 'hey'
                })

    def _process_token(self, token) -> Optional[str]:
        """Process a single spaCy token, returning cleaned form or None."""
        # Skip punctuation, spaces, numbers
        if token.is_punct or token.is_space or token.like_num:
            return None

        # Get token text (lemma or surface form)
        if self.lemmatize and token.lemma_:
            text = token.lemma_
        else:
            text = token.text

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Length filter
        if len(text) < self.min_word_length:
            return None

        # Stopword filter
        if self.remove_stopwords and text in self._stopwords:
            return None

        # Skip tokens that are purely non-alphabetic
        if not any(c.isalpha() for c in text):
            return None

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a single document.

        Parameters
        ----------
        text : str
            Document text.

        Returns
        -------
        List[str]
            List of filtered tokens.
        """
        self._load_model()

        if not text or not isinstance(text, str):
            return []

        doc = self._nlp(text)
        tokens = []
        for token in doc:
            processed = self._process_token(token)
            if processed:
                tokens.append(processed)

        return tokens

    def get_counts(self, text: str) -> Counter:
        """
        Get token counts for a single document.

        Parameters
        ----------
        text : str
            Document text.

        Returns
        -------
        Counter
            Token frequency counts.
        """
        return Counter(self.tokenize(text))

    def batch_tokenize(
        self,
        documents: List[str],
        verbose: bool = True
    ) -> List[List[str]]:
        """
        Batch tokenize documents using spaCy's pipe() for efficiency.

        This method is optimized for processing large document collections
        by using spaCy's parallel processing capabilities. Returns lists
        of tokens that can be reused for different distance computations.

        Parameters
        ----------
        documents : List[str]
            List of document texts.
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        List[List[str]]
            List of token lists for each document, in same order as input.
            Empty documents return empty lists.
        """
        self._load_model()

        n_docs = len(documents)
        if verbose:
            print(f"  Tokenizing {n_docs:,} documents with spaCy ({self.model_name})...")

        # Handle empty documents
        valid_docs = []
        valid_indices = []
        for i, doc in enumerate(documents):
            if doc and isinstance(doc, str) and len(doc.strip()) > 0:
                valid_docs.append(doc)
                valid_indices.append(i)

        if verbose:
            print(f"    {len(valid_docs):,} non-empty documents to process")

        # Determine number of processes
        n_process = self.n_process
        if n_process == -1:
            import os
            n_process = max(1, os.cpu_count() - 1)

        # Initialize with empty token lists
        doc_tokens: List[List[str]] = [[] for _ in range(n_docs)]

        try:
            # Try multiprocessing first
            processed = self._nlp.pipe(
                valid_docs,
                batch_size=self.batch_size,
                n_process=n_process
            )

            for idx, (doc_idx, spacy_doc) in enumerate(zip(valid_indices, processed)):
                tokens = []
                for token in spacy_doc:
                    processed_token = self._process_token(token)
                    if processed_token:
                        tokens.append(processed_token)
                doc_tokens[doc_idx] = tokens

                if verbose and (idx + 1) % 5000 == 0:
                    print(f"    Processed {idx + 1:,}/{len(valid_docs):,} documents")

        except Exception as e:
            # Fallback to single-process if multiprocessing fails
            if verbose:
                print(f"    Multiprocessing failed ({e}), using single process...")

            for idx, (doc_idx, text) in enumerate(zip(valid_indices, valid_docs)):
                doc_tokens[doc_idx] = self.tokenize(text)

                if verbose and (idx + 1) % 5000 == 0:
                    print(f"    Processed {idx + 1:,}/{len(valid_docs):,} documents")

        if verbose:
            total_tokens = sum(len(tokens) for tokens in doc_tokens)
            vocab_size = len(set(token for tokens in doc_tokens for token in tokens))
            print(f"  Tokenization complete: {total_tokens:,} tokens, {vocab_size:,} unique terms")

        return doc_tokens


class NLTKTokenizer:
    """
    NLTK-based tokenizer fallback (no spaCy dependency).

    Uses NLTK's word_tokenize with French language support, stopword removal,
    and basic lowercasing/filtering. Does NOT perform lemmatization (use
    SpaCyTokenizer for that).

    Parameters
    ----------
    lowercase : bool, default=True
        Convert tokens to lowercase.
    min_word_length : int, default=2
        Minimum token length to keep.
    remove_stopwords : bool, default=True
        Remove French stopwords.

    Examples
    --------
    >>> tokenizer = NLTKTokenizer()
    >>> tokens = tokenizer.batch_tokenize(["Le chat dort sur le canapé"])
    >>> tokens[0]
    ['chat', 'dort', 'canapé']
    """

    # Extended French stopwords (same as LDA script for consistency)
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
        'là', 'ça', 'cela', 'ceci', 'celui', 'celle', 'ceux',
        'oh', 'ah', 'eh', 'hé', 'ouais', 'yeah', 'yo', 'hey', 'ok', 'okay',
        'nan', 'non', 'oui', 'bah', 'ben', 'hein', 'genre',
        'moi', 'toi', 'soi', 'eux',
        # Filler words common in French rap
        'wesh', 'gros', 'frère', 'mec', 'poto', 'igo', 'izi', 'baby',
        'ouai', 'woh', 'wow', 'mmh', 'han',
    }

    def __init__(
        self,
        lowercase: bool = True,
        min_word_length: int = 2,
        remove_stopwords: bool = True,
    ):
        self.lowercase = lowercase
        self.min_word_length = min_word_length
        self.remove_stopwords = remove_stopwords
        self._word_tokenize = None

    def _load(self):
        """Lazy load NLTK resources."""
        if self._word_tokenize is None:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
            from nltk.tokenize import word_tokenize
            self._word_tokenize = word_tokenize

            # Also try to load NLTK French stopwords to augment ours
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            try:
                from nltk.corpus import stopwords
                self.FRENCH_STOPWORDS = self.FRENCH_STOPWORDS | set(stopwords.words('french'))
            except Exception:
                pass  # Use our built-in list

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a single document."""
        self._load()

        if not text or not isinstance(text, str):
            return []

        if self.lowercase:
            text = text.lower()

        tokens = self._word_tokenize(text, language='french')

        result = []
        for token in tokens:
            if len(token) < self.min_word_length:
                continue
            if not any(c.isalpha() for c in token):
                continue
            if self.remove_stopwords and token in self.FRENCH_STOPWORDS:
                continue
            result.append(token)

        return result

    def batch_tokenize(
        self,
        documents: List[str],
        verbose: bool = True
    ) -> List[List[str]]:
        """
        Batch tokenize documents using NLTK.

        Parameters
        ----------
        documents : List[str]
            List of document texts.
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        List[List[str]]
            Token lists for each document.
        """
        self._load()

        n_docs = len(documents)
        if verbose:
            print(f"  Tokenizing {n_docs:,} documents with NLTK (word_tokenize, French)...")

        doc_tokens: List[List[str]] = []
        for idx, doc in enumerate(documents):
            doc_tokens.append(self.tokenize(doc))
            if verbose and (idx + 1) % 5000 == 0:
                print(f"    Processed {idx + 1:,}/{n_docs:,} documents")

        if verbose:
            total_tokens = sum(len(tokens) for tokens in doc_tokens)
            vocab_size = len(set(token for tokens in doc_tokens for token in tokens))
            print(f"  Tokenization complete: {total_tokens:,} tokens, {vocab_size:,} unique terms")

        return doc_tokens


def batch_tokenize_documents(
    documents: List[str],
    tokenizer=None,
    model_name: str = 'fr_core_news_lg',
    **tokenizer_kwargs
) -> List[List[str]]:
    """
    Convenience function to batch tokenize documents.

    Parameters
    ----------
    documents : List[str]
        List of document texts.
    tokenizer : SpaCyTokenizer or NLTKTokenizer, optional
        Pre-configured tokenizer. If None, creates a SpaCyTokenizer.
    model_name : str or None, default='fr_core_news_lg'
        SpaCy model to use if creating new tokenizer.
        If None, uses NLTKTokenizer instead.
    **tokenizer_kwargs
        Additional arguments passed to tokenizer constructor.

    Returns
    -------
    List[List[str]]
        Token lists for each document.

    Examples
    --------
    >>> # SpaCy tokenization
    >>> tokens = batch_tokenize_documents(docs, model_name='fr_core_news_lg')
    >>>
    >>> # NLTK tokenization (no spaCy dependency)
    >>> tokens = batch_tokenize_documents(docs, model_name=None)
    """
    if tokenizer is None:
        if model_name is None:
            tokenizer = NLTKTokenizer(**tokenizer_kwargs)
        else:
            tokenizer = SpaCyTokenizer(model_name=model_name, **tokenizer_kwargs)

    return tokenizer.batch_tokenize(documents)


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

    Measures lexical similarity between two texts based on relative word
    frequencies. Standard metric in French stylometry and JADT community.

    Formula:
        D(A, B) = 0.5 * Σ|f_A(w) - f_B(w)|

    where f_A(w) and f_B(w) are the relative frequencies of word w in texts A and B.

    The distance is bounded [0, 1]:
    - 0: Identical word distributions
    - 1: No vocabulary overlap

    References
    ----------
    - Labbé, C., & Labbé, D. (2001). Inter-textual distance and authorship
      attribution. Journal of Quantitative Linguistics, 8(3), 213-231.
    - Labbé, D., & Monière, D. (2003). Le vocabulaire gouvernemental:
      Canada, Québec, France (1945-2000). Champion.

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
    0.333...  # 2 common words out of 3 unique
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
        Compute Labbé distance from pre-computed word counts (optimized).

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
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())

        if total1 == 0 or total2 == 0:
            return 1.0  # Maximum distance for empty texts

        # Union of vocabularies
        all_words = set(counts1.keys()) | set(counts2.keys())

        # Sum of absolute differences in relative frequencies
        total_diff = sum(
            abs(counts1.get(word, 0) / total1 - counts2.get(word, 0) / total2)
            for word in all_words
        )

        # Labbé distance is half the sum (bounded [0, 1])
        return total_diff / 2.0

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


class WMDDistance(BaseDistance):
    """
    Word Mover's Distance using FastText embeddings.

    WMD measures the minimum cumulative distance that words in one document
    need to "travel" to match words in another document, using word embeddings
    as the underlying space.

    STATUS: Stub implementation. Raises NotImplementedError.

    To be implemented with:
    - FastText French word embeddings (cc.fr.300.bin)
    - Gensim's wmdistance or custom EMD solver

    References
    ----------
    - Kusner, M., et al. (2015). From word embeddings to document distances.
      ICML, 957-966.

    Parameters
    ----------
    fasttext_path : str, optional
        Path to FastText binary model file.
    """

    def __init__(self, fasttext_path: Optional[str] = None):
        self.fasttext_path = fasttext_path
        self._model = None

    def compute(self, text1: str, text2: str) -> float:
        """
        Compute Word Mover's Distance between two texts.

        Raises
        ------
        NotImplementedError
            WMD + FastText implementation pending.
        """
        raise NotImplementedError(
            "WMD + FastText à implémenter. "
            "Requires: gensim, fasttext embeddings (cc.fr.300.bin)"
        )


def evaluate_topic_coherence(
    documents: List[str],
    topic_assignments: List[int],
    distance: BaseDistance,
    sample_size: int = 50,
    random_seed: Optional[int] = None,
    max_topics: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Evaluate intra-topic homogeneity using pairwise document distances.

    For each topic, computes the mean distance between documents assigned
    to that topic. Lower mean distance indicates more lexically homogeneous
    topics, suggesting better clustering quality.

    OPTIMIZED VERSION: Pre-tokenizes documents to avoid redundant processing.

    Parameters
    ----------
    documents : List[str]
        List of document texts.
    topic_assignments : List[int]
        Topic assignment for each document (same length as documents).
        Topic -1 is treated as outliers and excluded from analysis.
    distance : BaseDistance
        Distance metric instance to use for pairwise comparisons.
    sample_size : int, default=50
        Maximum number of document pairs to sample per topic.
        Set to -1 for exhaustive computation (may be slow for large topics).
    random_seed : int, optional
        Random seed for reproducible sampling.
    max_topics : int, optional
        Maximum number of topics to analyze (for very large topic sets).
        If None, all topics are analyzed.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'mean': float - Overall mean intra-topic distance
        - 'std': float - Standard deviation across topics
        - 'per_topic': Dict[int, dict] - Per-topic statistics with:
            - 'mean_distance': float
            - 'n_documents': int
            - 'n_pairs_sampled': int
        - 'distance_metric': str - Name of the distance metric used
        - 'total_documents': int - Number of documents analyzed (excluding outliers)
        - 'n_topics': int - Number of topics analyzed

    Notes
    -----
    - Topic -1 (BERTopic outliers) is automatically excluded.
    - For topics with only 1 document, distance is reported as 0.
    - Sampling is used for computational efficiency; increase sample_size
      for more precise estimates.

    Examples
    --------
    >>> from utils.comparaison_utils.topic_distances import (
    ...     LabbeDistance, evaluate_topic_coherence
    ... )
    >>> docs = ["le chat dort", "le chat mange", "le chien joue", "le chien court"]
    >>> topics = [0, 0, 1, 1]
    >>> dist = LabbeDistance()
    >>> results = evaluate_topic_coherence(docs, topics, dist)
    >>> print(f"Mean intra-topic distance: {results['mean']:.4f}")
    """
    if len(documents) != len(topic_assignments):
        raise ValueError(
            f"Length mismatch: {len(documents)} documents vs "
            f"{len(topic_assignments)} topic assignments"
        )

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Group document indices by topic (exclude outliers: topic -1)
    topic_doc_indices: Dict[int, List[int]] = {}
    for idx, topic in enumerate(topic_assignments):
        if topic == -1:
            continue  # Skip BERTopic outliers
        if topic not in topic_doc_indices:
            topic_doc_indices[topic] = []
        topic_doc_indices[topic].append(idx)

    if not topic_doc_indices:
        return {
            'mean': 0.0,
            'std': 0.0,
            'per_topic': {},
            'distance_metric': distance.__class__.__name__,
            'total_documents': 0,
            'n_topics': 0
        }

    # Limit number of topics if specified
    topic_ids = sorted(topic_doc_indices.keys())
    if max_topics is not None and len(topic_ids) > max_topics:
        topic_ids = topic_ids[:max_topics]
        if verbose:
            print(f"    Limiting analysis to {max_topics} topics")

    # Pre-compute word counts for all documents in selected topics
    # This is the key optimization: tokenize each document only once
    if verbose:
        print(f"    Pre-tokenizing documents...")

    doc_counts: Dict[int, Counter] = {}
    indices_needed = set()
    for topic_id in topic_ids:
        indices_needed.update(topic_doc_indices[topic_id])

    for idx in indices_needed:
        doc_text = documents[idx]
        if hasattr(distance, 'get_counts'):
            doc_counts[idx] = distance.get_counts(doc_text)
        else:
            # Fallback for distances without get_counts
            doc_counts[idx] = None

    # Compute intra-topic distances
    per_topic_results: Dict[int, dict] = {}
    all_topic_means: List[float] = []

    for topic_idx, topic_id in enumerate(topic_ids):
        doc_indices = topic_doc_indices[topic_id]
        n_docs = len(doc_indices)

        if verbose and topic_idx % 10 == 0:
            print(f"    Processing topic {topic_id} ({topic_idx + 1}/{len(topic_ids)}), {n_docs} docs")

        if n_docs < 2:
            # Single document: distance is 0 (perfectly homogeneous)
            per_topic_results[topic_id] = {
                'mean_distance': 0.0,
                'n_documents': n_docs,
                'n_pairs_sampled': 0
            }
            all_topic_means.append(0.0)
            continue

        # Generate pairs using itertools (memory efficient)
        n_total_pairs = n_docs * (n_docs - 1) // 2

        # Sample pairs if needed
        if sample_size > 0 and n_total_pairs > sample_size:
            # Random sampling without generating all pairs
            sampled_pairs = set()
            max_attempts = sample_size * 3  # Avoid infinite loop
            attempts = 0
            while len(sampled_pairs) < sample_size and attempts < max_attempts:
                i = random.randint(0, n_docs - 1)
                j = random.randint(0, n_docs - 1)
                if i != j:
                    pair = (min(i, j), max(i, j))
                    sampled_pairs.add(pair)
                attempts += 1
            sampled_pairs = list(sampled_pairs)
        else:
            # Use itertools.combinations (memory efficient iterator)
            sampled_pairs = list(itertools.combinations(range(n_docs), 2))

        # Compute distances using pre-computed counts
        distances = []
        for i, j in sampled_pairs:
            idx_i = doc_indices[i]
            idx_j = doc_indices[j]

            counts_i = doc_counts.get(idx_i)
            counts_j = doc_counts.get(idx_j)

            if counts_i is not None and counts_j is not None:
                # Use optimized compute_from_counts
                dist = distance.compute_from_counts(counts_i, counts_j)
            else:
                # Fallback to text-based computation
                dist = distance.compute(documents[idx_i], documents[idx_j])

            distances.append(dist)

        mean_dist = np.mean(distances) if distances else 0.0

        per_topic_results[topic_id] = {
            'mean_distance': float(mean_dist),
            'n_documents': n_docs,
            'n_pairs_sampled': len(sampled_pairs)
        }
        all_topic_means.append(mean_dist)

    # Aggregate statistics
    total_docs = sum(len(topic_doc_indices[t]) for t in topic_ids)

    return {
        'mean': float(np.mean(all_topic_means)) if all_topic_means else 0.0,
        'std': float(np.std(all_topic_means)) if all_topic_means else 0.0,
        'per_topic': per_topic_results,
        'distance_metric': distance.__class__.__name__,
        'total_documents': total_docs,
        'n_topics': len(topic_ids)
    }


def evaluate_topic_coherence_from_tokens(
    doc_tokens: List[List[str]],
    topic_assignments: List[int],
    distance_type: str = 'both',
    sample_size: int = 50,
    random_seed: Optional[int] = None,
    max_topics: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, dict]:
    """
    Evaluate intra-topic homogeneity using PRE-TOKENIZED documents.

    This function is optimized for computing distances across multiple models
    without re-tokenizing documents. Tokenize once with SpaCyTokenizer, then
    use this function for each model's topic assignments.

    Parameters
    ----------
    doc_tokens : List[List[str]]
        Pre-tokenized documents (from SpaCyTokenizer.batch_tokenize).
        Each element is a list of tokens for one document.
    topic_assignments : List[int]
        Topic assignment for each document. Topic -1 is treated as outliers.
    distance_type : str, default='both'
        Which distances to compute:
        - 'js': Jensen-Shannon only
        - 'labbe': Labbé only
        - 'both': Both distances
    sample_size : int, default=50
        Maximum document pairs to sample per topic.
    random_seed : int, optional
        Random seed for reproducible sampling.
    max_topics : int, optional
        Maximum number of topics to analyze.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'js': Jensen-Shannon results (if requested)
        - 'labbe': Labbé results (if requested)
        Each sub-dict has: mean, std, n_topics, per_topic

    Examples
    --------
    >>> # Tokenize all documents once
    >>> tokenizer = SpaCyTokenizer(model_name='fr_core_news_lg')
    >>> doc_tokens = tokenizer.batch_tokenize(all_documents)
    >>>
    >>> # Compute distances for BERTopic assignments
    >>> results_bert = evaluate_topic_coherence_from_tokens(doc_tokens, bert_topics)
    >>>
    >>> # Reuse same tokens for LDA assignments
    >>> results_lda = evaluate_topic_coherence_from_tokens(doc_tokens, lda_topics)
    >>>
    >>> # And for IRAMUTEQ
    >>> results_ira = evaluate_topic_coherence_from_tokens(doc_tokens, iramuteq_topics)
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
    for idx, topic in enumerate(topic_assignments):
        if topic == -1:
            continue
        if topic not in topic_doc_indices:
            topic_doc_indices[topic] = []
        topic_doc_indices[topic].append(idx)

    results = {}

    if not topic_doc_indices:
        empty_result = {
            'mean': 0.0, 'std': 0.0, 'per_topic': {}, 'n_topics': 0, 'total_documents': 0
        }
        if distance_type in ('js', 'both'):
            results['js'] = empty_result.copy()
        if distance_type in ('labbe', 'both'):
            results['labbe'] = empty_result.copy()
        return results

    topic_ids = sorted(topic_doc_indices.keys())
    if max_topics is not None and len(topic_ids) > max_topics:
        topic_ids = topic_ids[:max_topics]

    # Pre-compute token counts for all needed documents
    # (Only compute Counter once per document, even if it appears in multiple comparisons)
    if verbose:
        print("  Pre-computing token counts from token lists...")

    indices_needed = set()
    for topic_id in topic_ids:
        indices_needed.update(topic_doc_indices[topic_id])

    doc_counts: Dict[int, Counter] = {}
    for idx in indices_needed:
        doc_counts[idx] = Counter(doc_tokens[idx])

    # Helper function to compute mean distances for a metric
    def compute_distances_for_metric(compute_fn: Callable, metric_name: str) -> dict:
        per_topic_results: Dict[int, dict] = {}
        all_topic_means: List[float] = []

        for topic_idx, topic_id in enumerate(topic_ids):
            doc_indices = topic_doc_indices[topic_id]
            n_docs = len(doc_indices)

            if verbose and topic_idx % 10 == 0:
                print(f"    [{metric_name}] Topic {topic_id} ({topic_idx + 1}/{len(topic_ids)})")

            if n_docs < 2:
                per_topic_results[topic_id] = {
                    'mean_distance': 0.0, 'n_documents': n_docs, 'n_pairs_sampled': 0
                }
                all_topic_means.append(0.0)
                continue

            # Sample pairs
            n_total_pairs = n_docs * (n_docs - 1) // 2
            if sample_size > 0 and n_total_pairs > sample_size:
                sampled_pairs = set()
                max_attempts = sample_size * 3
                attempts = 0
                while len(sampled_pairs) < sample_size and attempts < max_attempts:
                    i = random.randint(0, n_docs - 1)
                    j = random.randint(0, n_docs - 1)
                    if i != j:
                        pair = (min(i, j), max(i, j))
                        sampled_pairs.add(pair)
                    attempts += 1
                sampled_pairs = list(sampled_pairs)
            else:
                sampled_pairs = list(itertools.combinations(range(n_docs), 2))

            # Compute distances using pre-computed counts
            distances = []
            for i, j in sampled_pairs:
                idx_i = doc_indices[i]
                idx_j = doc_indices[j]
                dist = compute_fn(doc_counts[idx_i], doc_counts[idx_j])
                distances.append(dist)

            mean_dist = np.mean(distances) if distances else 0.0
            per_topic_results[topic_id] = {
                'mean_distance': float(mean_dist),
                'n_documents': n_docs,
                'n_pairs_sampled': len(sampled_pairs)
            }
            all_topic_means.append(mean_dist)

        total_docs = sum(len(topic_doc_indices[t]) for t in topic_ids)

        return {
            'mean': float(np.mean(all_topic_means)) if all_topic_means else 0.0,
            'std': float(np.std(all_topic_means)) if all_topic_means else 0.0,
            'per_topic': per_topic_results,
            'n_topics': len(topic_ids),
            'total_documents': total_docs
        }

    # Define distance computation functions using Counters
    def labbe_from_counts(c1: Counter, c2: Counter) -> float:
        """Labbé distance from token counts."""
        total1 = sum(c1.values())
        total2 = sum(c2.values())
        if total1 == 0 or total2 == 0:
            return 1.0
        all_words = set(c1.keys()) | set(c2.keys())
        total_diff = sum(
            abs(c1.get(word, 0) / total1 - c2.get(word, 0) / total2)
            for word in all_words
        )
        return total_diff / 2.0

    def js_from_counts(c1: Counter, c2: Counter) -> float:
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

    # Compute requested distances
    if distance_type in ('labbe', 'both'):
        if verbose:
            print("  Computing Labbé distances...")
        results['labbe'] = compute_distances_for_metric(labbe_from_counts, 'Labbé')

    if distance_type in ('js', 'both'):
        if verbose:
            print("  Computing Jensen-Shannon distances...")
        results['js'] = compute_distances_for_metric(js_from_counts, 'JS')

    return results
