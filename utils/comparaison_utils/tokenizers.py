#!/usr/bin/env python3
"""
Tokenizers for Topic Distance Analysis
=======================================
Provides three tokenizer implementations with a common interface
(tokenize / batch_tokenize) for use in topic distance evaluation.

Classes:
--------
- SpaCyTokenizer: Accurate, POS-aware tokenization (requires spaCy)
- NLTKTokenizer: Lighter alternative (requires NLTK)
- SimpleSpaceTokenizer: Fast space-split tokenizer (no NLP dependencies)
"""

from collections import Counter
from typing import List, Optional, Tuple


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

    def batch_tokenize_dual(
        self,
        documents: List[str],
        verbose: bool = True
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Tokenize documents returning BOTH surface forms and lemmas in one SpaCy pass.

        Uses a single SpaCy pipe() call. For each valid token (passes all filters),
        appends both the surface form and the lemma to separate output lists.

        Parameters
        ----------
        documents : List[str]
            List of document texts.
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        Tuple[List[List[str]], List[List[str]]]
            (surface_tokens, lemma_tokens) — same length, same filtering.
        """
        self._load_model()

        n_docs = len(documents)
        if verbose:
            print(f"  Dual-tokenizing {n_docs:,} documents (surface + lemma) with spaCy ({self.model_name})...")

        valid_docs = []
        valid_indices = []
        for i, doc in enumerate(documents):
            if doc and isinstance(doc, str) and len(doc.strip()) > 0:
                valid_docs.append(doc)
                valid_indices.append(i)

        if verbose:
            print(f"    {len(valid_docs):,} non-empty documents to process")

        n_process = self.n_process
        if n_process == -1:
            import os
            n_process = max(1, os.cpu_count() - 1)

        surface_tokens: List[List[str]] = [[] for _ in range(n_docs)]
        lemma_tokens: List[List[str]] = [[] for _ in range(n_docs)]

        def _process_doc_dual(spacy_doc):
            surface = []
            lemma = []
            for token in spacy_doc:
                if token.is_punct or token.is_space or token.like_num:
                    continue
                text = token.text
                lemma_text = token.lemma_ if token.lemma_ else text
                if self.lowercase:
                    text = text.lower()
                    lemma_text = lemma_text.lower()
                if len(text) < self.min_word_length:
                    continue
                if self.remove_stopwords and text in self._stopwords:
                    continue
                if not any(c.isalpha() for c in text):
                    continue
                surface.append(text)
                lemma.append(lemma_text)
            return surface, lemma

        try:
            processed = self._nlp.pipe(
                valid_docs,
                batch_size=self.batch_size,
                n_process=n_process
            )
            for idx, (doc_idx, spacy_doc) in enumerate(zip(valid_indices, processed)):
                s, l = _process_doc_dual(spacy_doc)
                surface_tokens[doc_idx] = s
                lemma_tokens[doc_idx] = l
                if verbose and (idx + 1) % 5000 == 0:
                    print(f"    Processed {idx + 1:,}/{len(valid_docs):,} documents")

        except Exception as e:
            if verbose:
                print(f"    Multiprocessing failed ({e}), using single process...")
            for idx, (doc_idx, text) in enumerate(zip(valid_indices, valid_docs)):
                doc = self._nlp(text)
                s, l = _process_doc_dual(doc)
                surface_tokens[doc_idx] = s
                lemma_tokens[doc_idx] = l
                if verbose and (idx + 1) % 5000 == 0:
                    print(f"    Processed {idx + 1:,}/{len(valid_docs):,} documents")

        if verbose:
            vocab_surface = len(set(tok for t in surface_tokens for tok in t))
            vocab_lemma = len(set(tok for t in lemma_tokens for tok in t))
            total = sum(len(t) for t in surface_tokens)
            print(f"  Surface: {total:,} tokens, {vocab_surface:,} unique")
            print(f"  Lemma:   {total:,} tokens, {vocab_lemma:,} unique")

        return surface_tokens, lemma_tokens


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


class SimpleSpaceTokenizer:
    """
    Simple space-based tokenizer for fast testing (no NLP dependencies).

    This tokenizer is designed for rapid testing and debugging. It performs
    basic space-splitting with minimal filtering (lowercase, min length,
    stopwords). No lemmatization, POS tagging, or linguistic analysis.

    Use this for quick iterations when SpaCy/NLTK tokenization is too slow.
    For production analysis, prefer SpaCyTokenizer or NLTKTokenizer.

    Parameters
    ----------
    lowercase : bool, default=True
        Convert tokens to lowercase.
    min_word_length : int, default=2
        Minimum token length to keep.
    remove_stopwords : bool, default=True
        Remove common French stopwords.

    Examples
    --------
    >>> tokenizer = SimpleSpaceTokenizer()
    >>> tokens = tokenizer.batch_tokenize(["Le chat dort sur le canapé"])
    >>> tokens[0]
    ['chat', 'dort', 'canapé']
    """

    # Minimal stopwords for French (most common function words)
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
        # Common rap filler words
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

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a single document using space splitting."""
        if not text or not isinstance(text, str):
            return []

        if self.lowercase:
            text = text.lower()

        # Simple space split + basic punctuation removal
        import re
        # Remove punctuation but keep apostrophes within words
        text = re.sub(r"[^\w\s'-]", ' ', text)
        tokens = text.split()

        result = []
        for token in tokens:
            # Strip leading/trailing punctuation
            token = token.strip("'-")
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
        Batch tokenize documents using simple space splitting.

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
        n_docs = len(documents)
        if verbose:
            print(f"  Tokenizing {n_docs:,} documents with simple space tokenizer...")

        doc_tokens: List[List[str]] = []
        for idx, doc in enumerate(documents):
            doc_tokens.append(self.tokenize(doc))
            if verbose and (idx + 1) % 10000 == 0:
                print(f"    Processed {idx + 1:,}/{n_docs:,} documents")

        if verbose:
            total_tokens = sum(len(tokens) for tokens in doc_tokens)
            vocab_size = len(set(token for tokens in doc_tokens for token in tokens))
            print(f"  Tokenization complete: {total_tokens:,} tokens, {vocab_size:,} unique terms")

        return doc_tokens
