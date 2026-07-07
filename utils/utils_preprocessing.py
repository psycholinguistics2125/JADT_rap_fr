#!/usr/bin/env python3
"""
Shared Preprocessing Utilities for Topic Modeling
=================================================
Deduplication of documents before topic modeling.

Motivation (Antoniak, 2022, "Topic Modeling for the People"):
duplicate and near-duplicate documents severely damage topic coherence and are
the first thing to check when topics look bad. This is especially relevant for
rap lyrics, where choruses/hooks and ad-libs are repeated within and across songs.
"""

import re

import numpy as np
import pandas as pd


def _normalize_signature(text) -> str:
    """Lowercase and collapse punctuation/whitespace to a comparable signature."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _minhash_keep_mask(series, threshold: float = 0.8, num_perm: int = 128) -> pd.Series:
    """Near-duplicate detection via MinHash/LSH. Requires the `datasketch` package."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as e:
        raise ImportError(
            "method='minhash' requires the 'datasketch' package "
            "(pip install datasketch)."
        ) from e

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = np.ones(len(series), dtype=bool)
    for i, text in enumerate(series):
        tokens = set(_normalize_signature(text).split())
        if not tokens:
            continue  # empty docs are kept (treated as unique)
        mh = MinHash(num_perm=num_perm)
        for tok in tokens:
            mh.update(tok.encode("utf8"))
        if lsh.query(mh):
            keep[i] = False  # a near-duplicate of an earlier doc already indexed
        else:
            lsh.insert(str(i), mh)
    return pd.Series(keep, index=series.index)


def deduplicate_documents(df: pd.DataFrame, text_column: str = "lyrics_cleaned",
                          method: str = "exact") -> tuple:
    """
    Remove duplicate documents, keeping the first occurrence.

    Args:
        method:
            - "exact": drop rows whose normalized text (lowercased, punctuation and
              whitespace collapsed) is identical to an earlier row. No extra deps.
            - "minhash": near-duplicate removal via MinHash/LSH (needs datasketch).

    Returns:
        (df_dedup, stats) where df_dedup preserves the original positional order
        (index reset) and carries a 'source_position' column recording each row's
        position in the input df, so downstream code can map back to the full
        corpus for cross-model alignment. stats reports the counts.
    """
    df = df.reset_index(drop=True)
    n_before = len(df)
    positions = np.arange(n_before)

    if method == "minhash":
        keep_mask = _minhash_keep_mask(df[text_column]).values
    else:
        sig = df[text_column].map(_normalize_signature)
        is_empty = (sig.str.len() == 0).values
        # keep first occurrence of each signature; never collapse empty documents
        keep_mask = (~sig.duplicated(keep="first")).values | is_empty

    df_dedup = df[keep_mask].copy()
    df_dedup["source_position"] = positions[keep_mask]
    df_dedup = df_dedup.reset_index(drop=True)

    stats = {
        "method": method,
        "n_before": int(n_before),
        "n_after": int(len(df_dedup)),
        "n_removed": int(n_before - len(df_dedup)),
    }
    return df_dedup, stats
