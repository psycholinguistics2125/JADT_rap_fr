import pandas as pd
import numpy as np
import re

def remove_non_song(df):
    """
    Remove non-song entries from the French rap lyrics dataset.

    Uses multiple detection rules with mask logic to identify and flag non-songs:
    1. Missing/incomplete lyrics (keywords + short length)
    2. Extremely short content (< 30 words)
    3. Missing data (null/empty lyrics)
    4. Pure instrumental tracks
    5. Low vocabulary richness (< 10% unique words)
    6. Mostly English content (< 30% French)
    7. Freestyle compilations (> 3000 words)

    Parameters:
    -----------
    df : pandas.DataFrame
        French rap lyrics dataset with columns:
        - lyrics: song lyrics text
        - title: song title
        - n_words: word count
        - n_lines: line count
        - n_unique_words: unique word count
        - n_french_words: count of French words
        - n_non_french_words: count of non-French words

    Returns:
    --------
    df : pandas.DataFrame
        Same dataframe with 'is_song_candidate' column added/modified
        (True = valid song, False = non-song/removed)

    Output:
    -------
    Prints detailed statistics about flagged entries
    """

    print("=" * 80)
    print("REMOVING NON-SONG ENTRIES - COMPREHENSIVE DETECTION")
    print("=" * 80)
    print(f"\nStarting dataset size: {len(df):,} entries\n")

    # Initialize the flag column
    df['is_song_candidate'] = True

    # ========================================================================
    # RULE 1: Missing/Incomplete lyrics indicators
    # ========================================================================
    print("--- RULE 1: Missing/Incomplete lyrics ---")

    missing_keywords = [
        r'lyrics à venir',
        r'à venir',
        r'paroles à venir',
        r'yet to be transcribed',
        r'à compléter',
        r'\(à compléter\)',
        r'missing song info',
        r'paroles manquantes',
        r'lyrics coming soon',
        r'cette piste est une instrumentale',
        r'this track is instrumental',
        r'équipe rap genius france',
        r'équipe genius france',
        r'cliquez ici pour'
    ]

    pattern_missing = '|'.join(missing_keywords)
    has_missing_keywords = df['lyrics'].str.lower().str.contains(pattern_missing, na=False, regex=True)
    is_very_short = df['n_words'] < 100

    # ONLY flag if BOTH conditions are true
    mask_missing = has_missing_keywords & is_very_short
    df.loc[mask_missing, 'is_song_candidate'] = False
    print(f"   Flagged {mask_missing.sum()}: short and non-content music related")

    # ========================================================================
    # RULE 2: Extremely short content
    # ========================================================================
    print("\n--- RULE 2: Extremely short content ---")

    mask_too_short = (
        (df['n_words'] < 30) | 
        (df['n_lines'] < 2) |
        (df['lyrics'].str.len() < 100)
    )
    df.loc[mask_too_short, 'is_song_candidate'] = False
    print(f"   Flagged {mask_too_short.sum()}: very short songs")

    # ========================================================================
    # RULE 3: Missing critical data
    # ========================================================================
    print("\n--- RULE 3: Missing/null data ---")

    mask_missing_data = (
        df['lyrics'].isna() | 
        (df['lyrics'].str.strip() == '') |
        (df['n_words'] == 0)
    )
    df.loc[mask_missing_data, 'is_song_candidate'] = False
    print(f"   Flagged {mask_missing_data.sum()}: entries with missing/null lyrics data")

    # ========================================================================
    # RULE 4: Pure instrumental tracks
    # ========================================================================
    print("\n--- RULE 4: Pure instrumental tracks ---")

    instrumental_keywords = [
        r'^instrumental$',
        r'^instrumentale$',
        r'^\[instrumental\]$',
        r'^\(instrumental\)$'
    ]
    pattern_instrumental = '|'.join(instrumental_keywords)
    mask_instrumental = df['lyrics'].str.lower().str.strip().str.match(pattern_instrumental, na=False)
    df.loc[mask_instrumental, 'is_song_candidate'] = False
    print(f"   Flagged {mask_instrumental.sum()}: pure instrumental tracks")

    # ========================================================================
    # RULE 5: Low vocabulary richness
    # ========================================================================
    print("\n--- RULE 5: Low vocabulary richness ---")

    df['vocab_richness'] = df['n_unique_words'] / (df['n_words'] + 1)
    mask_low_vocab_richness = (df['vocab_richness'] < 0.1)
    df.loc[mask_low_vocab_richness, 'is_song_candidate'] = False
    print(f"   Flagged {mask_low_vocab_richness.sum()}: entries with low vocabulary richness (< 10% unique)")

    # ========================================================================
    # RULE 6: Mostly non-French content
    # ========================================================================
    print("\n--- RULE 6: Mostly English content ---")

    df['french_ratio'] = df['n_french_words'] / (df['n_french_words'] + df['n_non_french_words']).replace(0, 1)

    # Non-French content (< 30% French = > 70% English)
    mostly_english_mask = (df['french_ratio'] < 0.3)
    df.loc[mostly_english_mask, 'is_song_candidate'] = False
    print(f"   Flagged {mostly_english_mask.sum()}: more than 70% English content")

    # ========================================================================
    # RULE 7: Freestyle compilations (very long freestyles)
    # ========================================================================
    print("\n--- RULE 7: Freestyle compilations ---")

    freestyle_keywords = [
        r'freestyle',
        r'free-?style',
        r'freestyler'
    ]
    pattern_freestyle = '|'.join(freestyle_keywords)
    has_freestyle_mask = df['title'].str.lower().str.contains(pattern_freestyle, na=False, regex=True)
    is_very_long_mask = df['n_words'] > 3000

    mask_freestyle_long = has_freestyle_mask & is_very_long_mask
    df.loc[mask_freestyle_long, 'is_song_candidate'] = False
    print(f"   Flagged {mask_freestyle_long.sum()}: entries with 'freestyle' in title and > 3000 words")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    num_removed = (~df['is_song_candidate']).sum()
    num_kept = df['is_song_candidate'].sum()
    total = len(df)
    pct_removed = (num_removed / total * 100) if total > 0 else 0

    print(f"Total entries:        {total:,}")
    print(f"Valid songs (kept):   {num_kept:,}")
    print(f"Non-songs (removed):  {num_removed:,} ({pct_removed:.2f}%)")
    print("=" * 80)

    cleaned_df = df[df.is_song_candidate]

    return cleaned_df



def clean_lyrics_text(lyrics):
    """
    Clean individual lyrics text by removing non-lyric content.

    Removes:
    - Section labels ([Verse], [Chorus], etc.)
    - Producer/writer credits
    - Repetition annotations
    - Ad-lib markers
    - Sound effects
    - Genius editorial content
    - Excessive whitespace

    Args:
        lyrics (str): Raw lyrics text

    Returns:
        str: Cleaned lyrics text
    """
    if pd.isna(lyrics) or lyrics.strip() == '':
        return lyrics

    text = lyrics

    # A. Remove section labels (but keep the actual lyrics)
    # Pattern: [Intro], [Verse 1], [Couplet 1], [Refrain], etc.
    section_patterns = [
        r'\[Intro[^\]]*\]',
        r'\[Outro[^\]]*\]',
        r'\[Verse[^\]]*\]',
        r'\[Couplet[^\]]*\]',
        r'\[Refrain[^\]]*\]',
        r'\[Chorus[^\]]*\]',
        r'\[Bridge[^\]]*\]',
        r'\[Pont[^\]]*\]',
        r'\[Hook[^\]]*\]',
        r'\[Pre-[^\]]*\]',
        r'\[Post-[^\]]*\]',
        r'\[Instrumental[^\]]*\]',
        r'\[Beat[^\]]*\]',
        r'\[Sample[^\]]*\]',
        r'\[Skit[^\]]*\]',
        r'\[Interlude[^\]]*\]',
    ]

    for pattern in section_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # B. Remove producer/writer credits (usually at start or end)
    credit_patterns = [
        r'Produced by[^\n]*',
        r'Produit par[^\n]*',
        r'Written by[^\n]*',
        r'Écrit par[^\n]*',
        r'Paroles\s*:[^\n]*',
        r'Lyrics\s*:[^\n]*',
        r'Compositeur[^\n]*',
        r'Auteur[^\n]*',
    ]

    for pattern in credit_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # C. Remove repetition annotations
    # Pattern: (x2), (×2), (2x), etc.
    text = re.sub(r'\([×x]?\d+[×x]?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(répété[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(repeat[^)]*\)', '', text, flags=re.IGNORECASE)

    # D. Remove ad-lib/annotation markers (but keep the actual words if sung)
    # Pattern: (ad-lib), (ad lib), etc.
    text = re.sub(r'\(ad[-\s]?lib[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(background vocals?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(voix de fond\)', '', text, flags=re.IGNORECASE)

    # E. Remove sound effect descriptions (keep actual sung words)
    # Pattern: *rires*, *laughs*, *gunshot*, etc.
    text = re.sub(r'\*[^*]+\*', '', text)

    # F. Remove Genius editorial content
    editorial_patterns = [
        r'Cliquez ici[^\n]*',
        r'Voir la traduction[^\n]*',
        r'Voir les paroles[^\n]*',
        r'Équipe Rap Genius[^\n]*',
        r'Équipe Genius[^\n]*',
    ]

    for pattern in editorial_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # G. Remove excessive whitespace while preserving line structure
    # Remove spaces at start/end of lines
    lines = text.split('\n')
    lines = [line.strip() for line in lines]

    # Remove multiple consecutive blank lines (keep max 1 blank line between sections)
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == '':
            blank_count += 1
            if blank_count <= 1:  # Keep first blank line
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # H. Remove leading/trailing whitespace
    text = text.strip()

    return text


def clean_lyrics_column(df):
    """
    Apply lyrics cleaning to the entire dataframe.

    Args:
        df (pd.DataFrame): Dataframe with 'lyrics' column

    Returns:
        pd.DataFrame: Same dataframe with 'lyrics_cleaned' column added
    """
    print("\nCleaning lyrics text...")
    df['lyrics_cleaned'] = df['lyrics'].apply(clean_lyrics_text)

    # Calculate cleaning statistics
    original_chars = df['lyrics'].str.len().sum()
    cleaned_chars = df['lyrics_cleaned'].str.len().sum()
    removed_chars = original_chars - cleaned_chars
    pct_removed = (removed_chars / original_chars * 100) if original_chars > 0 else 0

    print(f"   Removed {removed_chars:,} characters ({pct_removed:.2f}% of total)")
    print(f"   Original: {original_chars:,} characters")
    print(f"   Cleaned:  {cleaned_chars:,} characters")

    return df


def sample_and_verify_cleaning(df, sample_size=15, random_seed=42):
    """
    Sample songs and display before/after cleaning for quality verification.

    Shows side-by-side comparison of original and cleaned lyrics to verify
    that the cleaning process is working correctly and not removing important content.

    Args:
        df (pd.DataFrame): Dataframe with 'lyrics' and 'lyrics_cleaned' columns
        sample_size (int): Number of samples to display (default: 10)
        random_seed (int): Random seed for reproducibility (default: 42)

    Returns:
        pd.DataFrame: Sample dataframe used for verification
    """

    # Check if lyrics_cleaned column exists
    if 'lyrics_cleaned' not in df.columns:
        print("ERROR: 'lyrics_cleaned' column not found. Run clean_lyrics_column() first.")
        return None

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Sample songs (filter out very short ones for better verification)
    valid_for_sampling = df[df['lyrics'].str.len() > 100].copy()

    if len(valid_for_sampling) < sample_size:
        sample_size = len(valid_for_sampling)
        print(f"Warning: Only {sample_size} songs with enough content for sampling")

    sample_df = valid_for_sampling.sample(n=sample_size, random_state=random_seed)

    print("=" * 100)
    print("CLEANING QUALITY VERIFICATION - SAMPLE COMPARISON")
    print("=" * 100)
    print(f"\nShowing {sample_size} random samples to verify cleaning quality\n")

    for idx, (index, row) in enumerate(sample_df.iterrows(), 1):
        artist = row.get('artist', 'Unknown')
        title = row.get('title', 'Unknown')
        original = row['lyrics']
        cleaned = row['lyrics_cleaned']

        # Calculate statistics for this song
        orig_len = len(original)
        cleaned_len = len(cleaned)
        removed_pct = (orig_len - cleaned_len) / orig_len * 100 if orig_len > 0 else 0

        print("-" * 100)
        print(f"SAMPLE {idx}/{sample_size}: {artist} - {title}")
        print(f"Characters: {orig_len:,} → {cleaned_len:,} ({removed_pct:.1f}% removed)")
        print("-" * 100)

        print("\nORIGINAL (first 300 chars):")
        print(original[:300])
        print("...")

        print("\nCLEANED (first 300 chars):")
        print(cleaned[:300])
        print("...")

        print("\n")

    # Summary statistics
    print("=" * 100)
    print("SUMMARY STATISTICS FOR SAMPLE")
    print("=" * 100)

    sample_df['original_len'] = sample_df['lyrics'].str.len()
    sample_df['cleaned_len'] = sample_df['lyrics_cleaned'].str.len()
    sample_df['removed_pct'] = (sample_df['original_len'] - sample_df['cleaned_len']) / sample_df['original_len'] * 100

    print(f"\nAverage characters removed: {sample_df['removed_pct'].mean():.2f}%")
    print(f"Median characters removed:   {sample_df['removed_pct'].median():.2f}%")
    print(f"Min characters removed:      {sample_df['removed_pct'].min():.2f}%")
    print(f"Max characters removed:      {sample_df['removed_pct'].max():.2f}%")

    print(f"\nOriginal avg song length: {sample_df['original_len'].mean():.0f} chars")
    print(f"Cleaned avg song length:   {sample_df['cleaned_len'].mean():.0f} chars")

    print("\n" + "=" * 100)

    return sample_df


def detailed_sample_inspection(df, artist=None, title=None, index=None, display_all=False):
    """
    Detailed inspection of a single song's cleaning.

    Useful for debugging specific songs or examining edge cases.

    Args:
        df (pd.DataFrame): Dataframe with 'lyrics' and 'lyrics_cleaned' columns
        artist (str): Artist name to search for
        title (str): Song title to search for
        index (int): Direct index of song to inspect
        display_all (bool): If True, show entire lyrics (not truncated)

    Returns:
        None (prints detailed comparison)
    """

    if 'lyrics_cleaned' not in df.columns:
        print("ERROR: 'lyrics_cleaned' column not found.")
        return

    # Find the song
    if index is not None:
        row = df.iloc[index]
    elif artist and title:
        matches = df[(df['artist'].str.contains(artist, case=False, na=False)) & 
                     (df['title'].str.contains(title, case=False, na=False))]
        if len(matches) == 0:
            print(f"No song found with artist='{artist}' and title='{title}'")
            return
        row = matches.iloc[0]
    else:
        print("Please provide either index or (artist, title)")
        return

    original = row['lyrics']
    cleaned = row['lyrics_cleaned']

    print("=" * 100)
    print("DETAILED CLEANING INSPECTION")
    print("=" * 100)
    print(f"\nArtist: {row.get('artist', 'Unknown')}")
    print(f"Title:  {row.get('title', 'Unknown')}")
    print(f"Original length: {len(original):,} characters")
    print(f"Cleaned length:  {len(cleaned):,} characters")
    print(f"Removed: {len(original) - len(cleaned):,} characters ({(len(original) - len(cleaned)) / len(original) * 100:.2f}%)")

    print("\n" + "=" * 100)
    print("ORIGINAL LYRICS:")
    print("=" * 100)
    if display_all:
        print(original)
    else:
        print(original[:1000])
        if len(original) > 1000:
            print("\n... (truncated) ...[showing first 1000 chars]\n")

    print("\n" + "=" * 100)
    print("CLEANED LYRICS:")
    print("=" * 100)
    if display_all:
        print(cleaned)
    else:
        print(cleaned[:1000])
        if len(cleaned) > 1000:
            print("\n... (truncated) ...[showing first 1000 chars]\n")

    print("\n" + "=" * 100)


def estimate_year_from_artist_patterns(artist, df_ref):
    """Add df_ref parameter"""
    artist_songs = df_ref[(df_ref['artist'] == artist) & (df_ref['year'].notna())]
    
    if len(artist_songs) > 0:
        years = pd.to_numeric(artist_songs['year'], errors='coerce').dropna()
        if len(years) > 0:
            return int(years.median())
    
    return None
