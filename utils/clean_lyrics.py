import pandas as pd
import re
from collections import Counter


# ============================================================================
# FRENCH APOSTROPHE CONTRACTIONS DATABASE
# ============================================================================

# Most common French apostrophe contractions in rap lyrics
FRENCH_APOSTROPHE_EXPANSIONS = {
    # P'TENSES - Most common in rap
    r"p'tit": "petit",
    r"p'tite": "petite",
    r"p'tites": "petites",
    r"p'tits": "petits",

    # C'ESTS - Fixed expressions
    r"c'est": "c'est",  # Keep as is (already expanded)

    # D'EPOS - Keep (already expanded)
    r"d'un": "d'un",  # Keep
    r"d'une": "d'une",  # Keep
    r"d'abord": "d'abord",  # Keep

    # L'APOSTROPHES - Keep (already expanded)
    r"l'ami": "l'ami",  # Keep
    r"l'amour": "l'amour",  # Keep
    r"l'heure": "l'heure",  # Keep
    r"l'école": "l'école",  # Keep
    r"l'homme": "l'homme",  # Keep

    # T'AURAIS/T'ES - Keep (already expanded)
    r"t'as": "t'as",  # Keep
    r"t'es": "t'es",  # Keep

    # S'EN - Keep (already expanded)
    r"s'en": "s'en",  # Keep

    # N'APOSTROPHES - Keep (already expanded)
    r"n'oublie": "n'oublie",  # Keep

    # QU'APOSTROPHES - Keep (already expanded)
    r"qu'on": "qu'on",  # Keep
    r"qu'il": "qu'il",  # Keep
}

# Censoring replacements (highest priority)
CENSORING_REPLACEMENTS = {
    r'p\*': 'pute',  # p* → pute (word boundary)
    r'p\*\*': 'pute',
    r'p\*ute': 'pute',
    r'p\*utes': 'pute',
    r'p\*\s': 'pute ',  # p* followed by space

    r'f\*': 'fuck',  # f* → fuck (word boundary)
    r'f\*\*': 'fuck',
    r'f\*ck': 'fuck',
    r'f\*\s': 'fuck ',  # f* followed by space
}


def analyze_apostrophe_patterns(df, lyrics_column='lyrics_cleaned', sample_size=None):
    """
    Analyze all apostrophe contractions in the dataset.
    Identifies patterns that need expansion.

    Args:
        df (pd.DataFrame): Dataset with lyrics
        lyrics_column (str): Column name with lyrics
        sample_size (int): Limit analysis to sample

    Returns:
        dict: Statistics about apostrophe contractions
    """

    if sample_size:
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        df_sample = df

    print("=" * 100)
    print("FRENCH APOSTROPHE CONTRACTION ANALYSIS")
    print("=" * 100)
    print(f"\nAnalyzing {len(df_sample)} songs for apostrophe contractions...\n")

    all_text = ' '.join(df_sample[lyrics_column].fillna(''))

    # Find all apostrophe patterns (word followed by apostrophe, then word)
    apostrophe_patterns = re.findall(r'[a-z]+['']?[a-z]+', all_text.lower())

    # Filter to only apostrophe patterns
    contraction_patterns = [p for p in apostrophe_patterns if '\'  in p or \''  in p]

    print(f"Total apostrophe contractions found: {len(contraction_patterns)}")
    print(f"\nMost common apostrophe patterns:")
    print("-" * 100)

    most_common = Counter(contraction_patterns).most_common(30)

    for pattern, count in most_common:
        # Suggest what it might expand to
        if pattern.lower().startswith("p'"):
            suggestion = pattern[0] + "etit" + pattern[2:]  # p'tit → petit
        elif pattern.lower().startswith("c'"):
            suggestion = "c'est"  # Usually c'est (already expanded)
        else:
            suggestion = "(analyze manually)"

        print(f"  {pattern:20} : {count:6,} times  → {suggestion}")

    print("\n" + "=" * 100)

    return {
        'contractions': Counter(contraction_patterns),
        'most_common': most_common
    }


def expand_apostrophes_french(text, custom_expansions=None):
    """
    Expand French apostrophe contractions to full words.
    e.g., p'tit → petit, d'un → de un (optional), etc.

    Args:
        text (str): Text with French apostrophe contractions
        custom_expansions (dict): Custom expansion rules

    Returns:
        str: Text with apostrophes expanded
    """

    if pd.isna(text) or text.strip() == '':
        return text

    result = text

    # Merge custom with defaults
    expansions = FRENCH_APOSTROPHE_EXPANSIONS.copy()
    if custom_expansions:
        expansions.update(custom_expansions)

    # Apply expansions (case-insensitive, preserve case)
    for pattern, replacement in expansions.items():
        # Case-insensitive replacement while preserving case when possible
        def replace_func(match):
            original = match.group(0)

            # If replacement has apostrophe, keep it as is (already expanded)
            if "'" in replacement or '\'' in replacement:
                return replacement

            # Otherwise preserve case for first letter
            if original[0].isupper():
                return replacement.capitalize()
            else:
                return replacement

        result = re.sub(pattern, replace_func, result, flags=re.IGNORECASE)

    return result


def replace_censoring_and_apostrophes(text):
    """
    Handle both censoring replacement and apostrophe expansion.

    Step 1: Replace p* → putain
    Step 2: Replace f* → fuck  
    Step 3: Expand apostrophes (p'tit → petit, etc.)

    Args:
        text (str): Raw text

    Returns:
        str: Cleaned text
    """

    if pd.isna(text) or text.strip() == '':
        return text

    result = text

    # Step 1: Replace censoring (highest priority to avoid conflicts)
    for pattern, replacement in CENSORING_REPLACEMENTS.items():
        def replace_func(match):
            original = match.group(0)
            # Preserve case if first letter is uppercase
            if original[0].isupper():
                return replacement.capitalize()
            else:
                return replacement

        result = re.sub(pattern, replace_func, result, flags=re.IGNORECASE)

    # Step 2: Standardize all apostrophe types to regular apostrophe
    apostrophe_variations = ["'", "'", "`", "´", "'"]
    for apos in apostrophe_variations[1:]:  # Skip regular apostrophe
        result = result.replace(apos, "'")

    # Step 3: Expand French apostrophe contractions
    result = expand_apostrophes_french(result)

    return result


def clean_censoring_apostrophes_v2(df, lyrics_column='lyrics'):
    """
    Complete cleaning: censoring replacement + apostrophe expansion.

    Process:
    1. Replace p* with putain
    2. Replace f* with fuck
    3. Standardize apostrophe types
    4. Expand French apostrophe contractions (p'tit → petit, etc.)

    Args:
        df (pd.DataFrame): Dataset with lyrics
        lyrics_column (str): Column with lyrics

    Returns:
        pd.DataFrame: Dataset with 'lyrics_cleaned_v3' column
    """

    print("\n" + "=" * 100)
    print("CENSORING & APOSTROPHE EXPANSION - VERSION 2")
    print("=" * 100)

    print("\nStep 1: Analyzing original text...")
    original_censoring = (df[lyrics_column].str.contains(r'\bp\*',).sum() +
                         df[lyrics_column].str.contains(r'\bf\*',).sum())
    original_apostrophes = df[lyrics_column].str.count(r"[''`´']").sum()

    print(f"   Found {original_censoring} censored words (p*, f*)")
    print(f"   Found {original_apostrophes} apostrophes to clean")

    print("\nStep 2: Applying cleaning...")
    df['lyrics_cleaned_v3'] = df[lyrics_column].apply(replace_censoring_and_apostrophes)

    print("   ✓ Censoring replaced (p* → putain, f* → fuck)")
    print("   ✓ Apostrophes standardized")
    print("   ✓ Apostrophe contractions expanded (p'tit → petit)")

    # Verify
    print("\nStep 3: Verification...")
    remaining_censoring = (df['lyrics_cleaned_v3'].str.contains(r'\bp\*', na=False, regex=True).sum() +
                          df['lyrics_cleaned_v3'].str.contains(r'\bf\*', na=False, regex=True).sum())

    print(f"   Censored words remaining: {remaining_censoring}")
    print(f"   Censored words replaced: {original_censoring - remaining_censoring}")

    # Character statistics
    print("\n" + "-" * 100)
    print("STATISTICS")
    print("-" * 100)

    orig_chars = df[lyrics_column].str.len().sum()
    clean_chars = df['lyrics_cleaned_v3'].str.len().sum()
    change_chars = clean_chars - orig_chars

    print(f"Original characters: {orig_chars:,}")
    print(f"Cleaned characters:  {clean_chars:,}")
    print(f"Character difference: {change_chars:+,} ({change_chars/orig_chars*100:+.2f}%)")

    songs_changed = (df[lyrics_column] != df['lyrics_cleaned_v3']).sum()
    print(f"\nSongs modified: {songs_changed:,} / {len(df):,} ({songs_changed/len(df)*100:.1f}%)")

    print("\n" + "=" * 100)

    return df


# ============================================================================
# INTERACTIVE FINDER & CUSTOM EXPANSION BUILDER
# ============================================================================

def find_apostrophe_context(df, pattern, context_size=50, lyrics_column='lyrics'):
    """
    Find specific apostrophe pattern with context.

    Args:
        df (pd.DataFrame): Dataset
        pattern (str): Apostrophe pattern to find (e.g., "p'tit")
        context_size (int): Characters before/after to show
        lyrics_column (str): Column with lyrics

    Returns:
        list: List of (song, context) tuples
    """

    results = []

    for idx, lyrics in enumerate(df[lyrics_column].items()):
        if pd.isna(lyrics):
            continue

        # Find all occurrences
        for match in re.finditer(pattern, lyrics, re.IGNORECASE):
            start = max(0, match.start() - context_size)
            end = min(len(lyrics), match.end() + context_size)
            context = lyrics[start:end]

            artist = df.iloc[idx].get('artist', 'Unknown')
            title = df.iloc[idx].get('title', 'Unknown')

            results.append({
                'artist': artist,
                'title': title,
                'context': context,
                'match': match.group(0)
            })

    return results


def suggest_expansions(df, sample_size=50, lyrics_column='lyrics'):
    """
    Analyze dataset and suggest custom expansion rules.

    Args:
        df (pd.DataFrame): Dataset
        sample_size (int): Number of top patterns to analyze
        lyrics_column (str): Column with lyrics

    Returns:
        dict: Suggested expansions
    """

    print("\n" + "=" * 100)
    print("APOSTROPHE EXPANSION SUGGESTIONS")
    print("=" * 100)

    stats = analyze_apostrophe_patterns(df, lyrics_column=lyrics_column, sample_size=None)

    most_common = stats['most_common'][:sample_size]

    suggestions = {}

    print("\nSuggested expansions based on frequency:\n")

    for pattern, count in most_common:
        lower = pattern.lower()

        if "'" not in lower and '\'' not in lower:
            continue

        # Try to suggest expansion
        suggestion = None

        if lower.startswith("p'"):
            # p'tit → petit, p'tite → petite
            suffix = lower[2:]
            if suffix == 'tit':
                suggestion = 'petit'
            elif suffix == 'tite':
                suggestion = 'petite'
            elif suffix == 'tits':
                suggestion = 'petits'
            elif suffix == 'tites':
                suggestion = 'petites'

        elif lower.startswith("c'"):
            suggestion = "c'est"  # Keep as is

        elif lower.startswith("l'"):
            suggestion = None  # Already expanded

        elif lower.startswith("d'"):
            suggestion = None  # Already expanded

        if suggestion:
            suggestions[pattern.lower()] = suggestion
            print(f"  {pattern:20} ({count:6,} times) → {suggestion}")

    print("\n" + "=" * 100)

    return suggestions



import re

def decode_censored_words(text):
    """
    Reconstructs censored words from LRFAF French rap lyrics.
    
    Rules extracted and validated from actual corpus data:
    
    Validated mappings:
    - b*** → bite
    - b*tes → bites
    - cette ***** → cette pute
    - la ch***e → la chatte
    - fils de p****** → fils de pute
    - cassez pas les ******* → cassez pas les couilles
    - sh*t → shit
    - cette p******* → cette pute
    - n*gros → negros
    - mangeurs de ***** → mangeurs de bites
    - fils de p*te → fils de putes
    """
    
    if pd.isna(text):
        return text
    
    text = str(text)
    
    # Rule 1: Remove repeat notation like (*4), (*8), (*2), (*3), etc.
    text = re.sub(r'\(\*\d+\)', '', text)
    
    # Rule 2: sh*t → shit (English word)
    text = re.sub(r'\bsh\*t\b', 'shit', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsh\*+\b', 'shit', text, flags=re.IGNORECASE)
    
    # Rule 3: b*** → bite, b*tes → bites
    text = re.sub(r'\bb\*tes\b', 'bites', text, flags=re.IGNORECASE)
    text = re.sub(r'\bb\*te\b', 'bite', text, flags=re.IGNORECASE)
    text = re.sub(r'\bb\*+s\b', 'bites', text, flags=re.IGNORECASE)
    text = re.sub(r'\bb\*+\b', 'bite', text, flags=re.IGNORECASE)
    
    # Rule 4: ch***e → chatte
    text = re.sub(r'\bla\s+ch\*+e\b', 'la chatte', text, flags=re.IGNORECASE)
    text = re.sub(r'\bch\*+e\b', 'chatte', text, flags=re.IGNORECASE)
    text = re.sub(r'\bch\*+\b', 'chatte', text, flags=re.IGNORECASE)
    
    # Rule 5: p****** / p******* / p*te(s) → pute/putes
    # fils de p*te → fils de putes
    text = re.sub(r'\bfils\s+de\s+p\*te\b', 'fils de putes', text, flags=re.IGNORECASE)
    text = re.sub(r'\bp\*tes\b', 'putes', text, flags=re.IGNORECASE)
    text = re.sub(r'\bp\*te\b', 'pute', text, flags=re.IGNORECASE)
    # General p*** with 5+ asterisks → pute
    text = re.sub(r'\bp\*{5,}\b', 'pute', text, flags=re.IGNORECASE)
    text = re.sub(r'\bp\*{3,4}\b', 'putain', text, flags=re.IGNORECASE)
    text = re.sub(r'\bp\*{1,2}\b', 'pute', text, flags=re.IGNORECASE)
    
    # Rule 6: cette ***** / cette p******* → cette pute
    text = re.sub(r'\bcette\s+\*{5,}\b', 'cette pute', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcette\s+p\*{5,}\b', 'cette pute', text, flags=re.IGNORECASE)
    
    # Rule 7: cassez pas les ******* → cassez pas les couilles (7+ asterisks)
    text = re.sub(r'\bcassez\s+pas\s+les\s+\*{7,}\b', 'cassez pas les couilles', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpas\s+les\s+\*{7,}\b', 'pas les couilles', text, flags=re.IGNORECASE)
    text = re.sub(r'\bles\s+\*{7,}\b', 'les couilles', text, flags=re.IGNORECASE)
    
    # Rule 8: mangeurs de ***** → mangeurs de bites (5 asterisks after "de")
    text = re.sub(r'\bmangeurs\s+de\s+\*{5}\b', 'mangeurs de bites', text, flags=re.IGNORECASE)
    text = re.sub(r'\bde\s+\*{5}\b', 'de bites', text, flags=re.IGNORECASE)
    
    # Rule 9: n*gros → negros, n* variants → negro/nigga
    text = re.sub(r'\bn\*gros\b', 'negros', text, flags=re.IGNORECASE)
    text = re.sub(r'\bn\*gro\b', 'negro', text, flags=re.IGNORECASE)
    text = re.sub(r'\bn\*gre\b', 'negre', text, flags=re.IGNORECASE)
    # Long sequences: n**** (4+) → nigga, n******* (7+) → niggas
    text = re.sub(r'\bn\*{10,}\b', 'niggas', text, flags=re.IGNORECASE)
    text = re.sub(r'\bn\*{4,9}\b', 'nigga', text, flags=re.IGNORECASE)
    text = re.sub(r'\bn\*{1,3}\b', 'negro', text, flags=re.IGNORECASE)

    # Rule 13: Cleanup - remove standalone asterisks that are surrounded by letters
    text = re.sub(r'(?<=[a-z])\*(?=[a-z])', '', text, flags=re.IGNORECASE)

    text = str(text)
    # Remove all asterisks (single or multiple)
    text = re.sub(r'\*+', '', text)
    # Clean up extra spaces created by asterisk removal
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def apply_to_dataframe(df, column_name='lyrics_cleaned', output_column='lyrics_cleaned'):
    """
    Apply decensoring to a DataFrame column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Your dataframe
    column_name : str
        Name of column with censored text (default: 'lyrics_cleaned')
    output_column : str
        Name of new column for decensored text (default: 'lyrics_decensored')
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with new decensored column
    
    Example:
    --------
    df = apply_to_dataframe(df, column_name='lyrics_cleaned', output_column='lyrics_decensored')
    print(df[['lyrics_cleaned', 'lyrics_decensored']].head())
    """
    df[output_column] = df[column_name].apply(decode_censored_words)
    return df


#df = clean_censoring_apostrophes_v2(df)