import re
import pandas as pd
from typing import List, Dict, Any

from utils.clean_lyrics import clean_censoring_apostrophes_v2
from utils.clean_songs import remove_non_song, clean_lyrics_column, estimate_year_from_artist_patterns


def split_verses(lyrics: str, min_words: int = 50, max_words: int = 500) -> List[Dict[str, Any]]:
    """
    Split lyrics into verses, respecting word count constraints by design.
    - Merges verses smaller than min_words with adjacent ones
    - Splits verses larger than max_words
    Returns: List of dicts with guaranteed word counts in [min_words, max_words]
    """
    verses = []
    
    # Strategy 1: Explicit [Verse ...] or [Chorus ...] markers
    marked_verses = extract_marked_verses(lyrics)
    if marked_verses:
        verses = marked_verses
    
    # Strategy 2: Split by double newlines
    if not verses:
        verses = split_by_newlines(lyrics)
    
    # Strategy 3: Heuristic grouping (fallback)
    if not verses:
        verses = split_by_heuristic(lyrics)
    
    # Now enforce word count constraints WITHOUT losing data
    constrained_verses = enforce_word_count_constraints(
        verses, min_words=min_words, max_words=max_words
    )
    
    return constrained_verses

def extract_marked_verses(lyrics: str) -> List[Dict[str, Any]]:
    """Extract verses with explicit markers like [Verse 1]"""
    verses = []
    
    for match in re.finditer(r'\[([^\]]+)\]', lyrics):
        label = match.group(1)
        start = match.end()
        next_match = re.search(r'\[([^\]]+)\]', lyrics[start:])
        end = start + next_match.start() if next_match else len(lyrics)
        
        verse_text = lyrics[start:end].strip()
        if verse_text:
            verses.append({
                "verse_text": verse_text,
                "verse_type": classify_verse_type(label),
                "verse_number": extract_verse_number(label),
                "line_count": len(verse_text.split('\n')),
                "word_count": len(verse_text.split())
            })
    
    return verses

def split_by_newlines(lyrics: str) -> List[Dict[str, Any]]:
    """Split by double newlines (structural breaks)"""
    verses = []
    sections = re.split(r'\n\s*\n+', lyrics)
    
    for idx, sec in enumerate(sections):
        sec = sec.strip()
        if sec:  # Keep even short sections for now
            verses.append({
                "verse_text": sec,
                "verse_type": infer_verse_type_from_content(sec, idx, len(sections)),
                "verse_number": idx + 1,
                "line_count": len(sec.split('\n')),
                "word_count": len(sec.split())
            })
    
    return verses

def split_by_heuristic(lyrics: str) -> List[Dict[str, Any]]:
    """Fallback: Split by heuristic (e.g., 12-line groups)"""
    verses = []
    lines = [l for l in lyrics.split('\n') if l.strip()]
    target_lines = 12
    
    for idx in range(0, len(lines), target_lines):
        group = lines[idx:idx+target_lines]
        if group:
            verse_text = '\n'.join(group)
            verses.append({
                "verse_text": verse_text,
                "verse_type": "verse",
                "verse_number": idx // target_lines + 1,
                "line_count": len(group),
                "word_count": len(verse_text.split())
            })
    
    return verses

def enforce_word_count_constraints(verses: List[Dict], 
                                   min_words: int = 50, 
                                   max_words: int = 500) -> List[Dict[str, Any]]:
    """
    Enforce word count constraints by merging and splitting verses.
    NO DATA LOSS.
    """
    constrained = []
    i = 0
    
    while i < len(verses):
        current_verse = verses[i]
        current_text = current_verse["verse_text"]
        current_wc = current_verse["word_count"]
        
        # Case 1: Verse is too small (< min_words) — merge with next
        if current_wc < min_words:
            merged_text = current_text
            j = i + 1
            
            # Keep merging until we reach min_words or run out of verses
            while j < len(verses) and len(merged_text.split()) < min_words:
                merged_text += "\n\n" + verses[j]["verse_text"]
                j += 1
            
            # If still too small, try merging backwards (previous verse was also small)
            if len(merged_text.split()) < min_words and constrained:
                # Merge with last constrained verse
                last = constrained.pop()
                merged_text = last["verse_text"] + "\n\n" + merged_text
            
            constrained.append({
                "verse_text": merged_text,
                "verse_type": current_verse["verse_type"],
                "verse_number": len(constrained) + 1,
                "line_count": len(merged_text.split('\n')),
                "word_count": len(merged_text.split())
            })
            
            i = j
        
        # Case 2: Verse is too large (> max_words) — split intelligently
        elif current_wc > max_words:
            split_verses_list = split_large_verse(current_text, max_words)
            for idx, split_text in enumerate(split_verses_list):
                constrained.append({
                    "verse_text": split_text,
                    "verse_type": current_verse["verse_type"],
                    "verse_number": len(constrained) + 1,
                    "line_count": len(split_text.split('\n')),
                    "word_count": len(split_text.split())
                })
            i += 1
        
        # Case 3: Verse is just right — keep as is
        else:
            current_verse["verse_number"] = len(constrained) + 1
            constrained.append(current_verse)
            i += 1
    
    return constrained

def split_large_verse(verse_text: str, max_words: int) -> List[str]:
    """
    Split a large verse into smaller chunks, respecting line boundaries.
    Each chunk tries to stay close to max_words without breaking in the middle of a line.
    """
    lines = verse_text.split('\n')
    chunks = []
    current_chunk = []
    current_wc = 0
    
    for line in lines:
        line_wc = len(line.split())
        
        # If adding this line exceeds max_words, save current chunk and start new one
        if current_wc + line_wc > max_words and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_wc = line_wc
        else:
            current_chunk.append(line)
            current_wc += line_wc
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def classify_verse_type(label: str) -> str:
    label = label.lower()
    if 'verse' in label: return 'verse'
    if 'chorus' in label or 'refrain' in label: return 'chorus'
    if 'intro' in label: return 'intro'
    if 'outro' in label: return 'outro'
    return 'unknown'

def extract_verse_number(label: str) -> int:
    match = re.search(r'\d+', label)
    return int(match.group()) if match else 0

def infer_verse_type_from_content(text: str, idx: int, total: int) -> str:
    if idx == 0: return 'intro'
    if idx == total - 1: return 'outro'
    return 'verse'

def build_df_verses(df_songs: pd.DataFrame, 
                    min_words: int = 50, 
                    max_words: int = 500) -> pd.DataFrame:
    """
    Build df_verses with all metadata preserved, no data loss.
    Each verse respects word count constraints by design.
    """
    data = []
    
    for i, row in df_songs.iterrows():
        song_meta = {k: row[k] for k in ['artist', 'title', 'year', 'born_in_france', 'url','birthdate_artist','age_artist']}
        
        # Split verses with automatic merging/splitting
        verses = split_verses(row['lyrics'], min_words=min_words, max_words=max_words)
        
        for v in verses:
            
            data.append({
                **song_meta,
                "song_id": i,
                "lyrics": v['verse_text'],
                "verse_type": v['verse_type'],
                "verse_number": v['verse_number'],
                "line_count": v['line_count'],
                "word_count": v['word_count']
            })
    
    df_verses = pd.DataFrame(data)
    return df_verses


if __name__ == "__main__" : 
    df_songs = pd.read_csv("data/corpus_with_date_itunes.csv")
    # remove song that are not song
    df_songs = remove_non_song(df_songs)
    # Usage
    df_verses = build_df_verses(df_songs, min_words=50, max_words=500)
    print(f"✓ Created verse dataset: {len(df_verses)} verses from {len(df_songs)} songs")
    print(f"  Expansion ratio: {len(df_verses) / len(df_songs):.2f}x (avg verses per song)")
    print(f"  Verse columns: {list(df_verses.columns)}\n")

    df_verses = df_verses[df_verses['word_count']>50]
    
    print(f" removing verses under 50 words")
    
    # Describe verse statistics
    print("📊 VERSE STATISTICS:")
    print(f"  Word count range: {df_verses['word_count'].min():.0f} - {df_verses['word_count'].max():.0f}")
    print(f"  Mean words per verse: {df_verses['word_count'].mean():.1f}")
    print(f"  Median words per verse: {df_verses['word_count'].median():.1f}")
    print(f"  Verse type distribution:")
    for vtype, count in df_verses['verse_type'].value_counts().items():
        pct = count / len(df_verses) * 100
        print(f"    - {vtype}: {count} ({pct:.1f}%)")
    print()
    
    # ============================================================================
    # CLEAN LYRICS - BASIC CLEANING
    # ============================================================================
    print("=" * 80)
    print("STEP 3: CLEANING LYRICS - BASIC PASS")
    print("=" * 80)
    
    df_verses = clean_lyrics_column(df_verses)
    print(f"✓ Applied basic cleaning to all {len(df_verses)} verses")
    print(f"  New column: 'lyrics_cleaned'\n")
    
    # ============================================================================
    # CLEAN CENSORING & EXPAND APOSTROPHES
    # ============================================================================
    print("=" * 80)
    print("STEP 4: CLEANING CENSORING & EXPANDING APOSTROPHES")
    print("=" * 80)
    
    df_verses = clean_censoring_apostrophes_v2(df_verses)
    print(f"✓ Applied censoring removal and apostrophe expansion")
    print(f"  Verses processed: {len(df_verses)}\n")
    
    # ============================================================================
    # HANDLE MISSING YEARS - ESTIMATE FROM ARTIST PATTERNS
    # ============================================================================
    print("=" * 80)
    print("STEP 5: HANDLING MISSING YEARS")
    print("=" * 80)
    
    # Count missing years before estimation
    missing_before = len(df_verses[df_verses['year'].isna()])
    print(f"⚠️  Missing years BEFORE estimation: {missing_before} verses ({missing_before/len(df_verses)*100:.2f}%)\n")
    
    # Apply year estimation using artist patterns
    missing_mask = df_verses['year'].isna()
    df_verses.loc[missing_mask, 'year'] = df_verses.loc[missing_mask, 'artist'].apply(
        lambda artist: estimate_year_from_artist_patterns(artist, df_verses)
    )
    
    # Count missing years after estimation
    missing_after = len(df_verses[df_verses['year'].isna()])
    estimated_count = missing_before - missing_after
    print(f"✓ Estimated years using artist patterns: {estimated_count} verses filled")
    print(f"⚠️  Missing years AFTER estimation: {missing_after} verses ({missing_after/len(df_verses)*100:.2f}%)")
    
    if missing_after > 0:
        print(f"  → Removing {missing_after} verses with no year data\n")
    


    # ============================================================================
    # FINAL DATASET PREPARATION
    # ============================================================================
    print("=" * 80)
    print("STEP 6: FINAL DATASET PREPARATION")
    print("=" * 80)
    
    # Remove any remaining verses with missing years
    final_data = df_verses[~df_verses['year'].isna()]
    print(f"✓ Final dataset size: {len(final_data)} verses")
    print(f"  Data retention: {len(final_data)/len(df_verses)*100:.2f}% of verse dataset\n")

    final_data = final_data[final_data['word_count']>50]
    print(f"✓ Final dataset size: {len(final_data)} verses")
    print(f" removing verses under 50 words")
    
    # Describe final dataset
    print("📊 FINAL DATASET STATISTICS:")
    print(f"  Unique artists: {final_data['artist'].nunique()}")
    print(f"  Unique songs: {final_data['title'].nunique()}")
    print(f"  Year range: {final_data['year'].min():.0f} - {final_data['year'].max():.0f}")
    print(f"  Birth year range: {final_data['birthdate_artist'].min():.0f} - {final_data['birthdate_artist'].max():.0f}")
    print(f"  Artists born in France: {final_data['born_in_france'].sum()} ({final_data['born_in_france'].sum()/final_data['born_in_france'].notna().sum()*100:.1f}%)")
    print(f"  Mean age at release: {final_data['age_artist'].mean():.1f} years\n")
    
    # Select columns for output
    col_select = [
        'artist', 
        'title', 
        'year', 
        'lyrics', 
        'lyrics_cleaned',
        'born_in_france', 
        'url',
        'birthdate_artist',
        'age_artist'
    ]
    
    # Verify all selected columns exist
    missing_cols = [col for col in col_select if col not in final_data.columns]
    if missing_cols:
        print(f"⚠️  WARNING: Missing columns in dataset: {missing_cols}")
        col_select = [col for col in col_select if col in final_data.columns]
    
    # ============================================================================
    # SAVE FINAL DATASET
    # ============================================================================
    print("=" * 80)
    print("STEP 7: SAVING FINAL DATASET")
    print("=" * 80)
    
    output_file = 'data/20251126_cleaned_verses_lrfaf_corpus.csv'
    final_data[col_select].to_csv(output_file, index=False)
    
    print(f"✓ Saved cleaned verses dataset to: {output_file}")
    print(f"  Rows: {len(final_data)}")
    print(f"  Columns: {len(col_select)}")
    print(f"  File size: {final_data[col_select].memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")
    
    print("=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)
