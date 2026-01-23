#!/usr/bin/env python3
"""
French Rap Dataset - API Year Finder (Rate-Limit Safe)
Handles iTunes and MusicBrainz APIs with proper delays to avoid 429 errors
"""

import pandas as pd
import requests
import time
from datetime import datetime
from tqdm import tqdm
import datasets
from datasets import load_dataset

# =============================================================================
# API FUNCTIONS WITH RATE LIMITING
# =============================================================================

def get_year_itunes(artist, title, delay=0.5):
    """
    iTunes API with rate limiting.
    Delay: 0.3s = ~200 songs/hour (safe for iTunes limits)
    """
    try:
        time.sleep(delay)  # CRITICAL: Prevents 429 Rate Limit errors
        
        clean_title = title.replace('*', '').strip()
        
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                'term': f"{artist} {clean_title}",
                'entity': 'song',
                'limit': 1,
                'country': 'FR'
            },
            timeout=10
        )
        
        # Handle rate limiting
        if response.status_code == 429:
            print("  ⚠️  Rate limited by iTunes, waiting 10s...")
            time.sleep(10)
            return None, 'rate_limited'
        
        data = response.json()
        
        if data.get('resultCount', 0) > 0:
            release_date = data['results'][0].get('releaseDate')
            if release_date:
                year = datetime.fromisoformat(release_date.rstrip('Z')).year
                return year, 'itunes'
        
        return None, 'not_found'
        
    except Exception as e:
        return None, 'error'


def get_year_musicbrainz(artist, title, delay=1.1):
    """
    MusicBrainz API with rate limiting.
    Delay: 1.1s = ~80 songs/hour (respects 1 request/second limit)
    """
    try:
        time.sleep(delay)  # REQUIRED: MusicBrainz enforces 1 req/sec
        
        clean_title = title.replace('*', '').strip()
        
        response = requests.get(
            "https://musicbrainz.org/ws/2/recording/",
            params={
                'query': f'artist:"{artist}" AND recording:"{clean_title}"',
                'fmt': 'json',
                'limit': 1
            },
            headers={
                'User-Agent': 'FrenchRapResearch/1.0 (educational)'
            },
            timeout=10
        )
        
        data = response.json()
        
        if 'recordings' in data and data['recordings']:
            date_str = data['recordings'][0].get('first-release-date', '')
            if date_str and len(date_str) >= 4:
                year = int(date_str[:4])
                return year, 'musicbrainz'
        
        return None, 'not_found'
        
    except Exception as e:
        return None, 'error'


# =============================================================================
# MAIN PROCESSING SCRIPT
# =============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("FRENCH RAP DATASET - API YEAR FINDER")
    print("="*70)
    
    # Load dataset
    

    df  = pd.read_csv("data/data_without_year_3.csv")



    
    # Add columns if not exist
    if 'year' not in df.columns:
        df['year'] = None
    if 'source' not in df.columns:
        df['source'] = None
    
    print(f"\nTotal songs: {len(df):,}")
    print(f"Songs without year: {df['year'].isna().sum():,}")
    
    # Calculate time estimate
    missing = df['year'].isna().sum()
    hours_estimate = (missing * 1.4) / 3600  # Average 1.4s per song
    print(f"Estimated time: {hours_estimate:.1f} hours")
    print("\nStarting API requests...")
    print("(Press Ctrl+C to stop and save progress)\n")
    
    # Process only songs without years
    to_process = df[df['year'].isna()]
    found_count = 0
    not_found_count = 0
    
    try:
        for idx, row in tqdm(to_process.iterrows(), 
                            total=len(to_process), 
                            desc="Processing"):
            
            # Try iTunes first (faster API)
            year, source = get_year_itunes(row['artist'], row['title'])
            
            # If not found in iTunes, try MusicBrainz (slower but more data)
            if not year:
                year, source = get_year_musicbrainz(row['artist'], row['title'])
            
            # Save result if found
            if year:
                df.loc[idx, 'year'] = year
                df.loc[idx, 'source'] = source
                found_count += 1
            else:
                not_found_count += 1
            
            # Save progress every 50 songs
            if (found_count + not_found_count) % 50 == 0:
                df.to_csv('data_with_years_PROGRESS.csv', index=False)
                print(f"\n  💾 Progress saved: {found_count} found, {not_found_count} not found")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving progress...")
    
    # Final save
    df.to_csv('data_with_years_COMPLETED.csv', index=False)
    
    # Print statistics
    completed = df['year'].notna().sum()
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    print(f"Total songs: {len(df):,}")
    print(f"Years found: {completed:,} ({100*completed/len(df):.1f}%)")
    print(f"Still missing: {len(df) - completed:,}")
    print(f"\nOutput file: data_with_years_COMPLETED.csv")
    
    # Source breakdown
    if 'source' in df.columns:
        print(f"\nSource breakdown:")
        print(df['source'].value_counts())
