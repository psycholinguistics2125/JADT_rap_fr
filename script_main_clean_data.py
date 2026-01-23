
"""
mainfile to clean the datasets

"""

import pandas as pd
import re

from utils.clean_lyrics import clean_censoring_apostrophes_v2
from utils.clean_songs import remove_non_song, clean_lyrics_column, estimate_year_from_artist_patterns




if __name__ == "__main__" : 
    df = pd.read_csv("data/corpus_with_date_itunes.csv")
    # remove song that are not song
    df = remove_non_song(df)

    # add a columns clean_lyrics
    df = clean_lyrics_column(df)

    # 3. Clean censoring & expand apostrophes
    df = clean_censoring_apostrophes_v2(df)

    #4. add years for songs that still do not have year using the median (n= 668)
    # Apply ONLY to missing years
    print(f"there are still {len(df[df['year'].isna()])} without year, we use the median for those")
    
    # Use with lambda
    missing_mask = df['year'].isna()
    df.loc[missing_mask, 'year'] = df.loc[missing_mask, 'artist'].apply(lambda artist: estimate_year_from_artist_patterns(artist, df))
    
    print(f"there are still {len(df[df['year'].isna()])} without year, event after median estimation, we remove them")

    final_data = df[~df['year'].isna()]
    
    col_select = ['artist', 'title', 'year', 'lyrics', "lyrics_cleaned","born_in_france", 'url','birthdate_artist','age_artist', 'born_in_france']
    
    final_data[col_select].to_csv('data/20251125_cleaned_lrfaf_corpus.csv', index = 0)
   

   

