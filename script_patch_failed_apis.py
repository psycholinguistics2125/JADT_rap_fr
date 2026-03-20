#!/usr/bin/env python3
"""
Patch Failed API Calls for Artist Biographies
==============================================
Re-runs only the API calls that failed during the main collection:
  - MusicBrainz (connection reset errors)
  - Claude web search (credit balance ran out)

Reads existing artist_biographies.json, patches missing fields in-place,
and saves back. Does NOT re-run Wikidata, Wikipedia, Genius, Discogs, or OpenAI.

Usage:
  python script_patch_failed_apis.py                  # patch both MB + Claude
  python script_patch_failed_apis.py --only-mb        # patch MusicBrainz only
  python script_patch_failed_apis.py --only-claude    # patch Claude only
  python script_patch_failed_apis.py --limit 10       # patch first 10 artists
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Re-use functions from main script
from script_collect_rapper_biography import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    MUSICBRAINZ_DELAY,
    OUTPUT_CSV,
    OUTPUT_JSON,
    format_musicbrainz_as_text,
    save_progress,
    search_musicbrainz_artist,
    synthesize_biography_with_claude,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Patch failed MusicBrainz and/or Claude API calls"
    )
    parser.add_argument("--only-mb", action="store_true",
                        help="Only retry MusicBrainz calls")
    parser.add_argument("--only-claude", action="store_true",
                        help="Only retry Claude calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of artists to patch")
    args = parser.parse_args()

    do_mb = not args.only_claude
    do_claude = not args.only_mb

    if not do_mb and not do_claude:
        print("Nothing to do (both --only-mb and --only-claude cancel out).")
        return

    # Backup before patching
    backup_path = OUTPUT_JSON.with_suffix(".json.bak")
    shutil.copy2(OUTPUT_JSON, backup_path)
    print(f"Backup saved to: {backup_path}")

    # Load existing data
    print("Loading existing artist_biographies.json...")
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} artists.")

    # Identify artists that need patching
    needs_mb = []
    needs_claude = []
    for i, r in enumerate(results):
        if do_mb and not r.get("musicbrainz_id"):
            needs_mb.append(i)
        if do_claude and not r.get("biography_summary"):
            needs_claude.append(i)

    # Deduplicate indices
    all_indices = sorted(set(needs_mb) | set(needs_claude))
    needs_mb_set = set(needs_mb)
    needs_claude_set = set(needs_claude)

    print(f"\nArtists needing MusicBrainz retry: {len(needs_mb)}")
    print(f"Artists needing Claude retry:      {len(needs_claude)}")
    print(f"Total artists to patch:            {len(all_indices)}")

    if not all_indices:
        print("Nothing to patch!")
        return

    if args.limit:
        all_indices = all_indices[:args.limit]
        print(f"(Limited to first {args.limit})")

    # Init clients
    session = requests.Session()
    anthropic_client = None
    if do_claude and ANTHROPIC_API_KEY:
        anthropic_client = anthropic.Anthropic()
        print(f"\nClaude ({ANTHROPIC_MODEL}): ENABLED")
    elif do_claude:
        print("\nClaude: DISABLED (no ANTHROPIC_API_KEY)")
        do_claude = False

    count_mb_patched = 0
    count_claude_patched = 0
    count_since_save = 0

    try:
        for idx in tqdm(all_indices, desc="Patching"):
            r = results[idx]
            artist_name = r["artist"]

            # --- MusicBrainz retry ---
            if idx in needs_mb_set and do_mb:
                mb_data = search_musicbrainz_artist(artist_name, session)
                if mb_data and mb_data.get("mbid"):
                    r["musicbrainz_id"] = mb_data.get("mbid")
                    r["mb_type"] = mb_data.get("mb_type")
                    r["mb_name"] = mb_data.get("mb_name")
                    r["mb_country"] = mb_data.get("mb_country")
                    r["mb_area"] = mb_data.get("mb_area")
                    r["mb_begin_area"] = mb_data.get("mb_begin_area")
                    r["mb_begin_date"] = mb_data.get("mb_begin_date")
                    r["mb_tags"] = mb_data.get("mb_tags", [])
                    r["mb_labels"] = mb_data.get("mb_labels", [])
                    r["mb_related_artists"] = mb_data.get("mb_related_artists", [])
                    r["mb_urls"] = mb_data.get("mb_urls", {})

                    # Update is_a_band if MB says Group
                    if mb_data.get("mb_type") == "Group":
                        r["is_a_band"] = True

                    # Update source string
                    if "musicbrainz" not in r.get("source", ""):
                        r["source"] = r.get("source", "").replace("+llm", "+musicbrainz+llm")

                    count_mb_patched += 1

            # --- Claude retry ---
            if idx in needs_claude_set and do_claude and anthropic_client:
                is_a_band = r.get("is_a_band", False)
                claude_data = synthesize_biography_with_claude(
                    artist_name, is_a_band, anthropic_client
                )
                if claude_data and claude_data.get("biography_summary"):
                    r["biography_summary"] = claude_data["biography_summary"]
                    r["biography_sources"] = claude_data.get("biography_sources", [])

                    # Update source string
                    if "claude" not in r.get("source", ""):
                        r["source"] = r.get("source", "") + "+claude"

                    count_claude_patched += 1

            count_since_save += 1
            if count_since_save >= 10:
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                count_since_save = 0

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving progress...")

    # Final save
    save_progress(results, [])

    print(f"\n{'=' * 60}")
    print("PATCH RESULTS")
    print("=" * 60)
    print(f"MusicBrainz patched:  {count_mb_patched}/{len(needs_mb)}")
    print(f"Claude patched:       {count_claude_patched}/{len(needs_claude)}")
    print(f"\nSaved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
