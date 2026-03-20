#!/usr/bin/env python3
"""
Collect Structured Biographies for French Rappers
==================================================
Pipeline per artist (7 sources):
  1. Wikidata — resolve entity + SPARQL structured fields
  2. Wikipedia FR — full article text
  3. MusicBrainz — music metadata (labels, tags, area, relations)
  4. Genius — artist bio, social links, alternate names
  5. Discogs — profile/bio, real name, labels, groups, URLs
  6. GPT-4o-mini — extract structured biography from ALL sources
  7. Claude (Anthropic) — web search + biography synthesis with sources
  8. Merge & save

Outputs:
  - data/artist_biographies.json   (full data + resume file)
  - data/artist_biographies.csv    (flat tabular summary)
  - data/artist_biographies_skipped.csv (artists not found)

Usage:
  python script_collect_rapper_biography.py
  python script_collect_rapper_biography.py --limit 5
  python script_collect_rapper_biography.py --no-resume
  python script_collect_rapper_biography.py --no-claude

Environment variables (in .env file):
  OPENAI_API_KEY          - OpenAI API key (required)
  GENIUS_ACCESS_TOKEN     - Genius API access token (optional)
  DISCOGS_ACCESS_TOKEN    - Discogs personal access token (optional)
  ANTHROPIC_API_KEY       - Anthropic API key (optional, for web search)
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import anthropic
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "20260123_filter_verses_lrfaf_corpus.csv"
OUTPUT_JSON = DATA_DIR / "artist_biographies.json"
OUTPUT_CSV = DATA_DIR / "artist_biographies.csv"
OUTPUT_SKIPPED = DATA_DIR / "artist_biographies_skipped.csv"

# Rate-limiting delays (seconds)
WIKIMEDIA_DELAY = 1.0
MUSICBRAINZ_DELAY = 1.1  # MusicBrainz requires max 1 req/sec
GENIUS_DELAY = 0.5
DISCOGS_DELAY = 1.0      # Discogs: 60 req/min authenticated
OPENAI_DELAY = 0.5
ANTHROPIC_DELAY = 1.0

# Text truncation limits
MAX_WIKI_WORDS = 6000
MAX_GENIUS_BIO_WORDS = 2000
MAX_DISCOGS_PROFILE_WORDS = 2000

WIKIMEDIA_HEADERS = {
    "User-Agent": "FrenchRapBiographyResearch/1.0 (educational; Python/requests)"
}

# MusicBrainz config (no API key needed)
MUSICBRAINZ_BASE = "https://musicbrainz.org/ws/2"
MUSICBRAINZ_HEADERS = {
    "User-Agent": "FrenchRapBiographyResearch/1.0 (educational; Python/requests)",
    "Accept": "application/json",
}

# Genius API config
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN", "")
GENIUS_BASE = "https://api.genius.com"

# Anthropic API config (for web search + biography synthesis)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

# Discogs API config
DISCOGS_TOKEN = os.getenv("DISCOGS_ACCESS_TOKEN", "")
DISCOGS_BASE = "https://api.discogs.com"
DISCOGS_HEADERS = {
    "User-Agent": "FrenchRapBiographyResearch/1.0",
    "Accept": "application/json",
}

# Wikidata occupation QIDs for disambiguation
OCCUPATION_QIDS = {
    "Q639669",   # musician
    "Q177220",   # singer
    "Q36834",    # composer
    "Q753110",   # songwriter
    "Q2252262",  # rapper
    "Q488205",   # singer-songwriter
    "Q806349",   # bandleader
    "Q183945",   # record producer
}

# SPARQL endpoint
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


# =============================================================================
# 1. RESOLVE WIKIDATA ENTITY
# =============================================================================

def resolve_wikidata_entity(artist_name, session):
    """
    Search Wikidata for the artist and disambiguate by occupation (P106).
    Returns (wikidata_id, wikipedia_fr_title) or (None, None).
    """
    time.sleep(WIKIMEDIA_DELAY)
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": artist_name,
        "language": "fr",
        "format": "json",
        "limit": 10,
        "type": "item",
    }
    try:
        resp = session.get(url, params=params, headers=WIKIMEDIA_HEADERS, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("search", [])
    except Exception as e:
        print(f"\n  [Wikidata] Search error for '{artist_name}': {e}")
        return None, None

    for candidate in results:
        qid = candidate.get("id")
        if not qid:
            continue
        time.sleep(WIKIMEDIA_DELAY)
        try:
            entity_resp = session.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "claims|sitelinks",
                    "format": "json",
                },
                headers=WIKIMEDIA_HEADERS,
                timeout=15,
            )
            entity_resp.raise_for_status()
            entity_data = entity_resp.json().get("entities", {}).get(qid, {})
        except Exception:
            continue

        claims = entity_data.get("claims", {})
        occupations = claims.get("P106", [])
        occupation_ids = set()
        for claim in occupations:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict):
                occupation_ids.add(value.get("id", ""))

        if not occupation_ids & OCCUPATION_QIDS:
            continue

        sitelinks = entity_data.get("sitelinks", {})
        fr_wiki = sitelinks.get("frwiki", {})
        wiki_title = fr_wiki.get("title")
        return qid, wiki_title

    return None, None


# =============================================================================
# 2. FETCH WIKIDATA STRUCTURED FIELDS (SPARQL)
# =============================================================================

def fetch_wikidata_structured(wikidata_id, session):
    """Fetch structured biographical fields via SPARQL."""
    time.sleep(WIKIMEDIA_DELAY)
    query = f"""
    SELECT ?birthDate ?birthPlaceLabel
           (GROUP_CONCAT(DISTINCT ?labelLabel; SEPARATOR=" | ") AS ?recordLabels)
           (GROUP_CONCAT(DISTINCT ?genreLabel; SEPARATOR=" | ") AS ?genres)
           (GROUP_CONCAT(DISTINCT ?eduLabel; SEPARATOR=" | ") AS ?education)
           (GROUP_CONCAT(DISTINCT ?assocLabel; SEPARATOR=" | ") AS ?associatedActs)
    WHERE {{
      OPTIONAL {{ wd:{wikidata_id} wdt:P569 ?birthDate. }}
      OPTIONAL {{ wd:{wikidata_id} wdt:P19 ?birthPlace. }}
      OPTIONAL {{ wd:{wikidata_id} wdt:P264 ?label. }}
      OPTIONAL {{ wd:{wikidata_id} wdt:P136 ?genre. }}
      OPTIONAL {{ wd:{wikidata_id} wdt:P69 ?edu. }}
      OPTIONAL {{
        {{ wd:{wikidata_id} wdt:P527 ?assoc. }}
        UNION
        {{ wd:{wikidata_id} wdt:P361 ?assoc. }}
        UNION
        {{ wd:{wikidata_id} wdt:P463 ?assoc. }}
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "fr,en". }}
    }}
    GROUP BY ?birthDate ?birthPlaceLabel
    LIMIT 1
    """
    try:
        resp = session.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers=WIKIMEDIA_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
    except Exception as e:
        print(f"\n  [SPARQL] Error for {wikidata_id}: {e}")
        return {}

    if not bindings:
        return {}

    row = bindings[0]

    def get_val(key):
        v = row.get(key, {}).get("value", "")
        return v if v else None

    return {
        "birth_date": get_val("birthDate"),
        "birth_place": get_val("birthPlaceLabel"),
        "record_labels": get_val("recordLabels"),
        "genres": get_val("genres"),
        "education": get_val("education"),
        "associated_acts": get_val("associatedActs"),
    }


# =============================================================================
# 3. FETCH WIKIPEDIA FR TEXT
# =============================================================================

def fetch_wikipedia_fr_text(title, session):
    """Fetch plain-text extract of a Wikipedia FR article."""
    if not title:
        return None
    time.sleep(WIKIMEDIA_DELAY)
    url = "https://fr.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "format": "json",
    }
    try:
        resp = session.get(url, params=params, headers=WIKIMEDIA_HEADERS, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
    except Exception as e:
        print(f"\n  [Wikipedia] Error for '{title}': {e}")
        return None

    for page_id, page_data in pages.items():
        if page_id == "-1":
            return None
        text = page_data.get("extract", "")
        if text:
            words = text.split()
            if len(words) > MAX_WIKI_WORDS:
                text = " ".join(words[:MAX_WIKI_WORDS])
            return text
    return None


# =============================================================================
# 4. MUSICBRAINZ API
# =============================================================================

def search_musicbrainz_artist(artist_name, session):
    """
    Search MusicBrainz for an artist and return structured metadata.
    No API key needed — just requires a proper User-Agent and max 1 req/sec.
    """
    time.sleep(MUSICBRAINZ_DELAY)

    params = {
        "query": f'artist:"{artist_name}"',
        "fmt": "json",
        "limit": 5,
    }
    try:
        resp = session.get(
            f"{MUSICBRAINZ_BASE}/artist",
            params=params,
            headers=MUSICBRAINZ_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"\n  [MusicBrainz] Search error for '{artist_name}': {e}")
        return {}

    artists = data.get("artists", [])
    if not artists:
        return {}

    # Pick best match: prefer exact name match, then top result
    best = None
    for a in artists:
        name = a.get("name", "").lower()
        if name == artist_name.lower():
            best = a
            break
    if not best:
        best = artists[0]

    mbid = best.get("id", "")

    result = {
        "mbid": mbid,
        "mb_name": best.get("name"),
        "mb_country": best.get("country"),
        "mb_type": best.get("type"),
        "mb_disambiguation": best.get("disambiguation"),
        "mb_begin_date": best.get("life-span", {}).get("begin"),
        "mb_end_date": best.get("life-span", {}).get("end"),
        "mb_area": best.get("area", {}).get("name"),
        "mb_begin_area": best.get("begin-area", {}).get("name"),
        "mb_tags": [],
        "mb_labels": [],
        "mb_related_artists": [],
        "mb_urls": {},
    }

    tags = best.get("tags", [])
    if tags:
        sorted_tags = sorted(tags, key=lambda t: t.get("count", 0), reverse=True)
        result["mb_tags"] = [t.get("name", "") for t in sorted_tags[:10]]

    # Fetch detailed info with relationships
    if mbid:
        time.sleep(MUSICBRAINZ_DELAY)
        try:
            detail_resp = session.get(
                f"{MUSICBRAINZ_BASE}/artist/{mbid}",
                params={
                    "fmt": "json",
                    "inc": "tags+artist-rels+label-rels+url-rels",
                },
                headers=MUSICBRAINZ_HEADERS,
                timeout=15,
            )
            detail_resp.raise_for_status()
            detail = detail_resp.json()
        except Exception as e:
            print(f"\n  [MusicBrainz] Detail error for '{artist_name}': {e}")
            return result

        relations = detail.get("relations", [])
        related_artists = []
        labels_from_rels = []
        urls = {}

        for rel in relations:
            rel_type = rel.get("type", "")
            target_type = rel.get("target-type", "")

            if target_type == "artist":
                related_name = rel.get("artist", {}).get("name", "")
                if related_name:
                    related_artists.append(f"{related_name} ({rel_type})")
            elif target_type == "label":
                label_name = rel.get("label", {}).get("name", "")
                if label_name:
                    labels_from_rels.append(label_name)
            elif target_type == "url":
                url_resource = rel.get("url", {}).get("resource", "")
                if url_resource:
                    urls[rel_type] = url_resource

        result["mb_related_artists"] = related_artists[:15]
        result["mb_labels"] = list(set(labels_from_rels))
        result["mb_urls"] = urls

        if not result["mb_tags"]:
            detail_tags = detail.get("tags", [])
            sorted_tags = sorted(detail_tags, key=lambda t: t.get("count", 0), reverse=True)
            result["mb_tags"] = [t.get("name", "") for t in sorted_tags[:10]]

    return result


def format_musicbrainz_as_text(mb_data):
    """Format MusicBrainz data into a text block for LLM context."""
    if not mb_data or not mb_data.get("mbid"):
        return None

    lines = []
    if mb_data.get("mb_name"):
        lines.append(f"Nom : {mb_data['mb_name']}")
    if mb_data.get("mb_begin_date"):
        lines.append(f"Date de début : {mb_data['mb_begin_date']}")
    if mb_data.get("mb_begin_area"):
        lines.append(f"Lieu d'origine : {mb_data['mb_begin_area']}")
    if mb_data.get("mb_area"):
        lines.append(f"Zone : {mb_data['mb_area']}")
    if mb_data.get("mb_country"):
        lines.append(f"Pays : {mb_data['mb_country']}")
    if mb_data.get("mb_type"):
        lines.append(f"Type : {mb_data['mb_type']}")
    if mb_data.get("mb_disambiguation"):
        lines.append(f"Précision : {mb_data['mb_disambiguation']}")
    if mb_data.get("mb_tags"):
        lines.append(f"Tags/Genres : {', '.join(mb_data['mb_tags'])}")
    if mb_data.get("mb_labels"):
        lines.append(f"Labels : {', '.join(mb_data['mb_labels'])}")
    if mb_data.get("mb_related_artists"):
        lines.append(f"Artistes liés : {', '.join(mb_data['mb_related_artists'][:10])}")

    return "\n".join(lines) if lines else None


# =============================================================================
# 5. GENIUS API
# =============================================================================

def search_genius_artist(artist_name, session):
    """
    Search Genius for an artist and return bio + metadata.
    Requires GENIUS_ACCESS_TOKEN (free at https://genius.com/api-clients).
    """
    if not GENIUS_ACCESS_TOKEN:
        return {}

    time.sleep(GENIUS_DELAY)

    headers = {
        "Authorization": f"Bearer {GENIUS_ACCESS_TOKEN}",
        "Accept": "application/json",
    }

    # Search songs to find the artist (Genius has no direct artist search)
    try:
        resp = session.get(
            f"{GENIUS_BASE}/search",
            params={"q": artist_name},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"\n  [Genius] Search error for '{artist_name}': {e}")
        return {}

    hits = data.get("response", {}).get("hits", [])
    if not hits:
        return {}

    # Find the best matching artist
    artist_id = None
    artist_info_basic = None
    artist_name_lower = artist_name.lower().strip()

    for hit in hits:
        result = hit.get("result", {})
        primary_artist = result.get("primary_artist", {})
        pa_name = primary_artist.get("name", "")

        if pa_name.lower().strip() == artist_name_lower:
            artist_id = primary_artist.get("id")
            artist_info_basic = primary_artist
            break

    # Fallback: use first hit if name is close enough
    if not artist_id and hits:
        first_artist = hits[0].get("result", {}).get("primary_artist", {})
        fa_name = first_artist.get("name", "").lower().strip()
        if (artist_name_lower in fa_name) or (fa_name in artist_name_lower):
            artist_id = first_artist.get("id")
            artist_info_basic = first_artist

    if not artist_id:
        return {}

    # Fetch full artist info
    time.sleep(GENIUS_DELAY)
    try:
        artist_resp = session.get(
            f"{GENIUS_BASE}/artists/{artist_id}",
            headers=headers,
            timeout=15,
        )
        artist_resp.raise_for_status()
        artist_data = artist_resp.json().get("response", {}).get("artist", {})
    except Exception as e:
        print(f"\n  [Genius] Artist fetch error for '{artist_name}': {e}")
        return {
            "genius_id": artist_id,
            "genius_name": artist_info_basic.get("name") if artist_info_basic else None,
            "genius_url": artist_info_basic.get("url") if artist_info_basic else None,
        }

    # Extract bio (plain text)
    description = artist_data.get("description", {})
    bio_plain = None
    if isinstance(description, dict):
        bio_plain = description.get("plain", "")
    elif isinstance(description, str):
        bio_plain = description

    if bio_plain and bio_plain.strip() and bio_plain.strip() != "?":
        words = bio_plain.split()
        if len(words) > MAX_GENIUS_BIO_WORDS:
            bio_plain = " ".join(words[:MAX_GENIUS_BIO_WORDS])
    else:
        bio_plain = None

    alternate_names = artist_data.get("alternate_names", [])
    social = {}
    for key in ["facebook_name", "instagram_name", "twitter_name"]:
        val = artist_data.get(key)
        if val:
            social[key.replace("_name", "")] = val

    return {
        "genius_id": artist_id,
        "genius_name": artist_data.get("name"),
        "genius_bio": bio_plain,
        "genius_image_url": artist_data.get("image_url"),
        "genius_url": artist_data.get("url"),
        "genius_alternate_names": alternate_names,
        "genius_social": social,
    }


def format_genius_as_text(genius_data):
    """Format Genius data into a text block for LLM context."""
    if not genius_data or not genius_data.get("genius_id"):
        return None

    lines = []
    if genius_data.get("genius_name"):
        lines.append(f"Nom (Genius) : {genius_data['genius_name']}")
    if genius_data.get("genius_alternate_names"):
        lines.append(f"Noms alternatifs : {', '.join(genius_data['genius_alternate_names'])}")
    if genius_data.get("genius_social"):
        social_parts = [f"{k}: {v}" for k, v in genius_data["genius_social"].items()]
        lines.append(f"Réseaux sociaux : {', '.join(social_parts)}")
    if genius_data.get("genius_bio"):
        lines.append(f"\nBiographie (Genius) :\n{genius_data['genius_bio']}")

    return "\n".join(lines) if lines else None


# =============================================================================
# 6. DISCOGS API
# =============================================================================

def search_discogs_artist(artist_name, session):
    """
    Search Discogs for an artist and return profile, real name, labels, groups.
    Requires DISCOGS_TOKEN (free at https://www.discogs.com/settings/developers).
    """
    if not DISCOGS_TOKEN:
        return {}

    time.sleep(DISCOGS_DELAY)

    headers = {
        **DISCOGS_HEADERS,
        "Authorization": f"Discogs token={DISCOGS_TOKEN}",
    }

    # Step 1: Search for the artist
    try:
        resp = session.get(
            f"{DISCOGS_BASE}/database/search",
            params={
                "q": artist_name,
                "type": "artist",
                "per_page": 5,
            },
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"\n  [Discogs] Search error for '{artist_name}': {e}")
        return {}

    results = data.get("results", [])
    if not results:
        return {}

    # Pick best match: prefer exact title match
    # Discogs appends " (N)" for disambiguation (e.g. "Booba (2)"), so strip it
    def _strip_discogs_suffix(title):
        return re.sub(r"\s*\(\d+\)$", "", title).lower().strip()

    best = None
    artist_name_lower = artist_name.lower().strip()
    for r in results:
        title_clean = _strip_discogs_suffix(r.get("title", ""))
        if title_clean == artist_name_lower:
            best = r
            break
    if not best:
        # Fallback: check if name is contained
        for r in results:
            title_clean = _strip_discogs_suffix(r.get("title", ""))
            if artist_name_lower in title_clean or title_clean in artist_name_lower:
                best = r
                break
    if not best:
        best = results[0]

    artist_id = best.get("id")
    if not artist_id:
        return {}

    # Step 2: Fetch full artist details
    time.sleep(DISCOGS_DELAY)
    try:
        detail_resp = session.get(
            f"{DISCOGS_BASE}/artists/{artist_id}",
            headers=headers,
            timeout=15,
        )
        detail_resp.raise_for_status()
        detail = detail_resp.json()
    except Exception as e:
        print(f"\n  [Discogs] Detail error for '{artist_name}': {e}")
        return {
            "discogs_id": artist_id,
            "discogs_name": best.get("title"),
        }

    # Extract profile text
    profile = detail.get("profile", "")
    if profile:
        words = profile.split()
        if len(words) > MAX_DISCOGS_PROFILE_WORDS:
            profile = " ".join(words[:MAX_DISCOGS_PROFILE_WORDS])
    else:
        profile = None

    # Extract real name
    realname = detail.get("realname", "")

    # Extract name variations
    namevariations = detail.get("namevariations", [])

    # Extract groups (for solo artists) or members (for groups)
    groups = []
    for g in detail.get("groups", []):
        gname = g.get("name", "")
        if gname:
            groups.append(gname)

    members = []
    for m in detail.get("members", []):
        mname = m.get("name", "")
        if mname:
            members.append(mname)

    # Extract URLs
    urls = detail.get("urls", [])

    # Step 3: Fetch releases to get label info (first page only)
    labels_set = set()
    time.sleep(DISCOGS_DELAY)
    try:
        releases_resp = session.get(
            f"{DISCOGS_BASE}/artists/{artist_id}/releases",
            params={"per_page": 25, "sort": "year", "sort_order": "desc"},
            headers=headers,
            timeout=15,
        )
        releases_resp.raise_for_status()
        releases_data = releases_resp.json()

        for release in releases_data.get("releases", []):
            label = release.get("label")
            if label and label.lower() != "not on label":
                labels_set.add(label)
    except Exception as e:
        print(f"\n  [Discogs] Releases error for '{artist_name}': {e}")

    return {
        "discogs_id": artist_id,
        "discogs_name": detail.get("name"),
        "discogs_realname": realname if realname else None,
        "discogs_profile": profile,
        "discogs_namevariations": namevariations[:10],
        "discogs_groups": groups,
        "discogs_members": members,
        "discogs_labels": sorted(labels_set)[:20],
        "discogs_urls": urls[:10],
        "discogs_url": f"https://www.discogs.com/artist/{artist_id}",
    }


def format_discogs_as_text(discogs_data):
    """Format Discogs data into a text block for LLM context."""
    if not discogs_data or not discogs_data.get("discogs_id"):
        return None

    lines = []
    if discogs_data.get("discogs_name"):
        lines.append(f"Nom (Discogs) : {discogs_data['discogs_name']}")
    if discogs_data.get("discogs_realname"):
        lines.append(f"Vrai nom : {discogs_data['discogs_realname']}")
    if discogs_data.get("discogs_namevariations"):
        lines.append(f"Variations de nom : {', '.join(discogs_data['discogs_namevariations'])}")
    if discogs_data.get("discogs_groups"):
        lines.append(f"Groupes : {', '.join(discogs_data['discogs_groups'])}")
    if discogs_data.get("discogs_members"):
        lines.append(f"Membres : {', '.join(discogs_data['discogs_members'])}")
    if discogs_data.get("discogs_labels"):
        lines.append(f"Labels (discographie) : {', '.join(discogs_data['discogs_labels'])}")
    if discogs_data.get("discogs_profile"):
        lines.append(f"\nProfil (Discogs) :\n{discogs_data['discogs_profile']}")

    return "\n".join(lines) if lines else None


# =============================================================================
# 7. CLAUDE WEB SEARCH + BIOGRAPHY SYNTHESIS
# =============================================================================

CLAUDE_SEARCH_PROMPT = """Tu es un chercheur spécialisé dans le rap français.

Recherche des informations complètes sur {entity_type} de rap français **{artist}**.

Effectue des recherches web approfondies puis rédige une biographie structurée couvrant :

1. **Origines** : lieu de naissance/enfance, vrai nom, contexte familial et social
2. **Débuts** : comment {subject} a commencé le rap, premiers projets, premières scènes
3. **Carrière** : albums majeurs, singles marquants, évolution artistique, styles/sous-genres
4. **Collaborations** : featurings notables, collectifs, producteurs récurrents
5. **Reconnaissance** : récompenses, certifications, passages médiatiques importants
6. **Controverses** : polémiques, démêlés judiciaires, faits divers (si pertinent)
7. **Vie personnelle** : éléments publiquement connus (engagements, santé mentale, événements marquants)
8. **Actualité** : activité récente, dernier projet, statut actuel

{group_instruction}

Rédige en français, de manière factuelle et sourcée. Sois exhaustif mais concis (400-600 mots)."""


def synthesize_biography_with_claude(artist_name, is_a_band, client):
    """
    Use Claude with web search to produce a narrative biography with sources.
    Returns {"biography_summary": str, "biography_sources": list} or {}.
    """
    time.sleep(ANTHROPIC_DELAY)

    if is_a_band:
        entity_type = "le groupe"
        subject = "le groupe"
        group_instruction = (
            "Inclus aussi : formation du groupe, membres (passés et actuels), "
            "dynamique interne, raisons de séparation le cas échéant."
        )
    else:
        entity_type = "l'artiste"
        subject = "l'artiste"
        group_instruction = ""

    prompt = CLAUDE_SEARCH_PROMPT.format(
        entity_type=entity_type,
        artist=artist_name,
        subject=subject,
        group_instruction=group_instruction,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=2048,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 3,
                    "user_location": {
                        "type": "approximate",
                        "country": "FR",
                        "timezone": "Europe/Paris",
                    },
                }],
                messages=[{"role": "user", "content": prompt}],
            )

            # Handle pause_turn: continue the conversation
            while response.stop_reason == "pause_turn":
                response = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=2048,
                    tools=[{
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                        "user_location": {
                            "type": "approximate",
                            "country": "FR",
                            "timezone": "Europe/Paris",
                        },
                    }],
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response.content},
                    ],
                )

            # Extract text and citation URLs from response
            text_parts = []
            sources = []
            seen_urls = set()

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    # Extract citations from text blocks
                    if hasattr(block, "citations") and block.citations:
                        for citation in block.citations:
                            url = getattr(citation, "url", None)
                            title = getattr(citation, "title", None)
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                sources.append({
                                    "url": url,
                                    "title": title or "",
                                })

            biography_text = "\n".join(text_parts).strip()

            if not biography_text:
                return {}

            return {
                "biography_summary": biography_text,
                "biography_sources": sources,
            }

        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"\n  [Claude] Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n  [Claude] Error for '{artist_name}': {e}")
                return {}
            time.sleep(1)

    return {}


# =============================================================================
# 8. EXTRACT BIOGRAPHY WITH LLM
# =============================================================================

LLM_SYSTEM_PROMPT = """Tu es un assistant de recherche en linguistique. Tu extrais des informations biographiques structurées sur des rappeurs français.

RÈGLES STRICTES :
- Retourne UNIQUEMENT du JSON valide, sans texte avant ou après.
- Si tu n'es pas sûr d'une information, retourne null pour ce champ.
- Ne fabrique JAMAIS d'information. Préfère null à une supposition.
- Base-toi uniquement sur les textes fournis et tes connaissances fiables.
- Priorité des sources : Wikipedia > Wikidata > Genius > Discogs > MusicBrainz.
- Croise les sources pour valider les informations quand c'est possible.
- Pour is_a_band : retourne true si l'entité est un groupe, duo ou collectif musical ; false si c'est un artiste solo."""

LLM_USER_TEMPLATE = """Extrais les informations biographiques de l'artiste **{artist}** à partir des textes ci-dessous.

{source_note}

Texte :
\"\"\"
{text}
\"\"\"

Retourne un objet JSON avec exactement ces clés :
{{
  "city_birth_childhood": "ville de naissance ou d'enfance (string ou null)",
  "record_labels": ["liste des labels (strings)"],
  "associated_acts": ["artistes ou groupes associés (strings)"],
  "narrative_biography": {{
    "childhood": "enfance et environnement familial (string ou null)",
    "rap_debut": "comment l'artiste a commencé le rap (string ou null)",
    "breakthrough": "moment de percée / reconnaissance (string ou null)"
  }},
  "family_early_environment": "contexte familial, origine sociale (string ou null)",
  "education": "parcours scolaire ou études (string ou null)",
  "mental_health_disclosures": "mentions de santé mentale dans sa musique ou interviews (string ou null)",
  "justice_events": "démêlés judiciaires, incarcérations (string ou null)",
  "trauma_related_facts": "événements traumatiques mentionnés (string ou null)",
  "is_a_band": "true si groupe/duo/collectif, false si artiste solo (boolean)"
}}"""


def extract_biography_with_llm(artist_name, wiki_text, wikidata_fields,
                                mb_text, genius_text, discogs_text, client):
    """
    Use GPT-4o-mini to extract structured biography from ALL available sources.
    """
    time.sleep(OPENAI_DELAY)

    parts = []
    source_notes = []

    # Wikipedia (highest priority)
    if wiki_text:
        parts.append("=== SOURCE : Wikipedia FR ===\n" + wiki_text)
        source_notes.append("Wikipedia FR")

    # Wikidata structured
    wd_lines = []
    if wikidata_fields.get("birth_place"):
        wd_lines.append(f"Lieu de naissance : {wikidata_fields['birth_place']}")
    if wikidata_fields.get("birth_date"):
        wd_lines.append(f"Date de naissance : {wikidata_fields['birth_date']}")
    if wikidata_fields.get("record_labels"):
        wd_lines.append(f"Labels : {wikidata_fields['record_labels']}")
    if wikidata_fields.get("genres"):
        wd_lines.append(f"Genres : {wikidata_fields['genres']}")
    if wikidata_fields.get("education"):
        wd_lines.append(f"Éducation : {wikidata_fields['education']}")
    if wikidata_fields.get("associated_acts"):
        wd_lines.append(f"Actes associés : {wikidata_fields['associated_acts']}")
    if wd_lines:
        parts.append("=== SOURCE : Wikidata ===\n" + "\n".join(wd_lines))
        source_notes.append("Wikidata")

    # Genius bio
    if genius_text:
        parts.append("=== SOURCE : Genius ===\n" + genius_text)
        source_notes.append("Genius")

    # Discogs
    if discogs_text:
        parts.append("=== SOURCE : Discogs ===\n" + discogs_text)
        source_notes.append("Discogs")

    # MusicBrainz
    if mb_text:
        parts.append("=== SOURCE : MusicBrainz ===\n" + mb_text)
        source_notes.append("MusicBrainz")

    if source_notes:
        source_note = "Sources disponibles : " + ", ".join(source_notes) + "."
    else:
        source_note = (
            "Aucune source externe trouvée. Utilise uniquement tes connaissances "
            "fiables. Retourne null pour tout champ incertain."
        )

    text = (
        "\n\n".join(parts)
        if parts
        else f"Artiste : {artist_name}. Aucune source externe disponible."
    )

    user_msg = LLM_USER_TEMPLATE.format(
        artist=artist_name,
        source_note=source_note,
        text=text,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError:
            return {}
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** (attempt + 1)
                print(f"\n  Rate limited by OpenAI, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt == max_retries - 1:
                return {}
            time.sleep(1)
    return {}


# =============================================================================
# 8. MERGE & SAVE UTILITIES
# =============================================================================

def build_artist_record(artist_name, wikidata_id, wiki_title, wikidata_fields,
                        wiki_text, mb_data, genius_data, discogs_data,
                        llm_result, claude_data=None):
    """Merge all sources into a single artist record."""
    has_wiki = wiki_text is not None
    has_wikidata = bool(wikidata_id)
    has_mb = bool(mb_data and mb_data.get("mbid"))
    has_genius = bool(genius_data and genius_data.get("genius_id"))
    has_discogs = bool(discogs_data and discogs_data.get("discogs_id"))
    has_claude = bool(claude_data and claude_data.get("biography_summary"))

    sources = []
    if has_wikidata:
        sources.append("wikidata")
    if has_wiki:
        sources.append("wikipedia")
    if has_genius:
        sources.append("genius")
    if has_discogs:
        sources.append("discogs")
    if has_mb:
        sources.append("musicbrainz")
    sources.append("llm")
    if has_claude:
        sources.append("claude")
    source = "+".join(sources)

    # Determine is_a_band from structural data + LLM
    mb_is_group = (mb_data.get("mb_type") == "Group") if mb_data else False
    discogs_is_group = bool(discogs_data.get("discogs_members")) if discogs_data else False
    llm_says_band = bool(llm_result.get("is_a_band")) if llm_result else False
    is_a_band = mb_is_group or discogs_is_group or llm_says_band

    if not claude_data:
        claude_data = {}

    return {
        "artist": artist_name,
        "source": source,
        "is_a_band": is_a_band,

        # === Wikidata ===
        "wikidata_id": wikidata_id,
        "wikipedia_fr_title": wiki_title,
        "birth_date": wikidata_fields.get("birth_date"),
        "birth_place": wikidata_fields.get("birth_place"),
        "wikidata_record_labels": wikidata_fields.get("record_labels"),
        "wikidata_genres": wikidata_fields.get("genres"),
        "wikidata_education": wikidata_fields.get("education"),
        "wikidata_associated_acts": wikidata_fields.get("associated_acts"),

        # === MusicBrainz ===
        "musicbrainz_id": mb_data.get("mbid") if mb_data else None,
        "mb_type": mb_data.get("mb_type") if mb_data else None,
        "mb_name": mb_data.get("mb_name") if mb_data else None,
        "mb_country": mb_data.get("mb_country") if mb_data else None,
        "mb_area": mb_data.get("mb_area") if mb_data else None,
        "mb_begin_area": mb_data.get("mb_begin_area") if mb_data else None,
        "mb_begin_date": mb_data.get("mb_begin_date") if mb_data else None,
        "mb_tags": mb_data.get("mb_tags", []) if mb_data else [],
        "mb_labels": mb_data.get("mb_labels", []) if mb_data else [],
        "mb_related_artists": mb_data.get("mb_related_artists", []) if mb_data else [],
        "mb_urls": mb_data.get("mb_urls", {}) if mb_data else {},

        # === Genius ===
        "genius_id": genius_data.get("genius_id") if genius_data else None,
        "genius_name": genius_data.get("genius_name") if genius_data else None,
        "genius_url": genius_data.get("genius_url") if genius_data else None,
        "genius_alternate_names": genius_data.get("genius_alternate_names", []) if genius_data else [],
        "genius_social": genius_data.get("genius_social", {}) if genius_data else {},
        "genius_image_url": genius_data.get("genius_image_url") if genius_data else None,

        # === Discogs ===
        "discogs_id": discogs_data.get("discogs_id") if discogs_data else None,
        "discogs_name": discogs_data.get("discogs_name") if discogs_data else None,
        "discogs_realname": discogs_data.get("discogs_realname") if discogs_data else None,
        "discogs_url": discogs_data.get("discogs_url") if discogs_data else None,
        "discogs_namevariations": discogs_data.get("discogs_namevariations", []) if discogs_data else [],
        "discogs_groups": discogs_data.get("discogs_groups", []) if discogs_data else [],
        "discogs_members": discogs_data.get("discogs_members", []) if discogs_data else [],
        "discogs_labels": discogs_data.get("discogs_labels", []) if discogs_data else [],
        "discogs_urls": discogs_data.get("discogs_urls", []) if discogs_data else [],

        # === LLM extracted ===
        "city_birth_childhood": llm_result.get("city_birth_childhood"),
        "record_labels": llm_result.get("record_labels"),
        "associated_acts": llm_result.get("associated_acts"),
        "narrative_biography": llm_result.get("narrative_biography"),
        "family_early_environment": llm_result.get("family_early_environment"),
        "education": llm_result.get("education"),
        "mental_health_disclosures": llm_result.get("mental_health_disclosures"),
        "justice_events": llm_result.get("justice_events"),
        "trauma_related_facts": llm_result.get("trauma_related_facts"),

        # === Claude web search ===
        "biography_summary": claude_data.get("biography_summary"),
        "biography_sources": claude_data.get("biography_sources", []),
    }


def save_progress(results, skipped):
    """Save current results to all output files."""
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        rows = []
        for r in results:
            flat = {}
            for k, v in r.items():
                if isinstance(v, list):
                    flat[k] = " | ".join(str(x) for x in v)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        flat[f"{k}_{sub_k}"] = sub_v
                else:
                    flat[k] = v
            rows.append(flat)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv(OUTPUT_SKIPPED, index=False)


# =============================================================================
# MAIN
# =============================================================================

def get_unique_artists():
    """Extract sorted unique artist names from the corpus CSV."""
    artists = set()
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            artists.add(row["artist"])
    return sorted(artists)


def main():
    parser = argparse.ArgumentParser(
        description="Collect structured biographies for French rappers"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N artists (for testing)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignoring any existing progress")
    parser.add_argument("--no-claude", action="store_true",
                        help="Skip Claude web search synthesis step")
    args = parser.parse_args()

    use_claude = ANTHROPIC_API_KEY and not args.no_claude

    print("=" * 70)
    print("FRENCH RAP - BIOGRAPHY COLLECTOR (7 sources)")
    print("=" * 70)

    # Check API configurations
    print("\nAPI Status:")
    print(f"  Wikidata + Wikipedia : ENABLED (no key needed)")
    print(f"  MusicBrainz          : ENABLED (no key needed)")
    if GENIUS_ACCESS_TOKEN:
        print(f"  Genius               : ENABLED")
    else:
        print(f"  Genius               : DISABLED (set GENIUS_ACCESS_TOKEN)")
    if DISCOGS_TOKEN:
        print(f"  Discogs              : ENABLED")
    else:
        print(f"  Discogs              : DISABLED (set DISCOGS_TOKEN)")
    print(f"  OpenAI (GPT-4o-mini) : ENABLED (via OPENAI_API_KEY)")
    if use_claude:
        print(f"  Claude ({ANTHROPIC_MODEL}) : ENABLED (web search)")
    elif args.no_claude:
        print(f"  Claude               : DISABLED (--no-claude flag)")
    else:
        print(f"  Claude               : DISABLED (set ANTHROPIC_API_KEY)")
    print()

    # Load artists
    all_artists = get_unique_artists()
    print(f"Total unique artists in corpus: {len(all_artists)}")

    # Load existing progress
    results = []
    skipped = []
    processed_names = set()

    if not args.no_resume and OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed_names = {r["artist"] for r in results}
        print(f"Resuming: {len(processed_names)} artists already processed")

    if not args.no_resume and OUTPUT_SKIPPED.exists():
        skipped_df = pd.read_csv(OUTPUT_SKIPPED)
        skipped = skipped_df.to_dict("records")
        processed_names |= {s["artist"] for s in skipped}

    # Filter to unprocessed artists
    to_process = [a for a in all_artists if a not in processed_names]
    if args.limit:
        to_process = to_process[:args.limit]

    print(f"Artists to process: {len(to_process)}")
    if not to_process:
        print("Nothing to do. All artists already processed.")
        return

    # Estimate: ~12s per artist without Claude, ~18s with Claude
    secs_per_artist = 18 if use_claude else 12
    est_minutes = len(to_process) * secs_per_artist / 60
    print(f"Estimated time: ~{est_minutes:.0f} minutes")
    print("(Press Ctrl+C to stop and save progress)\n")

    # Init clients
    client = OpenAI()
    anthropic_client = anthropic.Anthropic() if use_claude else None
    session = requests.Session()

    count_since_save = 0

    try:
        for artist_name in tqdm(to_process, desc="Collecting biographies"):
            # Step 1: Resolve Wikidata entity
            wikidata_id, wiki_title = resolve_wikidata_entity(artist_name, session)

            # Step 2: Fetch Wikidata structured fields
            wikidata_fields = {}
            if wikidata_id:
                wikidata_fields = fetch_wikidata_structured(wikidata_id, session)

            # Step 3: Fetch Wikipedia FR text
            wiki_text = fetch_wikipedia_fr_text(wiki_title, session)

            # Step 4: MusicBrainz
            mb_data = search_musicbrainz_artist(artist_name, session)
            mb_text = format_musicbrainz_as_text(mb_data)

            # Step 5: Genius
            genius_data = search_genius_artist(artist_name, session)
            genius_text = format_genius_as_text(genius_data)

            # Step 6: Discogs
            discogs_data = search_discogs_artist(artist_name, session)
            discogs_text = format_discogs_as_text(discogs_data)

            # Step 7: Extract biography with LLM (ALL sources)
            llm_result = extract_biography_with_llm(
                artist_name, wiki_text, wikidata_fields,
                mb_text, genius_text, discogs_text, client
            )

            # Pre-compute is_a_band for Claude prompt
            mb_is_group = (mb_data.get("mb_type") == "Group") if mb_data else False
            discogs_is_group = bool(discogs_data.get("discogs_members")) if discogs_data else False
            llm_says_band = bool(llm_result.get("is_a_band")) if llm_result else False
            is_a_band = mb_is_group or discogs_is_group or llm_says_band

            # Step 8: Claude web search + biography synthesis
            claude_data = {}
            if anthropic_client:
                claude_data = synthesize_biography_with_claude(
                    artist_name, is_a_band, anthropic_client
                )

            # Step 9: Merge and store
            has_any_source = (
                wikidata_id
                or (mb_data and mb_data.get("mbid"))
                or (genius_data and genius_data.get("genius_id"))
                or (discogs_data and discogs_data.get("discogs_id"))
                or llm_result
                or claude_data
            )

            if not has_any_source:
                skipped.append({
                    "artist": artist_name,
                    "reason": "no_sources_found_and_llm_failed",
                })
            else:
                record = build_artist_record(
                    artist_name, wikidata_id, wiki_title,
                    wikidata_fields, wiki_text,
                    mb_data, genius_data, discogs_data, llm_result,
                    claude_data,
                )
                results.append(record)

            count_since_save += 1
            if count_since_save >= 10:
                save_progress(results, skipped)
                count_since_save = 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")

    # Final save
    save_progress(results, skipped)

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"Artists processed:  {len(results)}")
    print(f"Artists skipped:    {len(skipped)}")

    source_counts = {}
    for r in results:
        s = r.get("source", "unknown")
        source_counts[s] = source_counts.get(s, 0) + 1

    print("\nSource breakdown:")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    print(f"\nOutputs:")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_SKIPPED}")


if __name__ == "__main__":
    main()
