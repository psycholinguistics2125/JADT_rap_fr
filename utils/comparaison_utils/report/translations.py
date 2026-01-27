#!/usr/bin/env python3
"""
Translation utilities for report generation.
"""

import json
from pathlib import Path

from ..constants import METRIC_REFERENCES


_TRANSLATIONS = None


def _load_translations() -> dict:
    """Load translations from JSON file (cached)."""
    global _TRANSLATIONS
    if _TRANSLATIONS is None:
        json_path = Path(__file__).parent / 'translations.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            _TRANSLATIONS = json.load(f)
    return _TRANSLATIONS


def get_text(key: str, lang: str = 'fr') -> str:
    """Get translated text for a key."""
    translations = _load_translations()
    return translations.get(lang, translations['fr']).get(key, key)


def get_metric_description(metric_key: str, lang: str = 'fr') -> str:
    """Get metric description in the appropriate language."""
    ref = METRIC_REFERENCES.get(metric_key, {})
    if lang == 'fr' and 'description_fr' in ref:
        return ref['description_fr']
    return ref.get('description', '')
