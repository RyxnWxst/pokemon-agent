"""
Vocabularies for Pokémon Showdown observations.

Loads vocab.json, builds the name-to-ID tables, and provides high-performance
string normalization and categorical ID lookups with unknown-entity tracking.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Counter as CounterType

from pokemon_agent.paths import VOCAB_PATH

# Configure logger
logger = logging.getLogger(__name__)

# Global trackers for debugging out-of-vocabulary (OOV) items
UNKNOWN_ENTITIES: CounterType[str] = Counter()

# High-performance memoization cache for string normalization
_NORM_CACHE: Dict[Any, str] = {}


def normalize_name(value: Any) -> Optional[str]:
    """
    Normalizes a poke-env object or string into a consistent lowercase format.
    
    Uses an internal memoization cache (_NORM_CACHE) to avoid repeated string 
    manipulation overhead during high-throughput rollout generation.

    Args:
        value: The raw object (string, Enum, or poke-env data class) to normalize.

    Returns:
        Optional[str]: The cleaned string, or None if the input represents a null state.
    """
    if value is None or value == "" or str(value).lower() == "none":
        return None

    target = value

    # EAFP (Easier to Ask Forgiveness than Permission) is the fastest approach in Python
    try:
        return _NORM_CACHE[target]
    except KeyError:
        pass # Hashable, but not yet cached
    except TypeError:
        pass # Unhashable type (e.g., dict/list), skip cache check

    # Extract string representation based on common poke-env object structures
    if hasattr(target, 'name'):
        val_str = str(target.name)
    elif hasattr(target, 'id'):
        val_str = str(target.id)
    else:
        # Fallback: Strip out specific formatting like ' (active)' or ' (fainted)'
        val_str = str(target).split(' (')[0]

    # Clean the string to match standard Showdown ID formats
    clean_val = val_str.lower().replace(" ", "").replace("-", "").replace("_", "").strip()
    
    # Attempt to cache the result for future lookups
    try:
        _NORM_CACHE[target] = clean_val
    except TypeError:
        pass # If target is unhashable, just return the value without caching
        
    return clean_val


def get_id(vocab_map: Dict[str, Dict[str, int]], path: str, value: Any) -> int:
    """
    Looks up a categorical value in the vocabulary map to get its integer ID.
    
    Note on 1-based indexing:
    Returns (ID + 1) because index 0 is strictly reserved across the model 
    architecture for padding, "None", or "Unknown" entities.

    Args:
        vocab_map: The nested dictionary of vocabularies.
        path: The specific vocabulary category (e.g., "pokemon.ability").
        value: The raw object or string to look up.

    Returns:
        int: The 1-based vocabulary index, or 0 if not found/unknown.
    """
    clean_val = normalize_name(value)
    
    if clean_val is None:
        return 0
    
    category_map = vocab_map.get(path)
    if not category_map:
        logger.debug(f"Vocabulary path '{path}' not found in vocab_map.")
        return 0

    val_id = category_map.get(clean_val, -1)
    
    # Track Out-Of-Vocabulary (OOV) items for later analysis
    if val_id == -1:
        actual_raw = str(getattr(value, 'name', value))
        UNKNOWN_ENTITIES[f"{path} | Raw: {actual_raw} | Normalized: {clean_val}"] += 1
        return 0
        
    return val_id + 1


def load_vocab(path: Optional[str | Path] = None) -> Dict[str, List[str]]:
    """Loads the vocabulary lists (category -> names); defaults to the bundled vocab.json."""
    path = Path(path) if path is not None else VOCAB_PATH
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_vocab_map(vocab_lists: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """Builds the name -> 0-based index tables that get_id reads."""
    return {
        category: {item.lower(): i for i, item in enumerate(items)}
        for category, items in vocab_lists.items()
    }
