# backend/db/entity_linking.py
"""
🔗 Enhanced Entity Linking
Multi-level fuzzy matching for better entity deduplication
"""

import re
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name for comparison
    - Remove extra spaces
    - Lowercase
    - Remove special characters (for acronym matching)
    """
    # Basic normalization
    normalized = name.strip().lower()
    return normalized


def extract_acronym(name: str) -> str:
    """
    Extract acronym from entity name
    Examples:
        "OpenAI Inc." → "openaiinc"
        "GPT-4" → "gpt4"
        "United States" → "us"
    """
    # Remove all non-alphanumeric characters
    clean = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
    return clean


def is_acronym_match(name1: str, name2: str) -> bool:
    """
    Check if one name is acronym/abbreviation of another
    
    Examples:
        "GPT4" vs "GPT-4" → True
        "OpenAI Inc" vs "OpenAI" → True
        "US" vs "United States" → False (too short, risky)
    """
    acronym1 = extract_acronym(name1)
    acronym2 = extract_acronym(name2)
    
    # Exact match after removing special chars
    if acronym1 == acronym2:
        return True
    
    # Check if one is substring of another (for abbreviations)
    # But only if both are reasonably long to avoid false positives
    if len(acronym1) >= 4 and len(acronym2) >= 4:
        if acronym1 in acronym2 or acronym2 in acronym1:
            return True
    
    return False


def calculate_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two entity names
    Uses SequenceMatcher ratio
    """
    normalized1 = normalize_entity_name(name1)
    normalized2 = normalize_entity_name(name2)
    
    return SequenceMatcher(None, normalized1, normalized2).ratio()


def fuzzy_match_entity(
    entity_name: str,
    existing_entities: List[Dict],
    strict_mode: bool = False,
    min_length: int = 3
) -> Optional[Tuple[str, float, str]]:
    """
    Multi-level fuzzy matching for entity linking
    
    Matching levels:
    1. Exact match (case-insensitive)
    2. High similarity (>0.9) - for strict mode
    3. Medium similarity (>0.8) - for normal mode  
    4. Acronym/abbreviation matching
    
    Args:
        entity_name: Entity name to match
        existing_entities: List of existing entity dicts with 'entity_name' field
        strict_mode: Use stricter threshold (0.9 vs 0.8)
        min_length: Minimum entity name length to consider (avoid matching short names)
    
    Returns:
        Tuple of (canonical_name, similarity_score, match_type) or None
        match_type: 'exact', 'high_similarity', 'medium_similarity', 'acronym'
    """
    if not entity_name or len(entity_name) < min_length:
        return None
    
    entity_lower = normalize_entity_name(entity_name)
    
    # Level 1: Exact match (case-insensitive)
    for existing in existing_entities:
        existing_name = existing['entity_name']
        if entity_lower == normalize_entity_name(existing_name):
            return (existing_name, 1.0, 'exact')
    
    # Level 2-3: Similarity matching
    threshold = 0.9 if strict_mode else 0.8
    best_match = None
    best_score = threshold
    
    for existing in existing_entities:
        existing_name = existing['entity_name']
        score = calculate_similarity(entity_name, existing_name)
        
        if score > best_score:
            best_score = score
            best_match = existing_name
    
    if best_match:
        match_type = 'high_similarity' if best_score >= 0.9 else 'medium_similarity'
        return (best_match, best_score, match_type)
    
    # Level 4: Acronym matching
    # Only for entities with reasonable length
    if len(entity_name) >= 4:
        for existing in existing_entities:
            existing_name = existing['entity_name']
            if len(existing_name) >= 4 and is_acronym_match(entity_name, existing_name):
                # Calculate similarity for logging
                score = calculate_similarity(entity_name, existing_name)
                return (existing_name, score, 'acronym')
    
    return None


def link_entities_batch(
    entities_dict: Dict[str, List[Dict]],
    existing_entities: List[Dict],
    strict_mode: bool = False
) -> Tuple[Dict[str, str], Dict[str, Tuple[float, str]]]:
    """
    Batch entity linking
    
    Args:
        entities_dict: Dict of {entity_name: [entity_dicts]}
        existing_entities: List of existing entities from DB
        strict_mode: Use stricter matching threshold
    
    Returns:
        Tuple of:
        - canonical_mapping: Dict of {original_name: canonical_name}
        - match_info: Dict of {original_name: (similarity_score, match_type)}
    """
    canonical_mapping = {}
    match_info = {}
    
    for entity_name in entities_dict.keys():
        match_result = fuzzy_match_entity(
            entity_name,
            existing_entities,
            strict_mode=strict_mode
        )
        
        if match_result:
            canonical_name, score, match_type = match_result
            canonical_mapping[entity_name] = canonical_name
            match_info[entity_name] = (score, match_type)
            
            logger.debug(
                f"🔗 Linked '{entity_name}' → '{canonical_name}' "
                f"(score: {score:.2f}, type: {match_type})"
            )
        else:
            # No match, use original name
            canonical_mapping[entity_name] = entity_name
    
    return canonical_mapping, match_info


def get_linking_statistics(match_info: Dict[str, Tuple[float, str]]) -> Dict:
    """
    Get statistics about entity linking
    
    Returns:
        Dict with counts by match type
    """
    stats = {
        'total': len(match_info),
        'exact': 0,
        'high_similarity': 0,
        'medium_similarity': 0,
        'acronym': 0,
        'no_match': 0
    }
    
    for entity_name, (score, match_type) in match_info.items():
        stats[match_type] = stats.get(match_type, 0) + 1
    
    return stats


# ================= Export =================

__all__ = [
    'fuzzy_match_entity',
    'link_entities_batch',
    'is_acronym_match',
    'calculate_similarity',
    'normalize_entity_name',
    'get_linking_statistics'
]
