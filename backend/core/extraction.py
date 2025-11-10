# backend/core/extraction_v2.py
"""
✅ IMPROVED: Extraction with keywords for relationships
Based on LightRAG architecture
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
llm_response_cache = {}
logger = logging.getLogger(__name__)


def clean_llm_output(text: str) -> str:
    """Clean LLM output"""
    if not text:
        return ""
    
    cleaned = text.strip()
    cleaned = cleaned.replace(" <|>", TUPLE_DELIMITER)
    cleaned = cleaned.replace("<|> ", TUPLE_DELIMITER)
    cleaned = cleaned.replace(" ## ", RECORD_DELIMITER)
    cleaned = cleaned.replace("## ", RECORD_DELIMITER)
    cleaned = cleaned.replace(" ##", RECORD_DELIMITER)
    cleaned = cleaned.replace(" <|COMPLETE|>", COMPLETION_DELIMITER)
    cleaned = cleaned.replace("\ufeff", "")
    
    return cleaned


# ==================== IMPROVED PROMPT ====================
def create_extraction_prompt(chunk_text: str, entity_types: List[str]) -> str:
    """
    ✅ IMPROVED: Prompt with keywords extraction for relationships
    """
    entity_types_str = ", ".join(entity_types)
    
    return f"""
-Goal-
Extract entities and relationships from the text. For relationships, also extract keywords.

-Steps-
1. Identify all entities:
   Format: ("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<description>)
   
   Entity types: {entity_types_str}

2. Identify relationships with keywords:
   Format: ("relationship"{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<description>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<strength>)
   
   - Keywords: 2-5 key verbs/nouns describing the relationship (comma-separated)
   - Strength: 0.0-1.0 (how confident you are about this relationship)

3. Output format:
   (<record>{RECORD_DELIMITER}<record>{RECORD_DELIMITER}...{COMPLETION_DELIMITER})

-Examples-
("entity"{TUPLE_DELIMITER}Apple Inc.{TUPLE_DELIMITER}ORGANIZATION{TUPLE_DELIMITER}Technology company that makes iPhone)
("entity"{TUPLE_DELIMITER}iPhone{TUPLE_DELIMITER}PRODUCT{TUPLE_DELIMITER}Smartphone product by Apple)
("relationship"{TUPLE_DELIMITER}Apple Inc.{TUPLE_DELIMITER}iPhone{TUPLE_DELIMITER}Apple manufactures and sells iPhone{TUPLE_DELIMITER}manufactures, sells, produces{TUPLE_DELIMITER}0.95)

-Real Data-
Text: {chunk_text}

Output:
"""


# ==================== IMPROVED PARSER ====================
def parse_extraction_result(result: str) -> List[Dict[str, Any]]:
    """
    ✅ IMPROVED: Parse with keywords support
    """
    records = []
    result = clean_llm_output(result)
    result = result.replace(COMPLETION_DELIMITER, "").strip()
    
    raw_records = result.split(RECORD_DELIMITER)
    
    for raw_record in raw_records:
        raw_record = raw_record.strip()
        if not raw_record:
            continue
        
        parts = [p.strip() for p in raw_record.split(TUPLE_DELIMITER)]
        if len(parts) < 2:
            continue
        
        record_type = parts[0].lower()
        
        # ========== ENTITY ==========
        if record_type == "entity" and len(parts) >= 4:
            records.append({
                "type": "entity",
                "entity_name": parts[1],
                "entity_type": parts[2],
                "description": parts[3],
            })
        
        # ========== RELATIONSHIP WITH KEYWORDS ==========
        elif record_type == "relationship":
            if len(parts) >= 6:
                # New format: source, target, description, keywords, strength
                try:
                    strength = float(parts[5])
                except ValueError:
                    strength = 1.0
                
                records.append({
                    "type": "relationship",
                    "source_entity": parts[1],
                    "target_entity": parts[2],
                    "description": parts[3],
                    "keywords": parts[4],  # ✅ NEW
                    "strength": strength,
                })
            
            elif len(parts) >= 5:
                # Old format: source, target, description, strength (no keywords)
                try:
                    strength = float(parts[4])
                except ValueError:
                    strength = 1.0
                
                # Extract keywords from description
                keywords = extract_keywords_from_description(parts[3])
                
                records.append({
                    "type": "relationship",
                    "source_entity": parts[1],
                    "target_entity": parts[2],
                    "description": parts[3],
                    "keywords": keywords,  # ✅ Extracted from description
                    "strength": strength,
                })
    
    return records


def extract_keywords_from_description(description: str) -> str:
    """
    ✅ NEW: Extract keywords from relationship description
    (Fallback when LLM doesn't provide keywords)
    """
    # Simple keyword extraction using common verbs
    verbs = [
        "produces", "creates", "develops", "founded", "located", "works",
        "manages", "owns", "uses", "builds", "sells", "provides", "offers",
        "leads", "operates", "manufactures", "designs", "invented"
    ]
    
    desc_lower = description.lower()
    found_keywords = [v for v in verbs if v in desc_lower]
    
    if found_keywords:
        return ", ".join(found_keywords[:3])  # Max 3 keywords
    
    # Fallback: first 3 words
    words = description.split()[:3]
    return ", ".join(words)


# ==================== HANDLERS ====================
def handle_single_entity_extraction(record: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    """Handle entity extraction"""
    return {
        "entity_name": record["entity_name"],
        "entity_type": record["entity_type"],
        "description": record["description"],
        "chunk_id": chunk_id,
        "source_id": chunk_id
    }


def handle_single_relationship_extraction(record: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    """
    ✅ IMPROVED: Handle relationship with keywords
    """
    return {
        "source_id": record["source_entity"],
        "target_id": record["target_entity"],
        "description": record["description"],
        "keywords": record.get("keywords", ""),  # ✅ NEW
        "strength": record.get("strength", 1.0),
        "chunk_id": chunk_id
    }


def process_extraction_result(records: List[Dict[str, Any]], chunk_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Process extraction results"""
    entities, relationships = [], []
    
    for r in records:
        if r["type"] == "entity":
            entities.append(handle_single_entity_extraction(r, chunk_id))
        elif r["type"] == "relationship":
            relationships.append(handle_single_relationship_extraction(r, chunk_id))
    
    return entities, relationships


# ==================== ASYNC EXTRACTION ====================
async def extract_single_chunk(
    chunk: Dict[str, Any],
    entity_types: List[str],
    model: str,
    use_cache: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """Extract from single chunk"""
    chunk_id = chunk.get("chunk_id", "")
    chunk_text = chunk.get("content", "")
    cache_key = f"{chunk_id}_{hash(chunk_text)}"
    
    try:
        # Check cache
        if use_cache and cache_key in llm_response_cache:
            result = llm_response_cache[cache_key]
        else:
            prompt = create_extraction_prompt(chunk_text, entity_types)
            from backend.utils.llm_utils import call_llm_with_retry
            import os
            model = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
            result = await call_llm_with_retry(prompt, model=model, max_retries=3)
            
            if use_cache:
                llm_response_cache[cache_key] = result
        
        # Parse and process
        records = parse_extraction_result(result)
        entities, relationships = process_extraction_result(records, chunk_id)
        
        return entities, relationships
    
    except asyncio.TimeoutError:
        logger.error(f"⏰ Timeout while extracting chunk {chunk_id}")
        return [], []
    except Exception as e:
        logger.error(f"❌ Error processing chunk {chunk_id}: {e}")
        return [], []


async def extract_entities(
    chunks: List[Dict[str, Any]],
    global_config: Dict[str, Any],
    max_concurrent: int = 5,
    use_cache: bool = True
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """Extract from multiple chunks in parallel"""
    entity_types = global_config.get("entity_types", [
        "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "CONCEPT", "TECHNOLOGY"
    ])
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(chunk):
        async with semaphore:
            return await extract_single_chunk(chunk, entity_types, "", use_cache)
    
    tasks = [process_with_semaphore(c) for c in chunks]
    results = await asyncio.gather(*tasks)
    
    entities_dict, relationships_dict = {}, {}
    
    for i, (ents, rels) in enumerate(results):
        chunk_id = chunks[i].get("chunk_id", f"chunk_{i}")
        if ents:
            entities_dict[chunk_id] = ents
        if rels:
            relationships_dict[chunk_id] = rels
    
    return entities_dict, relationships_dict


def extract_entities_relations(
    chunks: List[Dict[str, Any]],
    global_config: Optional[Dict[str, Any]] = None
) -> Tuple[Dict, Dict]:
    """Sync entry point for extraction"""
    if global_config is None:
        global_config = {
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "CONCEPT", "TECHNOLOGY"]
        }
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            logger.warning("⚠️ nest_asyncio not installed")
    
    nodes, edges = loop.run_until_complete(extract_entities(chunks, global_config))
    
    return nodes, edges


# ==================== KEYWORD ANALYSIS ====================
def analyze_relationship_keywords(relationships_dict: Dict[str, List[Dict]]) -> Dict[str, int]:
    """
    ✅ NEW: Analyze most common keywords in relationships
    """
    from collections import Counter
    
    all_keywords = []
    for rels in relationships_dict.values():
        for rel in rels:
            keywords = rel.get('keywords', '')
            if keywords:
                all_keywords.extend([k.strip() for k in keywords.split(',')])
    
    return dict(Counter(all_keywords).most_common(20))