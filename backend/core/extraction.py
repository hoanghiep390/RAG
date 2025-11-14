# backend/core/extraction.py
"""
✅ OPTIMIZED: Entity Extraction with Higher Concurrency
- 16 parallel LLM calls (increased from 8)
- Better error handling
- Retry logic for failed calls
- Progress tracking
"""

import asyncio
import logging
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
GRAPH_FIELD_SEP = "; "

def normalize_extracted_info(text: str, is_entity: bool = False) -> str:
    """Normalize extracted text following LightRAG rules"""
    if not text or not text.strip():
        return ""
    
    text = text.strip().strip('"').strip("'").strip()
    
    if is_entity:
        text = ' '.join(text.split())
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    else:
        text = ' '.join(text.split())
    
    return text


def clean_str(text: str) -> str:
    """Clean string by removing special characters"""
    if not text:
        return ""
    return text.strip().strip('"').strip("'")


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
) -> Optional[Dict]:
    """Handle single entity extraction following LightRAG logic"""
    if len(record_attributes) < 4 or '"entity"' not in record_attributes[0]:
        return None

    entity_name = clean_str(record_attributes[1]).strip()
    if not entity_name:
        logger.warning(f"Entity extraction error: empty entity name in: {record_attributes}")
        return None

    entity_name = normalize_extracted_info(entity_name, is_entity=True)

    if not entity_name or not entity_name.strip():
        logger.warning(
            f"Entity extraction error: entity name became empty after normalization. "
            f"Original: '{record_attributes[1]}'"
        )
        return None

    entity_type = clean_str(record_attributes[2]).strip('"')
    if not entity_type.strip() or entity_type.startswith('("'):
        logger.warning(f"Entity extraction error: invalid entity type in: {record_attributes}")
        return None

    entity_description = clean_str(record_attributes[3])
    entity_description = normalize_extracted_info(entity_description)

    if not entity_description.strip():
        logger.warning(
            f"Entity extraction error: empty description for entity '{entity_name}' "
            f"of type '{entity_type}'"
        )
        return None

    return {
        'entity_name': entity_name,
        'entity_type': entity_type,
        'description': entity_description,
        'source_id': chunk_key,
        'chunk_id': chunk_key,
        'file_path': file_path,
    }


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
) -> Optional[Dict]:
    """Handle single relationship extraction following LightRAG logic"""
    if len(record_attributes) < 5 or '"relationship"' not in record_attributes[0]:
        return None

    source = clean_str(record_attributes[1])
    target = clean_str(record_attributes[2])

    source = normalize_extracted_info(source, is_entity=True)
    target = normalize_extracted_info(target, is_entity=True)

    if not source or not source.strip():
        logger.warning(
            f"Relationship extraction error: source entity became empty after normalization. "
            f"Original: '{record_attributes[1]}'"
        )
        return None

    if not target or not target.strip():
        logger.warning(
            f"Relationship extraction error: target entity became empty after normalization. "
            f"Original: '{record_attributes[2]}'"
        )
        return None

    if source == target:
        logger.debug(f"Relationship source and target are the same in: {record_attributes}")
        return None

    edge_description = clean_str(record_attributes[3])
    edge_description = normalize_extracted_info(edge_description)

    edge_keywords = normalize_extracted_info(
        clean_str(record_attributes[4]), is_entity=True
    )
    edge_keywords = edge_keywords.replace("ï¼Œ", ",")

    weight = 1.0
    if len(record_attributes) > 5:
        try:
            weight_str = record_attributes[5].strip('"').strip("'")
            if weight_str and re.match(r'^[0-9]*\.?[0-9]+$', weight_str):
                weight = float(weight_str)
        except (ValueError, IndexError):
            pass

    return {
        'source_id': source,
        'target_id': target,
        'weight': weight,
        'strength': weight,
        'description': edge_description,
        'keywords': edge_keywords,
        'source_id': chunk_key,
        'chunk_id': chunk_key,
        'file_path': file_path,
    }


def split_string_by_multi_markers(text: str, markers: List[str]) -> List[str]:
    """Split string by multiple markers"""
    if not markers:
        return [text]
    
    pattern = '|'.join(re.escape(m) for m in markers)
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]


async def parse_extraction_result(
    result: str,
    chunk_key: str,
    file_path: str = "unknown_source"
) -> Tuple[Dict[str, List[Dict]], Dict[Tuple[str, str], List[Dict]]]:
    """Parse LLM extraction result into entities and relationships"""
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)

    context_base = {
        'tuple_delimiter': TUPLE_DELIMITER,
        'record_delimiter': RECORD_DELIMITER,
        'completion_delimiter': COMPLETION_DELIMITER,
    }

    records = split_string_by_multi_markers(
        result,
        [context_base["record_delimiter"], context_base["completion_delimiter"]],
    )

    for record in records:
        match = re.search(r"\((.*)\)", record)
        if match is None:
            continue
        
        record_content = match.group(1)
        record_attributes = split_string_by_multi_markers(
            record_content, [context_base["tuple_delimiter"]]
        )
        
        entity_data = await _handle_single_entity_extraction(
            record_attributes, chunk_key, file_path
        )
        if entity_data is not None:
            maybe_nodes[entity_data["entity_name"]].append(entity_data)
            continue
        
        relationship_data = await _handle_single_relationship_extraction(
            record_attributes, chunk_key, file_path
        )
        if relationship_data is not None:
            rel_key = (relationship_data["source_id"], relationship_data["target_id"])
            maybe_edges[rel_key].append(relationship_data)

    return dict(maybe_nodes), dict(maybe_edges)


def create_extraction_prompt(text: str, entity_types: List[str]) -> str:
    """Create extraction prompt following LightRAG format"""
    types_str = ", ".join(entity_types)
    
    return f"""
-Goal-
Extract entities and relationships from the text below.

-Steps-
1. Identify ALL entities in the text
2. Identify ALL relationships between entities
3. Format output EXACTLY as shown

-Entity Format-
("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<description>){RECORD_DELIMITER}

-Relationship Format-
("relationship"{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<description>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<strength>){RECORD_DELIMITER}

-Entity Types-
{types_str}

-Keywords-
2-5 action verbs or nouns describing the relationship

-Strength-
Float between 0.0 and 1.0 indicating relationship strength

-Text-
{text}

-Output-
"""


async def extract_single_chunk_with_retry(
    chunk: Dict,
    entity_types: List[str],
    llm_func,
    max_retries: int = 3
) -> Tuple[Dict, Dict]:
    """
    ✅ NEW: Extract with retry logic for failed calls
    """
    chunk_id = chunk.get("chunk_id", "")
    content = chunk.get("content", "")
    file_path = chunk.get("file_path", "unknown_source")
    
    prompt = create_extraction_prompt(content, entity_types)
    
    for attempt in range(max_retries):
        try:
            result = await llm_func(prompt)
            entities, relationships = await parse_extraction_result(
                result, chunk_id, file_path
            )
            return entities, relationships
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Extraction failed for chunk {chunk_id} after {max_retries} attempts: {e}")
                return {}, {}
            else:
                logger.warning(f"Extraction attempt {attempt+1} failed for chunk {chunk_id}: {e}, retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return {}, {}


async def extract_entities(
    chunks: List[Dict],
    global_config: Dict,
    max_concurrent: int = 16,  # ✅ INCREASED from 8
    progress_callback: Optional[callable] = None
) -> Tuple[Dict, Dict]:
    """
    ✅ OPTIMIZED: Extract entities with higher concurrency and progress tracking
    
    Args:
        chunks: List of chunk dicts
        global_config: Config with entity_types, llm_model_func
        max_concurrent: Max concurrent LLM calls (default 16)
        progress_callback: Optional callback(current, total)
    
    Returns:
        (entities_dict, relationships_dict)
    """
    entity_types = global_config.get("entity_types", [
        "PERSON", "ORGANIZATION", "LOCATION", "EVENT", 
        "PRODUCT", "CONCEPT", "TECHNOLOGY"
    ])
    
    llm_func = global_config.get("llm_model_func")
    if not llm_func:
        from backend.utils.llm_utils import call_llm_async
        llm_func = call_llm_async
    
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(chunks)
    
    async def process_with_semaphore(chunk):
        nonlocal completed
        async with semaphore:
            result = await extract_single_chunk_with_retry(
                chunk, entity_types, llm_func, max_retries=3
            )
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            return result
    
    logger.info(f"🔍 Extracting from {total} chunks (max_concurrent={max_concurrent})")
    
    tasks = [process_with_semaphore(c) for c in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_entities = defaultdict(list)
    all_relationships = defaultdict(list)
    
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed: {result}")
            continue
        
        entities, relationships = result
        
        for entity_name, entity_list in entities.items():
            all_entities[entity_name].extend(entity_list)
        
        for rel_key, rel_list in relationships.items():
            all_relationships[rel_key].extend(rel_list)
    
    total_ents = sum(len(v) for v in all_entities.values())
    total_rels = sum(len(v) for v in all_relationships.values())
    
    logger.info(f"✅ Extracted {total_ents} entities, {total_rels} relationships")
    
    return dict(all_entities), dict(all_relationships)


def extract_entities_relations(
    chunks: List[Dict],
    global_config: Optional[Dict] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[Dict, Dict]:
    """
    Sync wrapper for extraction with progress tracking
    
    Args:
        chunks: List of chunk dicts
        global_config: Configuration
        progress_callback: Optional callback(current, total)
    
    Returns:
        (entities_dict, relationships_dict)
    """
    if not global_config:
        global_config = {
            "entity_types": [
                "PERSON", "ORGANIZATION", "LOCATION", "EVENT",
                "PRODUCT", "CONCEPT", "TECHNOLOGY"
            ]
        }
    
    # ✅ OPTIMIZED: Higher concurrency
    max_concurrent = int(os.getenv('MAX_CONCURRENT_LLM_CALLS', 16))
    
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
            logger.warning("nest_asyncio not available")
    
    return loop.run_until_complete(
        extract_entities(chunks, global_config, max_concurrent, progress_callback)
    )