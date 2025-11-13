# backend/core/extraction.py

import asyncio
import logging
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict, defaultdict

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

GRAPH_FIELD_SEP = "; "  

def normalize_extracted_info(text: str, is_entity: bool = False) -> str:
    """
    Normalize extracted text following LightRAG rules
    
    Args:
        text: Text to normalize
        is_entity: True if normalizing entity name
    
    Returns:
        Normalized text
    """
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
    """
    Handle single entity extraction following LightRAG logic
    
    Args:
        record_attributes: ["entity", name, type, description]
        chunk_key: Chunk ID
        file_path: Source file path
    
    Returns:
        Entity dict or None if invalid
    """
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
    """
    Handle single relationship extraction following LightRAG logic
    
    Args:
        record_attributes: ["relationship", source, target, description, keywords?, strength?]
        chunk_key: Chunk ID
        file_path: Source file path
    
    Returns:
        Relationship dict or None if invalid
    """
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
    """
    Parse LLM extraction result into entities and relationships
    
    Args:
        result: LLM output text
        chunk_key: Chunk ID
        file_path: Source file
    
    Returns:
        (entities_dict, relationships_dict)
        - entities_dict: {entity_name: [entity_data]}
        - relationships_dict: {(source, target): [relation_data]}
    """
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
async def extract_single_chunk(
    chunk: Dict,
    entity_types: List[str],
    llm_func,
    cache_dict: Optional[Dict] = None
) -> Tuple[Dict, Dict]:
    """
    Extract entities and relationships from single chunk
    
    Args:
        chunk: Chunk data with 'chunk_id' and 'content'
        entity_types: List of entity types to extract
        llm_func: Async LLM function
        cache_dict: Optional cache dictionary
    
    Returns:
        (entities_dict, relationships_dict)
    """
    chunk_id = chunk.get("chunk_id", "")
    content = chunk.get("content", "")
    file_path = chunk.get("file_path", "unknown_source")
    
    cache_key = f"{chunk_id}_{hash(content)}"
    if cache_dict and cache_key in cache_dict:
        return cache_dict[cache_key]
    
    prompt = create_extraction_prompt(content, entity_types)
    
    try:
        result = await llm_func(prompt)
        entities, relationships = await parse_extraction_result(
            result, chunk_id, file_path
        )
        
        if cache_dict is not None:
            cache_dict[cache_key] = (entities, relationships)
        
        return entities, relationships
        
    except Exception as e:
        logger.error(f"Extraction failed for chunk {chunk_id}: {e}")
        return {}, {}


async def extract_entities(
    chunks: List[Dict],
    global_config: Dict,
    max_concurrent: int = 8,
) -> Tuple[Dict, Dict]:
    """
    Extract entities and relationships from all chunks with concurrency control
    
    Args:
        chunks: List of chunk dicts
        global_config: Config with entity_types, llm_model_func
        max_concurrent: Max concurrent LLM calls
    
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
    
    cache_dict = {}
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(chunk):
        async with semaphore:
            return await extract_single_chunk(chunk, entity_types, llm_func, cache_dict)
    
    logger.info(f"Extracting from {len(chunks)} chunks (max_concurrent={max_concurrent})")
    
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
    
    logger.info(f"Extracted {total_ents} entities, {total_rels} relationships")
    
    return dict(all_entities), dict(all_relationships)


def extract_entities_relations(
    chunks: List[Dict],
    global_config: Optional[Dict] = None
) -> Tuple[Dict, Dict]:
    """
    Sync wrapper for extraction
    
    Args:
        chunks: List of chunk dicts
        global_config: Configuration
    
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
    
    
    max_concurrent = int(os.getenv('MAX_CONCURRENT_LLM_CALLS', 8))
    
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
        extract_entities(chunks, global_config, max_concurrent)
    )