# ==========================================
# backend/core/extraction.py - FULL FIXED VERSION
# ==========================================
import asyncio
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"

def create_prompt(text: str) -> str:
    """Tạo prompt cho LLM với hướng dẫn rõ ràng về format"""
    return f"""Extract entities and relationships from this text.

IMPORTANT: Follow this EXACT format:

For entities:
("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<description>){RECORD_DELIMITER}

For relationships:
("relationship"{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<description>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<strength_0_to_1>){RECORD_DELIMITER}

RULES:
1. Entity types MUST be one of: PERSON, ORGANIZATION, LOCATION, EVENT, PRODUCT, CONCEPT, TECHNOLOGY
2. Strength MUST be a number between 0 and 1 (e.g., 0.8, 0.5, 1.0)
3. DO NOT use words like "strong", "weak", "HIGH" for strength - only numbers!
4. Use {TUPLE_DELIMITER} to separate fields
5. End each record with {RECORD_DELIMITER}

Example:
("entity"{TUPLE_DELIMITER}GPT-4{TUPLE_DELIMITER}TECHNOLOGY{TUPLE_DELIMITER}Advanced language model){RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}OpenAI{TUPLE_DELIMITER}GPT-4{TUPLE_DELIMITER}developed by{TUPLE_DELIMITER}AI, development{TUPLE_DELIMITER}0.9){RECORD_DELIMITER}

Text to extract from:
{text}

Output:"""

def safe_float_convert(value: str, default: float = 1.0) -> float:
    """✅ NEW: Safely convert string to float with fallback
    
    Handles cases where LLM returns words instead of numbers:
    - "strong" / "HIGH" -> 0.9
    - "medium" / "MEDIUM" -> 0.7
    - "weak" / "LOW" -> 0.5
    - Invalid -> default (1.0)
    """
    value = str(value).strip().strip('"\'').lower()
    
    # Try direct conversion first
    try:
        return float(value)
    except ValueError:
        pass
    
    # Map common words to scores
    strength_map = {
        'very strong': 0.95,
        'strong': 0.9,
        'high': 0.9,
        'medium high': 0.8,
        'medium': 0.7,
        'moderate': 0.7,
        'medium low': 0.6,
        'low': 0.5,
        'weak': 0.5,
        'very weak': 0.3
    }
    
    # Check if value matches any keyword
    for keyword, score in strength_map.items():
        if keyword in value:
            logger.debug(f"Mapped strength '{value}' to {score}")
            return score
    
    # Default fallback
    logger.warning(f"Could not parse strength '{value}', using default {default}")
    return default

def parse_result(result: str, chunk_id: str) -> Tuple[Dict, Dict]:
    """✅ FIXED: Parse LLM output with robust float conversion"""
    entities = defaultdict(list)
    relationships = defaultdict(list)
    
    records = re.split(f'{RECORD_DELIMITER}|<\\|COMPLETE\\|>', result)
    
    for record in records:
        match = re.search(r'\((.*?)\)', record)
        if not match:
            continue
        
        parts = match.group(1).split(TUPLE_DELIMITER)
        
        # Entity
        if len(parts) >= 4 and 'entity' in parts[0].lower():
            name = parts[1].strip().strip('"\'')
            if name:
                entities[name].append({
                    'entity_name': name,
                    'entity_type': parts[2].strip().strip('"\''),
                    'description': parts[3].strip().strip('"\''),
                    'source_id': chunk_id,
                    'chunk_id': chunk_id
                })
        
        # Relationship
        elif len(parts) >= 5 and 'relationship' in parts[0].lower():
            src = parts[1].strip().strip('"\'')
            tgt = parts[2].strip().strip('"\'')
            if src and tgt and src != tgt:
                # ✅ FIX: Use safe_float_convert for strength
                strength_str = parts[5].strip().strip('"\'') if len(parts) > 5 else '1.0'
                strength = safe_float_convert(strength_str, default=1.0)
                
                relationships[(src, tgt)].append({
                    'source_id': src,
                    'target_id': tgt,
                    'description': parts[3].strip().strip('"\''),
                    'keywords': parts[4].strip().strip('"\''),
                    'weight': strength,  # ✅ Now guaranteed to be float
                    'chunk_id': chunk_id
                })
    
    return dict(entities), dict(relationships)

async def extract_single(chunk: Dict, llm_func) -> Tuple[Dict, Dict]:
    """Extract từ 1 chunk"""
    try:
        prompt = create_prompt(chunk['content'])
        result = await llm_func(prompt)
        return parse_result(result, chunk['chunk_id'])
    except Exception as e:
        logger.error(f"Extract error for chunk {chunk.get('chunk_id')}: {e}")
        return {}, {}

async def extract_entities(chunks: List[Dict], llm_func, max_concurrent: int = 16) -> Tuple[Dict, Dict]:
    """Extract từ nhiều chunks song song"""
    sem = asyncio.Semaphore(max_concurrent)
    
    async def process(chunk):
        async with sem:
            return await extract_single(chunk, llm_func)
    
    results = await asyncio.gather(*[process(c) for c in chunks])
    
    all_entities = defaultdict(list)
    all_relationships = defaultdict(list)
    
    for entities, relationships in results:
        for k, v in entities.items():
            all_entities[k].extend(v)
        for k, v in relationships.items():
            all_relationships[k].extend(v)
    
    return dict(all_entities), dict(all_relationships)

def extract_entities_relations(chunks: List[Dict], global_config: Dict = None) -> Tuple[Dict, Dict]:
    """✅ FIXED: Sync wrapper with proper event loop handling"""
    if not chunks:
        logger.warning("⚠️ No chunks provided for extraction")
        return {}, {}
    
    llm_func = global_config.get('llm_model_func') if global_config else None
    if not llm_func:
        from backend.utils.llm_utils import call_llm_async
        llm_func = call_llm_async
    
    try:
        # ✅ FIX: Try to get existing event loop, create new if not exists
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            # No event loop in current thread, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        logger.info(f"🔍 Extracting entities from {len(chunks)} chunks...")
        
        # Run async extraction
        result = loop.run_until_complete(extract_entities(chunks, llm_func, 16))
        
        entities, relationships = result
        logger.info(
            f"✅ Extracted {sum(len(v) for v in entities.values())} entities, "
            f"{sum(len(v) for v in relationships.values())} relationships"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Extract entities error: {e}")
        return {}, {}