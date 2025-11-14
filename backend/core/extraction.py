# ==========================================
# backend/core/extraction.py
# ==========================================
import asyncio
import re
from typing import Dict, List, Tuple
from collections import defaultdict

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"

def create_prompt(text: str) -> str:
    """Tạo prompt cho LLM"""
    return f"""Extract entities and relationships from this text.

Format:
("entity"{TUPLE_DELIMITER}<name>{TUPLE_DELIMITER}<type>{TUPLE_DELIMITER}<description>){RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}<source>{TUPLE_DELIMITER}<target>{TUPLE_DELIMITER}<description>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<strength>){RECORD_DELIMITER}

Types: PERSON, ORGANIZATION, LOCATION, EVENT, PRODUCT, CONCEPT, TECHNOLOGY

Text:
{text}

Output:"""

def parse_result(result: str, chunk_id: str) -> Tuple[Dict, Dict]:
    """Parse LLM output"""
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
                relationships[(src, tgt)].append({
                    'source_id': src,
                    'target_id': tgt,
                    'description': parts[3].strip().strip('"\''),
                    'keywords': parts[4].strip().strip('"\''),
                    'weight': float(parts[5].strip().strip('"\'')) if len(parts) > 5 else 1.0,
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
        print(f"Extract error: {e}")
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
    """Sync wrapper"""
    llm_func = global_config.get('llm_model_func') if global_config else None
    if not llm_func:
        from backend.utils.llm_utils import call_llm_async
        llm_func = call_llm_async
    
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(extract_entities(chunks, llm_func, 16))