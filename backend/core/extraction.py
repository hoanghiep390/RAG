# backend/core/extraction.py - SIMPLIFIED & OPTIMIZED
"""
✅ Simplified extraction with batch processing
"""
import asyncio
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

logger = logging.getLogger(__name__)


# ==================== LRU CACHE ====================
class LRUCache:
    """Simple LRU cache"""
    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

llm_cache = LRUCache(maxsize=int(os.getenv('CACHE_MAX_SIZE', 1000)))


# ==================== UTILS ====================
def clean_llm_output(text: str) -> str:
    """Clean LLM output"""
    if not text:
        return ""
    
    cleaned = text.strip()
    for old, new in [
        (" <|>", TUPLE_DELIMITER), ("<|> ", TUPLE_DELIMITER),
        (" ## ", RECORD_DELIMITER), ("## ", RECORD_DELIMITER), (" ##", RECORD_DELIMITER),
        (" <|COMPLETE|>", COMPLETION_DELIMITER), ("\ufeff", "")
    ]:
        cleaned = cleaned.replace(old, new)
    
    return cleaned


def extract_keywords(description: str) -> str:
    """Extract keywords from description"""
    verbs = ["produces", "creates", "develops", "founded", "located", "works",
             "manages", "owns", "uses", "builds", "sells", "provides", "offers",
             "leads", "operates", "manufactures", "designs", "invented"]
    
    desc_lower = description.lower()
    found = [v for v in verbs if v in desc_lower]
    
    return ", ".join(found[:3]) if found else ", ".join(description.split()[:3])


# ==================== PROMPTS ====================
def create_batch_prompt(chunks: List[Dict], entity_types: List[str]) -> str:
    """Create prompt for multiple chunks"""
    types_str = ", ".join(entity_types)
    chunks_text = "\n\n---CHUNK_SEPARATOR---\n\n".join([
        f"CHUNK_ID: {c['chunk_id']}\n{c['content']}" for c in chunks
    ])
    
    return f"""
-Goal-
Extract entities and relationships from MULTIPLE chunks. Include CHUNK_ID for each.

-Format-
("entity"{TUPLE_DELIMITER}<chunk_id>{TUPLE_DELIMITER}<name>{TUPLE_DELIMITER}<type>{TUPLE_DELIMITER}<desc>)
("relationship"{TUPLE_DELIMITER}<chunk_id>{TUPLE_DELIMITER}<source>{TUPLE_DELIMITER}<target>{TUPLE_DELIMITER}<desc>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<strength>)

Entity Types: {types_str}
Keywords: 2-5 verbs/nouns
Strength: 0.0-1.0

-Chunks-
{chunks_text}

Output:
"""


def create_single_prompt(text: str, entity_types: List[str]) -> str:
    """Create prompt for single chunk"""
    types_str = ", ".join(entity_types)
    
    return f"""
-Goal-
Extract entities and relationships.

-Format-
("entity"{TUPLE_DELIMITER}<name>{TUPLE_DELIMITER}<type>{TUPLE_DELIMITER}<desc>)
("relationship"{TUPLE_DELIMITER}<source>{TUPLE_DELIMITER}<target>{TUPLE_DELIMITER}<desc>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<strength>)

Types: {types_str}

-Text-
{text}

Output:
"""


# ==================== PARSER ====================
def parse_extraction(result: str) -> List[Dict[str, Any]]:
    """Parse extraction result"""
    records = []
    result = clean_llm_output(result).replace(COMPLETION_DELIMITER, "").strip()
    
    for raw in result.split(RECORD_DELIMITER):
        raw = raw.strip()
        if not raw:
            continue
        
        parts = [p.strip() for p in raw.split(TUPLE_DELIMITER)]
        if len(parts) < 2:
            continue
        
        record_type = parts[0].lower().replace('(', '').replace('"', '')
        
        # ENTITY: ("entity", chunk_id?, name, type, desc)
        if record_type == "entity":
            if len(parts) >= 5:  # Batch format
                records.append({
                    "type": "entity",
                    "chunk_id": parts[1],
                    "entity_name": parts[2],
                    "entity_type": parts[3],
                    "description": parts[4],
                })
            elif len(parts) >= 4:  # Single format
                records.append({
                    "type": "entity",
                    "entity_name": parts[1],
                    "entity_type": parts[2],
                    "description": parts[3],
                })
        
        # RELATIONSHIP: ("relationship", chunk_id?, source, target, desc, keywords?, strength?)
        elif record_type == "relationship":
            try:
                if len(parts) >= 7:  # Batch with keywords
                    records.append({
                        "type": "relationship",
                        "chunk_id": parts[1],
                        "source_entity": parts[2],
                        "target_entity": parts[3],
                        "description": parts[4],
                        "keywords": parts[5],
                        "strength": float(parts[6]),
                    })
                elif len(parts) >= 6:  # Single with keywords
                    records.append({
                        "type": "relationship",
                        "source_entity": parts[1],
                        "target_entity": parts[2],
                        "description": parts[3],
                        "keywords": parts[4],
                        "strength": float(parts[5]),
                    })
                elif len(parts) >= 5:  # Without keywords
                    records.append({
                        "type": "relationship",
                        "source_entity": parts[1] if len(parts) == 6 else parts[2],
                        "target_entity": parts[2] if len(parts) == 6 else parts[3],
                        "description": parts[3] if len(parts) == 6 else parts[4],
                        "keywords": extract_keywords(parts[3] if len(parts) == 6 else parts[4]),
                        "strength": float(parts[4]) if len(parts) == 6 else 1.0,
                    })
            except (ValueError, IndexError):
                continue
    
    return records


def parse_batch(result: str, chunks: List[Dict]) -> Tuple[Dict, Dict]:
    """Parse batch result into dicts by chunk_id"""
    records = parse_extraction(result)
    entities_dict = {}
    rels_dict = {}
    
    for rec in records:
        chunk_id = rec.get('chunk_id', chunks[0]['chunk_id'])
        
        if rec['type'] == 'entity':
            if chunk_id not in entities_dict:
                entities_dict[chunk_id] = []
            entities_dict[chunk_id].append({
                'entity_name': rec['entity_name'],
                'entity_type': rec['entity_type'],
                'description': rec['description'],
                'chunk_id': chunk_id,
                'source_id': chunk_id
            })
        
        elif rec['type'] == 'relationship':
            if chunk_id not in rels_dict:
                rels_dict[chunk_id] = []
            rels_dict[chunk_id].append({
                'source_id': rec['source_entity'],
                'target_id': rec['target_entity'],
                'description': rec['description'],
                'keywords': rec.get('keywords', ''),
                'strength': rec.get('strength', 1.0),
                'chunk_id': chunk_id
            })
    
    return entities_dict, rels_dict


def process_result(records: List[Dict], chunk_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Process single chunk result"""
    entities = []
    rels = []
    
    for rec in records:
        if rec['type'] == 'entity':
            entities.append({
                'entity_name': rec['entity_name'],
                'entity_type': rec['entity_type'],
                'description': rec['description'],
                'chunk_id': chunk_id,
                'source_id': chunk_id
            })
        elif rec['type'] == 'relationship':
            rels.append({
                'source_id': rec['source_entity'],
                'target_id': rec['target_entity'],
                'description': rec['description'],
                'keywords': rec.get('keywords', ''),
                'strength': rec.get('strength', 1.0),
                'chunk_id': chunk_id
            })
    
    return entities, rels


# ==================== ASYNC EXTRACTION ====================
async def extract_single(chunk: Dict, entity_types: List[str], model: str, use_cache: bool = True) -> Tuple[List, List]:
    """Extract from single chunk"""
    chunk_id = chunk.get("chunk_id", "")
    text = chunk.get("content", "")
    cache_key = f"{chunk_id}_{hash(text)}"
    
    try:
        if use_cache:
            cached = llm_cache.get(cache_key)
            if cached:
                result = cached
                logger.debug(f"✅ Cache: {chunk_id}")
            else:
                from backend.utils.llm_utils import call_llm_with_retry
                prompt = create_single_prompt(text, entity_types)
                result = await call_llm_with_retry(prompt, model=model, max_retries=3)
                llm_cache.set(cache_key, result)
        else:
            from backend.utils.llm_utils import call_llm_with_retry
            prompt = create_single_prompt(text, entity_types)
            result = await call_llm_with_retry(prompt, model=model, max_retries=3)
        
        records = parse_extraction(result)
        return process_result(records, chunk_id)
    
    except Exception as e:
        logger.error(f"❌ {chunk_id}: {e}")
        return [], []


async def extract_batch(chunks: List[Dict], entity_types: List[str], model: str, use_cache: bool = True) -> Tuple[Dict, Dict]:
    """Extract from batch of chunks"""
    cache_key = f"batch_{'_'.join([c['chunk_id'] for c in chunks])}"
    
    try:
        if use_cache:
            cached = llm_cache.get(cache_key)
            if cached:
                logger.debug(f"✅ Cache: batch of {len(chunks)}")
                return cached
        
        from backend.utils.llm_utils import call_llm_with_retry
        prompt = create_batch_prompt(chunks, entity_types)
        result = await call_llm_with_retry(prompt, model=model, max_retries=3)
        
        entities_dict, rels_dict = parse_batch(result, chunks)
        
        if use_cache:
            llm_cache.set(cache_key, (entities_dict, rels_dict))
        
        return entities_dict, rels_dict
    
    except Exception as e:
        logger.error(f"❌ Batch: {e}")
        return {}, {}


async def extract_entities(chunks: List[Dict], global_config: Dict, 
                          max_concurrent: int = 5, batch_size: int = 10, 
                          use_cache: bool = True) -> Tuple[Dict, Dict]:
    """Extract with batch processing"""
    entity_types = global_config.get("entity_types", [
        "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "CONCEPT", "TECHNOLOGY"
    ])
    
    model = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
    
    # Group into batches
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    logger.info(f"🔄 Processing {len(chunks)} chunks in {len(batches)} batches (batch={batch_size})")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_sem(batch):
        async with semaphore:
            return await extract_batch(batch, entity_types, model, use_cache)
    
    tasks = [process_with_sem(b) for b in batches]
    results = await asyncio.gather(*tasks)
    
    # Merge results
    entities_dict = {}
    rels_dict = {}
    
    for ents, rels in results:
        entities_dict.update(ents)
        rels_dict.update(rels)
    
    total_ents = sum(len(v) for v in entities_dict.values())
    total_rels = sum(len(v) for v in rels_dict.values())
    
    logger.info(f"✅ Extracted {total_ents} entities, {total_rels} relationships")
    
    return entities_dict, rels_dict


# ==================== SYNC ENTRY POINT ====================
def extract_entities_relations(chunks: List[Dict], global_config: Optional[Dict] = None) -> Tuple[Dict, Dict]:
    """Sync entry point"""
    if not global_config:
        global_config = {
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "CONCEPT", "TECHNOLOGY"]
        }
    
    batch_size = int(os.getenv('EXTRACTION_BATCH_SIZE', 10))
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
            logger.warning("⚠️ nest_asyncio not available")
    
    return loop.run_until_complete(extract_entities(chunks, global_config, max_concurrent, batch_size))


# ==================== UTILITIES ====================
def analyze_keywords(relationships_dict: Dict[str, List[Dict]]) -> Dict[str, int]:
    """Analyze common keywords"""
    from collections import Counter
    
    keywords = []
    for rels in relationships_dict.values():
        for rel in rels:
            kw = rel.get('keywords', '')
            if kw:
                keywords.extend([k.strip() for k in kw.split(',')])
    
    return dict(Counter(keywords).most_common(20))