# backend/core/extraction.py - OPTIMIZED VERSION
"""
✅ OPTIMIZED: Extraction with batch processing and better caching
"""

import asyncio
import logging
import os
from typing import Dict, List, Tuple, Any, Optional

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"

# ✅ OPTIMIZED: Better cache with size limit
from collections import OrderedDict

class LRUCache:
    """Simple LRU cache for LLM responses"""
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

llm_response_cache = LRUCache(maxsize=int(os.getenv('CACHE_MAX_SIZE', 1000)))
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


# ==================== BATCH PROMPT ====================
def create_batch_extraction_prompt(chunks: List[Dict], entity_types: List[str]) -> str:
    """
    ✅ OPTIMIZED: Create prompt for multiple chunks at once
    Reduces LLM API calls by processing chunks in batches
    """
    entity_types_str = ", ".join(entity_types)
    
    # Combine multiple chunks with separator
    chunks_text = "\n\n---CHUNK_SEPARATOR---\n\n".join([
        f"CHUNK_ID: {c['chunk_id']}\n{c['content']}"
        for c in chunks
    ])
    
    return f"""
-Goal-
Extract entities and relationships from MULTIPLE text chunks. 
For each entity/relationship, include the CHUNK_ID it comes from.

-Steps-
1. For each chunk, identify entities:
   Format: ("entity"{TUPLE_DELIMITER}<chunk_id>{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<description>)
   
   Entity types: {entity_types_str}

2. For each chunk, identify relationships:
   Format: ("relationship"{TUPLE_DELIMITER}<chunk_id>{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<description>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<strength>)
   
   - Keywords: 2-5 key verbs/nouns (comma-separated)
   - Strength: 0.0-1.0

3. Output format:
   (<record>{RECORD_DELIMITER}<record>{RECORD_DELIMITER}...{COMPLETION_DELIMITER})

-Example-
("entity"{TUPLE_DELIMITER}chunk_123{TUPLE_DELIMITER}Apple Inc.{TUPLE_DELIMITER}ORGANIZATION{TUPLE_DELIMITER}Tech company)
("relationship"{TUPLE_DELIMITER}chunk_123{TUPLE_DELIMITER}Apple Inc.{TUPLE_DELIMITER}iPhone{TUPLE_DELIMITER}manufactures{TUPLE_DELIMITER}manufactures, produces{TUPLE_DELIMITER}0.95)

-Text Chunks-
{chunks_text}

Output:
"""


def create_extraction_prompt(chunk_text: str, entity_types: List[str]) -> str:
    """
    Standard single-chunk extraction prompt (fallback)
    """
    entity_types_str = ", ".join(entity_types)
    
    return f"""
-Goal-
Extract entities and relationships from the text.

-Steps-
1. Identify entities:
   Format: ("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<description>)
   Types: {entity_types_str}

2. Identify relationships:
   Format: ("relationship"{TUPLE_DELIMITER}<source>{TUPLE_DELIMITER}<target>{TUPLE_DELIMITER}<description>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<strength>)

-Text-
{chunk_text}

Output:
"""


# ==================== PARSER ====================
def parse_batch_extraction_result(result: str, chunks: List[Dict]) -> Tuple[Dict, Dict]:
    """
    ✅ OPTIMIZED: Parse batch extraction results
    Returns entities_dict and relationships_dict organized by chunk_id
    """
    records = parse_extraction_result(result)
    
    entities_dict = {}
    relationships_dict = {}
    
    for record in records:
        if record['type'] == 'entity':
            chunk_id = record.get('chunk_id', chunks[0]['chunk_id'])
            if chunk_id not in entities_dict:
                entities_dict[chunk_id] = []
            entities_dict[chunk_id].append({
                'entity_name': record['entity_name'],
                'entity_type': record['entity_type'],
                'description': record['description'],
                'chunk_id': chunk_id,
                'source_id': chunk_id
            })
        
        elif record['type'] == 'relationship':
            chunk_id = record.get('chunk_id', chunks[0]['chunk_id'])
            if chunk_id not in relationships_dict:
                relationships_dict[chunk_id] = []
            relationships_dict[chunk_id].append({
                'source_id': record['source_entity'],
                'target_id': record['target_entity'],
                'description': record['description'],
                'keywords': record.get('keywords', ''),
                'strength': record.get('strength', 1.0),
                'chunk_id': chunk_id
            })
    
    return entities_dict, relationships_dict


def parse_extraction_result(result: str) -> List[Dict[str, Any]]:
    """Parse extraction result (supports both single and batch formats)"""
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
        
        record_type = parts[0].lower().replace('(', '').replace('"', '')
        
        # ENTITY with chunk_id: ("entity", chunk_id, name, type, desc)
        if record_type == "entity":
            if len(parts) >= 5:
                # Batch format with chunk_id
                records.append({
                    "type": "entity",
                    "chunk_id": parts[1],
                    "entity_name": parts[2],
                    "entity_type": parts[3],
                    "description": parts[4],
                })
            elif len(parts) >= 4:
                # Single format without chunk_id
                records.append({
                    "type": "entity",
                    "entity_name": parts[1],
                    "entity_type": parts[2],
                    "description": parts[3],
                })
        
        # RELATIONSHIP with chunk_id
        elif record_type == "relationship":
            if len(parts) >= 7:
                # Batch format: chunk_id, source, target, desc, keywords, strength
                try:
                    strength = float(parts[6])
                except ValueError:
                    strength = 1.0
                
                records.append({
                    "type": "relationship",
                    "chunk_id": parts[1],
                    "source_entity": parts[2],
                    "target_entity": parts[3],
                    "description": parts[4],
                    "keywords": parts[5],
                    "strength": strength,
                })
            elif len(parts) >= 6:
                # Single format with keywords
                try:
                    strength = float(parts[5])
                except ValueError:
                    strength = 1.0
                
                keywords = parts[4] if len(parts) > 4 else extract_keywords_from_description(parts[3])
                
                records.append({
                    "type": "relationship",
                    "source_entity": parts[1],
                    "target_entity": parts[2],
                    "description": parts[3],
                    "keywords": keywords,
                    "strength": strength,
                })
    
    return records


def extract_keywords_from_description(description: str) -> str:
    """Extract keywords from description (fallback)"""
    verbs = [
        "produces", "creates", "develops", "founded", "located", "works",
        "manages", "owns", "uses", "builds", "sells", "provides", "offers",
        "leads", "operates", "manufactures", "designs", "invented"
    ]
    
    desc_lower = description.lower()
    found_keywords = [v for v in verbs if v in desc_lower]
    
    if found_keywords:
        return ", ".join(found_keywords[:3])
    
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
    """Handle relationship extraction"""
    return {
        "source_id": record["source_entity"],
        "target_id": record["target_entity"],
        "description": record["description"],
        "keywords": record.get("keywords", ""),
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
        if use_cache:
            cached = llm_response_cache.get(cache_key)
            if cached is not None:
                result = cached
                logger.debug(f"✅ Cache hit: {chunk_id}")
            else:
                prompt = create_extraction_prompt(chunk_text, entity_types)
                from backend.utils.llm_utils import call_llm_with_retry
                result = await call_llm_with_retry(prompt, model=model, max_retries=3)
                llm_response_cache.set(cache_key, result)
        else:
            prompt = create_extraction_prompt(chunk_text, entity_types)
            from backend.utils.llm_utils import call_llm_with_retry
            result = await call_llm_with_retry(prompt, model=model, max_retries=3)
        
        records = parse_extraction_result(result)
        entities, relationships = process_extraction_result(records, chunk_id)
        
        return entities, relationships
    
    except asyncio.TimeoutError:
        logger.error(f"⏰ Timeout: {chunk_id}")
        return [], []
    except Exception as e:
        logger.error(f"❌ Error: {chunk_id}: {e}")
        return [], []


async def extract_batch(
    chunks: List[Dict[str, Any]],
    entity_types: List[str],
    model: str,
    use_cache: bool = True
) -> Tuple[Dict, Dict]:
    """
    ✅ OPTIMIZED: Extract from batch of chunks with one LLM call
    """
    # Create cache key from all chunks
    cache_key = f"batch_{'_'.join([c['chunk_id'] for c in chunks])}"
    
    try:
        if use_cache:
            cached = llm_response_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"✅ Cache hit for batch of {len(chunks)} chunks")
                return cached
        
        # Create batch prompt
        prompt = create_batch_extraction_prompt(chunks, entity_types)
        
        # Call LLM once for all chunks
        from backend.utils.llm_utils import call_llm_with_retry
        result = await call_llm_with_retry(prompt, model=model, max_retries=3)
        
        # Parse batch results
        entities_dict, relationships_dict = parse_batch_extraction_result(result, chunks)
        
        # Cache result
        if use_cache:
            llm_response_cache.set(cache_key, (entities_dict, relationships_dict))
        
        return entities_dict, relationships_dict
    
    except Exception as e:
        logger.error(f"❌ Batch extraction error: {e}")
        return {}, {}


async def extract_entities(
    chunks: List[Dict[str, Any]],
    global_config: Dict[str, Any],
    max_concurrent: int = 5,
    batch_size: int = 10,
    use_cache: bool = True
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    ✅ OPTIMIZED: Extract with batch processing
    
    Args:
        chunks: List of chunks
        global_config: Configuration
        max_concurrent: Max concurrent API calls
        batch_size: Chunks per batch (1 LLM call per batch)
        use_cache: Use caching
    
    Returns:
        (entities_dict, relationships_dict)
    """
    entity_types = global_config.get("entity_types", [
        "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "CONCEPT", "TECHNOLOGY"
    ])
    
    model = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
    
    # ✅ OPTIMIZATION: Group chunks into batches
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    logger.info(f"🔄 Processing {len(chunks)} chunks in {len(batches)} batches (batch_size={batch_size})")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch_with_semaphore(batch):
        async with semaphore:
            return await extract_batch(batch, entity_types, model, use_cache)
    
    # Process all batches
    tasks = [process_batch_with_semaphore(batch) for batch in batches]
    results = await asyncio.gather(*tasks)
    
    # Merge results from all batches
    entities_dict, relationships_dict = {}, {}
    
    for batch_ents, batch_rels in results:
        entities_dict.update(batch_ents)
        relationships_dict.update(batch_rels)
    
    total_entities = sum(len(v) for v in entities_dict.values())
    total_rels = sum(len(v) for v in relationships_dict.values())
    
    logger.info(f"✅ Extracted {total_entities} entities, {total_rels} relationships")
    
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
    
    # Get batch size from env
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
            logger.warning("⚠️ nest_asyncio not installed")
    
    nodes, edges = loop.run_until_complete(
        extract_entities(chunks, global_config, max_concurrent, batch_size)
    )
    
    return nodes, edges


# ==================== KEYWORD ANALYSIS ====================
def analyze_relationship_keywords(relationships_dict: Dict[str, List[Dict]]) -> Dict[str, int]:
    """Analyze most common keywords"""
    from collections import Counter
    
    all_keywords = []
    for rels in relationships_dict.values():
        for rel in rels:
            keywords = rel.get('keywords', '')
            if keywords:
                all_keywords.extend([k.strip() for k in keywords.split(',')])
    
    return dict(Counter(all_keywords).most_common(20))