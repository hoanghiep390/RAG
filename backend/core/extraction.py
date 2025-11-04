import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional


TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
llm_response_cache = {}
logger = logging.getLogger(__name__)


def clean_llm_output(text: str) -> str:
    """
    Làm sạch output từ LLM:
    - Loại bỏ ký tự không cần thiết
    - Chuẩn hóa delimiter
    - Bỏ whitespace dư thừa
    """
    if not text:
        return ""
    
    cleaned = text.strip()
    
    # Đảm bảo delimiter thống nhất (phòng khi LLM xuống dòng hoặc thêm space)
    cleaned = cleaned.replace(" <|>", TUPLE_DELIMITER)
    cleaned = cleaned.replace("<|> ", TUPLE_DELIMITER)
    cleaned = cleaned.replace(" ## ", RECORD_DELIMITER)
    cleaned = cleaned.replace("## ", RECORD_DELIMITER)
    cleaned = cleaned.replace(" ##", RECORD_DELIMITER)
    cleaned = cleaned.replace(" <|COMPLETE|>", COMPLETION_DELIMITER)
    
    # Bỏ ký tự lạ hoặc BOM
    cleaned = cleaned.replace("\ufeff", "")
    
    return cleaned


# ========================
# LLM caller wrapper
# ========================
async def use_llm_func(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Hàm gọi LLM thực tế
    """
    try:
        from backend.utils.llm_utils import call_llm_async
        result = await call_llm_async(
            prompt=prompt,
            model=model,
            temperature=0.0,
            max_tokens=2000
        )
        return result
    except ImportError:
        # fallback giả lập để test offline
        logger.warning("⚠️ llm_utils not found — using mock extraction")
        await asyncio.sleep(0.1)
        return f"""entity{TUPLE_DELIMITER}Company A{TUPLE_DELIMITER}ORGANIZATION{TUPLE_DELIMITER}A technology company{RECORD_DELIMITER}
entity{TUPLE_DELIMITER}Product X{TUPLE_DELIMITER}PRODUCT{TUPLE_DELIMITER}Main product of Company A{RECORD_DELIMITER}
relationship{TUPLE_DELIMITER}Company A{TUPLE_DELIMITER}Product X{TUPLE_DELIMITER}PRODUCES{TUPLE_DELIMITER}0.9{COMPLETION_DELIMITER}"""


# ========================
# Prompt & Parser
# ========================
def create_extraction_prompt(chunk_text: str, entity_types: List[str]) -> str:
    """Tạo prompt cho LLM để trích xuất entities và relationships"""
    entity_types_str = ", ".join(entity_types)
    
    return f"""
-Goal-
Given a text document, identify all entities and relationships from the text.

-Steps-
1. Identify all entities. For each identified entity, extract:
   ("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<entity_description>)

2. Identify relationships between entities:
   ("relationship"{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<relationship_description>{TUPLE_DELIMITER}<relationship_strength>)

3. Output format:
   (<record>{RECORD_DELIMITER}<record>{RECORD_DELIMITER}...{COMPLETION_DELIMITER})

-Real Data-
Text: {chunk_text}

Output:
"""


def parse_extraction_result(result: str) -> List[Dict[str, Any]]:
    """Parse kết quả extraction từ LLM thành list of records"""
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
        
        if record_type == "entity" and len(parts) >= 4:
            records.append({
                "type": "entity",
                "entity_name": parts[1],
                "entity_type": parts[2],
                "description": parts[3],
            })
        elif record_type == "relationship" and len(parts) >= 5:
            try:
                strength = float(parts[4])
            except ValueError:
                strength = 1.0
            records.append({
                "type": "relationship",
                "source_entity": parts[1],
                "target_entity": parts[2],
                "description": parts[3],
                "strength": strength,
            })
    
    return records


# ========================
# Entity / Relationship Handlers
# ========================
def handle_single_entity_extraction(record: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    """Xử lý một entity record"""
    return {
        "entity_name": record["entity_name"],
        "entity_type": record["entity_type"],
        "description": record["description"],
        "chunk_id": chunk_id,
        "source_id": chunk_id
    }


def handle_single_relationship_extraction(record: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
    """Xử lý một relationship record"""
    return {
        "source_id": record["source_entity"],
        "target_id": record["target_entity"],
        "description": record["description"],
        "strength": record.get("strength", 1.0),
        "chunk_id": chunk_id
    }


def process_extraction_result(records: List[Dict[str, Any]], chunk_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Phân loại và xử lý các records thành entities và relationships"""
    entities, relationships = [], []
    
    for r in records:
        if r["type"] == "entity":
            entities.append(handle_single_entity_extraction(r, chunk_id))
        elif r["type"] == "relationship":
            relationships.append(handle_single_relationship_extraction(r, chunk_id))
    
    return entities, relationships


# ========================
# Async Extraction Flow
# ========================
async def extract_single_chunk(
    chunk: Dict[str, Any],
    entity_types: List[str],
    use_cache: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Trích xuất entities và relationships từ một chunk, có error handling
    """
    chunk_id = chunk.get("chunk_id", "")
    chunk_text = chunk.get("content", "")
    cache_key = f"{chunk_id}_{hash(chunk_text)}"
    
    try:
        # cache
        if use_cache and cache_key in llm_response_cache:
            result = llm_response_cache[cache_key]
        else:
            prompt = create_extraction_prompt(chunk_text, entity_types)
            result = await use_llm_func(prompt)
            if use_cache:
                llm_response_cache[cache_key] = result
        
        # parse + process
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
    """
    Trích xuất entities và relationships từ nhiều chunks song song
    """
    entity_types = global_config.get("entity_types", [
        "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT"
    ])
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(chunk):
        async with semaphore:
            return await extract_single_chunk(chunk, entity_types, use_cache)
    
    tasks = [process_with_semaphore(c) for c in chunks]
    results = await asyncio.gather(*tasks)
    
    maybe_nodes, maybe_edges = {}, {}
    
    for i, (ents, rels) in enumerate(results):
        chunk_id = chunks[i].get("chunk_id", f"chunk_{i}")
        if ents:
            maybe_nodes[chunk_id] = ents
        if rels:
            maybe_edges[chunk_id] = rels
    
    return maybe_nodes, maybe_edges


def extract_entities_relations(
    chunks: List[Dict[str, Any]],
    global_config: Optional[Dict[str, Any]] = None
) -> Tuple[Dict, Dict]:
    """
    Entry point đồng bộ cho extraction
    """
    if global_config is None:
        global_config = {
            "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "CONCEPT"]
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
            logger.warning("⚠️ nest_asyncio not installed — may cause loop conflict")
    
    nodes, edges = loop.run_until_complete(extract_entities(chunks, global_config))
    
    return nodes, edges