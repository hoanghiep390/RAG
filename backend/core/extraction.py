# backend/core/extraction.py 
"""
Trích xuất Entity & Relationship 
Trích xuất đơn giản một giai đoạn
"""

import asyncio
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"

#  Entity Types 

ENTITY_TYPES = [
    'person',
    'organization', 
    'location',
    'event',
    'product',
    'concept',
    'technology',
    'date',
    'metric',
    'equipment',
    'category',
    'other'
]

#  Helper Functions

def sanitize_text(text: str) -> str:
    """Làm sạch và chuẩn hóa văn bản đã trích xuất"""
    if not text:
        return ""
    
    # Xóa dấu ngoặc kép
    text = text.strip().strip('"').strip("'")
    
    # Xóa ký tự đặc biệt không nên có trong tên entity
    text = re.sub(r'[<>|/\\]', '', text)
    
    # Chuẩn hóa khoảng trắng
    text = ' '.join(text.split())
    
    return text.strip()


# ================= Prompt Creation =================

def create_extraction_prompt(text: str, entity_types: List[str] = None) -> str:
    """
    Tạo prompt trích xuất kiểu LightRAG
    
    Tham số:
        text: Văn bản cần trích xuất
        entity_types: Danh sách loại entity (mặc định: ENTITY_TYPES)
    
    Trả về:
        Chuỗi prompt đã định dạng
    """
    if entity_types is None:
        entity_types = ENTITY_TYPES
    
    entity_types_str = ', '.join(entity_types)
    
    prompt = f"""---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1. **Entity Extraction & Output:**
   - **Identification:** Identify clearly defined and meaningful entities in the input text.
   - **Entity Details:** For each identified entity, extract the following information:
       - `entity_name`: The name of the entity. Use title case for proper nouns. Ensure **consistent naming** across the entire extraction process.
       - `entity_type`: Categorize the entity using one of the following types: {entity_types_str}. If none apply, use `other`.
       - `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
       - `confidence` (optional): A score from 1-5 indicating confidence in this entity (5 = very confident, 1 = uncertain). If not provided, defaults to 3.
   - **Output Format - Entities:** Output 4-5 fields for each entity, delimited by `{TUPLE_DELIMITER}`, on a single line. The first field *must* be the literal string `entity`.
       - Format: `entity{TUPLE_DELIMITER}entity_name{TUPLE_DELIMITER}entity_type{TUPLE_DELIMITER}entity_description{TUPLE_DELIMITER}confidence` (confidence is optional)

2. **Relationship Extraction & Output:**
   - **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
   - **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities, decompose it into multiple binary (two-entity) relationship pairs.
       - **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X."
   - **Relationship Details:** For each binary relationship, extract the following fields:
       - `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction.
       - `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction.
       - `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{TUPLE_DELIMITER}` for separating multiple keywords within this field.**
       - `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
   - **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{TUPLE_DELIMITER}`, on a single line. The first field *must* be the literal string `relation`.
       - Format: `relation{TUPLE_DELIMITER}source_entity{TUPLE_DELIMITER}target_entity{TUPLE_DELIMITER}relationship_keywords{TUPLE_DELIMITER}relationship_description`

3. **Delimiter Usage Protocol:**
   - The `{TUPLE_DELIMITER}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
   - **Incorrect Example:** `entity{TUPLE_DELIMITER}Tokyo<|location|>Tokyo is the capital of Japan.`
   - **Correct Example:** `entity{TUPLE_DELIMITER}Tokyo{TUPLE_DELIMITER}location{TUPLE_DELIMITER}Tokyo is the capital of Japan.`

4. **Relationship Direction & Duplication:**
   - Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
   - Avoid outputting duplicate relationships.

5. **Output Order & Prioritization:**
   - Output all extracted entities first, followed by all extracted relationships.
   - Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6. **Context & Objectivity:**
   - Ensure all entity names and descriptions are written in the **third person**.
   - Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7. **Completion Signal:** Output the literal string `{COMPLETION_DELIMITER}` only after all entities and relationships have been completely extracted and outputted.

---Examples---

<Input Text>
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

<Output>
entity{TUPLE_DELIMITER}World Athletics Championship{TUPLE_DELIMITER}event{TUPLE_DELIMITER}The World Athletics Championship is a global sports competition featuring top athletes in track and field.
entity{TUPLE_DELIMITER}Tokyo{TUPLE_DELIMITER}location{TUPLE_DELIMITER}Tokyo is the host city of the World Athletics Championship.
entity{TUPLE_DELIMITER}Noah Carter{TUPLE_DELIMITER}person{TUPLE_DELIMITER}Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship.
entity{TUPLE_DELIMITER}100m Sprint Record{TUPLE_DELIMITER}metric{TUPLE_DELIMITER}The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter.
entity{TUPLE_DELIMITER}Carbon-Fiber Spikes{TUPLE_DELIMITER}equipment{TUPLE_DELIMITER}Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction.
relation{TUPLE_DELIMITER}World Athletics Championship{TUPLE_DELIMITER}Tokyo{TUPLE_DELIMITER}event location, international competition{TUPLE_DELIMITER}The World Athletics Championship is being hosted in Tokyo.
relation{TUPLE_DELIMITER}Noah Carter{TUPLE_DELIMITER}100m Sprint Record{TUPLE_DELIMITER}athlete achievement, record-breaking{TUPLE_DELIMITER}Noah Carter set a new 100m sprint record at the championship.
relation{TUPLE_DELIMITER}Noah Carter{TUPLE_DELIMITER}Carbon-Fiber Spikes{TUPLE_DELIMITER}athletic equipment, performance boost{TUPLE_DELIMITER}Noah Carter used carbon-fiber spikes to enhance performance during the race.
relation{TUPLE_DELIMITER}Noah Carter{TUPLE_DELIMITER}World Athletics Championship{TUPLE_DELIMITER}athlete participation, competition{TUPLE_DELIMITER}Noah Carter is competing at the World Athletics Championship.
{COMPLETION_DELIMITER}

---Real Data to be Processed---

<Input Text>
```
{text}
```

<Output>
"""
    
    return prompt


def create_continue_extraction_prompt(text: str, previous_result: str) -> str:
    """
    Tạo prompt tiếp tục trích xuất cho gleaning (kiểu LightRAG)
    
    Tham số:
        text: Văn bản gốc
        previous_result: Kết quả trích xuất trước đó
    
    Trả về:
        Prompt tiếp tục trích xuất
    """
    prompt = f"""---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1. **Strict Adherence to Format:** Follow the same format as before with `{TUPLE_DELIMITER}` delimiter.
2. **Focus on Corrections/Additions:**
   - **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
   - If an entity or relationship was **missed**, extract and output it now.
   - If an entity or relationship was **truncated or incorrectly formatted**, re-output the corrected version.
3. **Output Format - Entities:** `entity{TUPLE_DELIMITER}entity_name{TUPLE_DELIMITER}entity_type{TUPLE_DELIMITER}entity_description`
4. **Output Format - Relationships:** `relation{TUPLE_DELIMITER}source{TUPLE_DELIMITER}target{TUPLE_DELIMITER}keywords{TUPLE_DELIMITER}description`
5. **Completion Signal:** Output `{COMPLETION_DELIMITER}` when done.

---Previous Extraction---
```
{previous_result}
```

---Original Text---
```
{text}
```

<Output>
"""
    return prompt


# ================= Parsing Functions =================

def parse_extraction_result(result: str, chunk_id: str) -> Tuple[Dict, Dict]:
    """
    Phân tích kết quả trích xuất kiểu LightRAG
    
    Tham số:
        result: Chuỗi output của LLM
        chunk_id: Mã nhận diện chunk
    
    Trả về:
        (entities_dict, relationships_dict)
    """
    entities = defaultdict(list)
    relationships = defaultdict(list)
    
    # Chia theo dòng mới
    lines = result.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Bỏ qua dấu phân cách hoàn thành
        if COMPLETION_DELIMITER in line:
            continue
        
        # Chia theo dấu phân cách tuple
        parts = line.split(TUPLE_DELIMITER)
        
        if len(parts) < 4:
            continue
        
        record_type = parts[0].strip().lower()
        
        # Phân tích entity
        if 'entity' in record_type and len(parts) >= 4:
            entity_name = sanitize_text(parts[1])
            entity_type = sanitize_text(parts[2]).lower()
            entity_description = sanitize_text(parts[3])
            
            # Parse optional confidence score (default: 3)
            confidence = 3
            if len(parts) >= 5:
                try:
                    confidence = int(sanitize_text(parts[4]))
                    confidence = max(1, min(5, confidence))  # Clamp to 1-5
                except (ValueError, TypeError):
                    confidence = 3
            
            # Xác thực
            if not entity_name or not entity_description:
                logger.debug(f"⚠️ Bỏ qua entity không hợp lệ: {parts}")
                continue
            
            # Xác thực loại entity
            if entity_type not in ENTITY_TYPES:
                entity_type = 'other'
            
            # NEW: Xác thực chất lượng entity
            temp_entity = {
                'entity_name': entity_name,
                'entity_type': entity_type,
                'description': entity_description
            }
            
            if not validate_entity_quality(temp_entity):
                logger.debug(f"⚠️ Lọc entity chất lượng thấp: {entity_name}")
                continue
            
            entities[entity_name].append({
                'entity_name': entity_name,
                'entity_type': entity_type,
                'description': entity_description,
                'confidence': confidence,
                'source_id': chunk_id,
                'chunk_id': chunk_id
            })
        
        # Phân tích relationship
        elif 'relation' in record_type and len(parts) >= 5:
            source = sanitize_text(parts[1])
            target = sanitize_text(parts[2])
            keywords = sanitize_text(parts[3])
            description = sanitize_text(parts[4])
            
            # Xác thực
            if not source or not target or source == target:
                logger.debug(f" Bỏ qua relationship không hợp lệ: {parts}")
                continue
            
            relationships[(source, target)].append({
                'src_id': source,
                'tgt_id': target,
                'keywords': keywords,
                'description': description,
                'source_id': chunk_id,
                'chunk_id': chunk_id,
                'weight': 1.0  # Default weight
            })
    
    return dict(entities), dict(relationships)


def validate_entity_quality(entity: Dict) -> bool:
    """
    Xác thực chất lượng entity trước khi lưu
    
    Args:
        entity: Entity dict với 'entity_name' và 'description'
    
    Returns:
        True nếu entity có chất lượng tốt, False nếu nên bỏ qua
    """
    name = entity.get('entity_name', '')
    description = entity.get('description', '')
    
    # Lọc tên quá ngắn (< 2 ký tự)
    if len(name) < 2:
        return False
    
    # Lọc stop words phổ biến (tiếng Anh và tiếng Việt)
    stop_words = {
        # English
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        # Vietnamese
        'và', 'hoặc', 'nhưng', 'trong', 'trên', 'tại', 'của', 'với', 'bởi',
        'từ', 'là', 'được', 'có', 'sẽ', 'đã', 'đang', 'này', 'đó', 'các',
        'những', 'một', 'cho', 'về', 'theo', 'như', 'khi', 'nếu', 'thì'
    }
    
    if name.lower().strip() in stop_words:
        return False
    
    # Lọc tên chỉ chứa số
    if name.strip().isdigit():
        return False
    
    # Yêu cầu description có độ dài tối thiểu
    if not description or len(description.strip()) < 10:
        return False
    
    # Lọc entities có tên chỉ chứa ký tự đặc biệt
    import re
    if not re.search(r'[a-zA-Z0-9\u00C0-\u1EF9]', name):  # Bao gồm tiếng Việt có dấu
        return False
    
    return True


# ================= Entity Deduplication =================

def deduplicate_entities(entities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Merge similar entities using case-insensitive matching
    
    Args:
        entities: Dict of {entity_name: [entity_dicts]}
    
    Returns:
        Deduplicated entities dict
    """
    if not entities:
        return {}
    
    # Case-insensitive merge
    canonical = {}
    name_mapping = {}  # lowercase -> canonical name
    
    for name, ents in entities.items():
        name_lower = name.lower().strip()
        
        if name_lower in name_mapping:
            # Merge with existing
            canonical_name = name_mapping[name_lower]
            canonical[canonical_name].extend(ents)
        else:
            # New entity
            canonical[name] = ents
            name_mapping[name_lower] = name
    
    # Merge descriptions for duplicates
    for name, ents in canonical.items():
        if len(ents) > 1:
            # Combine descriptions
            descriptions = [e['description'] for e in ents if e.get('description')]
            unique_descriptions = []
            seen = set()
            
            for desc in descriptions:
                desc_lower = desc.lower()
                if desc_lower not in seen:
                    unique_descriptions.append(desc)
                    seen.add(desc_lower)
            
            merged_desc = '; '.join(unique_descriptions)[:500]
            
            # Keep first entity with merged description
            best = ents[0].copy()
            best['description'] = merged_desc
            
            canonical[name] = [best]
    
    reduction = len(entities) - len(canonical)
    if reduction > 0:
        logger.info(f" Gộp entities: {len(entities)} → {len(canonical)} entities ({reduction} đã gộp)")
    
    return canonical


async def deduplicate_entities_with_llm(
    entities: Dict[str, List[Dict]], 
    llm_func,
    min_descriptions_for_llm: int = 3
) -> Dict[str, List[Dict]]:
    """
    Advanced entity deduplication with LLM-based description summarization
    
    Args:
        entities: Dict of {entity_name: [entity_dicts]}
        llm_func: Async LLM function for summarization
        min_descriptions_for_llm: Minimum unique descriptions to trigger LLM (default: 3)
    
    Returns:
        Deduplicated entities dict with LLM-summarized descriptions
    """
    if not entities:
        return {}
    
    # First do basic deduplication
    canonical = {}
    name_mapping = {}
    
    for name, ents in entities.items():
        name_lower = name.lower().strip()
        
        if name_lower in name_mapping:
            canonical_name = name_mapping[name_lower]
            canonical[canonical_name].extend(ents)
        else:
            canonical[name] = ents
            name_mapping[name_lower] = name
    
    # Merge descriptions with LLM for complex cases
    llm_merge_count = 0
    
    for name, ents in canonical.items():
        if len(ents) > 1:
            # Get unique descriptions
            descriptions = [e['description'] for e in ents if e.get('description')]
            unique_descriptions = []
            seen = set()
            
            for desc in descriptions:
                desc_lower = desc.lower()
                if desc_lower not in seen:
                    unique_descriptions.append(desc)
                    seen.add(desc_lower)
            
            # Use LLM if many unique descriptions
            if len(unique_descriptions) >= min_descriptions_for_llm and llm_func:
                try:
                    # Create summarization prompt
                    prompt = f"""Summarize the following descriptions of "{name}" into one concise, comprehensive description.
Combine all unique information without repetition. Keep it under 200 words.

Descriptions:
{chr(10).join(f"{i+1}. {d}" for i, d in enumerate(unique_descriptions))}

Provide only the final summarized description, no additional text."""
                    
                    merged_desc = await llm_func(prompt)
                    merged_desc = merged_desc.strip()[:500]
                    llm_merge_count += 1
                    
                except Exception as e:
                    logger.warning(f" Gộp bằng LLM thất bại cho {name}: {e}, sử dụng gộp đơn giản")
                    merged_desc = '; '.join(unique_descriptions)[:500]
            else:
                # Simple concatenation for few descriptions
                merged_desc = '; '.join(unique_descriptions)[:500]
            
            # Keep first entity with merged description
            best = ents[0].copy()
            best['description'] = merged_desc
            
            # Track all source_ids
            all_sources = set()
            for e in ents:
                if 'source_id' in e:
                    all_sources.add(e['source_id'])
            best['source_ids'] = list(all_sources)
            
            canonical[name] = [best]
    
    reduction = len(entities) - len(canonical)
    if reduction > 0:
        logger.info(
            f" Gộp entities: {len(entities)} → {len(canonical)} entities "
            f"({reduction} đã gộp, {llm_merge_count} với LLM)"
        )
    
    return canonical


# ================= Relationship Validation =================

def validate_relationships(
    relationships: Dict[Tuple[str, str], List[Dict]],
    entities: Dict[str, List[Dict]]
) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Validate relationships - ensure both entities exist
    
    Args:
        relationships: Dict of {(src, tgt): [relationship_dicts]}
        entities: Dict of {entity_name: [entity_dicts]}
    
    Returns:
        Validated relationships dict
    """
    if not relationships or not entities:
        return relationships
    
    valid_relationships = {}
    entity_names_lower = {n.lower(): n for n in entities.keys()}
    
    filtered_count = 0
    
    for (src, tgt), rels in relationships.items():
        # Case-insensitive lookup
        src_lower = src.lower()
        tgt_lower = tgt.lower()
        
        src_canonical = entity_names_lower.get(src_lower)
        tgt_canonical = entity_names_lower.get(tgt_lower)
        
        if src_canonical and tgt_canonical and src_canonical != tgt_canonical:
            # Update relationship with canonical names
            for rel in rels:
                rel['src_id'] = src_canonical
                rel['tgt_id'] = tgt_canonical
            
            valid_relationships[(src_canonical, tgt_canonical)] = rels
        else:
            filtered_count += len(rels)
    
    if filtered_count > 0:
        logger.info(f" Xác thực: {filtered_count} relationships không hợp lệ đã bị lọc")
    
    return valid_relationships


async def deduplicate_relationships_with_llm(
    relationships: Dict[Tuple[str, str], List[Dict]],
    llm_func,
    min_descriptions_for_llm: int = 3
) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Advanced relationship deduplication with LLM-based description summarization
    
    Args:
        relationships: Dict of {(src, tgt): [relationship_dicts]}
        llm_func: Async LLM function for summarization
        min_descriptions_for_llm: Minimum unique descriptions to trigger LLM (default: 3)
    
    Returns:
        Deduplicated relationships dict with LLM-summarized descriptions
    """
    if not relationships:
        return {}
    
    merged_relationships = {}
    llm_merge_count = 0
    
    for (src, tgt), rels in relationships.items():
        if len(rels) > 1:
            # Get unique descriptions
            descriptions = [r['description'] for r in rels if r.get('description')]
            unique_descriptions = []
            seen = set()
            
            for desc in descriptions:
                desc_lower = desc.lower()
                if desc_lower not in seen:
                    unique_descriptions.append(desc)
                    seen.add(desc_lower)
            
            # Merge keywords
            all_keywords = []
            for r in rels:
                if r.get('keywords'):
                    all_keywords.extend([k.strip() for k in r['keywords'].split(',')])
            unique_keywords = list(set(all_keywords))
            merged_keywords = ', '.join(unique_keywords[:5])  # Top 5 keywords
            
            # Use LLM if many unique descriptions
            if len(unique_descriptions) >= min_descriptions_for_llm and llm_func:
                try:
                    # Create summarization prompt
                    prompt = f"""Summarize the following descriptions of the relationship between "{src}" and "{tgt}" into one concise description.
Combine all unique information without repetition. Keep it under 150 words.

Descriptions:
{chr(10).join(f"{i+1}. {d}" for i, d in enumerate(unique_descriptions))}

Provide only the final summarized description, no additional text."""
                    
                    merged_desc = await llm_func(prompt)
                    merged_desc = merged_desc.strip()[:400]
                    llm_merge_count += 1
                    
                except Exception as e:
                    logger.warning(f" Gộp bằng LLM thất bại cho relationship ({src}, {tgt}): {e}, sử dụng gộp đơn giản")
                    merged_desc = '; '.join(unique_descriptions)[:400]
            else:
                # Simple concatenation for few descriptions
                merged_desc = '; '.join(unique_descriptions)[:400]
            
            # Keep first relationship with merged description
            best = rels[0].copy()
            best['description'] = merged_desc
            best['keywords'] = merged_keywords
            
            # Track all source_ids
            all_sources = set()
            for r in rels:
                if 'source_id' in r:
                    all_sources.add(r['source_id'])
            best['source_ids'] = list(all_sources)
            
            merged_relationships[(src, tgt)] = [best]
        else:
            # Single relationship, keep as is
            merged_relationships[(src, tgt)] = rels
    
    reduction = sum(len(v) for v in relationships.values()) - sum(len(v) for v in merged_relationships.values())
    if reduction > 0:
        logger.info(
            f" Gộp relationships: {sum(len(v) for v in relationships.values())} → "
            f"{sum(len(v) for v in merged_relationships.values())} relationships "
            f"({reduction} đã gộp, {llm_merge_count} với LLM)"
        )
    
    return merged_relationships


# ================= Async Extraction =================

async def extract_from_chunk(chunk: Dict, llm_func, max_gleaning: int = 1) -> Tuple[Dict, Dict]:
    """
    Multi-stage extraction from one chunk with gleaning (LightRAG-style)
    
    Args:
        chunk: Chunk dict with 'content' and 'chunk_id'
        llm_func: Async LLM function
        max_gleaning: Max number of continue extraction attempts (default: 1)
    
    Returns:
        (entities_dict, relationships_dict)
    """
    try:
        # First extraction
        prompt = create_extraction_prompt(chunk['content'])
        result = await llm_func(prompt)
        
        # Continue extraction (gleaning) if incomplete
        for i in range(max_gleaning):
            # Check if complete
            if COMPLETION_DELIMITER in result:
                break
            
            # Continue extraction
            logger.debug(f" Tiếp tục trích xuất {i+1}/{max_gleaning} cho chunk {chunk.get('chunk_id')}")
            continue_prompt = create_continue_extraction_prompt(
                chunk['content'], 
                result
            )
            continue_result = await llm_func(continue_prompt)
            result += "\n" + continue_result
        
        # Parse combined result
        entities, relationships = parse_extraction_result(result, chunk['chunk_id'])
        
        return entities, relationships
    
    except Exception as e:
        logger.error(f" Trích xuất thất bại cho chunk {chunk.get('chunk_id')}: {e}")
        return {}, {}


async def extract_async(
    chunks: List[Dict],
    llm_func,
    max_concurrent: int = 16
) -> Tuple[Dict, Dict]:
    """
    Async extraction from multiple chunks
    
    Args:
        chunks: List of chunk dicts
        llm_func: Async LLM function
        max_concurrent: Max concurrent LLM calls
    
    Returns:
        (entities_dict, relationships_dict)
    """
    sem = asyncio.Semaphore(max_concurrent)
    
    async def process(chunk):
        async with sem:
            return await extract_from_chunk(chunk, llm_func)
    
    results = await asyncio.gather(*[process(c) for c in chunks])
    
    # Merge all results
    all_entities = defaultdict(list)
    all_relationships = defaultdict(list)
    
    for entities, relationships in results:
        for k, v in entities.items():
            all_entities[k].extend(v)
        for k, v in relationships.items():
            all_relationships[k].extend(v)
    
    # Post-processing with LLM-based deduplication
    all_entities = await deduplicate_entities_with_llm(
        dict(all_entities), 
        llm_func,
        min_descriptions_for_llm=3
    )
    all_relationships = validate_relationships(dict(all_relationships), all_entities)
    
    # Deduplicate relationships with LLM
    all_relationships = await deduplicate_relationships_with_llm(
        all_relationships,
        llm_func,
        min_descriptions_for_llm=3
    )
    
    return dict(all_entities), dict(all_relationships)


# ================= Main Entry Point =================

def extract_entities_relations(
    chunks: List[Dict],
    global_config: Dict = None,
    **kwargs  # Accept but ignore legacy parameters
) -> Tuple[Dict, Dict]:
    """
    Main extraction function - LightRAG style
    
    Args:
        chunks: List of chunks with 'content' and 'chunk_id'
        global_config: Config dict with 'llm_model_func' (optional)
        **kwargs: Legacy parameters (ignored)
    
    Returns:
        (entities_dict, relationships_dict)
    """
    if not chunks:
        logger.warning(" Không có chunks được cung cấp")
        return {}, {}
    
    # Get LLM function
    llm_func = global_config.get('llm_model_func') if global_config else None
    if not llm_func:
        from backend.utils.llm_utils import call_llm_async
        llm_func = call_llm_async
    
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        logger.info(f" Trích xuất kiểu LightRAG với gleaning (chunks={len(chunks)})")
        
        # Extract
        entities, relationships = loop.run_until_complete(
            extract_async(chunks, llm_func, max_concurrent=16)
        )
        
        # Stats
        entity_count = sum(len(v) for v in entities.values())
        rel_count = sum(len(v) for v in relationships.values())
        logger.info(f" Đã trích xuất: {entity_count} entities, {rel_count} relationships")
        
        return entities, relationships
    
    except Exception as e:
        logger.error(f" Trích xuất thất bại: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}


# ================= Statistics =================

def get_extraction_statistics(entities: Dict, relationships: Dict) -> Dict:
    """Get detailed extraction statistics"""
    entity_types = defaultdict(int)
    
    for ents in entities.values():
        for ent in ents:
            entity_types[ent['entity_type']] += 1
    
    total_entities = sum(len(v) for v in entities.values())
    unique_entities = len(entities)
    
    return {
        'total_entities': total_entities,
        'unique_entities': unique_entities,
        'entity_types': dict(entity_types),
        'total_relationships': sum(len(v) for v in relationships.values()),
        'unique_relationship_pairs': len(relationships),
        'deduplication_ratio': 1 - (unique_entities / max(total_entities, 1))
    }


# ================= Export =================

__all__ = [
    'extract_entities_relations',
    'get_extraction_statistics',
    'deduplicate_entities',
    'deduplicate_entities_with_llm',
    'validate_relationships',
    'ENTITY_TYPES',
    'TUPLE_DELIMITER',
    'COMPLETION_DELIMITER'
]