# backend/core/extraction.py 
"""
🚀 Enhanced Entity & Relationship Extraction 
"""

import asyncio
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"

# ================= Configuration =================

ENTITY_TYPES = {
    'PERSON': 'Individuals, characters, historical figures',
    'ORGANIZATION': 'Companies, institutions, groups',
    'LOCATION': 'Places, geographical locations',
    'EVENT': 'Meetings, conferences, occurrences',
    'PRODUCT': 'Goods, services, brands',
    'CONCEPT': 'Ideas, theories, methodologies',
    'TECHNOLOGY': 'Tools, systems, platforms',
    'DATE': 'Temporal references',
    'METRIC': 'Numbers, measurements, statistics'
}

STATIC_TYPES = {
    'WORKS_FOR': 'Employment', 'MANAGES': 'Management', 'REPORTS_TO': 'Hierarchy',
    'COLLABORATES_WITH': 'Partnership', 'COMPETES_WITH': 'Competition',
    'PRODUCES': 'Manufacturing', 'PROVIDES': 'Service', 'USES': 'Utilization',
    'DEVELOPS': 'Innovation', 'OWNS': 'Ownership', 'SELLS': 'Commercial',
    'LOCATED_IN': 'Location', 'OPERATES_IN': 'Operation', 'BASED_IN': 'Headquarters',
    'PARTICIPATES_IN': 'Participation', 'ORGANIZES': 'Organization', 'ATTENDS': 'Attendance',
    'IMPLEMENTS': 'Implementation', 'RESEARCHES': 'Research', 'INVENTED_BY': 'Invention',
    'RELATED_TO': 'General', 'PART_OF': 'Membership',
    'FOUNDED': 'Establishment', 'ACQUIRED': 'Acquisition', 'MERGED_WITH': 'Merger',
    'INVESTED_IN': 'Investment', 'ASSOCIATED_WITH': 'Association',
    'INFLUENCES': 'Influence', 'DEPENDS_ON': 'Dependency'
}

CATEGORIES = {
    'HIERARCHICAL': ['part of', 'belongs to', 'member of', 'within', 'reports to'],
    'FUNCTIONAL': ['creates', 'produces', 'develops', 'operates', 'uses', 'implements'],
    'TEMPORAL': ['founded', 'acquired', 'merged', 'started', 'ended'],
    'LOCATIONAL': ['in', 'at', 'based', 'located', 'operates in'],
    'ASSOCIATIVE': ['works', 'employed', 'member', 'collaborates', 'partners'],
    'CAUSAL': ['causes', 'leads to', 'results in', 'affects', 'impacts'],
    'COMPARATIVE': ['similar to', 'differs from', 'competes with']
}

DOMAIN_KEYWORDS = {
    'tech': ['software', 'AI', 'algorithm', 'cloud', 'API', 'code', 'data'],
    'medical': ['patient', 'treatment', 'drug', 'clinical', 'disease', 'doctor'],
    'legal': ['court', 'law', 'case', 'judge', 'ruling', 'contract'],
    'business': ['revenue', 'market', 'sales', 'investment', 'profit']
}

# ================= Core Functions =================

def detect_domain(chunks: List[Dict]) -> str:
    """Detect domain from content"""
    text = " ".join([c['content'][:500] for c in chunks[:5]]).lower()
    scores = {d: sum(1 for k in kws if k in text) for d, kws in DOMAIN_KEYWORDS.items()}
    return max(scores, key=scores.get) if max(scores.values()) >= 3 else 'general'


def select_mode(chunks: List[Dict], domain: Optional[str] = None, force_mode: Optional[str] = None) -> str:
    """Select extraction mode"""
    if force_mode and force_mode != 'auto':
        return force_mode
    
    domain = domain or detect_domain(chunks)
    mode_map = {'medical': 'dynamic', 'legal': 'dynamic'}
    selected = mode_map.get(domain, 'static')
    
    logger.info(f"🎯 Mode: {selected} (domain: {domain})")
    return selected


def categorize(text: str) -> str:
    """Categorize relationship type"""
    text_lower = text.lower()
    for cat, patterns in CATEGORIES.items():
        if any(p in text_lower for p in patterns):
            return cat
    return 'ASSOCIATIVE'


def safe_float(value: str, default: float = 0.7) -> float:
    """Convert to float safely"""
    try:
        return max(0.0, min(1.0, float(str(value).strip().strip('"\'').lower())))
    except:
        word_map = {'strong': 0.85, 'high': 0.8, 'medium': 0.6, 'low': 0.4, 'weak': 0.35}
        for word, score in word_map.items():
            if word in str(value).lower():
                return score
        return default


# ================= STAGE 1: Coarse Extraction =================

def create_coarse_prompt(text: str, context: Optional[str] = None) -> str:
    """
    Stage 1: Quick entity identification
    Fast, inclusive extraction
    """
    ctx = f"\n**Context**: {context}\n" if context else ""
    
    return f"""Identify ALL important entities in this text.{ctx}

# ENTITY TYPES
{', '.join(ENTITY_TYPES.keys())}

# INSTRUCTIONS
- Extract EVERY significant entity (people, organizations, locations, concepts)
- Be INCLUSIVE - when in doubt, extract it
- One entity per line

# FORMAT
EntityName | EntityType | Brief description

# EXAMPLE
OpenAI | ORGANIZATION | AI research company
GPT-4 | PRODUCT | Large language model
Sam Altman | PERSON | CEO of OpenAI

# TEXT
{text}

# OUTPUT:"""


def parse_coarse_result(result: str, chunk_id: str) -> Dict[str, List[Dict]]:
    """Parse coarse extraction results"""
    entities = defaultdict(list)
    
    for line in result.strip().split('\n'):
        line = line.strip()
        if not line or '|' not in line:
            continue
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 2:
            name = parts[0].strip().strip('"\'')
            etype = parts[1].strip().upper() if len(parts) > 1 else 'CONCEPT'
            desc = parts[2].strip() if len(parts) > 2 else ''
            
            # Filter noise
            if name and len(name) > 1 and name.lower() not in ['entity', 'name', 'type']:
                entities[name].append({
                    'entity_name': name,
                    'entity_type': etype if etype in ENTITY_TYPES else 'CONCEPT',
                    'description': desc,
                    'source_id': chunk_id,
                    'chunk_id': chunk_id,
                    'stage': 'coarse'
                })
    
    return dict(entities)


# ================= STAGE 2: Fine Extraction =================

def create_fine_prompt(text: str, coarse_entities: List[str], mode: str, context: Optional[str] = None) -> str:
    """
    Stage 2: Detailed extraction with relationships
    Validates entities + extracts connections
    """
    entities_list = ', '.join(coarse_entities[:20]) if coarse_entities else 'Extract from text'
    ctx = f"\n**Context**: {context}\n" if context else ""
    
    if mode == 'static':
        rel_types = '\n'.join([f"- {t}" for t in list(STATIC_TYPES.keys())[:15]])
        rel_format = "<RELATIONSHIP_TYPE from list>"
        example = "DEVELOPS"
    else:  # dynamic
        rel_types = "Use natural verbs: develops, manages, located_in, works_for, etc."
        rel_format = "<verb_phrase>"
        example = "develops and maintains"
    
    return f"""Refine entities and extract relationships.{ctx}

# KNOWN ENTITIES
{entities_list}

# RELATIONSHIP TYPES
{rel_types}

# FORMAT
Entity: ("entity"{TUPLE_DELIMITER}<name>{TUPLE_DELIMITER}<type>{TUPLE_DELIMITER}<description>){RECORD_DELIMITER}
Relation: ("relationship"{TUPLE_DELIMITER}<source>{TUPLE_DELIMITER}<target>{TUPLE_DELIMITER}{rel_format}{TUPLE_DELIMITER}<description>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<0.0-1.0>){RECORD_DELIMITER}

# EXAMPLE
("entity"{TUPLE_DELIMITER}OpenAI{TUPLE_DELIMITER}ORGANIZATION{TUPLE_DELIMITER}Leading AI research organization){RECORD_DELIMITER}
("entity"{TUPLE_DELIMITER}GPT-4{TUPLE_DELIMITER}PRODUCT{TUPLE_DELIMITER}Advanced large language model){RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}OpenAI{TUPLE_DELIMITER}GPT-4{TUPLE_DELIMITER}{example}{TUPLE_DELIMITER}OpenAI developed and maintains GPT-4{TUPLE_DELIMITER}AI, LLM, development{TUPLE_DELIMITER}0.95){RECORD_DELIMITER}

# TEXT
{text}

# OUTPUT:"""


def parse_fine_result(result: str, chunk_id: str, mode: str) -> Tuple[Dict, Dict]:
    """Parse fine extraction with validation"""
    entities = defaultdict(list)
    relationships = defaultdict(list)
    
    for record in re.split(f'{RECORD_DELIMITER}|<\\|COMPLETE\\|>', result):
        match = re.search(r'\((.*?)\)', record)
        if not match:
            continue
        
        parts = match.group(1).split(TUPLE_DELIMITER)
        
        # Entity
        if len(parts) >= 4 and 'entity' in parts[0].lower():
            name = parts[1].strip().strip('"\'')
            if name and len(name) > 1:
                entities[name].append({
                    'entity_name': name,
                    'entity_type': parts[2].strip().upper(),
                    'description': parts[3].strip().strip('"\''),
                    'source_id': chunk_id,
                    'chunk_id': chunk_id,
                    'stage': 'fine'
                })
        
        # Relationship
        elif len(parts) >= 6 and 'relationship' in parts[0].lower():
            src = parts[1].strip().strip('"\'')
            tgt = parts[2].strip().strip('"\'')
            rel_type = parts[3].strip().strip('"\'')
            
            if src and tgt and src != tgt and rel_type and len(src) > 1 and len(tgt) > 1:
                # Normalize
                if mode == 'static':
                    rel_type_upper = rel_type.upper()
                    rel_type_final = rel_type_upper if rel_type_upper in STATIC_TYPES else 'RELATED_TO'
                    verb = rel_type.lower()
                else:  # dynamic
                    verb = rel_type.lower()
                    rel_type_final = verb.upper().replace(' ', '_')
                
                relationships[(src, tgt)].append({
                    'source_id': src,
                    'target_id': tgt,
                    'relationship_type': rel_type_final,
                    'verb_phrase': verb,
                    'category': categorize(verb),
                    'description': parts[4].strip().strip('"\''),
                    'keywords': parts[5].strip().strip('"\''),
                    'weight': safe_float(parts[6] if len(parts) > 6 else '0.7'),
                    'chunk_id': chunk_id,
                    'extraction_mode': mode,
                    'stage': 'fine'
                })
    
    return dict(entities), dict(relationships)


# ================= Entity Deduplication =================

def deduplicate_entities(entities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """
    Merge similar entities using fuzzy matching
    Inspired by LightRAG's entity merging
    """
    if not entities:
        return {}
    
    # Step 1: Exact match (case-insensitive)
    canonical = {}
    for name, ents in entities.items():
        name_lower = name.lower().strip()
        
        found = False
        for canon_name in list(canonical.keys()):
            if name_lower == canon_name.lower():
                canonical[canon_name].extend(ents)
                found = True
                break
        
        if not found:
            canonical[name] = ents
    
    # Step 2: Fuzzy match (similarity > 0.85)
    merged = {}
    processed = set()
    names = list(canonical.keys())
    
    for i, name1 in enumerate(names):
        if name1 in processed:
            continue
        
        cluster = [name1]
        
        for j, name2 in enumerate(names[i+1:], start=i+1):
            if name2 in processed:
                continue
            
            # Calculate similarity
            ratio = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
            
            if ratio > 0.85:
                cluster.append(name2)
                processed.add(name2)
        
        # Use longest name as canonical
        canonical_name = max(cluster, key=len)
        
        merged[canonical_name] = []
        for name in cluster:
            merged[canonical_name].extend(canonical[name])
        
        processed.add(name1)
    
    # Step 3: Merge descriptions
    for name, ents in merged.items():
        if len(ents) > 1:
            descriptions = [e['description'] for e in ents if e.get('description')]
            merged_desc = '; '.join(set(descriptions))[:500]
            
            best = max(ents, key=lambda e: len(e.get('description', '')))
            best['description'] = merged_desc
            
            merged[name] = [best]
    
    reduction = len(entities) - len(merged)
    if reduction > 0:
        logger.info(f"✅ Deduplication: {len(entities)} → {len(merged)} entities ({reduction} removed)")
    
    return merged


# ================= Relationship Validation =================

def validate_relationships(
    relationships: Dict[Tuple[str, str], List[Dict]],
    entities: Dict[str, List[Dict]]
) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Validate relationships - ensure both entities exist
    """
    if not relationships or not entities:
        return relationships
    
    valid_relationships = {}
    entity_names = set(entities.keys())
    entity_names_lower = {n.lower(): n for n in entity_names}
    
    filtered_count = 0
    
    for (src, tgt), rels in relationships.items():
        # Try exact match
        src_found = src in entity_names
        tgt_found = tgt in entity_names
        
        # Try case-insensitive
        if not src_found:
            src_lower = src.lower()
            if src_lower in entity_names_lower:
                src = entity_names_lower[src_lower]
                src_found = True
        
        if not tgt_found:
            tgt_lower = tgt.lower()
            if tgt_lower in entity_names_lower:
                tgt = entity_names_lower[tgt_lower]
                tgt_found = True
        
        if src_found and tgt_found and src != tgt:
            valid_relationships[(src, tgt)] = rels
        else:
            filtered_count += len(rels)
    
    if filtered_count > 0:
        logger.info(f"✅ Validation: {filtered_count} invalid relationships filtered")
    
    return valid_relationships


# ================= Two-Stage Extraction =================

async def extract_two_stage(
    chunk: Dict,
    llm_func,
    mode: str,
    context: Optional[str] = None
) -> Tuple[Dict, Dict]:
    """
    Two-stage extraction: coarse → fine
    """
    try:
        # Stage 1: Coarse (fast)
        coarse_prompt = create_coarse_prompt(chunk['content'], context)
        coarse_result = await llm_func(coarse_prompt)
        coarse_entities = parse_coarse_result(coarse_result, chunk['chunk_id'])
        
        if not coarse_entities:
            logger.debug(f"No entities in coarse stage for {chunk['chunk_id']}")
            return {}, {}
        
        # Stage 2: Fine (detailed)
        entity_names = list(coarse_entities.keys())
        fine_prompt = create_fine_prompt(chunk['content'], entity_names, mode, context)
        fine_result = await llm_func(fine_prompt)
        fine_entities, relationships = parse_fine_result(fine_result, chunk['chunk_id'], mode)
        
        # Merge (prioritize fine)
        merged_entities = {**coarse_entities, **fine_entities}
        
        return merged_entities, relationships
    
    except Exception as e:
        logger.error(f"Two-stage extraction failed for {chunk.get('chunk_id')}: {e}")
        return {}, {}


# ================= Single-Stage Fallback =================

async def extract_single_stage(
    chunk: Dict,
    llm_func,
    mode: str,
    context: Optional[str] = None
) -> Tuple[Dict, Dict]:
    """Single-stage extraction (fallback)"""
    try:
        prompt = create_fine_prompt(chunk['content'], [], mode, context)
        result = await llm_func(prompt)
        return parse_fine_result(result, chunk['chunk_id'], mode)
    except Exception as e:
        logger.error(f"Single-stage extraction failed: {e}")
        return {}, {}


# ================= Main Async Extraction =================

async def extract_async(
    chunks: List[Dict],
    llm_func,
    mode: str,
    max_concurrent: int = 16,
    context: Optional[str] = None,
    use_two_stage: bool = True
) -> Tuple[Dict, Dict]:
    """
    Main async extraction with enhancements
    """
    sem = asyncio.Semaphore(max_concurrent)
    
    async def process(chunk):
        async with sem:
            if use_two_stage:
                return await extract_two_stage(chunk, llm_func, mode, context)
            else:
                return await extract_single_stage(chunk, llm_func, mode, context)
    
    results = await asyncio.gather(*[process(c) for c in chunks])
    
    # Merge all results
    all_entities = defaultdict(list)
    all_relationships = defaultdict(list)
    
    for entities, relationships in results:
        for k, v in entities.items():
            all_entities[k].extend(v)
        for k, v in relationships.items():
            all_relationships[k].extend(v)
    
    # Post-processing
    all_entities = deduplicate_entities(dict(all_entities))
    all_relationships = validate_relationships(dict(all_relationships), all_entities)
    
    return dict(all_entities), dict(all_relationships)


# ================= Main Entry Point =================

def extract_entities_relations(
    chunks: List[Dict],
    global_config: Dict = None,
    mode: str = 'auto',
    domain: Optional[str] = None,
    context: Optional[str] = None,
    use_two_stage: bool = True
) -> Tuple[Dict, Dict]:
    """
    Main extraction function with LightRAG enhancements
    
    Args:
        chunks: List of chunks
        global_config: Config with 'llm_model_func'
        mode: 'static', 'dynamic', or 'auto'
        domain: Domain hint
        context: Optional context
        use_two_stage: Enable two-stage extraction (default: True)
    
    Returns:
        (entities_dict, relationships_dict)
    """
    if not chunks:
        logger.warning("⚠️ No chunks provided")
        return {}, {}
    
    # Select mode
    selected_mode = select_mode(chunks, domain, mode)
    
    # Get LLM
    llm_func = global_config.get('llm_model_func') if global_config else None
    if not llm_func:
        from backend.utils.llm_utils import call_llm_async
        llm_func = call_llm_async
    
    try:
        # Event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        extraction_type = "two-stage" if use_two_stage else "single-stage"
        logger.info(f"🚀 Enhanced extraction ({extraction_type}, mode={selected_mode}, chunks={len(chunks)})")
        
        # Extract
        entities, relationships = loop.run_until_complete(
            extract_async(chunks, llm_func, selected_mode, 16, context, use_two_stage)
        )
        
        # Stats
        entity_count = sum(len(v) for v in entities.values())
        rel_count = sum(len(v) for v in relationships.values())
        logger.info(f"✅ Extracted: {entity_count} entities, {rel_count} relationships")
        
        return entities, relationships
    
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return {}, {}


# ================= Statistics =================

def get_statistics(relationships: Dict) -> Dict:
    """Get relationship statistics"""
    stats = {
        'total': sum(len(v) for v in relationships.values()),
        'by_type': defaultdict(int),
        'by_category': defaultdict(int),
        'by_mode': defaultdict(int)
    }
    
    for rels in relationships.values():
        for rel in rels:
            stats['by_type'][rel.get('relationship_type', 'UNKNOWN')] += 1
            stats['by_category'][rel.get('category', 'UNKNOWN')] += 1
            stats['by_mode'][rel.get('extraction_mode', 'unknown')] += 1
    
    return stats


def get_extraction_statistics(entities: Dict, relationships: Dict) -> Dict:
    """Get detailed extraction statistics"""
    entity_types = defaultdict(int)
    rel_categories = defaultdict(int)
    
    for ents in entities.values():
        for ent in ents:
            entity_types[ent['entity_type']] += 1
    
    for rels in relationships.values():
        for rel in rels:
            rel_categories[rel['category']] += 1
    
    total_entities = sum(len(v) for v in entities.values())
    unique_entities = len(entities)
    
    return {
        'total_entities': total_entities,
        'unique_entities': unique_entities,
        'entity_types': dict(entity_types),
        'total_relationships': sum(len(v) for v in relationships.values()),
        'relationship_categories': dict(rel_categories),
        'deduplication_ratio': 1 - (unique_entities / max(total_entities, 1))
    }


# ================= Export =================

__all__ = [
    'extract_entities_relations',
    'select_mode',
    'get_statistics',
    'get_extraction_statistics',
    'deduplicate_entities',
    'validate_relationships',
    'STATIC_TYPES',
    'CATEGORIES',
    'ENTITY_TYPES'
]