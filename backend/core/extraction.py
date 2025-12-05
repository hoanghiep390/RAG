# backend/core/extraction.py - HYBRID COMPACT VERSION
"""
🔍 Hybrid Entity & Relationship Extraction
- Static: Pre-defined types (fast)
- Dynamic: Free-form verbs (flexible)
- Auto: Smart selection
"""

import asyncio
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"

# ================= Configuration =================

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
    'HIERARCHICAL': ['part of', 'belongs to', 'member of', 'within'],
    'FUNCTIONAL': ['creates', 'produces', 'develops', 'operates', 'uses'],
    'TEMPORAL': ['founded', 'acquired', 'merged', 'started'],
    'LOCATIONAL': ['in', 'at', 'based', 'located'],
    'ASSOCIATIVE': ['works', 'employed', 'member', 'CEO', 'manages'],
    'INFLUENCE': ['affects', 'impacts', 'competes', 'influences']
}

DOMAIN_KEYWORDS = {
    'tech': ['software', 'AI', 'algorithm', 'cloud', 'API', 'code'],
    'medical': ['patient', 'treatment', 'drug', 'clinical', 'disease'],
    'legal': ['court', 'law', 'case', 'judge', 'ruling'],
    'business': ['revenue', 'market', 'sales', 'investment']
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
    """Categorize verb/type into category"""
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


# ================= Prompts =================

def create_prompt(text: str, mode: str, context: Optional[str] = None) -> str:
    """Create extraction prompt"""
    ctx = f"\n**Context**: {context}\n" if context else ""
    
    if mode == 'static':
        types_list = '\n'.join([f"- {t}" for t in list(STATIC_TYPES.keys())[:12]])
        rel_format = f"<RELATIONSHIP_TYPE from list above>"
        example_type = "DEVELOPS"
    else:  # dynamic
        types_list = "Use natural verb phrases like:\n- develops, is CEO of, operates, competes with"
        rel_format = f"<verb_phrase in natural language>"
        example_type = "develops and maintains"
    
    return f"""Extract entities and relationships.{ctx}

# ENTITIES
Types: PERSON, ORGANIZATION, LOCATION, EVENT, PRODUCT, CONCEPT, TECHNOLOGY

# RELATIONSHIPS
{types_list}

# FORMAT
Entity: ("entity"{TUPLE_DELIMITER}<name>{TUPLE_DELIMITER}<type>{TUPLE_DELIMITER}<description>){RECORD_DELIMITER}
Relation: ("relationship"{TUPLE_DELIMITER}<source>{TUPLE_DELIMITER}<target>{TUPLE_DELIMITER}{rel_format}{TUPLE_DELIMITER}<description>{TUPLE_DELIMITER}<keywords>{TUPLE_DELIMITER}<0.0-1.0>){RECORD_DELIMITER}

# EXAMPLE
("entity"{TUPLE_DELIMITER}OpenAI{TUPLE_DELIMITER}ORGANIZATION{TUPLE_DELIMITER}AI company){RECORD_DELIMITER}
("relationship"{TUPLE_DELIMITER}OpenAI{TUPLE_DELIMITER}GPT-4{TUPLE_DELIMITER}{example_type}{TUPLE_DELIMITER}OpenAI developed GPT-4{TUPLE_DELIMITER}AI, LLM{TUPLE_DELIMITER}0.95){RECORD_DELIMITER}

# TEXT
{text}

# OUTPUT:"""


# ================= Parsing =================

def parse_result(result: str, chunk_id: str, mode: str) -> Tuple[Dict, Dict]:
    """Parse LLM output"""
    entities, relationships = defaultdict(list), defaultdict(list)
    
    for record in re.split(f'{RECORD_DELIMITER}|<\\|COMPLETE\\|>', result):
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
                    'entity_type': parts[2].strip().strip('"\'').upper(),
                    'description': parts[3].strip().strip('"\''),
                    'source_id': chunk_id,
                    'chunk_id': chunk_id
                })
        
        # Relationship
        elif len(parts) >= 6 and 'relationship' in parts[0].lower():
            src = parts[1].strip().strip('"\'')
            tgt = parts[2].strip().strip('"\'')
            rel_type = parts[3].strip().strip('"\'')
            
            if src and tgt and src != tgt and rel_type:
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
                    'weight': safe_float(parts[6].strip().strip('"\'') if len(parts) > 6 else '0.7'),
                    'chunk_id': chunk_id,
                    'extraction_mode': mode
                })
    
    return dict(entities), dict(relationships)


# ================= Async Extraction =================

async def extract_single(chunk: Dict, llm_func, mode: str, context: Optional[str] = None) -> Tuple[Dict, Dict]:
    """Extract from single chunk"""
    try:
        prompt = create_prompt(chunk['content'], mode, context)
        result = await llm_func(prompt)
        return parse_result(result, chunk['chunk_id'], mode)
    except Exception as e:
        logger.error(f"Extract error for {chunk.get('chunk_id')}: {e}")
        return {}, {}


async def extract_async(chunks: List[Dict], llm_func, mode: str, max_concurrent: int = 16, 
                       context: Optional[str] = None) -> Tuple[Dict, Dict]:
    """Extract from multiple chunks"""
    sem = asyncio.Semaphore(max_concurrent)
    
    async def process(chunk):
        async with sem:
            return await extract_single(chunk, llm_func, mode, context)
    
    results = await asyncio.gather(*[process(c) for c in chunks])
    
    # Merge
    all_entities, all_relationships = defaultdict(list), defaultdict(list)
    for entities, relationships in results:
        for k, v in entities.items():
            all_entities[k].extend(v)
        for k, v in relationships.items():
            all_relationships[k].extend(v)
    
    return dict(all_entities), dict(all_relationships)


# ================= Main Entry =================

def extract_entities_relations(
    chunks: List[Dict],
    global_config: Dict = None,
    mode: str = 'auto',
    domain: Optional[str] = None,
    context: Optional[str] = None
) -> Tuple[Dict, Dict]:
    """
    Main extraction function
    
    Args:
        chunks: List of chunks
        global_config: Config with 'llm_model_func'
        mode: 'static', 'dynamic', or 'auto'
        domain: Domain hint ('tech', 'medical', 'legal', 'business')
        context: Optional context string
    
    Returns:
        (entities_dict, relationships_dict)
    """
    if not chunks:
        logger.warning("⚠️ No chunks")
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
        
        logger.info(f"🔍 Extracting (mode={selected_mode}, chunks={len(chunks)})")
        
        # Extract
        entities, relationships = loop.run_until_complete(
            extract_async(chunks, llm_func, selected_mode, 16, context)
        )
        
        # Stats
        logger.info(
            f"✅ {sum(len(v) for v in entities.values())} entities, "
            f"{sum(len(v) for v in relationships.values())} relationships"
        )
        
        return entities, relationships
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return {}, {}


# ================= Utility =================

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
# ================= Export =================
__all__ = [
    'extract_entities_relations',
    'select_mode',
    'get_statistics',
    'STATIC_TYPES',
    'CATEGORIES'
]