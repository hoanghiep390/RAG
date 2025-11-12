# backend/core/graph_builder.py
"""
✅ Graph Builder following LightRAG architecture
Includes: merge logic, entity type voting, LLM summarization
"""

import networkx as nx
import logging
import asyncio
from typing import Tuple, Dict, List, Any, Optional
from pathlib import Path
from collections import Counter
import time

from backend.utils.file_utils import save_to_json, load_from_json

logger = logging.getLogger(__name__)


GRAPH_FIELD_SEP = "; "  
MAX_DESCRIPTION_TOKENS = 500  


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(enc.encode(text))
    except:
        return int(len(text.split()) * 1.3)


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """
    Summarize long entity/relation descriptions using LLM
    Following LightRAG's summarization logic
    
    Args:
        entity_or_relation_name: Name of entity/relation
        description: Combined description (may contain GRAPH_FIELD_SEP)
        global_config: Config with llm_model_func, tokenizer
    
    Returns:
        Summarized description
    """
    llm_func = global_config.get("llm_model_func")
    if not llm_func:
        from backend.utils.llm_utils import call_llm_async
        llm_func = call_llm_async
    
    max_tokens = global_config.get("summary_max_tokens", MAX_DESCRIPTION_TOKENS)
    description_list = description.split(GRAPH_FIELD_SEP)
    
    prompt = f"""Summarize the following descriptions for "{entity_or_relation_name}":

Descriptions:
{chr(10).join(f"- {d}" for d in description_list)}

Provide a concise summary in under {max_tokens} tokens that captures the key facts.

Summary:"""
    
    try:
        summary = await llm_func(prompt, temperature=0.0, max_tokens=max_tokens)
        logger.debug(f"Summarized: {entity_or_relation_name}")
        return summary.strip()
    except Exception as e:
        logger.warning(f"Summarization failed for {entity_or_relation_name}: {e}")
        return description[:1000]

def vote_entity_type(entity_types: List[str]) -> str:
    """
    Vote for most common entity type
    Following LightRAG's voting logic
    
    Args:
        entity_types: List of entity types
    
    Returns:
        Most common entity type
    """
    if not entity_types:
        return "UNKNOWN"
    
    type_counts = Counter(entity_types)
    most_common = type_counts.most_common(1)[0][0]
    
    return most_common

def build_file_path(
    already_file_paths: List[str],
    nodes_data: List[Dict],
    entity_name: str
) -> str:
    """
    Build combined file path string
    
    Args:
        already_file_paths: Existing file paths
        nodes_data: New node data
        entity_name: Entity name (for logging)
    
    Returns:
        Combined file path string
    """
    all_paths = set(already_file_paths)
    
    for node in nodes_data:
        if node.get('file_path'):
            all_paths.add(node['file_path'])
    
    return GRAPH_FIELD_SEP.join(sorted(all_paths))


# ==================== MERGE NODES ====================
async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: List[Dict],
    knowledge_graph_inst: 'KnowledgeGraph',
    global_config: Dict,
) -> Dict:
    """
    Merge node data and upsert to graph
    Following LightRAG's merge logic
    
    Args:
        entity_name: Entity name
        nodes_data: List of entity data dicts
        knowledge_graph_inst: KnowledgeGraph instance
        global_config: Configuration
    
    Returns:
        Merged node data
    """
    already_entity_types = []
    already_source_ids = []
    already_description = []
    already_file_paths = []

    # Get existing node
    already_node = knowledge_graph_inst.get_node(entity_name)
    if already_node:
        already_entity_types.append(already_node.get('type', 'UNKNOWN'))
        
        if already_node.get('source_id'):
            already_source_ids.extend(
                already_node['source_id'].split(GRAPH_FIELD_SEP)
            )
        
        if already_node.get('file_path'):
            already_file_paths.extend(
                already_node['file_path'].split(GRAPH_FIELD_SEP)
            )
        
        if already_node.get('description'):
            already_description.append(already_node['description'])

    # Vote for entity type
    all_types = [nd['entity_type'] for nd in nodes_data] + already_entity_types
    entity_type = vote_entity_type(all_types)

    # Merge descriptions
    all_descriptions = [nd['description'] for nd in nodes_data] + already_description
    description = GRAPH_FIELD_SEP.join(sorted(set(all_descriptions)))

    # Merge source IDs
    all_source_ids = [nd['source_id'] for nd in nodes_data] + already_source_ids
    source_id = GRAPH_FIELD_SEP.join(set(all_source_ids))

    # Build file path
    file_path = build_file_path(already_file_paths, nodes_data, entity_name)

    # Check if summarization needed
    force_llm_summary = global_config.get("force_llm_summary_on_merge", 5)
    num_fragments = description.count(GRAPH_FIELD_SEP) + 1
    
    if num_fragments >= force_llm_summary:
        logger.info(f"LLM merge N: {entity_name} | {num_fragments} fragments")
        description = await _handle_entity_relation_summary(
            entity_name, description, global_config
        )
    elif num_fragments > 1:
        logger.debug(f"Merge N: {entity_name} | {num_fragments} fragments")

    # Create node data
    node_data = {
        'entity_id': entity_name,
        'type': entity_type,
        'description': description,
        'source_id': source_id,
        'file_path': file_path,
        'created_at': int(time.time()),
    }

    # Upsert to graph
    knowledge_graph_inst.add_entity(
        entity_name=entity_name,
        entity_type=entity_type,
        description=description,
        source_id=source_id,
        source_document=file_path.split(GRAPH_FIELD_SEP)[0] if file_path else "unknown"
    )

    node_data['entity_name'] = entity_name
    return node_data


# ==================== MERGE EDGES ====================
async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: List[Dict],
    knowledge_graph_inst: 'KnowledgeGraph',
    global_config: Dict,
) -> Optional[Dict]:
    """
    Merge edge data and upsert to graph
    Following LightRAG's merge logic
    
    Args:
        src_id: Source entity
        tgt_id: Target entity
        edges_data: List of relationship data dicts
        knowledge_graph_inst: KnowledgeGraph instance
        global_config: Configuration
    
    Returns:
        Merged edge data or None
    """
    if src_id == tgt_id:
        return None

    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_file_paths = []

    # Get existing edge
    already_edge = knowledge_graph_inst.get_edge(src_id, tgt_id)
    if already_edge:
        already_weights.append(already_edge.get('weight', 1.0))
        
        if already_edge.get('source_id'):
            already_source_ids.extend(
                already_edge['source_id'].split(GRAPH_FIELD_SEP)
            )
        
        if already_edge.get('file_path'):
            already_file_paths.extend(
                already_edge['file_path'].split(GRAPH_FIELD_SEP)
            )
        
        if already_edge.get('description'):
            already_description.append(already_edge['description'])
        
        if already_edge.get('keywords'):
            already_keywords.extend(
                already_edge['keywords'].split(GRAPH_FIELD_SEP)
            )

    # Merge weights (sum)
    all_weights = [ed['weight'] for ed in edges_data] + already_weights
    weight = sum(all_weights)

    # Merge descriptions
    all_descriptions = [ed['description'] for ed in edges_data if ed.get('description')] + already_description
    description = GRAPH_FIELD_SEP.join(sorted(set(all_descriptions)))

    # Merge keywords (deduplicate)
    all_keywords = set()
    for kw_str in already_keywords:
        if kw_str:
            all_keywords.update(k.strip() for k in kw_str.split(',') if k.strip())
    for edge in edges_data:
        if edge.get('keywords'):
            all_keywords.update(k.strip() for k in edge['keywords'].split(',') if k.strip())
    keywords = ','.join(sorted(all_keywords))

    # Merge source IDs
    all_source_ids = [ed['source_id'] for ed in edges_data if ed.get('source_id')] + already_source_ids
    source_id = GRAPH_FIELD_SEP.join(set(all_source_ids))

    # Build file path
    file_path = build_file_path(already_file_paths, edges_data, f"{src_id}-{tgt_id}")

    # Ensure source and target nodes exist
    for node_id in [src_id, tgt_id]:
        if not knowledge_graph_inst.has_node(node_id):
            knowledge_graph_inst.add_entity(
                entity_name=node_id,
                entity_type='UNKNOWN',
                description='',
                source_id=source_id,
                source_document=file_path.split(GRAPH_FIELD_SEP)[0] if file_path else "unknown"
            )

    # Check if summarization needed
    force_llm_summary = global_config.get("force_llm_summary_on_merge", 5)
    num_fragments = description.count(GRAPH_FIELD_SEP) + 1
    
    if num_fragments >= force_llm_summary:
        logger.info(f"LLM merge E: {src_id} - {tgt_id} | {num_fragments} fragments")
        description = await _handle_entity_relation_summary(
            f"({src_id}, {tgt_id})", description, global_config
        )
    elif num_fragments > 1:
        logger.debug(f"Merge E: {src_id} - {tgt_id} | {num_fragments} fragments")

    # Upsert to graph
    knowledge_graph_inst.add_relationship(
        source_entity=src_id,
        target_entity=tgt_id,
        description=description,
        strength=weight,
        chunk_id=None,
        source_document=file_path.split(GRAPH_FIELD_SEP)[0] if file_path else "unknown",
        keywords=keywords
    )

    edge_data = {
        'src_id': src_id,
        'tgt_id': tgt_id,
        'description': description,
        'keywords': keywords,
        'source_id': source_id,
        'file_path': file_path,
        'weight': weight,
        'created_at': int(time.time()),
    }

    return edge_data


# ==================== KNOWLEDGE GRAPH CLASS ====================
class KnowledgeGraph:
    """
    Knowledge Graph with LightRAG-compatible structure
    """
    
    def __init__(self, enable_summarization: bool = True):
        self.G = nx.DiGraph()
        self.enable_summarization = enable_summarization
    
    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        description: str,
        source_id: str,
        source_document: str,
        **kwargs
    ):
        """Add or merge entity"""
        if self.G.has_node(entity_name):
            # Merge
            node = self.G.nodes[entity_name]
            
            # Merge description
            if description and description not in node.get('description', ''):
                existing = node.get('description', '')
                if existing:
                    node['description'] = f"{existing}{GRAPH_FIELD_SEP}{description}"
                else:
                    node['description'] = description
            
            # Track sources
            sources = node.get('sources', set())
            sources.add(source_id)
            node['sources'] = sources
            
            docs = node.get('source_documents', set())
            docs.add(source_document)
            node['source_documents'] = docs
        else:
            # New node
            self.G.add_node(
                entity_name,
                type=entity_type,
                description=description,
                sources={source_id},
                source_documents={source_document},
                **kwargs
            )
    
    def add_relationship(
        self,
        source_entity: str,
        target_entity: str,
        description: str,
        strength: float = 1.0,
        chunk_id: Optional[str] = None,
        source_document: Optional[str] = None,
        **kwargs
    ):
        """Add or merge relationship"""
        if self.G.has_edge(source_entity, target_entity):
            # Merge
            edge = self.G.edges[source_entity, target_entity]
            
            if description and description not in edge.get('description', ''):
                existing = edge.get('description', '')
                if existing:
                    edge['description'] = f"{existing}{GRAPH_FIELD_SEP}{description}"
                else:
                    edge['description'] = description
            
            edge['strength'] = edge.get('strength', 1.0) + strength
            
            chunks = edge.get('chunks', set())
            if chunk_id:
                chunks.add(chunk_id)
            edge['chunks'] = chunks
            
            docs = edge.get('source_documents', set())
            if source_document:
                docs.add(source_document)
            edge['source_documents'] = docs
        else:
            # New edge
            self.G.add_edge(
                source_entity,
                target_entity,
                description=description,
                strength=strength,
                chunks={chunk_id} if chunk_id else set(),
                source_documents={source_document} if source_document else set(),
                **kwargs
            )
    
    def get_node(self, entity_name: str) -> Optional[Dict]:
        """Get node data"""
        if self.G.has_node(entity_name):
            return dict(self.G.nodes[entity_name])
        return None
    
    def has_node(self, entity_name: str) -> bool:
        """Check if node exists"""
        return self.G.has_node(entity_name)
    
    def get_edge(self, src: str, tgt: str) -> Optional[Dict]:
        """Get edge data"""
        if self.G.has_edge(src, tgt):
            return dict(self.G.edges[src, tgt])
        return None
    
    def has_edge(self, src: str, tgt: str) -> bool:
        """Check if edge exists"""
        return self.G.has_edge(src, tgt)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON"""
        data = nx.node_link_data(self.G)
        
        # Convert sets to lists
        for node in data['nodes']:
            for field in ['sources', 'source_documents']:
                if field in node and isinstance(node[field], set):
                    node[field] = list(node[field])
        
        for link in data['links']:
            for field in ['chunks', 'source_documents']:
                if field in link and isinstance(link[field], set):
                    link[field] = list(link[field])
        
        return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        type_counts = {}
        for _, data in self.G.nodes(data=True):
            t = data.get('type', 'UNKNOWN')
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            'num_entities': self.G.number_of_nodes(),
            'num_relationships': self.G.number_of_edges(),
            'entity_types': type_counts,
            'avg_degree': sum(dict(self.G.degree()).values()) / max(self.G.number_of_nodes(), 1),
            'density': nx.density(self.G),
        }


# ==================== BUILD GRAPH ====================
async def build_knowledge_graph_async(
    entities_dict: Dict[str, List[Dict]],
    relationships_dict: Dict[Tuple[str, str], List[Dict]],
    global_config: Dict,
    enable_summarization: bool = True,
) -> KnowledgeGraph:
    """
    Build knowledge graph with async merge
    Following LightRAG's architecture
    
    Args:
        entities_dict: {entity_name: [entity_data]}
        relationships_dict: {(src, tgt): [relation_data]}
        global_config: Configuration
        enable_summarization: Enable LLM summarization
    
    Returns:
        KnowledgeGraph instance
    """
    kg = KnowledgeGraph(enable_summarization=enable_summarization)
    
    logger.info("Building knowledge graph...")
    
    # Process all entities
    entity_tasks = []
    for entity_name, nodes_data in entities_dict.items():
        task = _merge_nodes_then_upsert(
            entity_name, nodes_data, kg, global_config
        )
        entity_tasks.append(task)
    
    if entity_tasks:
        await asyncio.gather(*entity_tasks)
        logger.info(f"Processed {len(entity_tasks)} entities")
    
    # Process all relationships
    edge_tasks = []
    for (src, tgt), edges_data in relationships_dict.items():
        task = _merge_edges_then_upsert(
            src, tgt, edges_data, kg, global_config
        )
        edge_tasks.append(task)
    
    if edge_tasks:
        results = await asyncio.gather(*edge_tasks)
        valid_edges = sum(1 for r in results if r is not None)
        logger.info(f"Processed {valid_edges} relationships")
    
    stats = kg.get_statistics()
    logger.info(f"Graph built: {stats['num_entities']} nodes, {stats['num_relationships']} edges")
    
    return kg


def build_knowledge_graph(
    entities_dict: Dict[str, List[Dict]],
    relationships_dict: Dict[Tuple[str, str], List[Dict]],
    global_config: Optional[Dict] = None,
    enable_summarization: bool = False,
    vector_db=None
) -> KnowledgeGraph:
    """
    Sync wrapper for building knowledge graph
    
    Args:
        entities_dict: Entity data
        relationships_dict: Relationship data
        global_config: Configuration
        enable_summarization: Enable LLM summarization
        vector_db: Unused (for compatibility)
    
    Returns:
        KnowledgeGraph instance
    """
    if not global_config:
        global_config = {
            "force_llm_summary_on_merge": 5,
            "summary_max_tokens": MAX_DESCRIPTION_TOKENS,
        }
    
    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Apply nest_asyncio if needed
    if loop.is_running():
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
    
    return loop.run_until_complete(
        build_knowledge_graph_async(
            entities_dict, relationships_dict, global_config, enable_summarization
        )
    )


# ==================== MERGE GRAPHS ====================
def merge_admin_graphs(user_id: str, enable_summarization: bool = False) -> Optional[KnowledgeGraph]:
    """
    Merge all user graphs into combined graph
    
    Args:
        user_id: User ID
        enable_summarization: Enable summarization
    
    Returns:
        Combined KnowledgeGraph or None
    """
    graphs_dir = Path(f"backend/data/{user_id}/graphs")
    if not graphs_dir.exists():
        logger.warning(f"No graphs dir for {user_id}")
        return None

    files = [f for f in graphs_dir.glob("*_graph.json") if f.name != "COMBINED_graph.json"]
    if not files:
        return None

    kg = KnowledgeGraph(enable_summarization=enable_summarization)
    logger.info(f"Merging {len(files)} graphs for {user_id}")

    for f in files:
        try:
            data = load_from_json(str(f))
            graph_data = data.get('graph', data)
            doc_name = f.stem.replace("_graph", "")
            
            for node in graph_data.get('nodes', []):
                src = list(node.get('sources', [])) or [f.stem]
                docs = list(node.get('source_documents', [])) or [doc_name]
                kg.add_entity(
                    entity_name=node['id'],
                    entity_type=node.get('type', 'UNKNOWN'),
                    description=node.get('description', ''),
                    source_id=src[0],
                    source_document=docs[0]
                )
            
            for link in graph_data.get('links', []):
                chunks = list(link.get('chunks', [])) or []
                docs = list(link.get('source_documents', [])) or [doc_name]
                kg.add_relationship(
                    source_entity=link['source'],
                    target_entity=link['target'],
                    description=link.get('description', ''),
                    strength=link.get('strength', 1.0),
                    chunk_id=chunks[0] if chunks else None,
                    source_document=docs[0]
                )
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")

    # Save combined
    out_path = graphs_dir / "COMBINED_graph.json"
    save_to_json(kg.to_dict(), str(out_path))
    logger.info(f"Saved COMBINED_graph.json: {kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges")
    
    return kg