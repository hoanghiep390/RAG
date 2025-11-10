# backend/core/graph_builder_v2.py
"""
✅ IMPROVED: Graph Builder with LLM Summarization & Entity Type Voting
Based on LightRAG original architecture
"""

import networkx as nx
import logging
import asyncio
from typing import Tuple, Dict, List, Any, Optional
from pathlib import Path
from collections import Counter

from backend.utils.file_utils import save_to_json, load_from_json
from backend.core.chunking import process_document_to_chunks
from backend.core.extraction import extract_entities_relations

logger = logging.getLogger(__name__)

# ==================== CONFIG ====================
MAX_DESCRIPTION_TOKENS = 500  # Max tokens before LLM summarization
GRAPH_FIELD_SEP = "; "  # Separator for merging descriptions


# ==================== TOKEN COUNTER ====================
def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        return len(enc.encode(text))
    except:
        # Fallback: rough estimate
        return len(text.split()) * 1.3


# ==================== LLM SUMMARIZATION ====================
async def summarize_description_async(
    entity_name: str,
    description: str,
    max_tokens: int = MAX_DESCRIPTION_TOKENS
) -> str:
    """
    ✅ NEW: Summarize long descriptions using LLM
    
    Args:
        entity_name: Name of entity
        description: Long description to summarize
        max_tokens: Target token count
    
    Returns:
        Summarized description
    """
    if count_tokens(description) <= max_tokens:
        return description
    
    try:
        from backend.utils.llm_utils import call_llm_async
        
        prompt = f"""Summarize the following description for entity "{entity_name}" in under {max_tokens} tokens.
Keep the most important facts. Be concise and factual.

Original description:
{description}

Summarized description:"""
        
        summary = await call_llm_async(
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens
        )
        
        logger.info(f"Summarized description for {entity_name}: {count_tokens(description)} → {count_tokens(summary)} tokens")
        return summary.strip()
        
    except Exception as e:
        logger.warning(f"Failed to summarize description for {entity_name}: {e}")
        # Fallback: truncate
        tokens = description.split()[:max_tokens]
        return " ".join(tokens)


def summarize_description_sync(entity_name: str, description: str, max_tokens: int = MAX_DESCRIPTION_TOKENS) -> str:
    """Sync wrapper for summarization"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(summarize_description_async(entity_name, description, max_tokens))


# ==================== ENTITY TYPE VOTING ====================
def vote_entity_type(entity_types: List[str]) -> str:
    """
    ✅ NEW: Vote for most common entity type when merging
    
    Args:
        entity_types: List of entity types from different sources
    
    Returns:
        Most common entity type
    """
    if not entity_types:
        return "UNKNOWN"
    
    # Count occurrences
    type_counts = Counter(entity_types)
    
    # Return most common
    most_common = type_counts.most_common(1)[0][0]
    
    logger.debug(f"Entity type voting: {dict(type_counts)} → {most_common}")
    return most_common


# ==================== IMPROVED KNOWLEDGE GRAPH ====================
class KnowledgeGraph:
    """
    ✅ IMPROVED: Knowledge Graph with LLM summarization and entity type voting
    """
    
    def __init__(self, enable_summarization: bool = True):
        self.G = nx.DiGraph()
        self.enable_summarization = enable_summarization
        self._pending_summaries = {}  # Cache for batch summarization
    
    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        description: str,
        source_id: str,
        source_document: str,
        **kwargs
    ):
        """
        ✅ IMPROVED: Add entity with type voting and description management
        """
        if self.G.has_node(entity_name):
            # ========== MERGE EXISTING ENTITY ==========
            node = self.G.nodes[entity_name]
            
            # 1. VOTE FOR ENTITY TYPE
            existing_type = node.get('type', 'UNKNOWN')
            types_to_vote = node.get('_type_history', [existing_type])
            types_to_vote.append(entity_type)
            voted_type = vote_entity_type(types_to_vote)
            node['type'] = voted_type
            node['_type_history'] = types_to_vote
            
            # 2. MERGE DESCRIPTIONS
            if description and description not in node.get('description', ''):
                existing_desc = node.get('description', '')
                
                # Merge with separator
                if existing_desc:
                    merged_desc = f"{existing_desc}{GRAPH_FIELD_SEP}{description}"
                else:
                    merged_desc = description
                
                node['description'] = merged_desc
                
                # Mark for summarization if too long
                if self.enable_summarization and count_tokens(merged_desc) > MAX_DESCRIPTION_TOKENS:
                    self._pending_summaries[entity_name] = merged_desc
            
            # 3. TRACK SOURCES
            sources = node.get('sources', set())
            sources.add(source_id)
            node['sources'] = sources
            
            # 4. TRACK DOCUMENTS
            docs = node.get('source_documents', set())
            docs.add(source_document)
            node['source_documents'] = docs
            
            # 5. UPDATE OTHER ATTRIBUTES
            for k, v in kwargs.items():
                if k not in node:
                    node[k] = v
        
        else:
            # ========== NEW ENTITY ==========
            self.G.add_node(
                entity_name,
                type=entity_type,
                description=description,
                sources={source_id},
                source_documents={source_document},
                _type_history=[entity_type],  # Track for voting
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
        """
        ✅ IMPROVED: Add relationship with description management
        """
        if self.G.has_edge(source_entity, target_entity):
            # ========== MERGE EXISTING RELATIONSHIP ==========
            edge = self.G.edges[source_entity, target_entity]
            
            # 1. MERGE DESCRIPTIONS
            if description and description not in edge.get('description', ''):
                existing_desc = edge.get('description', '')
                if existing_desc:
                    merged_desc = f"{existing_desc}{GRAPH_FIELD_SEP}{description}"
                else:
                    merged_desc = description
                edge['description'] = merged_desc
            
            # 2. AGGREGATE STRENGTH (take max)
            edge['strength'] = max(edge.get('strength', 1.0), strength)
            
            # 3. TRACK CHUNKS
            chunks = edge.get('chunks', set())
            if chunk_id:
                chunks.add(chunk_id)
            edge['chunks'] = chunks
            
            # 4. TRACK DOCUMENTS
            docs = edge.get('source_documents', set())
            if source_document:
                docs.add(source_document)
            edge['source_documents'] = docs
        
        else:
            # ========== NEW RELATIONSHIP ==========
            self.G.add_edge(
                source_entity,
                target_entity,
                description=description,
                strength=strength,
                chunks={chunk_id} if chunk_id else set(),
                source_documents={source_document} if source_document else set(),
                **kwargs
            )
    
    async def apply_pending_summaries_async(self):
        """
        ✅ NEW: Apply LLM summarization to long descriptions
        """
        if not self._pending_summaries:
            return
        
        logger.info(f"Summarizing {len(self._pending_summaries)} long descriptions...")
        
        for entity_name, long_desc in self._pending_summaries.items():
            if self.G.has_node(entity_name):
                summary = await summarize_description_async(entity_name, long_desc)
                self.G.nodes[entity_name]['description'] = summary
        
        self._pending_summaries.clear()
        logger.info("Summarization complete")
    
    def apply_pending_summaries(self):
        """Sync wrapper for summarization"""
        if not self._pending_summaries:
            return
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(self.apply_pending_summaries_async())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dict for JSON serialization"""
        data = nx.node_link_data(self.G)
        
        # Clean up internal fields and convert sets to lists
        for node in data['nodes']:
            # Remove internal fields
            node.pop('_type_history', None)
            
            # Convert sets to lists
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
        return {
            'num_entities': self.G.number_of_nodes(),
            'num_relationships': self.G.number_of_edges(),
            'entity_types': self._count_entity_types(),
            'avg_degree': sum(dict(self.G.degree()).values()) / max(self.G.number_of_nodes(), 1),
            'density': nx.density(self.G),
            'pending_summaries': len(self._pending_summaries)
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        """Count entities by type"""
        counts = {}
        for _, data in self.G.nodes(data=True):
            t = data.get('type', 'UNKNOWN')
            counts[t] = counts.get(t, 0) + 1
        return counts
    
    def get_isolated_entities(self) -> List[str]:
        """
        ✅ NEW: Find entities with no relationships (might be missing connections)
        """
        isolated = []
        for node in self.G.nodes():
            if self.G.degree(node) == 0:
                isolated.append(node)
        return isolated


# ==================== GRAPH BUILDING ====================
def build_knowledge_graph(
    entities_dict: Dict[str, List[Dict]],
    relationships_dict: Dict[str, List[Dict]],
    enable_summarization: bool = True,
    vector_db=None
) -> KnowledgeGraph:
    """
    ✅ IMPROVED: Build knowledge graph with summarization and type voting
    
    Args:
        entities_dict: {chunk_id: [entity1, entity2, ...]}
        relationships_dict: {chunk_id: [rel1, rel2, ...]}
        enable_summarization: Enable LLM summarization for long descriptions
        vector_db: Optional vector database reference
    
    Returns:
        KnowledgeGraph instance
    """
    kg = KnowledgeGraph(enable_summarization=enable_summarization)
    
    # Add entities
    for chunk_id, ents in entities_dict.items():
        doc = ents[0].get('source_document', Path(chunk_id).stem) if ents else 'unknown'
        
        for e in ents:
            e['source_document'] = e.get('source_document', doc)
            kg.add_entity(
                entity_name=e['entity_name'],
                entity_type=e['entity_type'],
                description=e['description'],
                source_id=e.get('source_id', chunk_id),
                source_document=e['source_document']
            )
    
    # Add relationships
    for chunk_id, rels in relationships_dict.items():
        doc = rels[0].get('source_document', Path(chunk_id).stem) if rels else 'unknown'
        
        for r in rels:
            r['source_document'] = r.get('source_document', doc)
            
            # Auto-create missing entities
            if not kg.G.has_node(r['source_id']):
                kg.add_entity(r['source_id'], 'UNKNOWN', '', chunk_id, doc)
            if not kg.G.has_node(r['target_id']):
                kg.add_entity(r['target_id'], 'UNKNOWN', '', chunk_id, doc)
            
            kg.add_relationship(
                source_entity=r['source_id'],
                target_entity=r['target_id'],
                description=r['description'],
                strength=r.get('strength', 1.0),
                chunk_id=chunk_id,
                source_document=doc
            )
    
    # Apply summarization if enabled
    if enable_summarization:
        kg.apply_pending_summaries()
    
    return kg


# ==================== PROCESS FILE ====================
def process_file(
    filepath: str,
    global_config: Optional[Dict] = None,
    enable_summarization: bool = True,
    vector_db=None
) -> Dict:
    """
    ✅ IMPROVED: Process file with summarization
    """
    logger.info(f"Processing: {filepath}")
    
    chunks = process_document_to_chunks(filepath)
    entities, relationships = extract_entities_relations(chunks, global_config)
    kg = build_knowledge_graph(entities, relationships, enable_summarization, vector_db)
    
    result = {
        'graph': kg.to_dict(),
        'statistics': kg.get_statistics(),
        'metadata': {
            'source_file': filepath,
            'num_chunks': len(chunks),
            'num_entity_chunks': len(entities),
            'num_relationship_chunks': len(relationships),
            'summarization_enabled': enable_summarization
        }
    }
    
    # Save graph
    safe_name = Path(filepath).stem
    output_dir = Path(f"backend/data/{Path(filepath).parent.name}/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{safe_name}_graph.json"
    save_to_json(result, str(output))
    
    return result


# ==================== MERGE GRAPHS ====================
def merge_admin_graphs(user_id: str, enable_summarization: bool = True) -> Optional[KnowledgeGraph]:
    """
    ✅ IMPROVED: Merge graphs with summarization
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

    # Apply summarization
    if enable_summarization:
        kg.apply_pending_summaries()

    # Save combined graph
    out_path = graphs_dir / "COMBINED_graph.json"
    save_to_json(kg.to_dict(), str(out_path))
    logger.info(f"COMBINED_graph.json saved: {kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges")
    
    return kg


# ==================== GLEANING PROCESS ====================
async def gleaning_process(
    entities_dict: Dict[str, List[Dict]],
    relationships_dict: Dict[str, List[Dict]],
    chunks: List[Dict],
    knowledge_graph: KnowledgeGraph,
    max_iterations: int = 2
) -> Tuple[Dict, Dict]:
    """
    ✅ IMPROVED: Refinement with better prompts
    """
    from backend.utils.llm_utils import call_llm_async
    from backend.core.extraction import parse_extraction_result, process_extraction_result

    current_entities = entities_dict.copy()
    current_relations = relationships_dict.copy()

    for it in range(max_iterations):
        logger.info(f"[Gleaning] Iteration {it+1}/{max_iterations}")

        # Build entity summary
        entities_text = "\n".join([
            f"- {e['entity_name']} ({e['entity_type']}): {e.get('description', '')}"
            for ents in current_entities.values() for e in ents
        ]) or "None"

        # Build relationship summary
        relations_text = "\n".join([
            f"- {r['source_id']} → {r['target_id']}: {r.get('description', '')} (strength: {r['strength']})"
            for rels in current_relations.values() for r in rels
        ]) or "None"

        prompt = f"""You are a knowledge graph expert. Review and improve this graph.

Current Entities:
{entities_text}

Current Relationships:
{relations_text}

Tasks:
1. Fix incorrect entity types
2. Standardize entity names (e.g., "Apple Inc." → "Apple")
3. Add missing entities or relationships
4. Remove duplicates

Output format:
entity<|>name<|>type<|>description##
relationship<|>source<|>target<|>description<|>strength##

Output:
"""

        try:
            response = await call_llm_async(prompt, temperature=0.0, max_tokens=1500)
            records = parse_extraction_result(response)
            new_ents, new_rels = process_extraction_result(records, chunk_id=f"gleaning_{it}")

            # Merge new results
            for chunk_id, ents in new_ents.items():
                current_entities.setdefault(chunk_id, []).extend(ents)
            for chunk_id, rels in new_rels.items():
                current_relations.setdefault(chunk_id, []).extend(rels)

            if len(new_ents) + len(new_rels) == 0:
                logger.info("[Gleaning] No improvement. Stopping.")
                break
        except Exception as e:
            logger.error(f"[Gleaning] Error in iteration {it+1}: {e}")
            break

    return current_entities, current_relations


# ==================== DIAGNOSTICS ====================
def diagnose_graph(kg: KnowledgeGraph) -> Dict[str, Any]:
    """
    ✅ NEW: Diagnose graph quality
    
    Returns:
        Dict with potential issues
    """
    issues = {
        'isolated_entities': kg.get_isolated_entities(),
        'entities_needing_summary': [
            node for node, data in kg.G.nodes(data=True)
            if count_tokens(data.get('description', '')) > MAX_DESCRIPTION_TOKENS
        ],
        'weak_relationships': [
            (u, v) for u, v, data in kg.G.edges(data=True)
            if data.get('strength', 1.0) < 0.3
        ],
        'unknown_entities': [
            node for node, data in kg.G.nodes(data=True)
            if data.get('type') == 'UNKNOWN'
        ]
    }
    
    return issues