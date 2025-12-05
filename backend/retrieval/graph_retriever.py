# backend/retrieval/graph_retriever.py 
"""
🕸️ Graph Retriever - With full relationship metadata
"""
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class GraphContext:
    """Entity context with enhanced relationship info"""
    entity_name: str
    entity_type: str
    description: str
    neighbors: List[str]
    relationships: List[Dict]  
    score: float
    hop_distance: int
    
    def __repr__(self):
        return f"GraphContext({self.entity_name}, score={self.score:.3f}, neighbors={len(self.neighbors)})"

class GraphRetriever:
    """Enhanced graph search with relationship metadata"""
    
    def __init__(self, mongo_storage):
        self.storage = mongo_storage
        self._graph_cache = None
    
    def _load_graph(self) -> Dict:
        """Lazy load graph"""
        if self._graph_cache is None:
            self._graph_cache = self.storage.get_graph()
            logger.info(f"📊 Loaded: {len(self._graph_cache.get('nodes', []))} nodes, "
                       f"{len(self._graph_cache.get('links', []))} edges")
        return self._graph_cache
    
    def search(
        self,
        entity_names: List[str],
        k_hops: int = 1,
        max_neighbors: int = 5,
        min_strength: float = 0.0,
        filter_category: Optional[List[str]] = None,
        filter_rel_type: Optional[List[str]] = None
    ) -> List[GraphContext]:
        """
        Enhanced graph search with relationship filtering
        
        Args:
            entity_names: Starting entities
            k_hops: Traversal depth (1-2)
            max_neighbors: Max neighbors per entity
            min_strength: Min edge strength
            filter_category: Filter by categories (e.g., ['FUNCTIONAL', 'HIERARCHICAL'])
            filter_rel_type: Filter by types (e.g., ['DEVELOPS', 'MANAGES'])
        """
        if not entity_names:
            return []
        
        try:
            graph = self._load_graph()
            
            if not graph or not graph.get('nodes'):
                logger.warning("⚠️ Empty graph")
                return []
            
            node_map = {n['id']: n for n in graph['nodes']}
            edges_by_source = defaultdict(list)
            
            for edge in graph.get('links', []):
                edges_by_source[edge['source']].append(edge)
            
            matched = self._find_entities(entity_names, node_map)
            
            if not matched:
                logger.info(f"❌ No entities found: {entity_names}")
                return []
            
            contexts = []
            visited = set()
            
            for entity_name in matched:
                if entity_name in visited:
                    continue
                
                context = self._build_context(
                    entity_name=entity_name,
                    node_map=node_map,
                    edges_by_source=edges_by_source,
                    k_hops=k_hops,
                    max_neighbors=max_neighbors,
                    min_strength=min_strength,
                    filter_category=filter_category,
                    filter_rel_type=filter_rel_type,
                    visited=visited
                )
                
                if context:
                    contexts.append(context)
                    visited.add(entity_name)
            
            contexts.sort(key=lambda x: x.score, reverse=True)
            return contexts
        
        except Exception as e:
            logger.error(f"❌ Graph search failed: {e}")
            return []
    
    def _find_entities(self, query_entities: List[str], node_map: Dict) -> List[str]:
        """Fuzzy entity matching"""
        matched = []
        query_lower = [e.lower() for e in query_entities]
        
        for node_id in node_map.keys():
            node_lower = node_id.lower()
            
            if node_lower in query_lower:
                matched.append(node_id)
                continue
            
            for q in query_lower:
                if q in node_lower or node_lower in q:
                    matched.append(node_id)
                    break
        
        return matched
    
    def _build_context(
        self,
        entity_name: str,
        node_map: Dict,
        edges_by_source: Dict,
        k_hops: int,
        max_neighbors: int,
        min_strength: float,
        filter_category: Optional[List[str]],
        filter_rel_type: Optional[List[str]],
        visited: Set[str]
    ) -> Optional[GraphContext]:
        """Build enhanced context with full relationship metadata"""
        if entity_name not in node_map:
            return None
        
        node = node_map[entity_name]
        neighbors = []
        relationships = []
        
        for edge in edges_by_source.get(entity_name, []):
            strength = edge.get('strength', 1.0)
            rel_type = edge.get('relationship_type', 'UNKNOWN')
            category = edge.get('category', 'UNKNOWN')
            
            # Apply filters
            if strength < min_strength:
                continue
            if filter_category and category not in filter_category:
                continue
            if filter_rel_type and rel_type not in filter_rel_type:
                continue
            
            target = edge['target']
            
            if target not in visited:
                neighbors.append(target)
                
                # ✅ ENHANCED: Include full relationship metadata
                relationships.append({
                    'target': target,
                    'relationship_type': rel_type,
                    'verb_phrase': edge.get('verb_phrase', rel_type.lower()),
                    'category': category,
                    'description': edge.get('description', ''),
                    'strength': strength,
                    'keywords': edge.get('keywords', '')
                })
        
        # 2-hop expansion
        if k_hops >= 2:
            second_hop = set()
            for neighbor in neighbors[:max_neighbors]:
                for edge in edges_by_source.get(neighbor, []):
                    if edge['target'] not in visited and edge['target'] != entity_name:
                        second_hop.add(edge['target'])
            neighbors.extend(list(second_hop)[:max_neighbors])
        
        neighbors = neighbors[:max_neighbors]
        relationships = relationships[:max_neighbors]
        
        score = min(1.0, len(neighbors) / 10.0)
        
        return GraphContext(
            entity_name=entity_name,
            entity_type=node.get('type', 'UNKNOWN'),
            description=node.get('description', ''),
            neighbors=neighbors,
            relationships=relationships,
            score=score,
            hop_distance=1
        )
    
    def get_subgraph(self, entity_names: List[str], k_hops: int = 2) -> Dict:
        """Extract subgraph"""
        try:
            graph = self._load_graph()
            node_map = {n['id']: n for n in graph['nodes']}
            
            to_visit = set(entity_names)
            visited = set()
            current_hop = 0
            
            while to_visit and current_hop < k_hops:
                current_level = to_visit.copy()
                to_visit.clear()
                
                for node_id in current_level:
                    if node_id in visited or node_id not in node_map:
                        continue
                    
                    visited.add(node_id)
                    
                    for edge in graph.get('links', []):
                        if edge['source'] == node_id:
                            to_visit.add(edge['target'])
                
                current_hop += 1
            
            subgraph_nodes = [node_map[n] for n in visited if n in node_map]
            subgraph_links = [
                e for e in graph.get('links', [])
                if e['source'] in visited and e['target'] in visited
            ]
            
            return {'nodes': subgraph_nodes, 'links': subgraph_links}
        
        except Exception as e:
            logger.error(f"❌ Subgraph failed: {e}")
            return {'nodes': [], 'links': []}
    
    def format_context_text(self, contexts: List[GraphContext]) -> str:
        """
        ✅ ENHANCED: Format with relationship types & categories
        """
        if not contexts:
            return ""
        
        lines = ["=== 🕸️ Knowledge Graph Context ===\n"]
        
        for i, ctx in enumerate(contexts, 1):
            lines.append(f"{i}. **{ctx.entity_name}** ({ctx.entity_type})")
            
            if ctx.description:
                lines.append(f"   📝 {ctx.description}")
            
            if ctx.relationships:
                lines.append("   🔗 Relationships:")
                for rel in ctx.relationships[:5]:  # Top 5
                    rel_type = rel['relationship_type']
                    verb = rel['verb_phrase']
                    category = rel['category']
                    target = rel['target']
                    desc = rel.get('description', '')
                    
                    # Format: "OpenAI DEVELOPS (FUNCTIONAL) GPT-4: develops and maintains the model"
                    lines.append(
                        f"     • {ctx.entity_name} **{rel_type}** ({category}) {target}: "
                        f"\"{verb}\" - {desc[:80]}"
                    )
            
            if ctx.neighbors:
                extra = len(ctx.neighbors) - len(ctx.relationships)
                if extra > 0:
                    lines.append(f"   ➕ {extra} more connections: {', '.join(ctx.neighbors[:3])}...")
            
            lines.append("")
        
        return "\n".join(lines)

# Convenience function
def search_graph(
    entity_names: List[str],
    mongo_storage,
    k_hops: int = 1,
    **kwargs
) -> List[GraphContext]:
    """Quick search"""
    retriever = GraphRetriever(mongo_storage)
    return retriever.search(entity_names, k_hops=k_hops, **kwargs)