# backend/retrieval/graph_retriever.py
"""
🕸️ Graph Retriever 
Knowledge graph traversal & entity-based retrieval
"""
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Data Classes
@dataclass
class GraphContext:
    """Entity context từ knowledge graph"""
    entity_name: str
    entity_type: str
    description: str
    neighbors: List[str]  # Related entities
    relationships: List[Dict]  # Edge info
    score: float  
    hop_distance: int  
    
    def __repr__(self):
        return f"GraphContext({self.entity_name}, score={self.score:.3f}, neighbors={len(self.neighbors)})"

#  Main Retriever 
class GraphRetriever:
    """Simple knowledge graph search"""
    
    def __init__(self, mongo_storage):
        """
        Args:
            mongo_storage: MongoStorage instance from backend.db.mongo_storage
        """
        self.storage = mongo_storage
        self._graph_cache = None
    
    def _load_graph(self) -> Dict:
        """Lazy load graph from MongoDB"""
        if self._graph_cache is None:
            self._graph_cache = self.storage.get_graph()
            logger.info(f"Loaded graph: {len(self._graph_cache.get('nodes', []))} nodes, "
                       f"{len(self._graph_cache.get('links', []))} edges")
        return self._graph_cache
    
    def search(
        self,
        entity_names: List[str],
        k_hops: int = 1,
        max_neighbors: int = 5,
        min_strength: float = 0.0
    ) -> List[GraphContext]:
        """
        Graph search - Main entry point
        
        Args:
            entity_names: List of entity names to start from
            k_hops: Number of hops to traverse (1 or 2)
            max_neighbors: Max neighbors per entity
            min_strength: Minimum edge strength
        
        Returns:
            List of GraphContext, sorted by relevance
        """
        if not entity_names:
            return []
        
        try:
            graph = self._load_graph()
            
            if not graph or not graph.get('nodes'):
                logger.warning("Empty knowledge graph")
                return []
            
            # Build lookup structures
            node_map = {n['id']: n for n in graph['nodes']}
            edges_by_source = defaultdict(list)
            
            for edge in graph.get('links', []):
                edges_by_source[edge['source']].append(edge)
            
            # Find matching entities
            matched_entities = self._find_entities(entity_names, node_map)
            
            if not matched_entities:
                logger.info(f"No entities found for: {entity_names}")
                return []
            
            # Traverse graph
            contexts = []
            visited = set()
            
            for entity_name in matched_entities:
                if entity_name in visited:
                    continue
                
                context = self._build_context(
                    entity_name=entity_name,
                    node_map=node_map,
                    edges_by_source=edges_by_source,
                    k_hops=k_hops,
                    max_neighbors=max_neighbors,
                    min_strength=min_strength,
                    visited=visited
                )
                
                if context:
                    contexts.append(context)
                    visited.add(entity_name)
            
            # Sort by score
            contexts.sort(key=lambda x: x.score, reverse=True)
            
            return contexts
        
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    def _find_entities(self, query_entities: List[str], node_map: Dict) -> List[str]:
        """Find entities in graph (fuzzy matching)"""
        matched = []
        query_lower = [e.lower() for e in query_entities]
        
        for node_id in node_map.keys():
            node_lower = node_id.lower()
            
            # Exact match
            if node_lower in query_lower:
                matched.append(node_id)
                continue
            
            # Partial match
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
        visited: Set[str]
    ) -> Optional[GraphContext]:
        """Build context for one entity"""
        if entity_name not in node_map:
            return None
        
        node = node_map[entity_name]
        
        # Get neighbors (1-hop)
        neighbors = []
        relationships = []
        
        for edge in edges_by_source.get(entity_name, []):
            strength = edge.get('strength', 1.0)
            
            if strength < min_strength:
                continue
            
            target = edge['target']
            
            if target not in visited:
                neighbors.append(target)
                relationships.append({
                    'target': target,
                    'description': edge.get('description', ''),
                    'strength': strength,
                    'keywords': edge.get('keywords', '')
                })
        
        # If k_hops=2, get 2-hop neighbors
        if k_hops >= 2:
            second_hop = set()
            for neighbor in neighbors[:max_neighbors]:
                for edge in edges_by_source.get(neighbor, []):
                    if edge['target'] not in visited and edge['target'] != entity_name:
                        second_hop.add(edge['target'])
            neighbors.extend(list(second_hop)[:max_neighbors])
        
        # Limit neighbors
        neighbors = neighbors[:max_neighbors]
        relationships = relationships[:max_neighbors]
        
        # Calculate score (based on centrality)
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
    
    def get_subgraph(
        self,
        entity_names: List[str],
        k_hops: int = 2
    ) -> Dict:
        """
        Get subgraph around entities
        
        Returns:
            Dict with 'nodes' and 'links' for visualization
        """
        try:
            graph = self._load_graph()
            node_map = {n['id']: n for n in graph['nodes']}
            
            # BFS to find all nodes within k_hops
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
                    
                    # Add neighbors
                    for edge in graph.get('links', []):
                        if edge['source'] == node_id:
                            to_visit.add(edge['target'])
                
                current_hop += 1
            
            # Extract subgraph
            subgraph_nodes = [node_map[n] for n in visited if n in node_map]
            subgraph_links = [
                e for e in graph.get('links', [])
                if e['source'] in visited and e['target'] in visited
            ]
            
            return {
                'nodes': subgraph_nodes,
                'links': subgraph_links
            }
        
        except Exception as e:
            logger.error(f"Subgraph extraction failed: {e}")
            return {'nodes': [], 'links': []}
    
    def format_context_text(self, contexts: List[GraphContext]) -> str:
        """
        Format graph contexts as readable text for LLM
        
        Returns:
            Formatted string ready for prompt
        """
        if not contexts:
            return ""
        
        lines = ["=== Knowledge Graph Context ===\n"]
        
        for i, ctx in enumerate(contexts, 1):
            lines.append(f"{i}. **{ctx.entity_name}** ({ctx.entity_type})")
            
            if ctx.description:
                lines.append(f"   Description: {ctx.description}")
            
            if ctx.neighbors:
                lines.append(f"   Related to: {', '.join(ctx.neighbors[:5])}")
            
            if ctx.relationships:
                lines.append("   Key relationships:")
                for rel in ctx.relationships[:3]:
                    lines.append(f"     → {rel['target']}: {rel['description']}")
            
            lines.append("")  
        
        return "\n".join(lines)

# Convenience Function 
def search_graph(
    entity_names: List[str],
    mongo_storage,
    k_hops: int = 1,
    **kwargs
) -> List[GraphContext]:
    """
    Quick graph search function
    
    Usage:
        from backend.db.mongo_storage import MongoStorage
        from backend.retrieval.graph_retriever import search_graph
        
        storage = MongoStorage(user_id='admin_00000000')
        results = search_graph(['GPT-4', 'OpenAI'], storage, k_hops=2)
        
        for r in results:
            print(f"{r.entity_name}: {r.description[:100]}")
    """
    retriever = GraphRetriever(mongo_storage)
    return retriever.search(entity_names, k_hops=k_hops, **kwargs)
