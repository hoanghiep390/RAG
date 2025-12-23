# backend/retrieval/graph_retriever.py 
"""
🕸️ Graph Retriever - LightRAG Style
Simplified to work with keywords instead of relationship_type/category
"""
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class GraphContext:
    """Entity context with relationship info"""
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
    """Graph search with keyword-based scoring"""
    
    def __init__(self, mongo_storage):
        self.storage = mongo_storage
        self._graph_cache = None
    
    def _load_graph(self) -> Dict:
        """Lazy load graph"""
        if self._graph_cache is None:
            self._graph_cache = self.storage.get_graph()
            logger.info(f"📊 Đã tải: {len(self._graph_cache.get('nodes', []))} nodes, "
                       f"{len(self._graph_cache.get('links', []))} edges")
        return self._graph_cache
    
    def search(
        self,
        entity_names: List[str],
        k_hops: int = 1,
        max_neighbors: int = 5,
        min_strength: float = 0.0,
        filter_keywords: Optional[List[str]] = None
    ) -> List[GraphContext]:
        """
        Graph search with keyword filtering
        
        Args:
            entity_names: Starting entities
            k_hops: Traversal depth (1-2)
            max_neighbors: Max neighbors per entity
            min_strength: Min edge strength
            filter_keywords: Filter by keywords (e.g., ['development', 'management'])
        """
        if not entity_names:
            return []
        
        try:
            graph = self._load_graph()
            
            if not graph or not graph.get('nodes'):
                logger.warning("⚠️ Đồ thị rỗng")
                return []
            
            node_map = {n['id']: n for n in graph['nodes']}
            edges_by_source = defaultdict(list)
            
            for edge in graph.get('links', []):
                edges_by_source[edge['source']].append(edge)
            
            matched = self._find_entities(entity_names, node_map)
            
            if not matched:
                logger.info(f"❌ Không tìm thấy entities: {entity_names}")
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
                    filter_keywords=filter_keywords,
                    visited=visited
                )
                
                if context:
                    contexts.append(context)
                    visited.add(entity_name)
            
            contexts.sort(key=lambda x: x.score, reverse=True)
            return contexts
        
        except Exception as e:
            logger.error(f"❌ Tìm kiếm đồ thị thất bại: {e}")
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
    
    def _calculate_edge_score(self, edge: Dict, filter_keywords: Optional[List[str]] = None) -> float:
        """
        Calculate edge score based on strength and keywords
        
        Args:
            edge: Edge dict with 'strength', 'keywords', 'description'
            filter_keywords: Optional keywords to boost score
        
        Returns:
            Score (0.0-2.0+)
        """
        # Base score from strength
        score = edge.get('strength', 1.0)
        
        # Boost based on description quality
        description = edge.get('description', '')
        if len(description) > 50:
            score *= 1.2
        elif len(description) > 20:
            score *= 1.1
        
        # Boost based on keywords
        keywords = edge.get('keywords', '').lower()
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(',')]
            
            # More keywords = stronger relationship
            if len(keyword_list) >= 3:
                score *= 1.15
            elif len(keyword_list) >= 2:
                score *= 1.1
            
            # Boost if matches filter keywords
            if filter_keywords:
                filter_lower = [k.lower() for k in filter_keywords]
                matches = sum(1 for k in keyword_list if any(f in k or k in f for f in filter_lower))
                if matches > 0:
                    score *= (1.0 + 0.2 * matches)  
        
        return score
    
    def _build_context(
        self,
        entity_name: str,
        node_map: Dict,
        edges_by_source: Dict,
        k_hops: int,
        max_neighbors: int,
        min_strength: float,
        filter_keywords: Optional[List[str]],
        visited: Set[str]
    ) -> Optional[GraphContext]:
        """Build context with keyword-based scoring"""
        if entity_name not in node_map:
            return None
        
        node = node_map[entity_name]
        neighbors = []
        relationships = []
        
        # Score and rank edges
        scored_edges = []
        
        for edge in edges_by_source.get(entity_name, []):
            strength = edge.get('strength', 1.0)
            
            # Apply strength filter
            if strength < min_strength:
                continue
            
            target = edge['target']
            
            if target not in visited:
                # Calculate importance score
                importance_score = self._calculate_edge_score(edge, filter_keywords)
                scored_edges.append((importance_score, edge))
        
        # Sort by importance score (descending)
        scored_edges.sort(key=lambda x: x[0], reverse=True)
        
        # Build relationships from top-scored edges
        for importance_score, edge in scored_edges[:max_neighbors]:
            target = edge['target']
            keywords = edge.get('keywords', '')
            description = edge.get('description', '')
            strength = edge.get('strength', 1.0)
            
            neighbors.append(target)
            
            relationships.append({
                'target': target,
                'keywords': keywords,
                'description': description,
                'strength': strength,
                'importance_score': importance_score
            })
        
        # 2-hop expansion (with same scoring)
        if k_hops >= 2:
            second_hop_scored = []
            for neighbor in neighbors[:max_neighbors]:
                for edge in edges_by_source.get(neighbor, []):
                    if edge['target'] not in visited and edge['target'] != entity_name:
                        # Score 2nd hop edges
                        score = self._calculate_edge_score(edge, filter_keywords)
                        second_hop_scored.append((score, edge['target']))
            
            # Sort and add top 2nd hop neighbors
            second_hop_scored.sort(key=lambda x: x[0], reverse=True)
            second_hop_neighbors = [target for _, target in second_hop_scored[:max_neighbors]]
            neighbors.extend(second_hop_neighbors)
        
        neighbors = neighbors[:max_neighbors * 2]  # Allow more neighbors with 2-hop
        
        # Calculate context score based on relationship quality
        if relationships:
            avg_importance = sum(r['importance_score'] for r in relationships) / len(relationships)
            score = min(1.0, avg_importance * (len(neighbors) / 10.0))
        else:
            score = 0.0
        
        return GraphContext(
            entity_name=entity_name,
            entity_type=node.get('type', 'unknown'),
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
            logger.error(f"❌ Subgraph thất bại: {e}")
            return {'nodes': [], 'links': []}
    
    def format_context_text(self, contexts: List[GraphContext]) -> str:
        """
        Format context with keywords
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
                    keywords = rel.get('keywords', '')
                    target = rel['target']
                    desc = rel.get('description', '')
                    strength = rel.get('strength', 1.0)
                    
                    # Format: "OpenAI → GPT-4 [development, AI] (strength: 0.95): develops and maintains"
                    keyword_str = f"[{keywords}]" if keywords else ""
                    lines.append(
                        f"     • {ctx.entity_name} → {target} {keyword_str} "
                        f"(strength: {strength:.2f}): {desc[:80]}"
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