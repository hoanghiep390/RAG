# backend/core/graph_builder.py 
import networkx as nx
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """NetworkX wrapper for knowledge graph - LightRAG style"""
    
    def __init__(self):
        self.G = nx.DiGraph()
    
    def add_entity(self, entity_name: str, entity_type: str, description: str, 
                   source_id: str, **kwargs):
        """Add or merge entity node"""
        if self.G.has_node(entity_name):
            node = self.G.nodes[entity_name]
            
            # Merge descriptions
            if description and description not in node.get('description', ''):
                existing_desc = node.get('description', '')
                node['description'] = f"{existing_desc}; {description}".strip('; ')
            
            # Merge sources
            node['sources'] = node.get('sources', set()) | {source_id}
        else:
            # New node
            self.G.add_node(
                entity_name, 
                type=entity_type, 
                description=description,
                sources={source_id},
                **kwargs
            )
    
    def add_relationship(self, source_entity: str, target_entity: str,
                        keywords: str = '', description: str = '',
                        strength: float = 1.0, chunk_id: str = None, **kwargs):
        """
        Add or merge relationship edge - LightRAG style
        
        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            keywords: Comma-separated keywords
            description: Relationship description
            strength: Relationship strength (default: 1.0)
            chunk_id: Source chunk ID
        """
        # Validate entities exist
        if not self.has_node(source_entity):
            logger.warning(f"⚠️ Không tìm thấy source entity: {source_entity}")
            return False
        
        if not self.has_node(target_entity):
            logger.warning(f"⚠️ Không tìm thấy target entity: {target_entity}")
            return False
        
        # Check self-loop
        if source_entity == target_entity:
            logger.warning(f"⚠️ Không cho phép vòng lặp tự thân: {source_entity}")
            return False
        
        if self.G.has_edge(source_entity, target_entity):
            # Merge existing edge
            edge = self.G.edges[source_entity, target_entity]
            
            # Merge descriptions
            if description and description not in edge.get('description', ''):
                existing_desc = edge.get('description', '')
                edge['description'] = f"{existing_desc}; {description}".strip('; ')
            
            # Merge keywords
            if keywords:
                existing_keywords = edge.get('keywords', '')
                if existing_keywords:
                    all_keywords = set(existing_keywords.split(',')) | set(keywords.split(','))
                    edge['keywords'] = ','.join(sorted(all_keywords))
                else:
                    edge['keywords'] = keywords
            
            # Accumulate strength
            edge['strength'] = edge.get('strength', 0) + strength
            
            # Merge chunks
            if chunk_id:
                edge['chunks'] = edge.get('chunks', set()) | {chunk_id}
        else:
            # New edge
            self.G.add_edge(
                source_entity, target_entity,
                keywords=keywords,
                description=description,
                strength=strength,
                chunks={chunk_id} if chunk_id else set(),
                **kwargs
            )
        
        return True
    
    def get_node(self, name: str):
        return dict(self.G.nodes[name]) if self.G.has_node(name) else None
    
    def has_node(self, name: str):
        return self.G.has_node(name)
    
    def get_edge(self, src: str, tgt: str):
        return dict(self.G.edges[src, tgt]) if self.G.has_edge(src, tgt) else None
    
    def has_edge(self, src: str, tgt: str):
        return self.G.has_edge(src, tgt)
    
    def to_dict(self):
        """Convert to JSON-serializable dict"""
        data = nx.node_link_data(self.G, edges="links")
        
        # Convert sets to lists
        for node in data.get('nodes', []):
            for field in ['sources']:
                if field in node and isinstance(node[field], set):
                    node[field] = list(node[field])
        
        for link in data.get('links', []):
            for field in ['chunks']:
                if field in link and isinstance(link[field], set):
                    link[field] = list(link[field])
        
        return data
    
    def get_statistics(self):
        """Get graph statistics"""
        types = {}
        for _, d in self.G.nodes(data=True):
            t = d.get('type', 'unknown')
            types[t] = types.get(t, 0) + 1
        
        return {
            'num_entities': self.G.number_of_nodes(),
            'num_relationships': self.G.number_of_edges(),
            'entity_types': types,
            'avg_degree': sum(dict(self.G.degree()).values()) / max(self.G.number_of_nodes(), 1),
            'density': nx.density(self.G)
        }


def build_knowledge_graph(entities_dict: Dict, relationships_dict: Dict, 
                         global_config: Dict = None, **kwargs) -> KnowledgeGraph:
    """
    Build knowledge graph - LightRAG style
    
    Args:
        entities_dict: Dict of {entity_name: [entity_dicts]}
        relationships_dict: Dict of {(src, tgt): [relationship_dicts]}
        global_config: Optional config dict
        **kwargs: Additional arguments (ignored)
    
    Returns:
        KnowledgeGraph instance
    """
    kg = KnowledgeGraph()
    
    # Add entities
    for entity_name, nodes in entities_dict.items():
        for node in nodes:
            kg.add_entity(
                entity_name=entity_name,
                entity_type=node.get('entity_type', 'other'),
                description=node.get('description', ''),
                source_id=node.get('source_id', node.get('chunk_id', ''))
            )
    
    # Add relationships
    for (src, tgt), edges in relationships_dict.items():
        for edge in edges:
            kg.add_relationship(
                source_entity=src,
                target_entity=tgt,
                keywords=edge.get('keywords', ''),
                description=edge.get('description', ''),
                strength=edge.get('weight', 1.0),
                chunk_id=edge.get('chunk_id', edge.get('source_id'))
            )
    
    stats = kg.get_statistics()
    logger.info(
        f"✅ Đã xây dựng đồ thị: {stats['num_entities']} entities, "
        f"{stats['num_relationships']} relationships"
    )
    
    return kg