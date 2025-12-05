# backend/core/graph_builder.py 
import networkx as nx
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """NetworkX wrapper with relationship_type support"""
    
    def __init__(self):
        self.G = nx.DiGraph()
    
    def add_entity(self, entity_name: str, entity_type: str, description: str, 
                   source_id: str, source_document: str, **kwargs):
        """Add/merge node"""
        if self.G.has_node(entity_name):
            node = self.G.nodes[entity_name]
            if description and description not in node.get('description', ''):
                node['description'] = f"{node.get('description', '')}; {description}".strip('; ')
            node['sources'] = node.get('sources', set()) | {source_id}
            node['source_documents'] = node.get('source_documents', set()) | {source_document}
        else:
            self.G.add_node(entity_name, type=entity_type, description=description,
                          sources={source_id}, source_documents={source_document}, **kwargs)
    
    def add_relationship(self, source_entity: str, target_entity: str, 
                        relationship_type: str = None, verb_phrase: str = None,
                        category: str = None, description: str = '',
                        strength: float = 1.0, chunk_id: str = None, 
                        source_document: str = None, **kwargs):
        """
        Add/merge edge with hybrid relationship support
        
        Args:
            relationship_type: Static type (e.g., 'DEVELOPS') or dynamic verb_phrase
            verb_phrase: Natural language verb (e.g., 'develops', 'is CEO of')
            category: Relationship category (e.g., 'FUNCTIONAL', 'HIERARCHICAL')
        """
        if self.G.has_edge(source_entity, target_entity):
            edge = self.G.edges[source_entity, target_entity]
            
            # Merge descriptions
            if description and description not in edge.get('description', ''):
                edge['description'] = f"{edge.get('description', '')}; {description}".strip('; ')
            
            # Accumulate strength
            edge['strength'] = edge.get('strength', 0) + strength
            
            # Merge metadata
            if chunk_id:
                edge['chunks'] = edge.get('chunks', set()) | {chunk_id}
            if source_document:
                edge['source_documents'] = edge.get('source_documents', set()) | {source_document}
            
            # Update type info (prioritize new data)
            if relationship_type:
                edge['relationship_type'] = relationship_type
            if verb_phrase:
                edge['verb_phrase'] = verb_phrase
            if category:
                edge['category'] = category
        else:
            # New edge
            self.G.add_edge(
                source_entity, target_entity,
                relationship_type=relationship_type or 'RELATED_TO',
                verb_phrase=verb_phrase or relationship_type.lower().replace('_', ' ') if relationship_type else 'related to',
                category=category or 'ASSOCIATIVE',
                description=description,
                strength=strength,
                chunks={chunk_id} if chunk_id else set(),
                source_documents={source_document} if source_document else set(),
                **kwargs
            )
    
    def get_node(self, name: str):
        return dict(self.G.nodes[name]) if self.G.has_node(name) else None
    
    def has_node(self, name: str):
        return self.G.has_node(name)
    
    def get_edge(self, src: str, tgt: str):
        return dict(self.G.edges[src, tgt]) if self.G.has_edge(src, tgt) else None
    
    def has_edge(self, src: str, tgt: str):
        return self.G.has_edge(src, tgt)
    
    def to_dict(self):
        """Convert to JSON with sets→lists"""
        data = nx.node_link_data(self.G, edges="links")
        
        for node in data.get('nodes', []):
            for field in ['sources', 'source_documents']:
                if field in node and isinstance(node[field], set):
                    node[field] = list(node[field])
        
        for link in data.get('links', []):
            for field in ['chunks', 'source_documents']:
                if field in link and isinstance(link[field], set):
                    link[field] = list(link[field])
        
        return data
    
    def get_statistics(self):
        """Enhanced stats with relationship types"""
        types = {}
        for _, d in self.G.nodes(data=True):
            t = d.get('type', 'UNKNOWN')
            types[t] = types.get(t, 0) + 1
        
        rel_types = {}
        rel_categories = {}
        for _, _, d in self.G.edges(data=True):
            rt = d.get('relationship_type', 'UNKNOWN')
            cat = d.get('category', 'UNKNOWN')
            rel_types[rt] = rel_types.get(rt, 0) + 1
            rel_categories[cat] = rel_categories.get(cat, 0) + 1
        
        return {
            'num_entities': self.G.number_of_nodes(),
            'num_relationships': self.G.number_of_edges(),
            'entity_types': types,
            'relationship_types': rel_types,
            'relationship_categories': rel_categories,
            'avg_degree': sum(dict(self.G.degree()).values()) / max(self.G.number_of_nodes(), 1),
            'density': nx.density(self.G)
        }


def build_knowledge_graph(entities_dict: Dict, relationships_dict: Dict, 
                         global_config: Dict = None, **kwargs) -> KnowledgeGraph:
    """Build graph with hybrid relationship support"""
    kg = KnowledgeGraph()
    
    # Add entities
    for entity_name, nodes in entities_dict.items():
        for node in nodes:
            kg.add_entity(
                entity_name=entity_name,
                entity_type=node['entity_type'],
                description=node.get('description', ''),
                source_id=node.get('source_id', ''),
                source_document=node.get('source_id', '')
            )
    
    # Add relationships with hybrid support
    for (src, tgt), edges in relationships_dict.items():
        for edge in edges:
            kg.add_relationship(
                source_entity=src,
                target_entity=tgt,
                relationship_type=edge.get('relationship_type', 'RELATED_TO'),
                verb_phrase=edge.get('verb_phrase', ''),
                category=edge.get('category', 'ASSOCIATIVE'),
                description=edge.get('description', ''),
                strength=edge.get('weight', 1.0),
                chunk_id=edge.get('chunk_id'),
                source_document=edge.get('chunk_id')
            )
    
    stats = kg.get_statistics()
    logger.info(
        f"✅ Graph built: {stats['num_entities']} entities, "
        f"{stats['num_relationships']} relationships "
        f"({len(stats['relationship_types'])} types, "
        f"{len(stats['relationship_categories'])} categories)"
    )
    
    return kg