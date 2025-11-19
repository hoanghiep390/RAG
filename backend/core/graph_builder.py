# backend/core/graph_builder.py
import networkx as nx
from typing import Dict, List, Tuple
import time

class KnowledgeGraph:
    """Simple NetworkX wrapper"""
    
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
    
    def add_relationship(self, source_entity: str, target_entity: str, description: str,
                        strength: float = 1.0, chunk_id: str = None, source_document: str = None, **kwargs):
        """Add/merge edge"""
        if self.G.has_edge(source_entity, target_entity):
            edge = self.G.edges[source_entity, target_entity]
            if description and description not in edge.get('description', ''):
                edge['description'] = f"{edge.get('description', '')}; {description}".strip('; ')
            edge['strength'] = edge.get('strength', 0) + strength
            if chunk_id:
                edge['chunks'] = edge.get('chunks', set()) | {chunk_id}
            if source_document:
                edge['source_documents'] = edge.get('source_documents', set()) | {source_document}
        else:
            self.G.add_edge(source_entity, target_entity, description=description, strength=strength,
                          chunks={chunk_id} if chunk_id else set(),
                          source_documents={source_document} if source_document else set(), **kwargs)
    
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
            for field in ['sources', 'source_documents']:
                if field in node and isinstance(node[field], set):
                    node[field] = list(node[field])
        
        for link in data.get('links', []):
            for field in ['chunks', 'source_documents']:
                if field in link and isinstance(link[field], set):
                    link[field] = list(link[field])
        
        return data
    
    def get_statistics(self):
        types = {}
        for _, d in self.G.nodes(data=True):
            t = d.get('type', 'UNKNOWN')
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
    """Build graph từ entities và relationships"""
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
    
    # Add relationships
    for (src, tgt), edges in relationships_dict.items():
        for edge in edges:
            kg.add_relationship(
                source_entity=src,
                target_entity=tgt,
                description=edge.get('description', ''),
                strength=edge.get('weight', 1.0),
                chunk_id=edge.get('chunk_id'),
                source_document=edge.get('chunk_id')
            )
    
    return kg