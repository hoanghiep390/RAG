# backend/core/graph_builder.py 
import networkx as nx
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Wrapper NetworkX cho knowledge graph"""
    
    def __init__(self):
        self.G = nx.DiGraph()
    
    def add_entity(self, entity_name: str, entity_type: str, description: str, 
                   source_id: str, **kwargs):
        """Thêm hoặc gộp entity node"""
        if self.G.has_node(entity_name):
            node = self.G.nodes[entity_name]
            
            # Gộp descriptions
            if description and description not in node.get('description', ''):
                existing_desc = node.get('description', '')
                node['description'] = f"{existing_desc}; {description}".strip('; ')
            
            # Gộp sources
            node['sources'] = node.get('sources', set()) | {source_id}
            # Node mới
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
        Thêm hoặc gộp relationship edge - kiểu LightRAG
        
        Tham số:
            source_entity: Tên entity nguồn
            target_entity: Tên entity đích
            keywords: Các keywords phân cách bằng dấu phẩy
            description: Mô tả relationship
            strength: Độ mạnh relationship (mặc định: 1.0)
            chunk_id: ID chunk nguồn
        """
        # Xác thực entities tồn tại
        if not self.has_node(source_entity):
            logger.warning(f" Không tìm thấy source entity: {source_entity}")
            return False
        
        if not self.has_node(target_entity):
            logger.warning(f" Không tìm thấy target entity: {target_entity}")
            return False
        
        # Kiểm tra vòng lặp tự thân
        if source_entity == target_entity:
            logger.warning(f" Không cho phép vòng lặp tự thân: {source_entity}")
            return False
        
            # Gộp edge hiện có
            edge = self.G.edges[source_entity, target_entity]
            
            # Gộp descriptions
            if description and description not in edge.get('description', ''):
                existing_desc = edge.get('description', '')
                edge['description'] = f"{existing_desc}; {description}".strip('; ')
            
            # Gộp keywords
            if keywords:
                existing_keywords = edge.get('keywords', '')
                if existing_keywords:
                    all_keywords = set(existing_keywords.split(',')) | set(keywords.split(','))
                    edge['keywords'] = ','.join(sorted(all_keywords))
                else:
                    edge['keywords'] = keywords
            
            # Tích lũy strength
            edge['strength'] = edge.get('strength', 0) + strength
            
            # Gộp chunks
            if chunk_id:
                edge['chunks'] = edge.get('chunks', set()) | {chunk_id}
            # Edge mới
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
        """Chuyển sang dict có thể serialize JSON"""
        data = nx.node_link_data(self.G, edges="links")
        
        # Chuyển sets sang lists
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
        """Lấy thống kê đồ thị"""
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
    Xây dựng knowledge graph - kiểu LightRAG
    
    Tham số:
        entities_dict: Dict của {entity_name: [entity_dicts]}
        relationships_dict: Dict của {(src, tgt): [relationship_dicts]}
        global_config: Config dict tùy chọn
        **kwargs: Tham số bổ sung (bỏ qua)
    
    Trả về:
        Instance KnowledgeGraph
    """
    kg = KnowledgeGraph()
    
    # Thêm entities
    for entity_name, nodes in entities_dict.items():
        for node in nodes:
            kg.add_entity(
                entity_name=entity_name,
                entity_type=node.get('entity_type', 'other'),
                description=node.get('description', ''),
                source_id=node.get('source_id', node.get('chunk_id', ''))
            )
    
    # Thêm relationships
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
        f" Đã xây dựng đồ thị: {stats['num_entities']} entities, "
        f"{stats['num_relationships']} relationships"
    )
    
    return kg