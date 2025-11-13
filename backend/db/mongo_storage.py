# backend/db/mongo_storage.py
"""
MongoDB Storage Manager for LightRAG Data
Collections:
- documents: Metadata về tài liệu upload
- chunks: Text chunks từ documents
- entities: Extracted entities
- relationships: Extracted relationships
- graph_nodes: Knowledge graph nodes
- graph_edges: Knowledge graph edges
- embeddings: Vector embeddings
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
from backend.config import get_mongodb

logger = logging.getLogger(__name__)

class MongoStorage:
    """Unified storage manager for all LightRAG data"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db = get_mongodb()
        self.documents = self.db['documents']
        self.chunks = self.db['chunks']
        self.entities = self.db['entities']
        self.relationships = self.db['relationships']
        self.graph_nodes = self.db['graph_nodes']
        self.graph_edges = self.db['graph_edges']
        self.embeddings = self.db['embeddings']
        
        
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for efficient queries"""
        
        self.documents.create_index([('user_id', 1), ('doc_id', 1)])
        
        
        self.chunks.create_index([('user_id', 1), ('doc_id', 1)])
        self.chunks.create_index([('chunk_id', 1)])
        
        
        self.entities.create_index([('user_id', 1), ('entity_name', 1)])
        self.entities.create_index([('user_id', 1), ('entity_type', 1)])
        
        
        self.relationships.create_index([('user_id', 1), ('source_id', 1)])
        self.relationships.create_index([('user_id', 1), ('target_id', 1)])
        
        
        self.graph_nodes.create_index([('user_id', 1), ('node_id', 1)], unique=True)
        self.graph_edges.create_index([('user_id', 1), ('source', 1), ('target', 1)])
        
        
        self.embeddings.create_index([('user_id', 1), ('chunk_id', 1)])
        self.embeddings.create_index([('user_id', 1), ('doc_id', 1)])
    
    
    
    def save_document(self, doc_id: str, filename: str, filepath: str, 
                     metadata: Dict = None) -> str:
        """Save document metadata"""
        doc = {
            'user_id': self.user_id,
            'doc_id': doc_id,
            'filename': filename,
            'filepath': filepath,
            'uploaded_at': datetime.now(),
            'status': 'processing',
            'metadata': metadata or {}
        }
        
        result = self.documents.insert_one(doc)
        logger.info(f"📄 Saved document: {doc_id}")
        return str(result.inserted_id)
    
    def update_document_status(self, doc_id: str, status: str, stats: Dict = None):
        """Update document processing status"""
        update_data = {
            'status': status,
            'updated_at': datetime.now()
        }
        if stats:
            update_data['stats'] = stats
        
        self.documents.update_one(
            {'user_id': self.user_id, 'doc_id': doc_id},
            {'$set': update_data}
        )
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID"""
        return self.documents.find_one(
            {'user_id': self.user_id, 'doc_id': doc_id}
        )
    
    def list_documents(self) -> List[Dict]:
        """List all documents for user"""
        return list(self.documents.find(
            {'user_id': self.user_id}
        ).sort('uploaded_at', -1))
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document and all related data"""
        try:
            
            self.documents.delete_one({'user_id': self.user_id, 'doc_id': doc_id})          
            self.chunks.delete_many({'user_id': self.user_id, 'doc_id': doc_id})
            self.entities.delete_many({'user_id': self.user_id, 'doc_id': doc_id})
            self.relationships.delete_many({'user_id': self.user_id, 'doc_id': doc_id})
            self.embeddings.delete_many({'user_id': self.user_id, 'doc_id': doc_id})
                        
            logger.info(f"🗑️ Deleted document and related data: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting document: {e}")
            return False
    
    
    
    def save_chunks(self, doc_id: str, chunks: List[Dict]):
        """Save chunks for document"""
        chunk_docs = []
        for chunk in chunks:
            chunk_docs.append({
                'user_id': self.user_id,
                'doc_id': doc_id,
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content'],
                'tokens': chunk.get('tokens', 0),
                'order': chunk.get('order', 0),
                'hierarchy': chunk.get('hierarchy', []),
                'file_path': chunk.get('file_path', ''),
                'file_type': chunk.get('file_type', ''),
                'created_at': datetime.now()
            })
        
        if chunk_docs:
            self.chunks.insert_many(chunk_docs)
            logger.info(f"📦 Saved {len(chunk_docs)} chunks for {doc_id}")
    
    def get_chunks(self, doc_id: str) -> List[Dict]:
        """Get all chunks for document"""
        return list(self.chunks.find(
            {'user_id': self.user_id, 'doc_id': doc_id}
        ).sort('order', 1))
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get single chunk by ID"""
        return self.chunks.find_one({'chunk_id': chunk_id})
    
    
    
    def save_entities(self, doc_id: str, entities_dict: Dict[str, List[Dict]]):
        """Save extracted entities"""
        entity_docs = []
        for entity_name, entity_list in entities_dict.items():
            for entity in entity_list:
                entity_docs.append({
                    'user_id': self.user_id,
                    'doc_id': doc_id,
                    'entity_name': entity_name,
                    'entity_type': entity['entity_type'],
                    'description': entity.get('description', ''),
                    'source_id': entity.get('source_id', ''),
                    'chunk_id': entity.get('chunk_id', ''),
                    'file_path': entity.get('file_path', ''),
                    'created_at': datetime.now()
                })
        
        if entity_docs:
            self.entities.insert_many(entity_docs)
            logger.info(f"🏷️ Saved {len(entity_docs)} entities for {doc_id}")
    
    def get_entities(self, doc_id: str = None, entity_type: str = None) -> List[Dict]:
        """Get entities with optional filters"""
        query = {'user_id': self.user_id}
        if doc_id:
            query['doc_id'] = doc_id
        if entity_type:
            query['entity_type'] = entity_type
        
        return list(self.entities.find(query))
    
    def get_entity_by_name(self, entity_name: str) -> Optional[Dict]:
        """Get entity by name (first match)"""
        return self.entities.find_one({
            'user_id': self.user_id,
            'entity_name': entity_name
        })
    

    
    def save_relationships(self, doc_id: str, relationships_dict: Dict[tuple, List[Dict]]):
        """Save extracted relationships"""
        rel_docs = []
        for (source, target), rel_list in relationships_dict.items():
            for rel in rel_list:
                rel_docs.append({
                    'user_id': self.user_id,
                    'doc_id': doc_id,
                    'source_id': source,
                    'target_id': target,
                    'description': rel.get('description', ''),
                    'keywords': rel.get('keywords', ''),
                    'weight': rel.get('weight', 1.0),
                    'strength': rel.get('strength', 1.0),
                    'chunk_id': rel.get('chunk_id', ''),
                    'created_at': datetime.now()
                })
        
        if rel_docs:
            self.relationships.insert_many(rel_docs)
            logger.info(f"🔗 Saved {len(rel_docs)} relationships for {doc_id}")
    
    def get_relationships(self, doc_id: str = None, 
                         source: str = None, target: str = None) -> List[Dict]:
        """Get relationships with optional filters"""
        query = {'user_id': self.user_id}
        if doc_id:
            query['doc_id'] = doc_id
        if source:
            query['source_id'] = source
        if target:
            query['target_id'] = target
        
        return list(self.relationships.find(query))
    

    
    def save_graph(self, graph_data: Dict):
        """Save knowledge graph nodes and edges"""
        
        node_docs = []
        for node in graph_data.get('nodes', []):
            node_docs.append({
                'user_id': self.user_id,
                'node_id': node['id'],
                'type': node.get('type', 'UNKNOWN'),
                'description': node.get('description', ''),
                'sources': list(node.get('sources', set())),
                'source_documents': list(node.get('source_documents', set())),
                'updated_at': datetime.now()
            })
        
        
        for node_doc in node_docs:
            self.graph_nodes.update_one(
                {'user_id': self.user_id, 'node_id': node_doc['node_id']},
                {'$set': node_doc},
                upsert=True
            )
        
        
        edge_docs = []
        for link in graph_data.get('links', []):
            edge_docs.append({
                'user_id': self.user_id,
                'source': link['source'],
                'target': link['target'],
                'description': link.get('description', ''),
                'keywords': link.get('keywords', ''),
                'strength': link.get('strength', 1.0),
                'chunks': list(link.get('chunks', set())),
                'source_documents': list(link.get('source_documents', set())),
                'updated_at': datetime.now()
            })
        
        
        for edge_doc in edge_docs:
            self.graph_edges.update_one(
                {
                    'user_id': self.user_id,
                    'source': edge_doc['source'],
                    'target': edge_doc['target']
                },
                {'$set': edge_doc},
                upsert=True
            )
        
        logger.info(f"🕸️ Saved graph: {len(node_docs)} nodes, {len(edge_docs)} edges")
    
    def get_graph(self) -> Dict:
        """Get complete knowledge graph"""
        nodes = list(self.graph_nodes.find({'user_id': self.user_id}))
        edges = list(self.graph_edges.find({'user_id': self.user_id}))
        
        
        graph_data = {
            'nodes': [
                {
                    'id': n['node_id'],
                    'type': n.get('type', 'UNKNOWN'),
                    'description': n.get('description', ''),
                    'sources': n.get('sources', []),
                    'source_documents': n.get('source_documents', [])
                }
                for n in nodes
            ],
            'links': [
                {
                    'source': e['source'],
                    'target': e['target'],
                    'description': e.get('description', ''),
                    'keywords': e.get('keywords', ''),
                    'strength': e.get('strength', 1.0),
                    'chunks': e.get('chunks', []),
                    'source_documents': e.get('source_documents', [])
                }
                for e in edges
            ]
        }
        
        return graph_data
    
    def get_graph_statistics(self) -> Dict:
        """Get graph statistics"""
        nodes_count = self.graph_nodes.count_documents({'user_id': self.user_id})
        edges_count = self.graph_edges.count_documents({'user_id': self.user_id})
        
        
        pipeline = [
            {'$match': {'user_id': self.user_id}},
            {'$group': {'_id': '$type', 'count': {'$sum': 1}}}
        ]
        type_counts = {
            item['_id']: item['count'] 
            for item in self.graph_nodes.aggregate(pipeline)
        }
        
        return {
            'num_entities': nodes_count,
            'num_relationships': edges_count,
            'entity_types': type_counts
        }
    

    
    def save_embeddings(self, doc_id: str, embeddings: List[Dict]):
        """Save embeddings"""
        emb_docs = []
        for emb in embeddings:
            emb_docs.append({
                'user_id': self.user_id,
                'doc_id': doc_id,
                'chunk_id': emb['id'],
                'text': emb['text'],
                'embedding': emb['embedding'],  
                'entity_name': emb.get('entity_name'),
                'entity_type': emb.get('entity_type', 'CHUNK'),
                'hierarchy': emb.get('hierarchy', ''),
                'tokens': emb.get('tokens', 0),
                'created_at': datetime.now()
            })
        
        if emb_docs:
            self.embeddings.insert_many(emb_docs)
            logger.info(f"🧮 Saved {len(emb_docs)} embeddings for {doc_id}")
    
    def get_embeddings(self, doc_id: str = None, 
                      entity_type: str = None) -> List[Dict]:
        """Get embeddings with filters"""
        query = {'user_id': self.user_id}
        if doc_id:
            query['doc_id'] = doc_id
        if entity_type:
            query['entity_type'] = entity_type
        
        return list(self.embeddings.find(query))
    
    def search_similar_chunks(self, query_embedding: List[float], 
                             top_k: int = 5) -> List[Dict]:
        """
        Search similar chunks using vector similarity
        Note: For production, use MongoDB Atlas Vector Search or external vector DB
        This is a simple implementation using cosine similarity
        """
        import numpy as np
        
    
        all_embeddings = list(self.embeddings.find(
            {'user_id': self.user_id, 'entity_type': 'CHUNK'}
        ))
        
        if not all_embeddings:
            return []
        
        
        query_vec = np.array(query_embedding)
        
        results = []
        for emb_doc in all_embeddings:
            emb_vec = np.array(emb_doc['embedding'])
            similarity = np.dot(query_vec, emb_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)
            )
            
            results.append({
                'chunk_id': emb_doc['chunk_id'],
                'text': emb_doc['text'],
                'similarity': float(similarity),
                'doc_id': emb_doc['doc_id'],
                'hierarchy': emb_doc.get('hierarchy', ''),
            })
        
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    
    
    def get_user_statistics(self) -> Dict:
        """Get overall statistics for user"""
        return {
            'total_documents': self.documents.count_documents({'user_id': self.user_id}),
            'total_chunks': self.chunks.count_documents({'user_id': self.user_id}),
            'total_entities': self.entities.count_documents({'user_id': self.user_id}),
            'total_relationships': self.relationships.count_documents({'user_id': self.user_id}),
            'graph_nodes': self.graph_nodes.count_documents({'user_id': self.user_id}),
            'graph_edges': self.graph_edges.count_documents({'user_id': self.user_id}),
            'total_embeddings': self.embeddings.count_documents({'user_id': self.user_id})
        }