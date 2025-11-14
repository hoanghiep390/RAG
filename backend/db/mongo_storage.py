# ==========================================
# backend/db/mongo_storage.py 
# ==========================================
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from backend.config import get_mongodb

class MongoStorage:
    """Simple MongoDB wrapper"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db = get_mongodb()
        self.documents = self.db['documents']
        self.chunks = self.db['chunks']
        self.entities = self.db['entities']
        self.relationships = self.db['relationships']
        self.graph_nodes = self.db['graph_nodes']
        self.graph_edges = self.db['graph_edges']
        
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes"""
        self.documents.create_index([('user_id', 1), ('doc_id', 1)], unique=True)
        self.chunks.create_index([('user_id', 1), ('doc_id', 1)])
        self.chunks.create_index([('chunk_id', 1)], unique=True)
        self.entities.create_index([('user_id', 1), ('entity_name', 1)])
        self.relationships.create_index([('user_id', 1), ('source_id', 1)])
        self.graph_nodes.create_index([('user_id', 1), ('node_id', 1)], unique=True)
        self.graph_edges.create_index([('user_id', 1), ('source', 1), ('target', 1)])
    
    def save_document(self, doc_id: str, filename: str, filepath: str, metadata: Dict = None) -> str:
        """Save doc metadata"""
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
        return str(result.inserted_id)
    
    def update_document_status(self, doc_id: str, status: str, stats: Dict = None):
        """Update status"""
        update_data = {'status': status, 'updated_at': datetime.now()}
        if stats:
            update_data['stats'] = stats
        self.documents.update_one(
            {'user_id': self.user_id, 'doc_id': doc_id},
            {'$set': update_data}
        )
    
    def get_document(self, doc_id: str):
        return self.documents.find_one({'user_id': self.user_id, 'doc_id': doc_id})
    
    def list_documents(self):
        return list(self.documents.find({'user_id': self.user_id}).sort('uploaded_at', -1))
    
    def save_chunks_bulk(self, doc_id: str, chunks: List[Dict]):
        """Bulk save chunks"""
        if not chunks:
            return
        chunk_docs = [
            {
                'user_id': self.user_id,
                'doc_id': doc_id,
                'chunk_id': c['chunk_id'],
                'content': c['content'],
                'tokens': c.get('tokens', 0),
                'order': c.get('order', 0),
                'file_path': c.get('file_path', ''),
                'file_type': c.get('file_type', ''),
                'created_at': datetime.now()
            }
            for c in chunks
        ]
        self.chunks.insert_many(chunk_docs, ordered=False)
    
    def save_entities_bulk(self, doc_id: str, entities_dict: Dict):
        """Bulk save entities"""
        if not entities_dict:
            return
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
                    'created_at': datetime.now()
                })
        if entity_docs:
            self.entities.insert_many(entity_docs, ordered=False)
    
    def save_relationships_bulk(self, doc_id: str, relationships_dict: Dict):
        """Bulk save relationships"""
        if not relationships_dict:
            return
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
                    'chunk_id': rel.get('chunk_id', ''),
                    'created_at': datetime.now()
                })
        if rel_docs:
            self.relationships.insert_many(rel_docs, ordered=False)
    
    def save_graph_bulk(self, graph_data: Dict):
        """Bulk save graph"""
        if graph_data.get('nodes'):
            for node in graph_data['nodes']:
                self.graph_nodes.update_one(
                    {'user_id': self.user_id, 'node_id': node['id']},
                    {'$set': {
                        'type': node.get('type', 'UNKNOWN'),
                        'description': node.get('description', ''),
                        'sources': list(node.get('sources', [])),
                        'source_documents': list(node.get('source_documents', [])),
                        'updated_at': datetime.now()
                    }},
                    upsert=True
                )
        
        if graph_data.get('links'):
            for link in graph_data['links']:
                self.graph_edges.update_one(
                    {'user_id': self.user_id, 'source': link['source'], 'target': link['target']},
                    {'$set': {
                        'description': link.get('description', ''),
                        'strength': link.get('strength', 1.0),
                        'chunks': list(link.get('chunks', [])),
                        'source_documents': list(link.get('source_documents', [])),
                        'updated_at': datetime.now()
                    }},
                    upsert=True
                )
    
    def save_document_complete(self, doc_id: str, filename: str, filepath: str,
                               chunks: List[Dict], entities: Dict = None,
                               relationships: Dict = None, graph: Dict = None, stats: Dict = None):
        """Save everything"""
        self.save_document(doc_id, filename, filepath)
        if chunks:
            self.save_chunks_bulk(doc_id, chunks)
        if entities:
            self.save_entities_bulk(doc_id, entities)
        if relationships:
            self.save_relationships_bulk(doc_id, relationships)
        if graph:
            self.save_graph_bulk(graph)
        self.update_document_status(doc_id, 'completed', stats)
    
    def delete_document_cascade(self, doc_id: str) -> Dict:
        """Delete everything for a doc"""
        stats = {'document': 0, 'chunks': 0, 'entities': 0, 'relationships': 0, 'files_deleted': []}
        
        doc = self.get_document(doc_id)
        if not doc:
            return stats
        
        stats['document'] = self.documents.delete_one({'user_id': self.user_id, 'doc_id': doc_id}).deleted_count
        stats['chunks'] = self.chunks.delete_many({'user_id': self.user_id, 'doc_id': doc_id}).deleted_count
        stats['entities'] = self.entities.delete_many({'user_id': self.user_id, 'doc_id': doc_id}).deleted_count
        stats['relationships'] = self.relationships.delete_many({'user_id': self.user_id, 'doc_id': doc_id}).deleted_count
        
        if doc.get('filepath'):
            filepath = Path(doc['filepath'])
            if filepath.exists():
                filepath.unlink()
                stats['files_deleted'].append(str(filepath))
        
        return stats
    
    def get_graph(self) -> Dict:
        """Get full graph"""
        nodes = list(self.graph_nodes.find({'user_id': self.user_id}))
        edges = list(self.graph_edges.find({'user_id': self.user_id}))
        
        return {
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
                    'strength': e.get('strength', 1.0),
                    'chunks': e.get('chunks', []),
                    'source_documents': e.get('source_documents', [])
                }
                for e in edges
            ]
        }
    
    def get_user_statistics(self) -> Dict:
        """Get stats"""
        return {
            'total_documents': self.documents.count_documents({'user_id': self.user_id}),
            'total_chunks': self.chunks.count_documents({'user_id': self.user_id}),
            'total_entities': self.entities.count_documents({'user_id': self.user_id}),
            'total_relationships': self.relationships.count_documents({'user_id': self.user_id}),
            'graph_nodes': self.graph_nodes.count_documents({'user_id': self.user_id}),
            'graph_edges': self.graph_edges.count_documents({'user_id': self.user_id})
        }