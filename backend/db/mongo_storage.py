# ==========================================
# backend/db/mongo_storage.py 
# ==========================================
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from backend.config import get_mongodb
import logging

logger = logging.getLogger(__name__)

class MongoStorage:
    """Enhanced MongoDB storage with better error handling"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        try:
            self.db = get_mongodb()
            self.documents = self.db['documents']
            self.chunks = self.db['chunks']
            self.entities = self.db['entities']
            self.relationships = self.db['relationships']
            self.graph_nodes = self.db['graph_nodes']
            self.graph_edges = self.db['graph_edges']
            
            self._create_indexes()
            logger.info(f"✅ MongoStorage initialized for user: {user_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MongoStorage: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for better performance"""
        try:
            self.documents.create_index([('user_id', 1), ('doc_id', 1)], unique=True)
            self.chunks.create_index([('user_id', 1), ('doc_id', 1)])
            self.chunks.create_index([('chunk_id', 1)], unique=True)
            self.entities.create_index([('user_id', 1), ('entity_name', 1)])
            self.relationships.create_index([('user_id', 1), ('source_id', 1)])
            self.graph_nodes.create_index([('user_id', 1), ('node_id', 1)], unique=True)
            self.graph_edges.create_index([('user_id', 1), ('source', 1), ('target', 1)])
            logger.debug("✅ MongoDB indexes created")
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")
    
    def save_document(self, doc_id: str, filename: str, filepath: str, metadata: Dict = None) -> str:
        """Save document metadata"""
        try:
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
            logger.info(f"✅ Saved document: {filename}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"❌ Failed to save document {filename}: {e}")
            raise
    
    def update_document_status(self, doc_id: str, status: str, stats: Dict = None):
        """Update document status"""
        try:
            update_data = {'status': status, 'updated_at': datetime.now()}
            if stats:
                update_data['stats'] = stats
            result = self.documents.update_one(
                {'user_id': self.user_id, 'doc_id': doc_id},
                {'$set': update_data}
            )
            if result.modified_count > 0:
                logger.info(f"✅ Updated document status: {doc_id} -> {status}")
            return result.modified_count
        except Exception as e:
            logger.error(f"❌ Failed to update document status: {e}")
            raise
    
    def get_document(self, doc_id: str):
        """Get document by ID"""
        try:
            return self.documents.find_one({'user_id': self.user_id, 'doc_id': doc_id})
        except Exception as e:
            logger.error(f"❌ Failed to get document {doc_id}: {e}")
            return None
    
    def list_documents(self):
        """List all documents for user"""
        try:
            return list(self.documents.find({'user_id': self.user_id}).sort('uploaded_at', -1))
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            return []
    
    def save_chunks_bulk(self, doc_id: str, chunks: List[Dict]):
        """Bulk save chunks"""
        if not chunks:
            return 0
        
        try:
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
                    'section': c.get('section', 'ROOT'),
                    'created_at': datetime.now()
                }
                for c in chunks
            ]
            result = self.chunks.insert_many(chunk_docs, ordered=False)
            logger.info(f"✅ Saved {len(result.inserted_ids)} chunks")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"❌ Failed to save chunks: {e}")
            return 0
    
    def save_entities_bulk(self, doc_id: str, entities_dict: Dict):
        """Bulk save entities"""
        if not entities_dict:
            return 0
        
        try:
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
                result = self.entities.insert_many(entity_docs, ordered=False)
                logger.info(f"✅ Saved {len(result.inserted_ids)} entities")
                return len(result.inserted_ids)
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to save entities: {e}")
            return 0
    
    def save_relationships_bulk(self, doc_id: str, relationships_dict: Dict):
        """Bulk save relationships"""
        if not relationships_dict:
            return 0
        
        try:
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
                result = self.relationships.insert_many(rel_docs, ordered=False)
                logger.info(f"✅ Saved {len(result.inserted_ids)} relationships")
                return len(result.inserted_ids)
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to save relationships: {e}")
            return 0
    
    def save_graph_bulk(self, graph_data: Dict):
        """Bulk save graph (upsert nodes and edges)"""
        nodes_saved = 0
        edges_saved = 0
        
        try:
            # Save nodes
            if graph_data.get('nodes'):
                for node in graph_data['nodes']:
                    try:
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
                        nodes_saved += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save node {node.get('id')}: {e}")
            
            # Save edges
            if graph_data.get('links'):
                for link in graph_data['links']:
                    try:
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
                        edges_saved += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save edge {link.get('source')}->{link.get('target')}: {e}")
            
            logger.info(f"✅ Saved graph: {nodes_saved} nodes, {edges_saved} edges")
            return {'nodes': nodes_saved, 'edges': edges_saved}
        
        except Exception as e:
            logger.error(f"❌ Failed to save graph: {e}")
            return {'nodes': nodes_saved, 'edges': edges_saved}
    
    def save_document_complete(self, doc_id: str, filename: str, filepath: str,
                               chunks: List[Dict], entities: Dict = None,
                               relationships: Dict = None, graph: Dict = None, stats: Dict = None):
        """Complete save operation for a document"""
        try:
            # Save document metadata
            self.save_document(doc_id, filename, filepath)
            
            # Save chunks
            if chunks:
                self.save_chunks_bulk(doc_id, chunks)
            
            # Save entities
            if entities:
                self.save_entities_bulk(doc_id, entities)
            
            # Save relationships
            if relationships:
                self.save_relationships_bulk(doc_id, relationships)
            
            # Save graph
            if graph:
                self.save_graph_bulk(graph)
            
            # Update status
            self.update_document_status(doc_id, 'completed', stats)
            
            logger.info(f"✅ Complete save for document: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save document completely: {e}")
            self.update_document_status(doc_id, 'failed', {'error': str(e)})
            return False
    
    def delete_document_cascade(self, doc_id: str) -> Dict:
        """Cascade delete everything for a document"""
        stats = {
            'document': 0, 
            'chunks': 0, 
            'entities': 0, 
            'relationships': 0, 
            'files_deleted': []
        }
        
        try:
            # Get document info
            doc = self.get_document(doc_id)
            if not doc:
                logger.warning(f"⚠️ Document {doc_id} not found")
                return stats
            
            # Delete from collections
            stats['document'] = self.documents.delete_one(
                {'user_id': self.user_id, 'doc_id': doc_id}
            ).deleted_count
            
            stats['chunks'] = self.chunks.delete_many(
                {'user_id': self.user_id, 'doc_id': doc_id}
            ).deleted_count
            
            stats['entities'] = self.entities.delete_many(
                {'user_id': self.user_id, 'doc_id': doc_id}
            ).deleted_count
            
            stats['relationships'] = self.relationships.delete_many(
                {'user_id': self.user_id, 'doc_id': doc_id}
            ).deleted_count
            
            # Delete physical file
            if doc.get('filepath'):
                filepath = Path(doc['filepath'])
                if filepath.exists():
                    try:
                        filepath.unlink()
                        stats['files_deleted'].append(str(filepath))
                        logger.info(f"✅ Deleted file: {filepath}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to delete file {filepath}: {e}")
            
            logger.info(f"✅ Cascade delete completed for {doc_id}: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"❌ Failed to cascade delete {doc_id}: {e}")
            return stats
    
    def delete_user_cascade(self, user_id: str = None) -> Dict:
        """🆕 Delete all data for a user"""
        target_user = user_id or self.user_id
        
        stats = {
            'documents': 0,
            'chunks': 0,
            'entities': 0,
            'relationships': 0,
            'graph_nodes': 0,
            'graph_edges': 0,
            'files_deleted': []
        }
        
        try:
            # Delete all collections
            stats['documents'] = self.documents.delete_many({'user_id': target_user}).deleted_count
            stats['chunks'] = self.chunks.delete_many({'user_id': target_user}).deleted_count
            stats['entities'] = self.entities.delete_many({'user_id': target_user}).deleted_count
            stats['relationships'] = self.relationships.delete_many({'user_id': target_user}).deleted_count
            stats['graph_nodes'] = self.graph_nodes.delete_many({'user_id': target_user}).deleted_count
            stats['graph_edges'] = self.graph_edges.delete_many({'user_id': target_user}).deleted_count
            
            # Delete user directory
            import shutil
            user_dir = Path(f"backend/data/{target_user}")
            if user_dir.exists():
                shutil.rmtree(user_dir)
                stats['files_deleted'].append(str(user_dir))
                logger.info(f"✅ Deleted user directory: {user_dir}")
            
            logger.info(f"✅ User cascade delete completed: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"❌ Failed to cascade delete user {target_user}: {e}")
            return stats
    
    def get_graph(self) -> Dict:
        """Get combined knowledge graph for user"""
        try:
            nodes = list(self.graph_nodes.find({'user_id': self.user_id}))
            edges = list(self.graph_edges.find({'user_id': self.user_id}))
            
            graph = {
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
            
            logger.info(f"✅ Retrieved graph: {len(graph['nodes'])} nodes, {len(graph['links'])} edges")
            return graph
        
        except Exception as e:
            logger.error(f"❌ Failed to get graph: {e}")
            return {'nodes': [], 'links': []}
    
    def get_user_statistics(self) -> Dict:
        """Get comprehensive user statistics"""
        try:
            stats = {
                'total_documents': self.documents.count_documents({'user_id': self.user_id}),
                'total_chunks': self.chunks.count_documents({'user_id': self.user_id}),
                'total_entities': self.entities.count_documents({'user_id': self.user_id}),
                'total_relationships': self.relationships.count_documents({'user_id': self.user_id}),
                'graph_nodes': self.graph_nodes.count_documents({'user_id': self.user_id}),
                'graph_edges': self.graph_edges.count_documents({'user_id': self.user_id})
            }
            return stats
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'total_entities': 0,
                'total_relationships': 0,
                'graph_nodes': 0,
                'graph_edges': 0
            }
    
    def health_check(self) -> bool:
        """Check MongoDB connection health"""
        try:
            self.db.command('ping')
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB health check failed: {e}")
            return False