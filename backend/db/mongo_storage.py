# backend/db/mongo_storage.py

from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from backend.config import get_mongodb
import logging

logger = logging.getLogger(__name__)

class MongoStorage:
    """✅ FIXED: Proper graph deletion with doc_id tracking"""
    
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
        """Create indexes including doc_id for graph collections"""
        try:
            # Existing indexes
            self.documents.create_index([('user_id', 1), ('doc_id', 1)], unique=True)
            self.chunks.create_index([('user_id', 1), ('doc_id', 1)])
            self.chunks.create_index([('chunk_id', 1)], unique=True)
            self.entities.create_index([('user_id', 1), ('entity_name', 1)])
            self.entities.create_index([('user_id', 1), ('doc_id', 1)])
            self.relationships.create_index([('user_id', 1), ('source_id', 1)])
            self.relationships.create_index([('user_id', 1), ('doc_id', 1)])
            
            # ✅ CRITICAL FIX: Add doc_id indexes for graph collections
            self.graph_nodes.create_index([('user_id', 1), ('node_id', 1)], unique=True)
            self.graph_nodes.create_index([('user_id', 1), ('doc_id', 1)])  
            
            self.graph_edges.create_index([('user_id', 1), ('source', 1), ('target', 1)])
            self.graph_edges.create_index([('user_id', 1), ('doc_id', 1)])  
            
            logger.debug("✅ MongoDB indexes created (with doc_id for graph)")
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")
    
    # ========== SAVE METHODS (Keep existing) ==========
    
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
                        'relationship_type': rel.get('relationship_type', 'RELATED_TO'),
                        'verb_phrase': rel.get('verb_phrase', ''),
                        'category': rel.get('category', 'ASSOCIATIVE'),
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
    
    # ========== ✅ FIX 1: SAVE GRAPH WITH DOC_ID TRACKING ==========
    
    def save_graph_bulk(self, graph_data: Dict, doc_id: str = None):
        """
        ✅ FIXED: Track doc_id for proper deletion
        
        Args:
            graph_data: Graph dict with nodes/links
            doc_id: Document ID (REQUIRED for deletion tracking)
        """
        if not doc_id:
            logger.error("❌ doc_id is required for save_graph_bulk!")
            return {'nodes': 0, 'edges': 0}
        
        nodes_saved = 0
        edges_saved = 0
        
        try:
            # Save nodes with doc_id
            if graph_data.get('nodes'):
                for node in graph_data['nodes']:
                    try:
                        self.graph_nodes.update_one(
                            {'user_id': self.user_id, 'node_id': node['id']},
                            {
                                '$set': {
                                    'type': node.get('type', 'UNKNOWN'),
                                    'description': node.get('description', ''),
                                    'sources': list(node.get('sources', [])),
                                    'updated_at': datetime.now()
                                },
                                # ✅ CRITICAL FIX: Track doc_id
                                '$addToSet': {'doc_id': doc_id}
                            },
                            upsert=True
                        )
                        nodes_saved += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save node {node.get('id')}: {e}")
            
            # Save edges with doc_id
            if graph_data.get('links'):
                for link in graph_data['links']:
                    try:
                        self.graph_edges.update_one(
                            {
                                'user_id': self.user_id,
                                'source': link['source'],
                                'target': link['target']
                            },
                            {
                                '$set': {
                                    'relationship_type': link.get('relationship_type', 'RELATED_TO'),
                                    'verb_phrase': link.get('verb_phrase', ''),
                                    'category': link.get('category', 'ASSOCIATIVE'),
                                    'description': link.get('description', ''),
                                    'strength': link.get('strength', 1.0),
                                    'chunks': list(link.get('chunks', [])),
                                    'updated_at': datetime.now()
                                },
                                # ✅ CRITICAL FIX: Track doc_id
                                '$addToSet': {'doc_id': doc_id}
                            },
                            upsert=True
                        )
                        edges_saved += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save edge {link.get('source')}->{link.get('target')}: {e}")
            
            logger.info(f"✅ Saved graph: {nodes_saved} nodes, {edges_saved} edges (doc_id={doc_id})")
            return {'nodes': nodes_saved, 'edges': edges_saved}
        
        except Exception as e:
            logger.error(f"❌ Failed to save graph: {e}")
            return {'nodes': nodes_saved, 'edges': edges_saved}
    
    def save_document_complete(self, doc_id: str, filename: str, filepath: str,
                               chunks: List[Dict], entities: Dict = None,
                               relationships: Dict = None, graph: Dict = None, stats: Dict = None):
        """Complete save with doc_id tracking"""
        try:
            self.save_document(doc_id, filename, filepath)
            
            if chunks:
                self.save_chunks_bulk(doc_id, chunks)
            
            if entities:
                self.save_entities_bulk(doc_id, entities)
            
            if relationships:
                self.save_relationships_bulk(doc_id, relationships)
            
            if graph:
                # ✅ CRITICAL: Pass doc_id to save_graph_bulk
                self.save_graph_bulk(graph, doc_id=doc_id)
            
            self.update_document_status(doc_id, 'completed', stats)
            
            logger.info(f"✅ Complete save for document: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save document completely: {e}")
            self.update_document_status(doc_id, 'failed', {'error': str(e)})
            return False
    
    # ========== ✅ FIX 2: DELETE WITH DOC_ID QUERY ==========
    
    def delete_document_cascade(self, doc_id: str) -> Dict:
        """
        ✅ FIXED: Delete graph using doc_id query
        
        Strategy:
        1. Delete all data with doc_id (chunks, entities, relationships)
        2. For graph nodes/edges: Check if ONLY this doc created them
        3. If yes → DELETE, if no → Keep but remove doc_id from array
        """
        stats = {
            'document': 0,
            'chunks': 0,
            'entities': 0,
            'relationships': 0,
            'graph_nodes_deleted': 0,
            'graph_nodes_updated': 0,
            'graph_edges_deleted': 0,
            'graph_edges_updated': 0,
            'files_deleted': [],
            'errors': []
        }
        
        try:
            # Get document info
            doc = self.get_document(doc_id)
            if not doc:
                logger.warning(f"⚠️ Document {doc_id} not found")
                stats['errors'].append(f"Document {doc_id} not found")
                return stats
            
            # 1. Delete collections data (simple - have doc_id field)
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
            
            # ========== ✅ FIX: DELETE GRAPH NODES WITH DOC_ID ==========
            
            # Find nodes that have this doc_id
            nodes_with_doc = list(self.graph_nodes.find({
                'user_id': self.user_id,
                'doc_id': doc_id  # ✅ CORRECT QUERY
            }))
            
            logger.info(f"🔍 Found {len(nodes_with_doc)} nodes with doc_id={doc_id}")
            
            for node in nodes_with_doc:
                doc_ids = node.get('doc_id', [])
                
                # Handle both array and single value
                if not isinstance(doc_ids, list):
                    doc_ids = [doc_ids] if doc_ids else []
                
                # Remove current doc_id
                remaining_docs = [d for d in doc_ids if d != doc_id]
                
                if not remaining_docs:
                    # ✅ No other docs → DELETE
                    self.graph_nodes.delete_one({
                        'user_id': self.user_id,
                        'node_id': node['node_id']
                    })
                    stats['graph_nodes_deleted'] += 1
                    logger.debug(f"🗑️ Deleted node: {node['node_id']}")
                else:
                    # ✅ Other docs exist → UPDATE
                    self.graph_nodes.update_one(
                        {'user_id': self.user_id, 'node_id': node['node_id']},
                        {'$set': {'doc_id': remaining_docs}}
                    )
                    stats['graph_nodes_updated'] += 1
                    logger.debug(f"📝 Updated node: {node['node_id']} (removed {doc_id})")
            
            # ========== ✅ FIX: DELETE GRAPH EDGES WITH DOC_ID ==========
            
            # Find edges that have this doc_id
            edges_with_doc = list(self.graph_edges.find({
                'user_id': self.user_id,
                'doc_id': doc_id  # ✅ CORRECT QUERY
            }))
            
            logger.info(f"🔍 Found {len(edges_with_doc)} edges with doc_id={doc_id}")
            
            for edge in edges_with_doc:
                doc_ids = edge.get('doc_id', [])
                
                # Handle both array and single value
                if not isinstance(doc_ids, list):
                    doc_ids = [doc_ids] if doc_ids else []
                
                # Remove current doc_id
                remaining_docs = [d for d in doc_ids if d != doc_id]
                
                if not remaining_docs:
                    # ✅ No other docs → DELETE
                    self.graph_edges.delete_one({
                        'user_id': self.user_id,
                        'source': edge['source'],
                        'target': edge['target']
                    })
                    stats['graph_edges_deleted'] += 1
                    logger.debug(f"🗑️ Deleted edge: {edge['source']} → {edge['target']}")
                else:
                    # ✅ Other docs exist → UPDATE
                    self.graph_edges.update_one(
                        {
                            'user_id': self.user_id,
                            'source': edge['source'],
                            'target': edge['target']
                        },
                        {'$set': {'doc_id': remaining_docs}}
                    )
                    stats['graph_edges_updated'] += 1
                    logger.debug(f"📝 Updated edge: {edge['source']} → {edge['target']} (removed {doc_id})")
            
            # 3. Delete physical file
            if doc.get('filepath'):
                filepath = Path(doc['filepath'])
                if filepath.exists():
                    try:
                        filepath.unlink()
                        stats['files_deleted'].append(str(filepath))
                        logger.info(f"✅ Deleted file: {filepath}")
                    except Exception as e:
                        error_msg = f"Failed to delete file {filepath}: {e}"
                        logger.warning(f"⚠️ {error_msg}")
                        stats['errors'].append(error_msg)
                else:
                    logger.warning(f"⚠️ File not found: {filepath}")
                    stats['errors'].append(f"File not found: {filepath.name}")
            
            logger.info(f"✅ Cascade delete completed for {doc_id}:")
            logger.info(f"   📊 Nodes: {stats['graph_nodes_deleted']} deleted, {stats['graph_nodes_updated']} updated")
            logger.info(f"   🔗 Edges: {stats['graph_edges_deleted']} deleted, {stats['graph_edges_updated']} updated")
            
            return stats
        
        except Exception as e:
            error_msg = f"Failed to cascade delete {doc_id}: {e}"
            logger.error(f"❌ {error_msg}")
            stats['errors'].append(error_msg)
            return stats
    
    # ========== OTHER METHODS (Unchanged) ==========
    
    def delete_user_cascade(self, user_id: str = None) -> Dict:
        """Delete all data for a user"""
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
            stats['documents'] = self.documents.delete_many({'user_id': target_user}).deleted_count
            stats['chunks'] = self.chunks.delete_many({'user_id': target_user}).deleted_count
            stats['entities'] = self.entities.delete_many({'user_id': target_user}).deleted_count
            stats['relationships'] = self.relationships.delete_many({'user_id': target_user}).deleted_count
            stats['graph_nodes'] = self.graph_nodes.delete_many({'user_id': target_user}).deleted_count
            stats['graph_edges'] = self.graph_edges.delete_many({'user_id': target_user}).deleted_count
            
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
                        'doc_id': n.get('doc_id', [])  # ✅ Include for debugging
                    }
                    for n in nodes
                ],
                'links': [
                    {
                        'source': e['source'],
                        'target': e['target'],
                        'relationship_type': e.get('relationship_type', 'RELATED_TO'),
                        'verb_phrase': e.get('verb_phrase', ''),
                        'category': e.get('category', 'ASSOCIATIVE'),
                        'description': e.get('description', ''),
                        'strength': e.get('strength', 1.0),
                        'chunks': e.get('chunks', []),
                        'doc_id': e.get('doc_id', [])  # ✅ Include for debugging
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