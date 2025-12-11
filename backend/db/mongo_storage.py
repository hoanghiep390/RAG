# backend/db/mongo_storage.py

from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from backend.config import get_mongodb
import logging

# ✅ Import EntityValidator
from backend.db.entity_validator import EntityValidator

logger = logging.getLogger(__name__)

class MongoStorage:
    """ Proper graph deletion with doc_id tracking"""
    
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
            logger.info(f" MongoStorage initialized for user: {user_id}")
        except Exception as e:
            logger.error(f" Failed to initialize MongoStorage: {e}")
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
            
            #  Add doc_id indexes for graph collections
            self.graph_nodes.create_index([('user_id', 1), ('node_id', 1)], unique=True)
            self.graph_nodes.create_index([('user_id', 1), ('doc_id', 1)])  
            
            self.graph_edges.create_index([('user_id', 1), ('source', 1), ('target', 1)])
            self.graph_edges.create_index([('user_id', 1), ('doc_id', 1)])  
            
            logger.debug(" MongoDB indexes created (with doc_id for graph)")
        except Exception as e:
            logger.warning(f" Index creation warning: {e}")
    
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
            logger.info(f" Saved document: {filename}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f" Failed to save document {filename}: {e}")
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
                logger.info(f"Updated document status: {doc_id} -> {status}")
            return result.modified_count
        except Exception as e:
            logger.error(f" Failed to update document status: {e}")
            raise
    
    def get_document(self, doc_id: str):
        """Get document by ID"""
        try:
            return self.documents.find_one({'user_id': self.user_id, 'doc_id': doc_id})
        except Exception as e:
            logger.error(f" Failed to get document {doc_id}: {e}")
            return None
    
    def list_documents(self):
        """List all documents for user"""
        try:
            return list(self.documents.find({'user_id': self.user_id}).sort('uploaded_at', -1))
        except Exception as e:
            logger.error(f" Failed to list documents: {e}")
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
            logger.info(f" Saved {len(result.inserted_ids)} chunks")
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f" Failed to save chunks: {e}")
            return 0
    
    def save_entities_bulk(self, doc_id: str, entities_dict: Dict, enable_linking: bool = True, strict_mode: bool = False):
        """
        Bulk save entities with enhanced entity linking support
        
        Args:
            doc_id: Document ID
            entities_dict: Dict of {entity_name: [entity_dicts]}
            enable_linking: Enable cross-document entity linking (fuzzy matching)
            strict_mode: Use stricter matching threshold (0.9 vs 0.8)
        """
        if not entities_dict:
            return 0
        
        try:
            # ✅ ENHANCED: Use new entity linking module
            from backend.db.entity_linking import link_entities_batch, get_linking_statistics
            
            entity_docs = []
            
            # ✅ OPTIMIZATION: Load existing entities once
            existing_entities = []
            if enable_linking:
                existing_entities = list(self.entities.find(
                    {'user_id': self.user_id},
                    {'entity_name': 1}
                ).limit(1000))  # Limit for performance
            
            # ✅ ENHANCED: Batch entity linking with multi-level matching
            canonical_mapping, match_info = link_entities_batch(
                entities_dict,
                existing_entities,
                strict_mode=strict_mode
            )
            
            # Build entity documents with canonical names
            for entity_name, entity_list in entities_dict.items():
                canonical_name = canonical_mapping.get(entity_name, entity_name)
                
                # Save entities with canonical name
                for entity in entity_list:
                    entity_docs.append({
                        'user_id': self.user_id,
                        'doc_id': doc_id,
                        'entity_name': canonical_name,  # Use canonical name
                        'original_name': entity_name if canonical_name != entity_name else None,  # Track original
                        'entity_type': entity['entity_type'],
                        'description': entity.get('description', ''),
                        'source_id': entity.get('source_id', ''),
                        'chunk_id': entity.get('chunk_id', ''),
                        'created_at': datetime.now()
                    })
            
            if entity_docs:
                result = self.entities.insert_many(entity_docs, ordered=False)
                
                # ✅ ENHANCED: Log detailed statistics
                stats = get_linking_statistics(match_info)
                linked_count = stats['exact'] + stats['high_similarity'] + stats['medium_similarity'] + stats['acronym']
                
                logger.info(
                    f"✅ Saved {len(result.inserted_ids)} entities "
                    f"({linked_count} linked: {stats['exact']} exact, "
                    f"{stats['high_similarity']} high-sim, "
                    f"{stats['medium_similarity']} med-sim, "
                    f"{stats['acronym']} acronym)"
                )
                return len(result.inserted_ids)
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to save entities: {e}")
            return 0
    
    def save_relationships_bulk(self, doc_id: str, relationships_dict: Dict):
        """Bulk save relationships - LightRAG style"""
        if not relationships_dict:
            return 0
        
        try:
            # Validate relationships before saving
            validator = EntityValidator(self)
            valid_rels, invalid_rels = validator.validate_relationships_bulk(
                relationships_dict,
                use_cache=True
            )
            
            if invalid_rels:
                logger.warning(
                    f"⚠️ Filtered {len(invalid_rels)} invalid relationships "
                    f"(entities not found)"
                )
            
            if not valid_rels:
                logger.warning("⚠️ No valid relationships to save")
                return 0
            
            # Build relationship documents from valid relationships
            rel_docs = []
            for (source, target), rel_list in valid_rels.items():
                for rel in rel_list:
                    rel_docs.append({
                        'user_id': self.user_id,
                        'doc_id': doc_id,
                        'source_id': source,
                        'target_id': target,
                        'keywords': rel.get('keywords', ''),
                        'description': rel.get('description', ''),
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

    
    # ==========  SAVE GRAPH WITH DOC_ID TRACKING ==========
    
    def save_graph_bulk(self, graph_data: Dict, doc_id: str = None):
        """
        Save graph with doc_id tracking - LightRAG style
        
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
                                    'type': node.get('type', 'unknown'),
                                    'description': node.get('description', ''),
                                    'sources': list(node.get('sources', [])),
                                    'updated_at': datetime.now()
                                },
                                '$addToSet': {'doc_id': doc_id}
                            },
                            upsert=True
                        )
                        nodes_saved += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to save node {node.get('id')}: {e}")
            
            # Save edges with doc_id - LightRAG style (only keywords + description)
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
                                    'keywords': link.get('keywords', ''),
                                    'description': link.get('description', ''),
                                    'strength': link.get('strength', 1.0),
                                    'chunks': list(link.get('chunks', [])),
                                    'updated_at': datetime.now()
                                },
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
        """Complete save with doc_id tracking and graph rebuild"""
        try:
            self.save_document(doc_id, filename, filepath)
            
            if chunks:
                self.save_chunks_bulk(doc_id, chunks)
            
            if entities:
                self.save_entities_bulk(doc_id, entities)
            
            if relationships:
                self.save_relationships_bulk(doc_id, relationships)
            
            # ✅ CHANGE: Rebuild graph from entities + relationships
            # Graph nodes/edges are now cache, not source of truth
            if entities and relationships:
                logger.info(f"🔄 Rebuilding graph cache for doc: {doc_id}")
                self.sync_graph_cache(doc_id)
            elif graph:
                # Fallback: use provided graph if no entities/relationships
                logger.warning(f"⚠️ Using provided graph (no entities/relationships)")
                self.save_graph_bulk(graph, doc_id=doc_id)
            
            self.update_document_status(doc_id, 'completed', stats)
            
            logger.info(f"✅ Complete save for document: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save document completely: {e}")
            self.update_document_status(doc_id, 'failed', {'error': str(e)})
            return False
    
    # DELETE WITH DOC_ID QUERY 
    
    def delete_document_cascade(self, doc_id: str) -> Dict:
        """
             Delete graph using doc_id query
        
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
            
            # ========== : DELETE GRAPH NODES WITH DOC_ID ==========
            
            # Find nodes that have this doc_id
            nodes_with_doc = list(self.graph_nodes.find({
                'user_id': self.user_id,
                'doc_id': doc_id  
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
                    #  No other docs → DELETE
                    self.graph_nodes.delete_one({
                        'user_id': self.user_id,
                        'node_id': node['node_id']
                    })
                    stats['graph_nodes_deleted'] += 1
                    logger.debug(f" Deleted node: {node['node_id']}")
                else:
                    #  Other docs exist → UPDATE
                    self.graph_nodes.update_one(
                        {'user_id': self.user_id, 'node_id': node['node_id']},
                        {'$set': {'doc_id': remaining_docs}}
                    )
                    stats['graph_nodes_updated'] += 1
                    logger.debug(f"📝 Updated node: {node['node_id']} (removed {doc_id})")
            
            #  DELETE GRAPH EDGES WITH DOC_ID 
            
            # Find edges that have this doc_id
            edges_with_doc = list(self.graph_edges.find({
                'user_id': self.user_id,
                'doc_id': doc_id  
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
        """Get combined knowledge graph for user - LightRAG style"""
        try:
            nodes = list(self.graph_nodes.find({'user_id': self.user_id}))
            edges = list(self.graph_edges.find({'user_id': self.user_id}))
            
            graph = {
                'nodes': [
                    {
                        'id': n['node_id'],
                        'type': n.get('type', 'unknown'),
                        'description': n.get('description', ''),
                        'sources': n.get('sources', []),
                        'doc_id': n.get('doc_id', [])
                    }
                    for n in nodes
                ],
                'links': [
                    {
                        'source': e['source'],
                        'target': e['target'],
                        'keywords': e.get('keywords', ''),
                        'description': e.get('description', ''),
                        'strength': e.get('strength', 1.0),
                        'chunks': e.get('chunks', []),
                        'doc_id': e.get('doc_id', [])
                    }
                    for e in edges
                ]
            }
            
            logger.info(f"✅ Retrieved graph: {len(graph['nodes'])} nodes, {len(graph['links'])} edges")
            return graph
        
        except Exception as e:
            logger.error(f"❌ Failed to get graph: {e}")
            return {'nodes': [], 'links': []}

    
    # ========== 🔄 GRAPH REBUILD METHODS (NEW) ==========
    
    def sync_graph_cache(self, doc_id: str) -> Dict:
        """
        Sync graph cache for specific document
        Rebuild graph nodes/edges from entities + relationships
        
        Args:
            doc_id: Document ID to sync
        
        Returns:
            Stats dict with nodes/edges counts
        """
        stats = {'nodes': 0, 'edges': 0}
        
        try:
            # 1. Get entities for this document
            entities_cursor = self.entities.find({
                'user_id': self.user_id,
                'doc_id': doc_id
            })
            
            # Group entities by name (for deduplication)
            from collections import defaultdict
            entities_by_name = defaultdict(list)
            
            for entity in entities_cursor:
                entities_by_name[entity['entity_name']].append(entity)
            
            # 2. Create/update graph nodes
            for entity_name, entity_list in entities_by_name.items():
                # Merge descriptions
                descriptions = [e.get('description', '') for e in entity_list if e.get('description')]
                merged_desc = '; '.join(set(descriptions))[:500]
                
                # Get type (use first non-empty)
                entity_type = next((e['entity_type'] for e in entity_list if e.get('entity_type')), 'UNKNOWN')
                
                # Get sources
                sources = list(set(e.get('source_id', '') for e in entity_list if e.get('source_id')))
                
                # Upsert node
                self.graph_nodes.update_one(
                    {'user_id': self.user_id, 'node_id': entity_name},
                    {
                        '$set': {
                            'type': entity_type,
                            'description': merged_desc,
                            'sources': sources,
                            'updated_at': datetime.now()
                        },
                        '$addToSet': {'doc_id': doc_id}
                    },
                    upsert=True
                )
                stats['nodes'] += 1
            
            # 3. Get relationships for this document
            relationships_cursor = self.relationships.find({
                'user_id': self.user_id,
                'doc_id': doc_id
            })
            
            # Group relationships by (source, target)
            relationships_by_pair = defaultdict(list)
            
            for rel in relationships_cursor:
                key = (rel['source_id'], rel['target_id'])
                relationships_by_pair[key].append(rel)
            
            # 4. Create/update graph edges - LightRAG style
            for (source, target), rel_list in relationships_by_pair.items():
                # Merge descriptions
                descriptions = [r.get('description', '') for r in rel_list if r.get('description')]
                merged_desc = '; '.join(set(descriptions))[:500]
                
                # Merge keywords
                all_keywords = []
                for r in rel_list:
                    if r.get('keywords'):
                        all_keywords.extend(r['keywords'].split(','))
                merged_keywords = ','.join(sorted(set(k.strip() for k in all_keywords if k.strip())))
                
                # Calculate strength (average weight)
                weights = [r.get('weight', 1.0) for r in rel_list]
                avg_strength = sum(weights) / len(weights)
                
                # Get chunks
                chunks = list(set(r.get('chunk_id', '') for r in rel_list if r.get('chunk_id')))
                
                # Upsert edge
                self.graph_edges.update_one(
                    {
                        'user_id': self.user_id,
                        'source': source,
                        'target': target
                    },
                    {
                        '$set': {
                            'keywords': merged_keywords,
                            'description': merged_desc,
                            'strength': avg_strength,
                            'chunks': chunks,
                            'updated_at': datetime.now()
                        },
                        '$addToSet': {'doc_id': doc_id}
                    },
                    upsert=True
                )
                stats['edges'] += 1

            
            logger.info(
                f"✅ Synced graph cache for {doc_id}: "
                f"{stats['nodes']} nodes, {stats['edges']} edges"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to sync graph cache: {e}")
            return stats
    
    def rebuild_graph_from_entities(self, user_id: str = None) -> Dict:
        """
        Rebuild entire graph from entities + relationships
        
        Args:
            user_id: User ID (default: self.user_id)
        
        Returns:
            Stats dict
        """
        target_user = user_id or self.user_id
        
        stats = {
            'nodes_created': 0,
            'edges_created': 0,
            'nodes_deleted': 0,
            'edges_deleted': 0
        }
        
        try:
            logger.info(f"🔄 Rebuilding graph for user: {target_user}")
            
            # 1. Clear existing graph
            delete_nodes = self.graph_nodes.delete_many({'user_id': target_user})
            delete_edges = self.graph_edges.delete_many({'user_id': target_user})
            
            stats['nodes_deleted'] = delete_nodes.deleted_count
            stats['edges_deleted'] = delete_edges.deleted_count
            
            logger.info(
                f"🗑️ Cleared: {stats['nodes_deleted']} nodes, "
                f"{stats['edges_deleted']} edges"
            )
            
            # 2. Get all documents for user
            documents = self.documents.find(
                {'user_id': target_user},
                {'doc_id': 1}
            )
            
            doc_ids = [doc['doc_id'] for doc in documents]
            
            # 3. Rebuild for each document
            for doc_id in doc_ids:
                doc_stats = self.sync_graph_cache(doc_id)
                stats['nodes_created'] += doc_stats['nodes']
                stats['edges_created'] += doc_stats['edges']
            
            logger.info(
                f"✅ Rebuilt graph: {stats['nodes_created']} nodes, "
                f"{stats['edges_created']} edges"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild graph: {e}")
            return stats
    
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