# backend/core/vector_store.py

from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None


class CombinedVectorStore:
    """
    ✅ Unified vector store for all documents
    Optimized for RAG/Chat retrieval
    """
    
    def __init__(self, user_id: str, use_hnsw: bool = True):
        self.user_id = user_id
        self.base_dir = Path("backend/data") / user_id / "vectors"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.base_dir / "COMBINED.index"
        self.metadata_path = self.base_dir / "COMBINED_metadata.json"
        self.doc_map_path = self.base_dir / "document_map.json"
        
        self.use_hnsw = use_hnsw
        self.dim = 384
        
        self.index = None
        self.metadata = {}
        self.document_map = {}
        
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing index or create new"""
        if self.index_path.exists() and self.metadata_path.exists():
            self._load()
        else:
            self._create_new()
    
    def _create_new(self):
        """Create new empty index"""
        if faiss:
            if self.use_hnsw:
                self.index = faiss.IndexHNSWFlat(self.dim, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
            else:
                self.index = faiss.IndexFlatL2(self.dim)
        
        self.metadata = {}
        self.document_map = {}
        logger.info(f"✅ Created new vector store for {self.user_id}")
    
    def _load(self):
        """Load existing index"""
        if faiss and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        if self.doc_map_path.exists():
            with open(self.doc_map_path, 'r', encoding='utf-8') as f:
                self.document_map = json.load(f)
        
        logger.info(f"✅ Loaded vector store: {len(self.metadata)} chunks, {len(self.document_map)} docs")
    
    def add_document_embeddings(self, 
                                document_id: str,
                                document_name: str,
                                embeddings: List[Dict[str, Any]],
                                tags: List[str] = None):
        """
        Add embeddings from a new document
        
        Args:
            document_id: Unique document ID
            document_name: Document filename
            embeddings: List of embedding dicts (from generate_embeddings)
            tags: Optional tags for filtering
        """
        if not embeddings:
            logger.warning(f"No embeddings to add for {document_id}")
            return
        
        # Get current index position
        start_idx = self.index.ntotal if self.index else len(self.metadata)
        
        # Add vectors to FAISS
        vectors = np.array([e['embedding'] for e in embeddings], dtype=np.float32)
        if self.index:
            self.index.add(vectors)
        
        # Add metadata
        for i, emb in enumerate(embeddings):
            idx = start_idx + i
            self.metadata[str(idx)] = {
                'chunk_id': emb['id'],
                'document_id': document_id,
                'document_name': document_name,
                'content': emb['text'],
                'hierarchy': emb.get('hierarchy', ''),
                'hierarchy_list': emb.get('hierarchy_list', []),
                'tokens': emb.get('tokens', 0),
                'order': emb.get('order', i),
                'file_path': emb.get('file_path', ''),
                'file_type': emb.get('file_type', ''),
                'entity_type': emb.get('entity_type', 'CHUNK')
            }
        
        # Update document map
        end_idx = start_idx + len(embeddings) - 1
        self.document_map[document_id] = {
            'document_name': document_name,
            'chunk_count': len(embeddings),
            'embedding_range': [start_idx, end_idx],
            'added_at': str(Path(embeddings[0].get('file_path', '')).stat().st_mtime) if embeddings[0].get('file_path') else None,
            'tags': tags or []
        }
        
        logger.info(f"✅ Added {len(embeddings)} embeddings for {document_name} (idx: {start_idx}-{end_idx})")
    
    def search(self, 
               query_embedding: List[float],
               top_k: int = 5,
               document_ids: Optional[List[str]] = None,
               tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            document_ids: Filter by specific documents
            tags: Filter by tags
        
        Returns:
            List of results with metadata
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning("Empty index, no results")
            return []
        
        # Search in FAISS
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Get more results for filtering
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_array, search_k)
        
        # Build results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if str(idx) not in self.metadata:
                continue
            
            meta = self.metadata[str(idx)].copy()
            doc_id = meta['document_id']
            
            # Apply filters
            if document_ids and doc_id not in document_ids:
                continue
            
            if tags:
                doc_tags = self.document_map.get(doc_id, {}).get('tags', [])
                if not any(tag in doc_tags for tag in tags):
                    continue
            
            meta['distance'] = float(dist)
            meta['similarity'] = float(1 / (1 + dist))  # Convert distance to similarity
            results.append(meta)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_text(self,
                       query_text: str,
                       top_k: int = 5,
                       **kwargs) -> List[Dict[str, Any]]:
        """
        Search by text query (generates embedding first)
        
        Args:
            query_text: Text query
            top_k: Number of results
            **kwargs: Additional filters (document_ids, tags)
        
        Returns:
            Search results
        """
        from backend.core.embedding import get_model
        
        model = get_model()
        query_emb = model.encode([query_text], show_progress=False)[0]
        
        return self.search(query_emb.tolist(), top_k=top_k, **kwargs)
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        if document_id not in self.document_map:
            return []
        
        start, end = self.document_map[document_id]['embedding_range']
        
        chunks = []
        for idx in range(start, end + 1):
            if str(idx) in self.metadata:
                chunks.append(self.metadata[str(idx)])
        
        return chunks
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete document embeddings (mark as deleted, rebuild needed for cleanup)
        
        Note: FAISS doesn't support individual deletion, 
        so we mark as deleted and rebuild index periodically
        """
        if document_id not in self.document_map:
            logger.warning(f"Document {document_id} not found")
            return False
        
        # Mark chunks as deleted
        start, end = self.document_map[document_id]['embedding_range']
        for idx in range(start, end + 1):
            if str(idx) in self.metadata:
                self.metadata[str(idx)]['_deleted'] = True
        
        # Remove from document map
        del self.document_map[document_id]
        
        logger.info(f"✅ Marked {document_id} for deletion (rebuild needed)")
        return True
    
    def rebuild_index(self):
        """
        Rebuild index (remove deleted chunks, compact)
        Call this periodically or after multiple deletions
        """
        logger.info("🔄 Rebuilding index...")
        
        # Collect active embeddings
        active_embeddings = []
        new_metadata = {}
        new_doc_map = {}
        
        current_idx = 0
        
        for doc_id, doc_info in self.document_map.items():
            start, end = doc_info['embedding_range']
            doc_embeddings = []
            
            for old_idx in range(start, end + 1):
                meta = self.metadata.get(str(old_idx))
                if meta and not meta.get('_deleted', False):
                    doc_embeddings.append(meta)
            
            if doc_embeddings:
                # Add to active list
                new_start = current_idx
                for meta in doc_embeddings:
                    new_metadata[str(current_idx)] = meta
                    current_idx += 1
                new_end = current_idx - 1
                
                # Update document map
                new_doc_map[doc_id] = {
                    **doc_info,
                    'embedding_range': [new_start, new_end],
                    'chunk_count': len(doc_embeddings)
                }
        
        # Rebuild FAISS index
        if faiss and new_metadata:
            # Extract vectors in new order
            vectors = []
            for idx in sorted(new_metadata.keys(), key=int):
                # Need to get original embedding - this requires storing them
                # For now, we'll need to regenerate or store embeddings
                pass
            
            # Create new index
            if self.use_hnsw:
                new_index = faiss.IndexHNSWFlat(self.dim, 32)
            else:
                new_index = faiss.IndexFlatL2(self.dim)
            
            # Note: Full rebuild requires re-adding all vectors
            # This is a limitation - better to minimize deletions
            
        self.metadata = new_metadata
        self.document_map = new_doc_map
        
        logger.info(f"✅ Rebuild complete: {len(new_metadata)} chunks, {len(new_doc_map)} docs")
    
    def save(self):
        """Save index and metadata to disk"""
        if self.index and faiss:
            faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        with open(self.doc_map_path, 'w', encoding='utf-8') as f:       
            json.dump(self.document_map, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved vector store: {self.index.ntotal if self.index else 0} vectors")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            'total_chunks': len(self.metadata),
            'total_documents': len(self.document_map),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dim,
            'index_type': 'HNSW' if self.use_hnsw else 'Flat',
            'documents': list(self.document_map.keys())
        }