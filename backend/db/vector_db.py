# backend/db/vector_db.py
"""
✅ OPTIMIZED: FAISS Vector Database with Better Delete & Batch Operations
- Efficient batch addition
- Mark-and-sweep deletion
- Auto-save after operations
"""

from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ FAISS not installed. Vector search will not work.")


class VectorDatabase:
    """Optimized FAISS vector database with consistent delete"""
    
    def __init__(self, user_id: str, dim: int = 384, use_hnsw: bool = True, auto_save: bool = True):
        """
        Initialize vector database
        
        Args:
            user_id: User ID for data isolation
            dim: Embedding dimension (default 384 for all-MiniLM-L6-v2)
            use_hnsw: Use HNSW index for faster search
            auto_save: Auto-save after operations
        """
        self.user_id = user_id
        self.dim = dim
        self.use_hnsw = use_hnsw
        self.auto_save = auto_save
        
        # Setup directories
        self.base_dir = Path("backend/data") / user_id / "vectors"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_path = self.base_dir / "combined.index"
        self.metadata_path = self.base_dir / "combined_metadata.json"
        self.doc_map_path = self.base_dir / "document_map.json"
        
        # Initialize
        self.index = None
        self.metadata = {}
        self.document_map = {}
        self._deleted_count = 0
        
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing index or create new"""
        if not FAISS_AVAILABLE:
            logger.error("❌ FAISS not available. Cannot create vector database.")
            return
        
        if self.index_path.exists() and self.metadata_path.exists():
            self._load()
        else:
            self._create_new()
    
    def _create_new(self):
        """Create new empty index"""
        if self.use_hnsw:
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
        else:
            self.index = faiss.IndexFlatL2(self.dim)
        
        self.metadata = {}
        self.document_map = {}
        self._deleted_count = 0
        logger.info(f"✅ Created new vector store for {self.user_id}")
    
    def _load(self):
        """Load existing index"""
        try:
            self.index = faiss.read_index(str(self.index_path))
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            if self.doc_map_path.exists():
                with open(self.doc_map_path, 'r', encoding='utf-8') as f:
                    self.document_map = json.load(f)
            
            # Count deleted
            self._deleted_count = sum(1 for m in self.metadata.values() if m.get('_deleted'))
            
            logger.info(f"✅ Loaded vector store: {len(self.metadata)} vectors, "
                       f"{len(self.document_map)} docs, {self._deleted_count} deleted")
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {e}")
            self._create_new()
    
    def add_document_embeddings_batch(self, 
                                      doc_id: str,
                                      filename: str,
                                      embeddings: List[Dict[str, Any]],
                                      tags: List[str] = None) -> int:
        """
        ✅ OPTIMIZED: Batch add embeddings for a document
        
        Returns:
            Number of embeddings added
        """
        if not FAISS_AVAILABLE or not embeddings:
            logger.warning(f"Cannot add embeddings for {doc_id}")
            return 0
        
        start_idx = self.index.ntotal
        
        # Extract vectors and add to FAISS in batch
        vectors = np.array([e['embedding'] for e in embeddings], dtype=np.float32)
        self.index.add(vectors)
        
        # Add metadata in batch
        for i, emb in enumerate(embeddings):
            idx = start_idx + i
            self.metadata[str(idx)] = {
                'chunk_id': emb['id'],
                'doc_id': doc_id,
                'filename': filename,
                'content': emb['text'],
                'hierarchy': emb.get('hierarchy', ''),
                'hierarchy_list': emb.get('hierarchy_list', []),
                'tokens': emb.get('tokens', 0),
                'order': emb.get('order', i),
                'file_path': emb.get('file_path', ''),
                'file_type': emb.get('file_type', ''),
                'entity_type': emb.get('entity_type', 'CHUNK'),
                '_deleted': False
            }
        
        # Update document map
        end_idx = start_idx + len(embeddings) - 1
        self.document_map[doc_id] = {
            'filename': filename,
            'chunk_count': len(embeddings),
            'embedding_range': [start_idx, end_idx],
            'tags': tags or []
        }
        
        logger.info(f"✅ Batch added {len(embeddings)} embeddings for {filename} (idx: {start_idx}-{end_idx})")
        
        if self.auto_save:
            self.save()
        
        return len(embeddings)
    
    def search(self, 
               query_embedding: List[float],
               top_k: int = 5,
               doc_ids: Optional[List[str]] = None,
               tags: Optional[List[str]] = None,
               skip_deleted: bool = True) -> List[Dict[str, Any]]:
        """
        ✅ OPTIMIZED: Search with deleted items filtering
        """
        if not FAISS_AVAILABLE or not self.index or self.index.ntotal == 0:
            logger.warning("Empty or unavailable index")
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Get more results for filtering
        search_k = min(top_k * 5, self.index.ntotal)
        distances, indices = self.index.search(query_array, search_k)
        
        # Build results with metadata and filters
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if str(idx) not in self.metadata:
                continue
            
            meta = self.metadata[str(idx)].copy()
            
            # Skip deleted
            if skip_deleted and meta.get('_deleted'):
                continue
            
            doc_id = meta['doc_id']
            
            # Apply filters
            if doc_ids and doc_id not in doc_ids:
                continue
            
            if tags:
                doc_tags = self.document_map.get(doc_id, {}).get('tags', [])
                if not any(tag in doc_tags for tag in tags):
                    continue
            
            meta['distance'] = float(dist)
            meta['similarity'] = float(1 / (1 + dist))
            results.append(meta)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_text(self,
                       query_text: str,
                       top_k: int = 5,
                       **kwargs) -> List[Dict[str, Any]]:
        """Search by text query (generates embedding first)"""
        from backend.core.embedding import get_model
        
        model = get_model()
        query_emb = model.encode([query_text], show_progress=False)[0]
        
        return self.search(query_emb.tolist(), top_k=top_k, **kwargs)
    
    def get_document_chunks(self, doc_id: str, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        if doc_id not in self.document_map:
            return []
        
        start, end = self.document_map[doc_id]['embedding_range']
        
        chunks = []
        for idx in range(start, end + 1):
            if str(idx) in self.metadata:
                meta = self.metadata[str(idx)]
                if include_deleted or not meta.get('_deleted'):
                    chunks.append(meta)
        
        return chunks
    
    def delete_document(self, doc_id: str) -> Dict[str, int]:
        """
        ✅ IMPROVED: Mark document embeddings as deleted
        
        Returns:
            Statistics: {'marked': count, 'needs_rebuild': bool}
        """
        if doc_id not in self.document_map:
            logger.warning(f"Document {doc_id} not found in FAISS")
            return {'marked': 0, 'needs_rebuild': False}
        
        # Mark chunks as deleted
        start, end = self.document_map[doc_id]['embedding_range']
        marked_count = 0
        
        for idx in range(start, end + 1):
            if str(idx) in self.metadata:
                self.metadata[str(idx)]['_deleted'] = True
                marked_count += 1
        
        # Remove from document map
        del self.document_map[doc_id]
        
        self._deleted_count += marked_count
        
        # Check if rebuild needed (>20% deleted)
        total = len(self.metadata)
        needs_rebuild = self._deleted_count > total * 0.2
        
        logger.info(f"✅ Marked {marked_count} embeddings as deleted for {doc_id}")
        
        if needs_rebuild:
            logger.warning(f"⚠️ {self._deleted_count}/{total} deleted ({self._deleted_count/total*100:.1f}%). Consider rebuild.")
        
        if self.auto_save:
            self.save()
        
        return {
            'marked': marked_count,
            'needs_rebuild': needs_rebuild,
            'total_deleted': self._deleted_count,
            'total_vectors': total
        }
    
    def rebuild_index(self) -> Dict[str, int]:
        """
        ✅ OPTIMIZED: Rebuild index (remove deleted chunks, compact)
        
        Returns:
            Statistics: {'before': int, 'after': int, 'removed': int}
        """
        if not FAISS_AVAILABLE:
            return {}
        
        before_count = len(self.metadata)
        logger.info(f"🔄 Rebuilding index ({self._deleted_count} deleted)...")
        
        # Collect active embeddings
        active_vectors = []
        new_metadata = {}
        new_doc_map = {}
        
        current_idx = 0
        
        for doc_id, doc_info in self.document_map.items():
            start, end = doc_info['embedding_range']
            doc_vectors = []
            doc_metadata = []
            doc_indices = []
            
            for old_idx in range(start, end + 1):
                meta = self.metadata.get(str(old_idx))
                if meta and not meta.get('_deleted'):
                    doc_vectors.append(old_idx)
                    doc_metadata.append(meta)
                    doc_indices.append(old_idx)
            
            if doc_vectors:
                # Get vectors from old index
                vectors = np.zeros((len(doc_vectors), self.dim), dtype=np.float32)
                for i, old_idx in enumerate(doc_indices):
                    vectors[i] = self.index.reconstruct(int(old_idx))
                
                # Add to new metadata
                new_start = current_idx
                for i, meta in enumerate(doc_metadata):
                    meta_copy = meta.copy()
                    meta_copy['_deleted'] = False
                    new_metadata[str(current_idx)] = meta_copy
                    active_vectors.append(vectors[i])
                    current_idx += 1
                new_end = current_idx - 1
                
                # Update document map
                new_doc_map[doc_id] = {
                    **doc_info,
                    'embedding_range': [new_start, new_end],
                    'chunk_count': len(doc_metadata)
                }
        
        # Create new index
        if self.use_hnsw:
            new_index = faiss.IndexHNSWFlat(self.dim, 32)
            new_index.hnsw.efConstruction = 200
            new_index.hnsw.efSearch = 50
        else:
            new_index = faiss.IndexFlatL2(self.dim)
        
        # Add vectors
        if active_vectors:
            vectors_array = np.array(active_vectors, dtype=np.float32)
            new_index.add(vectors_array)
        
        # Replace old with new
        self.index = new_index
        self.metadata = new_metadata
        self.document_map = new_doc_map
        self._deleted_count = 0
        
        after_count = len(new_metadata)
        removed = before_count - after_count
        
        logger.info(f"✅ Rebuild complete: {before_count} → {after_count} ({removed} removed)")
        
        if self.auto_save:
            self.save()
        
        return {
            'before': before_count,
            'after': after_count,
            'removed': removed
        }
    
    def save(self):
        """Save index and metadata to disk"""
        if not FAISS_AVAILABLE:
            return
        
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        with open(self.doc_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.document_map, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved vector store: {self.index.ntotal if self.index else 0} vectors, "
                   f"{self._deleted_count} deleted")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics"""
        total = len(self.metadata)
        active = total - self._deleted_count
        
        return {
            'total_vectors': total,
            'active_vectors': active,
            'deleted_vectors': self._deleted_count,
            'deletion_ratio': self._deleted_count / total if total > 0 else 0,
            'total_documents': len(self.document_map),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dim,
            'index_type': 'HNSW' if self.use_hnsw else 'Flat',
            'needs_rebuild': self._deleted_count > total * 0.2,
            'documents': list(self.document_map.keys())
        }
    
    def cleanup_orphans(self):
        """Remove metadata for vectors not in document_map"""
        doc_ranges = set()
        for doc_info in self.document_map.values():
            start, end = doc_info['embedding_range']
            doc_ranges.update(range(start, end + 1))
        
        orphans = []
        for idx_str in list(self.metadata.keys()):
            if int(idx_str) not in doc_ranges:
                orphans.append(idx_str)
                del self.metadata[idx_str]
        
        if orphans:
            logger.info(f"🧹 Cleaned up {len(orphans)} orphaned metadata entries")
            if self.auto_save:
                self.save()
        
        return len(orphans)