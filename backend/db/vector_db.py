# ==========================================
# backend/db/vector_db.py 
# ==========================================
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Optional
import importlib
import logging

logger = logging.getLogger(__name__)

# Try to import real FAISS
try:
    faiss = importlib.import_module('faiss')
    FAISS_AVAILABLE = True
    logger.info("✅ Real FAISS library loaded")
except Exception:
    FAISS_AVAILABLE = False
    logger.warning("⚠️ FAISS not available, using NumPy fallback")

    # NumPy-based fallback implementation
    class IndexFlatL2:
        """NumPy-based fallback for FAISS IndexFlatL2"""
        def __init__(self, dim):
            self.dim = dim
            self._vectors = np.zeros((0, dim), dtype=np.float32)

        @property
        def ntotal(self):
            return self._vectors.shape[0]

        def add(self, vectors):
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            if self._vectors.size == 0:
                self._vectors = vectors.copy()
            else:
                self._vectors = np.vstack([self._vectors, vectors])

        def search(self, query_array, k):
            q = np.asarray(query_array, dtype=np.float32)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            # L2 distance
            dists = np.sum((q[:, None, :] - self._vectors[None, :, :]) ** 2, axis=2)
            idx = np.argsort(dists, axis=1)[:, :k]
            dist_sorted = np.take_along_axis(dists, idx, axis=1)
            return dist_sorted, idx

        def reconstruct(self, index):
            return self._vectors[int(index)].copy()

    class IndexHNSWFlat:
        """NumPy-based fallback for FAISS IndexHNSWFlat (same as Flat for simplicity)"""
        def __init__(self, dim, M=32):
            self.dim = dim
            self.M = M
            self._vectors = np.zeros((0, dim), dtype=np.float32)
            logger.info(f"⚠️ Using NumPy fallback for HNSW (M={M}), no actual HNSW optimization")

        @property
        def ntotal(self):
            return self._vectors.shape[0]

        def add(self, vectors):
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            if self._vectors.size == 0:
                self._vectors = vectors.copy()
            else:
                self._vectors = np.vstack([self._vectors, vectors])

        def search(self, query_array, k):
            q = np.asarray(query_array, dtype=np.float32)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            dists = np.sum((q[:, None, :] - self._vectors[None, :, :]) ** 2, axis=2)
            idx = np.argsort(dists, axis=1)[:, :k]
            dist_sorted = np.take_along_axis(dists, idx, axis=1)
            return dist_sorted, idx

        def reconstruct(self, index):
            return self._vectors[int(index)].copy()

    def write_index(index, path):
        arr = getattr(index, '_vectors', np.zeros((0, index.dim), dtype=np.float32))
        np.save(str(path), arr)

    def read_index(path):
        p = str(path)
        try:
            arr = np.load(p, allow_pickle=False)
        except Exception:
            arr = np.load(p + '.npy', allow_pickle=False)
        
        # Detect if it was HNSW by checking metadata file
        metadata_path = Path(str(path).replace('.index', '_metadata.json'))
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                if meta.get('index_type') == 'HNSW':
                    idx = IndexHNSWFlat(arr.shape[1] if arr.ndim == 2 else 0, M=meta.get('M', 32))
                else:
                    idx = IndexFlatL2(arr.shape[1] if arr.ndim == 2 else 0)
        else:
            idx = IndexFlatL2(arr.shape[1] if arr.ndim == 2 else 0)
        
        if arr.size:
            idx.add(arr)
        return idx

    # Create mock faiss module
    faiss = type('faiss_module', (), {
        'IndexFlatL2': IndexFlatL2,
        'IndexHNSWFlat': IndexHNSWFlat,
        'write_index': staticmethod(write_index),
        'read_index': staticmethod(read_index)
    })()


class VectorDatabase:
    """Enhanced FAISS wrapper with Config integration"""
    
    def __init__(self, user_id: str, dim: int = None, use_hnsw: bool = None, auto_save: bool = True):
        """Initialize vector DB for a user
        
        Args:
            user_id: User ID for data isolation
            dim: Embedding dimension (default from Config)
            use_hnsw: Use HNSW index (default from Config)
            auto_save: Auto-save changes to disk
        """
        from backend.config import Config
        
        self.user_id = user_id
        self.dim = dim or Config.EMBEDDING_DIM
        self.use_hnsw = use_hnsw if use_hnsw is not None else Config.USE_HNSW
        self.auto_save = auto_save
        
        # Get user vector directory from Config
        self.base_dir = Config.get_user_vector_dir(user_id)
        
        self.index_path = self.base_dir / "combined.index"
        self.metadata_path = self.base_dir / "combined_metadata.json"
        self.doc_map_path = self.base_dir / "document_map.json"
        
        self.index = None
        self.metadata = {}
        self.document_map = {}
        self._deleted_count = 0
        self._index_type = 'FLAT'
        
        self._load_or_create()
    
    def _load_or_create(self):
        """Load or create index"""
        if not FAISS_AVAILABLE:
            logger.warning("⚠️ Using NumPy fallback for FAISS")
        
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                
                if self.doc_map_path.exists():
                    with open(self.doc_map_path, 'r', encoding='utf-8') as f:
                        self.document_map = json.load(f)
                
                self._deleted_count = sum(1 for m in self.metadata.values() if m.get('_deleted'))
                
                logger.info(f"✅ Loaded FAISS index: {self.index.ntotal} vectors ({self._deleted_count} deleted)")
            except Exception as e:
                logger.error(f"❌ Failed to load index: {e}, creating new one")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        from backend.config import Config
        
        if self.use_hnsw:
            try:
                if hasattr(faiss, 'IndexHNSWFlat'):
                    self.index = faiss.IndexHNSWFlat(self.dim, Config.HNSW_M)
                    self._index_type = 'HNSW'
                    logger.info(f"✅ Created HNSW index (M={Config.HNSW_M})")
                else:
                    self.index = faiss.IndexFlatL2(self.dim)
                    self._index_type = 'FLAT'
                    logger.warning("⚠️ HNSW not available, using Flat index")
            except Exception as e:
                logger.error(f"❌ HNSW creation failed: {e}, using Flat index")
                self.index = faiss.IndexFlatL2(self.dim)
                self._index_type = 'FLAT'
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self._index_type = 'FLAT'
            logger.info("✅ Created Flat L2 index")
        
        self.metadata = {}
        self.document_map = {}
        self._deleted_count = 0
    
    def add_document_embeddings_batch(self, doc_id: str, filename: str, 
                                      embeddings: List[Dict], tags: List[str] = None) -> int:
        """Batch add embeddings for a document"""
        if not FAISS_AVAILABLE and not embeddings:
            return 0
        
        if not embeddings:
            logger.warning(f"⚠️ No embeddings to add for {filename}")
            return 0
        
        start_idx = self.index.ntotal
        
        # Add vectors to index
        vectors = np.array([e['embedding'] for e in embeddings], dtype=np.float32)
        self.index.add(vectors)
        
        # Add metadata
        for i, emb in enumerate(embeddings):
            idx = start_idx + i
            self.metadata[str(idx)] = {
                'chunk_id': emb['id'],
                'doc_id': doc_id,
                'filename': filename,
                'content': emb['text'],
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
        
        if self.auto_save:
            self.save()
        
        logger.info(f"✅ Added {len(embeddings)} embeddings for {filename}")
        return len(embeddings)
    
    def search(self, query_embedding: List[float], top_k: int = 5,
               doc_ids: Optional[List[str]] = None, skip_deleted: bool = True) -> List[Dict]:
        """Vector similarity search"""
        if not FAISS_AVAILABLE and (not self.index or self.index.ntotal == 0):
            logger.warning("⚠️ Empty index or FAISS unavailable")
            return []
        
        if self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        search_k = min(top_k * 5, self.index.ntotal)
        
        try:
            distances, indices = self.index.search(query_array, search_k)
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if str(idx) not in self.metadata:
                continue
            
            meta = self.metadata[str(idx)].copy()
            
            if skip_deleted and meta.get('_deleted'):
                continue
            
            if doc_ids and meta['doc_id'] not in doc_ids:
                continue
            
            meta['distance'] = float(dist)
            meta['similarity'] = float(1 / (1 + dist))
            results.append(meta)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_text(self, query_text: str, top_k: int = 5, **kwargs) -> List[Dict]:
        """Search by text query"""
        from backend.core.embedding import get_model
        model = get_model()
        query_emb = model.encode([query_text], show_progress=False)[0]
        return self.search(query_emb.tolist(), top_k=top_k, **kwargs)
    
    def delete_document(self, doc_id: str) -> Dict:
        """Mark document as deleted (soft delete)"""
        if doc_id not in self.document_map:
            logger.warning(f"⚠️ Document {doc_id} not found")
            return {'marked': 0, 'needs_rebuild': False}
        
        start, end = self.document_map[doc_id]['embedding_range']
        marked_count = 0
        
        for idx in range(start, end + 1):
            if str(idx) in self.metadata:
                self.metadata[str(idx)]['_deleted'] = True
                marked_count += 1
        
        del self.document_map[doc_id]
        self._deleted_count += marked_count
        
        from backend.config import Config
        total = len(self.metadata)
        needs_rebuild = self._deleted_count > total * Config.AUTO_REBUILD_THRESHOLD
        
        if self.auto_save:
            self.save()
        
        logger.info(f"✅ Marked {marked_count} vectors as deleted for {doc_id}")
        
        return {
            'marked': marked_count,
            'needs_rebuild': needs_rebuild,
            'total_deleted': self._deleted_count,
            'total_vectors': total
        }
    
    def rebuild_index(self) -> Dict:
        """Rebuild index, removing deleted entries"""
        if not FAISS_AVAILABLE:
            logger.warning("⚠️ Cannot rebuild without FAISS")
            return {}
        
        before_count = len(self.metadata)
        logger.info(f"🔨 Rebuilding index ({before_count} vectors, {self._deleted_count} deleted)...")
        
        # Collect active vectors
        active_vectors = []
        new_metadata = {}
        new_doc_map = {}
        current_idx = 0
        
        for doc_id, doc_info in self.document_map.items():
            start, end = doc_info['embedding_range']
            doc_vectors = []
            
            for old_idx in range(start, end + 1):
                meta = self.metadata.get(str(old_idx))
                if meta and not meta.get('_deleted'):
                    doc_vectors.append((old_idx, meta))
            
            if doc_vectors:
                new_start = current_idx
                for old_idx, meta in doc_vectors:
                    vector = self.index.reconstruct(int(old_idx))
                    active_vectors.append(vector)
                    meta_copy = meta.copy()
                    meta_copy['_deleted'] = False
                    new_metadata[str(current_idx)] = meta_copy
                    current_idx += 1
                new_end = current_idx - 1
                
                new_doc_map[doc_id] = {
                    **doc_info,
                    'embedding_range': [new_start, new_end],
                    'chunk_count': len(doc_vectors)
                }
        
        # Create new index
        from backend.config import Config
        
        if self._index_type == 'HNSW' and hasattr(faiss, 'IndexHNSWFlat'):
            new_index = faiss.IndexHNSWFlat(self.dim, Config.HNSW_M)
        else:
            new_index = faiss.IndexFlatL2(self.dim)
        
        if active_vectors:
            vectors_array = np.array(active_vectors, dtype=np.float32)
            new_index.add(vectors_array)
        
        self.index = new_index
        self.metadata = new_metadata
        self.document_map = new_doc_map
        self._deleted_count = 0
        
        if self.auto_save:
            self.save()
        
        removed = before_count - len(new_metadata)
        logger.info(f"✅ Rebuild complete: {removed} vectors removed, {len(new_metadata)} remaining")
        
        return {
            'before': before_count,
            'after': len(new_metadata),
            'removed': removed
        }
    
    def save(self):
        """Save index and metadata to disk"""
        if not FAISS_AVAILABLE:
            return
        
        try:
            if self.index:
                faiss.write_index(self.index, str(self.index_path))
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            with open(self.doc_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_map, f, ensure_ascii=False, indent=2)
            
            # Save index metadata
            meta_file = self.base_dir / "combined_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump({
                    'index_type': self._index_type,
                    'M': getattr(self, 'M', 32) if self._index_type == 'HNSW' else None
                }, f)
            
            logger.debug(f"💾 Saved index ({self.index.ntotal} vectors)")
        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")
    
    def get_statistics(self) -> Dict:
        """Get detailed statistics"""
        total = len(self.metadata)
        active = total - self._deleted_count
        
        from backend.config import Config
        
        return {
            'total_vectors': total,
            'active_vectors': active,
            'deleted_vectors': self._deleted_count,
            'deletion_ratio': self._deleted_count / total if total > 0 else 0,
            'total_documents': len(self.document_map),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dim,
            'index_type': self._index_type,
            'needs_rebuild': self._deleted_count > total * Config.AUTO_REBUILD_THRESHOLD,
            'documents': list(self.document_map.keys()),
            'faiss_available': FAISS_AVAILABLE
        }