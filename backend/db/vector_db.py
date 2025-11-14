# ==========================================
# backend/db/vector_db.py 
# ==========================================
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Optional
import importlib

try:
    faiss = importlib.import_module('faiss')
    FAISS_AVAILABLE = True
except Exception:
    # If importing real faiss fails, mark FAISS_AVAILABLE False and provide a NumPy-based fallback
    FAISS_AVAILABLE = False

    class IndexFlatL2:
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
            # compute L2 distances between q (m x d) and self._vectors (n x d) -> (m x n)
            dists = np.sum((q[:, None, :] - self._vectors[None, :, :]) ** 2, axis=2)
            idx = np.argsort(dists, axis=1)[:, :k]
            dist_sorted = np.take_along_axis(dists, idx, axis=1)
            return dist_sorted, idx

        def reconstruct(self, index):
            return self._vectors[int(index)].copy()

    def write_index(index, path):
        arr = getattr(index, '_vectors', np.zeros((0, index.dim), dtype=np.float32))
        # Save as .npy for the simple fallback
        np.save(str(path), arr)

    def read_index(path):
        p = str(path)
        # try with and without .npy extension
        try:
            arr = np.load(p, allow_pickle=False)
        except Exception:
            arr = np.load(p + '.npy', allow_pickle=False)
        idx = IndexFlatL2(arr.shape[1] if arr.ndim == 2 else 0)
        if arr.size:
            idx.add(arr)
        return idx

    faiss = type('faiss_module', (), {
        'IndexFlatL2': IndexFlatL2,
        'write_index': staticmethod(write_index),
        'read_index': staticmethod(read_index)
    })()

class VectorDatabase:
    """Simple FAISS wrapper"""
    
    def __init__(self, user_id: str, dim: int = 384, use_hnsw: bool = False, auto_save: bool = True):
        """Initialize the vector DB for a user.

        Args:
            user_id: owner id used to create per-user storage folder
            dim: embedding dimension
            use_hnsw: if True, attempt to create an HNSW index (when available)
            auto_save: whether to automatically save changes to disk
        """
        self.user_id = user_id
        self.dim = dim
        self.use_hnsw = bool(use_hnsw)
        self.auto_save = auto_save
        
        self.base_dir = Path("backend/data") / user_id / "vectors"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.base_dir / "combined.index"
        self.metadata_path = self.base_dir / "combined_metadata.json"
        self.doc_map_path = self.base_dir / "document_map.json"
        
        self.index = None
        self.metadata = {}
        self.document_map = {}
        self._deleted_count = 0
        
        self._load_or_create()
    
    def _load_or_create(self):
        """Load hoặc tạo mới"""
        if not FAISS_AVAILABLE:
            return
        
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            if self.doc_map_path.exists():
                with open(self.doc_map_path, 'r', encoding='utf-8') as f:
                    self.document_map = json.load(f)
            
            self._deleted_count = sum(1 for m in self.metadata.values() if m.get('_deleted'))
        else:
            # Create index. Prefer HNSW if requested and available, else flat L2.
            if self.use_hnsw:
                try:
                    # Try common FAISS HNSW constructors
                    if hasattr(faiss, 'IndexHNSWFlat'):
                        # default M (connectivity) ~32 if constructor supports it
                        try:
                            self.index = faiss.IndexHNSWFlat(self.dim, 32)
                        except TypeError:
                            # fallback constructor without M
                            self.index = faiss.IndexHNSWFlat(self.dim)
                    elif hasattr(faiss, 'index_factory'):
                        # use factory string (e.g. "HNSW32") if available
                        try:
                            self.index = faiss.index_factory(self.dim, "HNSW32")
                        except Exception:
                            self.index = faiss.IndexFlatL2(self.dim)
                    else:
                        self.index = faiss.IndexFlatL2(self.dim)
                except Exception:
                    # If anything goes wrong, fallback to flat index
                    self.index = faiss.IndexFlatL2(self.dim)
            else:
                self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = {}
            self.document_map = {}
            self._deleted_count = 0
    
    def add_document_embeddings_batch(self, doc_id: str, filename: str, 
                                      embeddings: List[Dict], tags: List[str] = None) -> int:
        """Batch add embeddings"""
        if not FAISS_AVAILABLE or not embeddings:
            return 0
        
        start_idx = self.index.ntotal
        
        # Add vectors
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
        
        # Update doc map
        end_idx = start_idx + len(embeddings) - 1
        self.document_map[doc_id] = {
            'filename': filename,
            'chunk_count': len(embeddings),
            'embedding_range': [start_idx, end_idx],
            'tags': tags or []
        }
        
        if self.auto_save:
            self.save()
        
        return len(embeddings)
    
    def search(self, query_embedding: List[float], top_k: int = 5,
               doc_ids: Optional[List[str]] = None, skip_deleted: bool = True) -> List[Dict]:
        """Vector search"""
        if not FAISS_AVAILABLE or not self.index or self.index.ntotal == 0:
            return []
        
        query_array = np.array([query_embedding], dtype=np.float32)
        search_k = min(top_k * 5, self.index.ntotal)
        distances, indices = self.index.search(query_array, search_k)
        
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
        """Search by text"""
        from backend.core.embedding import get_model
        model = get_model()
        query_emb = model.encode([query_text], show_progress=False)[0]
        return self.search(query_emb.tolist(), top_k=top_k, **kwargs)
    
    def delete_document(self, doc_id: str) -> Dict:
        """Mark deleted"""
        if doc_id not in self.document_map:
            return {'marked': 0, 'needs_rebuild': False}
        
        start, end = self.document_map[doc_id]['embedding_range']
        marked_count = 0
        
        for idx in range(start, end + 1):
            if str(idx) in self.metadata:
                self.metadata[str(idx)]['_deleted'] = True
                marked_count += 1
        
        del self.document_map[doc_id]
        self._deleted_count += marked_count
        
        total = len(self.metadata)
        needs_rebuild = self._deleted_count > total * 0.2
        
        if self.auto_save:
            self.save()
        
        return {
            'marked': marked_count,
            'needs_rebuild': needs_rebuild,
            'total_deleted': self._deleted_count,
            'total_vectors': total
        }
    
    def rebuild_index(self) -> Dict:
        """Rebuild index (remove deleted)"""
        if not FAISS_AVAILABLE:
            return {}
        
        before_count = len(self.metadata)
        
        # Collect active
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
        
        return {
            'before': before_count,
            'after': len(new_metadata),
            'removed': before_count - len(new_metadata)
        }
    
    def save(self):
        """Save to disk"""
        if not FAISS_AVAILABLE:
            return
        
        if self.index:
            faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        with open(self.doc_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.document_map, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> Dict:
        """Get stats"""
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
            'needs_rebuild': self._deleted_count > total * 0.2,
            'documents': list(self.document_map.keys())
        }

