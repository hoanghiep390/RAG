# backend/core/embedding.py 
"""
✅ FIXED: Simplified embedding with proper directory handling
"""
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import logging
from pathlib import Path
from backend.utils.file_utils import save_to_json
from backend.core.chunking import process_document_to_chunks, normalize_hierarchy
from backend.utils.cache_utils import embedding_cache

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Optimized embedding model with GPU support"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = 384
        logger.info(f"🚀 Embedding model on {self.device}")
    
    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Encode texts in batches"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

_model: Optional[EmbeddingModel] = None

def get_model() -> EmbeddingModel:
    """Get singleton model"""
    global _model
    if _model is None:
        _model = EmbeddingModel()
    return _model

class VectorDatabase:
    """✅ FIXED: Optimized vector database with proper directory handling"""
    
    def __init__(self, db_path: str = "faiss.index", metadata_path: str = "faiss_meta.json", 
                 dim: int = 384, use_hnsw: bool = True):
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.dim = dim
        self.use_hnsw = use_hnsw
        self.metadata = {}
        
        # ✅ FIX: Create parent directories first
        db_dir = Path(self.db_path).parent
        meta_dir = Path(self.metadata_path).parent
        
        if str(db_dir) != '.':
            db_dir.mkdir(parents=True, exist_ok=True)
        if str(meta_dir) != '.':
            meta_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_or_create()
    
    def _load_or_create(self):
        """Load or create optimized index"""
        try:
            import faiss
            import json
            import os
            
            if os.path.exists(self.db_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.db_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"📂 Loaded: {self.index.ntotal} vectors")
            else:
                if self.use_hnsw:
                    self.index = faiss.IndexHNSWFlat(self.dim, 32)
                    self.index.hnsw.efConstruction = 200
                    self.index.hnsw.efSearch = 50
                    logger.info("🚀 Created HNSW index")
                else:
                    self.index = faiss.IndexFlatL2(self.dim)
                    logger.warning("⚠️ Using flat index")
                self.metadata = {}
        except ImportError:
            logger.error("FAISS not installed")
            self.index = None
            self.vectors = []
            self.metadata = {}
    
    def add_embedding(self, id: str, text: str, embedding: List[float], 
                     entity_name: str = None, entity_type: str = None,
                     chunk_id: str = None, **kwargs):
        """Add single embedding"""
        embedding_array = np.array([embedding], dtype=np.float32)
        
        if self.index is not None:
            idx = self.index.ntotal
            self.index.add(embedding_array)
        else:
            idx = len(self.vectors)
            self.vectors.append(embedding_array[0])
        
        self.metadata[str(idx)] = {
            'id': id, 'text': text, 'entity_name': entity_name,
            'entity_type': entity_type, 'chunk_id': chunk_id, **kwargs
        }
    
    def add_embeddings(self, embeddings: List[Dict[str, Any]]):
        """Add multiple embeddings (batch)"""
        if not embeddings:
            return
        
        vectors = np.array([e['embedding'] for e in embeddings], dtype=np.float32)
        
        if self.index is not None:
            start_idx = self.index.ntotal
            self.index.add(vectors)
            
            for i, emb in enumerate(embeddings):
                meta = {k: v for k, v in emb.items() if k != 'embedding'}
                self.metadata[str(start_idx + i)] = meta
        else:
            for emb in embeddings:
                self.vectors.append(np.array(emb['embedding']))
                idx = len(self.vectors) - 1
                self.metadata[str(idx)] = {k: v for k, v in emb.items() if k != 'embedding'}
        
        logger.info(f"➕ Added {len(embeddings)} embeddings")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search top-k similar embeddings"""
        query_array = np.array([query_embedding], dtype=np.float32)
        
        if self.index is not None:
            distances, indices = self.index.search(query_array, top_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if str(idx) in self.metadata:
                    result = self.metadata[str(idx)].copy()
                    result['distance'] = float(dist)
                    results.append(result)
        else:
            if not self.vectors:
                return []
            vectors_array = np.array(self.vectors)
            distances = np.linalg.norm(vectors_array - query_array, axis=1)
            top_indices = np.argsort(distances)[:top_k]
            results = [
                {**self.metadata[str(idx)], 'distance': float(distances[idx])}
                for idx in top_indices if str(idx) in self.metadata
            ]
        
        return results
    
    def save(self):
        """✅ FIXED: Save database with proper directory handling"""
        import json
        from pathlib import Path
        
        # ✅ FIX: Ensure parent directories exist before writing
        db_path = Path(self.db_path)
        meta_path = Path(self.metadata_path)
        
        # Create parent directories
        if not db_path.parent.exists():
            logger.warning(f"Parent dir missing, creating: {db_path.parent}")
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not meta_path.parent.exists():
            logger.warning(f"Parent dir missing, creating: {meta_path.parent}")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            import faiss
            try:
                faiss.write_index(self.index, str(db_path))
                logger.info(f"💾 Saved index: {db_path}")
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
                raise
        
        # Save metadata
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved metadata: {meta_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        total = self.index.ntotal if self.index else len(self.vectors)
        type_counts = {}
        for meta in self.metadata.values():
            t = meta.get('entity_type', 'CHUNK')
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            'total_vectors': total,
            'dimension': self.dim,
            'index_type': 'HNSW' if self.use_hnsw else 'Flat',
            'entity_type_counts': type_counts,
            'total_metadata_entries': len(self.metadata)
        }

def generate_embeddings(chunks: List[Dict[str, Any]], batch_size: int = 64, 
                       use_cache: bool = True) -> List[Dict[str, Any]]:
    """Generate embeddings with caching"""
    if not chunks:
        return []
    
    if use_cache:
        cache_key = f"chunks_{'_'.join([c.get('chunk_id', '')[:8] for c in chunks[:5]])}"
        cached = embedding_cache.get(cache_key)
        if cached:
            logger.info(f"✅ Cache hit: {len(chunks)} chunks")
            return cached
    
    model = get_model()
    texts = [c['content'] for c in chunks]
    
    logger.info(f"🔄 Generating {len(chunks)} embeddings (batch={batch_size})")
    embeddings_array = model.encode(texts, batch_size=batch_size)
    
    result = []
    for chunk, emb in zip(chunks, embeddings_array):
        hierarchy = chunk.get('hierarchy', '')
        result.append({
            'id': chunk.get('chunk_id', ''),
            'text': chunk.get('content', ''),
            'embedding': emb.tolist(),
            'chunk_id': chunk.get('chunk_id', ''),
            'tokens': chunk.get('tokens', 0),
            'order': chunk.get('order', 0),
            'hierarchy': normalize_hierarchy(hierarchy),
            'hierarchy_list': hierarchy if isinstance(hierarchy, list) else [hierarchy],
            'file_path': chunk.get('file_path', ''),
            'file_type': chunk.get('file_type', ''),
            'entity_type': 'CHUNK'
        })
    
    if use_cache:
        embedding_cache.set(cache_key, result)
    
    logger.info(f"✅ Generated {len(result)} embeddings")
    return result

def generate_entity_embeddings(entities_dict: Dict[str, List[Dict]], 
                              knowledge_graph=None, batch_size: int = 64) -> List[Dict[str, Any]]:
    """Generate entity embeddings"""
    texts = []
    metadata = []
    
    for chunk_id, entities in entities_dict.items():
        for entity in entities:
            name = entity['entity_name']
            etype = entity['entity_type']
            desc = entity.get('description', '')
            
            if knowledge_graph and knowledge_graph.G.has_node(name):
                desc = knowledge_graph.G.nodes[name].get('description', desc)
            
            texts.append(f"{name} ({etype}): {desc}")
            metadata.append({'id': f"entity_{name}", 'entity_name': name, 
                           'entity_type': etype, 'chunk_id': chunk_id})
    
    if not texts:
        return []
    
    model = get_model()
    logger.info(f"🔄 Generating {len(texts)} entity embeddings")
    embeddings_array = model.encode(texts, batch_size=batch_size)
    
    result = [
        {'id': meta['id'], 'text': text, 'embedding': emb.tolist(), **meta}
        for text, emb, meta in zip(texts, embeddings_array, metadata)
    ]
    
    logger.info(f"✅ Generated {len(result)} entity embeddings")
    return result

def generate_relationship_embeddings(relationships_dict: Dict[str, List[Dict]], 
                                    batch_size: int = 64) -> List[Dict[str, Any]]:
    """Generate relationship embeddings"""
    texts = []
    metadata = []
    
    for chunk_id, rels in relationships_dict.items():
        for rel in rels:
            source = rel['source_id']
            target = rel['target_id']
            desc = rel.get('description', '')
            
            texts.append(f"{source} -> {target}: {desc}")
            metadata.append({'id': f"rel_{source}_{target}_{chunk_id}", 
                           'source': source, 'target': target, 'chunk_id': chunk_id})
    
    if not texts:
        return []
    
    model = get_model()
    logger.info(f"🔄 Generating {len(texts)} relationship embeddings")
    embeddings_array = model.encode(texts, batch_size=batch_size)
    
    result = [
        {'id': meta['id'], 'text': text, 'embedding': emb.tolist(), 
         'entity_type': 'RELATIONSHIP', **meta}
        for text, emb, meta in zip(texts, embeddings_array, metadata)
    ]
    
    logger.info(f"✅ Generated {len(result)} relationship embeddings")
    return result

def search_similar(query: str, vector_db: VectorDatabase, top_k: int = 5, 
                  filter_type: str = None) -> List[Dict[str, Any]]:
    """Search similar items"""
    model = get_model()
    query_emb = model.encode([query], show_progress=False)[0]
    results = vector_db.search(query_emb.tolist(), top_k=top_k * 2)
    
    if filter_type:
        results = [r for r in results if r.get('entity_type') == filter_type]
    
    return results[:top_k]

def clear_cache():
    """Clear embedding cache"""
    embedding_cache.clear()
    logger.info("🗑️ Cleared cache")

def reset_model():
    """Reset model"""
    global _model
    if _model:
        del _model
        _model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🗑️ Reset model")