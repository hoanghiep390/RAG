# backend/core/embedding.py - OPTIMIZED VERSION
"""
✅ OPTIMIZED: Faster embedding with HNSW index and batch processing
"""
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import logging

from backend.utils.file_utils import save_to_json
from backend.core.chunking import process_document_to_chunks, normalize_hierarchy
from backend.utils.cache_utils import embedding_cache

logger = logging.getLogger(__name__)

# ==================== OPTIMIZED MODEL ====================
class OptimizedEmbeddingModel:
    """
    ✅ OPTIMIZED: Sentence transformer with GPU support and larger batches
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 384  # Faster processing
        
        logger.info(f"🚀 Embedding model loaded on {device}")
        if device == 'cuda':
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    def encode_batch(self, 
                     texts: List[str], 
                     batch_size: int = 64,
                     show_progress: bool = True,
                     normalize: bool = True) -> np.ndarray:
        """
        Encode texts with optimized batch size
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size (larger = faster on GPU)
            show_progress: Show progress bar
            normalize: Normalize embeddings (faster cosine search)
        
        Returns:
            Array of embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )


# Global model instance (singleton pattern)
_embedding_model: Optional[OptimizedEmbeddingModel] = None

def get_embedding_model() -> OptimizedEmbeddingModel:
    """Get or create singleton embedding model"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OptimizedEmbeddingModel()
    return _embedding_model


# ==================== OPTIMIZED VECTOR DATABASE ====================
class VectorDatabase:
    """
    ✅ OPTIMIZED: Vector database with HNSW index for fast search
    """
    
    def __init__(self, 
                 db_path: str = "faiss.index", 
                 metadata_path: str = "faiss_meta.json", 
                 dim: int = 384,
                 use_hnsw: bool = True):
        """
        Initialize vector database
        
        Args:
            db_path: Path to FAISS index file
            metadata_path: Path to metadata JSON
            dim: Embedding dimension
            use_hnsw: Use HNSW index (10-100x faster search)
        """
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.dim = dim
        self.use_hnsw = use_hnsw
        self.metadata = {}
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing database or create new one with optimized index"""
        try:
            import faiss
            import json
            import os
            
            if os.path.exists(self.db_path) and os.path.exists(self.metadata_path):
                # Load existing index
                self.index = faiss.read_index(self.db_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"📂 Loaded existing index: {self.index.ntotal} vectors")
            else:
                # Create new optimized index
                if self.use_hnsw:
                    # HNSW index: much faster search (10-100x)
                    M = 32  # Number of connections per layer
                    self.index = faiss.IndexHNSWFlat(self.dim, M)
                    self.index.hnsw.efConstruction = 200  # Build quality
                    self.index.hnsw.efSearch = 50  # Search quality
                    logger.info("🚀 Created HNSW index (fast search)")
                else:
                    # Fallback to flat index
                    self.index = faiss.IndexFlatL2(self.dim)
                    logger.warning("⚠️ Using flat index (slower search)")
                
                self.metadata = {}
                
        except ImportError:
            logger.error("FAISS not installed. Using in-memory storage.")
            self.index = None
            self.vectors = []
            self.metadata = {}
    
    def add_embedding(self, id: str, text: str, embedding: List[float], 
                     entity_name: str = None, entity_type: str = None,
                     chunk_id: str = None, **metadata):
        """Add a single embedding"""
        embedding_array = np.array([embedding], dtype=np.float32)
        
        if self.index is not None:
            idx = self.index.ntotal
            self.index.add(embedding_array)
        else:
            idx = len(self.vectors)
            self.vectors.append(embedding_array[0])
        
        
        self.metadata[str(idx)] = {
            'id': id,
            'text': text,
            'entity_name': entity_name,
            'entity_type': entity_type,
            'chunk_id': chunk_id,
            **metadata
        }
    
    def add_embeddings(self, embeddings: List[Dict[str, Any]]):
        """
        ✅ OPTIMIZED: Add multiple embeddings at once
        Much faster than adding one by one
        """
        if not embeddings:
            return
        
        # Extract embeddings and metadata
        vectors = np.array([emb['embedding'] for emb in embeddings], dtype=np.float32)
        
        if self.index is not None:
            start_idx = self.index.ntotal
            # Add all vectors at once (faster)
            self.index.add(vectors)
            
            # Store metadata
            for i, emb in enumerate(embeddings):
                idx = start_idx + i
                meta = {k: v for k, v in emb.items() if k != 'embedding'}
                self.metadata[str(idx)] = meta
        else:
            # Fallback
            for emb in embeddings:
                self.vectors.append(np.array(emb['embedding']))
                idx = len(self.vectors) - 1
                self.metadata[str(idx)] = {k: v for k, v in emb.items() if k != 'embedding'}
        
        logger.info(f"➕ Added {len(embeddings)} embeddings")
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for top-k most similar embeddings"""
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
            
            results = []
            for idx in top_indices:
                if str(idx) in self.metadata:
                    result = self.metadata[str(idx)].copy()
                    result['distance'] = float(distances[idx])
                    results.append(result)
        
        return results
    
    def save(self):
        """Save database and metadata"""
        import json
        
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, self.db_path)
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved: {self.db_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.index is not None:
            total_vectors = self.index.ntotal
        else:
            total_vectors = len(self.vectors)
        
        type_counts = {}
        for meta in self.metadata.values():
            entity_type = meta.get('entity_type', 'CHUNK')
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        return {
            'total_vectors': total_vectors,
            'dimension': self.dim,
            'index_type': 'HNSW' if self.use_hnsw else 'Flat',
            'entity_type_counts': type_counts,
            'total_metadata_entries': len(self.metadata)
        }


# ==================== OPTIMIZED EMBEDDING GENERATION ====================
def generate_embeddings(chunks: List[Dict[str, Any]], 
                       batch_size: int = 64,
                       use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    ✅ OPTIMIZED: Generate embeddings with caching and larger batches
    
    Args:
        chunks: List of chunks from chunking module
        batch_size: Batch size for encoding (larger = faster)
        use_cache: Use disk cache
    
    Returns:
        List of embeddings with metadata
    """
    if not chunks:
        return []
    
    # Check cache
    if use_cache:
        cache_key = f"chunks_{'_'.join([c.get('chunk_id', '')[:8] for c in chunks[:5]])}"
        cached = embedding_cache.get(cache_key)
        if cached is not None:
            logger.info(f"✅ Cache hit for {len(chunks)} chunks")
            return cached
    
    # Get model
    model = get_embedding_model()
    
    # Extract texts
    texts = [chunk['content'] for chunk in chunks]
    
    # Generate embeddings in batches
    logger.info(f"🔄 Generating embeddings for {len(chunks)} chunks (batch_size={batch_size})")
    embeddings_array = model.encode_batch(texts, batch_size=batch_size, show_progress=True)
    
    # Build output
    out = []
    for chunk, emb in zip(chunks, embeddings_array):
        hierarchy = chunk.get('hierarchy', '')
        hierarchy_str = normalize_hierarchy(hierarchy)
        
        out.append({
            'id': chunk.get('chunk_id', ''),
            'text': chunk.get('content', ''),
            'embedding': emb.tolist(),
            'chunk_id': chunk.get('chunk_id', ''),
            'tokens': chunk.get('tokens', 0),
            'order': chunk.get('order', 0),
            'hierarchy': hierarchy_str,
            'hierarchy_list': hierarchy if isinstance(hierarchy, list) else [hierarchy],
            'file_path': chunk.get('file_path', ''),
            'file_type': chunk.get('file_type', ''),
            'entity_type': 'CHUNK'
        })
    
    # Cache result
    if use_cache:
        embedding_cache.set(cache_key, out)
    
    logger.info(f"✅ Generated {len(out)} embeddings")
    return out

# PART 2: Entity and Relationship Embeddings + Main Functions

def generate_entity_embeddings(entities_dict: Dict[str, List[Dict]], 
                              knowledge_graph=None,
                              batch_size: int = 64) -> List[Dict[str, Any]]:
    """
    ✅ OPTIMIZED: Generate embeddings for entities with batching
    """
    embeddings = []
    texts = []
    entity_metadata = []
    
    for chunk_id, chunk_entities in entities_dict.items():
        for entity in chunk_entities:
            entity_name = entity['entity_name']
            entity_type = entity['entity_type']
            description = entity.get('description', '')
            
            # Enhanced with graph context if available
            if knowledge_graph and knowledge_graph.G.has_node(entity_name):
                node_data = knowledge_graph.G.nodes[entity_name]
                description = node_data.get('description', description)
            
            # Format: "EntityName (TYPE): description"
            text = f"{entity_name} ({entity_type}): {description}"
            texts.append(text)
            
            entity_metadata.append({
                'id': f"entity_{entity_name}",
                'entity_name': entity_name,
                'entity_type': entity_type,
                'chunk_id': chunk_id
            })
    
    if not texts:
        return []
    
    # Generate embeddings in batch
    model = get_embedding_model()
    logger.info(f"🔄 Generating {len(texts)} entity embeddings")
    embeddings_array = model.encode_batch(texts, batch_size=batch_size, show_progress=True)
    
    result = []
    for i, emb in enumerate(embeddings_array):
        result.append({
            'id': entity_metadata[i]['id'],
            'text': texts[i],
            'embedding': emb.tolist(),
            'entity_name': entity_metadata[i]['entity_name'],
            'entity_type': entity_metadata[i]['entity_type'],
            'chunk_id': entity_metadata[i]['chunk_id']
        })
    
    logger.info(f"✅ Generated {len(result)} entity embeddings")
    return result


def generate_relationship_embeddings(relationships_dict: Dict[str, List[Dict]],
                                    batch_size: int = 64) -> List[Dict[str, Any]]:
    """
    ✅ OPTIMIZED: Generate embeddings for relationships with batching
    """
    embeddings = []
    texts = []
    rel_metadata = []
    
    for chunk_id, chunk_rels in relationships_dict.items():
        for rel in chunk_rels:
            source = rel['source_id']
            target = rel['target_id']
            description = rel.get('description', '')
            
            # Format: "Source -> Target: description"
            text = f"{source} -> {target}: {description}"
            texts.append(text)
            
            rel_metadata.append({
                'id': f"rel_{source}_{target}_{chunk_id}",
                'source': source,
                'target': target,
                'chunk_id': chunk_id
            })
    
    if not texts:
        return []
    
    # Generate embeddings in batch
    model = get_embedding_model()
    logger.info(f"🔄 Generating {len(texts)} relationship embeddings")
    embeddings_array = model.encode_batch(texts, batch_size=batch_size, show_progress=True)
    
    result = []
    for i, emb in enumerate(embeddings_array):
        result.append({
            'id': rel_metadata[i]['id'],
            'text': texts[i],
            'embedding': emb.tolist(),
            'source': rel_metadata[i]['source'],
            'target': rel_metadata[i]['target'],
            'chunk_id': rel_metadata[i]['chunk_id'],
            'entity_type': 'RELATIONSHIP'
        })
    
    logger.info(f"✅ Generated {len(result)} relationship embeddings")
    return result


def search_similar(query: str, 
                  vector_db: VectorDatabase, 
                  top_k: int = 5, 
                  filter_type: str = None) -> List[Dict[str, Any]]:
    """
    Search for similar items
    
    Args:
        query: Search query
        vector_db: Vector database instance
        top_k: Number of results
        filter_type: Filter by entity_type
    
    Returns:
        List of search results
    """
    model = get_embedding_model()
    query_embedding = model.encode_batch([query], show_progress=False)[0]
    results = vector_db.search(query_embedding.tolist(), top_k=top_k * 2)
    
    # Filter by type if specified
    if filter_type:
        results = [r for r in results if r.get('entity_type') == filter_type]
    
    return results[:top_k]


def process_file(filepath: str, 
                entities_dict: Optional[Dict] = None,
                relationships_dict: Optional[Dict] = None,
                knowledge_graph=None,
                batch_size: int = 64,
                use_hnsw: bool = True) -> VectorDatabase:
    """
    ✅ OPTIMIZED: Process file with optimized settings
    
    Args:
        filepath: Path to file
        entities_dict: Extracted entities
        relationships_dict: Extracted relationships
        knowledge_graph: Knowledge graph instance
        batch_size: Batch size for embeddings
        use_hnsw: Use HNSW index for fast search
    
    Returns:
        VectorDatabase instance
    """
    logger.info(f"🔄 Processing file: {filepath}")
    
    # Generate chunk embeddings
    chunks = process_document_to_chunks(filepath)
    chunk_embeddings = generate_embeddings(chunks, batch_size=batch_size)
    
    # Initialize optimized vector database
    dim = len(chunk_embeddings[0]["embedding"]) if chunk_embeddings else 384
    vector_db = VectorDatabase(
        db_path="faiss.index", 
        metadata_path="faiss_meta.json", 
        dim=dim,
        use_hnsw=use_hnsw
    )
    
    # Add chunk embeddings
    vector_db.add_embeddings(chunk_embeddings)
    logger.info(f"✅ Added {len(chunk_embeddings)} chunk embeddings")
    
    # Add entity embeddings if provided
    if entities_dict:
        entity_embeddings = generate_entity_embeddings(
            entities_dict, 
            knowledge_graph, 
            batch_size=batch_size
        )
        if entity_embeddings:
            vector_db.add_embeddings(entity_embeddings)
            save_to_json(entity_embeddings, "entity_embeddings.json")
    
    # Add relationship embeddings if provided
    if relationships_dict:
        rel_embeddings = generate_relationship_embeddings(
            relationships_dict, 
            batch_size=batch_size
        )
        if rel_embeddings:
            vector_db.add_embeddings(rel_embeddings)
            save_to_json(rel_embeddings, "relationship_embeddings.json")
    
    # Save all embeddings
    all_embeddings = {
        'chunks': chunk_embeddings,
        'entities': entity_embeddings if entities_dict else [],
        'relationships': rel_embeddings if relationships_dict else []
    }
    save_to_json(all_embeddings, "embedding_output.json")
    
    # Save database
    vector_db.save()
    
    # Print statistics
    stats = vector_db.get_statistics()
    logger.info(f"📊 Vector DB Stats: {stats}")
    
    return vector_db


# ==================== BATCH PROCESSING UTILITIES ====================
def process_multiple_files(filepaths: List[str],
                          entities_dicts: Optional[List[Dict]] = None,
                          relationships_dicts: Optional[List[Dict]] = None,
                          batch_size: int = 64,
                          use_hnsw: bool = True) -> VectorDatabase:
    """
    ✅ NEW: Process multiple files into single vector database
    
    Useful for bulk document processing
    """
    from pathlib import Path
    
    logger.info(f"🔄 Processing {len(filepaths)} files in batch")
    
    # Initialize database
    vector_db = VectorDatabase(
        db_path="faiss_combined.index",
        metadata_path="faiss_combined_meta.json",
        dim=384,
        use_hnsw=use_hnsw
    )
    
    for i, filepath in enumerate(filepaths):
        try:
            logger.info(f"📄 [{i+1}/{len(filepaths)}] {Path(filepath).name}")
            
            # Process chunks
            chunks = process_document_to_chunks(filepath)
            chunk_embeddings = generate_embeddings(chunks, batch_size=batch_size)
            vector_db.add_embeddings(chunk_embeddings)
            
            # Process entities if provided
            if entities_dicts and i < len(entities_dicts):
                entity_embeds = generate_entity_embeddings(
                    entities_dicts[i], 
                    batch_size=batch_size
                )
                if entity_embeds:
                    vector_db.add_embeddings(entity_embeds)
            
            # Process relationships if provided
            if relationships_dicts and i < len(relationships_dicts):
                rel_embeds = generate_relationship_embeddings(
                    relationships_dicts[i],
                    batch_size=batch_size
                )
                if rel_embeds:
                    vector_db.add_embeddings(rel_embeds)
        
        except Exception as e:
            logger.error(f"❌ Failed to process {filepath}: {e}")
            continue
    
    # Save
    vector_db.save()
    logger.info(f"✅ Processed {len(filepaths)} files")
    logger.info(f"📊 {vector_db.get_statistics()}")
    
    return vector_db


# ==================== MEMORY CLEANUP ====================
def clear_embedding_cache():
    """Clear embedding cache to free memory"""
    embedding_cache.clear()
    logger.info("🗑️ Cleared embedding cache")


def reset_model():
    """Reset global model instance"""
    global _embedding_model
    if _embedding_model is not None:
        del _embedding_model
        _embedding_model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🗑️ Reset embedding model")