# backend/core/embedding.py
"""
Embedding generation with Config integration - FIXED show_progress issue
"""
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

_model = None

def get_model():
    """Get or initialize embedding model (singleton)"""
    global _model
    if _model is None:
        from backend.config import Config
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = Config.EMBEDDING_MODEL
        
        logger.info(f"Loading embedding model: {model_name} on {device}")
        _model = SentenceTransformer(model_name, device=device)
        
        # Verify dimension
        test_emb = _model.encode(["test"])
        actual_dim = test_emb.shape[1]
        
        if actual_dim != Config.EMBEDDING_DIM:
            logger.warning(
                f"⚠️ Model dimension mismatch! "
                f"Config: {Config.EMBEDDING_DIM}, Actual: {actual_dim}"
            )
        
        logger.info(f"✅ Embedding model loaded: {model_name} ({actual_dim}-dim)")
    
    return _model

def generate_embeddings(chunks: List[Dict], batch_size: int = None) -> List[Dict]:
    """✅ FIXED: Generate embeddings without show_progress parameter
    
    Args:
        chunks: List of chunk dictionaries with 'content' and 'chunk_id'
        batch_size: Batch size for encoding (default from Config)
    
    Returns:
        List of embedding dictionaries
    """
    if not chunks:
        logger.warning("⚠️ No chunks provided for embedding")
        return []
    
    from backend.config import Config
    batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
    
    try:
        model = get_model()
        texts = [c['content'] for c in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks (batch={batch_size})...")
        
        # ✅ FIX: Remove show_progress_bar parameter if not supported
        try:
            embeddings = model.encode(
                texts, 
                batch_size=batch_size, 
                normalize_embeddings=True,
                show_progress_bar=True
            )
        except TypeError:
            # Model doesn't support show_progress_bar, encode without it
            logger.debug("Model doesn't support show_progress_bar, encoding without it")
            embeddings = model.encode(
                texts, 
                batch_size=batch_size, 
                normalize_embeddings=True
            )
        
        result = []
        for chunk, emb in zip(chunks, embeddings):
            result.append({
                'id': chunk['chunk_id'],
                'text': chunk['content'],
                'embedding': emb.tolist(),
                'chunk_id': chunk['chunk_id'],
                'tokens': chunk.get('tokens', 0),
                'order': chunk.get('order', 0),
                'file_path': chunk.get('file_path', ''),
                'file_type': chunk.get('file_type', ''),
                'entity_type': 'CHUNK'
            })
        
        logger.info(f"✅ Generated {len(result)} embeddings")
        return result
    
    except Exception as e:
        logger.error(f"❌ Failed to generate embeddings: {e}")
        return []

def generate_entity_embeddings(entities_dict: Dict, batch_size: int = None) -> List[Dict]:
    """✅ FIXED: Generate embeddings for entities without show_progress
    
    Args:
        entities_dict: Dictionary of entity_name -> list of entity dicts
        batch_size: Batch size for encoding (default from Config)
    
    Returns:
        List of entity embedding dictionaries
    """
    from backend.config import Config
    batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
    
    texts, metadata = [], []
    
    for chunk_id, entities in entities_dict.items():
        for entity in entities:
            # Create rich text representation
            entity_text = (
                f"{entity['entity_name']} "
                f"({entity['entity_type']}): "
                f"{entity.get('description', '')}"
            )
            texts.append(entity_text)
            
            metadata.append({
                'id': f"entity_{entity['entity_name']}",
                'entity_name': entity['entity_name'],
                'entity_type': entity['entity_type'],
                'chunk_id': chunk_id
            })
    
    if not texts:
        logger.warning("⚠️ No entities to embed")
        return []
    
    try:
        logger.info(f"Generating embeddings for {len(texts)} entities (batch={batch_size})...")
        
        model = get_model()
        
        # ✅ FIX: Try with show_progress_bar, fallback without it
        try:
            embeddings = model.encode(
                texts, 
                batch_size=batch_size, 
                normalize_embeddings=True,
                show_progress_bar=True
            )
        except TypeError:
            embeddings = model.encode(
                texts, 
                batch_size=batch_size, 
                normalize_embeddings=True
            )
        
        result = [
            {**meta, 'text': text, 'embedding': emb.tolist()}
            for text, emb, meta in zip(texts, embeddings, metadata)
        ]
        
        logger.info(f"✅ Generated {len(result)} entity embeddings")
        return result
    
    except Exception as e:
        logger.error(f"❌ Failed to generate entity embeddings: {e}")
        return []

def generate_text_embedding(text: str) -> np.ndarray:
    """✅ FIXED: Generate embedding for a single text (for queries)
    
    Args:
        text: Input text
    
    Returns:
        Numpy array of embedding
    """
    try:
        model = get_model()
        
        # ✅ FIX: Remove show_progress
        embedding = model.encode([text], normalize_embeddings=True)[0]
        return embedding
    except Exception as e:
        logger.error(f"❌ Failed to generate text embedding: {e}")
        return None

def batch_encode_texts(texts: List[str], batch_size: int = None) -> np.ndarray:
    """✅ FIXED: Batch encode multiple texts
    
    Args:
        texts: List of texts to encode
        batch_size: Batch size (default from Config)
    
    Returns:
        Numpy array of embeddings (N x D)
    """
    from backend.config import Config
    batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
    
    if not texts:
        return np.array([])
    
    try:
        model = get_model()
        
        # ✅ FIX: Try with show_progress_bar, fallback without
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100
            )
        except TypeError:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True
            )
        
        return embeddings
    except Exception as e:
        logger.error(f"❌ Batch encoding failed: {e}")
        return np.array([])

def get_embedding_dimension() -> int:
    """Get actual embedding dimension from loaded model
    
    Returns:
        Embedding dimension
    """
    try:
        model = get_model()
        test_emb = model.encode(["test"])
        return test_emb.shape[1]
    except Exception as e:
        logger.error(f"❌ Failed to get embedding dimension: {e}")
        from backend.config import Config
        return Config.EMBEDDING_DIM