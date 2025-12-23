# backend/core/embedding.py
"""
Embedding generation with Config integration 
"""
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import torch
import logging
from backend.config import Config

logger = logging.getLogger(__name__)

_model = None

def get_model():
    """Get or initialize embedding model (singleton)"""
    global _model
    if _model is None:
        from backend.config import Config
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = Config.EMBEDDING_MODEL
        
        logger.info(f"📦 Đang tải mô hình embedding: {model_name} trên {device}")
        _model = SentenceTransformer(model_name, device=device)
        
        test_emb = _model.encode(["test"])
        actual_dim = test_emb.shape[1]
        
        if actual_dim != Config.EMBEDDING_DIM:
            logger.warning(
                f"⚠️ Không khớp kích thước mô hình! "
                f"Config: {Config.EMBEDDING_DIM}, Thực tế: {actual_dim}"
            )
        
        logger.info(f"✅ Đã tải mô hình embedding: {model_name} ({actual_dim}-chiều)")
    
    return _model

def generate_embeddings(chunks: List[Dict], batch_size: int = None) -> List[Dict]:
    """Generate embeddings for chunks"""
    if not chunks:
        logger.warning("⚠️ Không có chunks để tạo embedding")
        return []
    
    from backend.config import Config
    batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
    
    try:
        model = get_model()
        texts = [c['content'] for c in chunks]
        
        logger.info(f"📊 Đang tạo embeddings cho {len(texts)} chunks (batch={batch_size})...")
        
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
        
        logger.info(f"✅ Đã tạo {len(result)} embeddings")
        return result
    
    except Exception as e:
        logger.error(f"❌ Không thể tạo embeddings: {e}")
        return []

def generate_entity_embeddings(entities_dict: Dict, batch_size: int = None) -> List[Dict]:
    """Generate embeddings for entities"""
    from backend.config import Config
    batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
    
    texts, metadata = [], []
    
    for chunk_id, entities in entities_dict.items():
        for entity in entities:
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
        logger.warning("⚠️ Không có entities để tạo embedding")
        return []
    
    try:
        logger.info(f"📊 Đang tạo embeddings cho {len(texts)} entities (batch={batch_size})...")
        
        model = get_model()
        
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
        
        logger.info(f"✅ Đã tạo {len(result)} entity embeddings")
        return result
    
    except Exception as e:
        logger.error(f"❌ Không thể tạo entity embeddings: {e}")
        return []


def generate_relationship_embeddings(relationships_dict: Dict, batch_size: int = None) -> List[Dict]:
    """Generate embeddings for relationships (global retrieval)
    
    Args:
        relationships_dict: Dict of (source, target) -> list of relationships
        batch_size: Batch size for encoding
    
    Returns:
        List of relationship embedding dicts with entity_type='RELATIONSHIP'
    """
    from backend.config import Config
    batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
    
    texts, metadata = [], []
    
    for (source, target), rels in relationships_dict.items():
        for rel in rels:
            # Create rich text for relationship
            rel_text = (
                f"{source} -> {target}: "
                f"{rel.get('description', '')} "
                f"[{rel.get('keywords', '')}]"
            )
            texts.append(rel_text)
            
            metadata.append({
                'id': f"rel_{source}_{target}_{rel.get('chunk_id', '')}",
                'source_id': source,
                'target_id': target,
                'description': rel.get('description', ''),
                'keywords': rel.get('keywords', ''),
                'weight': rel.get('weight', 1.0),
                'chunk_id': rel.get('chunk_id', ''),
                'entity_type': 'RELATIONSHIP'  
            })
    
    if not texts:
        logger.warning("⚠️ Không có relationships để tạo embedding")
        return []
    
    try:
        logger.info(f"🔗 Đang tạo embeddings cho {len(texts)} relationships (batch={batch_size})...")
        
        model = get_model()
        
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
        
        logger.info(f"✅ Đã tạo {len(result)} relationship embeddings")
        return result
    
    except Exception as e:
        logger.error(f"❌ Không thể tạo relationship embeddings: {e}")
        return []

# Remaining functions unchanged...
def generate_text_embedding(text: str) -> np.ndarray:
    """Generate embedding for a single text (for queries)"""
    try:
        model = get_model()
        embedding = model.encode([text], normalize_embeddings=True)[0]
        return embedding
    except Exception as e:
        logger.error(f"❌ Không thể tạo text embedding: {e}")
        return None

def batch_encode_texts(texts: List[str], batch_size: int = None) -> np.ndarray:
    """Batch encode multiple texts"""
    batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE  
    
    if not texts:
        return np.array([])
    
    try:
        model = get_model()
        
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,        
            show_progress_bar=len(texts) > 50  
        )
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Mã hóa batch thất bại: {e}")
        return np.array([])
def get_embedding_dimension() -> int:
    """Get actual embedding dimension from loaded model"""
    try:
        model = get_model()
        test_emb = model.encode(["test"], normalize_embeddings=True)
        return test_emb.shape[1]
    except Exception as e:
        logger.warning(f"⚠️ Sử dụng kích thước từ config: {e}")
        return Config.EMBEDDING_DIM