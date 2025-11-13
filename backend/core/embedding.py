# backend/core/embedding.py
"""
✅ CLEANED: Embedding module - NO FILE SAVING
Chỉ generate embeddings và return data, không lưu file
"""
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import logging

from backend.core.chunking import normalize_hierarchy

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


def generate_embeddings(chunks: List[Dict[str, Any]], batch_size: int = 64, 
                       use_cache: bool = False) -> List[Dict[str, Any]]:
    """
    ✅ CLEANED: Generate embeddings - NO CACHING, NO FILE SAVING
    Pure processing function that returns data
    """
    if not chunks:
        return []
    
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
    
    logger.info(f"✅ Generated {len(result)} embeddings")
    return result


def generate_entity_embeddings(entities_dict: Dict[str, List[Dict]], 
                              knowledge_graph=None, batch_size: int = 64) -> List[Dict[str, Any]]:
    """
    ✅ CLEANED: Generate entity embeddings - NO FILE SAVING
    """
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
    """
    ✅ CLEANED: Generate relationship embeddings - NO FILE SAVING
    """
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