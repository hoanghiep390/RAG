from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np
import torch

_model = None

def get_model():
    global _model
    if _model is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return _model

def generate_embeddings(chunks: List[Dict], batch_size: int = 128) -> List[Dict]:
    """Tạo embeddings cho chunks"""
    if not chunks:
        return []
    
    model = get_model()
    texts = [c['content'] for c in chunks]
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    
    result = []
    for chunk, emb in zip(chunks, embeddings):
        result.append({
            'id': chunk['chunk_id'],
            'text': chunk['content'],
            'embedding': emb.tolist(),
            'chunk_id': chunk['chunk_id'],
            'tokens': chunk['tokens'],
            'order': chunk['order'],
            'file_path': chunk['file_path'],
            'file_type': chunk['file_type'],
            'entity_type': 'CHUNK'
        })
    
    return result

def generate_entity_embeddings(entities_dict: Dict, batch_size: int = 128) -> List[Dict]:
    """Tạo embeddings cho entities"""
    texts, metadata = [], []
    
    for chunk_id, entities in entities_dict.items():
        for entity in entities:
            texts.append(f"{entity['entity_name']} ({entity['entity_type']}): {entity.get('description', '')}")
            metadata.append({
                'id': f"entity_{entity['entity_name']}",
                'entity_name': entity['entity_name'],
                'entity_type': entity['entity_type'],
                'chunk_id': chunk_id
            })
    
    if not texts:
        return []
    
    model = get_model()
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    
    return [
        {**meta, 'text': text, 'embedding': emb.tolist()}
        for text, emb, meta in zip(texts, embeddings, metadata)
    ]