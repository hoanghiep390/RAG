# backend/retrieval/vector_retriever.py
"""
🔍 Vector Retriever 
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

#  Data Classes 
@dataclass
class ScoredChunk:
    """Kết quả từ vector search"""
    chunk_id: str
    content: str
    score: float  
    doc_id: str
    filename: str
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"ScoredChunk(score={self.score:.3f}, doc={self.filename})"

#  Main Retriever 
class VectorRetriever:
    """Simple vector search wrapper"""
    
    def __init__(self, vector_db):
        """
        Args:
            vector_db: VectorDatabase instance from backend.db.vector_db
        """
        self.vector_db = vector_db
        self.embed_model = None
    
    def _get_embed_model(self):
        """Lazy load embedding model"""
        if self.embed_model is None:
            from backend.core.embedding import get_model
            self.embed_model = get_model()
        return self.embed_model
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[ScoredChunk]:
        """
        Vector search - Main entry point
        
        Args:
            query: User query text
            top_k: Number of results to return
            doc_ids: Filter by specific documents (optional)
            min_score: Minimum similarity score (0-1)
        
        Returns:
            List of ScoredChunk, sorted by score descending
        """
        try:
            # 1. Embed query
            model = self._get_embed_model()
            query_embedding = model.encode([query], normalize_embeddings=True)[0]
            
            # 2. Search vector DB
            raw_results = self.vector_db.search(
                query_embedding=query_embedding.tolist(),
                top_k=top_k * 2, 
                doc_ids=doc_ids,
                skip_deleted=True
            )
            
            # 3. Convert to ScoredChunk
            scored_chunks = []
            for r in raw_results:
                # Convert distance to similarity (0-1 range)
                similarity = r.get('similarity', 0.0)
                
                # Filter by min_score
                if similarity < min_score:
                    continue
                
                chunk = ScoredChunk(
                    chunk_id=r.get('chunk_id', ''),
                    content=r.get('content', ''),
                    score=similarity,
                    doc_id=r.get('doc_id', ''),
                    filename=r.get('filename', 'unknown'),
                    metadata={
                        'file_path': r.get('file_path', ''),
                        'file_type': r.get('file_type', ''),
                        'tokens': r.get('tokens', 0),
                        'order': r.get('order', 0),
                        'entity_type': r.get('entity_type', 'CHUNK')
                    }
                )
                scored_chunks.append(chunk)
            
            # 4. Return top_k after filtering
            return scored_chunks[:top_k]
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        **kwargs
    ) -> List[ScoredChunk]:
        """
        Search by pre-computed embedding
        
        Useful when you already have the embedding and want to avoid re-encoding.
        """
        try:
            raw_results = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k,
                **kwargs
            )
            
            return self._convert_to_scored_chunks(raw_results)[:top_k]
        
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return []
    
    def _convert_to_scored_chunks(self, raw_results: List[Dict]) -> List[ScoredChunk]:
        """Convert raw vector DB results to ScoredChunk"""
        chunks = []
        for r in raw_results:
            chunk = ScoredChunk(
                chunk_id=r.get('chunk_id', ''),
                content=r.get('content', ''),
                score=r.get('similarity', 0.0),
                doc_id=r.get('doc_id', ''),
                filename=r.get('filename', 'unknown'),
                metadata={
                    'file_path': r.get('file_path', ''),
                    'file_type': r.get('file_type', ''),
                    'tokens': r.get('tokens', 0),
                    'order': r.get('order', 0),
                    'entity_type': r.get('entity_type', 'CHUNK')
                }
            )
            chunks.append(chunk)
        return chunks
    
    def get_statistics(self) -> Dict:
        """Get vector DB statistics"""
        return self.vector_db.get_statistics()

# Convenience Function 
def search_vectors(
    query: str,
    vector_db,
    top_k: int = 5,
    **kwargs
) -> List[ScoredChunk]:
    """
    Quick search function - Singleton pattern
    
    Usage:
        from backend.db.vector_db import VectorDatabase
        from backend.retrieval.vector_retriever import search_vectors
        
        vector_db = VectorDatabase(user_id='admin_00000000')
        results = search_vectors("What is AI?", vector_db, top_k=5)
        
        for r in results:
            print(f"{r.score:.3f} - {r.content[:100]}")
    """
    retriever = VectorRetriever(vector_db)
    return retriever.search(query, top_k=top_k, **kwargs)
