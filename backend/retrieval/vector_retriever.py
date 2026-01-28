# backend/retrieval/vector_retriever.py
"""
Bộ Trích xuất Vector
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


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

# Bộ Retriever Chính 
class VectorRetriever:
    """Wrapper tìm kiếm vector đơn giản"""
    
    def __init__(self, vector_db):
        """
        Tham số:
            vector_db: Instance VectorDatabase từ backend.db.vector_db
        """
        self.vector_db = vector_db
        self.embed_model = None
    
    def _get_embed_model(self):
        """Tải mô hình embedding lazy"""
        if self.embed_model is None:
            from backend.core.embedding import get_model
            self.embed_model = get_model()
        return self.embed_model
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        doc_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[ScoredChunk]:
        """
        Tìm kiếm vector - Điểm vào chính
        
        Tham số:
            query: Văn bản query của người dùng
            top_k: Số kết quả trả về
            doc_ids: Lọc theo tài liệu cụ thể (tùy chọn)
            min_score: Điểm tương đồng tối thiểu (0-1)
        
        Trả về:
            Danh sách ScoredChunk, sắp xếp theo điểm giảm dần
        """
        try:
            model = self._get_embed_model()
            query_embedding = model.encode([query], normalize_embeddings=True)[0]
            
            raw_results = self.vector_db.search(
                query_embedding=query_embedding.tolist(),
                top_k=top_k * 2, 
                doc_ids=doc_ids,
                skip_deleted=True
            )
            
            
            scored_chunks = []
            for r in raw_results:
                
                similarity = r.get('similarity', 0.0)
                
                
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
            
    
            return scored_chunks[:top_k]
        
        except Exception as e:
            logger.error(f" Tìm kiếm vector thất bại: {e}")
            return []
    
    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        **kwargs
    ) -> List[ScoredChunk]:
        """
        Tìm kiếm bằng embedding đã tính sẵn
        
        Hữu ích khi bạn đã có embedding và muốn tránh encode lại.
        """
        try:
            raw_results = self.vector_db.search(
                query_embedding=query_embedding,
                top_k=top_k,
                **kwargs
            )
            
            return self._convert_to_scored_chunks(raw_results)[:top_k]
        
        except Exception as e:
            logger.error(f" Tìm kiếm embedding thất bại: {e}")
            return []
    
    def _convert_to_scored_chunks(self, raw_results: List[Dict]) -> List[ScoredChunk]:
        """Chuyển kết quả thô từ vector DB sang ScoredChunk"""
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
        """Lấy thống kê vector DB"""
        return self.vector_db.get_statistics()

# Hàm Tiện Ích 
def search_vectors(
    query: str,
    vector_db,
    top_k: int = 10,
    **kwargs
) -> List[ScoredChunk]:
    """
    Hàm tìm kiếm nhanh - Mẫu Singleton
    
    Cách dùng:
        from backend.db.vector_db import VectorDatabase
        from backend.retrieval.vector_retriever import search_vectors
        
        vector_db = VectorDatabase(user_id='admin_00000000')
        results = search_vectors("What is AI?", vector_db, top_k=5)
        
        for r in results:
            print(f"{r.score:.3f} - {r.content[:100]}")
    """
    retriever = VectorRetriever(vector_db)
    return retriever.search(query, top_k=top_k, **kwargs)
