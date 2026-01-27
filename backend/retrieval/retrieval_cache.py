# backend/retrieval/retrieval_cache.py
"""
Bộ nhớ cache truy xuất - Tăng tốc các truy vấn lặp lại
"""

import hashlib
import time
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class RetrievalCache:
    """
    Cache  cho kết quả truy xuất
    
    Tính năng:
    - Cache keys dựa trên hash (query + mode + top_k)
    - TTL (Time To Live) cho các mục cache
    - Kích thước tối đa với loại bỏ LRU
    - Thao tác thread-safe
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        """
        Khởi tạo retrieval cache
        
        Tham số:
            max_size: Số mục cache tối đa (mặc định: 100)
            ttl: Thời gian sống tính bằng giây (mặc định: 300 = 5 phút)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {} 
        self.access_order: Dict[str, float] = {}  
        
        logger.info(f" Khởi tạo RetrievalCache (max_size={max_size}, ttl={ttl}s)")
    
    def _generate_key(self, query: str, mode: str, top_k: int) -> str:
        """Tạo cache key từ tham số query"""
        key_string = f"{query}|{mode}|{top_k}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Kiểm tra cache entry đã hết hạn chưa"""
        return (time.time() - timestamp) > self.ttl
    
    def _evict_lru(self):
        """Loại bỏ entry ít dùng nhất (LRU)"""
        if not self.access_order:
            return
        
        # Tìm entry LRU
        lru_key = min(self.access_order, key=self.access_order.get)
        
        # Xóa khỏi cache
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_order[lru_key]
        
        logger.debug(f" Evicted LRU entry: {lru_key[:8]}...")
    
    def get(self, query: str, mode: str, top_k: int) -> Optional[Any]:
        """
        Lấy kết quả đã cache
        
        Args:
            query: Truy vấn tìm kiếm
            mode: Chế độ truy xuất (vector/graph/hybrid/auto)
            top_k: Số kết quả
        
        Returns:
            Kết quả đã cache hoặc None nếu không tìm thấy/hết hạn
        """
        cache_key = self._generate_key(query, mode, top_k)
        
        # Kiểm tra tồn tại
        if cache_key not in self.cache:
            logger.debug(f" Cache miss: {cache_key[:8]}...")
            return None
        
        result, timestamp = self.cache[cache_key]
        
        # Kiểm tra hết hạn
        if self._is_expired(timestamp):
            logger.debug(f" Cache expired: {cache_key[:8]}...")
            del self.cache[cache_key]
            del self.access_order[cache_key]
            return None
        
        # Cập nhật thời gian truy cập
        self.access_order[cache_key] = time.time()
        
        logger.debug(f"Cache hit: {cache_key[:8]}...")
        return result
    
    def set(self, query: str, mode: str, top_k: int, result: Any):
        """
        Lưu kết quả vào cache
        
        Args:
            query: Truy vấn tìm kiếm
            mode: Chế độ truy xuất
            top_k: Số kết quả
            result: Kết quả truy xuất cần cache
        """
        cache_key = self._generate_key(query, mode, top_k)
        
        # Loại bỏ nếu đạt giới hạn
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            self._evict_lru()
        
        # Lưu vào cache
        self.cache[cache_key] = (result, time.time())
        self.access_order[cache_key] = time.time()
        
        logger.debug(f"Cached result: {cache_key[:8]}... (size: {len(self.cache)}/{self.max_size})")
    
    def clear(self):
        """Xóa tất cả cache entries"""
        self.cache.clear()
        self.access_order.clear()
        logger.info(" Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê cache"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl': self.ttl,
            'entries': list(self.cache.keys())[:5]  
        }
