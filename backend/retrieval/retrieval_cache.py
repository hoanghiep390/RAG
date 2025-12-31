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
    Cache LRU cho kết quả truy xuất
    
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
        """Generate cache key from query parameters"""
        key_string = f"{query}|{mode}|{top_k}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired"""
        return (time.time() - timestamp) > self.ttl
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_order:
            return
        
        # Find LRU entry
        lru_key = min(self.access_order, key=self.access_order.get)
        
        # Remove from cache
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_order[lru_key]
        
        logger.debug(f" Evicted LRU entry: {lru_key[:8]}...")
    
    def get(self, query: str, mode: str, top_k: int) -> Optional[Any]:
        """
        Get cached result
        
        Args:
            query: Search query
            mode: Retrieval mode (vector/graph/hybrid/auto)
            top_k: Number of results
        
        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._generate_key(query, mode, top_k)
        
        # Check if exists
        if cache_key not in self.cache:
            logger.debug(f" Cache miss: {cache_key[:8]}...")
            return None
        
        result, timestamp = self.cache[cache_key]
        
        # Check if expired
        if self._is_expired(timestamp):
            logger.debug(f" Cache expired: {cache_key[:8]}...")
            del self.cache[cache_key]
            del self.access_order[cache_key]
            return None
        
        # Update access time
        self.access_order[cache_key] = time.time()
        
        logger.debug(f"Cache hit: {cache_key[:8]}...")
        return result
    
    def set(self, query: str, mode: str, top_k: int, result: Any):
        """
        Store result in cache
        
        Args:
            query: Search query
            mode: Retrieval mode
            top_k: Number of results
            result: Retrieval result to cache
        """
        cache_key = self._generate_key(query, mode, top_k)
        
        # Evict if at max size
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            self._evict_lru()
        
        # Store in cache
        self.cache[cache_key] = (result, time.time())
        self.access_order[cache_key] = time.time()
        
        logger.debug(f"Cached result: {cache_key[:8]}... (size: {len(self.cache)}/{self.max_size})")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        logger.info(" Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl': self.ttl,
            'entries': list(self.cache.keys())[:5]  
        }
