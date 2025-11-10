# backend/utils/cache_utils.py
"""
Caching utilities for performance optimization
"""

import hashlib
import pickle
import json
from pathlib import Path
from functools import wraps
from typing import Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class DiskCache:
    """
    Persistent disk cache for expensive operations
    Uses pickle for complex objects
    """
    
    def __init__(self, cache_dir: str = "backend/data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key"""
        cache_file = self.cache_dir / f"{self._hash(key)}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    logger.debug(f"✅ Cache hit: {key[:50]}...")
                    return data
            except Exception as e:
                logger.warning(f"❌ Cache read error: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value"""
        cache_file = self.cache_dir / f"{self._hash(key)}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            logger.debug(f"💾 Cached: {key[:50]}...")
        except Exception as e:
            logger.error(f"❌ Cache write error: {e}")
    
    def clear(self) -> int:
        """Clear all cache files"""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except:
                pass
        logger.info(f"🗑️ Cleared {count} cache files")
        return count
    
    def _hash(self, key: str) -> str:
        """Generate hash for cache key"""
        return hashlib.md5(key.encode()).hexdigest()


class JSONCache:
    """
    JSON-based cache for simple data structures
    Faster than pickle but limited to JSON-serializable objects
    """
    
    def __init__(self, cache_dir: str = "backend/data/cache/json"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Optional[Any]:
        cache_file = self.cache_dir / f"{self._hash(key)}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        cache_file = self.cache_dir / f"{self._hash(key)}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON cache error: {e}")
    
    def _hash(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()


# Decorator for caching function results
def disk_cached(cache_dir: str = "backend/data/cache"):
    """
    Decorator to cache function results on disk
    
    Usage:
        @disk_cached("backend/data/cache/extractions")
        def expensive_function(arg1, arg2):
            # ... expensive computation
            return result
    """
    cache = DiskCache(cache_dir)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


def async_disk_cached(cache_dir: str = "backend/data/cache"):
    """
    Async version of disk_cached decorator
    
    Usage:
        @async_disk_cached("backend/data/cache/async")
        async def async_function(arg):
            # ... async computation
            return result
    """
    cache = DiskCache(cache_dir)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            result = await func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


# Pre-configured cache instances
extraction_cache = DiskCache("backend/data/cache/extractions")
embedding_cache = DiskCache("backend/data/cache/embeddings")
chunk_cache = DiskCache("backend/data/cache/chunks")