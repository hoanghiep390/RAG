# backend/utils/cache_utils.py

import logging
logger = logging.getLogger(__name__)

logger.warning(
    "⚠️ cache_utils.py is DEPRECATED and will be removed. "
    "Core modules no longer use caching. "
    "Implement caching at application level if needed."
)


class DiskCache:
    """ DEPRECATED - Do not use"""
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "DiskCache is deprecated. Use Redis or memory cache at application level."
        )


class JSONCache:
    """ DEPRECATED - Do not use"""
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "JSONCache is deprecated. Use Redis or memory cache at application level."
        )


def disk_cached(*args, **kwargs):
    """ DEPRECATED - Do not use"""
    raise DeprecationWarning(
        "disk_cached decorator is deprecated. Implement caching at application level."
    )


def async_disk_cached(*args, **kwargs):
    """ DEPRECATED - Do not use"""
    raise DeprecationWarning(
        "async_disk_cached decorator is deprecated. Implement caching at application level."
    )

extraction_cache = None
embedding_cache = None
chunk_cache = None

logger.error(
    "❌ Attempting to use deprecated cache objects. "
    "Update your imports to remove cache_utils dependencies."
) 