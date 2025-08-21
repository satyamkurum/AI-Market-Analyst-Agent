# app/utils/cache.py

import diskcache
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

def get_embedding_cache():
    """Get or create disk cache for embeddings."""
    try:
        cache = diskcache.Cache('./cache/embeddings')
        logger.info("Embedding cache initialized successfully")
        return cache
    except Exception as e:
        logger.error(f"Failed to initialize embedding cache: {e}")
        raise