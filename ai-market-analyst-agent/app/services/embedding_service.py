# app/services/embedding_service.py

import logging
from typing import List, Literal, Optional
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.utils.cache import get_embedding_cache
from app.core.exceptions import EmbeddingGenerationError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service to handle text embedding using multiple providers."""

    def __init__(self):
        self.local_model = None
        self.gemini_client = None
        self.cache = get_embedding_cache()

        # Initialize local model
        try:
            logger.info(f"Loading local embedding model: {settings.LOCAL_EMBEDDING_MODEL}")
            self.local_model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
            logger.info("✓ Local embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load local model {settings.LOCAL_EMBEDDING_MODEL}: {e}")
            self.local_model = None

        # Initialize Gemini client if API key is provided
        if settings.GEMINI_API_KEY:
            try:
                logger.info("Initializing Gemini client for API embeddings")
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_client = genai
                logger.info("✓ Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
        else:
            logger.warning("GEMINI_API_KEY not set, Gemini embeddings disabled")

    def _get_cached_embedding(self, text: str, model_type: str) -> Optional[List[float]]:
        """Get embedding from cache if exists."""
        cache_key = f"{model_type}:{text}"
        return self.cache.get(cache_key)

    def _set_cached_embedding(self, text: str, model_type: str, embedding: List[float]):
        """Store embedding in cache."""
        cache_key = f"{model_type}:{text}"
        self.cache.set(cache_key, embedding)

    def embed_local(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        if not self.local_model:
            raise EmbeddingGenerationError("Local embedding model is not initialized")

        embeddings = []
        for text in texts:
            text = text.strip()
            if not text:
                logger.warning("Skipping empty text chunk during embedding")
                embeddings.append([0.0] * settings.EMBEDDING_DIMENSION)
                continue

            # Check cache
            cached = self._get_cached_embedding(text, "local")
            if cached:
                embeddings.append(cached)
                continue

            try:
                emb = self.local_model.encode(text).tolist()
                if emb is None:
                    raise EmbeddingGenerationError("Local model returned None embedding")
                if len(emb) != settings.EMBEDDING_DIMENSION:
                    logger.warning(f"Embedding dimension mismatch: got {len(emb)}, expected {settings.EMBEDDING_DIMENSION}")
                    # Pad or trim to expected dimension
                    emb = (emb + [0.0] * settings.EMBEDDING_DIMENSION)[:settings.EMBEDDING_DIMENSION]

                embeddings.append(emb)
                self._set_cached_embedding(text, "local", emb)

            except Exception as e:
                logger.error(f"Local embedding failed for text '{text[:50]}...': {e}")
                # Use zero-vector fallback
                embeddings.append([0.0] * settings.EMBEDDING_DIMENSION)

        logger.info(f"✓ Successfully generated {len(embeddings)} local embeddings")
        return embeddings

    def embed_with_gemini(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google Gemini API."""
        if not self.gemini_client:
            raise EmbeddingGenerationError("Gemini client not available")

        embeddings = []
        try:
            uncached_texts = []
            cached_embeddings = []

            for text in texts:
                text = text.strip()
                if not text:
                    cached_embeddings.append([0.0] * 768)
                    continue
                cached = self._get_cached_embedding(text, "gemini")
                if cached:
                    cached_embeddings.append(cached)
                else:
                    uncached_texts.append(text)

            new_embeddings = []
            for text in uncached_texts:
                result = self.gemini_client.embed_content(
                    model=settings.GEMINI_EMBED_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                emb = result.get("embedding", [0.0] * 768)
                if len(emb) != 768:
                    logger.warning(f"Gemini embedding dimension mismatch, adjusting to 768")
                    emb = (emb + [0.0] * 768)[:768]

                new_embeddings.append(emb)
                self._set_cached_embedding(text, "gemini", emb)

            embeddings = cached_embeddings + new_embeddings
            logger.info(f"✓ Successfully generated {len(embeddings)} Gemini embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Gemini embedding generation failed: {e}")
            # fallback: return zero embeddings
            return [[0.0] * 768 for _ in texts]

    def embed(self, texts: List[str], model_type: Literal["local", "gemini"] = "local") -> List[List[float]]:
        """Main method to generate embeddings."""
        if model_type == "local":
            return self.embed_local(texts)
        elif model_type == "gemini":
            return self.embed_with_gemini(texts)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Global instance
embedding_service = EmbeddingService()
