# app/services/vector_store.py

import logging
import uuid
import hashlib
from typing import List, Dict, Optional
from pinecone import Pinecone
from app.core.config import settings
from app.core.exceptions import VectorStoreConnectionError, VectorStoreOperationError

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Service to handle all Pinecone vector store operations with session support."""
    
    def __init__(self):
        self.pc = None
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and connect to index."""
        try:
            logger.info("Initializing Pinecone client...")
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Verify index exists
            index_names = [index.name for index in self.pc.list_indexes()]
            if settings.PINECONE_INDEX_NAME not in index_names:
                error_msg = f"Index '{settings.PINECONE_INDEX_NAME}' not found. Please create it first."
                logger.error(error_msg)
                raise VectorStoreConnectionError(error_msg)
            
            # Connect to the index
            self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
            logger.info(f"Connected to Pinecone index '{settings.PINECONE_INDEX_NAME}'")
            
        except Exception as e:
            error_msg = f"Failed to initialize Pinecone: {e}"
            logger.exception(error_msg)
            raise VectorStoreConnectionError(error_msg) from e

    def initialize_session(self, session_id: str) -> str:
        """
        Initialize a new session namespace.
        """
        namespace = f"session_{session_id}"
        logger.info(f"Initializing session namespace: {namespace}")
        return namespace  # Pinecone creates namespaces lazily

    def upsert_embeddings(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        session_id: str,
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """Upsert text embeddings to Pinecone index for a specific session."""
        try:
            if len(texts) != len(embeddings):
                raise ValueError("Texts and embeddings must have the same length")
            
            namespace = f"session_{session_id}"
            
            if metadata is None:
                metadata = [{} for _ in texts]
            
            # Prepare vectors for upsert
            vectors = []
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
                # Use deterministic MD5 hash for stable, collision-free IDs
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                vector_id = f"{session_id}_{i}_{text_hash}"
                
                vector_metadata = {
                    "text": text,
                    "chunk_index": i,
                    "session_id": session_id,
                    **meta
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": vector_metadata
                })
            
            logger.info(f"Upserting {len(vectors)} vectors to session '{session_id}' (namespace={namespace})")
            
            # Perform upsert and capture response
            upsert_response = self.index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            
            logger.info(f"✓ Upsert complete for session '{session_id}' | Response: {upsert_response}")
            
        except Exception as e:
            error_msg = f"Failed to upsert embeddings for session {session_id}: {e}"
            logger.exception(error_msg)
            raise VectorStoreOperationError(error_msg) from e

    def search_similar(
        self, 
        query_embedding: List[float], 
        session_id: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Search for similar vectors in the specific session namespace."""
        try:
            namespace = f"session_{session_id}"
            logger.info(f"Searching for top {top_k} similar vectors in session '{session_id}'")
            
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            
            formatted_results = []
            for match in query_response.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", "") if match.metadata else "",
                    "session_id": match.metadata.get("session_id", "") if match.metadata else "",
                    "metadata": match.metadata or {}
                })
            
            logger.info(f"Found {len(formatted_results)} similar vectors in session '{session_id}'")
            return formatted_results
            
        except Exception as e:
            error_msg = f"Failed to search vectors in session {session_id}: {e}"
            logger.exception(error_msg)
            raise VectorStoreOperationError(error_msg) from e

    def delete_session(self, session_id: str) -> None:
        """Delete all vectors for a specific session."""
        try:
            namespace = f"session_{session_id}"
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted all vectors for session '{session_id}'")
        except Exception as e:
            if "Namespace not found" in str(e) or "code\":5" in str(e):
                logger.info(f"No vectors to delete for session '{session_id}' (namespace not found)")
            else:
                error_msg = f"Failed to delete session {session_id}: {e}"
                logger.exception(error_msg)
                raise VectorStoreOperationError(error_msg) from e

    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session namespace with proper error handling."""
        try:
            namespace = f"session_{session_id}"
            
            # Get global stats
            global_stats = self.index.describe_index_stats()
            
            # Get session-specific count - handle missing namespaces gracefully
            session_stats = global_stats.namespaces.get(namespace, {})
            session_vector_count = session_stats.get('vector_count', 0)
            
            # Additional verification: actually count vectors in the namespace
            actual_count = 0
            try:
                # Do a quick query to verify vectors exist
                test_vector = [0.0] * settings.EMBEDDING_DIMENSION
                results = self.index.query(
                    vector=test_vector,
                    top_k=1,
                    include_metadata=False,
                    namespace=namespace
                )
                actual_count = len(results.matches)
            except:
                actual_count = 0
            
            return {
                "total_vectors": global_stats.total_vector_count,
                "session_vectors": session_vector_count,
                "actual_vectors_found": actual_count,
                "session_id": session_id,
                "namespace": namespace,
                "index_dimension": global_stats.dimension,
                "status": "healthy" if actual_count > 0 else "no_vectors"
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for session {session_id}: {e}")
            return {
                "total_vectors": 0,
                "session_vectors": 0,
                "actual_vectors_found": 0,
                "session_id": session_id,
                "namespace": f"session_{session_id}",
                "index_dimension": 1024,
                "status": f"error: {str(e)}"
            }

    def verify_session_vectors(self, session_id: str, expected_count: int) -> bool:
        """Verify that the correct number of vectors exist in the session namespace."""
        try:
            namespace = f"session_{session_id}"
            test_vector = [0.0] * settings.EMBEDDING_DIMENSION
            
            results = self.index.query(
                vector=test_vector,
                top_k=1,
                include_metadata=False,
                namespace=namespace
            )
            
            has_vectors = len(results.matches) > 0
            stats = self.get_session_stats(session_id)
            actual_count = stats["session_vectors"]
            
            logger.info(
                f"Session '{session_id}' verification: expected={expected_count}, "
                f"actual={actual_count}, has_vectors={has_vectors}"
            )
            
            return actual_count == expected_count and has_vectors
            
        except Exception as e:
            logger.error(f"Session verification failed for '{session_id}': {e}")
            return False


# Create a global instance
vector_store = VectorStoreService()
