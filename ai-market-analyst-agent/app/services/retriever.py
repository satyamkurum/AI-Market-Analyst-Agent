import logging
from typing import List, Dict
from app.services.vector_store import vector_store
from app.services.bm25_service import bm25_service
from app.services.embedding_service import embedding_service
import numpy as np 

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self):
        pass

    def retrieve(self, query: str, session_id: str, top_k: int = 5, strategy: str = "hybrid") -> List[Dict]:
        """Retrieve relevant chunks using hybrid search."""
        try:
            if strategy == "hybrid":
                # Get vector results
                vector_results = self._vector_search(query, session_id, top_k)
                # Get BM25 results  
                bm25_results = self._bm25_search(query, session_id, top_k)
                
                # Combine and deduplicate results
                all_results = self._merge_results(vector_results, bm25_results, top_k)
                return all_results
                
            elif strategy == "vector":
                return self._vector_search(query, session_id, top_k)
            elif strategy == "bm25":
                return self._bm25_search(query, session_id, top_k)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Hybrid retrieval failed for query '{query}': {e}")
            # Fall back to BM25 only if vector search fails
            try:
                return self._bm25_search(query, session_id, top_k)
            except Exception as fallback_error:
                logger.error(f"BM25 fallback also failed: {fallback_error}")
                return []

    def _vector_search(self, query: str, session_id: str, top_k: int) -> List[Dict]:
        """Perform vector similarity search."""
        try:
            # Generate query embedding
            query_embedding = embedding_service.embed([query])[0]  # ← CORRECT METHOD NAME
            
            # ✅ ADD DEBUG CODE RIGHT HERE:
            logger.info(f"🔍 VECTOR DEBUG: Query='{query}'")
            logger.info(f"🔍 VECTOR DEBUG: Embedding length={len(query_embedding)}")
            logger.info(f"🔍 VECTOR DEBUG: First 3 values={query_embedding[:3]}")
            
            # Search vector store
            results = vector_store.search_similar(
                query_embedding=query_embedding,
                session_id=session_id,
                top_k=top_k
            )
            
            logger.info(f"Vector search found {len(results)} results for '{query}'")
            return results
            
        except Exception as e:
            logger.warning(f"Vector search failed for '{query}': {e}")
            return []  # Return empty list instead of crashing

    def _bm25_search(self, query: str, session_id: str, top_k: int) -> List[Dict]:
        """Perform BM25 keyword search."""
        try:
            results = bm25_service.search(query, session_id, top_k=top_k)
            logger.debug(f"BM25 search found {len(results)} results for '{query}'")
            return results
        except Exception as e:
            logger.error(f"BM25 search failed for '{query}': {e}")
            return []

    def _merge_results(self, vector_results: List[Dict], bm25_results: List[Dict], top_k: int) -> List[Dict]:
        """Merge and deduplicate results from both retrieval methods."""
        # Simple deduplication by text content
        seen_texts = set()
        merged_results = []
        
        # Add vector results first (higher precision)
        for result in vector_results:
            text = result.get('text', '')
            if text and text not in seen_texts:
                seen_texts.add(text)
                merged_results.append(result)
        
        # Add BM25 results (higher recall)
        for result in bm25_results:
            text = result.get('text', '')
            if text and text not in seen_texts:
                seen_texts.add(text)
                merged_results.append(result)
        
        # Return top_k results
        return merged_results[:top_k]

    def ingest_document(self, chunks: List, session_id: str) -> None:
        """Ingest document chunks into both vector store and BM25 index."""
        try:
            # Handle both string chunks and dictionary chunks
            if chunks and isinstance(chunks[0], dict) and 'text' in chunks[0]:
                texts = [chunk['text'] for chunk in chunks]  # Dictionary format
            else:
                texts = chunks  # Assume it's already a list of strings
            
            # ✅ ADD DEBUG HERE - RIGHT AFTER GETTING TEXTS
            logger.info(f"📊 INGESTION DEBUG: First chunk text: '{texts[0][:50]}...'")
            logger.info(f"📊 INGESTION DEBUG: Number of chunks: {len(texts)}")
            
            # Generate embeddings
            embeddings = embedding_service.embed(texts)
            
            # ✅ ADD DEBUG HERE - RIGHT AFTER GENERATING EMBEDDINGS
            logger.info(f"📊 INGESTION DEBUG: First embedding length: {len(embeddings[0])}")
            logger.info(f"📊 INGESTION DEBUG: First embedding sample: {embeddings[0][:3]}")
            
            # Upsert to vector store
            vector_store.upsert_embeddings(
                texts=texts,
                embeddings=embeddings,
                session_id=session_id,
                metadata=[{"chunk_index": i} for i in range(len(chunks))]
            )
            
            # Build BM25 index
            bm25_service.build_index(texts, session_id)
            
            logger.info(f"✓ Successfully ingested document for session '{session_id}'")
            
        except Exception as e:
            logger.error(f"Document ingestion failed for session '{session_id}': {e}")
            raise


# Create global instance
hybrid_retriever = HybridRetriever()
