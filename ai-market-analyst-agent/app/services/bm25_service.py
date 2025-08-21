# app/services/bm25_service.py

import logging
import pickle
import os
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from app.core.config import settings

logger = logging.getLogger(__name__)

class BM25Service:
    """Service to handle BM25 keyword indexing and search."""
    
    def __init__(self):
        self.bm25_indexes = {}  # session_id -> BM25 index
        self.session_chunks = {}  # session_id -> list of chunks
        
    def build_index(self, chunks: List[str], session_id: str) -> None:
        """
        Build BM25 index for a session.
        
        Args:
            chunks: List of text chunks to index
            session_id: Session identifier
        """
        try:
            logger.info(f"Building BM25 index for session '{session_id}' with {len(chunks)} chunks")
            
            # Tokenize chunks (simple whitespace tokenizer)
            tokenized_chunks = [chunk.lower().split() for chunk in chunks]
            
            # Create BM25 index
            bm25 = BM25Okapi(tokenized_chunks)
            
            # Store index and chunks
            self.bm25_indexes[session_id] = bm25
            self.session_chunks[session_id] = chunks
            
            logger.info(f"✓ BM25 index built for session '{session_id}'")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index for session '{session_id}': {e}")
            raise
    
    def search(self, query: str, session_id: str, top_k: int = 5) -> List[Dict]:
        """
        Search using BM25 keyword matching.
        
        Args:
            query: Search query
            session_id: Session identifier
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and text
        """
        try:
            if session_id not in self.bm25_indexes:
                raise ValueError(f"No BM25 index found for session '{session_id}'")
            
            logger.info(f"BM25 search for query: '{query}' in session '{session_id}'")
            
            bm25 = self.bm25_indexes[session_id]
            chunks = self.session_chunks[session_id]
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get scores
            scores = bm25.get_scores(tokenized_query)
            
            # Get top-k results
            top_indices = scores.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    results.append({
                        "score": float(scores[idx]),
                        "text": chunks[idx],
                        "index": int(idx),
                        "retriever": "bm25"
                    })
            
            logger.info(f"BM25 found {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed for session '{session_id}': {e}")
            raise
    
    def save_index(self, session_id: str, cache_dir: str = "./cache/bm25") -> None:
        """Save BM25 index to disk."""
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{session_id}.pkl")
            
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25_indexes.get(session_id),
                    'chunks': self.session_chunks.get(session_id)
                }, f)
            
            logger.info(f"Saved BM25 index for session '{session_id}' to {cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to save BM25 index for session '{session_id}': {e}")
    
    def load_index(self, session_id: str, cache_dir: str = "./cache/bm25") -> bool:
        """Load BM25 index from disk."""
        try:
            cache_path = os.path.join(cache_dir, f"{session_id}.pkl")
            
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                self.bm25_indexes[session_id] = data['bm25']
                self.session_chunks[session_id] = data['chunks']
                
                logger.info(f"Loaded BM25 index for session '{session_id}' from {cache_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to load BM25 index for session '{session_id}': {e}")
            return False
    
    def delete_session(self, session_id: str) -> None:
        """Remove session data from memory."""
        self.bm25_indexes.pop(session_id, None)
        self.session_chunks.pop(session_id, None)
        logger.info(f"Removed BM25 data for session '{session_id}' from memory")

# Create a global instance
bm25_service = BM25Service()