# test_pinecone_session_simple.py

import logging
from app.services.vector_store import vector_store
from app.services.embedding_service import embedding_service
from app.utils.document_processor import document_processor

logging.basicConfig(level=logging.INFO)

def test_session_management():
    """Test Pinecone operations with proper session management."""
    try:
        # Generate a unique session ID
        session_id = "test_session_001"
        print(f"Testing session: {session_id}")
        
        # 1. Initialize session (just gets namespace - no deletion needed)
        namespace = vector_store.initialize_session(session_id)
        print(f"Using namespace: {namespace}")
        
        # 2. Load and chunk the document
        text = document_processor.load_document("data/Innovate-Inc-Report.txt")
        chunks = document_processor.chunk_document(text)
        print(f"✓ Loaded {len(chunks)} chunks from document")
        
        # 3. Generate embeddings
        embeddings = embedding_service.embed(chunks)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        # 4. Upsert to Pinecone with session ID
        vector_store.upsert_embeddings(chunks, embeddings, session_id)
        
        # 5. Test search within the same session
        test_query = "What is the market share of FutureFlow?"
        query_embedding = embedding_service.embed([test_query])[0]
        
        results = vector_store.search_similar(query_embedding, session_id, top_k=3)
        
        print(f"\n🔍 Search results for: '{test_query}'")
        print("=" * 60)
        for i, result in enumerate(results):
            print(f"{i+1}. Score: {result['score']:.3f}")
            print(f"   Text: {result['text'][:100]}...")
            print()
        
        # 6. Show session stats
        stats = vector_store.get_session_stats(session_id)
        print(f"Session stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_session_management()
    exit(0 if success else 1)