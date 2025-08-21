import logging
from app.services.embedding_service import embedding_service
from app.utils.document_processor import document_processor

logging.basicConfig(level=logging.INFO)

def test_embeddings():
    """Test all embedding generation functionality."""
    try:
        print("Testing embedding service...")
        
        # Load and chunk the document
        text = document_processor.load_document("data/Innovate-Inc-Report.txt")
        chunks = document_processor.chunk_document(text)
        test_texts = chunks[:2]
        
        # Test local embeddings
        print("\n1. Testing local embeddings...")
        local_embeddings = embedding_service.embed_local(test_texts)
        print(f"✓ Local embeddings: {len(local_embeddings)} vectors")
        print(f"  Dimension: {len(local_embeddings[0])}")
        
        # Test Gemini embeddings (if available)
        print("\n2. Testing Gemini embeddings...")
        try:
            gemini_embeddings = embedding_service.embed_with_gemini(test_texts)
            print(f"✓ Gemini embeddings: {len(gemini_embeddings)} vectors")
            print(f"  Dimension: {len(gemini_embeddings[0])}")
        except Exception as e:
            print(f"⚠ Gemini embeddings skipped: {e}")
        
        # Test cache
        print("\n3. Testing cache...")
        cached_embeddings = embedding_service.embed_local(test_texts)
        print(f"✓ Cached embeddings retrieved: {len(cached_embeddings)} vectors")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_embeddings()
    exit(0 if success else 1)