# test_chunking.py

import logging
from app.utils.document_processor import document_processor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_document_processing():
    """Test the document loading and chunking functionality."""
    try:
        print("Testing document processing...")
        
        # Load the document
        text = document_processor.load_document("data/Innovate-Inc-Report.txt")
        print(f"✓ Successfully loaded document ({len(text)} characters)")
        
        # Chunk the document
        chunks = document_processor.chunk_document(text)
        print(f"✓ Successfully created {len(chunks)} chunks")
        
        # Show sample chunks
        print("\nSample chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"Chunk {i+1}: {chunk[:100]}...")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_document_processing()
    exit(0 if success else 1)