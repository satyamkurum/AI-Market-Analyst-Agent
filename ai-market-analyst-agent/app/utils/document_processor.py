# app/utils/document_processor.py

import logging
from typing import List, Optional
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles loading and chunking of documents for the RAG pipeline."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        logger.info(f"Initialized DocumentProcessor with chunk_size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}")

    def load_document(self, file_path: str) -> str:
        """
        Load text content from a file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            str: Content of the file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Document file not found: {file_path}")
            
            logger.info(f"Loading document from: {file_path}")
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            logger.info(f"Successfully loaded document, size: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    def chunk_document(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Input text to split
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            if not text or len(text.strip()) == 0:
                logger.warning("Attempted to chunk empty text")
                return []
            
            logger.info(f"Chunking text of length {len(text)} characters")
            chunks = self.text_splitter.split_text(text)
            
            logger.info(f"Created {len(chunks)} chunks from document")
            for i, chunk in enumerate(chunks):
                logger.debug(f"Chunk {i+1}: {len(chunk)} chars - {chunk[:100]}...")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error during document chunking: {str(e)}")
            raise

# Create a global instance for easy access
document_processor = DocumentProcessor()