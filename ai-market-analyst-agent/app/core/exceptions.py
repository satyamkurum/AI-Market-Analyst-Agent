# app/core/exceptions.py

class VectorStoreConnectionError(Exception):
    """Raised when connection to vector database fails."""
    pass

class VectorStoreOperationError(Exception):
    """Raised when vector store operations fail."""
    pass

class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails."""
    pass

class RetrievalError(Exception):
    """Raised when document retrieval fails."""
    pass

class LLMGenerationError(Exception):
    """Raised when LLM response generation fails."""
    pass

# ADD THIS NEW EXCEPTION
class ToolExecutionError(Exception):
    """Raised when tool execution fails."""
    pass

