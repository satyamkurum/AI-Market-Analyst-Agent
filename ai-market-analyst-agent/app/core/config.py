# app/core/config.py

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Priority:
    1. Environment variables (e.g. set in container or OS).
    2. .env file (loaded via python-dotenv).
    """

    # Pinecone Settings (vector DB - serverless index only)
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "ai-market-analyst"
    PINECONE_ENV: str = "us-east-1"

    # OpenAI Settings (for LLM agent)
    OPENAI_API_KEY: str

    # Gemini Settings (testing/comparison only, optional)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_EMBED_MODEL: str = "models/embedding-001"

    # Embedding Model Settings (main: local BGE model)
    LOCAL_EMBEDDING_MODEL: str = "thenlper/gte-large"
    EMBEDDING_DIMENSION: int = 1024  # must match Pinecone index dimension
  
    # ✅ ADD THESE NEW SETTINGS:
    # FastAPI Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # Application Settings
    PROJECT_NAME: str = "AI Market Analyst Agent"
    LOG_LEVEL: str = "INFO"
    CHUNK_SIZE: int = 512   # Token size for text splitting
    CHUNK_OVERLAP: int = 50 # Overlap between chunks

    def check_required_vars(self) -> None:
        """Ensure required environment variables are set."""
        if not self.PINECONE_API_KEY:
            raise ValueError("❌ PINECONE_API_KEY is not set in environment variables")
        if not self.OPENAI_API_KEY:
            raise ValueError("❌ OPENAI_API_KEY is not set in environment variables")

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Validate required vars at startup
settings.check_required_vars()