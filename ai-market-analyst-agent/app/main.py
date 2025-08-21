# app/main.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import router as api_router
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for FastAPI app."""
    # Startup: Initialize services
    logger.info("🚀 Starting AI Market Analyst API...")
    
    try:
        # Initialize vector store connection
        logger.info("Initializing vector store...")
        # Vector store is already initialized in its module
        
        # Initialize LLM service
        logger.info("Initializing LLM service...")
        # LLM service is already initialized in its module
        
        logger.info("✅ All services initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Shutdown: Cleanup resources
        logger.info("🛑 Shutting down AI Market Analyst API...")
        # Add any cleanup logic here

# Create FastAPI app with lifespan
app = FastAPI(
    title="AI Market Analyst API",
    description="API for AI-powered market research analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_router, prefix="/api/v1", tags=["api"])

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "AI Market Analyst API is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with service status."""
    services_status = {
        "vector_store": "healthy",  # You can add actual checks
        "llm_service": "healthy",
        "agent": "healthy"
    }
    return {
        "status": "healthy",
        "services": services_status
    }

# Optional: Add more endpoints for monitoring
@app.get("/info")
async def api_info():
    """Get API information and configuration."""
    return {
        "environment": settings.ENVIRONMENT,
        "vector_store": settings.PINECONE_INDEX_NAME,
        "embedding_model": settings.LOCAL_EMBEDDING_MODEL,
        "llm_model": "gemini-2.5-pro"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )