# app/core/models.py

from pydantic import BaseModel
from typing import Any, Dict, Optional

class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    query: str
    session_id: str

class StandardResponse(BaseModel):
    """Standard response model for all API endpoints."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"response": "This is a sample response"},
                "message": "Operation completed successfully",
                "session_id": "session_12345"
            }
        }

# Health check models
class HealthCheck(BaseModel):
    """Health check response model."""
    status: str
    services: Dict[str, str]

class ServiceInfo(BaseModel):
    """Service information response model."""
    app_name: str
    environment: str
    vector_store: str
    embedding_model: str
    llm_model: str