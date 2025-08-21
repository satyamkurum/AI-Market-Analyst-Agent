# app/api/endpoints.py
from fastapi import APIRouter, HTTPException
import json
from app.core.models import QueryRequest, StandardResponse
from app.agents.tools import answer_question, summarize_document, extract_structured_data

router = APIRouter()

@router.post("/query", response_model=StandardResponse)
def query_endpoint(request: QueryRequest):
    """Main endpoint for agent queries."""
    try:
        # ✅ FIRST check if session has data
        from app.services.vector_store import vector_store
        session_stats = vector_store.get_session_stats(request.session_id)
        
        if session_stats["actual_vectors_found"] == 0:
            return StandardResponse(
                success=False,
                data={"response": "Error: No data found for this session. Please ingest documents first."},
                session_id=request.session_id
            )
        
        # ✅ THEN process query
        query_lower = request.query.lower()
        
        if any(keyword in query_lower for keyword in ["summary", "summarize", "overview"]):
            result = summarize_document(request.session_id)
        elif any(keyword in query_lower for keyword in ["extract", "json", "data", "structure"]):
            result = extract_structured_data(request.session_id)
        else:
            result = answer_question(json.dumps({
                "question": request.query,
                "session_id": request.session_id
            }))
        
        return StandardResponse(
            success=True,
            data={"response": result},
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_document(file: UploadFile, session_id: str):
    """Endpoint for document ingestion"""
    try:
        # Save uploaded file
        contents = await file.read()
        file_path = f"data/uploads/{session_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Process and ingest
        text = document_processor.load_document(file_path)
        chunks = document_processor.chunk_document(text)
        hybrid_retriever.ingest_document(chunks, session_id)
        
        return {"status": "success", "message": "Document ingested"}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/session")
async def create_session(user_id: str = "anonymous"):
    """Create new session"""
    session_id = session_manager.create_session(user_id)
    return {"session_id": session_id}        