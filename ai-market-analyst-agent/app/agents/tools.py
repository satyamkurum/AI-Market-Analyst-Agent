# app/agents/tools.py

import logging
import json
from typing import Dict, List
from app.services.retriever import hybrid_retriever
from app.services.llm_service import llm_service
from app.agents.prompts import (
    QA_SYSTEM_PROMPT,
    SUMMARIZER_SYSTEM_PROMPT,
    EXTRACTOR_SYSTEM_PROMPT
)
from langchain.tools import Tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# -------------------------
# Input schemas for tools
# -------------------------
class QAInput(BaseModel):
    input_str: str = Field(description="JSON string with query and session_id")

class SessionInput(BaseModel):
    input_str: str = Field(description="Session ID string")

# -------------------------
# Helper functions
# -------------------------
def _parse_input(input_str: str) -> Dict:
    """
    Parse input consistently across all tools.
    Returns a dict with keys 'session_id' and optionally 'question'/'query'.
    """
    try:
        if isinstance(input_str, dict):
            data = input_str
        elif input_str.strip().startswith("{"):
            data = json.loads(input_str)
        else:
            data = {"session_id": input_str.strip()}
    except json.JSONDecodeError:
        data = {"session_id": input_str.strip()}

    # Ensure keys exist
    if "session_id" not in data:
        data["session_id"] = ""
    if "question" not in data:
        # fallback to 'query'
        data["question"] = data.get("query", "")
    return data

def _get_context(session_id: str, query: str = "", top_k: int = 5) -> str:
    """
    Retrieve context from hybrid retriever and BM25.
    Always returns a non-empty string.
    """
    try:
        context_chunks = hybrid_retriever.retrieve(query=query, session_id=session_id, top_k=top_k, strategy="hybrid")
        if not context_chunks:
            return "No relevant content found for this query."

        # Concatenate all text chunks
        context_text = "\n\n".join([chunk.get("text", "") for chunk in context_chunks if chunk.get("text")])
        if not context_text.strip():
            return "No relevant content found for this query."
        return context_text

    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        return "Error: Failed to retrieve document content. Please try again."

# -------------------------
# Tools implementations
# -------------------------
def answer_question(input_str: str) -> str:
    """Answer questions about the market research report."""
    try:
        data = _parse_input(input_str)
        logger.debug(f"DEBUG: Parsed input: {data}")

        query = data.get("question", "")
        session_id = data.get("session_id", "")

        if not query:
            return "Error: No question provided"
        if not session_id:
            return "Error: No session ID provided"

        logger.info(f"Q&A Tool: Answering '{query}' for session {session_id}")

        context_text = _get_context(session_id, query, top_k=5)

        if "Error" in context_text or "No relevant content" in context_text:
            return "I cannot find relevant information to answer this question in the report."

        response = llm_service.generate_response(
            system_prompt=QA_SYSTEM_PROMPT.format(context=context_text),
            user_prompt=query,
            temperature=0.1
        )
        return response

    except Exception as e:
        logger.error(f"Q&A tool failed: {e}")
        return f"Error: Q&A tool failed: {e}"

def summarize_document(input_str: str) -> str:
    """Generate a summary of the market research report."""
    try:
        data = _parse_input(input_str)
        session_id = data.get("session_id", "")

        if not session_id:
            return "Error: No session ID provided"

        logger.info(f"Summarizer Tool: Generating summary for session {session_id}")

        context_text = _get_context(session_id, "market analysis", top_k=10)

        if "Error" in context_text or "No relevant content" in context_text:
            return "I cannot generate a summary as no document content was found."

        summary = llm_service.generate_response(
            system_prompt=SUMMARIZER_SYSTEM_PROMPT.format(context=context_text),
            user_prompt="Please provide a comprehensive summary of the market research report.",
            temperature=0.2,
            max_tokens=300
        )
        return summary

    except Exception as e:
        logger.error(f"Summarizer tool failed: {e}")
        return f"Error: Summarizer tool failed: {e}"

def extract_structured_data(input_str: str) -> Dict:
    """Extract structured data from the market research report."""
    try:
        data = _parse_input(input_str)
        session_id = data.get("session_id", "")

        if not session_id:
            return {"error": "No session ID provided"}

        logger.info(f"Data Extractor Tool: Extracting data for session {session_id}")

        context_text = _get_context(session_id, "financial data market size", top_k=8)

        if "Error" in context_text or "No relevant content" in context_text:
            return {"error": "No document content found for data extraction"}

        structured_data = llm_service.generate_json_response(
            system_prompt=EXTRACTOR_SYSTEM_PROMPT.format(context=context_text),
            user_prompt="Extract all available market research data in the specified JSON format.",
            temperature=0.1
        )

        if not structured_data or ("error" in structured_data):
            return {"error": "Failed to extract structured data"}

        return structured_data

    except Exception as e:
        logger.error(f"Data extractor tool failed: {e}")
        return {"error": f"Data extractor tool failed: {e}"}

# -------------------------
# LangChain Tool Wrappers
# -------------------------
qa_tool = Tool(
    name="qa_tool",
    func=answer_question,
    description="Answer specific questions about the report. Input: JSON with question and session_id",
    args_schema=QAInput
)

summarizer_tool = Tool(
    name="summarizer_tool",
    func=summarize_document,
    description="Generate a comprehensive summary of the report. Input: session_id",
    args_schema=SessionInput
)

data_extractor_tool = Tool(
    name="data_extractor_tool",
    func=extract_structured_data,
    description="Extract structured data from the report. Input: session_id",
    args_schema=SessionInput
)

# Export tools list
tools = [qa_tool, summarizer_tool, data_extractor_tool]
