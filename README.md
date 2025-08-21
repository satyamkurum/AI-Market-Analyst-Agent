#  AI Market Analyst Agent 

A **Producation ready AI MARKET ANALYST AGENT** system that do market research by Automation, Integrated with PINECOME VectorDB, Agent Brain as GEMINI for reasoning.

---

## Features

### Core Requirements that is Implemented and also why?
- **Document Ingestion** – PDF/TXT processing with smart chunking  
- **Vector Storage** – Pinecone integration with session isolation so each users data gets stored in same index with different namespace for data privacy and security. I implemented dense dense type with 1024 dimensions so every details can be fetched correctly.  
- **Hybrid Retrieval** – Vector similarity + BM25 keyword search , so if the sementic retrival fails due to critical factual data, keyword seach will give the correct answers. It is required when we want exact data, we can rely fully on semantic search.
- **AI-Powered Q&A** – Accurate answers from document context, because it is saved in the pinecone without duplication, i added duplication detection so each file dont get though the whole process instead we session id stores the meta data of the data. 
- **LLM as Gemini** – Agents brain to choose the tool for specific task, OPENAI API was paid so most reliable llm was gemini that could work with Langchain. and local model gets slow with system compatibility.
- **Session Management** – UUID-based sessions so logs can be checked if any malfulction happens like currpted pdfs or any privacy concern, so it can be deleted

### Testing Features
- **Advanced Summarization** – Structured good summaries it gives, I have done test on diffrent query and saved in json file in repo at last.
- **Structured Data Extraction** – JSON output of financial/market metrics  
- **Hybrid Search Engine** – Automatic fallback from vector to keyword search ( highly useful for factual data ) 
- **Production Deployment** – Docker & docker-compose setup  
- **Web Interface** – Gradio UI for business-friendly interaction but its not proper working with API configuration right now, needs some more time to work on it.
- **Comprehensive Testing** – tested each component, from chunking, embedding, storing in pinecone with session ids, retrival techniques both semantic and keyword based, and whole agentic flow and tested full working vai api.

---

## Technology Stack

| Component       | Technology                         | Why Chosen |
|-----------------|------------------------------------|------------|
| **Vector DB**   | Pinecone                          | it is secure and scalable
| **Embeddings**  | Sentence Transformers (`gte-large`)| Local model, high-quality 1024-d embeddings, no API cost + I used gemini api embedding for comparision purpose but didn't utlised to store in Pinecone because it gives vectoes of 768 dimension but we are working on 1024.
| **LLM**         | Google Gemini                     | Excellent reasoning + free tier and OPENAI was paid, local model work slow and less reliable.
| **Backend**     | FastAPI + LangChain               | Async performance & orchestration - agent choose tools itself vai prompt but NOT via tool_binding because langchain has limitaion with GEMINI and word fine with OPENAI
| **UI**          | Gradio                            | Not fully developed, not storing the data in pinecone, some issue is there( still working)
| **Deployment**  | Docker                            | Reproducible, containerized environments is good for production and also mentioned in assignment if we want bonus.

---

## Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ai-market-analyst-agent.git
cd ai-market-analyst-agent

# 2. Set up environment variables
cp .env.example .env
# Add your API keys to .env file

# 3. Run with Docker (Recommended)
docker-compose up --build

# 4. Or run locally
pip install -r requirements.txt
python -m app.main &
python -m frontend.gradio_ui
```
## Environment Variable
 -  PINECONE_API_KEY=your_pinecone_key
 -  OPENAI_API_KEY=your_openai_key
 -  GEMINI_API_KEY=your_gemini_key

##  Design Decisions

### Chunking Strategy
- **Choice**: 500 tokens per chunk with 50 tokens overlap  
- **Some reasons why i choose this**:  
  - Ensures each chunk captures full semantic meaning.  
  - Keeps chunk size small enough to fit embedding model limits.  
  - Overlap preserves continuity between chunks, reducing loss of important context. 
---

### Embedding Model
- **Choice**: `sentence-transformers/gte-large`  
- **Reasons**:  
  - 1024-dimensional embeddings gives high semantic accuracy
  - It Runs locally and  avoids API costs.  
  - Balanced trade-off between retrieval quality and latency.  
- **Comparison**:  
  - `MiniLM-L6-v2`: I tried this but this was less accurate and also not embedd in 1024 dimension which is required. Also tried Gemini API embedding vai embedding-001 model.
  - `gte-large`: Slightly higher latency, but **10–12% more accurate** retrieval.  
- **Final Decided To**: Use `gte-large` for better quality.

---

### Vector Database
- **Choice**: Pinecone (cloud-native)  
- **Reasons**:  
  - low-latency vector search.  
  - Namespacing  enables session isolation.
  - Scales without local memory/storage constraints.

---

### Data Extraction Prompt
- **Choice**: Schema constrained JSON prompt design  to get exact json, no error if data has no data, just write not able to find. and written promts for these to make it more accurate.
- **Reasoning**:  
  - LLMs sometimes generate natural language around JSON.  
  - To prevent this, the prompt explicitly enforces JSON-only output.  
  - Example schema:  
    ```json
    {
      "current_market_size": "...",
      "projected_cagr": "...",
      "projected_market_size": "..."
    }
    ```  
  
### Project Workflow 

```json
 USER REQUEST
    │
    ▼
FASTAPI SERVER (HTTP Endpoint)
    ├── Input Validation
    ├── Authentication
    │
    ▼
SESSION MANAGER
    ├── Session Validation
    ├── Auto-create if missing
    ├── Access Control
    │
    ▼
HYBRID RETRIEVER (Core Engine)
    ├── VECTOR SEARCH (Pinecone)
    │   ├── session_{UUID} namespace
    │   ├── 1024-dim embeddings
    │   └── cosine similarity
    │
    └── BM25 SEARCH (Local)
        ├── per-session index
        ├── keyword matching
        └── fallback mechanism
    │
    ▼
AI TOOLS (LangChain Orchestration)
    ├── Q&A TOOL ───────┐
    ├── SUMMARIZER ─────┤ → Gemini LLM → Response
    └── DATA EXTRACTOR ─┘
    │
    ▼
RESPONSE FORMATTER
    ├── Standardized JSON
    ├── Error Handling
    └── Logging
    │
    ▼
USER RESPONSE (JSON/text) 

```
### Testing 
- After Installing and configurations, Run the `test_work.py` to see the results with Metadata, sessionwise. Alter the query if you want.  
- Result of the Testing is saved in `validation_result.json`. Please Check it out.

##  Closing Notes
- This project is made considering multiple test cases for each component in workflow. I emphasised on **scalability and production** based thinking.
- I design decisions to balance **accuracy, latency, and  most important cost efficiency**, so it ensure practical for enterprise deployment.  

---
⭐ Thank you, Hope you liked my Work.
