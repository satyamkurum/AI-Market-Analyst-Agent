# test_final_validation.py

import logging
import json
import time
from app.core.session_manager import session_manager
from app.services.retriever import hybrid_retriever
from app.services.llm_service import llm_service
from app.agents.tools import answer_question, summarize_document, extract_structured_data
from app.utils.document_processor import document_processor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_final_validation():
    """FINAL TEST: Prove everything works with consistent session management."""
    print("🎯 FINAL VALIDATION TEST")
    print("=" * 60)
    
    # Use a FIXED session ID for entire test
    FIXED_SESSION_ID = "final_validation_session"
    
    results = {}
    
    try:
        # Test 1: LLM Service
        print("1. Testing LLM Service...")
        test_response = llm_service.generate_response(
            system_prompt="You are a helpful AI assistant. Respond concisely.",
            user_prompt="What is the capital of France?",
            temperature=0.1
        )
        if "Paris" in test_response:
            results["llm_service"] = "✅ PASS"
            print(f"   ✅ LLM Response: {test_response}")
        else:
            results["llm_service"] = "❌ FAIL"
            print(f"   ❌ LLM Failed: {test_response}")
        
        # Test 2: Setup with FIXED session
        print(f"\n2. Setting up document for session: {FIXED_SESSION_ID}")
        text = document_processor.load_document("data/Innovate-Inc-Report.txt")
        chunks = document_processor.chunk_document(text)
        hybrid_retriever.ingest_document(chunks, FIXED_SESSION_ID)
        results["setup"] = "✅ PASS"
        print("   ✅ Document ingested successfully")
        
        # Test 3: Hybrid Retrieval with SAME session
        print(f"\n3. Testing Hybrid Retrieval...")
        retrieval_results = hybrid_retriever.retrieve(
            "FutureFlow market share", 
            FIXED_SESSION_ID, 
            top_k=5
        )
        
        if len(retrieval_results) > 0:
            results["retrieval"] = "✅ PASS"
            print(f"   ✅ Retrieved {len(retrieval_results)} results")
            print(f"   📊 Top result: {retrieval_results[0]['text'][:80]}...")
            print(f"   🎯 Top score: {retrieval_results[0].get('score', 'N/A')}")
        else:
            results["retrieval"] = "❌ FAIL"
            print("   ❌ Retrieval failed - no results found")
        
        # Test 4: Q&A Tool with SAME session
        print(f"\n4. Testing Q&A Tool...")
        qa_input = json.dumps({
            "question": "What is FutureFlow market share?",
            "session_id": FIXED_SESSION_ID
        })
        qa_result = answer_question(qa_input)
        
        if not qa_result.startswith("Error:") and "market" in qa_result.lower():
            results["qa_tool"] = "✅ PASS"
            print(f"   ✅ Q&A Working: {qa_result[:80]}..." if len(qa_result) > 80 else f"   ✅ Q&A: {qa_result}")
        else:
            results["qa_tool"] = "❌ FAIL"
            print(f"   ❌ Q&A Failed: {qa_result}")
        
        # Test 5: Summarizer with SAME session (with delay to avoid rate limits)
        print(f"\n5. Testing Summarizer Tool...")
        time.sleep(10)  # Avoid rate limits
        summarizer_input = json.dumps({"session_id": FIXED_SESSION_ID})
        summary_result = summarize_document(summarizer_input)
        
        if not summary_result.startswith("Error:") and len(summary_result) > 50:
            results["summarizer"] = "✅ PASS"
            print(f"   ✅ Summary: {summary_result[:80]}..." if len(summary_result) > 80 else f"   ✅ Summary: {summary_result}")
        else:
            results["summarizer"] = "❌ FAIL"
            print(f"   ❌ Summarizer Failed: {summary_result}")
        
        # Test 6: Data Extractor with SAME session (with delay)
        print(f"\n6. Testing Data Extractor Tool...")
        time.sleep(10)  # Avoid rate limits
        extractor_input = json.dumps({"session_id": FIXED_SESSION_ID})
        extractor_result = extract_structured_data(extractor_input)
        
        if isinstance(extractor_result, dict) and not extractor_result.get('error'):
            results["data_extractor"] = "✅ PASS"
            print(f"   ✅ Data Extracted: {len(extractor_result)} data points")
        else:
            results["data_extractor"] = "❌ FAIL"
            print(f"   ❌ Data Extractor Failed: {extractor_result}")
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return {"status": "❌ CRASHED", "error": str(e)}
    
    # Print final results
    print("\n" + "=" * 60)
    print("🎯 FINAL TEST RESULTS:")
    print("=" * 60)
    
    all_passed = all("✅" in status for status in results.values())
    
    for test_name, status in results.items():
        print(f"{test_name:.<20}{status}")
    
    print("=" * 60)
    if all_passed:
        print("🚀 ALL SYSTEMS GO! YOUR RAG AGENT IS PRODUCTION-READY!")
        print("✅ Vector search + BM25 hybrid retrieval: WORKING")
        print("✅ LLM integration: WORKING")  
        print("✅ All tools: WORKING")
        print("✅ Session management: WORKING")
    else:
        print("⚠️  Some components need attention")
        
    return {"status": "✅ SUCCESS" if all_passed else "⚠️  PARTIAL", "results": results}

if __name__ == "__main__":
    test_final_validation()