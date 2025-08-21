# test_system_validation.py
import logging
import json
import time
import uuid
from datetime import datetime
from app.core.session_manager import session_manager
from app.utils.document_processor import document_processor
from app.services.retriever import hybrid_retriever
from app.agents.tools import answer_question, summarize_document, extract_structured_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_complete_system():
    """Test the complete system with various query types - WITH ANSWERS STORED"""
    print("🎯 COMPREHENSIVE SYSTEM VALIDATION")
    print("=" * 60)
    
    results = {}
    detailed_answers = {}
    test_session = None
    
    try:
        # 1. Setup session using YOUR EXACT create_session method
        print("1. Creating session with your session manager...")
        test_session = session_manager.create_session("validation_test")
        print(f"   ✅ Session created: {test_session}")
        
        # 2. Ingest document
        print("2. Ingesting document...")
        text = document_processor.load_document("data/Innovate-Inc-Report.txt")
        chunks = document_processor.chunk_document(text)
        hybrid_retriever.ingest_document(chunks, test_session)
        print(f"   ✅ Document ingested into session: {test_session}")
        
        # 3. Test different query types
        test_cases = [
            {
                "type": "Q&A_Specific",
                "query": "What is FutureFlow's market share percentage?",
                "description": "Specific fact extraction"
            },
            {
                "type": "Q&A_General", 
                "query": "Who are the main competitors mentioned in the report?",
                "description": "General information retrieval"
            },
            {
                "type": "Summary",
                "query": "Provide a comprehensive summary",
                "description": "Document summarization"
            },
            {
                "type": "Data_Extraction", 
                "query": "Extract all financial metrics and market data",
                "description": "Structured data extraction"
            },
            {
                "type": "Q&A_Numeric",
                "query": "What is the current market size value?",
                "description": "Numeric data extraction"
            }
        ]
        
        # 4. Execute all test cases
        for i, test_case in enumerate(test_cases, 3):
            print(f"\n{i}. Testing {test_case['type']}: '{test_case['query']}'")
            print(f"   📋 {test_case['description']}")
            
            try:
                if test_case['type'] == 'Summary':
                    result = summarize_document(test_session)
                elif test_case['type'] == 'Data_Extraction':
                    result = extract_structured_data(test_session)
                else:
                    # For Q&A cases
                    qa_input = json.dumps({
                        "question": test_case["query"],
                        "session_id": test_session
                    })
                    result = answer_question(qa_input)
                
                # Store the detailed answer
                detailed_answers[test_case['type']] = {
                    "query": test_case["query"],
                    "response": result,
                    "response_length": len(str(result)),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Validate result
                is_valid = False
                if test_case['type'] == 'Data_Extraction':
                    is_valid = isinstance(result, dict) and not result.get('error')
                else:
                    is_valid = result and not result.startswith(("Error:", "I cannot find")) and len(result.strip()) > 20
                
                if is_valid:
                    results[test_case['type']] = "✅ PASS"
                    print(f"   ✅ SUCCESS: {test_case['type']}")
                    print(f"   📝 Response length: {len(str(result))} characters")
                else:
                    results[test_case['type']] = "❌ FAIL"
                    print(f"   ❌ FAILED: {test_case['type']}")
                    print(f"   💬 Result: {result}")
                    
            except Exception as e:
                results[test_case['type']] = f"❌ ERROR: {str(e)}"
                detailed_answers[test_case['type']] = {
                    "query": test_case["query"],
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                print(f"   ❌ ERROR in {test_case['type']}: {e}")
            
            # Add delay to avoid rate limits
            time.sleep(3)
        
        # 5. Test direct retrieval
        print(f"\n{len(test_cases)+3}. Testing direct vector retrieval...")
        try:
            retrieval_results = hybrid_retriever.retrieve("market share", test_session, top_k=2)
            detailed_answers["Vector_Retrieval"] = {
                "query": "market share",
                "results_count": len(retrieval_results),
                "results": retrieval_results,
                "timestamp": datetime.now().isoformat()
            }
            
            if len(retrieval_results) > 0:
                results["Vector_Retrieval"] = "✅ PASS"
                print(f"   ✅ Retrieval SUCCESS: {len(retrieval_results)} results")
            else:
                results["Vector_Retrieval"] = "❌ FAIL"
                print("   ❌ Retrieval FAILED: No results found")
        except Exception as e:
            results["Vector_Retrieval"] = f"❌ ERROR: {str(e)}"
            detailed_answers["Vector_Retrieval"] = {
                "query": "market share",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            print(f"   ❌ Retrieval ERROR: {e}")
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return {"status": "❌ CRASHED", "error": str(e), "session": test_session}
    
    # Print final results
    print("\n" + "=" * 60)
    print("🎯 FINAL VALIDATION RESULTS:")
    print("=" * 60)
    
    for test_name, status in results.items():
        print(f"{test_name:.<25}{status}")
    
    # Calculate success rate
    total_tests = len(results)
    passed_tests = sum(1 for status in results.values() if "✅" in status)
    success_rate = (passed_tests / total_tests) * 100
    
    print("=" * 60)
    print(f"📊 Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    print(f"🎯 Session ID: {test_session}")
    
    if success_rate >= 80:
        print("🚀 SYSTEM VALIDATION: SUCCESS!")
        status = "✅ SUCCESS"
    elif success_rate >= 50:
        print("⚠️ SYSTEM VALIDATION: PARTIAL SUCCESS")
        status = "⚠️ PARTIAL"
    else:
        print("❌ SYSTEM VALIDATION: FAILED")
        status = "❌ FAILED"
    
    return {
        "status": status,
        "success_rate": f"{success_rate:.1f}%",
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "session_id": test_session,
        "results": results,
        "detailed_answers": detailed_answers,  # ← ALL ANSWERS STORED HERE
        "timestamp": datetime.now().isoformat(),
        "test_date": datetime.now().strftime("%Y-%m-%d"),
        "system_version": "1.0"
    }

def save_results_to_file(results, filename="validation_results.json"):
    """Save test results to JSON file with pretty formatting"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {filename}")
    
    # Also save a readable version
    readable_file = "validation_results_readable.json"
    with open(readable_file, 'w') as f:
        # Create a more readable format
        readable_data = {
            "summary": {
                "status": results["status"],
                "success_rate": results["success_rate"],
                "passed_tests": results["passed_tests"],
                "total_tests": results["total_tests"],
                "session_id": results["session_id"],
                "test_date": results["test_date"]
            },
            "detailed_results": results["detailed_answers"]
        }
        json.dump(readable_data, f, indent=2, default=str)
    print(f"📖 Readable version saved to: {readable_file}")

if __name__ == "__main__":
    # Run the comprehensive test
    print("🚀 Starting comprehensive system validation...")
    test_results = test_complete_system()
    
    # Save results to file
    save_results_to_file(test_results)
    
    # Print final summary
    print(f"\n🔚 TEST COMPLETE: {test_results['status']}")
    print(f"📈 Success Rate: {test_results['success_rate']}")
    print(f"🆔 Session ID: {test_results['session_id']}")
    print(f"📊 Total answers collected: {len(test_results['detailed_answers'])}")
    print(f"💾 Results stored in: validation_results.json")