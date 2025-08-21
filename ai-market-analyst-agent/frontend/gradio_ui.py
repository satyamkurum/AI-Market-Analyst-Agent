# frontend/gradio_ui.py - DEBUG VERSION
import gradio as gr
import time
import uuid
import json
from app.core.session_manager import session_manager
from app.utils.document_processor import document_processor
from app.services.retriever import hybrid_retriever
from app.agents.tools import answer_question
from app.services.vector_store import vector_store

class DebugUI:
    def __init__(self):
        self.session_id = None
    
    def process_document(self, file):
        """DEBUG VERSION: Test each step"""
        try:
            # 1. TEST SESSION CREATION
            self.session_id = f"session_{uuid.uuid4()}"  # session_ prefix for Pinecone
            yield f"📋 **Session Created:** {self.session_id}", ""
            time.sleep(1)
            
            # 2. TEST FILE SAVING
            file_path = f"data/uploads/{self.session_id}_{file.name}"
            with open(file_path, "wb") as f:
                file.save(file_path)
            yield f"💾 **2. File Saved:** {file.name}", ""
            time.sleep(1)
            
            # 3. TEST DOCUMENT LOADING
            yield "📄 **3. Loading document...**", ""
            text = document_processor.load_document(file_path)
            yield f"✅ **3. Document loaded:** {len(text)} characters", ""
            time.sleep(1)
            
            # 4. TEST CHUNKING
            yield "✂️ **4. Splitting into chunks...**", ""
            chunks = document_processor.chunk_document(text)
            yield f"✅ **4. Document chunked:** {len(chunks)} chunks", ""
            time.sleep(1)
            
            # 5. TEST EMBEDDING GENERATION
            yield "🧠 **5. Generating embeddings...**", ""
            from app.services.embedding_service import embedding_service
            sample_text = chunks[0][:100] if chunks else "test"
            embeddings = embedding_service.embed([sample_text])
            yield f"✅ **5. Embeddings generated:** {len(embeddings[0])} dimensions", ""
            time.sleep(1)
            
            # 6. TEST PINECONE INGESTION
            yield "🚀 **6. Storing in Pinecone...**", ""
            hybrid_retriever.ingest_document(chunks, self.session_id)
            time.sleep(2)
            
            # 7. VERIFY PINECONE STORAGE
            yield "🔍 **7. Verifying Pinecone storage...**", ""
            stats = vector_store.get_session_stats(self.session_id)
            yield f"📊 **7. Pinecone stats:** {stats}", ""
            
            # 8. TEST RETRIEVAL
            yield "🔎 **8. Testing retrieval...**", ""
            test_results = hybrid_retriever.retrieve("test", self.session_id, top_k=1)
            yield f"✅ **8. Retrieval test:** {len(test_results)} results found", ""
            
            return f"🎉 **DEBUG COMPLETE!** Session: {self.session_id}", ""
            
        except Exception as e:
            return f"❌ **ERROR at step:** {str(e)}", ""
    
    def ask_question(self, question):
        """DEBUG: Test Q&A"""
        if not self.session_id:
            yield "⚠️ Please upload a document first!"
            return
        
        try:
            # TEST RETRIEVAL FIRST
            yield "🔍 **Testing retrieval...**"
            results = hybrid_retriever.retrieve(question, self.session_id, top_k=3)
            yield f"📊 **Found {len(results)} relevant chunks**"
            time.sleep(1)
            
            if len(results) == 0:
                yield "❌ **No results found in Pinecone!**"
                return
            
            # TEST Q&A
            yield "🤖 **Asking AI...**"
            qa_input = json.dumps({
                "question": question,
                "session_id": self.session_id
            })
            result = answer_question(qa_input)
            
            yield f"✅ **Answer:**\n\n{result}"
            
        except Exception as e:
            yield f"❌ **Error:** {str(e)}"
    
    def launch_ui(self):
        with gr.Blocks(title="AI Market Analyst - DEBUG", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🐛 DEBUG MODE: AI Market Analyst")
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload PDF/TXT", file_types=[".pdf", ".txt"])
                    upload_btn = gr.Button("📤 Upload & Debug", variant="primary")
                    status_output = gr.Markdown(label="Debug Status")
                
                with gr.Column(scale=2):
                    question_input = gr.Textbox(label="Test Question", placeholder="Ask something...", lines=2)
                    ask_btn = gr.Button("🔍 Test Q&A", variant="primary")
                    answer_output = gr.Markdown(label="Test Results")
            
            upload_btn.click(self.process_document, inputs=file_input, outputs=[status_output, answer_output])
            ask_btn.click(self.ask_question, inputs=question_input, outputs=answer_output)
        
        demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    ui = DebugUI()
    ui.launch_ui()