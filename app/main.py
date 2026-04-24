from fastapi import FastAPI, HTTPException
from app.models import QueryRequest
from app.engine import RAGEngine
import sys

app = FastAPI(title="Enterprise RAG API", version="1.0")

print("Loading AI Engine into Memory... This will take ~30 seconds.")
try:
    # Initialize the engine globally so it stays warm
    rag = RAGEngine(document_path="data/AIAgents.pdf")
    print("AI Engine loaded successfully! Ready for chats.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load AI Engine: {e}")
    sys.exit(1) # Stop the server immediately if the AI fails to load

# Removed 'async' to prevent the model from blocking the server
@app.post("/api/v1/chat")
def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # The AI engine is already warm, so this will answer in 1-3 seconds
        answer = rag.ask(request.query)
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))