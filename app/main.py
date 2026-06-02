from fastapi import FastAPI, HTTPException, UploadFile, File
from app.models import QueryRequest
from app.engine import RAGEngine
import sys
import os
import threading

app = FastAPI(title="Enterprise RAG API", version="1.0")

MAX_FILE_SIZE_MB = 50
DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Engine startup
# ---------------------------------------------------------------------------

print("Initializing AI Engine...")
try:
    rag = RAGEngine()
    print("AI Engine ready. Models will load on first query.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize AI Engine: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ingest_in_background(file_path: str, filename: str):
    """Embed a PDF in a background thread so /chat stays responsive."""
    print(f"⏳ Ingesting '{filename}'...")
    try:
        result = rag.ingest_pdf(file_path)
        print(f"✓ '{filename}' ingested — {result['chunks']} chunks")
    except Exception as e:
        print(f"✗ Failed to ingest '{filename}': {e}")


def _auto_ingest_existing_pdfs():
    """
    On startup, scan the data/ folder and ingest any PDFs already present.
    Runs sequentially (not parallel) to avoid hammering the CPU at boot.
    """
    if not os.path.exists(DATA_DIR):
        print(f"'{DATA_DIR}/' folder not found — skipping auto-ingestion.")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found in data/ — waiting for uploads.")
        return

    print(f"Auto-ingesting {len(pdf_files)} PDF(s) found in data/...")
    for pdf_file in pdf_files:
        _ingest_in_background(os.path.join(DATA_DIR, pdf_file), pdf_file)


# Kick off auto-ingestion in the background so the server starts immediately
threading.Thread(target=_auto_ingest_existing_pdfs, daemon=True).start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/v1/chat")
def chat(request: QueryRequest):
    """Ask a question against all ingested documents."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        answer = rag.ask(request.query)
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and start embedding it in the background.
    Returns immediately with status='processing'.
    Poll GET /api/v1/documents to check when it becomes 'ready'.
    """
    # 1. Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # 2. Read content and check size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        )

    # 3. Reject duplicates that are already ingested or being processed
    current_docs = {doc["filename"]: doc["status"] for doc in rag.list_documents()}
    if file.filename in current_docs:
        status = current_docs[file.filename]
        if status == "ready":
            raise HTTPException(status_code=409, detail=f"'{file.filename}' is already ingested.")
        if status == "processing":
            raise HTTPException(status_code=409, detail=f"'{file.filename}' is already being processed.")

    # 4. Save to disk (persists across Docker restarts via the volume mount)
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    # 5. Mark as processing immediately (visible before background thread starts)
    rag._ingested_docs[file.filename] = {"status": "processing", "chunks": 0}

    # 6. Run embedding in the background so this request returns fast
    threading.Thread(
        target=_ingest_in_background,
        args=(file_path, file.filename),
        daemon=True,
    ).start()

    return {
        "status": "processing",
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "message": (
            f"'{file.filename}' uploaded successfully. "
            "Embedding is running in the background — "
            "use GET /api/v1/documents to track progress."
        ),
    }


@app.get("/api/v1/documents")
def list_documents():
    """List all ingested documents and their current status."""
    return {"status": "success", "documents": rag.list_documents()}