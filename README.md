# Containerized Local RAG API

A completely localized, offline Retrieval-Augmented Generation (RAG) pipeline. This microservice architecture separates a FastAPI backend from a Gradio frontend, completely containerized via Docker for consistent local deployment — no external APIs or cloud services required.

## Architecture & Tech Stack
- **Framework:** FastAPI, Python 3.10
- **AI Pipeline:** LangChain, Hugging Face
- **Local LLM:** `HuggingFaceTB/SmolLM2-135M-Instruct` (Local CPU execution)
- **Vector Database:** DuckDB (persistent, file-backed)
- **Containerization:** Docker & Docker Compose
- **Frontend:** Gradio (tabbed UI)

## Key Features
- **100% Local Inference:** No OpenAI API keys or internet connection required after the initial model download.
- **Runtime PDF Uploads:** Upload any PDF at runtime through the UI or API — no container restart needed. Multiple PDFs are supported simultaneously.
- **Auto-Ingestion on Boot:** PDFs placed in the `data/` folder are automatically embedded when the container starts.
- **Persistent Vector Store:** Embeddings are saved to `data/vectorstore.duckdb` via a Docker volume. On restart, the vector store is restored from disk instantly — no re-embedding.
- **Ingestion Manifest:** A `data/manifest.json` file tracks every document's state (`processing` → `ready`) across container restarts. Crash recovery is handled automatically.
- **Non-Blocking Ingestion:** PDF embedding runs in a background thread so the `/chat` endpoint stays responsive during uploads.
- **Upload Safety:** Duplicate detection, 50 MB file size limit, and PDF-only validation on the upload endpoint.
- **Live Status Tracking:** Poll `GET /api/v1/documents` to monitor ingestion progress in real time.
- **Docker Volume Caching:** HuggingFace model weights are cached to `hf_cache/` to prevent re-downloading on container restarts.
- **CPU Optimized:** Configured with `low_cpu_mem_usage` and `.safetensors` to run reliably on consumer laptops without a GPU.
- **Modular UI:** The Gradio frontend is fully decoupled from the API backend.

## Challenges & Engineering Workarounds

Building a local RAG pipeline without cloud GPUs presented several hardware and containerization limits that required specific workarounds:

* **Docker/WSL Storage Crashes:** Downloading massive PyTorch and Hugging Face weights inside an ephemeral Docker container repeatedly corrupted the WSL `.vhdx` virtual drive. **Fix:** Migrated the Docker engine to a secondary drive and implemented local volume mapping (`./hf_cache:/root/.cache/huggingface`). This safely stores the model on the host OS and eliminates 300MB+ downloads on every container restart.

* **CPU Bottlenecks & FastAPI Freezes:** Running a local LLM blocks the main execution thread, which originally caused the asynchronous FastAPI server to freeze. **Fix:** Changed the inference endpoint from `async def` to a synchronous `def` to utilize FastAPI's background thread pooling, and pre-warmed the AI model in memory during the app's startup sequence to cut the "cold start" delay.

* **Small Model Hallucinations:** The 135M parameter model lacks the reasoning capacity of larger models and frequently hallucinated facts outside the PDF context. **Fix:** Bound the model to strict ChatML formatting (`<|im_start|>`) in the prompt template, creating an explicit "escape hatch" that forces the model to surrender with *"I do not have info on that"* instead of guessing.

* **Blocking PDF Ingestion:** Embedding a new PDF is CPU-intensive and would have stalled all chat requests. **Fix:** Ingestion runs in a `threading.Thread` with a lock on the vector store writer, keeping the `/chat` endpoint available throughout.

* **In-Memory Vector Store Lost on Restart:** The default LangChain DuckDB integration creates an in-memory database that is destroyed on every container restart, forcing a full re-embedding of all documents. **Fix:** Replaced the in-memory connection with a persistent `duckdb.connect("data/vectorstore.duckdb")` handle backed by a Docker volume mount. A companion `manifest.json` tracks ingestion state, allowing the engine to skip re-embedding on normal restarts and gracefully handle crash recovery.

## Setup & Installation

### Prerequisites
- Docker Desktop (running)
- Python 3.10+ (for the Gradio UI, runs on the host)
- A Hugging Face account and access token

### 1. Clone & Configure
```bash
git clone https://github.com/shahzaibali-edu/Containerized-Local-RAG-API.git
cd Containerized-Local-RAG-API
```

Copy the environment template and fill in your token:
```bash
cp .env.example .env
# Open .env and set:  HF_TOKEN=your_token_here
```

### 2. (Optional) Pre-load PDFs
Drop any PDF files into the `data/` folder. They will be automatically ingested when the container starts and will persist across future restarts.

### 3. Start the Backend
```bash
docker-compose up --build
```
> **First boot:** Docker downloads model weights to `hf_cache/` and embeds any PDFs in `data/`. Subsequent boots restore from the persistent vector store instantly — no re-embedding.

Watch for these messages in the logs:
```
✓ 'YourFile.pdf' ingested — XX chunks     ← first boot
✓ Restored vectorstore: N document(s) loaded from disk  ← subsequent boots
```

### 4. Launch the Frontend
In a separate terminal:
```bash
pip install gradio requests
python ui.py
```
Open **http://127.0.0.1:7860** to access the chat interface.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Ask a question. Body: `{"query": "your question"}` |
| `POST` | `/api/v1/upload` | Upload a PDF for ingestion. Multipart form: `file=<pdf>` |
| `GET`  | `/api/v1/documents` | List all ingested documents and their current status |
| `GET`  | `/docs` | Auto-generated Swagger UI for API testing |

## Performance Notes
This pipeline is configured for local CPU-only inference. Response times range from 20–50 seconds depending on hardware. To upgrade, swap the `model_id` in `app/engine.py` for a larger model (e.g., `Qwen/Qwen2.5-1.5B-Instruct`).

## License
This project is licensed under the [MIT License](LICENSE).
