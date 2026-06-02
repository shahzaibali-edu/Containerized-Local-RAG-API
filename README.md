# Containerized Local RAG API

A completely localized, offline Retrieval-Augmented Generation (RAG) pipeline. This microservice architecture separates a FastAPI backend from a Gradio frontend, completely containerized via Docker for consistent local deployment — no external APIs or cloud services required.

## Architecture & Tech Stack
- **Framework:** FastAPI, Python 3.10
- **AI Pipeline:** LangChain, Hugging Face
- **Local LLM:** `HuggingFaceTB/SmolLM2-135M-Instruct` (Local CPU execution)
- **Vector Database:** DuckDB (in-memory)
- **Containerization:** Docker & Docker Compose
- **Frontend:** Gradio (tabbed UI)

## Key Features
- **100% Local Inference:** No OpenAI API keys or internet connection required after the initial model download.
- **Runtime PDF Uploads:** Upload any PDF at runtime through the UI or API — no container restart needed. Multiple PDFs are supported simultaneously.
- **Auto-Ingestion:** PDFs placed in the `data/` folder before startup are automatically embedded when the container boots.
- **Non-Blocking Ingestion:** PDF embedding runs in a background thread so the chat endpoint stays responsive during upload processing.
- **Upload Safety:** Duplicate detection, 50 MB file size limit, and PDF-only validation on the upload endpoint.
- **Live Status Tracking:** Poll `GET /api/v1/documents` to monitor ingestion progress (`processing` → `ready`).
- **Docker Volume Caching:** Model weights are cached to a local volume (`hf_cache/`) to prevent re-downloading on container restarts.
- **CPU Optimized:** Configured with `low_cpu_mem_usage` and `.safetensors` to run reliably on consumer laptops without a GPU.
- **Modular UI:** The Gradio frontend is completely decoupled from the API backend.

## Challenges & Engineering Workarounds

Building a local RAG pipeline without cloud GPUs presented several hardware and containerization limits that required specific workarounds:

* **Docker/WSL Storage Crashes:** Downloading massive PyTorch and Hugging Face weights inside an ephemeral Docker container repeatedly corrupted the WSL `.vhdx` virtual drive. **Fix:** Migrated the Docker engine to a secondary drive and implemented local volume mapping (`./hf_cache:/root/.cache/huggingface`). This safely stores the model on the host OS and eliminates 300MB+ downloads on every container restart.
* **CPU Bottlenecks & FastAPI Freezes:** Running a local LLM blocks the main execution thread, which originally caused the asynchronous FastAPI server to freeze. **Fix:** Changed the inference endpoint from `async def` to a synchronous `def` to utilize FastAPI's background thread pooling, and pre-warmed the AI model in memory during the app's startup sequence to cut the "cold start" delay from 30s to ~2s for warm queries.
* **Small Model Hallucinations:** The 135M parameter model lacks the reasoning capacity of larger models and frequently hallucinated facts outside the PDF context. **Fix:** Bound the model to strict ChatML formatting (`<|im_start|>`) in the prompt template, creating an explicit "escape hatch" that forces the model to surrender with *"I do not have info on that"* instead of guessing.
* **Blocking PDF Ingestion:** Embedding a new PDF is CPU-intensive and would have stalled all chat requests. **Fix:** Ingestion runs in a `threading.Thread` with a lock on the vector store writer, keeping the `/chat` endpoint available throughout.

## Setup & Installation

### Prerequisites
- Docker Desktop running
- Python 3.10+ (for the Gradio UI)
- A Hugging Face account and access token

### 1. Clone & Configure
```bash
git clone https://github.com/shahzaibali-edu/Containerized-Local-RAG-API.git
cd Containerized-Local-RAG-API
```

Copy the environment template and add your token:
```bash
cp .env.example .env
# Edit .env and set your HF_TOKEN
```

### 2. (Optional) Pre-load PDFs
Drop any PDF files into the `data/` folder. They will be automatically ingested when the container starts.

### 3. Start the Backend
```bash
docker-compose up --build
```
> **First boot:** Docker downloads model weights to `hf_cache/` — this takes a few minutes. Subsequent boots are instant.

Watch for this in the logs before proceeding:
```
✓ 'YourFile.pdf' ingested — XX chunks
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
| `GET`  | `/api/v1/documents` | List all ingested documents and their status |
| `GET`  | `/docs` | Auto-generated Swagger UI for API testing |

## Performance Notes
This pipeline is configured for local CPU-only inference. Response times range from 20–50 seconds depending on hardware. To upgrade, swap the `model_id` in `app/engine.py` for a larger model (e.g., `Qwen/Qwen2.5-1.5B-Instruct`).
