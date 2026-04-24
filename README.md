# Containerized Local RAG API

A completely localized, offline Retrieval-Augmented Generation (RAG) pipeline. This microservice architecture separates a FastAPI backend from a Gradio frontend, completely containerized via Docker for consistent local deployment without relying on external APIs.

## Architecture & Tech Stack
- **Framework:** FastAPI, Python 3.10
- **AI Pipeline:** LangChain, Hugging Face
- **Local LLM:** `HuggingFaceTB/SmolLM2-135M-Instruct` (Local CPU execution)
- **Vector Database:** DuckDB 
- **Containerization:** Docker & Docker Compose
- **Frontend:** Gradio

## Key Features
- **100% Local Inference:** No OpenAI API keys or internet connection required after the initial model download.
- **Docker Volume Caching:** Model weights are cached to a local volume (`hf_cache`) to prevent redownloading on container restarts and to bypass WSL storage limits.
- **CPU Optimized:** Configured with `low_cpu_mem_usage` and `.safetensors` to run reliably on consumer laptops without GPU acceleration.
- **Modular UI:** The Gradio frontend is completely decoupled from the API backend.

## Challenges & Engineering Workarounds

Building a local RAG pipeline without cloud GPUs presented several hardware and containerization limits that required specific workarounds:

* **Docker/WSL Storage Crashes:** Downloading massive PyTorch and Hugging Face weights inside an ephemeral Docker container repeatedly corrupted the WSL `.vhdx` virtual drive. **Fix:** Migrated the Docker engine to a secondary drive and implemented local volume mapping (`./hf_cache:/root/.cache/huggingface`). This safely stores the model on the host OS and eliminates 300MB+ downloads on every container restart.
* **CPU Bottlenecks & FastAPI Freezes:** Running a local LLM blocks the main execution thread, which originally caused the asynchronous FastAPI server to freeze. **Fix:** Changed the inference endpoint from `async def` to a synchronous `def` to utilize FastAPI's background thread pooling, and pre-warmed the AI model in memory during the app's startup sequence to cut the "cold start" delay from 30s to ~2s for warm queries.
* **Small Model Hallucinations:** The 135M parameter model lacks the reasoning capacity of larger models and frequently hallucinated facts outside the PDF context. **Fix:** Bound the model to strict ChatML formatting (`<|im_start|>`) in the prompt template, creating an explicit "escape hatch" that forces the model to surrender with *"I do not have info on that"* instead of guessing.

## Setup & Installation

### 1. Backend (Docker)
Ensure Docker Desktop is running. Clone the repository and navigate into the project directory:

```bash
git clone [https://github.com/YourUsername/Containerized-Local-RAG-API.git](https://github.com/YourUsername/Containerized-Local-RAG-API.git)
cd Containerized-Local-RAG-API
```

Place your target PDF document in the `/data` directory.

Build and start the container:
```bash
docker-compose up --build
```
*Note: The first boot will take some time to download the model weights to the local volume. Subsequent boots will be instant.*

### 2. Frontend (Gradio)
Open a separate standard terminal (not Docker), install the UI requirements, and launch the client:

```bash
pip install gradio requests
python ui.py
```
This will generate a local web address (usually `http://127.0.0.1:7860`) where you can interact with the RAG pipeline.

## API Endpoints
- `POST /api/v1/chat`: Main inference endpoint. Expects JSON: `{"query": "your question"}`.
- `GET /docs`: Auto-generated Swagger documentation for API testing.

## Performance Notes
This pipeline is currently configured for local CPU-only inference. Response times range from 20-50 seconds depending on local hardware. To upgrade for production or GPU-enabled environments, swap the `model_id` in `engine.py` to a larger parameter model.
