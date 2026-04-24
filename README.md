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

## Setup & Installation

### 1. Backend (Docker)
Ensure Docker Desktop is running. Clone the repository and navigate into the project directory:

```bash
git clone [https://github.com/YourUsername/Containerized-Local-RAG-API.git](https://github.com/YourUsername/Containerized-Local-RAG-API.git)
cd Containerized-Local-RAG-API
