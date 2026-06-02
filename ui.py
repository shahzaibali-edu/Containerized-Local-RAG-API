import gradio as gr
import requests
import os

API_BASE = "http://127.0.0.1:8000/api/v1"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def send_message(message, history):
    """Send a chat message to the RAG API and append the reply to history."""
    if not message.strip():
        return "", history

    try:
        response = requests.post(
            f"{API_BASE}/chat",
            json={"query": message},
            timeout=120,  # LLM inference can be slow on CPU
        )
        data = response.json()
        reply = data.get("response", "No response received from API.")
    except requests.exceptions.ConnectionError:
        reply = "⚠️ Cannot connect to the API. Is Docker running?"
    except requests.exceptions.Timeout:
        reply = "⏱️ Request timed out. The model may still be loading — try again in a moment."
    except Exception as e:
        reply = f"⚠️ Unexpected error: {str(e)}"

    # Gradio 6 uses OpenAI-style role dicts, not tuples
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history


def upload_pdf(file):
    """Upload a PDF file to the API for background ingestion."""
    if file is None:
        return "⚠️ Please select a PDF file first."

    # Gradio may pass a file path string or an object with a .name attribute
    filepath = file if isinstance(file, str) else file.name
    filename = os.path.basename(filepath)

    try:
        with open(filepath, "rb") as f:
            response = requests.post(
                f"{API_BASE}/upload",
                files={"file": (filename, f, "application/pdf")},
                timeout=30,
            )
        data = response.json()

        if response.status_code == 200:
            return f"✅ {data['message']}"
        elif response.status_code == 409:
            return f"ℹ️ {data.get('detail', 'Already uploaded.')}"
        elif response.status_code == 413:
            return f"❌ File too large: {data.get('detail', '')}"
        elif response.status_code == 400:
            return f"❌ Invalid file: {data.get('detail', '')}"
        else:
            return f"❌ Error {response.status_code}: {data.get('detail', 'Unknown error')}"

    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to the API. Is Docker running?"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def refresh_documents():
    """Fetch and format the list of ingested documents from the API."""
    try:
        response = requests.get(f"{API_BASE}/documents", timeout=10)
        data = response.json()
        docs = data.get("documents", [])

        if not docs:
            return "📭 No documents ingested yet. Upload a PDF above to get started."

        lines = [
            "| Document | Status | Chunks |",
            "|----------|--------|--------|",
        ]
        for doc in docs:
            icon = {"ready": "✅", "processing": "⏳", "error": "❌"}.get(doc["status"], "❓")
            label = doc["status"].capitalize()
            lines.append(f"| {doc['filename']} | {icon} {label} | {doc['chunks']} |")

        return "\n".join(lines)

    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to API. Is Docker running?"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Local RAG Pipeline") as demo:

    gr.Markdown(
        """
        # 🤖 Local RAG Pipeline
        Ask questions about your uploaded PDF documents — running fully locally via Docker & SmolLM2.
        """
    )

    with gr.Tabs():

        # ── Tab 1: Chat ────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=440,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about your documents…",
                    show_label=False,
                    scale=5,
                )
                send_btn = gr.Button("Send →", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

            send_btn.click(send_message, [msg_input, chatbot], [msg_input, chatbot])
            msg_input.submit(send_message, [msg_input, chatbot], [msg_input, chatbot])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

        # ── Tab 2: Upload PDFs ─────────────────────────────────────────────
        with gr.Tab("📄 Upload PDFs"):
            with gr.Row():

                # Left column — upload panel
                with gr.Column(scale=1):
                    gr.Markdown("### ⬆️ Upload a New PDF")
                    gr.Markdown(
                        "Drag & drop or click to select a PDF (max 50 MB). "
                        "Embedding runs in the background — the chat tab stays usable."
                    )
                    file_input = gr.File(label="Select PDF", file_types=[".pdf"])
                    upload_btn = gr.Button("Upload & Start Ingestion", variant="primary")
                    upload_status = gr.Markdown("")

                # Right column — document status
                with gr.Column(scale=1):
                    gr.Markdown("### 📚 Ingested Documents")
                    gr.Markdown(
                        "Click **Refresh** to check the latest ingestion status. "
                        "`⏳ Processing` means embedding is still running."
                    )
                    docs_display = gr.Markdown("*Click Refresh to load document list.*")
                    refresh_btn = gr.Button("🔄 Refresh Status")

            upload_btn.click(upload_pdf, inputs=[file_input], outputs=[upload_status])
            refresh_btn.click(refresh_documents, outputs=[docs_display])


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
    )