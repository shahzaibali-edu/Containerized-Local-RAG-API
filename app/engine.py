import os
import json
import duckdb
import threading
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import DuckDB as DuckDBStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Paths are relative to the app's working directory (/code inside Docker,
# or the project root when running locally). Both map to the same host
# folder via the docker-compose volume mount: ./data:/code/data
VECTORSTORE_PATH = "data/vectorstore.duckdb"
MANIFEST_PATH = "data/manifest.json"


class RAGEngine:
    def __init__(self):
        # Embedding model — loaded once, reused for all PDFs
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Lock to serialise writes to the vector store
        self._lock = threading.Lock()

        # LLM is lazy-loaded on first query
        self.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        self._llm = None

        # ------------------------------------------------------------------
        # Persistence bootstrap
        # ------------------------------------------------------------------
        os.makedirs("data", exist_ok=True)

        db_exists   = os.path.exists(VECTORSTORE_PATH)
        manifest    = self._load_manifest()

        # Open a single persistent connection for the lifetime of this object.
        # duckdb.connect() creates the file if it doesn't exist.
        self._db_conn = duckdb.connect(VECTORSTORE_PATH)

        if db_exists and manifest:
            # Normal restart path — restore only successfully-ingested docs.
            # Any "processing" entry in the manifest means the container crashed
            # mid-ingestion; we discard those so they get cleanly re-embedded.
            self._ingested_docs: dict = {
                k: v for k, v in manifest.items() if v.get("status") == "ready"
            }
            if self._ingested_docs:
                # Re-attach to the existing table in the db file (no re-embedding)
                self.vectorstore = DuckDBStore(
                    connection=self._db_conn,
                    embedding=self.embeddings,
                )
                print(
                    f"✓ Restored vectorstore: {len(self._ingested_docs)} document(s) "
                    "loaded from disk — no re-embedding needed."
                )
            else:
                # DB file exists but no ready docs (e.g. manifest was cleared)
                self.vectorstore = None
                print("Vectorstore file found but no completed documents in manifest — starting fresh.")
        else:
            # First boot, or manifest was deleted alongside the DB
            if db_exists and not manifest:
                # DB orphaned without a manifest — safest to start clean so
                # the file doesn't silently hold stale/unknown embeddings.
                self._db_conn.close()
                os.remove(VECTORSTORE_PATH)
                self._db_conn = duckdb.connect(VECTORSTORE_PATH)
                print("Orphaned vectorstore found (no manifest). Removed and starting fresh.")

            self._ingested_docs = {}
            self.vectorstore = None

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        """Read the document manifest from disk. Returns {} on any error."""
        if os.path.exists(MANIFEST_PATH):
            try:
                with open(MANIFEST_PATH, "r") as f:
                    return json.load(f)
            except Exception as exc:
                print(f"Warning: could not read manifest ({exc}). Treating as empty.")
        return {}

    def _save_manifest(self):
        """Persist the current _ingested_docs dict to disk."""
        try:
            with open(MANIFEST_PATH, "w") as f:
                json.dump(self._ingested_docs, f, indent=2)
        except Exception as exc:
            print(f"Warning: could not save manifest ({exc}).")

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_pdf(self, file_path: str) -> dict:
        """
        Load, chunk, embed, and persist a PDF.
        Thread-safe. Returns immediately with status info.
        """
        filename = os.path.basename(file_path)

        # Skip files already successfully embedded
        existing = self._ingested_docs.get(filename)
        if existing and existing["status"] == "ready":
            return {"status": "duplicate", "filename": filename, "chunks": existing["chunks"]}

        # Mark as in-progress and persist immediately so a crash is visible
        self._ingested_docs[filename] = {"status": "processing", "chunks": 0}
        self._save_manifest()

        try:
            loader = PyPDFLoader(file_path)
            docs   = loader.load()
            splits = self.text_splitter.split_documents(docs)

            with self._lock:
                if self.vectorstore is None:
                    # First document — create table in the persistent DB file
                    self.vectorstore = DuckDBStore.from_documents(
                        splits, self.embeddings, connection=self._db_conn
                    )
                else:
                    # Subsequent documents — append to the existing table
                    self.vectorstore.add_documents(splits)

            self._ingested_docs[filename] = {"status": "ready", "chunks": len(splits)}
            self._save_manifest()
            return {"status": "success", "filename": filename, "chunks": len(splits)}

        except Exception as exc:
            self._ingested_docs[filename] = {"status": "error", "chunks": 0, "error": str(exc)}
            self._save_manifest()
            raise

    def list_documents(self) -> list:
        """Return all tracked documents with their ingestion status."""
        return [
            {"filename": k, "status": v["status"], "chunks": v["chunks"]}
            for k, v in self._ingested_docs.items()
        ]

    def has_documents(self) -> bool:
        """True once at least one PDF has been successfully embedded."""
        return self.vectorstore is not None

    # ------------------------------------------------------------------
    # LLM (lazy-loaded on first query)
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            self._llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "use_safetensors": True,   # bypasses the HF security block
                },
                pipeline_kwargs={
                    "max_new_tokens": 150,
                    "return_full_text": False,
                },
            )
        return self._llm

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def ask(self, query: str) -> str:
        """Run a RAG query against all ingested documents."""
        if not self.has_documents():
            return (
                "No documents have been ingested yet. "
                "Please upload a PDF using the Upload tab first."
            )

        prompt = ChatPromptTemplate.from_template(
            "<|im_start|>system\n"
            "You are a strict assistant. Answer based ONLY on the context below. "
            "If the answer is not in the context, reply exactly with: "
            "'I do not have info on that.' Do not guess.<|im_end|>\n"
            "<|im_start|>user\n"
            "Context: {context}\n\n"
            "Question: {input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        qa_chain = create_retrieval_chain(
            self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            create_stuff_documents_chain(self.llm, prompt),
        )
        result = qa_chain.invoke({"input": query})
        return result["answer"]