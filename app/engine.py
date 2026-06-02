import os
import threading
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import DuckDB
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class RAGEngine:
    def __init__(self):
        # Load the embedding model once; reused for all PDFs
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Vector store starts empty; built incrementally as PDFs are ingested
        self.vectorstore = None

        # Tracks every ingested document: {filename: {"status": str, "chunks": int}}
        self._ingested_docs: dict = {}

        # Lock to prevent concurrent writes to the vector store
        self._lock = threading.Lock()

        # LLM is lazy-loaded on first query to keep startup fast
        self.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        self._llm = None

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_pdf(self, file_path: str) -> dict:
        """
        Load, split, and embed a PDF into the vector store.
        Thread-safe: multiple uploads can queue up without corrupting state.
        Returns a dict with status, filename, and chunk count.
        """
        filename = os.path.basename(file_path)

        # Guard: skip if already successfully ingested
        existing = self._ingested_docs.get(filename)
        if existing and existing["status"] == "ready":
            return {"status": "duplicate", "filename": filename, "chunks": existing["chunks"]}

        # Mark as in-progress before any heavy work
        self._ingested_docs[filename] = {"status": "processing", "chunks": 0}

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splits = self.text_splitter.split_documents(docs)

            with self._lock:
                if self.vectorstore is None:
                    # First PDF — create the vector store from scratch
                    self.vectorstore = DuckDB.from_documents(splits, self.embeddings)
                else:
                    # Subsequent PDFs — add to the existing store
                    self.vectorstore.add_documents(splits)

            self._ingested_docs[filename] = {"status": "ready", "chunks": len(splits)}
            return {"status": "success", "filename": filename, "chunks": len(splits)}

        except Exception as e:
            self._ingested_docs[filename] = {"status": "error", "chunks": 0, "error": str(e)}
            raise

    def list_documents(self) -> list:
        """Return a list of all tracked documents with their ingestion status."""
        return [
            {"filename": k, "status": v["status"], "chunks": v["chunks"]}
            for k, v in self._ingested_docs.items()
        ]

    def has_documents(self) -> bool:
        """True if at least one PDF has been successfully embedded."""
        return self.vectorstore is not None

    # ------------------------------------------------------------------
    # LLM (lazy-loaded)
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            self._llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "use_safetensors": True,  # bypasses the HF security block
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