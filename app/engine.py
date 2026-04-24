import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import DuckDB
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class RAGEngine:
    def __init__(self, document_path: str):
        loader = PyPDFLoader(document_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = DuckDB.from_documents(splits, embeddings, connection_string="duckdb.db")
        
        # We define the model but don't download it yet
        self.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                model_kwargs={
                    "low_cpu_mem_usage": True,
                    "use_safetensors": True  # <--- This bypasses the security block!
                },
                pipeline_kwargs={
                    "max_new_tokens": 150,
                    "return_full_text": False
                }
            )
        return self._llm

    def ask(self, query: str) -> str:
        # prompt = ChatPromptTemplate.from_template(
        #     "Based ONLY on the following context, answer the user's question briefly.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer: "
        # )
        # prompt = ChatPromptTemplate.from_template(
        #     "You are a strict assistant. Answer the user's question using ONLY the provided context.\n"
        #     "If the context does not contain the answer, you MUST reply exactly with: 'I do not have info on that.' Do not guess.\n\n"
        #     "Context: {context}\n\n"
        #     "Question: {input}\n\n"
        #     "Answer: "
        # )
        prompt = ChatPromptTemplate.from_template(
            "<|im_start|>system\n"
            "You are a strict assistant. Answer based ONLY on the context below. If the answer is not in the context, reply exactly with: 'I do not have info on that.' Do not guess.<|im_end|>\n"
            "<|im_start|>user\n"
            "Context: {context}\n\n"
            "Question: {input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        qa_chain = create_retrieval_chain(
            self.vectorstore.as_retriever(search_kwargs={"k": 3}), 
            create_stuff_documents_chain(self.llm, prompt)
        )
        result = qa_chain.invoke({"input": query})
        return result["answer"]