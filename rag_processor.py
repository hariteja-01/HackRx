# rag_processor_optimized.py

import os
import requests
import io
import asyncio
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Google API Key is missing.")
        self.api_key = api_key
        self.llm = None
        self.embeddings = None
        # --- Caching Mechanism ---
        # Stores processed vector stores in memory to avoid reprocessing the same document
        self.vector_store_cache = {}

    async def _initialize_clients(self):
        # Using async initialization if ever needed, for now, it's synchronous
        if self.llm is None or self.embeddings is None:
            print("Initializing Google AI clients for the first time...")
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, request_timeout=120)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            print("Google AI clients initialized.")

    async def _get_vector_store_from_url(self, doc_url: str):
        # --- Cache Check ---
        if doc_url in self.vector_store_cache:
            print(f"CACHE HIT: Loading vector store for {doc_url} from cache.")
            return self.vector_store_cache[doc_url]

        print(f"CACHE MISS: Processing new document from {doc_url}.")
        
        # 1. Download PDF
        try:
            response = requests.get(doc_url, timeout=30)
            response.raise_for_status()
            pdf_file = io.BytesIO(response.content)
        except requests.RequestException as e:
            print(f"Error downloading PDF: {e}")
            return None

        # 2. Extract Text
        pdf_reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        if not text:
            return None

        # 3. Chunk Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        text_chunks = text_splitter.split_text(text)

        # 4. Create and Cache Vector Store
        try:
            await self._initialize_clients()
            vector_store = await asyncio.to_thread(FAISS.from_texts, text_chunks, self.embeddings)
            self.vector_store_cache[doc_url] = vector_store
            print(f"SUCCESS: Document processed and cached.")
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    async def process_and_answer(self, doc_url: str, questions: list[str]) -> list[str]:
        await self._initialize_clients()
        
        vector_store = await self._get_vector_store_from_url(doc_url)
        if not vector_store:
            return ["Could not process the document from the provided URL."] * len(questions)

        # --- Advanced Retrieval (MMR) ---
        # Fetches more documents initially (fetch_k=20) then selects the most relevant
        # and diverse ones (k=8) to send to the LLM. This drastically improves context quality.
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 8, 'fetch_k': 20}
        )

        # --- Highly Constrained Prompt ---
        prompt_template = """
        You are a highly specialized Question-Answering bot. Your task is to answer the user's question with extreme precision,
        based ONLY on the context provided below. Do not use any external knowledge. If the answer is not
        explicitly stated in the context, you MUST reply with the exact phrase: "Information not available in the document."
        Do not add any explanations or apologies.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        PRECISE ANSWER:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # --- Asynchronous Processing ---
        # Run all question-answering tasks concurrently instead of in a loop
        async def get_answer(question):
            try:
                return await rag_chain.ainvoke(question)
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                return "An error occurred during answer generation."

        tasks = [get_answer(q) for q in questions]
        answers = await asyncio.gather(*tasks)
        
        return answers