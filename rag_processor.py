# rag_processor.py

import os
import requests
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
# --- New Imports for the Modern LCEL Approach ---
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Google API Key is missing.")
        self.api_key = api_key
        self.llm = None
        self.embeddings = None

    def _initialize_clients(self):
        if self.llm is None or self.embeddings is None:
            print("Initializing Google AI clients for the first time...")
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            print("Google AI clients initialized.")

    def _get_pdf_text(self, pdf_url: str) -> str:
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except requests.RequestException as e:
            print(f"Error downloading PDF: {e}")
            return ""

    def _get_text_chunks(self, text: str) -> list:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )
        return text_splitter.split_text(text)

    def _get_vector_store(self, text_chunks: list):
        if not text_chunks:
            return None
        try:
            self._initialize_clients()
            vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def process_and_answer(self, doc_url: str, questions: list[str]) -> list[str]:
        # --- This is the new, completely rewritten logic ---
        self._initialize_clients()
        print(f"Processing document: {doc_url}")

        raw_text = self._get_pdf_text(doc_url)
        if not raw_text:
            return ["Could not process the document from the provided URL."] * len(questions)

        text_chunks = self._get_text_chunks(raw_text)
        vector_store = self._get_vector_store(text_chunks)
        if not vector_store:
            return ["Failed to create a searchable index from the document."] * len(questions)

        # Create a retriever to fetch relevant documents
        retriever = vector_store.as_retriever()

        # Define the prompt template
        prompt_template = """
        You are an expert at answering questions about policy documents.
        Answer the question as precisely and detailed as possible from the provided context.
        Ensure your answer is directly based on the text. If the answer is not in the
        provided context, say "The answer is not available in the provided document."
        Do not make up information.

        Context:
        {context}

        Question:
        {question}

        Precise Answer:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the new RAG chain using LCEL
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answers = []
        for question in questions:
            try:
                # Invoke the new chain for each question
                answer = rag_chain.invoke(question)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                answers.append("An error occurred while generating the answer.")
        
        return answers