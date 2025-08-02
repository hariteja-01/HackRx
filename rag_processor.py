# rag_processor.py

import os
import requests
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.Youtubeing import load_qa_chain
from langchain.prompts import PromptTemplate

class RAGProcessor:
    def __init__(self, api_key: str):
        """
        Initializes the RAG processor, now only storing the API key.
        """
        if not api_key:
            raise ValueError("Google API Key is missing.")
        # Store the API key but don't use it yet
        self.api_key = api_key
        self.llm = None
        self.embeddings = None

    def _initialize_clients(self):
        """
        Initializes the Google AI clients on demand.
        This will be called when the first request comes in.
        """
        if self.llm is None or self.embeddings is None:
            print("Initializing Google AI clients for the first time...")
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            print("Google AI clients initialized.")

    def _get_pdf_text(self, pdf_url: str) -> str:
        # ... (this method remains the same)
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
        # ... (this method remains the same)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000
        )
        return text_splitter.split_text(text)

    def _get_vector_store(self, text_chunks: list):
        # ... (this method remains the same)
        if not text_chunks:
            return None
        try:
            # We need embeddings here, so ensure clients are initialized
            self._initialize_clients()
            vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def _get_conversational_chain(self):
        # ... (this method remains the same)
        # We need the LLM here, so ensure clients are initialized
        self._initialize_clients()
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
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain

    def process_and_answer(self, doc_url: str, questions: list[str]) -> list[str]:
        """
        Main method to process a document URL and answer a list of questions.
        """
        # Ensure clients are initialized before doing any work
        self._initialize_clients()

        print(f"Processing document: {doc_url}")
        raw_text = self._get_pdf_text(doc_url)
        if not raw_text:
            return ["Could not process the document from the provided URL."] * len(questions)

        text_chunks = self._get_text_chunks(raw_text)
        vector_store = self._get_vector_store(text_chunks)

        if not vector_store:
            return ["Failed to create a searchable index from the document."] * len(questions)

        chain = self._get_conversational_chain()
        answers = []

        for question in questions:
            try:
                docs = vector_store.similarity_search(question)
                response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                answers.append(response["output_text"])
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                answers.append("An error occurred while generating the answer.")
        
        return answers