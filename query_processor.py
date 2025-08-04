import os
import google.generativeai as genai
from typing import Dict, Any, List
import logging
import re
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        # Initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize embeddings model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.document_chunks = []
    
    def generate_embeddings(self, text: str, chunk_size: int = 1000) -> None:
        """Generate embeddings for document text"""
        # Split text into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        self.document_chunks = chunks
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
    
    def find_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """Find most relevant document chunks for a query"""
        if not self.index:
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query])
        
        # Search the index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return relevant chunks
        return [self.document_chunks[i] for i in indices[0]]
    
    def answer_question(self, document_text: str, question: str) -> str:
        """Answer a question based on document text"""
        try:
            # Generate embeddings if not already done
            if not self.index:
                self.generate_embeddings(document_text)
            
            # Find relevant chunks
            relevant_chunks = self.find_relevant_chunks(question)
            context = "\n\n".join(relevant_chunks)
            
            # Generate prompt
            prompt = f"""
            You are an expert document analyst. Answer the question based on the provided context.
            Be precise and only use information from the context. If you don't know, say "I don't know".

            Context:
            {context}

            Question: {question}
            Answer:
            """
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error processing question: {str(e)}"
    
    def process_decision_query(self, query: str, document_text: str = "") -> Dict[str, Any]:
        """Process a natural language query and return structured decision"""
        try:
            # Generate embeddings if document text is provided
            if document_text and not self.index:
                self.generate_embeddings(document_text)
            
            # Find relevant chunks if document exists
            context = ""
            if document_text:
                relevant_chunks = self.find_relevant_chunks(query)
                context = "\n\n".join(relevant_chunks)
            
            # Generate structured prompt
            prompt = f"""
            You are an insurance policy analyzer. Analyze the query and return a JSON response with:
            - decision: "approved" or "rejected" or "needs_more_info"
            - amount: if applicable, the payout amount
            - justification: brief explanation of the decision
            - clauses: list of policy clauses that support the decision

            Query: {query}
            {f"Policy Context:\n{context}" if context else "No policy context provided"}
            
            Return ONLY valid JSON in this format:
            {{
                "decision": "approved|rejected|needs_more_info",
                "amount": null|number,
                "justification": "string",
                "clauses": ["string"]
            }}
            """
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            try:
                # Sometimes Gemini adds markdown syntax
                json_str = response.text.replace('```json', '').replace('```', '').strip()
                decision_data = json.loads(json_str)
                
                # Validate structure
                if not all(key in decision_data for key in ['decision', 'justification', 'clauses']):
                    raise ValueError("Invalid response structure")
                
                return decision_data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing Gemini response: {response.text}")
                return {
                    "decision": "needs_more_info",
                    "amount": None,
                    "justification": "Error processing policy information",
                    "clauses": []
                }
        except Exception as e:
            logger.error(f"Error processing decision query: {str(e)}")
            return {
                "decision": "needs_more_info",
                "amount": None,
                "justification": "Error processing your request",
                "clauses": []
            }