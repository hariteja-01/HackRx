import os
import google.generativeai as genai
from typing import Dict, Any, List
import logging
import json
from sentence_transformers import SentenceTransformer
from embeddings_manager import EmbeddingsManager

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        # Initialize Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize embeddings
        self.embeddings_manager = EmbeddingsManager()
    
    def answer_question(self, document_text: str, question: str) -> str:
        """Answer a question based on document text"""
        try:
            # Create index if not exists
            if not self.embeddings_manager.chunks:
                self.embeddings_manager.create_index([document_text])
            
            # Find relevant chunks
            results = self.embeddings_manager.search(question)
            context = "\n\n".join([res["text"] for res in results])
            
            # Generate prompt
            prompt = f"""You are an expert document analyst. Answer the question based on the provided context.
Be precise and only use information from the context. If you don't know, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error processing question: {str(e)}"
    
    def process_decision_query(self, query: str, document_text: str = "") -> Dict[str, Any]:
        """Process a natural language query and return structured decision"""
        try:
            # Create index if document provided
            if document_text and not self.embeddings_manager.chunks:
                self.embeddings_manager.create_index([document_text])
            
            # Get context if document exists
            context = ""
            if document_text:
                results = self.embeddings_manager.search(query)
                context = "\n\n".join([res["text"] for res in results])
            
            # Generate structured prompt
            prompt = f"""You are an insurance policy analyzer. Analyze the query and return a JSON response with:
- decision: "approved" or "rejected" or "needs_more_info"
- amount: if applicable, the payout amount
- justification: brief explanation of the decision
- clauses: list of policy clauses that support the decision

Query: {query}
{'Policy Context:' + context if context else 'No policy context provided'}

Return ONLY valid JSON in this format:
{{
    "decision": "approved|rejected|needs_more_info",
    "amount": null|number,
    "justification": "string",
    "clauses": ["string"]
}}"""
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            try:
                json_str = response.text.replace('```json', '').replace('```', '').strip()
                decision_data = json.loads(json_str)
                
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