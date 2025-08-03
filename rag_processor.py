import os
import re
import io
import asyncio
import json
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from fastapi import HTTPException
from PyPDF2 import PdfReader
try:
    from docx import Document
except ImportError:
    Document = None
import faiss
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
MAX_RETRIEVAL_CHUNKS = 5
MAX_CONTEXT_LENGTH = 8000  # Reduced to prevent token overflow
GENERATION_TIMEOUT = 25
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash"

class Justification(BaseModel):
    clause: str = Field(..., description="Exact text from policy document")
    reasoning: str = Field(..., description="How this clause supports the decision")

class FinalResponse(BaseModel):
    decision: str = Field(..., description="Approved/Rejected/Partially Approved/Needs More Information")
    amount: Optional[str] = Field(None)
    justification: List[Justification] = Field(...)

class OptimizedRAGProcessor:
    def __init__(self, google_api_key: str):
        genai.configure(api_key=google_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.0,
            convert_system_message_to_human=True
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store_cache = {}
        self.clean_pattern = re.compile(r'\s+')
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def _download_and_extract_text(self, url: str) -> str:
        """Handles PDF, DOCX and plain text documents with robust error handling"""
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                response = await loop.run_in_executor(pool, lambda: requests.get(url, timeout=15))
            response.raise_for_status()

            if url.lower().endswith('.pdf'):
                with io.BytesIO(response.content) as f:
                    reader = PdfReader(f)
                    return "".join(page.extract_text() or "" for page in reader.pages)
            elif Document and url.lower().endswith(('.docx', '.doc')):
                with io.BytesIO(response.content) as f:
                    doc = Document(f)
                    return "\n".join(para.text for para in doc.paragraphs)
            return response.text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Document processing failed: {str(e)}")

    async def _get_vector_store(self, doc_url: str):
        """Creates or retrieves cached FAISS vector store with optimized chunking"""
        if doc_url in self.vector_store_cache:
            return self.vector_store_cache[doc_url]
            
        text = await self._download_and_extract_text(doc_url)
        clean_text = self.clean_pattern.sub(' ', text).strip()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=['\n\n', '\n', '•', ';', 'Clause', 'Section']
        )
        chunks = splitter.split_text(clean_text)
        
        embedded_chunks = await self.embeddings.aembed_documents(chunks)
        
        index = faiss.IndexFlatIP(len(embedded_chunks[0]))
        index.add(np.array(embedded_chunks, dtype=np.float32))
        
        vector_store = {"index": index, "chunks": chunks}
        self.vector_store_cache[doc_url] = vector_store
        return vector_store

    async def _retrieve_chunks(self, query: str, vs_data, top_k: int = MAX_RETRIEVAL_CHUNKS):
        """Retrieves most relevant chunks using semantic search"""
        query_embedding = np.array([await self.embeddings.aembed_query(query)], dtype=np.float32)
        distances, indices = vs_data["index"].search(query_embedding, top_k)
        return [vs_data["chunks"][i] for i in indices[0]]

    def _extract_json_from_response(self, response_content: str) -> dict:
        """Robust JSON extraction with multiple fallback methods"""
        try:
            # First try direct JSON parsing
            return json.loads(response_content)
        except json.JSONDecodeError:
            try:
                # Try extracting from markdown code block
                json_match = re.search(r'```json\n({.*?})\n```', response_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                
                # Try finding first/last braces
                start = response_content.find('{')
                end = response_content.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response_content[start:end])
                
                raise ValueError("No valid JSON found in response")
            except Exception as e:
                raise ValueError(f"Could not extract JSON: {str(e)}")

    async def _generate_structured_answer(self, question: str, context: str) -> FinalResponse:
        """Generates validated JSON response with strict formatting"""
        prompt = f"""**Insurance Policy Analysis Task**

Analyze this insurance policy question and return ONLY a JSON response.

**Policy Context:**
{context[:MAX_CONTEXT_LENGTH]}

**Question:**
{question}

**Response Requirements:**
1. Decision must be one of: "Approved", "Rejected", "Partially Approved", or "Needs More Information"
2. Include amount if applicable
3. Provide at least one justification with exact clause reference

**Return ONLY this JSON structure:**
```json
{{
  "decision": "decision_here",
  "amount": "amount_or_null",
  "justification": [
    {{
      "clause": "exact_policy_text",
      "reasoning": "how_this_applies"
    }}
  ]
}}
```"""

        try:
            response = await self.llm.ainvoke(prompt)
            response_json = self._extract_json_from_response(response.content)
            validated_response = FinalResponse.parse_obj(response_json)
            
            # Verify at least one justification exists
            if not validated_response.justification:
                raise ValueError("Response must include at least one justification")
                
            return validated_response
            
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(f"Response validation error: {str(e)}")
            return FinalResponse(
                decision="Error",
                justification=[Justification(
                    clause="Response Validation",
                    reasoning=f"Invalid response format: {str(e)}"
                )]
            )
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return FinalResponse(
                decision="Error",
                justification=[Justification(
                    clause="Generation Error",
                    reasoning=f"Failed to generate response: {str(e)}"
                )]
            )

    async def process_queries(self, doc_url: str, questions: List[str]) -> List[Dict]:
        """Main processing pipeline with comprehensive error handling"""
        try:
            vector_store = await self._get_vector_store(doc_url)
            answers = []
            
            for question in questions:
                try:
                    relevant_chunks = await self._retrieve_chunks(question, vector_store)
                    context = "\n\n---\n\n".join(relevant_chunks)
                    structured_response = await self._generate_structured_answer(question, context)
                    answers.append(structured_response.dict())
                except Exception as e:
                    answers.append(FinalResponse(
                        decision="Error",
                        justification=[Justification(
                            clause="Processing Error",
                            reasoning=f"Failed to process query: {str(e)}"
                        )]
                    ).dict())
            
            return answers
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))