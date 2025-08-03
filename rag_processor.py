# rag_processor.py

import os
import re
import io
import asyncio
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np
from pydantic import BaseModel, Field
from fastapi import HTTPException
from PyPDF2 import PdfReader
from docx import Document
import faiss
from openai import AsyncOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

# --- Pydantic Models ---
class Justification(BaseModel):
    clause: str = Field(..., description="The specific, exact clause or text excerpt from the document that justifies the reasoning.")
    reasoning: str = Field(..., description="A step-by-step explanation of how this clause applies to the query to support the overall decision.")

class FinalResponse(BaseModel):
    decision: str = Field(..., description="The final decision, must be one of: 'Approved', 'Rejected', 'Partially Approved', or 'Needs More Information'.")
    amount: Optional[str] = Field(None, description="The approved or payable amount, if applicable. Can be a number or a description like 'As per plan limits'.")
    justification: List[Justification] = Field(..., description="A list of justification objects, each linking a document clause to the reasoning for the decision.")

class CustomEmbeddings(Embeddings):
    """Wrapper for OpenAI embeddings."""
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(input=texts, model="text-embedding-3-small")
        return [e.embedding for e in response.data]

    async def aembed_query(self, text: str) -> List[float]:
        return (await self.aembed_documents([text]))[0]

class OptimizedRAGProcessor:
    def __init__(self, openai_api_key: str):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.embeddings = CustomEmbeddings(self.client)
        self.vector_store_cache = {}
        self.clean_pattern = re.compile(r'\s+')

    async def _download_and_extract_text(self, url: str) -> str:
        """Download and extract text from PDF/DOCX."""
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                response = await loop.run_in_executor(pool, lambda: requests.get(url, timeout=15))
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                with io.BytesIO(response.content) as f:
                    reader = PdfReader(f)
                    return "".join(page.extract_text() or "" for page in reader.pages)
            elif 'word' in content_type or url.lower().endswith(('.docx', '.doc')):
                with io.BytesIO(response.content) as f:
                    doc = Document(f)
                    return "\n".join(para.text for para in doc.paragraphs)
            return response.text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Document download/extraction failed: {str(e)}")

    async def _get_vector_store(self, doc_url: str):
        if doc_url in self.vector_store_cache:
            return self.vector_store_cache[doc_url]
            
        text = await self._download_and_extract_text(doc_url)
        clean_text = self.clean_pattern.sub(' ', text).strip()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(clean_text)
        
        embedded_chunks = await self.embeddings.aembed_documents(chunks)
        
        index = faiss.IndexFlatL2(len(embedded_chunks[0]))
        index.add(np.array(embedded_chunks, dtype=np.float32))
        
        vector_store = {"index": index, "chunks": chunks}
        self.vector_store_cache[doc_url] = vector_store
        return vector_store

    async def _retrieve_chunks(self, query: str, vs_data, top_k: int = 7):
        query_embedding = np.array([await self.embeddings.aembed_query(query)], dtype=np.float32)
        distances, indices = vs_data["index"].search(query_embedding, top_k)
        return [vs_data["chunks"][i] for i in indices[0]]

    async def _generate_structured_answer(self, question: str, context: str) -> FinalResponse:
        """Generates the final structured JSON response."""
        system_prompt = """You are an expert insurance claim adjudicator. Your task is to make a final decision based on the user's claim details and the provided policy clauses.
        Follow these steps strictly:
        1. Analyze the 'Claim Details'.
        2. Scrutinize all 'Policy Clauses' to find applicable rules, waiting periods, and exclusions.
        3. For each point of your reasoning, you MUST cite the specific clause that supports it in the 'justification' list.
        4. Formulate a final decision ('Approved', 'Rejected', 'Partially Approved', or 'Needs More Information').
        5. Provide your final answer ONLY in the specified JSON format."""

        user_prompt = f"Claim Details: {question}\n\nPolicy Clauses:\n{context}"
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            response_json = response.choices[0].message.content
            # Validate the JSON against our Pydantic model
            return FinalResponse.parse_raw(response_json)
        except Exception as e:
            print(f"Error during structured answer generation: {str(e)}")
            # Fallback for parsing or other errors
            return FinalResponse(
                decision="Error",
                justification=[Justification(clause="Generation Error", reasoning=f"Failed to generate a valid structured response: {str(e)}")]
            )

    async def _process_single_query(self, question: str, vs_data) -> Dict:
        relevant_chunks = await self._retrieve_chunks(question, vs_data)
        context = "\n\n---\n\n".join(relevant_chunks)
        structured_response = await self._generate_structured_answer(question, context)
        return structured_response.dict()

    async def process_queries(self, doc_url: str, questions: List[str]) -> List[Dict]:
        """Main processing pipeline optimized for speed and accuracy."""
        vector_store_data = await self._get_vector_store(doc_url)
        
        # Process questions in parallel
        tasks = [self._process_single_query(q, vector_store_data) for q in questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle cases where a task might have failed
        final_answers = []
        for i, ans in enumerate(answers):
            if isinstance(ans, Exception):
                final_answers.append(
                    FinalResponse(
                        decision="Error",
                        justification=[Justification(clause="Task Execution Error", reasoning=f"Error processing query '{questions[i]}': {str(ans)}")]
                    ).dict()
                )
            else:
                final_answers.append(ans)
        return final_answers