# rag_processor.py

import os
import requests
import io
import asyncio
from PyPDF2 import PdfReader
from typing import List, Dict, Any, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# --- Pydantic Models for internal processing ---
class StructuredQuery(BaseModel):
    age: Optional[int] = Field(None, description="Age of the person")
    gender: Optional[str] = Field(None, description="Gender of the person")
    procedure: Optional[str] = Field(None, description="The medical procedure or claim type")
    location: Optional[str] = Field(None, description="Location of the event")
    policy_duration_months: Optional[int] = Field(None, description="How many months the policy has been active")
    other_details: Optional[str] = Field(None, description="Any other relevant details from the query")

class Justification(BaseModel):
    clause: str = Field(..., description="The specific, exact clause or text excerpt from the document that justifies the reasoning.")
    reasoning: str = Field(..., description="A step-by-step explanation of how this clause applies to the query to support the overall decision.")

class FinalResponse(BaseModel):
    decision: str = Field(..., description="The final decision, must be one of: 'Approved', 'Rejected', 'Partially Approved', or 'Needs More Information'.")
    amount: Optional[str] = Field(None, description="The approved or payable amount, if applicable. Can be a number or a description like 'As per plan limits'.")
    justification: List[Justification] = Field(..., description="A list of justification objects, each linking a document clause to the reasoning for the decision.")

class RAGProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Google API Key is missing.")
        self.api_key = api_key
        self.llm = None
        self.embeddings = None
        self.vector_store_cache = {}

    async def _initialize_clients(self):
        if self.llm is None:
            print("Initializing Google AI clients...")
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.0, request_timeout=300)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            print("Google AI clients initialized.")

    async def _get_vector_store(self, doc_url: str):
        if doc_url in self.vector_store_cache:
            print(f"CACHE HIT: Using cached vector store for {doc_url}.")
            return self.vector_store_cache[doc_url]
        
        print(f"CACHE MISS: Processing new document: {doc_url}")
        try:
            response = requests.get(doc_url, timeout=30)
            response.raise_for_status()
            with io.BytesIO(response.content) as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            
            await self._initialize_clients()
            vector_store = await asyncio.to_thread(FAISS.from_texts, chunks, self.embeddings)
            self.vector_store_cache[doc_url] = vector_store
            return vector_store
        except Exception as e:
            print(f"Failed to process PDF from {doc_url}: {e}")
            return None

    async def _adjudicate_query(self, query: str, vector_store) -> Dict[str, Any]:
        await self._initialize_clients()

        # Step 1: Deconstruct the query into a structured format.
        parser_query = JsonOutputParser(pydantic_object=StructuredQuery)
        prompt_query = PromptTemplate(
            template="Parse the user's query to extract key details.\n{format_instructions}\nQuery: {query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser_query.get_format_instructions()},
        )
        chain_query = prompt_query | self.llm | parser_query
        structured_query = await chain_query.ainvoke({"query": query})

        # Step 2: Generate targeted search queries for intelligent retrieval.
        search_query_prompt = PromptTemplate.from_template(
            "Based on these claim details, generate 5 specific, distinct search queries for an insurance policy document. Focus on waiting periods, specific procedure coverage, exclusions, and policy conditions.\n\nDetails:\n{details}"
        )
        search_query_chain = search_query_prompt | self.llm | StrOutputParser()
        search_queries_str = await search_query_chain.ainvoke({"details": str(structured_query)})
        search_queries = [q.strip() for q in search_queries_str.split('\n') if q.strip()]
        search_queries.append(query)

        # Retrieve documents using the advanced MMR technique.
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 25})
        tasks = [retriever.ainvoke(q) for q in search_queries]
        retrieved_docs_lists = await asyncio.gather(*tasks)
        
        unique_docs = {doc.page_content: doc for sublist in retrieved_docs_lists for doc in sublist}
        context = "\n\n---\n\n".join([doc.page_content for doc in unique_docs.values()])

        # Step 3: Perform the final reasoning and decision making.
        parser_decision = JsonOutputParser(pydantic_object=FinalResponse)
        prompt_decision = PromptTemplate(
            template="You are an expert insurance claim adjudicator. Make a final decision based on the user's claim details and the provided policy clauses. Follow these steps:\n1. Analyze the 'Claim Details'.\n2. Scrutinize all 'Policy Clauses' to find applicable rules.\n3. For each point of your reasoning, you MUST cite the specific clause that supports it.\n4. Formulate a final decision and provide a structured JSON output as instructed.\n\n{format_instructions}\n\nClaim Details:\n{details}\n\nPolicy Clauses:\n{context}\n\nFinal Decision JSON:",
            input_variables=["details", "context"],
            partial_variables={"format_instructions": parser_decision.get_format_instructions()},
        )
        chain_decision = prompt_decision | self.llm | parser_decision
        final_decision = await chain_decision.ainvoke({"details": str(structured_query), "context": context})

        return final_decision

    async def process_claim_queries(self, doc_url: str, queries: list[str]) -> List[Dict[str, Any]]:
        vector_store = await self._get_vector_store(doc_url)
        if not vector_store:
            error_response = FinalResponse(
                decision="Error",
                justification=[Justification(clause="System Error", reasoning="Could not process the policy document from the provided URL.")]
            ).dict()
            return [error_response] * len(queries)
            
        # Process all queries concurrently for maximum speed.
        tasks = [self._adjudicate_query(q, vector_store) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Gracefully handle any errors during the processing of a single query.
        final_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                print(f"An exception occurred while adjudicating query '{queries[i]}': {res}")
                final_results.append(FinalResponse(
                    decision="Error",
                    justification=[Justification(clause="Processing Error", reasoning=f"An internal error occurred: {str(res)}")]
                ).dict())
            else:
                final_results.append(res)
        
        return final_results