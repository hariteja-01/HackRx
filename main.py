# main.py

import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field

# Ensure this import matches the filename of your processor
from rag_processor import RAGProcessor

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
HACKATHON_BEARER_TOKEN = os.getenv("HACKATHON_API_KEY")

# --- Initialize the RAG Processor ---
try:
    rag_processor = RAGProcessor(api_key=GEMINI_API_KEY)
except ValueError as e:
    print(f"FATAL ERROR: {e}. Please set GOOGLE_API_KEY in your .env file.")
    rag_processor = None

# --- API Definition ---
app = FastAPI(
    title="HackRx 6.0 Final Decision Engine",
    description="Optimized RAG system for claim adjudication using a 3-step reasoning pipeline.",
    version="FINAL"
)

# --- Security ---
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not HACKATHON_BEARER_TOKEN or credentials.credentials != HACKATHON_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

# --- Pydantic Models for a Structured Response ---
class Justification(BaseModel):
    clause: str = Field(..., description="The specific clause or text from the document.")
    reasoning: str = Field(..., description="How this clause supports the decision.")

class FinalResponse(BaseModel):
    decision: str = Field(..., description="The final decision, e.g., 'Approved', 'Rejected'.")
    amount: Optional[str] = Field(None, description="The approved or payable amount, if applicable.")
    justification: List[Justification] = Field(..., description="A list of clauses and reasons backing the decision.")

class FinalRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document to be processed.")
    questions: list[str] = Field(..., description="A list of natural language queries for adjudication.")

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=List[FinalResponse], dependencies=[Depends(verify_token)])
async def run_submission(request: FinalRequest):
    if rag_processor is None:
        raise HTTPException(status_code=500, detail="Server not configured. Missing API keys.")
    
    try:
        final_responses = await rag_processor.process_claim_queries(
            doc_url=request.documents,
            queries=request.questions
        )
        return final_responses
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/")
async def read_root():
    return {"status": "ok", "message": "HackRx Final Decision Engine API is running!"}