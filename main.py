# main.py

import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import time

# Import the processor
from rag_processor import OptimizedRAGProcessor

# --- Configuration ---
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY", "7294b64376d390e0c8800d2f7dd32943cbe143a7eeb1f7787d878ffb3d329995")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize processor
processor = OptimizedRAGProcessor(openai_api_key=OPENAI_API_KEY)

app = FastAPI(
    title="HackRx 6.0 Final Submission Engine",
    description="High-performance document query system with structured decision output.",
    version="FINAL"
)

# --- Security & Middleware ---
# <<< FIX: Import the correct credentials model
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
security = HTTPBearer()

# <<< FIX: Use the correct type hint for the credentials dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# --- Pydantic Models ---
class Justification(BaseModel):
    clause: str = Field(..., description="The specific clause from the document.")
    reasoning: str = Field(..., description="How this clause supports the decision.")

class FinalResponse(BaseModel):
    decision: str = Field(..., description="The final decision, e.g., 'Approved', 'Rejected'.")
    amount: Optional[str] = Field(None, description="The approved amount, if applicable.")
    justification: List[Justification] = Field(..., description="List of clauses and reasons.")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Endpoints ---
@app.post("/hackrx/run", response_model=List[FinalResponse])
async def run_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint to process queries and return structured decisions."""
    start_time = time.time()
    try:
        answers = await asyncio.wait_for(
            processor.process_queries(request.documents, request.questions),
            timeout=28.0
        )
        duration = time.time() - start_time
        print(f"SUCCESS: Processed {len(request.questions)} queries in {duration:.2f}s")
        return answers
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timeout after 28 seconds.")
    except Exception as e:
        print(f"ERROR: An exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}