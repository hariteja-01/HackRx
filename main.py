import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import asyncio
import time
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from rag_processor import OptimizedRAGProcessor, FinalResponse

# Configuration
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is required. "
        "Please create a .env file with your Google Gemini API key."
    )

# Initialize processor
processor = OptimizedRAGProcessor(google_api_key=GOOGLE_API_KEY)

app = FastAPI(
    title="HackRx Insurance Policy Analyzer",
    description="API for analyzing insurance policies using Google Gemini",
    version="3.0",
    docs_url="/docs",
    redoc_url=None
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# Endpoints
@app.post("/hackrx/run", response_model=List[FinalResponse])
async def run_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for policy analysis queries"""
    start_time = time.time()
    try:
        # Process with timeout
        answers = await asyncio.wait_for(
            processor.process_queries(request.documents, request.questions),
            timeout=GENERATION_TIMEOUT
        )
        duration = time.time() - start_time
        print(f"Processed {len(request.questions)} queries in {duration:.2f}s")
        return answers
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Processing timeout after {GENERATION_TIMEOUT} seconds"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """System health check endpoint"""
    return {
        "status": "healthy",
        "model": LLM_MODEL,
        "ready": True,
        "timestamp": time.time()
    }

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "success": False,
            "timestamp": time.time()
        }
    )