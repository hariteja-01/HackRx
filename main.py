import os
from fastapi import FastAPI, HTTPException, Depends, Request
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
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY", "7294b64376d390e0c8800d2f7dd32943cbe143a7eeb1f7787d878ffb3d329995")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GENERATION_TIMEOUT = 30  # seconds

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
    version="4.0",
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

@app.get("/")
async def root():
    return {"message": "HackRx API is running", "docs": "/docs"}

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
        print(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """System health check endpoint"""
    try:
        # Test a simple Gemini request to verify API connectivity
        test_prompt = "Hello, please respond with 'OK'"
        test_response = await processor.llm.ainvoke(test_prompt)
        return {
            "status": "healthy",
            "model": "gemini-1.5-flash",
            "api_connection": True,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "success": False,
            "path": request.url.path,
            "timestamp": time.time()
        }
    )