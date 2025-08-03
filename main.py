import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List
import asyncio
import time
from dotenv import load_dotenv

load_dotenv()

from rag_processor import OptimizedRAGProcessor, FinalResponse

# Configuration
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

processor = OptimizedRAGProcessor(google_api_key=GOOGLE_API_KEY)

app = FastAPI(
    title="HackRx Insurance Policy Analyzer",
    version="1.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.get("/")
async def root():
    return {"message": "HackRx API is running", "docs": "/docs"}

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")  # Add a favicon.ico file or remove this

@app.post("/hackrx/run", response_model=List[FinalResponse])
async def run_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    start_time = time.time()
    try:
        answers = await processor.process_queries(request.documents, request.questions)
        duration = time.time() - start_time
        print(f"Processed {len(request.questions)} queries in {duration:.2f}s")
        return answers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemini-1.5-flash"}