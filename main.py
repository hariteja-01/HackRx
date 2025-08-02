# main.py

import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
# Make sure this matches your filename
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
    title="HackRx 6.0 High-Performance Submission",
    description="Optimized LLM Query–Retrieval System with Caching and Async Processing.",
    version="2.0.0"
)

# --- Security ---
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not HACKATHON_BEARER_TOKEN or credentials.credentials != HACKATHON_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document to be processed.")
    questions: list[str] = Field(..., description="A list of questions to ask about the document.")

class HackRxResponse(BaseModel):
    answers: list[str] = Field(..., description="A list of answers corresponding to the questions.")

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: HackRxRequest):
    if rag_processor is None:
        raise HTTPException(status_code=500, detail="Server not configured. Missing API keys.")
    
    try:
        answers = await rag_processor.process_and_answer(
            doc_url=request.documents,
            questions=request.questions
        )
        return HackRxResponse(answers=answers)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- CORRECTED HEALTH CHECK ENDPOINT ---
@app.get("/")
async def read_root():
    return {"status": "ok", "message": "HackRx 6.0 High-Performance API is running!"}