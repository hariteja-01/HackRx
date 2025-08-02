# main.py
import os
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from rag_processor import RAGProcessor

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
HACKATHON_BEARER_TOKEN = os.getenv("HACKATHON_API_KEY")

# --- Initialize the RAG Processor ---
# This ensures the model is loaded once when the application starts.
try:
    rag_processor = RAGProcessor(api_key=GEMINI_API_KEY)
except ValueError as e:
    print(f"FATAL ERROR: {e}. Please set GOOGLE_API_KEY in your .env file.")
    rag_processor = None # Set to None to handle startup failure

# --- API Definition ---
app = FastAPI(
    title="HackRx 6.0 Submission",
    description="LLM-Powered Intelligent Query–Retrieval System",
    version="1.0.0"
)

# --- Security ---
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Dependency to verify the bearer token.
    """
    if not HACKATHON_BEARER_TOKEN or credentials.credentials != HACKATHON_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

# --- Pydantic Models for Request and Response ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document to be processed.")
    questions: list[str] = Field(..., description="A list of questions to ask about the document.")

class HackRxResponse(BaseModel):
    answers: list[str] = Field(..., description="A list of answers corresponding to the questions.")


# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def run_submission(request: HackRxRequest):
    """
    This endpoint processes a document and answers questions based on its content.
    It performs real-time Retrieval-Augmented Generation (RAG).
    """
    if rag_processor is None:
        raise HTTPException(status_code=500, detail="Server not configured. Missing API keys.")
    
    try:
        # The core logic is delegated to our RAGProcessor class
        answers = rag_processor.process_and_answer(
            doc_url=request.documents,
            questions=request.questions
        )
        return HackRxResponse(answers=answers)
    except Exception as e:
        # Catch-all for any unexpected errors during processing
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "HackRx 6.0 API is running!"}