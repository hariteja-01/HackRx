import os
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import time
from document_processor import DocumentProcessor
from query_processor import QueryProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Document Processing System",
    description="System for processing natural language queries against documents",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class DecisionRequest(BaseModel):
    query: str
    document_url: Optional[str] = None

class DecisionResponse(BaseModel):
    decision: str
    amount: Optional[float] = None
    justification: str
    clauses: List[str]

# Initialize processors
document_processor = DocumentProcessor()
query_processor = QueryProcessor()

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    authorization: str = Header(None)
):
    """
    Process multiple questions against a document
    """
    start_time = time.time()
    
    # Validate authorization
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    try:
        # Process each question
        answers = []
        for question in request.questions:
            try:
                # Process document and question
                document_text = document_processor.process_document(request.documents)
                answer = query_processor.answer_question(document_text, question)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        # Log performance
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(request.questions)} questions in {processing_time:.2f} seconds")
        
        return {"answers": answers}
    
    except Exception as e:
        logger.error(f"Error in processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/process_decision", response_model=DecisionResponse)
async def process_decision(
    request: DecisionRequest,
    authorization: str = Header(None)
):
    """
    Process a natural language query and return a structured decision
    """
    start_time = time.time()
    
    try:
        # Process document if provided
        document_text = ""
        if request.document_url:
            document_text = document_processor.process_document(request.document_url)
        
        # Process query and get decision
        decision_data = query_processor.process_decision_query(request.query, document_text)
        
        # Log performance
        processing_time = time.time() - start_time
        logger.info(f"Processed decision query in {processing_time:.2f} seconds")
        
        return decision_data
    
    except Exception as e:
        logger.error(f"Error in decision processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)