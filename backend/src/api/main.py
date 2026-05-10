from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv
load_dotenv()

import uvicorn
import logging

from rag.agent import ClinicalAgent

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedGuardAI API",
    description="Agentic RAG Assistant for Safety-First Medical Advice",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eagerly initialize the agent at module import. This ensures the agent is
# available regardless of how the app is mounted (uvicorn, sync TestClient,
# lifespan context manager).
agent = None
try:
    agent = ClinicalAgent()
    logger.info("ClinicalAgent initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ClinicalAgent: {e}")

class PatientContext(BaseModel):
    age: int = Field(..., description="Patient age in years")
    weight: float = Field(..., description="Patient weight in kg")
    allergies: List[str] = Field(default_factory=list, description="Known drug allergies")
    conditions: List[str] = Field(default_factory=list, description="Pre-existing medical conditions")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's medical or dosage query")
    patient_context: PatientContext

class QueryResponse(BaseModel):
    answer: str
    status: str

@app.post("/api/v1/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    """
    Main endpoint for the Agentic RAG system.
    Expects a query and a strictly verified patient context payload.
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent system not initialized.")

    try:
        logger.info(f"Processing query: {request.query}")
        
        # Guardrail check
        if agent.check_for_emergency(request.query):
            return QueryResponse(
                answer="[EMERGENCY GUARDRAIL TRIPPED] Please seek emergency medical assistance immediately. Do not wait.",
                status="emergency"
            )
            
        # The agent dynamically plans retrieval and answers based strictly on FDA context
        response_text = agent.process_query(
            user_query=request.query,
            patient_context=request.patient_context.dict()
        )
        
        return QueryResponse(
            answer=response_text,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error processing context: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during Agent execution.")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "components": {"rag_db": "connected" if agent and agent.retriever else "failed"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
