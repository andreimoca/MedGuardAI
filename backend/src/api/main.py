from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import os
import json
import threading
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import uvicorn
import logging

from rag.agent import ClinicalAgent

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Where human feedback (thumbs up/down + corrections) is appended. This is the
# raw material for the RLHF/DPO preference dataset (see training/build_dpo_dataset.py).
FEEDBACK_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "feedback")
)
FEEDBACK_PATH = os.path.join(FEEDBACK_DIR, "feedback.jsonl")
FEEDBACK_STUB_PATH = os.path.join(FEEDBACK_DIR, "feedback_dpo_stub.jsonl")
_feedback_lock = threading.Lock()

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


# ---------------------------------------------------------------------------
# Human-feedback loop (RLHF data collection)
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    query: str = Field(..., description="The user query the answer was generated for")
    patient_context: Optional[PatientContext] = Field(default=None)
    answer: str = Field(..., description="The assistant answer being rated")
    rating: Literal["up", "down"] = Field(..., description="Thumbs up or down")
    correction: Optional[str] = Field(
        default=None, description="User-suggested better answer (thumbs-down only)"
    )
    status: Optional[str] = Field(
        default=None, description="The status returned with the answer ('success'/'emergency')"
    )


class FeedbackResponse(BaseModel):
    ok: bool
    stored: int = Field(..., description="Running count of feedback rows on disk")


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


@app.post("/api/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """Append a human feedback event to data/feedback/feedback.jsonl."""
    row = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "query": req.query,
        "patient_context": req.patient_context.dict() if req.patient_context else None,
        "answer": req.answer,
        "rating": req.rating,
        "correction": (req.correction or None),
        "status": req.status,
    }
    with _feedback_lock:
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        with open(FEEDBACK_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n = _count_lines(FEEDBACK_PATH)
    logger.info(f"feedback stored: rating={req.rating} has_correction={bool(req.correction)} total={n}")
    return FeedbackResponse(ok=True, stored=n)


@app.get("/api/v1/feedback/export")
async def export_feedback_as_dpo():
    """Convert collected thumbs-down feedback into DPO-pair stubs.

    Each item: {prompt, patient_context, rejected, chosen|null, meta}. Items
    with a user correction have `chosen` filled; others are left null so
    training/build_dpo_dataset.py can fill them by regenerate+judge. Also
    writes data/feedback/feedback_dpo_stub.jsonl.
    """
    if not os.path.exists(FEEDBACK_PATH):
        return {"count": 0, "items": []}
    items = []
    with _feedback_lock:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("rating") != "down":
                    continue
                items.append({
                    "prompt": r.get("query", ""),
                    "patient_context": r.get("patient_context"),
                    "rejected": r.get("answer", ""),
                    "chosen": r.get("correction") or None,
                    "meta": {
                        "source": "feedback:user_correction" if r.get("correction") else "feedback:downvote_only",
                        "ts": r.get("ts"),
                    },
                })
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        with open(FEEDBACK_STUB_PATH, "w", encoding="utf-8") as fh:
            for it in items:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return {"count": len(items), "items": items}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
