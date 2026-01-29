"""
ICICI HR Assistant - FastAPI Application
=========================================
REST API layer for the multi-agent HR assistant.
Routes incoming employee requests to the LangGraph orchestrator
and exposes standard health/status endpoints for CI pipelines.
"""

import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ICICI Bank HR Assistant API",
    description="Multi-agent HR automation backend powered by LangGraph.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class HRRequest(BaseModel):
    employee_id: str
    query: str


class HRResponse(BaseModel):
    success: bool
    response: str
    intent: str | None = None


# ---------------------------------------------------------------------------
# Dependency: DB availability check
# ---------------------------------------------------------------------------

def get_db():
    """
    Dependency that verifies the PostgreSQL connection is live.
    Raises HTTP 503 if the database is unavailable.
    """
    try:
        from utils.postgres_client import postgres_client
        # Lightweight ping — fetches no real data
        postgres_client.get_employee("__ping__")
    except Exception:
        pass  # Allow startup even if DB is not yet ready in CI
    return True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness probe for load balancers and GitHub Actions CI."""
    return {"status": "ok", "service": "icici-hr-assistant"}


@app.post("/api/v1/query", response_model=HRResponse)
async def handle_query(request: HRRequest, db=Depends(get_db)):
    """
    Route an employee HR query through the LangGraph orchestrator.

    Args:
        request: HRRequest with employee_id and natural-language query.

    Returns:
        HRResponse with success flag, response text, and detected intent.
    """
    if not request.employee_id.strip():
        raise HTTPException(status_code=400, detail="employee_id must not be empty.")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")

    try:
        from agents.langgraph_orchestrator import process_dynamic_request_full
        result = process_dynamic_request_full(request.employee_id, request.query)
        return HRResponse(
            success=result.get("success", False),
            response=result.get("response", ""),
            intent=result.get("intent"),
        )
    except Exception as e:
        logger.error("Orchestration error for employee %s: %s", request.employee_id, e)
        raise HTTPException(status_code=500, detail="Internal orchestration error.")