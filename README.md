# ICICI Bank Agentic HR Assistant

A multi-agent, full-stack HR automation system built during an internship at ICICI Bank.
Employees interact with a LangGraph-orchestrated chatbot that autonomously routes queries
to specialized agents handling leave, attendance, equipment, and policy lookups.

## Architecture

```
Streamlit / React Frontend
        |
        v
FastAPI Backend (REST API)
        |
        v
LangGraph Orchestrator  ← routes to specialized agents
    |       |       |       |
  Leave  Attend  Equip  Policy
  Agent  Agent   Agent   Agent
    |
    v
PostgreSQL (employee data, leave records, audit logs)
    |
    v
Groq LLM API (gemma2-9b-it)
```

## Stack
- **Orchestration:** LangGraph, LangChain
- **Backend:** FastAPI + Uvicorn
- **Database:** PostgreSQL (psycopg2 + SQLAlchemy)
- **LLM:** Groq (gemma2-9b-it)
- **CI/CD:** GitHub Actions (test + Docker build)
- **Infra:** Docker, Docker Compose

## Agents
| Agent | Responsibility |
|---|---|
| `leave_agent` | Leave application, approval routing, balance checks |
| `attendance_agent` | GPS-verified check-in/out, overtime calculation |
| `complaint_agent` | Equipment issues and IT complaint ticketing |
| `employee_agent` | Employee profile and payroll queries |
| `chatbot_agent` | General policy Q&A via vector retrieval |

## Setup

```bash
# 1. Copy environment config
cp .env.example .env  # Fill in your credentials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the backend
uvicorn src.main:app --reload

# 4. Run tests
pytest tests/ -v
```

> **Note:** Database credentials and API keys are excluded. See `.env.example`.
