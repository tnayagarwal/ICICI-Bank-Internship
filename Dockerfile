# Enterprise Dockerfile for NextJs / FastAPI / Python orchestration
FROM python:3.10-slim

WORKDIR /app

# System dependencies for production Python networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install orchestrator requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy orchestration code
COPY src/ /app/src/

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
