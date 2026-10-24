# FinGuard AI Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FINGUARD_API_HOST=0.0.0.0 \
    FINGUARD_API_PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
COPY models/ ./models/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command starts FastAPI backend
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
