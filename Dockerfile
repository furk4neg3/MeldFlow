# Production-ready-ish Dockerfile for serving
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends     git     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

EXPOSE 8000

# Default command: start FastAPI server
CMD [ "uvicorn", "mmplat.serving.app:app", "--host", "0.0.0.0", "--port", "8000" ]
