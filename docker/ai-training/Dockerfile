# AI Training and Deployment Pipeline Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-ai.txt .
RUN pip install --no-cache-dir -r requirements-ai.txt

# Copy application code
COPY backend/ ./backend/
COPY k8s/ ./k8s/

# Create necessary directories
RUN mkdir -p /models /data /logs /tmp/k8s_manifests

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV KUBEFLOW_HOST=http://kubeflow-pipelines:8888

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "backend.core.ai_training_deployment_endpoints"]