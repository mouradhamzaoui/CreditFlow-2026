# Image Python légère
FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY pyproject.toml uv.lock ./

# Install Python deps
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r pyproject.toml

# Copy application code and model artifacts
COPY src/ ./src/
COPY models/ ./models/

# Create log directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Healthcheck crucial pour Kubernetes / AWS ECS
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]