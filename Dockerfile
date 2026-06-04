# Single multi-purpose image used by every docker-compose service
# (pipeline, api, gradio, mlflow, test). The service's `command:` decides
# what actually runs — see docker-compose.yml.
FROM python:3.12-slim

# Keep Python output unbuffered and skip .pyc generation (cleaner container logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so the layer is cached across code changes
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv pip install --system . \
    && uv pip install --system pytest pytest-cov

# Copy the rest of the project
COPY . .

# Ensure runtime output dirs exist even on a fresh build context
RUN mkdir -p models reports mlruns data/raw

EXPOSE 8000 7860 5000 3000

# Default command = serving API. Overridden per-service in docker-compose.yml.
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
