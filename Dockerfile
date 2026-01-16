
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (for Docker layer caching)
# If these don't change, Docker reuses this layer = faster rebuilds
COPY pyproject.toml uv.lock ./

# Install dependencies
# --frozen: Don't update lock file
# --no-dev: Skip development dependencies (ragas, testing tools)
RUN uv sync --frozen --no-dev


FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only the installed packages from builder stage
# This leaves behind build tools, cache, etc. = smaller image
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/
COPY streamlit/ ./streamlit/

# Copy pre-built indices (CRITICAL - these are your vector DB and BM25 index)
# Without these, the app won't work!
COPY chroma_db/ ./chroma_db/
COPY bm25_index.pkl ./bm25_index.pkl

# Pre-download HuggingFace models during build (avoids runtime download + rate limits)
RUN /app/.venv/bin/python -c "\
from sentence_transformers import CrossEncoder; \
print('Downloading cross-encoder model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Model cached successfully')"

# Environment variables
# PATH: Tell Python where to find installed packages
ENV PATH="/app/.venv/bin:$PATH"

# PYTHONUNBUFFERED: Print logs immediately (important for Cloud Run)
ENV PYTHONUNBUFFERED=1

# PORT: Cloud Run sets this dynamically, default to 8080
ENV PORT=8080

# Document which port the container listens on
EXPOSE 8080

# Run Streamlit with Cloud Run configuration
CMD ["sh", "-c", "streamlit run streamlit/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
