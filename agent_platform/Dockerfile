FROM python:3.12-slim AS base

# System dependencies for reportlab, matplotlib, weasyprint
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create artifact directory
RUN mkdir -p /tmp/agent_artifacts

# Non-root user
RUN groupadd -r agent && useradd -r -g agent agent \
    && chown -R agent:agent /app /tmp/agent_artifacts
USER agent

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); r.raise_for_status()"

# Production entrypoint: gunicorn with uvicorn workers
CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
