# Gymkhana REPL Server
# Isolated sandbox with pre-installed scientific packages

FROM python:3.11-slim

WORKDIR /app

# Install build dependencies used by scientific Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Gymkhana and its Pydantic AI inference router from this checkout.
COPY pyproject.toml README.md ./
COPY gymkhana ./gymkhana
RUN pip install --no-cache-dir .

# Install Python packages for math/code tasks
RUN pip install --no-cache-dir \
    flask \
    numpy \
    pandas \
    sympy \
    scipy \
    matplotlib \
    scikit-learn \
    networkx \
    requests \
    beautifulsoup4 \
    lxml \
    tenacity

# Create workspace directory
RUN mkdir -p /app/workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5003/health')" || exit 1

# Run server
CMD ["python", "-m", "gymkhana.core.services.sandboxes.server", "--port", "5003", "--host", "0.0.0.0"]
