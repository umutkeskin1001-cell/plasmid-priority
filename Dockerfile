# Plasmid Priority – reproducible analysis environment
# Build:  docker build -t plasmid-priority .
# Run:    docker run --rm plasmid-priority make quality

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps forduckdb / scikit-learn wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[analysis,dev]"

# Copy source
COPY . .

# Compile sources as a quick sanity check
RUN python -m compileall -q src

# Default entrypoint
CMD ["make", "quality"]
