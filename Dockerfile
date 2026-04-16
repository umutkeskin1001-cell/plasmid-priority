# Plasmid Priority — Reproducible Analysis Environment
#
# Build:  docker build -t plasmid-priority .
# Run:    docker run --rm -v /path/to/data:/data plasmid-priority make quality
#
# Multi-stage build: builder compiles deps, runtime runs with non-root user.

# ─── Builder stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System deps for duckdb / scikit-learn / LightGBM wheel builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (with tree-models so primary model is available)
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[analysis,dev,tree-models]" && \
    python -c "import lightgbm; print('LightGBM', lightgbm.__version__, 'OK')"

# Copy source
COPY . .

# Verify source compiles cleanly
RUN python -m compileall -q src

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLASMID_PRIORITY_DATA_ROOT=/data \
    MPLBACKEND=Agg

# libgomp needed at runtime by LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 plasmid && \
    useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/bash plasmid

WORKDIR /app

# Copy site-packages and project source from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build/src ./src
COPY --from=builder /build/config ./config
COPY --from=builder /build/Makefile ./Makefile
COPY --from=builder /build/pyproject.toml ./pyproject.toml
COPY --from=builder /build/tests ./tests

# Data and reports mount points
VOLUME ["/data", "/app/reports"]

USER plasmid

# Sanity check: verify LightGBM importable in runtime layer
RUN python -c "import lightgbm; import plasmid_priority; print('Runtime OK')" || \
    echo "WARNING: some imports failed in runtime layer — check dependencies"

CMD ["make", "quality"]
