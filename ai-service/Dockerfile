# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG APP_VERSION=1.0.0

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

# Install dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    if [ "$BUILD_ENV" = "development" ]; then \
        pip install --no-cache-dir -r /tmp/requirements-dev.txt; \
    fi

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Production stage
FROM python:3.11-slim

# Set labels
LABEL maintainer="your-team@company.com"
LABEL version=${APP_VERSION}
LABEL description="TaskMaster Pro AI Service"

# Create non-root user
RUN groupadd -r aiservice && useradd -r -g aiservice aiservice

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy application code
COPY --chown=aiservice:aiservice app/ /app/
COPY --chown=aiservice:aiservice scripts/ /app/scripts/
COPY --chown=aiservice:aiservice alembic.ini /app/
COPY --chown=aiservice:aiservice alembic/ /app/alembic/

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/cache && \
    chown -R aiservice:aiservice /app

# Copy startup script
COPY --chown=aiservice:aiservice scripts/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/ai/health || exit 1

# Switch to non-root user
USER aiservice

# Expose port
EXPOSE 8000

# Default command
CMD ["/app/start.sh"]