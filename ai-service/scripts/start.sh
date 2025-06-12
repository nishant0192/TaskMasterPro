# ai-service/scripts/start.sh
#!/bin/bash
"""
Production startup script for AI Service
Handles initialization, health checks, and graceful startup
"""

set -e

echo "🚀 Starting TaskMaster AI Service..."

# Environment variables with defaults
export ENVIRONMENT=${ENVIRONMENT:-production}
export DEBUG=${DEBUG:-false}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export WORKERS=${WORKERS:-4}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}

# Wait for dependencies
echo "⏳ Waiting for dependencies..."

# Wait for PostgreSQL
echo "Checking PostgreSQL connection..."
until pg_isready -h ${DB_HOST:-postgres} -p ${DB_PORT:-5432} -U ${DB_USER:-aiuser}; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 2
done
echo "✅ PostgreSQL is ready"

# Wait for Redis
echo "Checking Redis connection..."
until redis-cli -h ${REDIS_HOST:-redis} -p ${REDIS_PORT:-6379} ping; do
    echo "Redis is unavailable - sleeping"
    sleep 2
done
echo "✅ Redis is ready"

# Run database migrations
echo "🔄 Running database migrations..."
alembic upgrade head

# Download required models if not present
echo "📥 Checking AI models..."
python -c "
import os
from sentence_transformers import SentenceTransformer
import spacy

# Download sentence transformer model
model_dir = '/app/models/embedding'
if not os.path.exists(model_dir):
    print('Downloading sentence transformer model...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.save(model_dir)
    print('✅ Sentence transformer model ready')

# Verify spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy model ready')
except OSError:
    print('❌ spaCy model not found')
    exit(1)
"

# Pre-warm the application
echo "🔥 Pre-warming application..."
python -c "
import asyncio
from app.core.ai_engine import ai_engine
from app.services.personalization_engine import personalization_engine

async def warmup():
    try:
        await ai_engine.initialize()
        await personalization_engine.initialize()
        print('✅ AI components initialized')
    except Exception as e:
        print(f'❌ Warmup failed: {e}')
        exit(1)

asyncio.run(warmup())
"

echo "✅ AI Service initialization complete"

# Start the application
if [ "$ENVIRONMENT" = "development" ]; then
    echo "🔧 Starting in development mode..."
    exec uvicorn app.main:app \
        --host $HOST \
        --port $PORT \
        --reload \
        --log-level $LOG_LEVEL
else
    echo "🚀 Starting in production mode..."
    exec gunicorn app.main:app \
        -w $WORKERS \
        -k uvicorn.workers.UvicornWorker \
        --bind $HOST:$PORT \
        --log-level $LOG_LEVEL \
        --access-logfile - \
        --error-logfile - \
        --worker-tmp-dir /dev/shm \
        --worker-connections 1000 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --preload \
        --timeout 300 \
        --keep-alive 2
fi