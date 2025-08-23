#!/bin/bash
# Railway startup script

echo "🚀 Starting SAT Vocabulary System..."
echo "📁 Creating required directories..."
mkdir -p data/processed feedback_data

echo "🔧 Environment check..."
echo "PORT: ${PORT:-8000}"
echo "PYTHONPATH: $PYTHONPATH"

# Set default port if not provided
export PORT=${PORT:-8000}

echo "▶️  Starting application on port $PORT..."
exec uvicorn src.app:app --host 0.0.0.0 --port $PORT --log-level info
