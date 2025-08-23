#!/bin/bash
# Railway startup script

echo "🚀 Starting SAT Vocabulary System..."
echo "📁 Creating required directories..."
mkdir -p data/processed feedback_data

echo "🔧 Environment check..."
echo "PORT: ${PORT:-8000}"
echo "PYTHONPATH: $PYTHONPATH"

echo "▶️  Starting application..."
exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
