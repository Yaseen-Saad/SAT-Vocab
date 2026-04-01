# SAT Vocab RAG

FastAPI app that generates SAT vocabulary entries with mnemonics, context, and feedback-aware retrieval.

## Features
- FastAPI web UI and JSON API
- RAG-style context from stored entries
- Feedback capture (positive/negative examples)
- Hack Club AI integration via authenticated proxy API
- Vercel serverless entrypoint and local run support

## Tech Stack
- Python 3.11+
- FastAPI + Jinja2
- Requests
- Vercel Python runtime

## Project Structure
- src/app.py: main FastAPI application
- src/core/rag_engine_clean.py: retrieval and feedback context
- src/core/vocabulary_generator_clean.py: generation orchestration
- src/services/llm_service.py: Hack Club AI client
- api/index.py: Vercel entrypoint
- tests/test_smoke.py: basic local smoke tests

## Local Setup
1. Create a virtual environment and activate it.
2. Install dependencies:
   pip install -r requirements.txt
3. Copy env template and set your key:
   copy .env.example .env
4. Set HACKCLUB_API_KEY in .env.
5. Run locally:
   python main.py
6. Open:
   http://localhost:8000

## Local Testing
Install dev dependencies:
- pip install -r requirements-dev.txt

Run tests:
- pytest -q

## Required Environment Variables
- HACKCLUB_API_KEY: your key from https://ai.hackclub.com/dashboard

Optional:
- HACKCLUB_API_URL (default: https://ai.hackclub.com/proxy/v1)
- HACKCLUB_MODEL (default: google/gemini-2.5-flash)
- DEBUG (default: false)
- ENVIRONMENT (development or production)

## Deploy to Vercel (GitHub Import)
1. Import this repo in Vercel.
2. Keep Framework Preset as Other.
3. Root Directory: ./
4. Leave build/output commands empty (vercel.json is used).
5. Add env var:
   - HACKCLUB_API_KEY
6. Deploy.

## Health Endpoints
- /health
- /api/health

## Notes
- Vercel file system is read-only except temporary directories.
- Feedback persistence is best-effort in serverless; local runs persist to project folders.
