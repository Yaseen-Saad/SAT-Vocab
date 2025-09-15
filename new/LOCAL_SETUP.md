# SAT Vocabulary RAG System - Local Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- **NO API KEY NEEDED!** (Uses free Hack Club AI)
- Or OpenAI API key if you prefer GPT models

### 1. Set up Environment

Copy the environment template:
```bash
copy .env.example .env
```

**Default configuration uses FREE Hack Club AI - no API key needed!**

If you want to use OpenAI instead, edit `.env` file:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

**Option A: Use the startup script (recommended)**
```bash
python start.py
```

**Option B: Manual startup**
```bash
python main.py
```

**Option C: Using uvicorn directly**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📱 Accessing the Application

Once running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Root Info**: http://localhost:8000/

## 🔧 Configuration

### Environment Variables

Key settings in `.env`:

```bash
# Core Settings
ENVIRONMENT=development
DEBUG=true
HOST=localhost
PORT=8000

# LLM Configuration (FREE Hack Club AI!)
LLM_PROVIDER=hackclub
LLM_MODEL=qwen/qwen3-32b
LLM_API_URL=https://ai.hackclub.com

# Alternative: OpenAI (requires API key)
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_key_here
# LLM_MODEL=gpt-3.5-turbo

# Quality Thresholds
QUALITY_THRESHOLD=7.0
EXCELLENCE_THRESHOLD=8.5

# Database
DATABASE_URL=sqlite:///./data/vocabulary.db
VECTOR_DB_PATH=./data/vector_store
```

## 📋 API Endpoints

### Vocabulary Generation
- `POST /api/v1/vocabulary/generate` - Generate single entry
- `POST /api/v1/vocabulary/batch-generate` - Generate multiple entries

### Search & Discovery
- `POST /api/v1/vocabulary/search` - Search vocabulary entries
- `GET /api/v1/vocabulary/contextual/{word}` - Get contextual examples

### Feedback & Quality
- `POST /api/v1/feedback/submit` - Submit user feedback
- `POST /api/v1/vocabulary/assess-quality` - Assess entry quality
- `POST /api/v1/vocabulary/{word}/regenerate` - Regenerate with feedback

### System Management
- `GET /api/v1/stats/system` - System statistics
- `GET /health` - Health check

## 🧪 Testing the System

### 1. Generate a Vocabulary Entry

```bash
curl -X POST "http://localhost:8000/api/v1/vocabulary/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "word": "perspicacious",
    "quality_threshold": 0.7,
    "max_attempts": 3
  }'
```

### 2. Search for Similar Words

```bash
curl -X POST "http://localhost:8000/api/v1/vocabulary/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "intelligent",
    "top_k": 5,
    "search_type": "hybrid"
  }'
```

### 3. Submit Feedback

```bash
curl -X POST "http://localhost:8000/api/v1/feedback/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "word": "perspicacious",
    "feedback_type": "positive",
    "feedback_text": "Great mnemonic, very memorable!"
  }'
```

## 📊 Example Usage

### Python Client

```python
import requests

# Generate vocabulary entry
response = requests.post("http://localhost:8000/api/v1/vocabulary/generate", 
    json={"word": "perspicacious"})
entry = response.json()
print(f"Generated: {entry['word']} - {entry['definition']}")

# Search for similar entries
response = requests.post("http://localhost:8000/api/v1/vocabulary/search",
    json={"query": "smart", "top_k": 3})
results = response.json()
print(f"Found {results['total_found']} similar entries")
```

### JavaScript/Frontend

```javascript
// Generate vocabulary entry
const generateWord = async (word) => {
  const response = await fetch('/api/v1/vocabulary/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ word: word })
  });
  return response.json();
};

// Usage
generateWord('perspicacious').then(entry => {
  console.log('Generated:', entry);
});
```

## 🔍 Directory Structure

```
new/
├── main.py                 # FastAPI application
├── start.py               # Startup script
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
├── .env                  # Your environment config
├── data/                 # Data storage
│   ├── vector_store/     # Vector database
│   ├── feedback/         # User feedback
│   └── cache/           # Cache files
└── src/
    ├── api/             # API endpoints
    ├── core/            # Core business logic
    ├── models/          # Data models
    └── services/        # Services & config
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **LLM Provider Issues**
   - Default uses FREE Hack Club AI (no setup needed!)
   - For OpenAI: Add `OPENAI_API_KEY=your_key` to `.env`
   - Restart the application after changes

3. **Port Already in Use**
   - Change `PORT=8001` in `.env`
   - Or kill process on port 8000

4. **Permission Errors**
   - Ensure write permissions for `data/` directory
   - Run with appropriate user permissions

### Logs

Check logs for detailed error information:
```bash
python start.py 2>&1 | tee app.log
```

## 🚀 Next Steps

1. **Add Sample Data**: Import vocabulary from Gulotta PDF
2. **Configure Quality**: Adjust quality thresholds in `.env`
3. **Enable Feedback**: Start collecting user feedback
4. **Monitor Performance**: Check `/api/v1/stats/system`

## 💡 Tips

- Use the interactive API docs at `/docs` for testing
- Start with simple words to test the system
- Monitor quality scores and adjust thresholds
- Collect feedback to improve generation quality
- Check system stats regularly for performance insights