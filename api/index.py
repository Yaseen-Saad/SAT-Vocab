"""
Vercel serverless function entry point for SAT Vocabulary System
"""
import sys
import os
from pathlib import Path

# Add src to Python path
root_dir = Path(__file__).parent.parent
src_dir = root_dir / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(root_dir))

# Create necessary directories (use /tmp for serverless)
import tempfile
temp_dir = Path(tempfile.gettempdir())
data_dir = temp_dir / "sat_vocab_data" / "processed"
feedback_dir = temp_dir / "sat_vocab_feedback"
data_dir.mkdir(parents=True, exist_ok=True)
feedback_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables for the app
os.environ["DATA_DIR"] = str(data_dir.parent)
os.environ["FEEDBACK_DIR"] = str(feedback_dir)

# Import the FastAPI app
from src.app import app

# Export the app for Vercel
handler = app
