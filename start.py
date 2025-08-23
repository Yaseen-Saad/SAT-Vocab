#!/usr/bin/env python3
"""
Railway deployment startup script
Handles environment variables properly and starts the application
"""
import os
import subprocess
import sys

def main():
    print("🚀 Starting SAT Vocabulary System...")
    
    # Create required directories
    print("📁 Creating required directories...")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("feedback_data", exist_ok=True)
    
    # Get port from environment
    port = os.environ.get("PORT", "8000")
    print(f"🔧 Port: {port}")
    print(f"🐍 Python: {sys.version}")
    
    # Start uvicorn
    print("▶️  Starting application...")
    cmd = [
        "uvicorn", 
        "src.app:app", 
        "--host", "0.0.0.0", 
        "--port", str(port),
        "--log-level", "info"
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.exec_cmd = cmd
    os.execvp("uvicorn", cmd)

if __name__ == "__main__":
    main()
