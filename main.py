#!/usr/bin/env python3
"""
Startup script for SAT Vocabulary System
This ensures the app starts correctly in any environment
"""

import os
import sys
import uvicorn

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run the app
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
