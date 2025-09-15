#!/usr/bin/env python3
"""
Local development startup script
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")


def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            logger.info("Creating .env file from .env.example")
            import shutil
            shutil.copy(env_example, env_file)
            logger.warning("Please edit .env file and add your OpenAI API key")
        else:
            logger.error(".env.example file not found")
            sys.exit(1)
    
    # Load environment variables
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        logger.info("Environment variables loaded from .env")


def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.error(f"Error output: {e.stderr}")
        sys.exit(1)


def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/vector_store",
        "data/feedback",
        "data/cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def check_api_key():
    """Check LLM configuration"""
    llm_provider = os.getenv("LLM_PROVIDER", "hackclub")
    
    if llm_provider == "hackclub":
        logger.info("Using Hack Club AI - No API key needed! 🎉")
        logger.info("Free unlimited AI for Hack Club teens!")
        return True
    elif llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "not_needed":
            logger.warning("OPENAI_API_KEY not found in environment variables")
            logger.warning("Please add your OpenAI API key to the .env file")
            return False
        logger.info("OpenAI API key found ✓")
        return True
    else:
        logger.warning(f"Unknown LLM provider: {llm_provider}")
        return False


def start_server():
    """Start the FastAPI server"""
    logger.info("Starting SAT Vocabulary RAG System...")
    
    try:
        # Import here to ensure dependencies are installed
        import uvicorn
        
        # Run the server with app import string for reload
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


def main():
    """Main startup function"""
    logger.info("🚀 Starting SAT Vocabulary RAG System Setup...")
    
    # Check Python version
    check_python_version()
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Check LLM configuration
    has_valid_config = check_api_key()
    
    if not has_valid_config:
        llm_provider = os.getenv("LLM_PROVIDER", "hackclub")
        if llm_provider == "openai":
            logger.error("Cannot start without OpenAI API key")
            logger.info("Please:")
            logger.info("1. Edit the .env file")
            logger.info("2. Add your OpenAI API key: OPENAI_API_KEY=your_key_here")
            logger.info("3. Run this script again")
            logger.info("OR switch to Hack Club AI by setting LLM_PROVIDER=hackclub")
            sys.exit(1)
        else:
            logger.error("LLM configuration invalid")
            sys.exit(1)
    
    logger.info("✅ Setup complete! Starting server...")
    
    # Start server
    start_server()


if __name__ == "__main__":
    main()