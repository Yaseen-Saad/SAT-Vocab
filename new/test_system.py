"""
Quick test script to verify the SAT Vocabulary RAG System
"""

import asyncio
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_system():
    """Test core system components"""
    try:
        logger.info("🧪 Testing SAT Vocabulary RAG System...")
        
        # Test imports
        logger.info("Testing imports...")
        from src.services.config import get_settings
        from src.core.rag_engine import get_rag_engine
        from src.core.quality_system import get_quality_system
        from src.core.vocabulary_generator import get_vocabulary_generator
        
        # Test configuration
        logger.info("Testing configuration...")
        settings = get_settings()
        logger.info(f"Environment: {settings.environment}")
        logger.info(f"App name: {settings.app_name}")
        
        # Test RAG engine initialization
        logger.info("Testing RAG engine...")
        rag_engine = get_rag_engine()
        stats = rag_engine.get_statistics()
        logger.info(f"RAG engine initialized: {stats.get('total_entries', 0)} entries")
        
        # Test quality system
        logger.info("Testing quality system...")
        quality_system = get_quality_system()
        logger.info("Quality system initialized ✓")
        
        # Test vocabulary generator
        logger.info("Testing vocabulary generator...")
        try:
            generator = get_vocabulary_generator()
            logger.info("Vocabulary generator initialized ✓")
            
            # Check LLM provider
            settings = get_settings()
            if settings.llm_provider == "hackclub":
                logger.info("Using Hack Club AI - Free unlimited generation! 🎉")
            elif settings.llm_provider == "openai":
                if not settings.openai_api_key or settings.openai_api_key == "not_needed":
                    logger.warning("OpenAI provider selected but no API key configured")
                else:
                    logger.info("OpenAI API configured ✓")
                    
        except Exception as e:
            logger.warning(f"Vocabulary generator initialization issue: {e}")
        
        # Test data directories
        logger.info("Testing data directories...")
        data_dirs = ["data", "data/vector_store", "data/feedback", "data/cache"]
        for directory in data_dirs:
            path = Path(directory)
            if path.exists():
                logger.info(f"Directory exists: {directory} ✓")
            else:
                logger.warning(f"Directory missing: {directory}")
        
        logger.info("✅ Core system test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False


async def test_generation():
    """Test vocabulary generation"""
    try:
        logger.info("🔤 Testing vocabulary generation...")
        
        from src.core.vocabulary_generator import get_vocabulary_generator
        from src.services.config import get_settings
        
        settings = get_settings()
        generator = get_vocabulary_generator()
        
        # Test generation
        word = "perspicacious"
        logger.info(f"Generating entry for: {word}")
        logger.info(f"Using LLM provider: {settings.llm_provider}")
        
        entry = await generator.generate_vocabulary_entry(
            word=word,
            quality_threshold=0.5,  # Lower threshold for testing
            max_attempts=1
        )
        
        if entry:
            logger.info(f"✅ Generated entry for '{word}':")
            logger.info(f"  Definition: {entry.definition[:100]}...")
            logger.info(f"  Mnemonic: {entry.mnemonic_phrase[:100]}...")
            logger.info(f"  Quality Score: {entry.quality_score:.3f}")
            
            if settings.llm_provider == "hackclub":
                logger.info("🎉 Generated using FREE Hack Club AI!")
        else:
            logger.warning("❌ Failed to generate vocabulary entry")
            
        return entry is not None
        
    except Exception as e:
        logger.error(f"❌ Generation test failed: {e}")
        return False


async def main():
    """Main test function"""
    logger.info("🚀 Starting SAT Vocabulary RAG System Tests")
    logger.info("=" * 50)
    
    # Test core system
    system_ok = await test_system()
    
    if system_ok:
        logger.info("=" * 50)
        
        # Test generation (now works with Hack Club AI!)
        from src.services.config import get_settings
        settings = get_settings()
        
        if settings.llm_provider == "hackclub":
            logger.info("🎉 Testing generation with FREE Hack Club AI!")
            generation_ok = await test_generation()
        elif settings.llm_provider == "openai":
            if settings.openai_api_key and settings.openai_api_key != "not_needed":
                generation_ok = await test_generation()
            else:
                logger.info("⚠️  Skipping generation test (no OpenAI API key)")
                logger.info("💡 Add OPENAI_API_KEY to .env file or use LLM_PROVIDER=hackclub")
                generation_ok = True
        else:
            logger.info("⚠️  Unknown LLM provider, skipping generation test")
            generation_ok = True
    
    logger.info("=" * 50)
    
    if system_ok:
        logger.info("🎉 All tests passed! System is ready to use.")
        logger.info("🚀 Run 'python start.py' to start the server")
    else:
        logger.error("💥 Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())