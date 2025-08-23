"""
SAT Vocabulary AI System - Main Application
FastAPI web application for interactive vocabulary generation
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from core.vocabulary_generator_clean import SimpleVocabularyGenerator, GeneratedVocabularyEntry
try:
    from core.rag_engine_clean import get_rag_engine, VocabularyEntry
except ImportError:
    from core.rag_engine_simple import get_rag_engine, VocabularyEntry
from services.llm_service import get_llm_service

# Simple user feedback model
class UserFeedback:
    def __init__(self, rating, comments, word):
        self.rating = rating
        self.comments = comments  
        self.word = word

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)   
logger = logging.getLogger(__name__)


# Pydantic models for API
class GenerateRequest(BaseModel):
    word: str
    use_context: bool = True
    num_context_examples: int = 3


class BatchGenerateRequest(BaseModel):
    words: List[str]
    use_context: bool = True
    num_context_examples: int = 3


class SearchRequest(BaseModel):
    word: str
    top_k: int = 5
    similarity_threshold: float = 0.3


class FeedbackRequest(BaseModel):
    word: str
    entry_id: str
    satisfaction_score: int
    helpful_components: List[str] = []
    problematic_components: List[str] = []
    user_comments: str = ""
    would_recommend: bool = False


class RegenerateRequest(BaseModel):
    word: str
    part_of_speech: str = "noun"
    use_simple: bool = True
    regeneration_reason: str
    specific_issue: str = ""
    improvement_suggestions: str = ""


class RegenerateFeedbackRequest(BaseModel):
    word: str
    satisfaction_score: int
    helpful_components: List[str] = []
    user_comments: str = ""

# Global variables for services
generator = None
rag_engine = None
llm_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global generator, rag_engine, llm_service
    
    logger.info("🚀 Initializing SAT Vocabulary AI System...")
    
    try:
        # Initialize services
        logger.info("📊 Initializing RAG engine...")
        rag_engine = get_rag_engine()
        
        logger.info("🤖 Initializing LLM service...")
        llm_service = get_llm_service()
        
        logger.info("⚡ Initializing vocabulary generator...")
        generator = SimpleVocabularyGenerator(llm_service, rag_engine)
        
        # Store in app state for access in endpoints
        app.state.rag_engine = rag_engine
        app.state.llm_service = llm_service
        app.state.generator = generator
        
        logger.info("✅ Services initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        import traceback
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
        # Don't raise - allow app to start in degraded mode
        yield
    
    finally:
        logger.info("🔄 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SAT Vocabulary AI System",
    description="Generate authentic Gulotta-style vocabulary entries using RAG and AI",
    version="1.0.0",
    lifespan=lifespan
)

# Setup templates and static files
templates_dir = Path(__file__).parent / "web" / "templates"
static_dir = Path(__file__).parent / "web" / "static"

templates = Jinja2Templates(directory=str(templates_dir))

# Create static directory if it doesn't exist
static_dir.mkdir(parents=True, exist_ok=True)

# Mount static files if directory exists
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Health check endpoint for deployment platforms
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SAT Vocabulary AI", "timestamp": "2025-08-23"}


# Web interface routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with vocabulary generation form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/word/{word}", response_class=HTMLResponse)
async def word_detail(request: Request, word: str):
    """Word detail page showing generated entry"""
    try:
        # Get services from app state
        generator = request.app.state.generator
        rag_engine = request.app.state.rag_engine
        
        # Generate vocabulary entry
        entry = generator.generate_entry(word)
        
        # Get similar entries for context
        similar_entries = rag_engine.retrieve_similar_entries(word, top_k=3)
        
        # Generate unique entry ID for feedback
        import uuid
        entry_id = str(uuid.uuid4())
        
        return templates.TemplateResponse("word_detail.html", {
            "request": request,
            "word": word,
            "entry": entry,
            "similar_entries": similar_entries,
            "entry_id": entry_id,
            "is_regenerated": False
        })
        
    except Exception as e:
        logger.error(f"Error generating word detail for {word}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_class=HTMLResponse)
async def generate_form(
    request: Request,
    word: str = Form(...),
    use_context: bool = Form(True),
    format_output: str = Form("web")
):
    """Handle form submission for word generation"""
    try:
        # Get generator from app state
        generator = request.app.state.generator
        
        # Generate vocabulary entry
        entry = generator.generate_complete_entry(
            word=word,
            use_context=use_context,
            num_context_examples=3
        )
        
        if format_output == "json":
            # Return JSON response
            return JSONResponse({
                "word": entry.word,
                "pronunciation": entry.pronunciation,
                "part_of_speech": entry.part_of_speech,
                "definition": entry.definition,
                "mnemonic_type": entry.mnemonic_type,
                "mnemonic_phrase": entry.mnemonic_phrase,
                "picture_story": entry.picture_story,
                "other_forms": entry.other_forms,
                "example_sentence": entry.example_sentence,
                "quality_score": entry.quality_score,
                "validation_passed": entry.validation_passed
            })
        
        # Return web page
        # Generate unique entry ID for feedback
        import uuid
        entry_id = str(uuid.uuid4())
        
        return templates.TemplateResponse("word_detail.html", {
            "request": request,
            "word": word,
            "entry": entry,
            "similar_entries": [],
            "entry_id": entry_id,
            "is_regenerated": False
        })
        
    except Exception as e:
        logger.error(f"Error generating entry for {word}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API routes
@app.post("/api/generate")
async def api_generate(request: GenerateRequest, fastapi_request: Request):
    """API endpoint for generating a single vocabulary entry"""
    try:
        # Get generator from app state
        generator = fastapi_request.app.state.generator
        
        entry = generator.generate_entry(
            word=request.word)
        
        return {
            "word": entry.word,
            "pronunciation": entry.pronunciation,
            "part_of_speech": entry.part_of_speech,
            "definition": entry.definition,
            "mnemonic_type": entry.mnemonic_type,
            "mnemonic_phrase": entry.mnemonic_phrase,
            "picture_story": entry.picture_story,
            "other_forms": entry.other_forms,
            "example_sentence": entry.example_sentence,
            "quality_score": entry.quality_score,
            "validation_passed": entry.validation_passed,
            "generation_metadata": entry.generation_metadata
        }
        
    except Exception as e:
        logger.error(f"Error in API generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch-generate")
async def api_batch_generate(request: BatchGenerateRequest, fastapi_request: Request):
    """API endpoint for generating multiple vocabulary entries"""
    try:
        # Get generator from app state
        generator = fastapi_request.app.state.generator
        
        entries = generator.batch_generate(
            words=request.words,
            use_context=request.use_context,
            num_context_examples=request.num_context_examples
        )
        
        result = []
        for entry in entries:
            result.append({
                "word": entry.word,
                "pronunciation": entry.pronunciation,
                "part_of_speech": entry.part_of_speech,
                "definition": entry.definition,
                "mnemonic_type": entry.mnemonic_type,
                "mnemonic_phrase": entry.mnemonic_phrase,
                "picture_story": entry.picture_story,
                "other_forms": entry.other_forms,
                "example_sentence": entry.example_sentence,
                "quality_score": entry.quality_score,
                "validation_passed": entry.validation_passed
            })
        
        return {"entries": result}
        
    except Exception as e:
        logger.error(f"Error in API batch generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def api_search(request: SearchRequest, fastapi_request: Request):
    """API endpoint for searching similar vocabulary entries"""
    try:
        # Get rag_engine from app state
        rag_engine = fastapi_request.app.state.rag_engine
        
        similar_entries = rag_engine.retrieve_similar_entries(
            query_word=request.word,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        result = []
        for entry, score in similar_entries:
            result.append({
                "word": entry.word,
                "pronunciation": entry.pronunciation,
                "definition": entry.definition,
                "mnemonic_phrase": entry.mnemonic_phrase,
                "similarity_score": score
            })
        
        return {"similar_entries": result}
        
    except Exception as e:
        logger.error(f"Error in API search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(feedback_request: FeedbackRequest, request: Request):
    """Submit user feedback for a vocabulary entry"""
    try:
        # Get quality checker from app state
        quality_checker = request.app.state.quality_checker
        
        # Create UserFeedback object
        import uuid
        from datetime import datetime
        
        feedback = UserFeedback(
            word=feedback_request.word,
            entry_id=feedback_request.entry_id,
            satisfaction_score=feedback_request.satisfaction_score,
            helpful_components=feedback_request.helpful_components,
            problematic_components=feedback_request.problematic_components,
            user_comments=feedback_request.user_comments,
            would_recommend=feedback_request.would_recommend,
            timestamp=datetime.now().isoformat(),
            user_id=None  # Could be session-based in the future
        )
        
        # Store feedback
        quality_checker.record_user_feedback(feedback)
        
        logger.info(f"Feedback recorded for word: {feedback_request.word}, satisfaction: {feedback_request.satisfaction_score}")
        
        return {
            "success": True,
            "message": "Feedback recorded successfully",
            "feedback_id": feedback.entry_id
        }
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback/regenerated")
async def submit_regenerated_feedback(feedback_request: FeedbackRequest, request: Request):
    """Submit user feedback for a regenerated vocabulary entry"""
    try:
        # Get services from app state
        quality_checker = request.app.state.quality_checker
        rag_engine = request.app.state.rag_engine
        
        # Create UserFeedback object
        import uuid
        from datetime import datetime
        
        feedback = UserFeedback(
            word=feedback_request.word,
            entry_id=feedback_request.entry_id,
            satisfaction_score=feedback_request.satisfaction_score,
            helpful_components=feedback_request.helpful_components,
            problematic_components=feedback_request.problematic_components,
            user_comments=feedback_request.user_comments,
            would_recommend=feedback_request.would_recommend,
            timestamp=datetime.now().isoformat(),
            user_id=None  # Could be session-based in the future
        )
        
        # Store feedback
        quality_checker.record_user_feedback(feedback)
        
        # If satisfaction is high (7+), store as positive example for RAG
        if feedback_request.satisfaction_score >= 7:
            # Get the regenerated entry details if available
            positive_example = f"""
HIGH-QUALITY REGENERATED ENTRY FOR {feedback_request.word.upper()}:
Satisfaction Score: {feedback_request.satisfaction_score}/10
User Comments: {feedback_request.user_comments}
Helpful Components: {', '.join(feedback_request.helpful_components)}
Recommendation: {'Yes' if feedback_request.would_recommend else 'No'}

This entry received high user satisfaction and should be used as a template for future {feedback_request.word} generations.
"""
            rag_engine.add_positive_example(feedback_request.word, positive_example)
            logger.info(f"High-rated regenerated entry for '{feedback_request.word}' stored as positive RAG example")
        
        logger.info(f"Regenerated feedback recorded for word: {feedback_request.word}, satisfaction: {feedback_request.satisfaction_score}")
        
        return {
            "success": True,
            "message": "Regenerated feedback recorded successfully",
            "feedback_id": feedback.entry_id,
            "stored_as_positive": feedback_request.satisfaction_score >= 7
        }
        
    except Exception as e:
        logger.error(f"Error recording regenerated feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics dashboard page"""
    try:
        # Simple analytics - count feedback files
        feedback_dir = "feedback_data"
        analytics = {
            "total_feedback": 0,
            "avg_rating": 0,
            "entries_generated": 0
        }
        
        if os.path.exists(feedback_dir):
            neg_file = os.path.join(feedback_dir, "negative_examples.txt")
            pos_file = os.path.join(feedback_dir, "positive_examples.txt")
            
            if os.path.exists(neg_file):
                with open(neg_file, 'r') as f:
                    analytics["negative_examples"] = len([line for line in f if "NEGATIVE EXAMPLE" in line])
            
            if os.path.exists(pos_file):
                with open(pos_file, 'r') as f:
                    analytics["positive_examples"] = len([line for line in f if "POSITIVE EXAMPLE" in line])
        
        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "analytics": analytics
        })
        
    except Exception as e:
        logger.error(f"Error loading analytics page: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics")
async def get_analytics(request: Request):
    """Get feedback analytics"""
    try:
        # Simple analytics data
        analytics = {
            "total_feedback": 0,
            "avg_rating": 0,
            "recent_words": []
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def api_stats(request: Request):
    """API endpoint for vocabulary statistics"""
    try:
        # Get rag_engine from app state
        rag_engine = request.app.state.rag_engine
        
        entries = rag_engine.entries
        
        # Count by mnemonic type
        mnemonic_types = {}
        for entry in entries:
            mtype = entry.mnemonic_type or "Unknown"
            mnemonic_types[mtype] = mnemonic_types.get(mtype, 0) + 1
        
        # Count by part of speech
        pos_counts = {}
        for entry in entries:
            pos = entry.part_of_speech or "Unknown"
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        return {
            "total_entries": len(entries),
            "mnemonic_types": mnemonic_types,
            "parts_of_speech": pos_counts
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check(request: Request):
    """Health check endpoint"""
    try:
        # Get services from app state
        llm_service = request.app.state.llm_service
        rag_engine = request.app.state.rag_engine
        generator = request.app.state.generator
        
        # Test LLM service
        response = llm_service.generate_completion(
            prompt="Test",
            max_tokens=1
        )
        
        return {
            "status": "healthy",
            "llm_service": "connected" if response.success else "error",
            "rag_entries_loaded": len(rag_engine.entries),
            "services_initialized": all([generator, rag_engine, llm_service])
        }
        
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/regenerate/{word}", response_class=HTMLResponse)
async def regenerate_form(request: Request, word: str):
    """Show the regenerate form for a specific word"""
    return templates.TemplateResponse("regenerate.html", {
        "request": request,
        "word": word
    })


@app.post("/regenerate", response_class=HTMLResponse)
async def regenerate_entry_web(
    request: Request,
    word: str = Form(...),
    regeneration_reason: str = Form(...),
    specific_issue: str = Form(""),
    improvement_suggestions: str = Form(""),
    part_of_speech: str = Form("noun"),
    use_simple: bool = Form(True)
):
    """Web endpoint for regenerating vocabulary entry with feedback"""
    try:
        # Store negative feedback
        from datetime import datetime
        negative_feedback = {
            'word': word,
            'reason': regeneration_reason,
            'specific_issue': specific_issue,
            'improvement_suggestions': improvement_suggestions,
            'timestamp': str(datetime.now())
        }
        
        # Get services from app state
        rag_engine = request.app.state.rag_engine
        llm_service = request.app.state.llm_service
        
        # Store negative feedback in RAG
        rag_engine.add_negative_example(word, f"""
NEGATIVE FEEDBACK FOR {word.upper()}:
Issue: {regeneration_reason} - {specific_issue}
Avoid: {improvement_suggestions}
This type of generation should be avoided for {word}.
        """.strip())
        
        # Generate new entry with feedback context
        if use_simple:
            from src.core.vocabulary_generator_simple import SimpleVocabularyGenerator
            simple_generator = SimpleVocabularyGenerator(llm_service, rag_engine)
            entry = simple_generator.generate_entry(
                word=word,
                part_of_speech=part_of_speech,
                avoid_issues=negative_feedback
            )
        else:
            generator = request.app.state.generator
            entry = generator.generate_complete_entry(
                word=word,
                use_context=True,
                num_context_examples=5
            )
        
        # Render the word detail page with regenerated entry
        return templates.TemplateResponse("word_detail.html", {
            "request": request,
            "entry": entry,
            "word": word,
            "is_regenerated": True,
            "feedback_context": negative_feedback
        })
        
    except Exception as e:
        logger.error(f"Error regenerating entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/regenerate")
async def regenerate_entry_api(regenerate_request: RegenerateRequest, request: Request):
    """API endpoint for regenerating vocabulary entry with feedback"""
    try:
        # Store negative feedback
        from datetime import datetime
        negative_feedback = {
            'word': regenerate_request.word,
            'reason': regenerate_request.regeneration_reason,
            'specific_issue': regenerate_request.specific_issue,
            'improvement_suggestions': regenerate_request.improvement_suggestions,
            'timestamp': str(datetime.now())
        }
        
        # Get services from app state
        rag_engine = request.app.state.rag_engine
        llm_service = request.app.state.llm_service
        
        # Store negative feedback in RAG
        rag_engine.add_negative_example(regenerate_request.word, f"""
NEGATIVE FEEDBACK FOR {regenerate_request.word.upper()}:
Issue: {regenerate_request.regeneration_reason} - {regenerate_request.specific_issue}
Avoid: {regenerate_request.improvement_suggestions}
This type of generation should be avoided for {regenerate_request.word}.
        """.strip())
        
        # Generate new entry with feedback context
        if regenerate_request.use_simple:
            from src.core.vocabulary_generator_simple import SimpleVocabularyGenerator
            simple_generator = SimpleVocabularyGenerator(llm_service, rag_engine)
            entry = simple_generator.generate_entry(
                word=regenerate_request.word,
                part_of_speech=regenerate_request.part_of_speech,
                avoid_issues=negative_feedback
            )
        else:
            generator = request.app.state.generator
            entry = generator.generate_complete_entry(
                word=regenerate_request.word,
                use_context=True,
                num_context_examples=5
            )
        
        # Return the regenerated entry
        return {
            "success": True,
            "entry": {
                "word": entry.word,
                "pronunciation": entry.pronunciation,
                "part_of_speech": entry.part_of_speech,
                "definition": entry.definition,
                "mnemonic_type": entry.mnemonic_type,
                "mnemonic_phrase": entry.mnemonic_phrase,
                "picture_story": entry.picture_story,
                "other_forms": entry.other_forms,
                "example_sentence": entry.example_sentence,
                "quality_score": getattr(entry, 'quality_score', 0.0),
                "validation_passed": getattr(entry, 'validation_passed', True)
            },
            "feedback_stored": True,
            "regeneration_context": negative_feedback
        }
        
    except Exception as e:
        logger.error(f"Error regenerating entry via API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/regenerate-feedback")
async def submit_regenerate_feedback(feedback_request: RegenerateFeedbackRequest, request: Request):
    """Submit feedback for a regenerated entry"""
    try:
        # Get services from app state
        rag_engine = request.app.state.rag_engine
        quality_checker = request.app.state.quality_checker
        
        # If feedback is positive (≥7), store as positive example
        if feedback_request.satisfaction_score >= 7:
            # Create a positive example entry for RAG
            positive_example = f"""
EXCELLENT REGENERATED EXAMPLE for {feedback_request.word.upper()} (satisfaction: {feedback_request.satisfaction_score}/10):
User feedback: {feedback_request.user_comments or 'Highly rated regenerated example'}
Helpful components: {', '.join(feedback_request.helpful_components)}
This regeneration was successful and should guide future generations.
            """.strip()
            
            rag_engine.add_positive_example(feedback_request.word, positive_example)
            
            logger.info(f"Stored positive regeneration feedback for {feedback_request.word}")
        
        # Store feedback in quality system as well
        from datetime import datetime
        import uuid
        
        feedback = UserFeedback(
            word=feedback_request.word,
            entry_id=f"regenerated_{feedback_request.word}_{int(datetime.now().timestamp())}",
            satisfaction_score=feedback_request.satisfaction_score,
            helpful_components=feedback_request.helpful_components,
            problematic_components=[],
            user_comments=feedback_request.user_comments,
            would_recommend=feedback_request.satisfaction_score >= 7,
            timestamp=datetime.now().isoformat(),
            user_id=None
        )
        
        quality_checker.record_user_feedback(feedback)
        
        return {
            "success": True,
            "message": "Regeneration feedback recorded successfully",
            "positive_example_stored": feedback_request.satisfaction_score >= 7
        }
        
    except Exception as e:
        logger.error(f"Error recording regeneration feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error": "Page not found",
        "status_code": 404
    }, status_code=404)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error": "Internal server error",
        "status_code": 500
    }, status_code=500)


if __name__ == "__main__":
    import uvicorn
    
    # Load configuration
    host = os.getenv("HOST", "localhost")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info(f"Starting SAT Vocabulary AI System on {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )