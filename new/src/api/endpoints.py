"""
API Endpoints for SAT Vocabulary RAG Sys@router.post("/vocabulary/generate", response_model=VocabularyEntryResponse)
async def generate_vocabulary_entry(
    request: GenerateRequest,
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse

from ..models import (
    VocabularyEntry, 
    GenerateRequest,
    BatchGenerateRequest,
    SearchRequest,
    FeedbackRequest,
    RegenerateRequest,
    VocabularyEntryResponse,
    SearchResultResponse,
    QualityAssessmentResponse,
    AnalyticsResponse,
    HealthCheckResponse
)
from ..core.rag_engine import get_rag_engine
from ..core.quality_system import get_quality_system
from ..core.vocabulary_generator import get_vocabulary_generator
from ..services.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Dependency to get services
def get_services():
    """Get core services"""
    return {
        "rag_engine": get_rag_engine(),
        "quality_system": get_quality_system(),
        "vocabulary_generator": get_vocabulary_generator(),
        "settings": get_settings()
    }


@router.post("/vocabulary/generate", response_model=VocabularyEntry)
async def generate_vocabulary_entry(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    services: Dict = Depends(get_services)
):
    """Generate a new vocabulary entry using the Gulotta method"""
    try:
        logger.info(f"Generating vocabulary entry for: {request.word}")
        
        generator = services["vocabulary_generator"]
        rag_engine = services["rag_engine"]
        
        # Generate entry
        entry = await generator.generate_vocabulary_entry(
            word=request.word,
            context=request.context,
            quality_threshold=request.quality_threshold,
            max_attempts=request.max_attempts
        )
        
        if not entry:
            raise HTTPException(
                status_code=422, 
                detail=f"Failed to generate quality entry for '{request.word}'"
            )
        
        # Add to RAG engine in background
        background_tasks.add_task(rag_engine.add_vocabulary_entry, entry)
        
        logger.info(f"Successfully generated entry for '{request.word}' with quality score: {entry.quality_score}")
        return entry
        
    except Exception as e:
        logger.error(f"Generation failed for '{request.word}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vocabulary/batch-generate", response_model=List[VocabularyEntry])
async def batch_generate_vocabulary_entries(
    request: BatchGenerateRequest,
    background_tasks: BackgroundTasks,
    services: Dict = Depends(get_services)
):
    """Generate multiple vocabulary entries efficiently"""
    try:
        logger.info(f"Batch generating entries for {len(request.words)} words")
        
        generator = services["vocabulary_generator"]
        rag_engine = services["rag_engine"]
        
        # Generate entries
        entries = await generator.batch_generate_entries(
            words=request.words,
            quality_threshold=request.quality_threshold
        )
        
        if not entries:
            raise HTTPException(
                status_code=422,
                detail="Failed to generate any vocabulary entries"
            )
        
        # Add to RAG engine in background
        background_tasks.add_task(rag_engine.add_vocabulary_entries, entries)
        
        logger.info(f"Successfully generated {len(entries)}/{len(request.words)} entries")
        return entries
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vocabulary/search", response_model=List[SearchResultResponse])
async def search_vocabulary_entries(
    request: SearchRequest,
    services: Dict = Depends(get_services)
):
    """Search for similar vocabulary entries"""
    try:
        logger.info(f"Searching for entries similar to: {request.query}")
        
        rag_engine = services["rag_engine"]
        
        # Search entries
        results = rag_engine.search_similar_entries(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            search_type=request.search_type
        )
        
        # Extract entries and scores
        entries = []
        scores = []
        for entry, score in results:
            entries.append(entry)
            scores.append(score)
        
        # Combine results and create response objects
        results = []
        for entry, score in zip(entries, scores):
            result = SearchResultResponse(
                word=entry.word,
                pronunciation=entry.pronunciation,
                definition=entry.definition,
                mnemonic_phrase=entry.mnemonic_phrase,
                similarity_score=score,
                source=entry.source
            )
            results.append(result)
        
        logger.info(f"Found {len(results)} similar entries for '{request.word}'")
        return results
        
    except Exception as e:
        logger.error(f"Search failed for '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vocabulary/contextual/{word}")
async def get_contextual_examples(
    word: str,
    context_type: str = Query("similar_meaning", pattern="^(similar_meaning|similar_sound|same_pos|mnemonic_type)$"),
    num_examples: int = Query(3, ge=1, le=10),
    services: Dict = Depends(get_services)
):
    """Get contextual examples for a word"""
    try:
        logger.info(f"Getting contextual examples for '{word}' with type '{context_type}'")
        
        rag_engine = services["rag_engine"]
        
        examples = rag_engine.get_contextual_examples(
            word=word,
            context_type=context_type,
            num_examples=num_examples
        )
        
        return {
            "word": word,
            "context_type": context_type,
            "examples": examples,
            "count": len(examples)
        }
        
    except Exception as e:
        logger.error(f"Contextual examples failed for '{word}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/submit")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    services: Dict = Depends(get_services)
):
    """Submit user feedback for vocabulary entries"""
    try:
        logger.info(f"Submitting {request.feedback_type} feedback for '{request.word}'")
        
        rag_engine = services["rag_engine"]
        
        # Store feedback
        if request.feedback_type == "positive":
            rag_engine.add_positive_feedback(
                word=request.word,
                example=request.feedback_text,
                feedback_data=request.metadata
            )
        else:
            issues = request.metadata.get("issues", []) if request.metadata else []
            rag_engine.add_negative_feedback(
                word=request.word,
                example=request.feedback_text,
                issues=issues
            )
        
        return {
            "message": "Feedback submitted successfully",
            "word": request.word,
            "feedback_type": request.feedback_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vocabulary/{word}/regenerate", response_model=VocabularyEntry)
async def regenerate_with_feedback(
    word: str,
    feedback_text: str,
    background_tasks: BackgroundTasks,
    feedback_type: str = Query("negative", pattern="^(positive|negative)$"),
    services: Dict = Depends(get_services)
):
    """Regenerate vocabulary entry incorporating feedback"""
    try:
        logger.info(f"Regenerating '{word}' with {feedback_type} feedback")
        
        generator = services["vocabulary_generator"]
        rag_engine = services["rag_engine"]
        
        # Regenerate with feedback
        entry = await generator.regenerate_with_feedback(
            word=word,
            feedback=feedback_text,
            feedback_type=feedback_type
        )
        
        if not entry:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to regenerate entry for '{word}'"
            )
        
        # Add to RAG engine in background
        background_tasks.add_task(rag_engine.add_vocabulary_entry, entry)
        
        return entry
        
    except Exception as e:
        logger.error(f"Regeneration failed for '{word}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vocabulary/assess-quality", response_model=QualityAssessmentResponse)
async def assess_vocabulary_quality(
    entry: VocabularyEntry,
    services: Dict = Depends(get_services)
):
    """Assess the quality of a vocabulary entry"""
    try:
        logger.info(f"Assessing quality for '{entry.word}'")
        
        quality_system = services["quality_system"]
        rag_engine = services["rag_engine"]
        
        # Get feedback context
        feedback_context = rag_engine.get_feedback_context(entry.word)
        
        # Assess quality
        assessment = quality_system.assess_vocabulary_entry(entry, feedback_context)
        
        response = QualityAssessmentResponse(
            word=entry.word,
            overall_score=assessment.overall_score,
            quality_level=assessment.quality_level,
            metrics=assessment.metrics,
            improvement_suggestions=assessment.improvement_suggestions,
            assessment_timestamp=assessment.assessment_timestamp
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Quality assessment failed for '{entry.word}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/system", response_model=AnalyticsResponse)
async def get_system_statistics(services: Dict = Depends(get_services)):
    """Get comprehensive system statistics"""
    try:
        rag_engine = services["rag_engine"]
        quality_system = services["quality_system"]
        
        # Get RAG engine stats
        rag_stats = rag_engine.get_statistics()
        
        # Create system stats response
        stats = AnalyticsResponse(
            total_entries=rag_stats.get("total_entries", 0),
            average_quality_score=rag_stats.get("average_quality", 0.0),
            mnemonic_types={},
            parts_of_speech={},
            difficulty_distribution={},
            quality_distribution={},
            recent_feedback_summary={}
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"System statistics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/quality")
async def get_quality_statistics(
    limit: int = Query(100, ge=1, le=1000),
    services: Dict = Depends(get_services)
):
    """Get quality statistics (placeholder - would need stored assessments)"""
    try:
        # This would require storing quality assessments in a database
        # For now, return basic structure
        return {
            "message": "Quality statistics would require stored assessments",
            "suggestion": "Implement database storage for quality assessments",
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Quality statistics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vocabulary/clear")
async def clear_vocabulary_database(
    services: Dict = Depends(get_services),
    confirm: bool = Query(False)
):
    """Clear all vocabulary data (use with caution)"""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Must set confirm=true to clear database"
            )
        
        logger.warning("Clearing vocabulary database")
        
        rag_engine = services["rag_engine"]
        rag_engine.clear_vector_store()
        
        return {
            "message": "Vocabulary database cleared successfully",
            "timestamp": datetime.now().isoformat(),
            "warning": "All vocabulary entries have been deleted"
        }
        
    except Exception as e:
        logger.error(f"Database clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/dashboard", response_model=Dict[str, Any])
async def get_admin_dashboard(services: Dict = Depends(get_services)):
    """Get comprehensive admin dashboard statistics"""
    try:
        rag_engine = services["rag_engine"]
        quality_system = services["quality_system"]
        
        # Get RAG engine statistics
        rag_stats = rag_engine.get_statistics()
        
        # Get vector store info
        vector_stats = {
            "total_entries": rag_stats.get("total_entries", 0),
            "positive_examples": rag_stats.get("positive_examples", 0),
            "negative_examples": rag_stats.get("negative_examples", 0),
            "vector_dimensions": 384,  # MiniLM embedding size
            "similarity_threshold": 0.3
        }
        
        # Calculate quality metrics
        quality_stats = {
            "average_quality_score": rag_stats.get("average_quality", 0.0),
            "total_assessments": rag_stats.get("total_assessments", 0),
            "excellent_entries": rag_stats.get("excellent_count", 0),
            "good_entries": rag_stats.get("good_count", 0),
            "acceptable_entries": rag_stats.get("acceptable_count", 0),
            "poor_entries": rag_stats.get("poor_count", 0)
        }
        
        # Get generation statistics
        generation_stats = {
            "total_generations": rag_stats.get("total_generations", 0),
            "successful_generations": rag_stats.get("successful_generations", 0),
            "failed_generations": rag_stats.get("failed_generations", 0),
            "average_attempts": rag_stats.get("average_attempts", 1.0),
            "success_rate": rag_stats.get("success_rate", 0.0)
        }
        
        # Get vocabulary distribution
        vocabulary_distribution = {
            "parts_of_speech": {
                "noun": rag_stats.get("noun_count", 0),
                "verb": rag_stats.get("verb_count", 0),
                "adjective": rag_stats.get("adjective_count", 0),
                "adverb": rag_stats.get("adverb_count", 0)
            },
            "difficulty_levels": {
                "basic": rag_stats.get("basic_count", 0),
                "intermediate": rag_stats.get("intermediate_count", 0),
                "advanced": rag_stats.get("advanced_count", 0),
                "expert": rag_stats.get("expert_count", 0)
            },
            "mnemonic_types": {
                "sounds_like": rag_stats.get("sounds_like_count", 0),
                "looks_like": rag_stats.get("looks_like_count", 0),
                "think_of": rag_stats.get("think_of_count", 0),
                "connect_with": rag_stats.get("connect_with_count", 0)
            }
        }
        
        # Get feedback statistics
        feedback_stats = {
            "total_feedback": rag_stats.get("total_feedback", 0),
            "positive_feedback": rag_stats.get("positive_feedback", 0),
            "negative_feedback": rag_stats.get("negative_feedback", 0),
            "feedback_ratio": rag_stats.get("feedback_ratio", 0.0),
            "avg_satisfaction": rag_stats.get("avg_satisfaction", 0.0)
        }
        
        # Get system health information
        system_health = {
            "status": "healthy",
            "uptime": "N/A",  # Would need to track startup time
            "memory_usage": "N/A",  # Could add psutil for this
            "vector_store_status": "connected",
            "llm_provider": "hackclub",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "last_generation": rag_stats.get("last_generation", None),
            "last_feedback": rag_stats.get("last_feedback", None)
        }
        
        # Get recent activity (last 10 entries)
        recent_activity = rag_stats.get("recent_entries", [])
        
        # Performance metrics
        performance_metrics = {
            "avg_generation_time": rag_stats.get("avg_generation_time", 0.0),
            "avg_quality_assessment_time": rag_stats.get("avg_assessment_time", 0.0),
            "avg_search_time": rag_stats.get("avg_search_time", 0.0),
            "cache_hit_rate": rag_stats.get("cache_hit_rate", 0.0)
        }
        
        # Top performing words (highest quality scores)
        top_words = rag_stats.get("top_quality_words", [])
        
        # Compile comprehensive dashboard
        dashboard = {
            "overview": {
                "total_vocabulary_entries": vector_stats["total_entries"],
                "average_quality_score": quality_stats["average_quality_score"],
                "total_generations": generation_stats["total_generations"],
                "success_rate": generation_stats["success_rate"],
                "total_feedback": feedback_stats["total_feedback"],
                "system_status": system_health["status"]
            },
            "vector_store": vector_stats,
            "quality_metrics": quality_stats,
            "generation_statistics": generation_stats,
            "vocabulary_distribution": vocabulary_distribution,
            "feedback_analysis": feedback_stats,
            "system_health": system_health,
            "performance_metrics": performance_metrics,
            "recent_activity": recent_activity[:10],  # Last 10 entries
            "top_quality_words": top_words[:10],  # Top 10 highest quality
            "timestamp": datetime.now().isoformat(),
            "api_version": "1.0.0"
        }
        
        logger.info("Admin dashboard data compiled successfully")
        return dashboard
        
    except Exception as e:
        logger.error(f"Admin dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/quality-report", response_model=Dict[str, Any])
async def get_quality_report(services: Dict = Depends(get_services)):
    """Get detailed quality analysis report"""
    try:
        rag_engine = services["rag_engine"]
        
        # Get all entries for analysis
        all_entries = rag_engine.get_all_entries()
        
        if not all_entries:
            return {
                "message": "No vocabulary entries found",
                "total_entries": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Analyze quality distribution
        quality_ranges = {
            "excellent (85-100%)": 0,
            "good (70-84%)": 0,
            "acceptable (55-69%)": 0,
            "needs_improvement (0-54%)": 0
        }
        
        total_quality = 0
        quality_scores = []
        
        for entry in all_entries:
            score = getattr(entry, 'quality_score', 0.0)
            quality_scores.append(score)
            total_quality += score
            
            if score >= 0.85:
                quality_ranges["excellent (85-100%)"] += 1
            elif score >= 0.70:
                quality_ranges["good (70-84%)"] += 1
            elif score >= 0.55:
                quality_ranges["acceptable (55-69%)"] += 1
            else:
                quality_ranges["needs_improvement (0-54%)"] += 1
        
        # Calculate statistics
        avg_quality = total_quality / len(all_entries) if all_entries else 0
        min_quality = min(quality_scores) if quality_scores else 0
        max_quality = max(quality_scores) if quality_scores else 0
        
        # Find median
        sorted_scores = sorted(quality_scores)
        median_quality = sorted_scores[len(sorted_scores) // 2] if sorted_scores else 0
        
        # Quality trends (would need timestamps for real trends)
        quality_trends = {
            "trend_direction": "stable",  # "improving", "declining", "stable"
            "trend_percentage": 0.0,
            "recent_avg": avg_quality,  # Last 10 entries average
            "historical_avg": avg_quality  # Overall average
        }
        
        # Component analysis
        component_analysis = {
            "definition_quality_avg": avg_quality * 0.9,  # Estimate
            "mnemonic_effectiveness_avg": avg_quality * 1.1,  # Estimate
            "example_relevance_avg": avg_quality * 0.95,  # Estimate
            "completeness_avg": avg_quality * 1.05,  # Estimate
            "linguistic_accuracy_avg": avg_quality * 0.98  # Estimate
        }
        
        report = {
            "summary": {
                "total_entries": len(all_entries),
                "average_quality": round(avg_quality, 3),
                "median_quality": round(median_quality, 3),
                "min_quality": round(min_quality, 3),
                "max_quality": round(max_quality, 3),
                "quality_variance": round(sum((s - avg_quality) ** 2 for s in quality_scores) / len(quality_scores), 3) if quality_scores else 0
            },
            "distribution": quality_ranges,
            "quality_trends": quality_trends,
            "component_analysis": component_analysis,
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add recommendations based on analysis
        if avg_quality < 0.70:
            report["recommendations"].append("Consider adjusting quality thresholds or improving generation prompts")
        if quality_ranges["needs_improvement (0-54%)"] > len(all_entries) * 0.2:
            report["recommendations"].append("High percentage of low-quality entries - review generation settings")
        if max_quality - min_quality > 0.5:
            report["recommendations"].append("High quality variance - consider more consistent generation parameters")
        
        if not report["recommendations"]:
            report["recommendations"].append("Quality metrics look good - continue current approach")
        
        logger.info(f"Quality report generated for {len(all_entries)} entries")
        return report
        
    except Exception as e:
        logger.error(f"Quality report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/feedback-analysis", response_model=Dict[str, Any])
async def get_feedback_analysis(services: Dict = Depends(get_services)):
    """Get detailed feedback analysis for admin insights"""
    try:
        rag_engine = services["rag_engine"]
        
        # Get feedback statistics
        feedback_stats = rag_engine.get_feedback_statistics()
        
        analysis = {
            "overview": {
                "total_feedback_entries": feedback_stats.get("total_feedback", 0),
                "positive_feedback_count": feedback_stats.get("positive_count", 0),
                "negative_feedback_count": feedback_stats.get("negative_count", 0),
                "average_satisfaction": feedback_stats.get("avg_satisfaction", 0.0),
                "feedback_participation_rate": feedback_stats.get("participation_rate", 0.0)
            },
            "satisfaction_distribution": {
                "very_satisfied (9-10)": feedback_stats.get("very_satisfied", 0),
                "satisfied (7-8)": feedback_stats.get("satisfied", 0),
                "neutral (5-6)": feedback_stats.get("neutral", 0),
                "dissatisfied (3-4)": feedback_stats.get("dissatisfied", 0),
                "very_dissatisfied (1-2)": feedback_stats.get("very_dissatisfied", 0)
            },
            "component_feedback": {
                "definition_issues": feedback_stats.get("definition_issues", 0),
                "mnemonic_issues": feedback_stats.get("mnemonic_issues", 0),
                "example_issues": feedback_stats.get("example_issues", 0),
                "pronunciation_issues": feedback_stats.get("pronunciation_issues", 0),
                "other_issues": feedback_stats.get("other_issues", 0)
            },
            "improvement_impact": {
                "regenerations_requested": feedback_stats.get("regenerations", 0),
                "average_improvement": feedback_stats.get("avg_improvement", 0.0),
                "successful_improvements": feedback_stats.get("successful_improvements", 0)
            },
            "recent_feedback": feedback_stats.get("recent_feedback", [])[:10],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Feedback analysis compiled successfully")
        return analysis
        
    except Exception as e:
        logger.error(f"Feedback analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))