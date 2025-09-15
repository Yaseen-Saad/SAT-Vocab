"""
Main FastAPI Application
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.services.config import get_settings, get_config_service
from src.core.rag_engine import get_rag_engine, initialize_rag_engine
from src.core.quality_system import get_quality_system
from src.core.vocabulary_generator import get_vocabulary_generator
from src.api.endpoints import router as api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting SAT Vocabulary RAG System...")
    
    # Initialize services
    config_service = get_config_service()
    rag_engine = get_rag_engine()
    quality_system = get_quality_system()
    vocabulary_generator = get_vocabulary_generator()
    
    logger.info("All services initialized successfully")
    
    # Application startup complete
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down SAT Vocabulary RAG System...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Advanced SAT Vocabulary Learning System with RAG and Quality Assessment",
        docs_url="/docs" if settings.is_development() else None,
        redoc_url="/redoc" if settings.is_development() else None,
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware for production
    if settings.is_production():
        trusted_hosts = ["*"]  # Configure based on your deployment
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            # Check core services
            rag_engine = get_rag_engine()
            quality_system = get_quality_system()
            
            return {
                "status": "healthy",
                "environment": settings.environment,
                "version": settings.app_version,
                "services": {
                    "rag_engine": "operational",
                    "quality_system": "operational",
                    "vocabulary_generator": "operational"
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Service unavailable")
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "SAT Vocabulary RAG System API",
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1"
        }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler"""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload and settings.is_development(),
        log_level=settings.log_level.lower(),
        access_log=True
    )