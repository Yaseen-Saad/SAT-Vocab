"""
Configuration Management Service
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import validator


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application settings
    app_name: str = "SAT Vocabulary RAG System"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_cors_origins: str = "*"
    
    # LLM settings
    llm_provider: str = "hackclub"
    llm_api_url: str = "https://ai.hackclub.com"
    llm_model: str = "qwen/qwen3-32b"
    openai_api_key: Optional[str] = "not_needed"
    generation_temperature: float = 0.7
    max_tokens: int = 2000
    max_retries: int = 3
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Vector Database settings
    vector_db_path: str = "./data/vector_store"
    vector_db_collection: str = "sat_vocabulary"
    vector_db_persist: bool = True
    
    # Database settings
    database_url: str = "sqlite:///./data/vocabulary.db"
    database_echo: bool = False
    
    # Storage settings
    data_directory: str = "./data"
    feedback_storage_path: str = "./data/feedback"
    cache_directory: str = "./data/cache"
    
    # Quality settings
    quality_threshold_excellent: float = 0.85
    quality_threshold_good: float = 0.70
    quality_threshold_acceptable: float = 0.55
    default_quality_threshold: float = 0.70
    
    # Generation settings
    max_generation_attempts: int = 3
    batch_generation_size: int = 10
    concurrent_generation_limit: int = 3
    
    # Feedback settings
    feedback_learning_enabled: bool = True
    feedback_weight_positive: float = 0.1
    feedback_weight_negative: float = 0.15
    min_feedback_examples: int = 3
    
    # Cache settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Deployment settings
    use_https: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # Monitoring settings
    metrics_enabled: bool = True
    metrics_port: int = 9090
    health_check_enabled: bool = True
    
    # PDF processing settings
    pdf_processing_enabled: bool = True
    pdf_extraction_timeout: int = 300
    pdf_max_pages: int = 500
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("openai_api_key")
    def validate_api_key(cls, v, values):
        """Validate API key based on provider"""
        provider = values.get("llm_provider", "hackclub")
        if provider == "openai" and not v:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        elif provider == "hackclub":
            return "not_needed"  # Hack Club AI doesn't need API key
        return v
    
    @validator("api_cors_origins")
    def validate_cors_origins(cls, v):
        """Parse CORS origins"""
        if v == "*":
            return ["*"]
        return [origin.strip() for origin in v.split(",") if origin.strip()]
    
    @validator("vector_db_path", "data_directory", "feedback_storage_path", "cache_directory")
    def create_directories(cls, v):
        """Ensure directories exist"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")
        return v.upper()
    
    def get_database_url(self, async_driver: bool = False) -> str:
        """Get database URL with optional async driver"""
        if self.database_url.startswith("sqlite"):
            if async_driver:
                return self.database_url.replace("sqlite://", "sqlite+aiosqlite://")
        return self.database_url
    
    def get_cors_origins(self) -> list:
        """Get CORS origins as list"""
        if isinstance(self.api_cors_origins, str):
            return self.validate_cors_origins(self.api_cors_origins)
        return self.api_cors_origins
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"


class ConfigurationService:
    """Service for managing application configuration"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._setup_logging()
        self._validate_configuration()
        
        logger.info(f"Configuration service initialized for environment: {self.settings.environment}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            # Configure root logger
            log_level = getattr(logging, self.settings.log_level)
            
            # Create formatter
            formatter = logging.Formatter(self.settings.log_format)
            
            # Setup handlers
            handlers = []
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            handlers.append(console_handler)
            
            # File handler if specified
            if self.settings.log_file:
                file_handler = logging.FileHandler(self.settings.log_file)
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)
            
            # Configure root logger
            logging.basicConfig(
                level=log_level,
                handlers=handlers,
                force=True
            )
            
            # Set specific logger levels
            logging.getLogger("uvicorn").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            
            logger.info(f"Logging configured: level={self.settings.log_level}")
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        try:
            # Validate required directories
            required_dirs = [
                self.settings.data_directory,
                self.settings.vector_db_path,
                self.settings.feedback_storage_path,
                self.settings.cache_directory
            ]
            
            for directory in required_dirs:
                path = Path(directory)
                if not path.exists():
                    logger.warning(f"Creating missing directory: {directory}")
                    path.mkdir(parents=True, exist_ok=True)
            
            # Validate LLM configuration
            if self.settings.llm_provider == "openai":
                if not self.settings.openai_api_key or self.settings.openai_api_key == "not_needed":
                    raise ValueError("OpenAI API key is required for OpenAI provider")
            elif self.settings.llm_provider == "hackclub":
                logger.info("Using Hack Club AI - no API key required!")
            else:
                logger.warning(f"Unknown LLM provider: {self.settings.llm_provider}")
            
            # Validate quality thresholds
            thresholds = [
                self.settings.quality_threshold_excellent,
                self.settings.quality_threshold_good,
                self.settings.quality_threshold_acceptable
            ]
            
            if not all(0 <= t <= 1 for t in thresholds):
                raise ValueError("Quality thresholds must be between 0 and 1")
            
            if not (thresholds[0] > thresholds[1] > thresholds[2]):
                raise ValueError("Quality thresholds must be in descending order")
            
            logger.info("Configuration validation completed successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "provider": self.settings.llm_provider,
            "api_url": getattr(self.settings, 'llm_api_url', 'https://ai.hackclub.com'),
            "model": self.settings.llm_model,
            "api_key": self.settings.openai_api_key,
            "temperature": self.settings.generation_temperature,
            "max_tokens": self.settings.max_tokens,
            "max_retries": self.settings.max_retries
        }
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration"""
        return {
            "path": self.settings.vector_db_path,
            "collection": self.settings.vector_db_collection,
            "persist": self.settings.vector_db_persist,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimension": self.settings.embedding_dimension
        }
    
    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality assessment configuration"""
        return {
            "threshold_excellent": self.settings.quality_threshold_excellent,
            "threshold_good": self.settings.quality_threshold_good,
            "threshold_acceptable": self.settings.quality_threshold_acceptable,
            "default_threshold": self.settings.default_quality_threshold
        }
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation configuration"""
        return {
            "max_attempts": self.settings.max_generation_attempts,
            "batch_size": self.settings.batch_generation_size,
            "concurrent_limit": self.settings.concurrent_generation_limit,
            "quality_threshold": self.settings.default_quality_threshold
        }
    
    def get_feedback_config(self) -> Dict[str, Any]:
        """Get feedback configuration"""
        return {
            "learning_enabled": self.settings.feedback_learning_enabled,
            "weight_positive": self.settings.feedback_weight_positive,
            "weight_negative": self.settings.feedback_weight_negative,
            "min_examples": self.settings.min_feedback_examples,
            "storage_path": self.settings.feedback_storage_path
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return {
            "host": self.settings.api_host,
            "port": self.settings.api_port,
            "reload": self.settings.api_reload,
            "cors_origins": self.settings.get_cors_origins(),
            "use_https": self.settings.use_https,
            "ssl_cert_path": self.settings.ssl_cert_path,
            "ssl_key_path": self.settings.ssl_key_path
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            "metrics_enabled": self.settings.metrics_enabled,
            "metrics_port": self.settings.metrics_port,
            "health_check_enabled": self.settings.health_check_enabled
        }
    
    def update_setting(self, key: str, value: Any) -> bool:
        """Update a setting dynamically"""
        try:
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
                logger.info(f"Updated setting {key} = {value}")
                return True
            else:
                logger.warning(f"Setting {key} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update setting {key}: {e}")
            return False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information"""
        return {
            "app_name": self.settings.app_name,
            "app_version": self.settings.app_version,
            "environment": self.settings.environment,
            "debug": self.settings.debug,
            "python_version": os.sys.version,
            "working_directory": os.getcwd()
        }
    
    def export_settings(self) -> Dict[str, Any]:
        """Export all settings (excluding sensitive data)"""
        settings_dict = self.settings.dict()
        
        # Remove sensitive information
        sensitive_keys = ["openai_api_key", "database_url"]
        for key in sensitive_keys:
            if key in settings_dict:
                settings_dict[key] = "***hidden***"
        
        return settings_dict


# Global settings instance
@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)"""
    return Settings()


# Global configuration service instance
_config_service_instance = None


def get_config_service() -> ConfigurationService:
    """Get or create configuration service instance"""
    global _config_service_instance
    if _config_service_instance is None:
        _config_service_instance = ConfigurationService()
    return _config_service_instance


def initialize_config_service(settings: Optional[Settings] = None) -> ConfigurationService:
    """Initialize configuration service with custom settings"""
    global _config_service_instance
    _config_service_instance = ConfigurationService(settings)
    return _config_service_instance