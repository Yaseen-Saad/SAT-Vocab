"""
Data Models for SAT Vocabulary RAG System
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class MnemonicType(str, Enum):
    """Types of mnemonic devices"""
    SOUNDS_LIKE = "Sounds like"
    LOOKS_LIKE = "Looks like"
    THINK_OF = "Think of"
    CONNECT_WITH = "Connect with"


class PartOfSpeech(str, Enum):
    """Parts of speech"""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    ADJ = "adj"
    ADV = "adv"


class DifficultyLevel(str, Enum):
    """SAT difficulty levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class VocabularyEntry:
    """Core vocabulary entry data structure"""
    word: str
    pronunciation: str
    part_of_speech: str
    definition: str
    mnemonic_type: str
    mnemonic_phrase: str
    picture_story: str
    other_forms: str
    example_sentence: str
    raw_text: Optional[str] = ""
    quality_score: Optional[float] = 0.0
    source: Optional[str] = "generated"
    difficulty_level: Optional[str] = "intermediate"
    frequency_rank: Optional[int] = None
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = field(default_factory=datetime.now)
    id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "word": self.word,
            "pronunciation": self.pronunciation,
            "part_of_speech": self.part_of_speech,
            "definition": self.definition,
            "mnemonic_type": self.mnemonic_type,
            "mnemonic_phrase": self.mnemonic_phrase,
            "picture_story": self.picture_story,
            "other_forms": self.other_forms,
            "example_sentence": self.example_sentence,
            "raw_text": self.raw_text,
            "quality_score": self.quality_score,
            "source": self.source,
            "difficulty_level": self.difficulty_level,
            "frequency_rank": self.frequency_rank,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VocabularyEntry':
        """Create from dictionary"""
        # Handle datetime fields
        created_at = None
        updated_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'])
            
        return cls(
            word=data['word'],
            pronunciation=data.get('pronunciation', ''),
            part_of_speech=data.get('part_of_speech', 'noun'),
            definition=data.get('definition', ''),
            mnemonic_type=data.get('mnemonic_type', 'Sounds like'),
            mnemonic_phrase=data.get('mnemonic_phrase', ''),
            picture_story=data.get('picture_story', ''),
            other_forms=data.get('other_forms', ''),
            example_sentence=data.get('example_sentence', ''),
            raw_text=data.get('raw_text', ''),
            quality_score=data.get('quality_score', 0.0),
            source=data.get('source', 'generated'),
            difficulty_level=data.get('difficulty_level', 'intermediate'),
            frequency_rank=data.get('frequency_rank'),
            created_at=created_at,
            updated_at=updated_at,
            id=data.get('id', str(uuid.uuid4()))
        )


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    word: str
    overall_score: float
    component_scores: Dict[str, float]
    authenticity_score: float
    creativity_score: float
    memorability_score: float
    accuracy_score: float
    completeness_score: float
    format_compliance_score: float
    issues: List[str]
    strengths: List[str]
    suggestions: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class UserFeedback:
    """User feedback data structure"""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    word: str = ""
    entry_id: str = ""
    satisfaction_score: int = 5
    helpful_components: List[str] = field(default_factory=list)
    problematic_components: List[str] = field(default_factory=list)
    user_comments: str = ""
    would_recommend: bool = False
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# Pydantic Models for API

class GenerateRequest(BaseModel):
    """Request model for single word generation"""
    word: str = Field(..., min_length=1, max_length=50, pattern=r'^[a-zA-Z\-]+$')
    use_context: bool = True
    num_context_examples: int = Field(default=3, ge=1, le=10)
    part_of_speech: Optional[PartOfSpeech] = None
    difficulty_level: Optional[DifficultyLevel] = None

    @validator('word')
    def validate_word(cls, v):
        return v.strip().upper()


class BatchGenerateRequest(BaseModel):
    """Request model for batch generation"""
    words: List[str] = Field(..., min_items=1, max_items=10)
    use_context: bool = True
    num_context_examples: int = Field(default=3, ge=1, le=10)
    part_of_speech: Optional[PartOfSpeech] = None
    difficulty_level: Optional[DifficultyLevel] = None

    @validator('words')
    def validate_words(cls, v):
        return [word.strip().upper() for word in v if word.strip()]


class SearchRequest(BaseModel):
    """Request model for similarity search"""
    word: str = Field(..., min_length=1, max_length=50)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    search_type: str = Field(default="semantic", pattern=r'^(semantic|keyword|hybrid)$')


class FeedbackRequest(BaseModel):
    """Request model for user feedback"""
    word: str = Field(..., min_length=1, max_length=50)
    entry_id: str = Field(..., min_length=1, max_length=100)
    satisfaction_score: int = Field(..., ge=1, le=10)
    helpful_components: List[str] = Field(default=[], max_items=10)
    problematic_components: List[str] = Field(default=[], max_items=10)
    user_comments: str = Field(default="", max_length=1000)
    would_recommend: bool = False
    session_id: Optional[str] = None


class RegenerateRequest(BaseModel):
    """Request model for entry regeneration"""
    word: str = Field(..., min_length=1, max_length=50)
    part_of_speech: PartOfSpeech = PartOfSpeech.NOUN
    regeneration_reason: str = Field(..., min_length=1, max_length=500)
    specific_issue: str = Field(default="", max_length=500)
    improvement_suggestions: str = Field(default="", max_length=500)
    use_enhanced_generation: bool = True

    @validator('word')
    def validate_word(cls, v):
        return v.strip().upper()


class VocabularyEntryResponse(BaseModel):
    """Response model for vocabulary entries"""
    word: str
    pronunciation: str
    part_of_speech: str
    definition: str
    mnemonic_type: str
    mnemonic_phrase: str
    picture_story: str
    other_forms: str
    example_sentence: str
    quality_score: float
    validation_passed: bool
    generation_metadata: Dict[str, Any] = {}
    confidence_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class SearchResultResponse(BaseModel):
    """Response model for search results"""
    word: str
    pronunciation: str
    definition: str
    mnemonic_phrase: str
    similarity_score: float
    source: Optional[str] = None


class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment"""
    word: str
    overall_score: float
    component_scores: Dict[str, float]
    grade: str  # A, B, C, D, F
    issues: List[str]
    strengths: List[str]
    suggestions: List[str]
    meets_threshold: bool


class AnalyticsResponse(BaseModel):
    """Response model for analytics"""
    total_entries: int
    average_quality_score: float
    mnemonic_types: Dict[str, int]
    parts_of_speech: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    quality_distribution: Dict[str, int]
    recent_feedback_summary: Dict[str, Any]


class HealthCheckResponse(BaseModel):
    """Response model for health checks"""
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]
    performance_metrics: Dict[str, float]
    database_status: str
    vector_store_status: str


# Configuration Models

class QualityWeights(BaseModel):
    """Quality assessment weights configuration"""
    authenticity: float = 0.25
    creativity: float = 0.20
    memorability: float = 0.20
    accuracy: float = 0.15
    completeness: float = 0.10
    format_compliance: float = 0.10
    
    @validator('*')
    def validate_weight(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Weight must be between 0 and 1')
        return v


class SystemConfig(BaseModel):
    """System configuration model"""
    environment: str = "development"
    debug: bool = True
    api_version: str = "1.0.0"
    quality_threshold: float = 7.0
    cache_enabled: bool = True
    rate_limit_enabled: bool = True
    vector_db_collection: str = "vocabulary_entries"
    embedding_model: str = "all-MiniLM-L6-v2"
    quality_weights: QualityWeights = QualityWeights()