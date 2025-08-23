"""
Enhanced Quality Assessment and Feedback System
Provides detailed quality metrics and user satisfaction tracking for continuous improvement
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from .vocabulary_types import GeneratedVocabularyEntry, QualityMetrics, UserFeedback, FeedbackAnalytics

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    """Data structure for model improvement"""
    word: str
    generated_entry: str
    context_examples: List[str]
    quality_metrics: QualityMetrics
    user_feedback: Optional[UserFeedback]
    improvement_suggestions: List[str]
    timestamp: str


class EnhancedQualityChecker:
    """Advanced quality assessment system for vocabulary entries"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.db_path = self.config.get('feedback_db_path', 'feedback.db')
        self._init_database()
        
        # Quality thresholds
        self.excellence_threshold = 8.5
        self.good_threshold = 7.0
        self.acceptable_threshold = 5.5
        
        # Component weights
        self.weights = {
            'authenticity': 0.25,
            'creativity': 0.20,
            'memorability': 0.20,
            'accuracy': 0.15,
            'completeness': 0.10,
            'format_compliance': 0.10
        }
    
    def _init_database(self):
        """Initialize SQLite database for storing feedback and metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        word TEXT NOT NULL,
                        overall_score REAL,
                        component_scores TEXT,
                        authenticity_score REAL,
                        creativity_score REAL,
                        memorability_score REAL,
                        accuracy_score REAL,
                        completeness_score REAL,
                        format_compliance_score REAL,
                        issues TEXT,
                        strengths TEXT,
                        suggestions TEXT,
                        timestamp TEXT,
                        entry_text TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        word TEXT NOT NULL,
                        entry_id TEXT,
                        satisfaction_score INTEGER,
                        helpful_components TEXT,
                        problematic_components TEXT,
                        user_comments TEXT,
                        would_recommend BOOLEAN,
                        timestamp TEXT,
                        user_id TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        word TEXT NOT NULL,
                        generated_entry TEXT,
                        context_examples TEXT,
                        quality_score REAL,
                        user_satisfaction REAL,
                        improvement_suggestions TEXT,
                        timestamp TEXT
                    )
                ''')
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def assess_quality(self, entry: GeneratedVocabularyEntry, context_examples: List[str] = None) -> QualityMetrics:
        """Perform comprehensive quality assessment"""
        
        # Individual component assessments
        authenticity_score = self._assess_authenticity(entry, context_examples or [])
        creativity_score = self._assess_creativity(entry)
        memorability_score = self._assess_memorability(entry)
        accuracy_score = self._assess_accuracy(entry)
        completeness_score = self._assess_completeness(entry)
        format_compliance_score = self._assess_format_compliance(entry)
        
        # Component scores dictionary
        component_scores = {
            'authenticity': authenticity_score,
            'creativity': creativity_score,
            'memorability': memorability_score,
            'accuracy': accuracy_score,
            'completeness': completeness_score,
            'format_compliance': format_compliance_score
        }
        
        # Calculate overall score
        overall_score = sum(
            score * self.weights[component] 
            for component, score in component_scores.items()
        )
        
        # Identify issues, strengths, and suggestions
        issues = self._identify_issues(entry, component_scores)
        strengths = self._identify_strengths(entry, component_scores)
        suggestions = self._generate_suggestions(entry, component_scores, issues)
        
        metrics = QualityMetrics(
            word=entry.word,
            overall_score=overall_score,
            component_scores=component_scores,
            authenticity_score=authenticity_score,
            creativity_score=creativity_score,
            memorability_score=memorability_score,
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            format_compliance_score=format_compliance_score,
            issues=issues,
            strengths=strengths,
            suggestions=suggestions,
            timestamp=datetime.now().isoformat()
        )
        
        # Store metrics in database
        self._store_quality_metrics(metrics, "")  # No generated_text field in new structure
        
        return metrics
    
    def _assess_authenticity(self, entry: GeneratedVocabularyEntry, context_examples: List[str]) -> float:
        """Assess how authentic the entry is to Gulotta's style"""
        score = 10.0
        
        # Check for Gulotta-style mnemonic patterns
        if not entry.mnemonic_phrase:
            score -= 3.0
        elif entry.mnemonic_type and entry.mnemonic_type.lower() in ['sounds like', 'looks like']:
            score += 0.5
        
        # Check for vivid picture story characteristics
        if entry.picture_story:
            picture_words = entry.picture_story.split()
            if len(picture_words) < 15:
                score -= 1.5
            elif len(picture_words) > 60:
                score -= 1.0
            
            # Look for storytelling elements
            storytelling_indicators = ['imagine', 'picture', 'story', 'character', 'scene', 'visual']
            if any(word in entry.picture_story.lower() for word in storytelling_indicators):
                score += 0.5
        else:
            score -= 2.0
        
        # Check pronunciation format
        if entry.pronunciation and '-' in entry.pronunciation:
            score += 0.5
        
        return max(0, min(10, score))
    
    def _assess_creativity(self, entry: GeneratedVocabularyEntry) -> float:
        """Assess creativity of mnemonic and picture story"""
        score = 10.0
        
        # Assess mnemonic creativity
        if entry.mnemonic_phrase:
            mnemonic_lower = entry.mnemonic_phrase.lower()
            word_lower = entry.word.lower()
            
            # Check for actual sound similarity
            if any(syllable in mnemonic_lower for syllable in [word_lower[:3], word_lower[-3:]]):
                score += 1.0
            
            # Check for creative connections
            creative_words = ['like', 'imagine', 'think', 'picture', 'connect']
            if any(word in mnemonic_lower for word in creative_words):
                score += 0.5
        else:
            score -= 3.0
        
        # Assess picture story creativity
        if entry.picture_story:
            story_lower = entry.picture_story.lower()
            creative_elements = ['character', 'scene', 'action', 'dialogue', 'metaphor', 'vivid']
            creativity_bonus = sum(0.3 for element in creative_elements if element in story_lower)
            score += min(creativity_bonus, 2.0)
        else:
            score -= 2.0
        
        return max(0, min(10, score))
    
    def _assess_memorability(self, entry: GeneratedVocabularyEntry) -> float:
        """Assess how memorable the entry is likely to be"""
        score = 10.0
        
        # Check for memorable mnemonic
        if entry.mnemonic_phrase:
            # Short, catchy mnemonics are more memorable
            words_count = len(entry.mnemonic_phrase.split())
            if 2 <= words_count <= 5:
                score += 1.0
            elif words_count > 8:
                score -= 1.0
        else:
            score -= 2.5
        
        # Check for memorable picture elements
        if entry.picture_story:
            memorable_elements = ['visual', 'action', 'emotion', 'surprise', 'funny', 'dramatic']
            if any(element in entry.picture_story.lower() for element in memorable_elements):
                score += 1.0
            
            # Stories with dialogue are more memorable
            if '"' in entry.picture_story:
                score += 0.5
        else:
            score -= 2.0
        
        # Good example sentences help memorability
        if entry.example_sentence and entry.word.lower() in entry.example_sentence.lower():
            score += 0.5
        
        return max(0, min(10, score))
    
    def _assess_accuracy(self, entry: GeneratedVocabularyEntry) -> float:
        """Assess factual accuracy and proper usage"""
        score = 10.0
        
        # Check if example sentence properly uses the word
        if entry.example_sentence:
            if entry.word.lower() not in entry.example_sentence.lower():
                score -= 2.0
        else:
            score -= 1.5
        
        # Check pronunciation format
        if entry.pronunciation:
            if not any(char in entry.pronunciation for char in ['(', ')', '-']):
                score -= 1.0
        else:
            score -= 1.0
        
        # Check part of speech
        valid_pos = ['noun', 'verb', 'adj', 'adjective', 'adverb', 'adv']
        if entry.part_of_speech and entry.part_of_speech.lower() not in valid_pos:
            score -= 0.5
        elif not entry.part_of_speech:
            score -= 1.0
        
        return max(0, min(10, score))
    
    def _assess_completeness(self, entry: GeneratedVocabularyEntry) -> float:
        """Assess completeness of all required components"""
        score = 10.0
        
        required_fields = {
            'word': 2.0,
            'pronunciation': 1.5,
            'part_of_speech': 1.0,
            'definition': 2.0,
            'mnemonic_phrase': 1.5,
            'picture_story': 1.5,
            'example_sentence': 0.5
        }
        
        for field, penalty in required_fields.items():
            if not getattr(entry, field, '').strip():
                score -= penalty
        
        return max(0, min(10, score))
    
    def _assess_format_compliance(self, entry: GeneratedVocabularyEntry) -> float:
        """Assess compliance with Gulotta format standards"""
        score = 10.0
        
        # Check if word is in uppercase
        if entry.word and not entry.word.isupper():
            score -= 1.0
        
        # Check pronunciation format
        if entry.pronunciation and not ('(' in entry.pronunciation or '-' in entry.pronunciation):
            score -= 1.0
        
        # Check mnemonic type format
        if entry.mnemonic_type and entry.mnemonic_type not in ['Sounds like', 'Looks like', 'Think of', 'Connect with']:
            score -= 0.5
        
        return max(0, min(10, score))
    
    def _identify_issues(self, entry: GeneratedVocabularyEntry, component_scores: Dict[str, float]) -> List[str]:
        """Identify specific issues with the entry"""
        issues = []
        
        # Component-specific issues
        if component_scores['authenticity'] < 6:
            issues.append("Entry doesn't match authentic Gulotta style")
        
        if component_scores['creativity'] < 6:
            issues.append("Mnemonic or picture story lacks creativity")
        
        if component_scores['memorability'] < 6:
            issues.append("Entry may not be memorable enough for students")
        
        if component_scores['accuracy'] < 7:
            issues.append("Potential accuracy issues with word usage or pronunciation")
        
        if component_scores['completeness'] < 7:
            issues.append("Missing required components")
        
        if component_scores['format_compliance'] < 8:
            issues.append("Format doesn't comply with Gulotta standards")
        
        # Specific field issues
        if not entry.mnemonic_phrase:
            issues.append("Missing mnemonic device")
        
        if not entry.picture_story or len(entry.picture_story.split()) < 10:
            issues.append("Picture story too short or missing")
        
        if entry.example_sentence and entry.word.lower() not in entry.example_sentence.lower():
            issues.append("Example sentence doesn't use the target word")
        
        return issues
    
    def _identify_strengths(self, entry: GeneratedVocabularyEntry, component_scores: Dict[str, float]) -> List[str]:
        """Identify strengths of the entry"""
        strengths = []
        
        if component_scores['authenticity'] >= 8:
            strengths.append("Excellent authentic Gulotta style")
        
        if component_scores['creativity'] >= 8:
            strengths.append("Highly creative mnemonic and picture story")
        
        if component_scores['memorability'] >= 8:
            strengths.append("Very memorable for students")
        
        if entry.picture_story and len(entry.picture_story.split()) >= 20:
            strengths.append("Rich, detailed picture story")
        
        if entry.mnemonic_phrase and any(syllable in entry.mnemonic_phrase.lower() for syllable in [entry.word.lower()[:3], entry.word.lower()[-3:]]):
            strengths.append("Good phonetic connection in mnemonic")
        
        if entry.other_forms:
            strengths.append("Includes helpful word forms")
        
        return strengths
    
    def _generate_suggestions(self, entry: GeneratedVocabularyEntry, component_scores: Dict[str, float], issues: List[str]) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        if component_scores['creativity'] < 7:
            suggestions.append("Try a more creative 'sounds like' phrase that clearly connects to the word's sound")
        
        if not entry.picture_story or len(entry.picture_story.split()) < 15:
            suggestions.append("Expand the picture story with more vivid details and specific characters")
        
        if component_scores['memorability'] < 7:
            suggestions.append("Add more emotional or visual elements to make the story more memorable")
        
        if not entry.other_forms:
            suggestions.append("Consider adding related word forms (noun, verb, adjective, adverb)")
        
        if component_scores['format_compliance'] < 8:
            suggestions.append("Ensure format matches: WORD (pronunciation) part — definition")
        
        return suggestions
    
    def _store_quality_metrics(self, metrics: QualityMetrics, entry_text: str):
        """Store quality metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO quality_metrics (
                        word, overall_score, component_scores, authenticity_score,
                        creativity_score, memorability_score, accuracy_score,
                        completeness_score, format_compliance_score, issues,
                        strengths, suggestions, timestamp, entry_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.word,
                    metrics.overall_score,
                    json.dumps(metrics.component_scores),
                    metrics.authenticity_score,
                    metrics.creativity_score,
                    metrics.memorability_score,
                    metrics.accuracy_score,
                    metrics.completeness_score,
                    metrics.format_compliance_score,
                    json.dumps(metrics.issues),
                    json.dumps(metrics.strengths),
                    json.dumps(metrics.suggestions),
                    metrics.timestamp,
                    entry_text
                ))
        except Exception as e:
            logger.error(f"Failed to store quality metrics: {e}")
    
    def record_user_feedback(self, feedback: UserFeedback):
        """Record user satisfaction feedback"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO user_feedback (
                        word, entry_id, satisfaction_score, helpful_components,
                        problematic_components, user_comments, would_recommend,
                        timestamp, user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feedback.word,
                    feedback.entry_id,
                    feedback.satisfaction_score,
                    json.dumps(feedback.helpful_components),
                    json.dumps(feedback.problematic_components),
                    feedback.user_comments,
                    feedback.would_recommend,
                    feedback.timestamp,
                    feedback.user_id
                ))
        except Exception as e:
            logger.error(f"Failed to store user feedback: {e}")
    
    def get_quality_analytics(self) -> Dict[str, Any]:
        """Get quality analytics and trends"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Overall statistics
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_entries,
                        AVG(overall_score) as avg_quality,
                        AVG(authenticity_score) as avg_authenticity,
                        AVG(creativity_score) as avg_creativity,
                        AVG(memorability_score) as avg_memorability
                    FROM quality_metrics
                ''')
                stats = cursor.fetchone()
                
                # User satisfaction statistics
                cursor = conn.execute('''
                    SELECT 
                        COUNT(*) as total_feedback,
                        AVG(satisfaction_score) as avg_satisfaction,
                        COUNT(CASE WHEN would_recommend = 1 THEN 1 END) as would_recommend_count
                    FROM user_feedback
                ''')
                feedback_stats = cursor.fetchone()
                
                # Top issues
                cursor = conn.execute('''
                    SELECT word, issues, overall_score 
                    FROM quality_metrics 
                    WHERE overall_score < 6 
                    ORDER BY overall_score ASC 
                    LIMIT 10
                ''')
                low_quality_entries = cursor.fetchall()
                
                return {
                    'total_entries': stats[0] or 0,
                    'average_quality_score': stats[1] or 0,
                    'average_authenticity': stats[2] or 0,
                    'average_creativity': stats[3] or 0,
                    'average_memorability': stats[4] or 0,
                    'total_user_feedback': feedback_stats[0] or 0,
                    'average_user_satisfaction': feedback_stats[1] or 0,
                    'recommendation_rate': (feedback_stats[2] or 0) / max(feedback_stats[0] or 1, 1) * 100,
                    'low_quality_entries': low_quality_entries
                }
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}
    
    def get_training_data(self, min_quality_score: float = 7.0, min_satisfaction: int = 7) -> List[TrainingData]:
        """Get high-quality entries for training data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT DISTINCT 
                        qm.word, qm.entry_text, qm.overall_score,
                        qm.timestamp, qm.suggestions,
                        uf.satisfaction_score, uf.user_comments
                    FROM quality_metrics qm
                    LEFT JOIN user_feedback uf ON qm.word = uf.word
                    WHERE qm.overall_score >= ? 
                    AND (uf.satisfaction_score >= ? OR uf.satisfaction_score IS NULL)
                    ORDER BY qm.overall_score DESC, uf.satisfaction_score DESC
                ''', (min_quality_score, min_satisfaction))
                
                results = cursor.fetchall()
                training_data = []
                
                for row in results:
                    word, entry_text, quality_score, timestamp, suggestions, satisfaction, comments = row
                    
                    training_entry = TrainingData(
                        word=word,
                        generated_entry=entry_text,
                        context_examples=[],  # Would need to be reconstructed
                        quality_metrics=None,  # Would need to be reconstructed
                        user_feedback=None,  # Would need to be reconstructed
                        improvement_suggestions=json.loads(suggestions) if suggestions else [],
                        timestamp=timestamp
                    )
                    training_data.append(training_entry)
                
                return training_data
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []