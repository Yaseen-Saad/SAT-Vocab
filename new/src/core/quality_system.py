"""
Quality Assessment System for SAT Vocabulary Generation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import statistics
import re

from ..models import VocabularyEntry, QualityMetrics
from ..services.config import get_settings

logger = logging.getLogger(__name__)


class QualityAssessmentSystem:
    """Advanced quality assessment for vocabulary entries"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        
        # Quality thresholds
        self.thresholds = {
            "excellent": 0.85,
            "good": 0.70,
            "acceptable": 0.55,
            "needs_improvement": 0.40
        }
        
        # Quality weights for different aspects
        self.weights = {
            "definition_quality": 0.25,
            "mnemonic_effectiveness": 0.25,
            "example_relevance": 0.20,
            "content_completeness": 0.15,
            "linguistic_accuracy": 0.15
        }
        
        # Expected vocabulary structure from Gulotta method
        self.expected_fields = [
            'word', 'pronunciation', 'part_of_speech', 'definition',
            'mnemonic_type', 'mnemonic_phrase', 'picture_story',
            'other_forms', 'example_sentence'
        ]
        
        logger.info("Quality Assessment System initialized")
    
    def assess_vocabulary_entry(self, entry: VocabularyEntry, feedback_context: Dict = None) -> QualityMetrics:
        """Comprehensive quality assessment of a vocabulary entry"""
        try:
            # Individual quality scores
            definition_score = self._assess_definition_quality(entry)
            mnemonic_score = self._assess_mnemonic_effectiveness(entry)
            example_score = self._assess_example_relevance(entry)
            completeness_score = self._assess_content_completeness(entry)
            linguistic_score = self._assess_linguistic_accuracy(entry)
            
            # Calculate weighted overall score
            overall_score = (
                definition_score * self.weights["definition_quality"] +
                mnemonic_score * self.weights["mnemonic_effectiveness"] +
                example_score * self.weights["example_relevance"] +
                completeness_score * self.weights["content_completeness"] +
                linguistic_score * self.weights["linguistic_accuracy"]
            )
            
            # Incorporate feedback if available
            if feedback_context:
                overall_score = self._adjust_score_with_feedback(overall_score, feedback_context)
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Create component scores dictionary
            component_scores = {
                "definition_quality": definition_score,
                "mnemonic_effectiveness": mnemonic_score,
                "example_relevance": example_score,
                "content_completeness": completeness_score,
                "linguistic_accuracy": linguistic_score
            }
            
            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(
                entry, definition_score, mnemonic_score, example_score,
                completeness_score, linguistic_score
            )
            
            # Create quality assessment
            assessment = QualityMetrics(
                word=entry.word,
                overall_score=overall_score,
                component_scores=component_scores,
                authenticity_score=definition_score,
                creativity_score=mnemonic_score,
                memorability_score=mnemonic_score,
                accuracy_score=linguistic_score,
                completeness_score=completeness_score,
                format_compliance_score=0.8,  # Default value
                issues=[],
                strengths=[],
                suggestions=suggestions,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Quality assessment completed for '{entry.word}': {quality_level} ({overall_score:.3f})")
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed for '{entry.word}': {e}")
            # Return minimal assessment
            return QualityMetrics(
                word=entry.word,
                overall_score=0.0,
                component_scores={},
                authenticity_score=0.0,
                creativity_score=0.0,
                memorability_score=0.0,
                accuracy_score=0.0,
                completeness_score=0.0,
                format_compliance_score=0.0,
                issues=["Quality assessment failed"],
                strengths=[],
                suggestions=["Quality assessment failed"],
                timestamp=datetime.now().isoformat()
            )
    
    def _assess_definition_quality(self, entry: VocabularyEntry) -> float:
        """Assess the quality of the definition"""
        try:
            score = 0.0
            definition = entry.definition.strip()
            
            if not definition:
                return 0.0
            
            # Length check (20-200 characters is ideal)
            length = len(definition)
            if 20 <= length <= 200:
                score += 0.3
            elif 10 <= length < 20 or 200 < length <= 300:
                score += 0.15
            
            # Clarity indicators
            clarity_indicators = [
                definition.startswith(('a ', 'an ', 'the ')),  # Proper article usage
                ';' not in definition or definition.count(';') <= 2,  # Not overly complex
                definition.endswith('.'),  # Proper punctuation
                not definition.lower().startswith(entry.word.lower()),  # No circular definition
                len(definition.split()) >= 3  # At least 3 words
            ]
            score += sum(clarity_indicators) * 0.1
            
            # Avoid common definition problems
            problems = [
                definition.lower().count(entry.word.lower()) > 1,  # Repetitive
                definition.count('(') != definition.count(')'),  # Unbalanced parentheses
                len([w for w in definition.split() if len(w) > 15]) > 2,  # Too many long words
                definition.lower().startswith('definition of')  # Meta-definition
            ]
            score -= sum(problems) * 0.15
            
            # Vocabulary appropriateness (SAT level)
            sophisticated_words = ['sophisticated', 'complex', 'intricate', 'nuanced', 'elaborate']
            if any(word in definition.lower() for word in sophisticated_words):
                score += 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Definition quality assessment failed: {e}")
            return 0.0
    
    def _assess_mnemonic_effectiveness(self, entry: VocabularyEntry) -> float:
        """Assess the effectiveness of the mnemonic device"""
        try:
            score = 0.0
            mnemonic_phrase = entry.mnemonic_phrase.strip()
            picture_story = entry.picture_story.strip()
            mnemonic_type = entry.mnemonic_type.lower()
            
            if not mnemonic_phrase:
                return 0.0
            
            # Mnemonic type appropriateness
            type_scores = {
                'sounds like': 0.9,  # Most effective for SAT
                'looks like': 0.8,
                'association': 0.7,
                'story': 0.8,
                'acronym': 0.6,
                'rhyme': 0.7
            }
            score += type_scores.get(mnemonic_type, 0.5) * 0.3
            
            # Mnemonic phrase quality
            if mnemonic_phrase:
                # Length appropriateness (5-50 words)
                word_count = len(mnemonic_phrase.split())
                if 5 <= word_count <= 50:
                    score += 0.2
                elif 3 <= word_count < 5 or 50 < word_count <= 80:
                    score += 0.1
                
                # Contains target word or sound-alike
                word_lower = entry.word.lower()
                mnemonic_lower = mnemonic_phrase.lower()
                if word_lower in mnemonic_lower or any(
                    word_lower.startswith(part) or part.startswith(word_lower[:3])
                    for part in mnemonic_lower.split()
                ):
                    score += 0.2
                
                # Vivid imagery indicators
                imagery_words = ['see', 'picture', 'imagine', 'visual', 'bright', 'dark', 'loud', 'quiet']
                if any(word in mnemonic_lower for word in imagery_words):
                    score += 0.1
            
            # Picture story bonus
            if picture_story:
                if len(picture_story.split()) >= 10:
                    score += 0.2
                if any(sense in picture_story.lower() for sense in ['see', 'hear', 'feel', 'smell', 'taste']):
                    score += 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Mnemonic effectiveness assessment failed: {e}")
            return 0.0
    
    def _assess_example_relevance(self, entry: VocabularyEntry) -> float:
        """Assess the relevance and quality of the example sentence"""
        try:
            score = 0.0
            example = entry.example_sentence.strip()
            
            if not example:
                return 0.0
            
            # Basic structure check
            if example.endswith(('.', '!', '?')):
                score += 0.2
            
            # Contains the target word
            word_in_example = entry.word.lower() in example.lower()
            if word_in_example:
                score += 0.3
            
            # Length appropriateness (8-30 words)
            word_count = len(example.split())
            if 8 <= word_count <= 30:
                score += 0.2
            elif 5 <= word_count < 8 or 30 < word_count <= 40:
                score += 0.1
            
            # Context appropriateness (academic/SAT-level)
            academic_contexts = [
                'student', 'school', 'education', 'academic', 'study', 'research',
                'literature', 'history', 'science', 'analysis', 'argument'
            ]
            if any(context in example.lower() for context in academic_contexts):
                score += 0.15
            
            # Complexity appropriateness
            complex_sentence_indicators = [
                ',' in example,  # Has clauses
                len([w for w in example.split() if len(w) > 6]) >= 2,  # Multiple long words
                any(word in example.lower() for word in ['because', 'although', 'however', 'therefore'])
            ]
            score += sum(complex_sentence_indicators) * 0.05
            
            # Avoid common problems
            problems = [
                example.lower().startswith('the ' + entry.word.lower()),  # Boring start
                example.count(entry.word.lower()) > 2,  # Overuse
                len(example) < 20,  # Too short
                not any(c.isupper() for c in example)  # No capitals
            ]
            score -= sum(problems) * 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Example relevance assessment failed: {e}")
            return 0.0
    
    def _assess_content_completeness(self, entry: VocabularyEntry) -> float:
        """Assess completeness of all required fields"""
        try:
            score = 0.0
            total_fields = len(self.expected_fields)
            
            # Check each required field
            field_scores = {
                'word': 1.0 if entry.word.strip() else 0.0,
                'pronunciation': 1.0 if entry.pronunciation.strip() else 0.0,
                'part_of_speech': 1.0 if entry.part_of_speech.strip() else 0.0,
                'definition': 1.0 if entry.definition.strip() else 0.0,
                'mnemonic_type': 1.0 if entry.mnemonic_type.strip() else 0.0,
                'mnemonic_phrase': 1.0 if entry.mnemonic_phrase.strip() else 0.0,
                'picture_story': 0.8 if entry.picture_story.strip() else 0.0,  # Less critical
                'other_forms': 0.6 if entry.other_forms.strip() else 0.0,  # Optional
                'example_sentence': 1.0 if entry.example_sentence.strip() else 0.0
            }
            
            # Weighted average (core fields are more important)
            core_fields = ['word', 'definition', 'mnemonic_phrase', 'example_sentence']
            core_score = sum(field_scores[field] for field in core_fields) / len(core_fields)
            
            optional_score = sum(field_scores[field] for field in field_scores if field not in core_fields)
            optional_score /= (len(field_scores) - len(core_fields))
            
            score = core_score * 0.8 + optional_score * 0.2
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Content completeness assessment failed: {e}")
            return 0.0
    
    def _assess_linguistic_accuracy(self, entry: VocabularyEntry) -> float:
        """Assess linguistic accuracy and consistency"""
        try:
            score = 0.0
            
            # Word format check
            word = entry.word.strip()
            if word and word.replace('-', '').replace("'", "").isalpha():
                score += 0.2
            
            # Pronunciation format check
            pronunciation = entry.pronunciation.strip()
            if pronunciation:
                # Should contain phonetic indicators
                if any(char in pronunciation for char in ['(', ')', 'ˈ', 'ə', 'ɪ', 'ɛ', 'æ']):
                    score += 0.2
                elif pronunciation.count('-') >= 1:  # Syllable separation
                    score += 0.15
            
            # Part of speech validation
            valid_pos = ['noun', 'verb', 'adjective', 'adverb', 'preposition', 'conjunction', 'interjection']
            if entry.part_of_speech.lower() in valid_pos:
                score += 0.2
            
            # Definition grammar check
            definition = entry.definition.strip()
            if definition:
                # Should start with lowercase (after article)
                if definition.startswith(('a ', 'an ', 'the ')) and len(definition) > 4:
                    if definition.split()[1][0].islower():
                        score += 0.1
                
                # Should end with period
                if definition.endswith('.'):
                    score += 0.1
            
            # Example sentence grammar
            example = entry.example_sentence.strip()
            if example:
                # Should start with capital
                if example[0].isupper():
                    score += 0.1
                
                # Should end with punctuation
                if example.endswith(('.', '!', '?')):
                    score += 0.1
                
                # Should contain target word in appropriate form
                if entry.word.lower() in example.lower():
                    score += 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Linguistic accuracy assessment failed: {e}")
            return 0.0
    
    def _adjust_score_with_feedback(self, base_score: float, feedback_context: Dict) -> float:
        """Adjust score based on user feedback"""
        try:
            adjusted_score = base_score
            
            # Positive feedback boost
            positive_count = len(feedback_context.get("positive_examples", []))
            if positive_count > 0:
                boost = min(0.15, positive_count * 0.05)
                adjusted_score += boost
            
            # Negative feedback penalty
            negative_count = len(feedback_context.get("negative_examples", []))
            if negative_count > 0:
                penalty = min(0.2, negative_count * 0.05)
                adjusted_score -= penalty
            
            return min(1.0, max(0.0, adjusted_score))
            
        except Exception as e:
            logger.error(f"Feedback adjustment failed: {e}")
            return base_score
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level from score"""
        if score >= self.thresholds["excellent"]:
            return "excellent"
        elif score >= self.thresholds["good"]:
            return "good"
        elif score >= self.thresholds["acceptable"]:
            return "acceptable"
        elif score >= self.thresholds["needs_improvement"]:
            return "needs_improvement"
        else:
            return "poor"
    
    def _generate_improvement_suggestions(
        self, 
        entry: VocabularyEntry,
        definition_score: float,
        mnemonic_score: float,
        example_score: float,
        completeness_score: float,
        linguistic_score: float
    ) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Definition improvements
        if definition_score < 0.7:
            if not entry.definition.strip():
                suggestions.append("Add a clear, concise definition")
            elif len(entry.definition) < 20:
                suggestions.append("Expand the definition with more detail")
            elif len(entry.definition) > 200:
                suggestions.append("Shorten the definition for clarity")
            else:
                suggestions.append("Improve definition clarity and precision")
        
        # Mnemonic improvements
        if mnemonic_score < 0.7:
            if not entry.mnemonic_phrase.strip():
                suggestions.append("Add a memorable mnemonic phrase")
            else:
                suggestions.append("Enhance mnemonic with more vivid imagery")
                if not entry.picture_story.strip():
                    suggestions.append("Add a picture story to strengthen the mnemonic")
        
        # Example improvements
        if example_score < 0.7:
            if not entry.example_sentence.strip():
                suggestions.append("Add a relevant example sentence")
            elif entry.word.lower() not in entry.example_sentence.lower():
                suggestions.append("Ensure the example sentence uses the target word")
            else:
                suggestions.append("Improve example sentence relevance and academic context")
        
        # Completeness improvements
        if completeness_score < 0.8:
            missing_core = []
            if not entry.word.strip():
                missing_core.append("word")
            if not entry.definition.strip():
                missing_core.append("definition")
            if not entry.mnemonic_phrase.strip():
                missing_core.append("mnemonic phrase")
            if not entry.example_sentence.strip():
                missing_core.append("example sentence")
            
            if missing_core:
                suggestions.append(f"Complete missing core fields: {', '.join(missing_core)}")
            
            if not entry.pronunciation.strip():
                suggestions.append("Add pronunciation guide")
            if not entry.other_forms.strip():
                suggestions.append("Consider adding other word forms (noun, verb, etc.)")
        
        # Linguistic improvements
        if linguistic_score < 0.7:
            suggestions.append("Review grammar, punctuation, and formatting")
            
            if entry.part_of_speech.lower() not in ['noun', 'verb', 'adjective', 'adverb']:
                suggestions.append("Verify and correct part of speech")
        
        # General suggestions if overall quality is low
        overall_score = (definition_score + mnemonic_score + example_score + 
                        completeness_score + linguistic_score) / 5
        
        if overall_score < 0.5:
            suggestions.append("Consider regenerating this entry with different parameters")
        
        return suggestions
    
    def batch_assess_entries(self, entries: List[VocabularyEntry]) -> List[QualityMetrics]:
        """Assess multiple entries efficiently"""
        try:
            assessments = []
            
            for entry in entries:
                assessment = self.assess_vocabulary_entry(entry)
                assessments.append(assessment)
            
            logger.info(f"Batch assessment completed for {len(entries)} entries")
            return assessments
            
        except Exception as e:
            logger.error(f"Batch assessment failed: {e}")
            return []
    
    def get_quality_statistics(self, assessments: List[QualityMetrics]) -> Dict[str, Any]:
        """Generate quality statistics from assessments"""
        try:
            if not assessments:
                return {"error": "No assessments provided"}
            
            scores = [a.overall_score for a in assessments]
            quality_levels = [a.quality_level for a in assessments]
            
            # Basic statistics
            stats = {
                "total_entries": len(assessments),
                "average_score": statistics.mean(scores),
                "median_score": statistics.median(scores),
                "score_std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min_score": min(scores),
                "max_score": max(scores)
            }
            
            # Quality level distribution
            level_counts = {}
            for level in quality_levels:
                level_counts[level] = level_counts.get(level, 0) + 1
            stats["quality_distribution"] = level_counts
            
            # Score ranges
            score_ranges = {
                "excellent": len([s for s in scores if s >= 0.85]),
                "good": len([s for s in scores if 0.70 <= s < 0.85]),
                "acceptable": len([s for s in scores if 0.55 <= s < 0.70]),
                "needs_improvement": len([s for s in scores if 0.40 <= s < 0.55]),
                "poor": len([s for s in scores if s < 0.40])
            }
            stats["score_ranges"] = score_ranges
            
            # Top improvement suggestions
            all_suggestions = []
            for assessment in assessments:
                all_suggestions.extend(assessment.improvement_suggestions)
            
            suggestion_counts = {}
            for suggestion in all_suggestions:
                suggestion_counts[suggestion] = suggestion_counts.get(suggestion, 0) + 1
            
            # Sort by frequency
            top_suggestions = sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            stats["top_improvement_suggestions"] = top_suggestions
            
            return stats
            
        except Exception as e:
            logger.error(f"Quality statistics generation failed: {e}")
            return {"error": str(e)}


# Global quality assessment instance
_quality_system_instance = None


def get_quality_system() -> QualityAssessmentSystem:
    """Get or create the global quality assessment system"""
    global _quality_system_instance
    if _quality_system_instance is None:
        _quality_system_instance = QualityAssessmentSystem()
    return _quality_system_instance