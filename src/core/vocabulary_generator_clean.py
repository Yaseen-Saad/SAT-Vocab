"""
Clean Vocabulary Generator
Simple, focused vocabulary generation without over-engineering
"""

import logging
import re
from typing import Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class GeneratedVocabularyEntry:
    """Simple vocabulary entry structure"""
    word: str
    pronunciation: str
    part_of_speech: str
    definition: str
    mnemonic_type: str
    mnemonic_phrase: str
    picture_story: str
    other_forms: str
    example_sentence: str
    quality_score: float = 80.0
    validation_passed: bool = True
    generation_metadata: Dict = field(default_factory=dict)

class SimpleVocabularyGenerator:
    """Clean vocabulary generator focused on quality and learning"""
    
    def __init__(self, llm_service, rag_engine):
        self.llm_service = llm_service
        self.rag_engine = rag_engine
        
        # Simple word database for better definitions
        self.word_data = {
            "REVERE": ("rev-EER", "verb", "to respect and admire deeply"),
            "SERENE": ("suh-REEN", "adjective", "calm and peaceful"),
            "ARDENT": ("AHR-dent", "adjective", "passionate and enthusiastic"),
            "PLACID": ("PLAS-id", "adjective", "calm and peaceful"),
            "FRANK": ("FRANK", "adjective", "open and honest"),
            "CANDID": ("KAN-did", "adjective", "truthful and straightforward"),
            "ZEPHYR": ("ZEF-er", "noun", "a gentle breeze"),
            "VICISSITUDE": ("vi-SIS-i-tood", "noun", "natural change or variation")
        }
    
    def generate_entry(self, word: str, part_of_speech: str = None, avoid_issues: Dict = None) -> GeneratedVocabularyEntry:
        """Generate a vocabulary entry"""
        try:
            # Get learning context from RAG
            context = self._get_context(word)
            
            # Build focused prompt
            prompt = self._build_prompt(word, context, avoid_issues)
            
            # Generate response
            response = self.llm_service.generate_completion(prompt)
            content = self._extract_content(response)
            
            # Parse and validate
            entry = self._parse_entry(content, word)
            if self._validate_entry(entry):
                return entry
            
            # Fallback
            return self._create_basic_entry(word)
            
        except Exception as e:
            logger.error(f"Generation failed for {word}: {e}")
            return self._create_basic_entry(word)
    
    def _get_context(self, word: str) -> Dict[str, Any]:
        """Get learning context from RAG"""
        # Get similar entries for style reference
        similar = self.rag_engine.get_similar_entries(word, top_k=2)
        examples = []
        for entry, score in similar:
            if score > 0.1 and entry.definition:
                examples.append({
                    'word': entry.word,
                    'format': f"{entry.word} ({entry.pronunciation}) {entry.part_of_speech} — {entry.definition}"
                })
        
        # Get feedback to avoid mistakes
        feedback = ""
        if hasattr(self.rag_engine, 'get_feedback_context'):
            feedback = self.rag_engine.get_feedback_context(word)
        
        return {
            'examples': examples[:2],  # Limit examples
            'feedback': feedback
        }
    
    def _build_prompt(self, word: str, context: Dict[str, Any], avoid_issues: List[str] = None) -> str:
        """Build clean, focused prompt"""
        # Get word info
        pronunciation, pos, definition = self._get_word_info(word)
        
        # Example format
        examples = ""
        if context['examples']:
            examples = "EXAMPLES:\n"
            for ex in context['examples']:
                examples += f"- {ex['format']}\n"
        
        # Feedback warnings
        warnings = ""
        if context['feedback']:
            warnings = f"\nAVOID THESE MISTAKES:\n{context['feedback']}\n"
        
        if avoid_issues:
            warnings += f"\nFIX THESE ISSUES:\n" + "\n".join(f"- {issue}" for issue in avoid_issues)
        
        return f"""Create a vocabulary entry in this EXACT format:

{word.upper()} ({pronunciation}) {pos} — {definition}
Sounds like: [something that sounds like the word]
Picture: [2-3 sentence visual story]
Other forms: [related forms]
Sentence: [example using the word]

{examples}

RULES:
1. Make mnemonic sound similar to "{word}"
2. Picture connects mnemonic to meaning
3. Keep everything concise
4. Example sentence must use "{word}"

{warnings}

Generate only the 5 lines above, nothing else."""
    
    def _get_word_info(self, word: str) -> tuple:
        """Get pronunciation, part of speech, and definition"""
        if word.upper() in self.word_data:
            return self.word_data[word.upper()]
        
        # Default values
        return f"{word.lower()}", "noun", f"related to {word.lower()}"
    
    def _extract_content(self, response) -> str:
        """Extract content from LLM response"""
        if hasattr(response, 'content'):
            return response.content
        elif hasattr(response, 'text'):
            return response.text
        return str(response)
    
    def _parse_entry(self, content: str, word: str) -> GeneratedVocabularyEntry:
        """Parse response into structured entry"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Initialize with defaults
        entry = GeneratedVocabularyEntry(
            word=word,
            pronunciation="",
            part_of_speech="noun",
            definition="",
            mnemonic_type="Sounds like",
            mnemonic_phrase="",
            picture_story="",
            other_forms="",
            example_sentence=""
        )
        
        for line in lines:
            if '(' in line and ')' in line and '—' in line:
                # Main line: WORD (pronunciation) pos — definition
                match = re.match(r'([A-Z]+)\s*\(([^)]+)\)\s*(\w+)\s*[—–-]\s*(.+)', line)
                if match:
                    entry.word = match.group(1)
                    entry.pronunciation = match.group(2)
                    entry.part_of_speech = match.group(3)
                    entry.definition = match.group(4)
            
            elif line.lower().startswith('sounds like:'):
                entry.mnemonic_phrase = line.split(':', 1)[1].strip()
            elif line.lower().startswith('picture:'):
                entry.picture_story = line.split(':', 1)[1].strip()
            elif line.lower().startswith('other forms:'):
                entry.other_forms = line.split(':', 1)[1].strip()
            elif line.lower().startswith('sentence:'):
                entry.example_sentence = line.split(':', 1)[1].strip()
        
        return entry
    
    def _validate_entry(self, entry: GeneratedVocabularyEntry) -> bool:
        """Quick validation of entry quality"""
        # Check required fields
        required = [entry.definition, entry.mnemonic_phrase, entry.picture_story, entry.example_sentence]
        if not all(required):
            return False
        
        # Check for word usage in example
        if entry.word.lower() not in entry.example_sentence.lower():
            return False
        
        # Check for circular definition
        if entry.word.lower() in entry.definition.lower():
            return False
        
        return True
    
    def _create_basic_entry(self, word: str) -> GeneratedVocabularyEntry:
        """Create basic fallback entry"""
        pronunciation, pos, definition = self._get_word_info(word)
        
        return GeneratedVocabularyEntry(
            word=word.upper(),
            pronunciation=pronunciation,
            part_of_speech=pos,
            definition=definition,
            mnemonic_type="Sounds like",
            mnemonic_phrase=f"sounds like {word.lower()}",
            picture_story=f"Picture someone demonstrating {definition}.",
            other_forms="",
            example_sentence=f"The {word.lower()} was impressive.",
            quality_score=50.0
        )
    
    def generate_complete_entry(self, word: str, use_context: bool = True, num_context_examples: int = 3) -> GeneratedVocabularyEntry:
        """Generate a complete entry with context"""
        return self.generate_entry(word)
    
    def batch_generate(self, words: List[str], use_context: bool = True, num_context_examples: int = 3) -> List[GeneratedVocabularyEntry]:
        """Generate multiple entries"""
        return [self.generate_entry(word) for word in words]
