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
    generated_text: str = ""
    llm_error: str = ""
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

            if hasattr(response, 'success') and not response.success:
                logger.error(f"LLM request failed for {word}: {getattr(response, 'error', 'unknown error')}")
                return self._create_error_entry(word, getattr(response, 'error', 'LLM request failed'))

            content = self._extract_content(response)

            if not content or not content.strip():
                return self._create_error_entry(word, "LLM returned empty content")
            
            # Parse and validate
            entry = self._parse_entry(content, word)
            entry.generated_text = content
            entry.generation_metadata = {
                "llm_success": True,
                "model": getattr(response, 'model', ''),
                "finish_reason": getattr(response, 'finish_reason', '')
            }

            if self._validate_entry(entry):
                return entry

            # If parse/validation fails, try one strict repair pass.
            repair_prompt = self._build_repair_prompt(word, content)
            repair_response = self.llm_service.generate_completion(repair_prompt, temperature=0.2, max_tokens=300)
            if hasattr(repair_response, 'success') and repair_response.success:
                repaired_content = self._extract_content(repair_response)
                repaired_entry = self._parse_entry(repaired_content, word)
                repaired_entry.generated_text = repaired_content
                repaired_entry.generation_metadata = {
                    "llm_success": True,
                    "model": getattr(repair_response, 'model', ''),
                    "finish_reason": getattr(repair_response, 'finish_reason', ''),
                    "repair_pass": True
                }
                if self._validate_entry(repaired_entry):
                    return repaired_entry

            return self._create_error_entry(word, "LLM output format/quality validation failed")
            
        except Exception as e:
            logger.error(f"Generation failed for {word}: {e}")
            return self._create_error_entry(word, str(e))
    
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
        
        # Default values for unknown words; real definition should come from LLM.
        return f"{word.lower()}", "adjective", "definition to be generated by LLM"

    def _build_repair_prompt(self, word: str, raw_output: str) -> str:
        """Build a strict reformat prompt for malformed LLM output."""
        pronunciation, pos, definition = self._get_word_info(word)
        return f"""Reformat the text below into EXACTLY this 5-line format and improve quality if needed:

{word.upper()} ({pronunciation}) {pos} — {definition}
Sounds like: [mnemonic phrase]
Picture: [2-3 sentence vivid picture story]
Other forms: [word forms]
Sentence: [natural sentence that uses {word.lower()}]

Do not include extra lines or commentary.

TEXT TO REFORMAT:
{raw_output}
"""
    
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

    def _create_error_entry(self, word: str, error_message: str) -> GeneratedVocabularyEntry:
        """Create an explicit error entry instead of low-quality fake content."""
        pronunciation, pos, _ = self._get_word_info(word)
        return GeneratedVocabularyEntry(
            word=word.upper(),
            pronunciation=pronunciation,
            part_of_speech=pos,
            definition="LLM generation failed. Check API key and provider status.",
            mnemonic_type="Sounds like",
            mnemonic_phrase="Unavailable due to LLM error",
            picture_story="No picture story generated because the language model request failed.",
            other_forms="",
            example_sentence=f"A valid example for '{word.lower()}' could not be generated.",
            quality_score=0.0,
            validation_passed=False,
            generated_text="",
            llm_error=error_message,
            generation_metadata={
                "llm_success": False,
                "error": error_message
            }
        )
    
    def generate_complete_entry(self, word: str, use_context: bool = True, num_context_examples: int = 3) -> GeneratedVocabularyEntry:
        """Generate a complete entry with context"""
        return self.generate_entry(word)
    
    def batch_generate(self, words: List[str], use_context: bool = True, num_context_examples: int = 3) -> List[GeneratedVocabularyEntry]:
        """Generate multiple entries"""
        return [self.generate_entry(word) for word in words]
