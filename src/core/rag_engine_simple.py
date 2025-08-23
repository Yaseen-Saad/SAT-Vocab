"""
Simplified RAG Engine for SAT Vocabulary (without heavy dependencies)
Provides context retrieval from the sample vocabulary data using basic text matching.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class VocabularyEntry:
    """Structured representation of a vocabulary entry"""
    word: str
    pronunciation: str
    part_of_speech: str
    definition: str
    mnemonic_type: str
    mnemonic_phrase: str
    picture_story: str
    other_forms: str
    example_sentence: str
    raw_text: str

class SimpleRAGEngine:
    """Simplified RAG engine using basic text matching."""
    
    def __init__(self, sample_file_path: str):
        self.sample_file_path = sample_file_path
        self.entries = []
        self._load_sample_data()
    
    def _load_sample_data(self):
        """Load and parse the sample vocabulary data."""
        try:
            with open(self.sample_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse vocabulary entries
            self.entries = self._parse_vocabulary_entries(content)
            logger.info(f"Loaded {len(self.entries)} vocabulary entries")
            
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            # Create fallback entries
            self.entries = self._create_fallback_entries()
    
    def _parse_vocabulary_entries(self, content: str) -> List[VocabularyEntry]:
        """Parse vocabulary entries from the sample text."""
        entries = []
        
        # Split by word entries - look for word patterns like "ACCLAIM (..."
        word_pattern = r'\n([A-Z]{2,}(?:\s+[A-Z]+)*)\s*\([^)]+\)\s*(\w+)\s*—\s*([^\n]+)'
        
        matches = re.finditer(word_pattern, content)
        
        current_pos = 0
        for match in matches:
            word = match.group(1).strip()
            pronunciation = match.group(0).split('(')[1].split(')')[0]
            part_of_speech = match.group(2)
            definition = match.group(3)
            
            # Get the entry content (from current match to next match)
            start_pos = match.start()
            next_match = None
            
            # Find next match to get end position
            temp_pos = match.end()
            for next_match in re.finditer(word_pattern, content[temp_pos:]):
                break
            
            if next_match:
                end_pos = temp_pos + next_match.start()
            else:
                end_pos = len(content)
            
            entry_text = content[start_pos:end_pos].strip()
            
            # Parse the full entry
            parsed_entry = self._parse_entry_content(word, pronunciation, part_of_speech, definition, entry_text)
            if parsed_entry:
                entries.append(parsed_entry)
        
        # If no entries found, try simpler parsing
        if not entries:
            entries = self._simple_parse(content)
        
        logger.info(f"Parsed {len(entries)} entries")
        return entries[:100]  # Limit for performance
    
    def _parse_entry_content(self, word: str, pronunciation: str, part_of_speech: str, definition: str, content: str) -> Optional[VocabularyEntry]:
        """Parse a complete entry content."""
        try:
            # Extract mnemonic
            mnemonic_patterns = [
                r'(Sounds like|Looks like|Think of|Connect with):\s*([^\n]+)',
                r'(First|Picture lead-in):\s*([^\n]+)'
            ]
            
            mnemonic_type = ""
            mnemonic_phrase = ""
            
            for pattern in mnemonic_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    mnemonic_type = match.group(1)
                    mnemonic_phrase = match.group(2)
                    break
            
            # Extract picture story
            picture_match = re.search(r'Picture:\s*(.*?)(?=\n(?:Other forms?|Sentence|Connect|Note|Don\'t|[A-Z]{2,}\s*\(|$))', 
                                    content, re.DOTALL | re.IGNORECASE)
            picture_story = picture_match.group(1).strip() if picture_match else ""
            
            # Extract other forms
            other_forms_match = re.search(r'Other forms?:\s*([^\n]+)', content, re.IGNORECASE)
            other_forms = other_forms_match.group(1).strip() if other_forms_match else ""
            
            # Extract example sentence
            sentence_match = re.search(r'Sentence:\s*([^\n]+)', content, re.IGNORECASE)
            example_sentence = sentence_match.group(1).strip() if sentence_match else ""
            
            return VocabularyEntry(
                word=word,
                pronunciation=pronunciation,
                part_of_speech=part_of_speech,
                definition=definition,
                mnemonic_type=mnemonic_type,
                mnemonic_phrase=mnemonic_phrase,
                picture_story=picture_story,
                other_forms=other_forms,
                example_sentence=example_sentence,
                raw_text=content
            )
            
        except Exception as e:
            logger.error(f"Error parsing entry for {word}: {e}")
            return None
    
    def _simple_parse(self, content: str) -> List[VocabularyEntry]:
        """Simple fallback parsing."""
        entries = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if re.match(r'^\d+\.\s*[A-Z]+', line):
                # Found a numbered entry
                parts = line.split('—', 1)
                if len(parts) > 1:
                    word_part = parts[0].strip()
                    definition = parts[1].strip()
                    
                    # Extract just the word
                    word_match = re.search(r'[A-Z]+', word_part)
                    word = word_match.group() if word_match else "UNKNOWN"
                    
                    entry = VocabularyEntry(
                        word=word,
                        pronunciation="",
                        part_of_speech="",
                        definition=definition,
                        mnemonic_type="",
                        mnemonic_phrase="",
                        picture_story="",
                        other_forms="",
                        example_sentence="",
                        raw_text=line
                    )
                    entries.append(entry)
        
        return entries
    
    def _create_fallback_entries(self) -> List[VocabularyEntry]:
        """Create fallback entries if sample data can't be loaded."""
        return [
            VocabularyEntry(
                word='ABATE',
                pronunciation='uh-BAYT',
                part_of_speech='verb',
                definition='to reduce in intensity',
                mnemonic_type='Sounds like',
                mnemonic_phrase='A BAIT to catch fish reduces the fish population',
                picture_story='Storm clouds clearing',
                other_forms='abated, abating',
                example_sentence='The medication helped abate his symptoms',
                raw_text='ABATE (uh-BAYT) verb — to reduce in intensity'
            ),
            VocabularyEntry(
                word='VERBOSE',
                pronunciation='ver-BOHS',
                part_of_speech='adjective',
                definition='wordy; using more words than necessary',
                mnemonic_type='Sounds like',
                mnemonic_phrase='VERB OSE sounds like verbose - too many words',
                picture_story='Someone talking too much',
                other_forms='verbosity, verbosely',
                example_sentence='His verbose explanation confused everyone',
                raw_text='VERBOSE (ver-BOHS) adjective — wordy; using more words than necessary'
            )
        ]
    
    def retrieve_similar_entries(self, query_word: str, top_k: int = 3, similarity_threshold: float = 0.0) -> List[Tuple[VocabularyEntry, float]]:
        """Retrieve similar entries (alias for get_similar_entries for compatibility)."""
        results = self.get_similar_entries(query_word, top_k)
        # Filter by threshold if specified
        if similarity_threshold > 0:
            results = [(entry, score) for entry, score in results if score >= similarity_threshold]
        return results
    
    def get_similar_entries(self, query: str, top_k: int = 3) -> List[Tuple[VocabularyEntry, float]]:
        """Get vocabulary entries similar to the query using text matching."""
        if not self.entries:
            return []
        
        query_lower = query.lower()
        scored_entries = []
        
        for entry in self.entries:
            score = 0
            
            # Word matching (highest weight)
            if query_lower == entry.word.lower():
                score += 20
            elif query_lower in entry.word.lower():
                score += 15
            
            # Definition matching
            if query_lower in entry.definition.lower():
                score += 10
            
            # Mnemonic matching
            if query_lower in entry.mnemonic_phrase.lower():
                score += 5
            
            # Picture story matching
            if query_lower in entry.picture_story.lower():
                score += 3
            
            # Example sentence matching
            if query_lower in entry.example_sentence.lower():
                score += 2
            
            # Word similarity (basic)
            for word in query_lower.split():
                if word in entry.word.lower():
                    score += 8
                if word in entry.definition.lower():
                    score += 4
            
            if score > 0:
                # Normalize score
                normalized_score = min(score / 20.0, 1.0)
                scored_entries.append((entry, normalized_score))
        
        # Sort by score and return top k
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return scored_entries[:top_k]
    
    def get_context_examples(self, word: str, num_examples: int = 3) -> List[str]:
        """Get context examples for generating new vocabulary entries."""
        similar_entries = self.get_similar_entries(word, top_k=num_examples * 2)
        
        examples = []
        for entry, score in similar_entries:
            if len(examples) < num_examples:
                examples.append(entry.raw_text)
        
        # If we don't have enough, add some random ones
        if len(examples) < num_examples:
            remaining_entries = [e for e in self.entries if e.raw_text not in examples]
            for i, entry in enumerate(remaining_entries):
                if len(examples) >= num_examples:
                    break
                examples.append(entry.raw_text)
        
        return examples
    
    def get_random_examples(self, num_examples: int = 3) -> List[str]:
        """Get random examples for general context."""
        if not self.entries:
            return []
        
        import random
        selected = random.sample(self.entries, min(num_examples, len(self.entries)))
        return [entry.raw_text for entry in selected]
    
    def search_by_pattern(self, pattern_type: str, limit: int = 10) -> List[VocabularyEntry]:
        """Search entries by mnemonic pattern type."""
        matches = [entry for entry in self.entries 
                  if entry.mnemonic_type.lower() == pattern_type.lower()]
        return matches[:limit]
    
    def add_negative_example(self, word: str, negative_example: str):
        """Add a negative example to avoid similar generation patterns"""
        try:
            import os
            feedback_dir = os.path.join('feedback_data')
            os.makedirs(feedback_dir, exist_ok=True)
            
            negative_file = os.path.join(feedback_dir, 'negative_examples.txt')
            
            with open(negative_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{negative_example}\n{'='*50}\n")
                
            logger.info(f"Added negative feedback for {word}")
            
        except Exception as e:
            logger.error(f"Failed to store negative feedback: {e}")
    
    def add_positive_example(self, word: str, positive_example: str):
        """Add a positive example to guide future generations"""
        try:
            import os
            feedback_dir = os.path.join('feedback_data')
            os.makedirs(feedback_dir, exist_ok=True)
            
            positive_file = os.path.join(feedback_dir, 'positive_examples.txt')
            
            with open(positive_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{positive_example}\n{'='*50}\n")
                
            # Also try to add to the main sample file for immediate use
            self._add_to_active_examples(positive_example)
            
            logger.info(f"Added positive feedback for {word}")
            
        except Exception as e:
            logger.error(f"Failed to store positive feedback: {e}")
    
    def _add_to_active_examples(self, positive_example: str):
        """Add positive example to active examples for immediate use"""
        try:
            # Parse the positive example into a VocabularyEntry
            lines = positive_example.strip().split('\n')
            if not lines:
                return
                
            # Look for the main entry line
            main_line = None
            for line in lines:
                if '(' in line and ')' in line and '—' in line:
                    main_line = line
                    break
            
            if main_line:
                # Try to parse it as a vocabulary entry
                word_match = re.search(r'([A-Z]+(?:\s+[A-Z]+)*)', main_line)
                if word_match:
                    word = word_match.group(1)
                    
                    # Create a simplified entry
                    entry = VocabularyEntry(
                        word=word,
                        pronunciation="",
                        part_of_speech="",
                        definition="",
                        mnemonic_type="",
                        mnemonic_phrase="",
                        picture_story="",
                        other_forms="",
                        example_sentence="",
                        raw_text=positive_example
                    )
                    
                    # Add to our entries list
                    self.entries.append(entry)
                    logger.info(f"Added positive example for {word} to active examples")
                    
        except Exception as e:
            logger.warning(f"Failed to add positive example to active examples: {e}")
    
    def get_feedback_context(self, word: str) -> Dict[str, List[str]]:
        """Get both positive and negative feedback context for a word"""
        feedback = {'positive': [], 'negative': []}
        
        try:
            import os
            feedback_dir = os.path.join('feedback_data')
            
            # Load negative examples
            negative_file = os.path.join(feedback_dir, 'negative_examples.txt')
            if os.path.exists(negative_file):
                with open(negative_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for examples mentioning this word
                    if word.upper() in content:
                        sections = content.split('=' * 50)
                        for section in sections:
                            if word.upper() in section:
                                feedback['negative'].append(section.strip())
            
            # Load positive examples
            positive_file = os.path.join(feedback_dir, 'positive_examples.txt')
            if os.path.exists(positive_file):
                with open(positive_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for examples mentioning this word
                    if word.upper() in content:
                        sections = content.split('=' * 50)
                        for section in sections:
                            if word.upper() in section:
                                feedback['positive'].append(section.strip())
                                
        except Exception as e:
            logger.warning(f"Failed to load feedback context: {e}")
            
        return feedback

# Global instance
_simple_rag_engine_instance = None

def get_rag_engine(sample_file_path: str = None) -> SimpleRAGEngine:
    """Get or create the global simple RAG engine instance."""
    global _simple_rag_engine_instance
    
    if _simple_rag_engine_instance is None:
        if sample_file_path is None:
            # Default path
            current_dir = Path(__file__).parent.parent.parent
            sample_file_path = current_dir / "sample.txt"
        
        _simple_rag_engine_instance = SimpleRAGEngine(str(sample_file_path))
    
    return _simple_rag_engine_instance

def retrieve_context_for_word(word: str, num_examples: int = 3) -> List[str]:
    """Convenience function to get context examples for a word."""
    engine = get_rag_engine()
    return engine.get_context_examples(word, num_examples)

def find_similar_vocabulary(word: str, top_k: int = 5) -> List[Tuple[VocabularyEntry, float]]:
    """Convenience function to find similar vocabulary entries."""
    engine = get_rag_engine()
    return engine.get_similar_entries(word, top_k)
