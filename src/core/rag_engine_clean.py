"""
Clean RAG Engine
Focused on learning from user feedback and improving generation quality
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class VocabularyEntry:
    """Simple vocabulary entry structure"""
    word: str
    pronunciation: str = ""
    part_of_speech: str = ""
    definition: str = ""
    mnemonic_type: str = ""
    mnemonic_phrase: str = ""
    picture_story: str = ""
    other_forms: str = ""
    example_sentence: str = ""
    raw_text: str = ""

class CleanRAGEngine:
    """Simple, effective RAG engine that learns from feedback"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.entries: List[VocabularyEntry] = []
        self.feedback_dir = "feedback_data"
        
        # In-memory storage for serverless environments
        self.negative_examples = []
        self.positive_examples = []
        
        # Create directories (handle read-only file systems)
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.feedback_dir, exist_ok=True)
            self.can_write = True
        except (OSError, PermissionError):
            # Read-only file system (like Vercel serverless)
            self.can_write = False
            logger.info("Read-only file system detected - using in-memory storage")
        
        # Load initial data
        self._load_sample_entries()
        self._load_user_entries()
    
    def _load_sample_entries(self):
        """Load some quality sample entries for reference"""
        samples = [
            VocabularyEntry(
                word="REVERE",
                pronunciation="rev-EER",
                part_of_speech="verb",
                definition="to respect and admire deeply",
                mnemonic_type="Sounds like",
                mnemonic_phrase="reverend",
                picture_story="Picture a reverend being deeply respected by his congregation, everyone looking up to him with admiration.",
                other_forms="reverence, revered, revering",
                example_sentence="Students revere their wise professor.",
                raw_text="REVERE (rev-EER) verb — to respect and admire deeply"
            ),
            VocabularyEntry(
                word="SERENE",
                pronunciation="suh-REEN",
                part_of_speech="adjective",
                definition="calm and peaceful",
                mnemonic_type="Sounds like",
                mnemonic_phrase="seen",
                picture_story="Picture a serene lake you've seen, perfectly still and peaceful, reflecting the calm sky above.",
                other_forms="serenity, serenely",
                example_sentence="The monastery garden felt serene and tranquil.",
                raw_text="SERENE (suh-REEN) adjective — calm and peaceful"
            ),
            VocabularyEntry(
                word="CANDID",
                pronunciation="KAN-did",
                part_of_speech="adjective", 
                definition="truthful and straightforward",
                mnemonic_type="Sounds like",
                mnemonic_phrase="can did",
                picture_story="Picture someone saying 'I can tell you what I did' - being completely honest and straightforward about their actions.",
                other_forms="candidly, candidness",
                example_sentence="Her candid response surprised everyone.",
                raw_text="CANDID (KAN-did) adjective — truthful and straightforward"
            )
        ]
        self.entries.extend(samples)
    
    def _load_user_entries(self):
        """Load entries from user sessions"""
        entries_file = os.path.join(self.data_dir, "user_entries.json")
        if os.path.exists(entries_file):
            try:
                with open(entries_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        self.entries.append(VocabularyEntry(**item))
                logger.info(f"Loaded {len(data)} user entries")
            except Exception as e:
                logger.error(f"Error loading user entries: {e}")
    
    def add_entry(self, entry: VocabularyEntry):
        """Add a new entry to the database"""
        self.entries.append(entry)
        self._save_user_entries()
    
    def _save_user_entries(self):
        """Save user entries to file"""
        # Filter out sample entries (they start with known words)
        sample_words = {"REVERE", "SERENE", "CANDID"}
        user_entries = [entry for entry in self.entries if entry.word not in sample_words]
        
        entries_file = os.path.join(self.data_dir, "user_entries.json")
        try:
            with open(entries_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(entry) for entry in user_entries], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user entries: {e}")
    
    def get_similar_entries(self, query: str, top_k: int = 3) -> List[Tuple[VocabularyEntry, float]]:
        """Find similar entries using simple text matching"""
        if not self.entries:
            return []
        
        query_lower = query.lower()
        scored_entries = []
        
        for entry in self.entries:
            score = 0
            
            # Word matching (highest priority)
            if query_lower == entry.word.lower():
                score += 10
            elif query_lower in entry.word.lower():
                score += 5
            
            # Definition matching
            if query_lower in entry.definition.lower():
                score += 3
            
            # Content matching
            if query_lower in entry.mnemonic_phrase.lower():
                score += 2
            if query_lower in entry.picture_story.lower():
                score += 1
            
            if score > 0:
                scored_entries.append((entry, score))
        
        # Sort by score and return top k
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        return scored_entries[:top_k]
    
    def add_negative_example(self, word: str, bad_example: str):
        """Store negative feedback to avoid similar mistakes"""
        timestamp = datetime.now().isoformat()
        
        if self.can_write:
            # Try to save to file
            neg_file = os.path.join(self.feedback_dir, "negative_examples.txt")
            try:
                with open(neg_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== NEGATIVE EXAMPLE - {word} - {timestamp} ===\n")
                    f.write(bad_example)
                    f.write("\n" + "="*50 + "\n")
                logger.info(f"Stored negative example for {word} to file")
            except Exception as e:
                logger.error(f"Error storing negative example to file: {e}")
                # Fallback to memory
                self.negative_examples.append({
                    'word': word,
                    'example': bad_example,
                    'timestamp': timestamp
                })
        else:
            # Store in memory for serverless
            self.negative_examples.append({
                'word': word,
                'example': bad_example,
                'timestamp': timestamp
            })
            logger.info(f"Stored negative example for {word} in memory")
    
    def add_positive_example(self, word: str, good_example: str):
        """Store positive feedback as learning examples"""
        timestamp = datetime.now().isoformat()
        
        if self.can_write:
            # Try to save to file
            pos_file = os.path.join(self.feedback_dir, "positive_examples.txt")
            try:
                with open(pos_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n=== POSITIVE EXAMPLE - {word} - {timestamp} ===\n")
                    f.write(good_example)
                    f.write("\n" + "="*50 + "\n")
                logger.info(f"Stored positive example for {word} to file")
            except Exception as e:
                logger.error(f"Error storing positive example to file: {e}")
                # Fallback to memory
                self.positive_examples.append({
                    'word': word,
                    'example': good_example,
                    'timestamp': timestamp
                })
        else:
            # Store in memory for serverless
            self.positive_examples.append({
                'word': word,
                'example': good_example,
                'timestamp': timestamp
            })
            logger.info(f"Stored positive example for {word} in memory")
    
    def get_feedback_context(self, word: str) -> str:
        """Get feedback context to improve generation"""
        context = ""
        
        # Get negative examples to avoid
        neg_file = os.path.join(self.feedback_dir, "negative_examples.txt")
        if os.path.exists(neg_file):
            try:
                with open(neg_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for examples related to this word
                    if word.upper() in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if word.upper() in line and "NEGATIVE EXAMPLE" in line:
                                # Get the next few lines that contain the example
                                example_lines = []
                                j = i + 1
                                while j < len(lines) and not lines[j].startswith("==="):
                                    if lines[j].strip():
                                        example_lines.append(lines[j].strip())
                                    j += 1
                                if example_lines:
                                    context += f"AVOID: {' '.join(example_lines[:3])}\n"
            except Exception as e:
                logger.error(f"Error reading negative examples: {e}")
        
        # Get positive examples to follow
        pos_file = os.path.join(self.feedback_dir, "positive_examples.txt")
        if os.path.exists(pos_file):
            try:
                with open(pos_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if word.upper() in content:
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if word.upper() in line and "POSITIVE EXAMPLE" in line:
                                example_lines = []
                                j = i + 1
                                while j < len(lines) and not lines[j].startswith("==="):
                                    if lines[j].strip():
                                        example_lines.append(lines[j].strip())
                                    j += 1
                                if example_lines:
                                    context += f"FOLLOW: {' '.join(example_lines[:3])}\n"
            except Exception as e:
                logger.error(f"Error reading positive examples: {e}")
        
        return context

# Global instance
_rag_engine = None

def get_rag_engine() -> CleanRAGEngine:
    """Get the global RAG engine instance"""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = CleanRAGEngine()
    return _rag_engine
