import os
import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yaml

logger = logging.getLogger(__name__)


@dataclass
class VocabularyEntry:
    """Structured representation of a vocabulary entry"""
    word: str
    pronunciation: str
    part_of_speech: str
    definition: str
    mnemonic_type: str  # "sounds like", "looks like", etc.
    mnemonic_phrase: str
    picture_story: str
    other_forms: str
    example_sentence: str
    raw_text: str


class SampleParser:
    """
    Parses the authentic Gulotta entries from sample.txt
    """
    
    def __init__(self, sample_file_path: str):
        self.sample_file_path = sample_file_path
        self.entries = []
        self._load_and_parse()
    
    def _load_and_parse(self):
        """Load and parse the sample file"""
        try:
            with open(self.sample_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into individual entries
            # Entries typically start with a capitalized word followed by pronunciation
            entry_pattern = r'\n([A-Z][A-Z\s]+)\s*\([^)]+\)\s*\w+\s*—'
            entry_splits = re.split(entry_pattern, content)
            
            # Process each entry
            for i in range(1, len(entry_splits), 2):
                if i + 1 < len(entry_splits):
                    word_line = entry_splits[i].strip()
                    entry_content = entry_splits[i + 1]
                    
                    parsed_entry = self._parse_single_entry(word_line, entry_content)
                    if parsed_entry:
                        self.entries.append(parsed_entry)
            
            logger.info(f"Parsed {len(self.entries)} vocabulary entries from sample file")
            
        except Exception as e:
            logger.error(f"Error parsing sample file: {e}")
            self.entries = []
    
    def _parse_single_entry(self, word_line: str, content: str) -> Optional[VocabularyEntry]:
        """Parse a single vocabulary entry"""
        try:
            # Reconstruct the full entry text
            full_text = word_line + content
            
            # Extract word and pronunciation
            word_match = re.match(r'^([A-Z][A-Z\s]*)\s*\(([^)]+)\)\s*(\w+)\s*—\s*(.*?)(?=\n|$)', 
                                word_line + content, re.DOTALL)
            if not word_match:
                return None
            
            word = word_match.group(1).strip()
            pronunciation = word_match.group(2).strip()
            part_of_speech = word_match.group(3).strip()
            definition_start = word_match.group(4).strip()
            
            # Extract mnemonic (Sounds like, Looks like, etc.)
            mnemonic_match = re.search(r'(Sounds like|Looks like|Think of|Connect with):\s*([^\n]+)', content)
            mnemonic_type = mnemonic_match.group(1) if mnemonic_match else ""
            mnemonic_phrase = mnemonic_match.group(2) if mnemonic_match else ""
            
            # Extract picture story
            picture_match = re.search(r'Picture:\s*(.*?)(?=\n(?:Other forms|Sentence|Connect|Note|Don\'t|$))', 
                                    content, re.DOTALL)
            picture_story = picture_match.group(1).strip() if picture_match else ""
            
            # Extract other forms
            other_forms_match = re.search(r'Other forms?:\s*([^\n]+)', content)
            other_forms = other_forms_match.group(1).strip() if other_forms_match else ""
            
            # Extract example sentence
            sentence_match = re.search(r'Sentence:\s*([^\n]+)', content)
            example_sentence = sentence_match.group(1).strip() if sentence_match else ""
            
            # Get full definition (including the start from word line)
            definition_parts = [definition_start]
            
            # Look for definition continuation before mnemonic
            if mnemonic_match:
                def_text = content[:mnemonic_match.start()]
                def_lines = [line.strip() for line in def_text.split('\n') if line.strip()]
                if len(def_lines) > 1:
                    definition_parts.extend(def_lines[1:])
            
            definition = ' '.join(definition_parts).strip()
            
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
                raw_text=full_text
            )
            
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None
    
    def get_entries(self) -> List[VocabularyEntry]:
        """Get all parsed entries"""
        return self.entries


class EmbeddingGenerator:
    """
    Generates embeddings for vocabulary entries using sentence transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode([text])[0]
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(384)  # Default dimension
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.zeros((len(texts), 384))


class RAGEngine:
    """
    Main RAG engine for vocabulary pattern retrieval
    """
    
    def __init__(self, sample_file_path: str, config_path: str = None):
        self.sample_file_path = sample_file_path
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.parser = SampleParser(sample_file_path)
        self.embedding_generator = EmbeddingGenerator(
            self.config.get('embedding', {}).get('model', 'all-MiniLM-L6-v2')
        )
        
        # Generate embeddings for all entries
        self.entries = self.parser.get_entries()
        self.embeddings = self._generate_entry_embeddings()
        
        logger.info(f"RAG Engine initialized with {len(self.entries)} entries")
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}
    
    def _generate_entry_embeddings(self) -> np.ndarray:
        """Generate embeddings for all vocabulary entries"""
        if not self.entries:
            return np.array([])
        
        # Create searchable text for each entry (word + definition + mnemonic)
        entry_texts = []
        for entry in self.entries:
            search_text = f"{entry.word} {entry.definition} {entry.mnemonic_phrase}"
            entry_texts.append(search_text)
        
        embeddings = self.embedding_generator.generate_embeddings(entry_texts)
        logger.info(f"Generated embeddings for {len(entry_texts)} entries")
        return embeddings
    
    def retrieve_similar_entries(
        self, 
        query_word: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[VocabularyEntry, float]]:
        """
        Retrieve entries similar to the query word
        
        Args:
            query_word: The word to find similar entries for
            top_k: Number of top similar entries to return
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of (entry, similarity_score) tuples
        """
        if len(self.embeddings) == 0:
            return []
        
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(query_word)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k similar entries
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in similar_indices:
            similarity_score = similarities[idx]
            if similarity_score >= similarity_threshold:
                results.append((self.entries[idx], similarity_score))
        
        logger.info(f"Retrieved {len(results)} similar entries for '{query_word}'")
        return results
    
    def get_context_examples(
        self, 
        word: str, 
        num_examples: int = 3
    ) -> List[str]:
        """
        Get context examples for generating new vocabulary entries
        
        Args:
            word: The target word
            num_examples: Number of example entries to retrieve
        
        Returns:
            List of raw text examples
        """
        similar_entries = self.retrieve_similar_entries(word, top_k=num_examples * 2)
        
        # Select diverse examples (not just the most similar)
        examples = []
        used_patterns = set()
        
        for entry, score in similar_entries:
            # Create a pattern signature based on mnemonic type and structure
            pattern_sig = f"{entry.mnemonic_type}_{len(entry.picture_story.split())//10}"
            
            if pattern_sig not in used_patterns and len(examples) < num_examples:
                examples.append(entry.raw_text)
                used_patterns.add(pattern_sig)
        
        # If we don't have enough diverse examples, fill with any remaining
        while len(examples) < num_examples and len(examples) < len(similar_entries):
            for entry, score in similar_entries:
                if entry.raw_text not in examples:
                    examples.append(entry.raw_text)
                    if len(examples) >= num_examples:
                        break
        
        logger.info(f"Selected {len(examples)} context examples for '{word}'")
        return examples
    
    def get_random_examples(self, num_examples: int = 3) -> List[str]:
        """Get random examples for general context"""
        if len(self.entries) == 0:
            return []
        
        indices = np.random.choice(len(self.entries), 
                                 size=min(num_examples, len(self.entries)), 
                                 replace=False)
        examples = [self.entries[idx].raw_text for idx in indices]
        
        logger.info(f"Selected {len(examples)} random examples")
        return examples
    
    def search_by_pattern(self, pattern_type: str, limit: int = 10) -> List[VocabularyEntry]:
        """
        Search entries by mnemonic pattern type
        
        Args:
            pattern_type: Type of mnemonic pattern (e.g., "Sounds like", "Looks like")
            limit: Maximum number of results
        
        Returns:
            List of matching entries
        """
        matches = [entry for entry in self.entries 
                  if entry.mnemonic_type.lower() == pattern_type.lower()]
        
        return matches[:limit]


# Global RAG engine instance
_rag_engine_instance = None


def get_rag_engine(sample_file_path: str = None) -> RAGEngine:
    """Get or create the global RAG engine instance"""
    global _rag_engine_instance
    
    if _rag_engine_instance is None:
        if sample_file_path is None:
            # Default path
            base_dir = os.path.dirname(__file__)
            sample_file_path = os.path.join(base_dir, '..', '..', 'sample.txt')
        
        _rag_engine_instance = RAGEngine(sample_file_path)
    
    return _rag_engine_instance

# Convenience functions
def retrieve_context_for_word(word: str, num_examples: int = 3) -> List[str]:
    """Convenience function to get context examples for a word"""
    engine = get_rag_engine()
    return engine.get_context_examples(word, num_examples)

def find_similar_vocabulary(word: str, top_k: int = 5) -> List[Tuple[VocabularyEntry, float]]:
    """Convenience function to find similar vocabulary entries"""
    engine = get_rag_engine()
    return engine.retrieve_similar_entries(word, top_k)