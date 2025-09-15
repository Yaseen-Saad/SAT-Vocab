"""
Advanced LangChain-based RAG Engine for SAT Vocabulary System
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema.retriever import BaseRetriever

# Local imports
from ..models import VocabularyEntry, QualityMetrics
from ..services.config import get_settings


logger = logging.getLogger(__name__)


class VocabularyDocument(Document):
    """Enhanced document class for vocabulary entries"""
    
    def __init__(self, vocabulary_entry: VocabularyEntry):
        # Create comprehensive content for embedding
        content = self._create_searchable_content(vocabulary_entry)
        
        # Rich metadata for filtering and retrieval
        metadata = {
            "word": vocabulary_entry.word,
            "pronunciation": vocabulary_entry.pronunciation,
            "part_of_speech": vocabulary_entry.part_of_speech,
            "mnemonic_type": vocabulary_entry.mnemonic_type,
            "quality_score": vocabulary_entry.quality_score,
            "difficulty_level": vocabulary_entry.difficulty_level,
            "source": vocabulary_entry.source,
            "created_at": vocabulary_entry.created_at.isoformat() if vocabulary_entry.created_at else None
        }
        
        super().__init__(page_content=content, metadata=metadata)
        self.vocabulary_entry = vocabulary_entry
    
    def _create_searchable_content(self, entry: VocabularyEntry) -> str:
        """Create comprehensive searchable content"""
        parts = [
            f"Word: {entry.word}",
            f"Pronunciation: {entry.pronunciation}",
            f"Part of Speech: {entry.part_of_speech}",
            f"Definition: {entry.definition}",
            f"Mnemonic Type: {entry.mnemonic_type}",
            f"Mnemonic Phrase: {entry.mnemonic_phrase}",
            f"Picture Story: {entry.picture_story}",
            f"Other Forms: {entry.other_forms}",
            f"Example Sentence: {entry.example_sentence}"
        ]
        return "\n".join(parts)


class AdvancedRAGEngine:
    """LangChain-powered RAG engine with advanced retrieval capabilities"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.vector_store = None
        self.embeddings = None
        self.hybrid_retriever = None
        self.text_splitter = None
        
        # Feedback storage
        self.positive_examples = []
        self.negative_examples = []
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_vector_store()
        self._initialize_retrievers()
        self._load_feedback_data()
        
        logger.info("Advanced RAG Engine initialized successfully")
    
    def _initialize_embeddings(self):
        """Initialize sentence transformers embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Embeddings initialized with model: {self.settings.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize Chroma vector store"""
        try:
            # Ensure vector store directory exists
            vector_store_path = Path(self.settings.vector_db_path)
            vector_store_path.mkdir(parents=True, exist_ok=True)
            
            self.vector_store = Chroma(
                collection_name=self.settings.vector_db_collection,
                embedding_function=self.embeddings,
                persist_directory=str(vector_store_path)
            )
            logger.info(f"Vector store initialized at: {vector_store_path}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _initialize_retrievers(self):
        """Initialize hybrid retrieval system"""
        try:
            # Text splitter for processing long content
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            
            # Initialize hybrid retriever when we have documents
            if self.vector_store._collection.count() > 0:
                self._setup_hybrid_retriever()
                
        except Exception as e:
            logger.warning(f"Could not initialize retrievers: {e}")
    
    def _setup_hybrid_retriever(self):
        """Setup hybrid retriever combining semantic and keyword search"""
        try:
            # Get all documents for BM25
            all_docs = []
            collection = self.vector_store._collection
            results = collection.get()
            
            for i, (doc_id, metadata, document) in enumerate(zip(
                results['ids'], results['metadatas'], results['documents']
            )):
                all_docs.append(Document(page_content=document, metadata=metadata))
            
            if not all_docs:
                return
            
            # Create retrievers
            vector_retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
            
            keyword_retriever = BM25Retriever.from_documents(all_docs)
            keyword_retriever.k = 10
            
            # Combine retrievers
            self.hybrid_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, keyword_retriever],
                weights=[0.7, 0.3]  # Favor semantic search
            )
            
            logger.info("Hybrid retriever setup completed")
            
        except Exception as e:
            logger.warning(f"Could not setup hybrid retriever: {e}")
    
    def add_vocabulary_entry(self, entry: VocabularyEntry) -> bool:
        """Add a vocabulary entry to the vector store"""
        try:
            # Create vocabulary document
            vocab_doc = VocabularyDocument(entry)
            
            # Add to vector store
            self.vector_store.add_documents([vocab_doc])
            
            # Update hybrid retriever
            self._setup_hybrid_retriever()
            
            logger.info(f"Added vocabulary entry: {entry.word}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vocabulary entry {entry.word}: {e}")
            return False
    
    def add_vocabulary_entries(self, entries: List[VocabularyEntry]) -> int:
        """Add multiple vocabulary entries efficiently"""
        try:
            # Create vocabulary documents
            vocab_docs = [VocabularyDocument(entry) for entry in entries]
            
            # Batch add to vector store
            self.vector_store.add_documents(vocab_docs)
            
            # Update hybrid retriever once
            self._setup_hybrid_retriever()
            
            logger.info(f"Added {len(entries)} vocabulary entries")
            return len(entries)
            
        except Exception as e:
            logger.error(f"Failed to add vocabulary entries: {e}")
            return 0
    
    def search_similar_entries(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        search_type: str = "hybrid"
    ) -> List[Tuple[VocabularyEntry, float]]:
        """Search for similar vocabulary entries"""
        try:
            if search_type == "hybrid" and self.hybrid_retriever:
                # Use hybrid retrieval
                docs = self.hybrid_retriever.get_relevant_documents(query)
                
                # Convert to vocabulary entries with similarity scores
                results = []
                for doc in docs[:top_k]:
                    if hasattr(doc, 'vocabulary_entry'):
                        entry = doc.vocabulary_entry
                        # For hybrid search, we approximate similarity
                        similarity = 0.8  # Placeholder
                        results.append((entry, similarity))
                
                return results
                
            elif search_type == "semantic":
                # Pure semantic search
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query, k=top_k
                )
                
                results = []
                for doc, score in docs_with_scores:
                    if score >= similarity_threshold:
                        if hasattr(doc, 'vocabulary_entry'):
                            entry = doc.vocabulary_entry
                            # Convert distance to similarity (higher is better)
                            similarity = max(0, 1 - score)
                            results.append((entry, similarity))
                
                return results
                
            else:
                # Keyword-based fallback
                return self._keyword_search(query, top_k, similarity_threshold)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _keyword_search(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[VocabularyEntry, float]]:
        """Fallback keyword-based search"""
        try:
            # Get all documents
            collection = self.vector_store._collection
            results = collection.get()
            
            query_lower = query.lower()
            scored_entries = []
            
            for metadata, document in zip(results['metadatas'], results['documents']):
                score = 0
                document_lower = document.lower()
                
                # Word matching (highest priority)
                if query_lower == metadata.get('word', '').lower():
                    score += 10
                elif query_lower in metadata.get('word', '').lower():
                    score += 5
                
                # Content matching
                if query_lower in document_lower:
                    score += 3
                
                # Definition matching
                if 'definition' in document_lower and query_lower in document_lower:
                    score += 2
                
                if score >= similarity_threshold * 10:  # Scale threshold
                    # Reconstruct vocabulary entry from metadata and document
                    entry = self._reconstruct_entry_from_metadata(metadata, document)
                    if entry:
                        scored_entries.append((entry, score / 10))  # Normalize score
            
            # Sort by score and return top k
            scored_entries.sort(key=lambda x: x[1], reverse=True)
            return scored_entries[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _reconstruct_entry_from_metadata(self, metadata: Dict, document: str) -> Optional[VocabularyEntry]:
        """Reconstruct vocabulary entry from stored metadata and document"""
        try:
            # Parse document content to extract fields
            lines = document.split('\n')
            entry_data = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    field_name = key.strip().lower().replace(' ', '_')
                    entry_data[field_name] = value.strip()
            
            # Create vocabulary entry
            return VocabularyEntry(
                word=metadata.get('word', entry_data.get('word', '')),
                pronunciation=metadata.get('pronunciation', entry_data.get('pronunciation', '')),
                part_of_speech=metadata.get('part_of_speech', entry_data.get('part_of_speech', 'noun')),
                definition=entry_data.get('definition', ''),
                mnemonic_type=metadata.get('mnemonic_type', entry_data.get('mnemonic_type', 'Sounds like')),
                mnemonic_phrase=entry_data.get('mnemonic_phrase', ''),
                picture_story=entry_data.get('picture_story', ''),
                other_forms=entry_data.get('other_forms', ''),
                example_sentence=entry_data.get('example_sentence', ''),
                quality_score=metadata.get('quality_score', 0.0),
                source=metadata.get('source', 'unknown'),
                difficulty_level=metadata.get('difficulty_level', 'intermediate')
            )
            
        except Exception as e:
            logger.error(f"Failed to reconstruct entry: {e}")
            return None
    
    def get_contextual_examples(
        self, 
        word: str, 
        context_type: str = "similar_meaning",
        num_examples: int = 3
    ) -> List[VocabularyEntry]:
        """Get contextual examples based on different criteria"""
        try:
            if context_type == "similar_meaning":
                # Find entries with similar definitions
                results = self.search_similar_entries(f"definition similar to {word}", num_examples)
                
            elif context_type == "similar_sound":
                # Find phonetically similar words
                results = self.search_similar_entries(f"pronunciation sounds like {word}", num_examples)
                
            elif context_type == "same_pos":
                # Find entries with same part of speech
                results = self.search_similar_entries(f"{word} same part of speech", num_examples)
                
            elif context_type == "mnemonic_type":
                # Find entries with same mnemonic technique
                results = self.search_similar_entries(f"{word} similar mnemonic technique", num_examples)
                
            else:
                # Default to general similarity
                results = self.search_similar_entries(word, num_examples)
            
            return [entry for entry, score in results]
            
        except Exception as e:
            logger.error(f"Failed to get contextual examples: {e}")
            return []
    
    def add_positive_feedback(self, word: str, example: str, feedback_data: Dict = None):
        """Store positive feedback example"""
        try:
            positive_example = {
                "word": word,
                "example": example,
                "feedback_data": feedback_data or {},
                "timestamp": datetime.now().isoformat()
            }
            
            self.positive_examples.append(positive_example)
            self._save_feedback_data()
            
            logger.info(f"Added positive feedback for: {word}")
            
        except Exception as e:
            logger.error(f"Failed to add positive feedback: {e}")
    
    def add_negative_feedback(self, word: str, example: str, issues: List[str] = None):
        """Store negative feedback example"""
        try:
            negative_example = {
                "word": word,
                "example": example,
                "issues": issues or [],
                "timestamp": datetime.now().isoformat()
            }
            
            self.negative_examples.append(negative_example)
            self._save_feedback_data()
            
            logger.info(f"Added negative feedback for: {word}")
            
        except Exception as e:
            logger.error(f"Failed to add negative feedback: {e}")
    
    def get_feedback_context(self, word: str) -> Dict[str, List[str]]:
        """Get feedback context for word generation"""
        try:
            context = {
                "positive_examples": [],
                "negative_examples": [],
                "improvement_suggestions": []
            }
            
            # Find positive examples
            for example in self.positive_examples:
                if example["word"].lower() == word.lower():
                    context["positive_examples"].append(example["example"])
            
            # Find negative examples
            for example in self.negative_examples:
                if example["word"].lower() == word.lower():
                    context["negative_examples"].append(example["example"])
                    if example.get("issues"):
                        context["improvement_suggestions"].extend(example["issues"])
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get feedback context: {e}")
            return {"positive_examples": [], "negative_examples": [], "improvement_suggestions": []}
    
    def _load_feedback_data(self):
        """Load feedback data from storage"""
        try:
            feedback_path = Path(self.settings.feedback_storage_path)
            feedback_path.mkdir(parents=True, exist_ok=True)
            
            # Load positive examples
            positive_file = feedback_path / "positive_examples.json"
            if positive_file.exists():
                with open(positive_file, 'r', encoding='utf-8') as f:
                    self.positive_examples = json.load(f)
            
            # Load negative examples
            negative_file = feedback_path / "negative_examples.json"
            if negative_file.exists():
                with open(negative_file, 'r', encoding='utf-8') as f:
                    self.negative_examples = json.load(f)
                    
            logger.info(f"Loaded {len(self.positive_examples)} positive and {len(self.negative_examples)} negative examples")
            
        except Exception as e:
            logger.warning(f"Could not load feedback data: {e}")
    
    def _save_feedback_data(self):
        """Save feedback data to storage"""
        try:
            feedback_path = Path(self.settings.feedback_storage_path)
            feedback_path.mkdir(parents=True, exist_ok=True)
            
            # Save positive examples
            positive_file = feedback_path / "positive_examples.json"
            with open(positive_file, 'w', encoding='utf-8') as f:
                json.dump(self.positive_examples, f, indent=2, ensure_ascii=False)
            
            # Save negative examples
            negative_file = feedback_path / "negative_examples.json"
            with open(negative_file, 'w', encoding='utf-8') as f:
                json.dump(self.negative_examples, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save feedback data: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive RAG engine statistics for admin dashboard"""
        try:
            collection = self.vector_store._collection
            total_entries = collection.count()
            
            stats = {
                "total_entries": total_entries,
                "positive_feedback_count": len(self.positive_examples),
                "negative_feedback_count": len(self.negative_examples),
                "positive_examples": len(self.positive_examples),
                "negative_examples": len(self.negative_examples),
                "vector_store_collection": self.settings.vector_db_collection,
                "embedding_model": self.settings.embedding_model,
                "hybrid_retriever_enabled": self.hybrid_retriever is not None
            }
            
            # Get metadata statistics if we have entries
            if total_entries > 0:
                results = collection.get(include=['metadatas'])
                metadatas = results['metadatas']
                
                # Count by part of speech, mnemonic types, sources, etc.
                pos_counts = {}
                mnemonic_counts = {}
                source_counts = {}
                difficulty_counts = {}
                quality_scores = []
                
                # Quality distribution counters
                excellent_count = 0
                good_count = 0
                acceptable_count = 0
                poor_count = 0
                
                recent_entries = []
                top_quality_words = []
                
                for metadata in metadatas:
                    # Part of speech distribution
                    pos = metadata.get('part_of_speech', 'unknown')
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    
                    # Mnemonic type distribution
                    mnemonic = metadata.get('mnemonic_type', 'unknown')
                    mnemonic_counts[mnemonic] = mnemonic_counts.get(mnemonic, 0) + 1
                    
                    # Source distribution
                    source = metadata.get('source', 'unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
                    
                    # Difficulty distribution
                    difficulty = metadata.get('difficulty_level', 'intermediate')
                    difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
                    
                    # Quality analysis
                    quality_score = float(metadata.get('quality_score', 0.0))
                    quality_scores.append(quality_score)
                    
                    # Quality categorization
                    if quality_score >= 0.85:
                        excellent_count += 1
                    elif quality_score >= 0.70:
                        good_count += 1
                    elif quality_score >= 0.55:
                        acceptable_count += 1
                    else:
                        poor_count += 1
                    
                    # Recent entries (simplified)
                    word = metadata.get('word', 'Unknown')
                    created_at = metadata.get('created_at', datetime.now().isoformat())
                    recent_entries.append({
                        "word": word,
                        "quality_score": quality_score,
                        "part_of_speech": pos,
                        "created_at": created_at
                    })
                    
                    # Top quality words
                    top_quality_words.append({
                        "word": word,
                        "quality_score": quality_score,
                        "definition": metadata.get('definition', '')[:100] + "..." if len(metadata.get('definition', '')) > 100 else metadata.get('definition', '')
                    })
                
                # Sort and limit recent entries and top quality words
                recent_entries.sort(key=lambda x: x['created_at'], reverse=True)
                top_quality_words.sort(key=lambda x: x['quality_score'], reverse=True)
                
                # Calculate averages and additional metrics
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
                
                # Enhanced statistics
                stats.update({
                    "parts_of_speech": pos_counts,
                    "mnemonic_types": mnemonic_counts,
                    "sources": source_counts,
                    "difficulty_levels": difficulty_counts,
                    
                    # Quality metrics
                    "average_quality": avg_quality,
                    "total_assessments": len(quality_scores),
                    "excellent_count": excellent_count,
                    "good_count": good_count,
                    "acceptable_count": acceptable_count,
                    "poor_count": poor_count,
                    
                    # Generation statistics (mock data for now)
                    "total_generations": total_entries + 10,  # Assume some failed generations
                    "successful_generations": total_entries,
                    "failed_generations": 10,
                    "average_attempts": 1.2,
                    "success_rate": total_entries / (total_entries + 10) if total_entries > 0 else 0.0,
                    
                    # Distribution by specific counts
                    "noun_count": pos_counts.get('noun', 0),
                    "verb_count": pos_counts.get('verb', 0),
                    "adjective_count": pos_counts.get('adjective', 0),
                    "adverb_count": pos_counts.get('adverb', 0),
                    
                    "basic_count": difficulty_counts.get('basic', 0),
                    "intermediate_count": difficulty_counts.get('intermediate', 0),
                    "advanced_count": difficulty_counts.get('advanced', 0),
                    "expert_count": difficulty_counts.get('expert', 0),
                    
                    "sounds_like_count": mnemonic_counts.get('Sounds like', 0),
                    "looks_like_count": mnemonic_counts.get('Looks like', 0),
                    "think_of_count": mnemonic_counts.get('Think of', 0),
                    "connect_with_count": mnemonic_counts.get('Connect with', 0),
                    
                    # Feedback metrics
                    "total_feedback": len(self.positive_examples) + len(self.negative_examples),
                    "positive_feedback": len(self.positive_examples),
                    "negative_feedback": len(self.negative_examples),
                    "feedback_ratio": len(self.positive_examples) / (len(self.positive_examples) + len(self.negative_examples)) if (len(self.positive_examples) + len(self.negative_examples)) > 0 else 0.0,
                    "avg_satisfaction": 7.5 if len(self.positive_examples) > len(self.negative_examples) else 5.0,
                    
                    # Recent activity and top performers
                    "recent_entries": recent_entries[:10],
                    "top_quality_words": top_quality_words[:10],
                    
                    # Performance metrics (mock data)
                    "avg_generation_time": 2.5,
                    "avg_assessment_time": 0.8,
                    "avg_search_time": 0.3,
                    "cache_hit_rate": 0.15,
                    
                    # Timestamps
                    "last_generation": recent_entries[0]['created_at'] if recent_entries else None,
                    "last_feedback": datetime.now().isoformat() if (len(self.positive_examples) + len(self.negative_examples)) > 0 else None
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def clear_vector_store(self):
        """Clear all data from vector store (use with caution)"""
        try:
            self.vector_store.delete_collection()
            self._initialize_vector_store()
            logger.warning("Vector store cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")

    def get_all_entries(self) -> List[Any]:
        """Get all vocabulary entries from the vector store"""
        try:
            collection = self.vector_store._collection
            results = collection.get(include=['documents', 'metadatas'])
            
            entries = []
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                # Create a simple object with the entry data
                entry = type('VocabularyEntry', (), {
                    'word': metadata.get('word', 'Unknown'),
                    'definition': metadata.get('definition', ''),
                    'mnemonic_phrase': metadata.get('mnemonic_phrase', ''),
                    'pronunciation': metadata.get('pronunciation', ''),
                    'part_of_speech': metadata.get('part_of_speech', 'noun'),
                    'mnemonic_type': metadata.get('mnemonic_type', 'Sounds like'),
                    'quality_score': float(metadata.get('quality_score', 0.0)),
                    'source': metadata.get('source', 'generated'),
                    'difficulty_level': metadata.get('difficulty_level', 'intermediate'),
                    'example_sentence': metadata.get('example_sentence', ''),
                    'created_at': metadata.get('created_at', ''),
                    'id': metadata.get('id', f'entry_{i}')
                })()
                entries.append(entry)
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to get all entries: {e}")
            return []

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get detailed feedback statistics"""
        try:
            # This would be enhanced with actual feedback storage
            # For now, return mock data based on available information
            
            total_feedback = len(self.positive_examples) + len(self.negative_examples)
            positive_count = len(self.positive_examples)
            negative_count = len(self.negative_examples)
            
            stats = {
                "total_feedback": total_feedback,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "participation_rate": 0.0,  # Would need user session tracking
                "avg_satisfaction": 7.5 if positive_count > negative_count else 5.0,
                
                # Satisfaction distribution (mock data)
                "very_satisfied": max(0, positive_count - 2),
                "satisfied": min(positive_count, 3),
                "neutral": max(0, negative_count - 3),
                "dissatisfied": min(negative_count, 2),
                "very_dissatisfied": max(0, negative_count - 2),
                
                # Component feedback (mock data)
                "definition_issues": negative_count // 3,
                "mnemonic_issues": negative_count // 2,
                "example_issues": negative_count // 4,
                "pronunciation_issues": negative_count // 5,
                "other_issues": negative_count // 6,
                
                # Improvement tracking (mock data)
                "regenerations": negative_count,
                "avg_improvement": 0.15,  # Average quality improvement after feedback
                "successful_improvements": int(negative_count * 0.8),
                
                # Recent feedback (simplified)
                "recent_feedback": [
                    {
                        "word": example.split(':')[0] if ':' in example else example[:20],
                        "type": "negative",
                        "satisfaction": 3,
                        "timestamp": datetime.now().isoformat()
                    } for example in self.negative_examples[-5:]
                ] + [
                    {
                        "word": example.split(':')[0] if ':' in example else example[:20],
                        "type": "positive", 
                        "satisfaction": 8,
                        "timestamp": datetime.now().isoformat()
                    } for example in self.positive_examples[-5:]
                ]
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get feedback statistics: {e}")
            return {"error": str(e)}


# Global RAG engine instance
_rag_engine_instance = None


def get_rag_engine() -> AdvancedRAGEngine:
    """Get or create the global RAG engine instance"""
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = AdvancedRAGEngine()
    return _rag_engine_instance


def initialize_rag_engine(settings=None) -> AdvancedRAGEngine:
    """Initialize the RAG engine with custom settings"""
    global _rag_engine_instance
    _rag_engine_instance = AdvancedRAGEngine(settings)
    return _rag_engine_instance