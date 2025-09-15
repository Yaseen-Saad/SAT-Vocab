"""
LLM-based Vocabulary Generator with Gulotta Method Integration
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import json

# LangChain imports
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.callbacks.manager import get_openai_callback

# Local imports
from ..models import VocabularyEntry, QualityMetrics
from ..services.config import get_settings
from ..services.hackclub_ai import HackClubChatModel, get_hackclub_callback
from .quality_system import get_quality_system
from .rag_engine import get_rag_engine

logger = logging.getLogger(__name__)


class VocabularyGenerator:
    """Advanced vocabulary generator using LLM with quality-aware generation"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.llm = None
        self.chat_model = None
        self.quality_system = get_quality_system()
        self.rag_engine = get_rag_engine()
        
        # Generation templates
        self.gulotta_system_prompt = self._create_gulotta_system_prompt()
        self.generation_template = self._create_generation_template()
        self.improvement_template = self._create_improvement_template()
        
        # Output parser
        self.output_parser = PydanticOutputParser(pydantic_object=VocabularyEntry)
        
        # Initialize LLM
        self._initialize_llm()
        
        logger.info("Vocabulary Generator initialized")
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if self.settings.llm_provider == "hackclub":
                # Use Hack Club AI (free!)
                self.chat_model = HackClubChatModel(
                    model=self.settings.llm_model,
                    temperature=self.settings.generation_temperature,
                    max_tokens=self.settings.max_tokens
                )
                
                logger.info(f"Hack Club AI initialized: {self.settings.llm_model}")
                
            elif self.settings.llm_provider == "openai":
                # For structured output, use ChatOpenAI
                self.chat_model = ChatOpenAI(
                    model_name=self.settings.llm_model,
                    temperature=self.settings.generation_temperature,
                    openai_api_key=self.settings.openai_api_key,
                    max_tokens=self.settings.max_tokens
                )
                
                # For simple completions
                self.llm = OpenAI(
                    model_name=self.settings.llm_model,
                    temperature=self.settings.generation_temperature,
                    openai_api_key=self.settings.openai_api_key,
                    max_tokens=self.settings.max_tokens
                )
                
                logger.info(f"OpenAI initialized: {self.settings.llm_model}")
                
            else:
                raise ValueError(f"Unsupported LLM provider: {self.settings.llm_provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _create_gulotta_system_prompt(self) -> str:
        """Create system prompt based on Gulotta method"""
        return """You are an expert SAT vocabulary tutor specializing in the Gulotta method for memorable vocabulary learning.

The Gulotta method emphasizes:
1. SOUND-ALIKE MNEMONICS: Find words or phrases that sound like the target word
2. VIVID IMAGERY: Create memorable visual scenes that connect the sound-alike to the meaning
3. PICTURE STORIES: Develop detailed scenarios that make the connection unforgettable
4. MULTIPLE ASSOCIATIONS: Use various mnemonic techniques when appropriate

Your goal is to create vocabulary entries that help students remember SAT words permanently through:
- Clear, precise definitions appropriate for SAT level
- Effective sound-alike mnemonics that match the word's pronunciation
- Vivid picture stories that create strong memory associations
- Relevant example sentences showing proper usage
- Complete pronunciation guides and word forms

Always follow the Gulotta approach: find what the word SOUNDS LIKE, then create a PICTURE that connects the sound to the MEANING.

Focus on quality over quantity. Each entry should be memorable, accurate, and engaging."""
    
    def _create_generation_template(self) -> str:
        """Create the main generation template"""
        return """Generate a comprehensive SAT vocabulary entry for the word: {word}

Use the Gulotta method to create an effective mnemonic device:
1. Identify what the word sounds like (phonetic similarity)
2. Create a vivid picture story connecting the sound to the meaning
3. Ensure the mnemonic is memorable and appropriate for SAT students

{context_instructions}

{feedback_context}

Required format - provide a valid JSON object with these exact fields:
{{
    "word": "{word}",
    "pronunciation": "phonetic pronunciation with syllable breaks",
    "part_of_speech": "primary part of speech (noun/verb/adjective/adverb)",
    "definition": "clear, concise SAT-level definition",
    "mnemonic_type": "Sounds like" (or other appropriate type),
    "mnemonic_phrase": "what the word sounds like + memory connection",
    "picture_story": "detailed vivid scenario connecting sound to meaning",
    "other_forms": "related word forms (plural, verb forms, etc.)",
    "example_sentence": "SAT-appropriate sentence using the word",
    "quality_score": 0.0,
    "source": "generated",
    "difficulty_level": "advanced"
}}

Focus on creating a memorable, accurate entry that will help students remember this word permanently."""
    
    def _create_improvement_template(self) -> str:
        """Create template for improving existing entries"""
        return """Improve this SAT vocabulary entry based on the quality assessment and feedback:

Current Entry:
{current_entry}

Quality Issues Identified:
{improvement_suggestions}

User Feedback Context:
{feedback_context}

Create an improved version that addresses these issues while maintaining the Gulotta method approach.
Focus on: better mnemonics, clearer definitions, more vivid imagery, and stronger memory associations.

Provide the improved entry in the same JSON format."""
    
    async def generate_vocabulary_entry(
        self, 
        word: str, 
        context: Optional[str] = None,
        quality_threshold: float = 0.7,
        max_attempts: int = 3
    ) -> Optional[VocabularyEntry]:
        """Generate a high-quality vocabulary entry with multiple attempts if needed"""
        try:
            best_entry = None
            best_score = 0.0
            
            for attempt in range(max_attempts):
                logger.info(f"Generating vocabulary entry for '{word}' (attempt {attempt + 1}/{max_attempts})")
                
                # Generate entry
                entry = await self._generate_single_entry(word, context, attempt)
                
                if not entry:
                    continue
                
                # Get feedback context
                feedback_context = self.rag_engine.get_feedback_context(word)
                
                # Assess quality
                assessment = self.quality_system.assess_vocabulary_entry(entry, feedback_context)
                entry.quality_score = assessment.overall_score
                
                # Determine quality level based on score
                quality_level = self._get_quality_level(assessment.overall_score)
                logger.info(f"Generated entry quality: {quality_level} ({assessment.overall_score:.3f})")
                
                # Check if meets threshold
                if assessment.overall_score >= quality_threshold:
                    logger.info(f"Quality threshold met on attempt {attempt + 1}")
                    return entry
                
                # Keep best attempt
                if assessment.overall_score > best_score:
                    best_entry = entry
                    best_score = assessment.overall_score
                
                # If not last attempt, try improvement
                if attempt < max_attempts - 1 and best_entry:
                    logger.info(f"Attempting to improve entry (current score: {best_score:.3f})")
                    improved_entry = await self._improve_entry(best_entry, assessment.improvement_suggestions, feedback_context)
                    if improved_entry:
                        entry = improved_entry
            
            if best_entry:
                logger.warning(f"Returning best entry for '{word}' with score {best_score:.3f}")
                return best_entry
            else:
                logger.error(f"Failed to generate any valid entry for '{word}'")
                return None
                
        except Exception as e:
            logger.error(f"Vocabulary generation failed for '{word}': {e}")
            return None
    
    async def _generate_single_entry(
        self, 
        word: str, 
        context: Optional[str] = None,
        attempt: int = 0
    ) -> Optional[VocabularyEntry]:
        """Generate a single vocabulary entry"""
        try:
            # Get similar entries for context
            similar_entries = self.rag_engine.search_similar_entries(word, top_k=3)
            
            # Prepare context instructions
            context_instructions = ""
            if similar_entries:
                context_instructions = f"""
Consider these similar vocabulary entries for context and inspiration:
{self._format_similar_entries(similar_entries)}

Ensure your entry is unique and uses different mnemonic approaches.
"""
            
            if context:
                context_instructions += f"\nAdditional context: {context}"
            
            # Get feedback context
            feedback_context = self.rag_engine.get_feedback_context(word)
            feedback_text = self._format_feedback_context(feedback_context)
            
            # Add variation for multiple attempts
            if attempt > 0:
                context_instructions += f"\nThis is attempt {attempt + 1}. Try a different mnemonic approach or more creative imagery."
            
            # Create prompt
            prompt_text = self.generation_template.format(
                word=word,
                context_instructions=context_instructions,
                feedback_context=feedback_text
            )
            
            # Generate with LLM
            messages = [
                SystemMessage(content=self.gulotta_system_prompt),
                HumanMessage(content=prompt_text)
            ]
            
            # Use appropriate callback based on provider
            if self.settings.llm_provider == "hackclub":
                with get_hackclub_callback() as cb:
                    response = await self.chat_model.agenerate([messages])
                    logger.info(f"Hack Club AI usage: Free! 🎉")
            else:
                with get_openai_callback() as cb:
                    response = await self.chat_model.agenerate([messages])
                    logger.info(f"LLM usage: {cb.total_tokens} tokens (${cb.total_cost:.4f})")
            
            # Parse response
            raw_response = response.generations[0][0].text
            logger.debug(f"Raw response type: {type(raw_response)}")
            logger.debug(f"Raw response: {raw_response}")
            
            # Handle case where response_text might be a list
            if isinstance(raw_response, list):
                response_text = ' '.join(str(item) for item in raw_response)
            else:
                response_text = str(raw_response)
            
            response_text = response_text.strip()
            entry = self._parse_vocabulary_entry(response_text, word)
            
            return entry
            
        except Exception as e:
            logger.error(f"Single entry generation failed: {e}")
            return None
    
    async def _improve_entry(
        self, 
        entry: VocabularyEntry, 
        suggestions: List[str],
        feedback_context: Dict
    ) -> Optional[VocabularyEntry]:
        """Improve an existing entry based on suggestions"""
        try:
            logger.info(f"Improving entry for '{entry.word}'")
            
            # Format current entry
            current_entry_text = self._format_entry_for_improvement(entry)
            
            # Format feedback
            feedback_text = self._format_feedback_context(feedback_context)
            
            # Create improvement prompt
            prompt_text = self.improvement_template.format(
                current_entry=current_entry_text,
                improvement_suggestions="\n".join(f"- {s}" for s in suggestions),
                feedback_context=feedback_text
            )
            
            messages = [
                SystemMessage(content=self.gulotta_system_prompt + "\n\nYou are now improving an existing entry. Make specific enhancements while preserving what works well."),
                HumanMessage(content=prompt_text)
            ]
            
            # Use appropriate callback based on provider
            if self.settings.llm_provider == "hackclub":
                with get_hackclub_callback() as cb:
                    response = await self.chat_model.agenerate([messages])
                    logger.info(f"Improvement via Hack Club AI: Free! 🎉")
            else:
                with get_openai_callback() as cb:
                    response = await self.chat_model.agenerate([messages])
                    logger.info(f"Improvement LLM usage: {cb.total_tokens} tokens (${cb.total_cost:.4f})")
            
            # Parse improved entry
            raw_response = response.generations[0][0].text
            logger.debug(f"Raw improved response type: {type(raw_response)}")
            logger.debug(f"Raw improved response: {raw_response}")
            
            # Handle case where response_text might be a list
            if isinstance(raw_response, list):
                response_text = ' '.join(str(item) for item in raw_response)
            else:
                response_text = str(raw_response)
            
            response_text = response_text.strip()
            improved_entry = self._parse_vocabulary_entry(response_text, entry.word)
            
            if improved_entry:
                logger.info(f"Successfully improved entry for '{entry.word}'")
                return improved_entry
            else:
                logger.warning(f"Failed to parse improved entry for '{entry.word}'")
                return None
                
        except Exception as e:
            logger.error(f"Entry improvement failed: {e}")
            return None
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score"""
        if score >= 0.85:
            return "excellent"
        elif score >= 0.70:
            return "good"
        elif score >= 0.55:
            return "acceptable"
        else:
            return "needs_improvement"
    
    def _safe_string(self, value, default=''):
        """Safely convert value to string, handling lists and other types"""
        if value is None:
            return default
        elif isinstance(value, list):
            return ' '.join(str(item) for item in value)
        elif isinstance(value, str):
            return value
        else:
            return str(value)
    
    def _parse_vocabulary_entry(self, response_text: str, word: str) -> Optional[VocabularyEntry]:
        """Parse LLM response into VocabularyEntry object"""
        try:
            # Extract JSON from response
            json_text = self._extract_json_from_response(response_text)
            
            if not json_text:
                logger.error("No valid JSON found in LLM response")
                return None
            
            # Parse JSON
            entry_data = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['word', 'definition', 'mnemonic_phrase']
            for field in required_fields:
                field_value = entry_data.get(field, '')
                if isinstance(field_value, list):
                    field_value = ' '.join(str(item) for item in field_value)
                elif not isinstance(field_value, str):
                    field_value = str(field_value)
                
                if not field_value.strip():
                    logger.warning(f"Missing or empty required field: {field}")
                    return None
            
            # Create VocabularyEntry with defaults
            entry = VocabularyEntry(
                word=self._safe_string(entry_data.get('word', word)).strip(),
                pronunciation=self._safe_string(entry_data.get('pronunciation', '')).strip(),
                part_of_speech=self._safe_string(entry_data.get('part_of_speech', 'noun')).strip(),
                definition=self._safe_string(entry_data.get('definition', '')).strip(),
                mnemonic_type=self._safe_string(entry_data.get('mnemonic_type', 'Sounds like')).strip(),
                mnemonic_phrase=self._safe_string(entry_data.get('mnemonic_phrase', '')).strip(),
                picture_story=self._safe_string(entry_data.get('picture_story', '')).strip(),
                other_forms=self._safe_string(entry_data.get('other_forms', '')).strip(),
                example_sentence=self._safe_string(entry_data.get('example_sentence', '')).strip(),
                quality_score=entry_data.get('quality_score', 0.0),
                source=entry_data.get('source', 'generated'),
                difficulty_level=entry_data.get('difficulty_level', 'advanced'),
                created_at=datetime.now()
            )
            
            logger.info(f"Successfully parsed vocabulary entry for '{word}'")
            return entry
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response text: {response_text[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Entry parsing failed: {e}")
            return None
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract JSON object from LLM response"""
        try:
            # Find JSON object boundaries
            start_idx = response_text.find('{')
            if start_idx == -1:
                return None
            
            # Find matching closing brace
            brace_count = 0
            end_idx = start_idx
            
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if brace_count != 0:
                return None
            
            json_text = response_text[start_idx:end_idx]
            return json_text
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {e}")
            return None
    
    def _format_similar_entries(self, similar_entries: List[tuple]) -> str:
        """Format similar entries for context"""
        if not similar_entries:
            return "No similar entries found."
        
        formatted = []
        for entry, score in similar_entries:
            formatted.append(f"""
Word: {entry.word}
Mnemonic: {entry.mnemonic_phrase[:100]}...
Quality Score: {score:.2f}
""")
        
        return "\n".join(formatted)
    
    def _format_feedback_context(self, feedback_context: Dict) -> str:
        """Format feedback context for prompts"""
        if not feedback_context:
            return "No feedback context available."
        
        text_parts = []
        
        if feedback_context.get("positive_examples"):
            text_parts.append(f"Positive examples: {', '.join(feedback_context['positive_examples'][:3])}")
        
        if feedback_context.get("negative_examples"):
            text_parts.append(f"Negative examples: {', '.join(feedback_context['negative_examples'][:3])}")
        
        if feedback_context.get("improvement_suggestions"):
            text_parts.append(f"Suggestions: {', '.join(feedback_context['improvement_suggestions'][:3])}")
        
        return "\n".join(text_parts) if text_parts else "No specific feedback available."
    
    def _format_entry_for_improvement(self, entry: VocabularyEntry) -> str:
        """Format entry for improvement prompt"""
        return f"""
Word: {entry.word}
Pronunciation: {entry.pronunciation}
Part of Speech: {entry.part_of_speech}
Definition: {entry.definition}
Mnemonic Type: {entry.mnemonic_type}
Mnemonic Phrase: {entry.mnemonic_phrase}
Picture Story: {entry.picture_story}
Other Forms: {entry.other_forms}
Example Sentence: {entry.example_sentence}
Current Quality Score: {entry.quality_score:.3f}
"""
    
    async def batch_generate_entries(
        self, 
        words: List[str], 
        quality_threshold: float = 0.7
    ) -> List[VocabularyEntry]:
        """Generate multiple vocabulary entries efficiently"""
        try:
            logger.info(f"Starting batch generation for {len(words)} words")
            
            # Create tasks for concurrent generation
            tasks = []
            for word in words:
                task = self.generate_vocabulary_entry(word, quality_threshold=quality_threshold)
                tasks.append(task)
            
            # Execute with limited concurrency to avoid rate limits
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
            
            async def limited_generate(task):
                async with semaphore:
                    return await task
            
            limited_tasks = [limited_generate(task) for task in tasks]
            results = await asyncio.gather(*limited_tasks, return_exceptions=True)
            
            # Filter successful results
            successful_entries = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to generate entry for '{words[i]}': {result}")
                elif result is not None:
                    successful_entries.append(result)
                else:
                    logger.warning(f"No entry generated for '{words[i]}'")
            
            logger.info(f"Batch generation completed: {len(successful_entries)}/{len(words)} successful")
            return successful_entries
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return []
    
    async def regenerate_with_feedback(
        self, 
        word: str, 
        feedback: str, 
        feedback_type: str = "negative"
    ) -> Optional[VocabularyEntry]:
        """Regenerate entry incorporating specific feedback"""
        try:
            logger.info(f"Regenerating '{word}' with {feedback_type} feedback")
            
            # Store feedback
            if feedback_type == "positive":
                self.rag_engine.add_positive_feedback(word, feedback)
            else:
                self.rag_engine.add_negative_feedback(word, feedback, [feedback])
            
            # Generate new entry with updated feedback context
            entry = await self.generate_vocabulary_entry(word, context=f"Address this feedback: {feedback}")
            
            if entry:
                logger.info(f"Successfully regenerated entry for '{word}' with feedback")
            else:
                logger.error(f"Failed to regenerate entry for '{word}' with feedback")
            
            return entry
            
        except Exception as e:
            logger.error(f"Feedback regeneration failed: {e}")
            return None


# Global generator instance
_generator_instance = None


def get_vocabulary_generator() -> VocabularyGenerator:
    """Get or create the global vocabulary generator"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = VocabularyGenerator()
    return _generator_instance


async def generate_vocabulary_entry_async(word: str, **kwargs) -> Optional[VocabularyEntry]:
    """Convenience function for async vocabulary generation"""
    generator = get_vocabulary_generator()
    return await generator.generate_vocabulary_entry(word, **kwargs)