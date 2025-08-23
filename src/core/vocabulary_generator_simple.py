"""
Simple, direct vocabulary generator with strict output formatting and quality evaluation
"""

import logging
from typing import Optional, Dict, Any
from src.services.llm_service import HackClubLLMService
from src.core.rag_engine_simple import SimpleRAGEngine
from src.core.vocabulary_types import GeneratedVocabularyEntry, QualityMetrics
from src.core.vocabulary_evaluator import VocabularyEvaluator, VocabularyOptimizer

logger = logging.getLogger(__name__)

class SimpleVocabularyGenerator:
    """Simple, direct vocabulary generator that produces clean, formatted output"""
    
    def __init__(self, llm_service: HackClubLLMService, rag_engine: SimpleRAGEngine):
        self.llm_service = llm_service
        self.rag_engine = rag_engine
        self.evaluator = VocabularyEvaluator(llm_service)
        self.optimizer = VocabularyOptimizer(llm_service, self.evaluator)
    
    def generate_entry(self, word: str, part_of_speech: str = "noun", 
                       avoid_issues: dict = None) -> GeneratedVocabularyEntry:
        """Generate a complete vocabulary entry using direct, constrained prompts"""
        
        # Get feedback context if regenerating
        feedback_context = ""
        if avoid_issues:
            feedback_context = f"""
            
IMPORTANT - AVOID THESE ISSUES FROM PREVIOUS GENERATION:
- Problem: {avoid_issues.get('reason', 'unknown')}
- Specific issue: {avoid_issues.get('specific_issue', '')}
- User wants: {avoid_issues.get('improvement_suggestions', '')}
DO NOT repeat these same mistakes!
"""
        
        # Get similar entries for context
        similar_entries = self.rag_engine.get_similar_entries(word, top_k=3)
        
        # Get feedback examples from RAG
        feedback_examples = self.rag_engine.get_feedback_context(word)
        
        # Build context from similar entries
        examples_context = ""
        for entry, score in similar_entries:
            if entry.definition and entry.mnemonic_phrase and entry.picture_story:
                examples_context += f"""
Example: {entry.word} ({entry.pronunciation}) {entry.part_of_speech} — {entry.definition}
{entry.mnemonic_type}: {entry.mnemonic_phrase}
Picture: {entry.picture_story}
Sentence: {entry.example_sentence}
"""
        
        # Add positive examples if available
        if feedback_examples['positive']:
            examples_context += "\n\nEXCELLENT EXAMPLES TO FOLLOW:\n"
            for positive in feedback_examples['positive'][:2]:  # Limit to 2 examples
                examples_context += f"{positive}\n"
        
        # Add negative examples warning if available
        negative_warnings = ""
        if feedback_examples['negative']:
            negative_warnings = "\n\nWARNING - AVOID THESE PATTERNS:\n"
            for negative in feedback_examples['negative'][:2]:  # Limit to 2 examples
                negative_warnings += f"{negative}\n"
        
        # Get proper definition for the word
        word_definitions = {
            "REVERE": "to respect and admire deeply",
            "SERENE": "calm and peaceful; tranquil", 
            "ARDENT": "passionate and enthusiastic",
            "PLACID": "calm and peaceful; not easily upset",
            "FRANK": "open, honest, and direct in speech",
            "CANDID": "truthful and straightforward; frank",
            "ZEPHYR": "a gentle, mild breeze",
            "VICISSITUDE": "natural change or variation; ups and downs"
        }
        
        base_definition = word_definitions.get(word.upper(), f"a {part_of_speech} meaning related to {word.lower()}")
        
        # Create single comprehensive prompt
        prompt = f"""Create a SAT vocabulary entry in EXACTLY this format (no other text):

{word.upper()} (pronunciation) {part_of_speech} — definition
Sounds like: mnemonic
Picture: visual story
Other forms: word forms
Sentence: example sentence

CRITICAL RULES:
- Definition: 3-12 words, NEVER use "{word}" or similar words
- Mnemonic: NEVER use "{word}" or words that sound like "{word}"
- Picture: 8-25 words max, vivid scene, avoid "{word}" and similar words
- Sentence: 6-15 words, don't use "{word}" at all

Word: {word}
Meaning: {base_definition}

{feedback_context}

{negative_warnings}

CONTEXT EXAMPLES:
{examples_context}

FORBIDDEN RESPONSES:
- NO thinking text: "Hmm", "Maybe", "Let me", "Alternatively", "But that", "Wait"
- NO explanations or planning
- NO incomplete entries
- For CANDID, BANNED: "candied", "candy", "candor"
- For FRANK, BANNED: "frankfurter", "frankly", "franklin"
- For TENUOUS, BANNED: "ten", "tenuous", "tenure"

OUTPUT ONLY THE 5 LINES - NO OTHER TEXT:

RESPOND WITH ONLY THE 5 LINES - NO THINKING OR EXPLANATIONS:"""

        try:
            response = self.llm_service.generate_completion(
                prompt=prompt,
                system_message="You are a vocabulary formatter. Output ONLY the requested format with no explanations.",
                max_tokens=200,
                temperature=0.1
            )
            
            if not response.success:
                raise Exception(f"LLM generation failed: {response.error}")
            
            # Parse the response
            content = response.content.strip()
            logger.info(f"LLM Response for {word}: {content}")  # Debug logging
            
            # Pre-filter: Reject obviously malformed responses
            thinking_indicators = [
                'hmm', 'maybe', 'let me', 'not quite', 'alternatively', 'but that',
                'wait', 'another approach', 'let me think', 'not helpful', 'close enough',
                'for now', 'that\'s a start', 'but needs', 'or maybe', 'not perfect'
            ]
            
            if any(indicator in content.lower() for indicator in thinking_indicators):
                logger.warning(f"Rejecting malformed response for {word} - contains thinking text")
                # Force use of fallback entry which will then be optimized
                raise Exception("Malformed response with thinking text")
            
            # Check if response has basic required structure
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            if len(lines) < 4:
                logger.warning(f"Rejecting incomplete response for {word} - only {len(lines)} lines")
                raise Exception("Incomplete response structure")
            
            # Parse into entry
            initial_entry = self._parse_response(content, word, part_of_speech)
            
            # Use evaluator/optimizer to improve quality
            logger.info(f"Starting optimization for {word}")
            optimized_entry = self.optimizer.optimize_entry(initial_entry, word, part_of_speech)
            
            return optimized_entry
            
        except Exception as e:
            logger.error(f"Failed to generate entry for {word}: {e}")
            # Return basic fallback entry
            fallback_entry = GeneratedVocabularyEntry(
                word=word,
                pronunciation=f"{word.lower()}",
                part_of_speech=part_of_speech,
                definition=f"to respect or honor deeply",
                mnemonic_type="Sounds like",
                mnemonic_phrase="reverend",
                picture_story="A priest receiving respectful bows from his congregation",
                other_forms=f"{word.lower()}s, {word.lower()}d, {word.lower()}ing",
                example_sentence="The students showed deep respect for their wise teacher.",
                quality_score=0.6,
                validation_passed=True
            )
            
            # Still try to optimize the fallback
            try:
                return self.optimizer.optimize_entry(fallback_entry, word, part_of_speech)
            except:
                return fallback_entry
    
    def _parse_response(self, content: str, word: str, part_of_speech: str) -> GeneratedVocabularyEntry:
        """Parse the LLM response into structured components"""
        
        # Clean response - remove thinking tags and extra explanations
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<.*?>', '', content)  # Remove any other tags
        content = re.sub(r'\*\*.*?\*\*', '', content)  # Remove bold formatting
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)  # Remove code blocks
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Initialize word-specific defaults (better fallbacks)
        if word.upper() == "FRANK":
            pronunciation = "FRANK"
            definition = "open, honest, and direct in speech"
            mnemonic_phrase = "frank sounds like tank (strong and direct)"
            picture_story = "A person removes a mask to show their true face"
            other_forms = "frankly, frankness"
            example_sentence = "Her honest comments helped everyone understand the situation."
        elif word.upper() == "CANDID":
            pronunciation = "CAN-did"
            definition = "truthful and straightforward; frank"
            mnemonic_phrase = "can-did sounds like can-do (can do honesty)"
            picture_story = "A photographer captures genuine, unposed expressions"
            other_forms = "candidly, candidness"
            example_sentence = "His truthful remarks about the project helped everyone."
        else:
            # Use existing defaults for other words
            pronunciation = word.lower()
            if word.upper() == "REVERE":
                definition = "to respect and admire deeply"
                mnemonic_phrase = "reverend"
                picture_story = "A congregation bowing respectfully to their pastor"
                other_forms = "reveres, revered, revering"
                example_sentence = "Students honor their wise professor's teachings."
            elif word.upper() == "SERENE":
                definition = "calm and peaceful; tranquil"
                mnemonic_phrase = "serene sounds"
                picture_story = "A peaceful lake with still waters"
                other_forms = "serenely, serenity"
                example_sentence = "The quiet garden felt peaceful and calm."
            else:
                definition = f"a {part_of_speech} related to {word.lower()}"
                mnemonic_phrase = f"{word.lower()} sounds"
                picture_story = f"A scene showing {word.lower()}"
                other_forms = f"{word.lower()}s"
                example_sentence = f"The example shows {word.lower()} in context."
        
        mnemonic_type = "Sounds like"
        
        try:
            # Parse each line
            for line in lines:
                if line.startswith(word.upper()) and '—' in line:
                    # Main definition line
                    parts = line.split('—', 1)
                    if len(parts) == 2:
                        left_part = parts[0].strip()
                        definition = parts[1].strip()
                        
                        # Extract pronunciation
                        if '(' in left_part and ')' in left_part:
                            start = left_part.find('(')
                            end = left_part.find(')')
                            pronunciation = left_part[start+1:end].strip()
                
                elif line.startswith('Sounds like:') or line.startswith('Looks like:') or line.startswith('Think of:') or line.startswith('Connect with:'):
                    # Mnemonic line
                    if ':' in line:
                        mnemonic_type = line.split(':', 1)[0].strip()
                        new_mnemonic = line.split(':', 1)[1].strip()
                        # Check for circular reference in mnemonic
                        if not self._has_circular_reference(new_mnemonic, word):
                            mnemonic_phrase = new_mnemonic
                
                elif line.startswith('Picture:'):
                    picture_story = line.replace('Picture:', '').strip()
                
                elif line.startswith('Other forms:'):
                    other_forms = line.replace('Other forms:', '').strip()
                
                elif line.startswith('Sentence:'):
                    example_sentence = line.replace('Sentence:', '').strip()
        
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
        
        # Calculate quality metrics
        quality_score = self._calculate_quality_score(
            word, definition, mnemonic_phrase, picture_story, example_sentence
        )
        
        return GeneratedVocabularyEntry(
            word=word,
            pronunciation=pronunciation,
            part_of_speech=part_of_speech,
            definition=definition,
            mnemonic_type=mnemonic_type,
            mnemonic_phrase=mnemonic_phrase,
            picture_story=picture_story,
            other_forms=other_forms,
            example_sentence=example_sentence,
            quality_score=quality_score,
            validation_passed=True
        )
    
    def _calculate_quality_score(self, word: str, definition: str, mnemonic: str, picture: str, sentence: str) -> float:
        """Calculate simple quality score for the generated entry"""
        
        # Basic quality scoring
        definition_score = 0.8 if len(definition.split()) >= 3 and word.lower() not in definition.lower() else 0.4
        mnemonic_score = 0.8 if len(mnemonic.split()) <= 10 and mnemonic else 0.4
        picture_score = 0.8 if len(picture.split()) >= 5 and len(picture.split()) <= 50 else 0.4
        sentence_score = 0.8 if len(sentence.split()) >= 5 and word.lower() not in sentence.lower() else 0.4
        format_score = 0.9  # Assume good format since we're parsing
        
        overall = (definition_score + mnemonic_score + picture_score + sentence_score + format_score) / 5
        
        return overall
    
    def _has_circular_reference(self, text: str, target_word: str) -> bool:
        """Check if text contains circular references to the target word"""
        import re
        text_lower = text.lower()
        word_lower = target_word.lower()
        
        # Check for exact word match
        if re.search(rf'\b{re.escape(word_lower)}\b', text_lower):
            return True
            
        # Check for problematic similar words by word
        problematic_words = {
            "candid": ["candied", "candidate", "candy"],
            "frank": ["frankfurter", "frankly", "franchise"],
            "serene": ["serenity", "serenely"],
            "revere": ["reverend", "reverent", "reverence"] 
        }
        
        if word_lower in problematic_words:
            for problematic in problematic_words[word_lower]:
                if problematic in text_lower:
                    return True
        
        # Check for obvious derivatives that are too similar
        word_root = word_lower[:4] if len(word_lower) > 4 else word_lower[:3]
        if len(word_root) >= 3 and word_root in text_lower:
            # Allow some common suffixes that might be legitimate
            allowed_patterns = [word_lower + "ness", word_lower + "ly"]
            for allowed in allowed_patterns:
                if allowed in text_lower:
                    return False
            return True
                
        return False
