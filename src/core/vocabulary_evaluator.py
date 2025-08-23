"""
Vocabulary Entry Evaluator and Optimizer
"""

import logging
import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from src.core.vocabulary_types import GeneratedVocabularyEntry
from src.services.llm_service import HackClubLLMService

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of vocabulary entry evaluation"""
    is_valid: bool
    overall_score: float
    issues: List[str]
    suggestions: List[str]
    component_scores: Dict[str, float]

class VocabularyEvaluator:
    """Evaluates vocabulary entries for quality and authenticity"""
    
    def __init__(self, llm_service: HackClubLLMService):
        self.llm_service = llm_service
    
    def evaluate_entry(self, entry: GeneratedVocabularyEntry, target_word: str) -> EvaluationResult:
        """Comprehensive evaluation of a vocabulary entry"""
        
        issues = []
        suggestions = []
        scores = {}
        
        # Pre-check: Reject entries with obvious thinking text
        all_text = f"{entry.definition} {entry.mnemonic_phrase} {entry.picture_story} {entry.example_sentence}"
        thinking_indicators = [
            'hmm', 'let me', 'maybe', 'not quite', 'alternatively', 'but that',
            'wait', 'i think', 'another approach', 'let me think', 'not helpful',
            'close enough', 'for now', 'that\'s a start', 'but needs', 'or maybe',
            'not perfect', 'alternatively', 'but maybe better', 'the key is'
        ]
        
        has_thinking = any(indicator in all_text.lower() for indicator in thinking_indicators)
        if has_thinking:
            issues.append("CRITICAL: Contains thinking/planning text")
            issues.append("CRITICAL: Entry format is completely broken")
            return EvaluationResult(
                is_valid=False,
                overall_score=0.0,
                issues=issues + ["REJECT: Contains extensive thinking text"],
                suggestions=["Completely regenerate with proper format"],
                component_scores={"all": 0.0}
            )
        
        # 1. Check for circular references
        circular_score, circular_issues = self._check_circular_references(entry, target_word)
        scores['circular_avoidance'] = circular_score
        issues.extend(circular_issues)
        
        # 2. Check definition quality
        def_score, def_issues = self._check_definition_quality(entry, target_word)
        scores['definition'] = def_score
        issues.extend(def_issues)
        
        # 3. Check mnemonic effectiveness
        mnem_score, mnem_issues = self._check_mnemonic_quality(entry, target_word)
        scores['mnemonic'] = mnem_score
        issues.extend(mnem_issues)
        
        # 4. Check picture story quality
        pic_score, pic_issues = self._check_picture_quality(entry, target_word)
        scores['picture'] = pic_score
        issues.extend(pic_issues)
        
        # 5. Check example sentence
        sent_score, sent_issues = self._check_sentence_quality(entry, target_word)
        scores['sentence'] = sent_score
        issues.extend(sent_issues)
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores)
        
        # Determine if entry is valid (no critical issues)
        critical_keywords = ['CRITICAL', 'REJECT', 'uses the word', 'thinking']
        has_critical_issues = any(any(keyword.lower() in issue.lower() for keyword in critical_keywords) for issue in issues)
        
        # Be more lenient if the entry is mostly good
        minor_issues = ['too long', 'too short', 'should start', 'should end']
        only_minor_issues = all(any(minor.lower() in issue.lower() for minor in minor_issues) for issue in issues)
        
        if only_minor_issues and overall_score >= 0.7:
            is_valid = True
        else:
            is_valid = overall_score >= 0.8 and not has_critical_issues and len(issues) < 3
        
        # Generate suggestions
        if not is_valid:
            suggestions = self._generate_suggestions(entry, target_word, issues)
        
        return EvaluationResult(
            is_valid=is_valid,
            overall_score=overall_score,
            issues=issues,
            suggestions=suggestions,
            component_scores=scores
        )
    
    def _check_circular_references(self, entry: GeneratedVocabularyEntry, target_word: str) -> Tuple[float, List[str]]:
        """Check for circular references in all components"""
        issues = []
        score = 1.0
        
        word_lower = target_word.lower()
        word_root = word_lower[:4] if len(word_lower) > 4 else word_lower
        
        # Check definition
        if self._contains_word_or_similar(entry.definition, word_lower, word_root):
            issues.append(f"CRITICAL: Definition uses '{target_word}' or similar")
            score -= 0.4
        
        # Check mnemonic
        if self._contains_word_or_similar(entry.mnemonic_phrase, word_lower, word_root):
            issues.append(f"CRITICAL: Mnemonic uses '{target_word}' or similar")
            score -= 0.3
        
        # Check picture
        if self._contains_word_or_similar(entry.picture_story, word_lower, word_root):
            issues.append(f"CRITICAL: Picture uses '{target_word}' or similar")
            score -= 0.2
        
        # Check example sentence
        if self._contains_word_or_similar(entry.example_sentence, word_lower, word_root):
            issues.append(f"CRITICAL: Example sentence uses '{target_word}' or similar")
            score -= 0.4
        
        return max(score, 0.0), issues
    
    def _contains_word_or_similar(self, text: str, word: str, word_root: str) -> bool:
        """Check if text contains the word or very similar words"""
        text_lower = text.lower()
        
        # Exact word match
        if re.search(rf'\b{re.escape(word)}\b', text_lower):
            return True
        
        # Root similarity - be more conservative about catching variants
        if len(word_root) > 4:  # Only check roots longer than 4 chars
            pattern = rf'\b{re.escape(word_root)}\w+'  # Must be start of word with additional chars
            matches = re.findall(pattern, text_lower)
            # Only flag if the match is significantly longer than the root
            for match in matches:
                if len(match) > len(word_root) + 2:  # At least 3 chars longer
                    return True
        
        # Specific problematic patterns - be very strict but smarter
        problematic_pairs = {
            'candid': ['candied', 'candy', 'candies', 'candor'],
            'frank': ['frankfurter', 'franklin', 'frankish', 'franco'],
            'ardent': ['ardently', 'ardor'],
            'serene': ['serenity', 'serenade'], 
            'revere': ['reverend', 'reverent', 'reverence', 'revert'],
            'placid': ['placidly'],
            'tenuous': ['tenure', 'tenuous'],
            'demagogue': [],  # Let creative approaches work
            'supplant': []    # "plant" is actually a good mnemonic component
        }
        
        if word in problematic_pairs:
            for similar in problematic_pairs[word]:
                if similar in text_lower:
                    return True
        
        # Check for any word that starts with the same 4+ letters (more conservative)
        if len(word) >= 5:
            prefix = word[:4]
            words_in_text = re.findall(r'\b\w+', text_lower)
            for text_word in words_in_text:
                if len(text_word) > len(word) and text_word.startswith(prefix):
                    return True
        
        return False
    
    def _check_definition_quality(self, entry: GeneratedVocabularyEntry, target_word: str) -> Tuple[float, List[str]]:
        """Check definition quality"""
        issues = []
        score = 1.0
        
        definition = entry.definition.strip()
        word_count = len(definition.split())
        
        # Length check
        if word_count < 3:
            issues.append("Definition too short (min 3 words)")
            score -= 0.3
        elif word_count > 15:
            issues.append("Definition too long (max 15 words)")
            score -= 0.2
        
        # Quality check
        if not definition or definition == "":
            issues.append("CRITICAL: Empty definition")
            score = 0.0
        
        # Basic grammar check - allow "to" at start for infinitive definitions
        if definition and not definition[0].isupper() and not definition.lower().startswith('to '):
            issues.append("Definition should start with capital letter or 'to'")
            score -= 0.1
        
        return max(score, 0.0), issues
    
    def _check_mnemonic_quality(self, entry: GeneratedVocabularyEntry, target_word: str) -> Tuple[float, List[str]]:
        """Check mnemonic quality"""
        issues = []
        score = 1.0
        
        mnemonic = entry.mnemonic_phrase.strip()
        word_count = len(mnemonic.split())
        
        # Length check
        if word_count > 15:
            issues.append("Mnemonic too long (max 15 words)")
            score -= 0.2
        
        # Quality check
        if not mnemonic:
            issues.append("CRITICAL: Empty mnemonic")
            score = 0.0
        
        # Check for valid mnemonic type
        valid_types = ["Sounds like", "Looks like", "Think of", "Connect with"]
        if entry.mnemonic_type not in valid_types:
            issues.append(f"Invalid mnemonic type: {entry.mnemonic_type}")
            score -= 0.2
        
        return max(score, 0.0), issues
    
    def _check_picture_quality(self, entry: GeneratedVocabularyEntry, target_word: str) -> Tuple[float, List[str]]:
        """Check picture story quality"""
        issues = []
        score = 1.0
        
        picture = entry.picture_story.strip()
        word_count = len(picture.split())
        
        # Length check
        if word_count < 5:
            issues.append("Picture too short (min 5 words)")
            score -= 0.3
        elif word_count > 50:
            issues.append("Picture too long (max 50 words)")
            score -= 0.2
        
        # Quality check
        if not picture:
            issues.append("CRITICAL: Empty picture story")
            score = 0.0
        
        # Check for verbose thinking - be more aggressive
        thinking_patterns = [
            '<think>', '</think>', 'Maybe', 'For example', 'Yeah, that makes sense',
            'Need to make sure', 'That seems to work', 'Let me think', 'Actually',
            'Wait', 'Hmm', 'Perhaps', 'I think', 'So maybe', 'But how to',
            'Or maybe', 'Wait, maybe', 'But maybe better', 'Let\'s see',
            'etc.', '...', 'The story about', 'Need to', 'That makes sense'
        ]
        
        for pattern in thinking_patterns:
            if pattern.lower() in picture.lower():
                issues.append(f"CRITICAL: Picture contains thinking text: '{pattern}'")
                score -= 0.4
                break
        
        return max(score, 0.0), issues
    
    def _check_sentence_quality(self, entry: GeneratedVocabularyEntry, target_word: str) -> Tuple[float, List[str]]:
        """Check example sentence quality"""
        issues = []
        score = 1.0
        
        sentence = entry.example_sentence.strip()
        word_count = len(sentence.split())
        
        # Length check
        if word_count < 5:
            issues.append("Sentence too short (min 5 words)")
            score -= 0.3
        elif word_count > 20:
            issues.append("Sentence too long (max 20 words)")
            score -= 0.2
        
        # Quality check
        if not sentence:
            issues.append("CRITICAL: Empty example sentence")
            score = 0.0
        
        # Grammar check
        if sentence and not sentence[0].isupper():
            issues.append("Sentence should start with capital letter")
            score -= 0.1
        
        if sentence and not sentence.endswith('.'):
            issues.append("Sentence should end with period")
            score -= 0.1
        
        return max(score, 0.0), issues
    
    def _generate_suggestions(self, entry: GeneratedVocabularyEntry, target_word: str, issues: List[str]) -> List[str]:
        """Generate specific suggestions for improvement"""
        suggestions = []
        
        if any('circular' in issue.lower() or 'uses' in issue.lower() for issue in issues):
            suggestions.append(f"Avoid using '{target_word}' or similar words like '{target_word}furter', '{target_word}ed', etc.")
            suggestions.append("Use completely different words that sound similar or relate to the concept")
        
        if any('too long' in issue.lower() for issue in issues):
            suggestions.append("Keep responses concise and focused")
            suggestions.append("Remove thinking/planning text from the final output")
        
        if any('empty' in issue.lower() for issue in issues):
            suggestions.append("Ensure all components are properly filled")
        
        if any('mnemonic' in issue.lower() for issue in issues):
            suggestions.append("Create a clear sound-alike or visual connection")
            suggestions.append("Use simple, memorable phrases")
        
        return suggestions

class VocabularyOptimizer:
    """Optimizes vocabulary entries through iterative improvement"""
    
    def __init__(self, llm_service: HackClubLLMService, evaluator: VocabularyEvaluator):
        self.llm_service = llm_service
        self.evaluator = evaluator
        self.max_iterations = 3
    
    def _contains_word_or_similar(self, text: str, word: str, word_root: str) -> bool:
        """Check if text contains the word or very similar words - delegate to evaluator"""
        return self.evaluator._contains_word_or_similar(text, word, word_root)
    
    def optimize_entry(self, entry: GeneratedVocabularyEntry, target_word: str, part_of_speech: str) -> GeneratedVocabularyEntry:
        """Optimize entry through evaluation and regeneration"""
        
        current_entry = entry
        
        for iteration in range(self.max_iterations):
            # Evaluate current entry
            evaluation = self.evaluator.evaluate_entry(current_entry, target_word)
            
            logger.info(f"Optimization iteration {iteration + 1} for {target_word}: Score {evaluation.overall_score:.2f}")
            if evaluation.issues:
                logger.info(f"Issues found: {evaluation.issues}")
            
            if evaluation.is_valid:
                logger.info(f"Entry for {target_word} passed evaluation")
                current_entry.quality_score = evaluation.overall_score
                current_entry.validation_passed = True
                return current_entry
            
            # Generate improvement prompt
            improvement_prompt = self._create_improvement_prompt(
                current_entry, target_word, part_of_speech, evaluation
            )
            
            # Get improved entry
            try:
                response = self.llm_service.generate_completion(
                    prompt=improvement_prompt,
                    system_message="You are a vocabulary expert. Output ONLY the requested format with no explanations.",
                    max_tokens=250,
                    temperature=0.1
                )
                
                if response.success:
                    improved_entry = self._parse_improved_response(response.content, target_word, part_of_speech)
                    current_entry = improved_entry
                else:
                    logger.warning(f"LLM failed to improve entry for {target_word}: {response.error}")
                    break
                    
            except Exception as e:
                logger.error(f"Error during optimization iteration {iteration + 1}: {e}")
                break
        
        # If we reach here, optimization failed
        logger.warning(f"Failed to optimize entry for {target_word} after {self.max_iterations} iterations")
        current_entry.quality_score = evaluation.overall_score if 'evaluation' in locals() else 0.0
        current_entry.validation_passed = False
        current_entry.detailed_feedback = f"Failed optimization: {', '.join(evaluation.issues[:3])}"
        
        return current_entry
    
    def _create_improvement_prompt(self, entry: GeneratedVocabularyEntry, target_word: str, 
                                 part_of_speech: str, evaluation: EvaluationResult) -> str:
        """Create prompt for improving the entry based on evaluation"""
        
        issues_text = '\n'.join([f"- {issue}" for issue in evaluation.issues[:5]])
        suggestions_text = '\n'.join([f"- {suggestion}" for suggestion in evaluation.suggestions[:3]])
        
        # Create better mnemonic suggestions based on the word
        mnemonic_examples = {
            'supplant': 'Sounds like: "super plant" or "supply plant"',
            'tenuous': 'Sounds like: "ten new us" or "tennis"',
            'demagogue': 'Sounds like: "demo gag" or "demon vogue"',
            'candid': 'Sounds like: "can did" or "camera"',
            'frank': 'Sounds like: "frank-enstein" or "bank"'
        }
        
        suggested_mnemonic = mnemonic_examples.get(target_word.lower(), 'Sounds like: [find a completely different word that sounds similar]')
        
        return f"""Fix this vocabulary entry by addressing the issues below. BE CREATIVE with mnemonics!

CURRENT ENTRY HAS PROBLEMS:
{target_word.upper()} ({entry.pronunciation}) {entry.part_of_speech} — {entry.definition}
{entry.mnemonic_type}: {entry.mnemonic_phrase}
Picture: {entry.picture_story}
Other forms: {entry.other_forms}
Sentence: {entry.example_sentence}

PROBLEMS TO FIX:
{issues_text}

REQUIREMENTS:
{suggestions_text}

MNEMONIC GUIDANCE:
- {suggested_mnemonic}
- Make it memorable and fun!
- Link the sound to the meaning

STRICT RULES:
- NO thinking text (<think>, "Maybe", "For example", etc.)
- Definition must be 3+ words and start with capital
- Mnemonic cannot use {target_word.lower()} or similar words
- Be specific and engaging

OUTPUT FORMAT (NOTHING ELSE):
{target_word.upper()} (pronunciation) {part_of_speech} — Clear definition starting with capital
Sounds like: creative mnemonic phrase
Picture: vivid scene description
Other forms: proper word forms
Sentence: example using the word naturally"""
    
    def _parse_improved_response(self, content: str, word: str, part_of_speech: str) -> GeneratedVocabularyEntry:
        """Parse the improved response into a vocabulary entry"""
        
        # Aggressive cleaning of response
        import re
        
        # Remove thinking tags and content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<.*?>', '', content)
        
        # Remove thinking phrases
        thinking_phrases = [
            r'Maybe.*?\.',
            r'For example.*?\.',
            r'Yeah.*?\.',
            r'Need to.*?\.',
            r'That seems.*?\.',
            r'Let me.*?\.',
            r'Actually.*?\.',
            r'Wait.*?\.',
            r'But how.*?\.',
            r'The story about.*?\.',
            r'etc\.',
            r'\.\.\.+',
            r'That makes sense.*?\.'
        ]
        
        for pattern in thinking_phrases:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and newlines
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Initialize with safe defaults based on word
        word_definitions = {
            "FRANK": "honest and straightforward",
            "CANDID": "truthful and sincere", 
            "SERENE": "calm and peaceful",
            "ARDENT": "passionate and enthusiastic",
            "REVERE": "to respect deeply",
            "PLACID": "calm and peaceful",
            "SUPPLANT": "to replace or take the place of",
            "TENUOUS": "very weak or slight",
            "DEMAGOGUE": "a political leader who appeals to emotions"
        }
        
        # Safe mnemonics that avoid circular references
        safe_mnemonics = {
            "FRANK": "brick (solid and reliable)",
            "CANDID": "camera (captures truth)",
            "SERENE": "marine (peaceful waters)",
            "ARDENT": "garden (passionate growth)",
            "REVERE": "silver (precious and valued)",
            "PLACID": "classic (timelessly calm)",
            "SUPPLANT": "super plant (growing over something else)",
            "TENUOUS": "tennis (thin string about to break)",
            "DEMAGOGUE": "demo dog (barking loudly for attention)"
        }
        
        pronunciation = word.lower()
        definition = word_definitions.get(word.upper(), f"a {part_of_speech}")
        mnemonic_type = "Sounds like"
        mnemonic_phrase = safe_mnemonics.get(word.upper(), "safe alternative")
        
        # Better picture stories for fallbacks
        picture_stories = {
            "SUPPLANT": "A strong vine growing over and replacing an old fence",
            "TENUOUS": "A person walking carefully across a thin rope bridge",
            "DEMAGOGUE": "A speaker on a soapbox attracting a loud crowd",
            "FRANK": "Someone speaking honestly at a business meeting",
            "CANDID": "A photographer capturing genuine moments",
            "SERENE": "A peaceful lake reflecting calm mountains",
            "ARDENT": "A passionate artist painting with intense focus",
            "REVERE": "People respectfully bowing before an honored leader",
            "PLACID": "A sleeping cat on a quiet windowsill"
        }
        
        picture_story = picture_stories.get(word.upper(), f"A clear scene showing {definition}")
        other_forms = f"{word.lower()}ly, {word.lower()}ness"
        example_sentence = f"The person demonstrated this quality clearly."
        
        try:
            # Parse each line more carefully
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith(word.upper()) and '—' in line:
                    parts = line.split('—', 1)
                    if len(parts) == 2:
                        left_part = parts[0].strip()
                        new_definition = parts[1].strip()
                        
                        # Only use if it doesn't contain forbidden words
                        if not self._contains_word_or_similar(new_definition, word.lower(), word.lower()[:3]):
                            definition = new_definition
                        
                        if '(' in left_part and ')' in left_part:
                            start = left_part.find('(')
                            end = left_part.find(')')
                            pronunciation = left_part[start+1:end].strip()
                
                elif ':' in line and any(line.startswith(t) for t in ['Sounds like', 'Looks like', 'Think of', 'Connect with']):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        new_type = parts[0].strip()
                        new_phrase = parts[1].strip()
                        
                        # Only use if it doesn't contain forbidden words
                        if not self._contains_word_or_similar(new_phrase, word.lower(), word.lower()[:3]):
                            mnemonic_type = new_type
                            mnemonic_phrase = new_phrase
                        else:
                            # Keep the original if it's not terrible
                            if len(parts[1].strip()) > 5 and not word.lower() in parts[1].lower():
                                mnemonic_phrase = parts[1].strip()
                
                elif line.startswith('Picture:'):
                    new_picture = line.replace('Picture:', '').strip()
                    # Check for thinking text and forbidden words
                    has_thinking = any(pattern.lower() in new_picture.lower() for pattern in [
                        'maybe', 'for example', 'yeah', 'need to', 'that seems', 'etc', '...'
                    ])
                    has_forbidden = self._contains_word_or_similar(new_picture, word.lower(), word.lower()[:3])
                    
                    if not has_thinking and not has_forbidden and len(new_picture.split()) >= 5:
                        picture_story = new_picture
                
                elif line.startswith('Other forms:'):
                    other_forms = line.replace('Other forms:', '').strip()
                
                elif line.startswith('Sentence:'):
                    new_sentence = line.replace('Sentence:', '').strip()
                    # Only use if it doesn't contain forbidden words
                    if not self._contains_word_or_similar(new_sentence, word.lower(), word.lower()[:3]):
                        example_sentence = new_sentence
        
        except Exception as e:
            logger.warning(f"Error parsing improved response for {word}: {e}")
        
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
            quality_score=0.0,
            validation_passed=False
        )
