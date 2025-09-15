"""
LLM Service for integrating with ai.hackclub.com API
Provides clean integration with proper error handling, retries, and response processing.
Optimized for serverless deployment with caching and connection pooling.
"""

import os
import time
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from LLM API"""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    success: bool = True
    error: Optional[str] = None


class HackClubLLMService:
    """
    AI Hackclub API client with robust error handling and retry logic
    Optimized for serverless environments with connection pooling
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, default_model: str = None):
        # Hack Club AI doesn't require an API key
        self.api_key = api_key  # Optional for Hack Club AI
        self.base_url = base_url or os.getenv('HACKCLUB_API_URL', 'https://ai.hackclub.com')
        self.default_model = default_model or os.getenv('HACKCLUB_MODEL', 'qwen/qwen3-32b')
        
        # Setup session with retry strategy and connection pooling
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,  # Reduced for faster failover
            backoff_factor=0.5,  # Faster backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Connection pooling
            pool_maxsize=20
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Default headers (no auth needed for Hack Club AI)
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'SAT-Vocab-RAG/1.0.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        }
        if self.api_key:  # Only add auth if API key is provided
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        self.session.headers.update(headers)
        
        # Simple in-memory cache for repeated requests
        self._cache = {}
        self._cache_size = 100
    
    def generate_completion(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_message: str = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion using the Hackclub API with caching
        
        Args:
            prompt: The user prompt
            model: Model to use (defaults to configured model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_message: Optional system message
            **kwargs: Additional parameters
        
        Returns:
            LLMResponse with the generated content
        """
        try:
            # Create cache key
            cache_key = f"{prompt[:100]}_{model or self.default_model}_{temperature}_{max_tokens}"
            
            # Check cache first
            if cache_key in self._cache:
                logger.debug("Cache hit for request")
                return self._cache[cache_key]
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request payload
            payload = {
                "model": model or self.default_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            logger.info(f"Making API request to {self.base_url}/chat/completions")
            logger.debug(f"Payload: {payload}")
            
            # Make API request with reduced timeout for serverless
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=25  # Reduced for serverless constraints
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Extract content from response
            if 'choices' not in data or not data['choices']:
                raise ValueError("No choices in API response")
            
            choice = data['choices'][0]
            content = choice['message']['content']
            
            # Create structured response
            result = LLMResponse(
                content=content,
                usage=data.get('usage', {}),
                model=data.get('model', model or self.default_model),
                finish_reason=choice.get('finish_reason', 'unknown'),
                success=True
            )
            
            # Cache the result (with size limit)
            if len(self._cache) < self._cache_size:
                self._cache[cache_key] = result
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return LLMResponse(
                content="",
                usage={},
                model=model or self.default_model,
                finish_reason="error",
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return LLMResponse(
                content="",
                usage={},
                model=model or self.default_model,
                finish_reason="error",
                success=False,
                error=str(e)
            )
    
    def generate_vocabulary_entry(
        self,
        word: str,
        context_examples: List[str],
        template: str,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a Gulotta-style vocabulary entry
        
        Args:
            word: The vocabulary word
            context_examples: List of example entries for context
            template: Prompt template to use
            **kwargs: Additional generation parameters
        
        Returns:
            LLMResponse with the generated vocabulary entry
        """
        # Format the template with context
        examples_text = "\n\n".join(context_examples)
        prompt = template.format(
            word=word.upper(),
            examples=examples_text
        )
        
        # Add system message for vocabulary generation
        system_message = """You are Charles Gulotta, the renowned SAT vocabulary instructor. You have a unique gift for creating unforgettable mnemonic devices that help students remember difficult words through vivid "sounds like" associations and detailed picture stories. 

Your style characteristics:
- Creative "Sounds like" mnemonics that actually sound like the target word
- Detailed, narrative picture stories (2-4 sentences) that connect the mnemonic to the word's meaning
- Specific characters, settings, and visual details that make the stories memorable
- Perfect adherence to the exact format: WORD (pronunciation) part — definition, Sounds like: phrase, Picture: story, Other forms: forms, Sentence: example

Follow the examples provided EXACTLY. Do not deviate from the format or style."""
        
        return self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            **kwargs
        )
    
    def generate_definition(
        self,
        word: str,
        context: str = "",
        **kwargs
    ) -> LLMResponse:
        """
        Generate a clear SAT-appropriate definition
        
        Args:
            word: The vocabulary word
            context: Additional context from similar words
            **kwargs: Additional generation parameters
        
        Returns:
            LLMResponse with the generated definition
        """
        prompt = f"""Create a clear, SAT-appropriate definition for the word "{word}".
        
        Context from similar words: {context}
        
        The definition should be:
        - Precise and accurate
        - Appropriate for SAT level
        - Clear and concise
        - Include part of speech
        
        Format: [part of speech] — [definition]"""
        
        system_message = "You are a vocabulary expert creating precise, SAT-level definitions."
        
        return self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            max_tokens=200,
            temperature=0.3,
            **kwargs
        )
    
    def generate_example_sentence(
        self,
        word: str,
        definition: str,
        context: str = "",
        **kwargs
    ) -> LLMResponse:
        """
        Generate a natural example sentence
        
        Args:
            word: The vocabulary word
            definition: The word's definition
            context: Additional context from similar usage
            **kwargs: Additional generation parameters
        
        Returns:
            LLMResponse with the generated example sentence
        """
        prompt = f"""Create a natural example sentence using the word "{word}" with definition: {definition}
        
        Context from similar usage: {context}
        
        The sentence should:
        - Demonstrate proper usage of the word
        - Be contextually appropriate
        - Sound natural and authentic
        - Be suitable for SAT preparation
        
        Provide only the example sentence, nothing else."""
        
        system_message = "You are creating natural, educational example sentences for vocabulary learning."
        
        return self.generate_completion(
            prompt=prompt,
            system_message=system_message,
            max_tokens=100,
            temperature=0.5,
            **kwargs
        )


# Global service instance
_llm_service_instance = None


def get_llm_service() -> HackClubLLMService:
    """Get or create the global LLM service instance"""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = HackClubLLMService()
    return _llm_service_instance


# Convenience functions
def generate_vocabulary_entry(word: str, context_examples: List[str], template: str, **kwargs) -> LLMResponse:
    """Convenience function for generating vocabulary entries"""
    service = get_llm_service()
    return service.generate_vocabulary_entry(word, context_examples, template, **kwargs)


def generate_definition(word: str, context: str = "", **kwargs) -> LLMResponse:
    """Convenience function for generating definitions"""
    service = get_llm_service()
    return service.generate_definition(word, context, **kwargs)


def generate_example_sentence(word: str, definition: str, context: str = "", **kwargs) -> LLMResponse:
    """Convenience function for generating example sentences"""
    service = get_llm_service()
    return service.generate_example_sentence(word, definition, context, **kwargs)
