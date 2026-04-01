"""
LLM Service for integrating with either Hack Club AI or OpenRouter-style APIs.
Uses OpenAI-compatible chat completions and auto-detects the provider from config/key shape.
"""

import os
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load local .env when available (safe for production; platform env vars still win).
load_dotenv()


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
        raw_api_key = (
            api_key
            or os.getenv('LLM_API_KEY', '')
            or os.getenv('HACKCLUB_API_KEY', '')
            or os.getenv('OPENROUTER_API_KEY', '')
        ).strip()

        configured_provider = (
            os.getenv('LLM_PROVIDER')
            or os.getenv('HACKCLUB_PROVIDER')
            or os.getenv('OPENROUTER_PROVIDER')
            or ''
        ).strip().lower()

        # Auto-detect provider from the key shape when possible.
        if configured_provider:
            self.provider = configured_provider
        elif raw_api_key.startswith('sk-or-v1-'):
            self.provider = 'openrouter'
        else:
            self.provider = 'hackclub'

        self.api_key = raw_api_key

        if self.provider == 'openrouter':
            self.base_url = base_url or os.getenv('LLM_API_URL', 'https://openrouter.ai/api/v1')
            self.default_model = default_model or os.getenv('LLM_MODEL', 'anthropic/claude-3.7-sonnet')
        else:
            self.base_url = base_url or os.getenv('LLM_API_URL', os.getenv('HACKCLUB_API_URL', 'https://ai.hackclub.com/proxy/v1'))
            self.default_model = default_model or os.getenv('LLM_MODEL', os.getenv('HACKCLUB_MODEL', 'google/gemini-2.5-flash'))

        # Hack Club's current docs promote the Responses API; OpenRouter keeps chat completions.
        self.api_style = (
            os.getenv('LLM_API_STYLE', '').strip().lower()
            or ('responses' if self.provider == 'hackclub' else 'chat')
        )
        
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
        
        # Default headers for OpenAI-compatible providers.
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'SAT-Vocab-RAG/1.0.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        if self.provider == 'openrouter':
            # Recommended by OpenRouter to improve request handling and attribution.
            headers.setdefault('HTTP-Referer', 'http://localhost')
            headers.setdefault('X-Title', 'SAT Vocabulary AI System')
        
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
            if not self.api_key:
                return LLMResponse(
                    content="",
                    usage={},
                    model=model or self.default_model,
                    finish_reason="error",
                    success=False,
                    error=f"Missing API key for provider '{self.provider}'. Set LLM_API_KEY (or HACKCLUB_API_KEY / OPENROUTER_API_KEY) in .env or deployment env vars."
                )

            # Create cache key
            cache_key = f"{self.api_style}_{prompt[:100]}_{model or self.default_model}_{temperature}_{max_tokens}"
            
            # Check cache first
            if cache_key in self._cache:
                logger.debug("Cache hit for request")
                return self._cache[cache_key]
            
            endpoint, payload = self._build_payload(
                prompt=prompt,
                model=model or self.default_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system_message=system_message,
                **kwargs
            )
            logger.info(f"Making API request to {endpoint}")
            logger.debug(f"Payload: {payload}")
            
            # Make API request with reduced timeout for serverless
            response = self.session.post(
                endpoint,
                json=payload,
                timeout=25  # Reduced for serverless constraints
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()

            content, response_model, finish_reason = self._extract_content(data)
            if not content or not content.strip():
                raise ValueError("API returned empty content")
            
            # Create structured response
            result = LLMResponse(
                content=content,
                usage=data.get('usage', {}),
                model=data.get('model', response_model or model or self.default_model),
                finish_reason=finish_reason,
                success=True
            )
            
            # Cache the result (with size limit)
            if len(self._cache) < self._cache_size:
                self._cache[cache_key] = result
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed via {self.provider}: {e}")
            return LLMResponse(
                content="",
                usage={},
                model=model or self.default_model,
                finish_reason="error",
                success=False,
                error=str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error via {self.provider}: {e}")
            return LLMResponse(
                content="",
                usage={},
                model=model or self.default_model,
                finish_reason="error",
                success=False,
                error=str(e)
            )

    def _build_payload(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_message: Optional[str] = None,
        **kwargs
    ) -> tuple[str, Dict[str, Any]]:
        """Build the request endpoint and payload for the configured API style."""
        if self.api_style == 'responses':
            input_messages = []
            if system_message:
                input_messages.append({
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_message}]
                })
            input_messages.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            })

            payload = {
                "model": model,
                "input": input_messages,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            return f"{self.base_url.rstrip('/')}/responses", payload

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        return f"{self.base_url.rstrip('/')}/chat/completions", payload

    def _extract_content(self, data: Dict[str, Any]) -> tuple[str, Optional[str], str]:
        """Extract generated text from either chat completions or responses payloads."""
        # Chat completions format
        if 'choices' in data and data['choices']:
            choice = data['choices'][0]
            message = choice.get('message', {})
            content = message.get('content', '')
            if isinstance(content, list):
                content = ''.join(piece.get('text', '') for piece in content if isinstance(piece, dict))
            return str(content or '').strip(), data.get('model'), choice.get('finish_reason', 'unknown')

        # Responses API format
        outputs = data.get('output', [])
        for item in outputs:
            if item.get('type') == 'message':
                content_items = item.get('content', [])
                text_parts = []
                for piece in content_items:
                    if piece.get('type') in ('output_text', 'text'):
                        text_parts.append(piece.get('text', ''))
                if text_parts:
                    return ''.join(text_parts).strip(), data.get('model'), item.get('status', data.get('status', 'unknown'))

        return '', data.get('model'), data.get('status', 'unknown')
    
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
        system_message = """You create concise SAT vocabulary entries with vivid mnemonic clues, picture stories, related forms, and example sentences.

    Follow the requested five-line format exactly and do not add commentary."""
        
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
