"""
Hack Club AI Service Integration
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)


class HackClubAI:
    """Custom client for Hack Club AI service"""
    
    def __init__(self, api_url: str = "https://ai.hackclub.com", model: str = "qwen/qwen3-32b"):
        self.api_url = api_url.rstrip('/')
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)
        
        logger.info(f"Initialized Hack Club AI client with model: {model}")
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Send chat completion request to Hack Club AI"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add any additional parameters
            payload.update(kwargs)
            
            logger.debug(f"Sending request to Hack Club AI: {len(messages)} messages")
            
            response = await self.client.post(
                f"{self.api_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content from response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                logger.debug(f"Received response: {len(content)} characters")
                return content
            else:
                logger.error(f"Unexpected response format: {result}")
                raise ValueError("Invalid response format from Hack Club AI")
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Hack Club AI: {e}")
            raise
        except Exception as e:
            logger.error(f"Hack Club AI request failed: {e}")
            raise
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = await self.client.get(f"{self.api_url}/model")
            response.raise_for_status()
            
            # Response should be comma-separated model names
            models_text = response.text.strip()
            models = [model.strip() for model in models_text.split(',')]
            
            logger.info(f"Available models: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return [self.model]  # Return default model
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class HackClubChatModel:
    """LangChain-compatible wrapper for Hack Club AI"""
    
    def __init__(self, model: str = "qwen/qwen3-32b", temperature: float = 0.7, max_tokens: int = 2000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = HackClubAI(model=model)
        
    async def agenerate(self, messages_list: List[List], **kwargs):
        """Generate responses for multiple message lists (async)"""
        try:
            results = []
            
            for messages in messages_list:
                # Convert LangChain message format to Hack Club format
                formatted_messages = []
                for message in messages:
                    if hasattr(message, 'content'):
                        role = "system" if hasattr(message, '__class__') and "System" in message.__class__.__name__ else "user"
                        formatted_messages.append({
                            "role": role,
                            "content": message.content
                        })
                    else:
                        # Fallback for direct dict format
                        formatted_messages.append(message)
                
                # Get response from Hack Club AI
                response_text = await self.client.chat_completion(
                    messages=formatted_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **kwargs
                )
                
                # Create mock LangChain response structure
                generation = type('Generation', (), {
                    'text': response_text,
                    'generation_info': {'model': self.model}
                })()
                
                results.append([generation])
            
            # Create mock LLMResult structure
            result = type('LLMResult', (), {
                'generations': results,
                'llm_output': {'model': self.model}
            })()
            
            return result
            
        except Exception as e:
            logger.error(f"Hack Club AI generation failed: {e}")
            raise
    
    def generate(self, messages_list: List[List], **kwargs):
        """Synchronous generate (runs async internally)"""
        return asyncio.run(self.agenerate(messages_list, **kwargs))
    
    async def aclose(self):
        """Close the underlying client"""
        await self.client.close()


# Callback for tracking usage (mock since Hack Club AI is free)
class HackClubCallback:
    """Mock callback for usage tracking"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_hackclub_callback():
    """Get a mock callback for Hack Club AI (since it's free)"""
    return HackClubCallback()