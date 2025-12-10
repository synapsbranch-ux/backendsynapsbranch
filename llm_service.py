"""
LLM Service - AWS Bedrock Integration for SynapsBranch
Supports: Amazon Nova, Meta Llama, Mistral via AWS Bedrock
Fallback: OpenAI GPT models
"""

import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from abc import ABC, abstractmethod

import boto3
from botocore.config import Config

# Import settings
from settings import settings

logger = logging.getLogger(__name__)

# Model mapping for user-friendly names
MODEL_MAPPING = {
    # Bedrock models
    "nova-pro": "amazon.nova-pro-v1:0",
    "nova": "amazon.nova-pro-v1:0",
    "llama3-70b": "meta.llama3-70b-instruct-v1:0",
    "llama": "meta.llama3-70b-instruct-v1:0",
    "mistral-large": "mistral.mistral-large-2402-v1:0",
    "mistral": "mistral.mistral-large-2402-v1:0",
    # Direct model IDs
    "amazon.nova-pro-v1:0": "amazon.nova-pro-v1:0",
    "meta.llama3-70b-instruct-v1:0": "meta.llama3-70b-instruct-v1:0",
    "mistral.mistral-large-2402-v1:0": "mistral.mistral-large-2402-v1:0",
    # OpenAI fallback
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str,
        model: str
    ) -> str:
        """Send messages and get a response."""
        pass
    
    @abstractmethod
    async def stream_chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Stream response chunks."""
        pass


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider supporting Nova, Llama, and Mistral."""
    
    def __init__(self):
        self.region = settings.AWS_BEDROCK_REGION
        self.config = Config(
            region_name=self.region,
            retries={"max_attempts": 3, "mode": "adaptive"}
        )
        self.client = boto3.client(
            "bedrock-runtime",
            config=self.config,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        
    def _format_messages_for_model(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str,
        model_id: str
    ) -> Dict[str, Any]:
        """Format messages based on model type."""
        
        # Convert messages to Bedrock Converse API format
        formatted_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            formatted_messages.append({
                "role": role,
                "content": [{"text": msg["content"]}]
            })
        
        return {
            "modelId": model_id,
            "messages": formatted_messages,
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "maxTokens": 4096,
                "temperature": 0.7,
                "topP": 0.9,
            }
        }
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str,
        model: str
    ) -> str:
        """Send messages to Bedrock and get response."""
        try:
            model_id = MODEL_MAPPING.get(model, model)
            request_body = self._format_messages_for_model(messages, system_prompt, model_id)
            
            # Use Converse API (unified interface for all models)
            response = self.client.converse(**request_body)
            
            # Extract response text
            output_message = response.get("output", {}).get("message", {})
            content = output_message.get("content", [])
            
            if content and len(content) > 0:
                return content[0].get("text", "")
            
            return ""
            
        except Exception as e:
            logger.error(f"Bedrock chat error: {str(e)}")
            raise
    
    async def stream_chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Stream response from Bedrock."""
        try:
            model_id = MODEL_MAPPING.get(model, model)
            request_body = self._format_messages_for_model(messages, system_prompt, model_id)
            
            # Use Converse Stream API
            response = self.client.converse_stream(**request_body)
            
            stream = response.get("stream")
            if stream:
                for event in stream:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text
                            
        except Exception as e:
            logger.error(f"Bedrock stream error: {str(e)}")
            raise


class OpenAIProvider(LLMProvider):
    """OpenAI fallback provider."""
    
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                organization=settings.OPENAI_ORG_ID,
            )
            self.available = settings.openai_configured
        except ImportError:
            logger.warning("OpenAI package not installed, fallback unavailable")
            self.available = False
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str,
        model: str
    ) -> str:
        """Send messages to OpenAI and get response."""
        if not self.available:
            raise RuntimeError("OpenAI provider not available")
            
        try:
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=4096,
                temperature=0.7,
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.error(f"OpenAI chat error: {str(e)}")
            raise
    
    async def stream_chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI."""
        if not self.available:
            raise RuntimeError("OpenAI provider not available")
            
        try:
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=4096,
                temperature=0.7,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI stream error: {str(e)}")
            raise


class LLMService:
    """
    Unified LLM service with provider fallback.
    Primary: AWS Bedrock (Nova, Llama, Mistral)
    Fallback: OpenAI (GPT-4o)
    """
    
    def __init__(self):
        self.bedrock = BedrockProvider()
        self.openai = OpenAIProvider()
        self.default_model = settings.DEFAULT_LLM_MODEL
    
    def _is_openai_model(self, model: str) -> bool:
        """Check if model is an OpenAI model."""
        return model.startswith("gpt-") or model in ["gpt-4o", "gpt-4o-mini"]
    
    def _get_provider(self, model: str) -> LLMProvider:
        """Get the appropriate provider for the model."""
        if self._is_openai_model(model):
            return self.openai
        return self.bedrock
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str = "You are a helpful AI assistant.",
        model: Optional[str] = None
    ) -> str:
        """
        Send messages and get a response.
        Falls back to OpenAI if Bedrock fails.
        """
        model = model or self.default_model
        provider = self._get_provider(model)
        
        try:
            return await provider.chat(messages, system_prompt, model)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}, trying fallback...")
            
            # Try fallback if not already using OpenAI
            if not self._is_openai_model(model) and self.openai.available:
                try:
                    return await self.openai.chat(messages, system_prompt, "gpt-4o")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise fallback_error
            raise
    
    async def stream_chat(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: str = "You are a helpful AI assistant.",
        model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream response chunks.
        Falls back to OpenAI if Bedrock fails.
        """
        model = model or self.default_model
        provider = self._get_provider(model)
        
        try:
            async for chunk in provider.stream_chat(messages, system_prompt, model):
                yield chunk
        except Exception as e:
            logger.warning(f"Primary provider streaming failed: {e}, trying fallback...")
            
            # Try fallback if not already using OpenAI
            if not self._is_openai_model(model) and self.openai.available:
                try:
                    async for chunk in self.openai.stream_chat(messages, system_prompt, "gpt-4o"):
                        yield chunk
                except Exception as fallback_error:
                    logger.error(f"Fallback streaming also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise


# Available models for the frontend
AVAILABLE_MODELS = [
    {"id": "amazon.nova-pro-v1:0", "name": "Amazon Nova Pro", "provider": "AWS Bedrock"},
    {"id": "meta.llama3-70b-instruct-v1:0", "name": "Llama 3 70B", "provider": "AWS Bedrock"},
    {"id": "mistral.mistral-large-2402-v1:0", "name": "Mistral Large", "provider": "AWS Bedrock"},
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI"},
]


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
