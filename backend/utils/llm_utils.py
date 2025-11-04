"""
LLM utilities - Minimalist version
Supports: OpenAI and Groq only
"""

import os
import asyncio
from typing import Optional, List
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================
# === OpenAI Implementation ===
# ============================================

async def call_openai_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs
) -> str:
    """Async call to OpenAI API"""
    try:
        from openai import AsyncOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        client = AsyncOpenAI(api_key=api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        raise ImportError("openai package not installed. Install with: pip install openai")
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise


def call_openai_sync(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs
) -> str:
    """Sync wrapper for OpenAI API"""
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        client = OpenAI(api_key=api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        raise ImportError("openai package not installed. Install with: pip install openai")
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise


# ============================================
# === Groq Implementation ===
# ============================================

async def call_groq_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "llama-3.1-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs
) -> str:
    """Async call to Groq API"""
    try:
        from groq import AsyncGroq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        client = AsyncGroq(api_key=api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        raise ImportError("groq package not installed. Install with: pip install groq")
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise


def call_groq_sync(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "llama-3.1-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs
) -> str:
    """Sync wrapper for Groq API"""
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        client = Groq(api_key=api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        raise ImportError("groq package not installed. Install with: pip install groq")
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise


# ============================================
# === Universal LLM Caller ===
# ============================================

async def call_llm_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    provider: Optional[str] = None,
    **kwargs
) -> str:
    """
    Universal async LLM caller
    
    Args:
        prompt: User prompt
        system_prompt: System prompt (optional)
        model: Model name (optional, uses env default)
        temperature: Temperature
        max_tokens: Max tokens
        provider: 'openai' or 'groq'
        
    Returns:
        Generated text
    """
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if provider not in ["openai", "groq"]:
        raise ValueError(f"Provider must be 'openai' or 'groq', got: {provider}")
    
    if not model:
        model = os.getenv("LLM_MODEL")
        if not model:
            model = "gpt-4o-mini" if provider == "openai" else "llama-3.1-70b-versatile"
    
    if provider == "openai":
        return await call_openai_async(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    elif provider == "groq":
        return await call_groq_async(prompt, system_prompt, model, temperature, max_tokens, **kwargs)


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    provider: Optional[str] = None,
    **kwargs
) -> str:
    """
    Sync wrapper for LLM call
    """
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    if not model:
        model = os.getenv("LLM_MODEL")
        if not model:
            model = "gpt-4o-mini" if provider == "openai" else "llama-3.1-70b-versatile"
    
    if provider == "openai":
        return call_openai_sync(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    elif provider == "groq":
        return call_groq_sync(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# ============================================
# === Batch Processing ===
# ============================================

async def call_llm_batch(
    prompts: List[str],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_concurrent: int = 5,
    **kwargs
) -> List[str]:
    """
    Process multiple prompts concurrently
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(prompt):
        async with semaphore:
            try:
                return await call_llm_async(prompt, system_prompt, model, **kwargs)
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                return ""
    
    tasks = [process_with_semaphore(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    return results
async def call_llm_with_retry(
    prompt: str,
    max_retries: int = 3,
    **kwargs
) -> str:
    """
    Call LLM with automatic retry on failure
    """
    for attempt in range(max_retries):
        try:
            return await call_llm_async(prompt, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            await asyncio.sleep(2 ** attempt) 