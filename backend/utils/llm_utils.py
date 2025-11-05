"""
LLM utilities - Minimalist version (Strict .env-based configuration)
Supports: OpenAI and Groq only
"""

import os
import asyncio
from typing import Optional, List
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

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
    Universal async LLM caller (Strict .env-based version)
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()

    # Strict mode: bắt buộc .env phải có LLM_MODEL
    provider = provider or os.getenv("LLM_PROVIDER", "groq")
    model = model or os.getenv("LLM_MODEL")
    
    if not model:
        if provider == "groq":
            model = "llama-3.1-70b-versatile"
            logger.warning("LLM_MODEL not set, using default: llama-3.1-70b-versatile")
        elif provider == "openai":
            model = "gpt-4o-mini"
            logger.warning("LLM_MODEL not set, using default: gpt-4o-mini")
        else:
            raise ValueError("LLM_MODEL must be set in .env")


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
    Sync wrapper for LLM call (Strict .env-based version)
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()

    model = model or os.getenv("LLM_MODEL")
    if not model:
        raise ValueError(
            "❌ LLM_MODEL is not set in your .env file. "
            "Please define it, e.g. LLM_MODEL=gpt-4o-mini or LLM_MODEL=llama-3.1-70b-versatile"
        )

    if provider == "openai":
        return call_openai_sync(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    elif provider == "groq":
        return call_groq_sync(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


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

# Fallback nếu .env thiếu
DEFAULT_MODEL = os.getenv("LLM_MODEL")
if not DEFAULT_MODEL:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    DEFAULT_MODEL = "llama-3.1-70b-versatile" if provider == "groq" else "gpt-4o-mini"