# backend/utils/llm_utils.py
"""
✅ CLEAN: LLM utilities - No file I/O operations
Chỉ xử lý API calls, không lưu file

STATUS: ✅ ALREADY CLEAN - No changes needed
"""

import os
import asyncio
from typing import Optional, List
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================
# ✅ OPENAI FUNCTIONS - CLEAN
# ============================================

async def call_openai_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs
) -> str:
    """✅ CLEAN: Async OpenAI call - No file operations"""
    try:
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

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
        raise ImportError("openai package not installed")
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
    """✅ CLEAN: Sync OpenAI call - No file operations"""
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

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
        raise ImportError("openai package not installed")
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise


# ============================================
# ✅ GROQ FUNCTIONS - CLEAN
# ============================================

async def call_groq_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "llama-3.1-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs
) -> str:
    """✅ CLEAN: Async Groq call - No file operations"""
    try:
        from groq import AsyncGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")

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
        raise ImportError("groq package not installed")
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
    """✅ CLEAN: Sync Groq call - No file operations"""
    try:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")

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
        raise ImportError("groq package not installed")
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        raise


# ============================================
# ✅ UNIVERSAL LLM FUNCTIONS - CLEAN
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
    """✅ CLEAN: Universal async LLM call - No file operations"""
    provider = provider or os.getenv("LLM_PROVIDER", "groq")
    model = model or os.getenv("LLM_MODEL")
    
    if not model:
        if provider == "groq":
            model = "llama-3.1-70b-versatile"
        elif provider == "openai":
            model = "gpt-4o-mini"
        else:
            raise ValueError("LLM_MODEL not set in .env")
    
    if provider == "openai":
        return await call_openai_async(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    elif provider == "groq":
        return await call_groq_async(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    provider: Optional[str] = None,
    **kwargs
) -> str:
    """✅ CLEAN: Universal sync LLM call - No file operations"""
    provider = provider or os.getenv("LLM_PROVIDER", "groq")
    model = model or os.getenv("LLM_MODEL")
    
    if not model:
        if provider == "groq":
            model = "llama-3.1-70b-versatile"
        elif provider == "openai":
            model = "gpt-4o-mini"
        else:
            raise ValueError("LLM_MODEL not set in .env")
    
    if provider == "openai":
        return call_openai_sync(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    elif provider == "groq":
        return call_groq_sync(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


# ============================================
# ✅ BATCH & RETRY FUNCTIONS - CLEAN
# ============================================

async def call_llm_batch(
    prompts: List[str],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_concurrent: int = 5,
    **kwargs
) -> List[str]:
    """✅ CLEAN: Batch processing - No file operations"""
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
    """✅ CLEAN: LLM call with retry - No file operations"""
    for attempt in range(max_retries):
        try:
            return await call_llm_async(prompt, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            await asyncio.sleep(2 ** attempt)


# ============================================
# ✅ DEFAULT MODEL FALLBACK - CLEAN
# ============================================

DEFAULT_MODEL = os.getenv("LLM_MODEL")
if not DEFAULT_MODEL:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    DEFAULT_MODEL = "llama-3.1-70b-versatile" if provider == "groq" else "gpt-4o-mini"


# ============================================
# SUMMARY
# ============================================

"""
✅ STATUS: CLEAN - No file I/O operations detected

This module only handles:
- LLM API calls (OpenAI, Groq)
- Async/sync wrappers
- Batch processing
- Retry logic
- Configuration from environment variables

No file operations:
- ✅ No file saving
- ✅ No file loading
- ✅ No caching to disk
- ✅ Pure API communication

All functions are clean and can be used as-is.
"""