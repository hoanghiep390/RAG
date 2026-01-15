# backend/utils/llm_utils.py
"""
Tiện ích LLM xử lý API calls
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
    """ Async OpenAI call - No file operations"""
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
        logger.error(f" Lỗi OpenAI API: {str(e)}")
        raise


async def call_openai_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs
):
    """Stream OpenAI response token by token"""
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

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except ImportError:
        raise ImportError("openai package not installed")
    except Exception as e:
        logger.error(f" Lỗi OpenAI streaming: {str(e)}")
        raise




async def call_groq_async(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs
) -> str:
    """Async Groq call - No file operations"""
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
        logger.error(f" Lỗi Groq API: {str(e)}")
        raise


async def call_groq_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    **kwargs
):
    """Stream Groq response token by token"""
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

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except ImportError:
        raise ImportError("groq package not installed")
    except Exception as e:
        logger.error(f" Lỗi Groq streaming: {str(e)}")
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
    """ Universal async LLM call - No file operations"""
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


async def call_llm_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    provider: Optional[str] = None,
    **kwargs
):
    """Universal streaming LLM call - yields tokens as they arrive"""
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
        async for chunk in call_openai_stream(prompt, system_prompt, model, temperature, max_tokens, **kwargs):
            yield chunk
    elif provider == "groq":
        async for chunk in call_groq_stream(prompt, system_prompt, model, temperature, max_tokens, **kwargs):
            yield chunk
    else:
        raise ValueError(f"Unsupported provider: {provider}")




async def call_llm_batch(
    prompts: List[str],
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_concurrent: int = 5,
    **kwargs
) -> List[str]:
    """ Batch processing - No file operations"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(prompt):
        async with semaphore:
            try:
                return await call_llm_async(prompt, system_prompt, model, **kwargs)
            except Exception as e:
                logger.error(f" Lỗi xử lý batch: {str(e)}")
                return ""

    tasks = [process_with_semaphore(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    return results


async def call_llm_with_retry(
    prompt: str,
    max_retries: int = 3,
    **kwargs
) -> str:
    """ LLM call with retry - No file operations"""
    for attempt in range(max_retries):
        try:
            return await call_llm_async(prompt, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f" LLM call thất bại (lần thử {attempt + 1}/{max_retries}): {str(e)}")
            await asyncio.sleep(2 ** attempt)


# ============================================
# MÔ HÌNH MẶC ĐỊNH DỰ PHÒNG
# ============================================

DEFAULT_MODEL = os.getenv("LLM_MODEL")
if not DEFAULT_MODEL:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    DEFAULT_MODEL = "llama-3.1-70b-versatile" if provider == "groq" else "gpt-4o-mini"
