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
    """
    Gọi OpenAI API bất đồng bộ để tạo văn bản
    
    Args:
        prompt: Câu hỏi/yêu cầu từ người dùng
        system_prompt: Hướng dẫn hệ thống cho AI (tùy chọn)
        model: Tên model OpenAI (mặc định: gpt-4o-mini)
        temperature: Độ sáng tạo (0-1, thấp = ổn định hơn)
        max_tokens: Số token tối đa trong câu trả lời
        **kwargs: Các tham số bổ sung cho API
        
    Returns:
        str: Văn bản phản hồi từ OpenAI
        
    Raises:
        ImportError: Nếu chưa cài đặt package openai
        ValueError: Nếu thiếu OPENAI_API_KEY
    """
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
    """
    Stream phản hồi từ OpenAI theo từng token (real-time)
    
    Args:
        prompt: Câu hỏi/yêu cầu từ người dùng
        system_prompt: Hướng dẫn hệ thống cho AI (tùy chọn)
        model: Tên model OpenAI (mặc định: gpt-4o-mini)
        temperature: Độ sáng tạo (0-1)
        max_tokens: Số token tối đa
        **kwargs: Các tham số bổ sung
        
    Yields:
        str: Từng phần văn bản được tạo ra theo thời gian thực
    """
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
    """
    Gọi Groq API bất đồng bộ để tạo văn bản (nhanh hơn OpenAI)
    
    Args:
        prompt: Câu hỏi/yêu cầu từ người dùng
        system_prompt: Hướng dẫn hệ thống cho AI (tùy chọn)
        model: Tên model Groq (mặc định: llama-3.3-70b-versatile)
        temperature: Độ sáng tạo (0-1)
        max_tokens: Số token tối đa trong câu trả lời
        **kwargs: Các tham số bổ sung cho API
        
    Returns:
        str: Văn bản phản hồi từ Groq
        
    Raises:
        ImportError: Nếu chưa cài đặt package groq
        ValueError: Nếu thiếu GROQ_API_KEY
    """
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
    """
    Stream phản hồi từ Groq theo từng token (real-time)
    
    Args:
        prompt: Câu hỏi/yêu cầu từ người dùng
        system_prompt: Hướng dẫn hệ thống cho AI (tùy chọn)
        model: Tên model Groq (mặc định: llama-3.3-70b-versatile)
        temperature: Độ sáng tạo (0-1)
        max_tokens: Số token tối đa
        **kwargs: Các tham số bổ sung
        
    Yields:
        str: Từng phần văn bản được tạo ra theo thời gian thực
    """
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

def _should_enable_fallback() -> bool:
    """Kiểm tra xem có nên bật fallback hay không"""
    return os.getenv("ENABLE_LLM_FALLBACK", "true").lower() in ("true", "1", "yes")



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
    Gọi LLM API tổng quát - tự động chọn provider (OpenAI hoặc Groq)
    Có cơ chế fallback tự động từ OpenAI sang Groq nếu OpenAI lỗi
    
    Args:
        prompt: Câu hỏi/yêu cầu từ người dùng
        system_prompt: Hướng dẫn hệ thống cho AI (tùy chọn)
        model: Tên model (tùy chọn, lấy từ .env nếu không có)
        temperature: Độ sáng tạo (0-1)
        max_tokens: Số token tối đa
        provider: Provider LLM (openai/groq, lấy từ .env nếu không có)
        **kwargs: Các tham số bổ sung
        
    Returns:
        str: Văn bản phản hồi từ LLM
        
    Raises:
        ValueError: Nếu provider không được hỗ trợ hoặc thiếu cấu hình
        Exception: Nếu cả hai provider đều thất bại
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("LLM_MODEL")
    
    if not model:
        if provider == "openai":
            model = "gpt-4o-mini"
        elif provider == "groq":
            model = "llama-3.1-70b-versatile"
        else:
            raise ValueError("LLM_MODEL not set in .env")
    
    # Thử provider chính
    if provider == "groq":
        return await call_groq_async(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
    elif provider == "openai":
        try:
            return await call_openai_async(prompt, system_prompt, model, temperature, max_tokens, **kwargs)
        except Exception as e:
            # Fallback sang Groq nếu OpenAI lỗi và fallback được bật
            if _should_enable_fallback():
                logger.warning(f"⚠️ OpenAI API failed: {str(e)}. Falling back to Groq...")
                try:
                    fallback_model = "llama-3.3-70b-versatile"
                    result = await call_groq_async(prompt, system_prompt, fallback_model, temperature, max_tokens, **kwargs)
                    logger.info("✅ Fallback to Groq successful")
                    return result
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback to Groq also failed: {str(fallback_error)}")
                    raise Exception(f"Both OpenAI and Groq failed. OpenAI: {str(e)}, Groq: {str(fallback_error)}")
            else:
                logger.error(f"❌ OpenAI API failed and fallback is disabled: {str(e)}")
                raise
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
    """
    Stream phản hồi LLM tổng quát - tự động chọn provider
    Có cơ chế fallback tự động từ OpenAI sang Groq nếu OpenAI lỗi
    
    Args:
        prompt: Câu hỏi/yêu cầu từ người dùng
        system_prompt: Hướng dẫn hệ thống cho AI (tùy chọn)
        model: Tên model (tùy chọn, lấy từ .env nếu không có)
        temperature: Độ sáng tạo (0-1)
        max_tokens: Số token tối đa
        provider: Provider LLM (openai/groq, lấy từ .env nếu không có)
        **kwargs: Các tham số bổ sung
        
    Yields:
        str: Từng phần văn bản được tạo ra theo thời gian thực
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("LLM_MODEL")
    
    if not model:
        if provider == "openai":
            model = "gpt-4o-mini"
        elif provider == "groq":
            model = "llama-3.1-70b-versatile"
        else:
            raise ValueError("LLM_MODEL not set in .env")
    
    # Thử provider chính
    if provider == "groq":
        async for chunk in call_groq_stream(prompt, system_prompt, model, temperature, max_tokens, **kwargs):
            yield chunk
    elif provider == "openai":
        try:
            async for chunk in call_openai_stream(prompt, system_prompt, model, temperature, max_tokens, **kwargs):
                yield chunk
        except Exception as e:
            # Fallback sang Groq nếu OpenAI lỗi và fallback được bật
            if _should_enable_fallback():
                logger.warning(f"⚠️ OpenAI streaming failed: {str(e)}. Falling back to Groq...")
                try:
                    fallback_model = "llama-3.3-70b-versatile"
                    async for chunk in call_groq_stream(prompt, system_prompt, fallback_model, temperature, max_tokens, **kwargs):
                        yield chunk
                    logger.info("✅ Fallback to Groq streaming successful")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback to Groq streaming also failed: {str(fallback_error)}")
                    raise Exception(f"Both OpenAI and Groq streaming failed. OpenAI: {str(e)}, Groq: {str(fallback_error)}")
            else:
                logger.error(f"❌ OpenAI streaming failed and fallback is disabled: {str(e)}")
                raise
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
    Xử lý nhiều prompt cùng lúc (batch processing) với giới hạn đồng thời
    
    Args:
        prompts: Danh sách các câu hỏi/yêu cầu cần xử lý
        system_prompt: Hướng dẫn hệ thống cho AI (tùy chọn)
        model: Tên model (tùy chọn)
        max_concurrent: Số lượng request đồng thời tối đa (mặc định: 5)
        **kwargs: Các tham số bổ sung
        
    Returns:
        List[str]: Danh sách các phản hồi tương ứng với từng prompt
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(prompt):
        """Xử lý từng prompt với semaphore để giới hạn số request đồng thời"""
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
    """
    Gọi LLM với cơ chế retry tự động khi gặp lỗi
    
    Args:
        prompt: Câu hỏi/yêu cầu từ người dùng
        max_retries: Số lần thử lại tối đa (mặc định: 3)
        **kwargs: Các tham số bổ sung cho call_llm_async
        
    Returns:
        str: Văn bản phản hồi từ LLM
        
    Raises:
        Exception: Nếu vẫn lỗi sau khi thử lại max_retries lần
    """
    for attempt in range(max_retries):
        try:
            return await call_llm_async(prompt, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f" LLM call thất bại (lần thử {attempt + 1}/{max_retries}): {str(e)}")
            await asyncio.sleep(2 ** attempt)


# ============================================
# MÔ HÌNH DỰ PHÒNG
# ============================================

# Lấy model mặc định từ biến môi trường, nếu không có thì dùng model dự phòng
DEFAULT_MODEL = os.getenv("LLM_MODEL")
if not DEFAULT_MODEL:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    # Chọn model dự phòng dựa trên provider
    DEFAULT_MODEL = "gpt-4o-mini" if provider == "openai" else "llama-3.1-70b-versatile"
