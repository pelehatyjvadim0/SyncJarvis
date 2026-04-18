from __future__ import annotations

import asyncio
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError


def is_retryable_transport_error(exc: BaseException) -> bool:
    # Решает, стоит ли повторить запрос к API (сеть, таймаут, 5xx, rate limit).
    return isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)) or (
        isinstance(exc, APIStatusError) and getattr(exc, "status_code", 0) >= 500
    )


async def chat_with_retry(
    client: AsyncOpenAI,
    *,
    default_model: str,
    model_override: str | None,
    max_transport_retries: int,
    max_tokens: int,
    temperature: float,
    messages: Any,
    retry_backoff_base: float,
    failure_message: str,
):
    last_exc: BaseException | None = None
    for attempt in range(max(1, max_transport_retries)):
        try:
            return await client.chat.completions.create(
                model=model_override or default_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
        except Exception as exc:
            last_exc = exc
            if not is_retryable_transport_error(exc) or attempt >= max_transport_retries - 1:
                raise
            await asyncio.sleep(retry_backoff_base * (attempt + 1))
    raise last_exc or RuntimeError(failure_message)
