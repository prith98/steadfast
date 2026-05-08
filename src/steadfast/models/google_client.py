"""Google client wrapper — Gemini family via ``google-genai``.

Uses the unified Gen AI SDK (``google.genai``). Async calls go through
``client.aio.models.generate_content``. Roles map: Steadfast's ``user`` →
genai ``user``; Steadfast's ``assistant`` → genai ``model``; Steadfast's
``system`` → genai ``GenerateContentConfig.system_instruction``.
"""

from __future__ import annotations

from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from steadfast.models.base import (
    BaseModelClient,
    ChatMessage,
    ChatResponse,
    TokenUsage,
)
from steadfast.models.pricing import compute_cost


class GoogleClient(BaseModelClient):
    """:class:`BaseModelClient` implementation backed by ``google.genai.Client.aio``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_concurrent: int = 5,
        max_retries: int = 5,
    ) -> None:
        super().__init__(max_concurrent=max_concurrent, max_retries=max_retries)
        self._client = genai.Client(api_key=api_key)

    @classmethod
    def _is_retryable(cls, exc: BaseException) -> bool:
        if isinstance(exc, genai_errors.APIError):
            code = getattr(exc, "code", None)
            if code == 429:
                return True
            if isinstance(code, int) and code >= 500:
                return True
        return False

    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_output_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatResponse:
        system_msgs = [m.content for m in messages if m.role == "system"]
        contents: list[genai_types.Content] = []
        for m in messages:
            if m.role == "system":
                continue
            role = "user" if m.role == "user" else "model"
            contents.append(
                genai_types.Content(role=role, parts=[genai_types.Part(text=m.content)])
            )

        config_kwargs: dict[str, Any] = {"max_output_tokens": max_output_tokens}
        if system_msgs:
            config_kwargs["system_instruction"] = "\n\n".join(system_msgs)
        config_kwargs.update(kwargs)
        config = genai_types.GenerateContentConfig(**config_kwargs)

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        text = response.text or ""
        usage_meta = response.usage_metadata
        usage = TokenUsage(
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        )
        cost = compute_cost(model, usage)

        finish_reason: str | None = None
        if response.candidates:
            fr = response.candidates[0].finish_reason
            finish_reason = str(fr) if fr is not None else None

        raw_payload: dict[str, Any] = {}
        if hasattr(response, "model_dump"):
            raw_payload = response.model_dump()

        return ChatResponse(
            text=text,
            usage=usage,
            cost_usd=cost,
            model=model,
            finish_reason=finish_reason,
            raw=raw_payload,
        )
