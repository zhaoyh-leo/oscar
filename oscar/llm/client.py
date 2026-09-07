"""LLM client built on LangChain chat models.

Supports DeepSeek, OpenAI, Anthropic, and arbitrary OpenAI-compatible endpoints
via API key.  Structured outputs use LangChain's ``with_structured_output``
(default: ``function_calling``), so responses are returned as typed Pydantic
models.  No local model support.
"""

import json
import os
import re
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from oscar.config import config
from oscar.utils.cache import llm_cache_get, llm_cache_set

# Environment variable used to source the API key for each provider.
# 密钥只经环境变量/根 .env 提供(oscar.config import 时自动加载)——绝不
# 落到代码或已提交配置。
_ENV_KEY_MAP = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class LLMClient:
    """Thin wrapper around a LangChain chat model with structured output support."""

    def __init__(self):
        self._model: Optional[BaseChatModel] = None
        self._structured: dict[tuple[type[BaseModel], str], Any] = {}

    @property
    def model(self) -> BaseChatModel:
        """Lazily build and cache the configured chat model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    # -- Provider factory -------------------------------------------------

    def _build_model(self) -> BaseChatModel:
        provider = (config.llm.provider or "deepseek").strip().lower()
        api_key = self._resolve_api_key(provider)

        if provider in ("deepseek", "openai", "openai_compatible"):
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, Any] = {
                "model": config.llm.model,
                "api_key": api_key,
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
            }
            # deepseek and openai_compatible need an explicit base_url
            if provider in ("deepseek", "openai_compatible"):
                kwargs["base_url"] = config.llm.base_url
            return ChatOpenAI(**kwargs)

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=config.llm.model,
                api_key=api_key,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                timeout=config.llm.timeout,
            )

        raise ValueError(
            f"Unsupported LLM provider: {provider!r}. "
            "Supported providers: 'deepseek', 'openai', 'openai_compatible', 'anthropic'."
        )

    def _resolve_api_key(self, provider: str) -> str:
        """API key 只从环境变量读取(含根 .env 加载的),缺失即抛错。

        旧实现先取 config.llm.api_key(硬编码默认)再回落 env —— 提交的
        配置里带明文 key,且 env 永远不生效。现在 env 是唯一来源:缺 key
        宁可报错指明设置方式,也不静默带病运行。
        """
        env_name = _ENV_KEY_MAP.get(provider, "")
        if not env_name:
            raise ValueError(
                f"Provider {provider!r} has no API-key env var mapped in "
                "_ENV_KEY_MAP."
            )
        api_key = os.getenv(env_name, "").strip()
        if not api_key:
            raise ValueError(
                f"{env_name} is not set. Put your key in the git-ignored "
                "root .env (see .env.example) or export it in your shell."
            )
        return api_key

    # -- Public API -------------------------------------------------------

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat request and return the response text.

        Results are cached by (message_hash, model, temperature) to avoid
        redundant LLM calls on repeated runs.
        """
        model_name = config.llm.model
        temp = kwargs.get("temperature", 0.0)

        # Check cache
        cached = llm_cache_get(messages, model_name, temp)
        if cached and "content" in cached:
            return cached["content"]

        response = self.model.invoke(messages, **kwargs)
        text = response.content

        # Cache the response
        llm_cache_set(messages, model_name, temp, {"content": text})
        return text

    def structured_output(
        self,
        schema: type[BaseModel],
        messages: list[dict],
        *,
        method: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> BaseModel:
        """Invoke the model and return a typed ``schema`` instance.

        Uses ``with_structured_output`` with ``method`` defaulting to
        ``config.llm.structured_output_method`` (``function_calling`` by
        default). The bound structured model is cached per (schema, method).

        Responses are cached like ``chat`` (key salted with the schema name so
        different schemas over identical prompts never collide). A cached hit
        is re-validated against the schema; parse failures fall through to a
        fresh call.  This is what makes evidence grounding cheap on re-runs.
        """
        method = method or config.llm.structured_output_method
        key = (schema, method)
        bound = self._structured.get(key)
        if bound is None:
            bound = self.model.with_structured_output(schema, method=method)
            self._structured[key] = bound

        model_name = config.llm.model
        temp = config.llm.temperature if temperature is None else temperature
        salt = f"structured:{schema.__name__}"

        cached = llm_cache_get(messages, model_name, temp, salt)
        if cached and "content" in cached:
            try:
                return schema.model_validate_json(cached["content"])
            except Exception:
                pass  # 缓存内容与 schema 不符 → 旁路重调

        response = bound.invoke(messages)
        if isinstance(response, BaseModel):
            llm_cache_set(messages, model_name, temp, {"content": response.model_dump_json()}, salt)
        return response

    def chat_json(self, messages: list[dict], **kwargs):
        """Send a chat request and parse the response as JSON.

        Returns a ``dict``, or a ``list`` when the model answers with a
        top-level JSON array (claim extraction prompts ask for arrays).  Old
        implementation regex-extracted the first ``{...}`` object, which
        silently mangled array answers into ``{}`` — parse the whole text
        first, keep the object extraction as a fallback for prose-wrapped
        responses.  Callers expecting an object guard with ``isinstance`` or
        ``.get`` inside try/except.
        """
        text = self.chat(messages, **kwargs)

        # 模型偶用 ```json 围栏包裹 → 先剥掉再整体解析
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

        # 兜底:文本夹杂叙述时取第一个 JSON 对象
        stripped = re.sub(r"^.*?\{", "{", stripped, count=1, flags=re.DOTALL)
        stripped = re.sub(r"\}[^}]*$", "}", stripped)
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {}

    def close(self):
        """Release the cached model (allows re-creation after config changes)."""
        self._model = None
        self._structured.clear()


# Global singleton
llm_client = LLMClient()
