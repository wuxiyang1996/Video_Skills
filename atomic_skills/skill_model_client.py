"""Unified model client for atomic skill backends.

Supports two connection modes:
- api: remote LLM/VLM via OpenRouter or any OpenAI-compatible endpoint
- local: local model served at a base URL (vLLM, Ollama, TGI, etc.)

Skills call `client.reason(prompt)` for text LLM or `client.perceive(prompt, images)`
for VLM. The client handles JSON parsing and error recovery.
"""

from __future__ import annotations

import json
import re
import signal
import threading
from contextlib import contextmanager
from typing import Any

import requests


@contextmanager
def _total_timeout(seconds: int):
    # SIGALRM is process-main-thread only. Background workers remain bounded by
    # the requests connect/read timeout passed by SkillModelClient._post.
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handle_timeout(signum, frame):  # type: ignore[no-untyped-def]
        raise TimeoutError(f"Skill model request exceeded {seconds}s total timeout")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _parse_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


class SkillModelClient:
    """LLM/VLM client for atomic skill execution.

    Args:
        model: Model identifier (e.g. "qwen/qwen3.5-9b" for perception/reasoning skills)
        api_key: API key for authenticated endpoints
        api_base: Base URL; defaults to OpenRouter. Set to local URL for local models.
        max_tokens: Max generation tokens
        temperature: Sampling temperature
        timeout_s: Request timeout
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        api_base: str = "https://openrouter.ai/api/v1/chat/completions",
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: int = 60,
        seed: int | None = None,
        disable_thinking: bool = True,
        provider: dict[str, Any] | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.seed = seed
        self.disable_thinking = disable_thinking
        # OpenRouter routing preferences, e.g. {"order": [...], "allow_fallbacks": False}.
        # Needed because providers differ on whether they honour thinking-off.
        self.provider = provider
        self.last_response_metadata: dict[str, Any] = {}

    @staticmethod
    def _is_qwen_reasoning_model(model: str) -> bool:
        lower = (model or "").lower()
        return any(token in lower for token in ("qwen3", "qwen-3", "qwen3.5", "qwq"))

    @property
    def is_openrouter_endpoint(self) -> bool:
        return "openrouter.ai" in (self.api_base or "")

    def _post(self, messages: list[dict[str, Any]]) -> str:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        # Qwen3-family endpoints can default to thinking, which spends the whole
        # completion budget on hidden reasoning and returns empty content with
        # finish_reason "length".  Every skill then parse-fails and silently
        # falls back to its rule.  Mirror OpenRouterClient.chat and turn it off.
        if self.disable_thinking:
            if self._is_qwen_reasoning_model(self.model):
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            if self.is_openrouter_endpoint:
                # ``exclude`` only hides the reasoning field; ``effort`` is what
                # limits it, and some providers ignore both.  Verify with
                # usage.completion_tokens_details.reasoning_tokens, not by absence
                # of message.reasoning.
                payload["reasoning"] = {"exclude": True, "effort": "minimal"}
        if self.provider and self.is_openrouter_endpoint:
            payload["provider"] = dict(self.provider)

        prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
        self.last_response_metadata = {
            "model": self.model,
            "prompt_chars": prompt_chars,
            "prompt_approx_tokens": max(1, prompt_chars // 4),
            "output_chars": 0,
            "malformed_json": 0,
            "timeout_count": 0,
            "compact_retry_count": 0,
        }
        try:
            with _total_timeout(self.timeout_s):
                resp = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            usage = data.get("usage") or {}
            finish_reason = choice.get("finish_reason")
            reasoning_tokens = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
            self.last_response_metadata.update({
                "output_chars": len(content),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "reasoning_tokens": reasoning_tokens,
                "finish_reason": finish_reason,
                "provider": data.get("provider"),
                # Diagnosable rather than silent: the budget went to hidden
                # reasoning.  Read the usage counter -- ``reasoning.exclude``
                # strips message.reasoning while the tokens are still spent.
                "thinking_exhausted": bool(not content and (reasoning_tokens > 0 or finish_reason == "length")),
            })
            return content
        except (TimeoutError, requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            self.last_response_metadata["timeout_count"] = int(self.last_response_metadata.get("timeout_count") or 0) + 1
            self.last_response_metadata["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def reason(self, prompt: str, *, system: str = "You are a precise video reasoning assistant. Answer in JSON only.") -> dict[str, Any]:
        """Send a reasoning prompt and parse JSON response."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = self._post(messages)
        except (TimeoutError, requests.Timeout, requests.ConnectionError, requests.HTTPError, requests.RequestException) as exc:
            self.last_response_metadata["timeout_count"] = int(self.last_response_metadata.get("timeout_count") or 0) + 1
            self.last_response_metadata["error"] = f"{type(exc).__name__}: {exc}"
            return {"parse_error": True, "timeout": True, "error": str(exc)}
        try:
            return _parse_json_from_text(raw)
        except (json.JSONDecodeError, ValueError):
            self.last_response_metadata["malformed_json"] = int(self.last_response_metadata.get("malformed_json") or 0) + 1
            return {"raw_response": raw, "parse_error": True}

    def perceive(
        self,
        prompt: str,
        *,
        image_urls: list[str] | None = None,
        system: str = "You are a video perception assistant. You observe sampled frames from a video clip and describe what you see in structured JSON.",
    ) -> dict[str, Any]:
        """Send a VLM prompt with video clip frames and parse JSON response.

        image_urls can be:
        - base64 data URIs (data:image/jpeg;base64,...)
        - HTTP URLs to frame images
        - Local file paths (will be read and converted to data URIs)
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for url in image_urls or []:
            if url.startswith("data:") or url.startswith("http"):
                content.append({"type": "image_url", "image_url": {"url": url}})
            else:
                data_uri = self._file_to_data_uri(url)
                if data_uri:
                    content.append({"type": "image_url", "image_url": {"url": data_uri}})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        raw = self._post(messages)
        try:
            return _parse_json_from_text(raw)
        except (json.JSONDecodeError, ValueError):
            self.last_response_metadata["malformed_json"] = int(self.last_response_metadata.get("malformed_json") or 0) + 1
            return {"raw_response": raw, "parse_error": True}

    @staticmethod
    def _file_to_data_uri(path: str) -> str | None:
        """Convert a local image file to a base64 data URI."""
        import base64
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return None
        suffix = p.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "jpeg")
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:image/{mime};base64,{data}"

    @classmethod
    def from_openrouter(cls, *, model: str, api_key: str, **kwargs: Any) -> "SkillModelClient":
        return cls(model=model, api_key=api_key, api_base="https://openrouter.ai/api/v1/chat/completions", **kwargs)

    @classmethod
    def from_local(cls, *, model: str, base_url: str = "http://localhost:8000/v1/chat/completions", **kwargs: Any) -> "SkillModelClient":
        return cls(model=model, api_key="", api_base=base_url, **kwargs)
