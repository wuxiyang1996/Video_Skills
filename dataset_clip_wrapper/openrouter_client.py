"""Shared OpenRouter HTTP client."""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any

import requests

DEFAULT_API_BASE = "https://openrouter.ai/api/v1/chat/completions"


def load_openrouter_api_key(*, keys_py_path: str | None = None, env_var: str = "OPENROUTER_API_KEY") -> str:
    env_key = os.environ.get(env_var)
    if env_key:
        return env_key
    if keys_py_path:
        path = Path(keys_py_path)
        if path.exists():
            spec = importlib.util.spec_from_file_location("openrouter_keys", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                key = getattr(module, "OPENROUTER_API_KEY", None)
                if key:
                    return key
    raise RuntimeError(f"Missing API key: set {env_var} or provide keys_py_path")


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("model response must be a JSON object")
    return payload


class OpenRouterClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_base: str = DEFAULT_API_BASE,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning: dict[str, Any] | None = None,
        timeout_s: int = 180,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.timeout_s = timeout_s

    def chat(self, messages: list[dict[str, Any]], *, response_format: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if response_format is not None:
            payload["response_format"] = response_format
        response = requests.post(
            self.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        content = message.get("content")
        if content is None:
            raise ValueError(
                "OpenRouter response did not include assistant content; "
                f"finish_reason={response.json()['choices'][0].get('finish_reason')}, "
                f"reasoning_preview={repr(message.get('reasoning'))[:300]}"
            )
        return content.strip()

    def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return parse_json_response(self.chat(messages, response_format={"type": "json_object"}))
