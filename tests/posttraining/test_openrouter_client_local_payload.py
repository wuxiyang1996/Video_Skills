from typing import Any

from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient


class _Response:
    ok = True

    def json(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {},
        }


def test_local_transformers_endpoint_omits_openrouter_extra_body(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def post(*args, **kwargs):
        captured.update(kwargs["json"])
        return _Response()

    monkeypatch.setattr(
        "dataset_clip_wrapper.perception.openrouter_client.requests.post", post
    )
    client = OpenRouterClient(
        model="Qwen/Qwen3.5-9B",
        api_key="local",
        api_base="http://127.0.0.1:18000/v1/chat/completions",
    )
    assert client.chat([{"role": "user", "content": "test"}]) == "{}"
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in captured


def test_openrouter_qwen_payload_keeps_provider_extension(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def post(*args, **kwargs):
        captured.update(kwargs["json"])
        return _Response()

    monkeypatch.setattr(
        "dataset_clip_wrapper.perception.openrouter_client.requests.post", post
    )
    client = OpenRouterClient(
        model="qwen/qwen3.5-9b",
        api_key="test",
        api_base="https://openrouter.ai/api/v1/chat/completions",
    )
    client.chat([{"role": "user", "content": "test"}])
    assert captured["extra_body"]["enable_thinking"] is False
