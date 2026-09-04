"""SkillModelClient must not let a thinking model spend the budget on reasoning.

Observed live: qwen/qwen3.5-9b on OpenRouter returned empty content with
finish_reason "length" and 256 completion tokens, so every LLM-backed skill
parse-failed and fell back to its lexical rule without any error surfacing.
"""

from typing import Any

from atomic_skills.skill_model_client import SkillModelClient


class _Resp:
    def __init__(self, content="", reasoning=None, finish="stop", reasoning_tokens=0, provider="X"):
        self._d = {"choices": [{"message": {"content": content, "reasoning": reasoning}, "finish_reason": finish}],
                   "usage": {"completion_tokens": 7, "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
                   "provider": provider}
    def raise_for_status(self): pass
    def json(self): return self._d


def _capture(monkeypatch, resp):
    captured: dict[str, Any] = {}
    def post(url, headers=None, json=None, timeout=None):
        captured.update(json); return resp
    monkeypatch.setattr("atomic_skills.skill_model_client.requests.post", post)
    return captured


def test_qwen3_on_openrouter_disables_thinking(monkeypatch) -> None:
    captured = _capture(monkeypatch, _Resp('{"ok": true}'))
    c = SkillModelClient(model="qwen/qwen3.5-9b", api_key="k")
    c._post([{"role": "user", "content": "x"}])
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["reasoning"]["exclude"] is True


def test_non_qwen_on_openrouter_still_limits_reasoning(monkeypatch) -> None:
    captured = _capture(monkeypatch, _Resp('{"ok": true}'))
    SkillModelClient(model="openai/gpt-oss-120b", api_key="k")._post([{"role": "user", "content": "x"}])
    assert "chat_template_kwargs" not in captured          # Qwen-specific
    assert captured["reasoning"] == {"exclude": True, "effort": "minimal"}


def test_local_endpoint_omits_the_openrouter_extension(monkeypatch) -> None:
    # transformers-serve rejects unknown top-level fields with HTTP 422.
    captured = _capture(monkeypatch, _Resp('{"ok": true}'))
    SkillModelClient(model="Qwen/Qwen3.5-9B", api_base="http://127.0.0.1:18000/v1/chat/completions")._post([{"role": "user", "content": "x"}])
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning" not in captured


def test_can_opt_out(monkeypatch) -> None:
    captured = _capture(monkeypatch, _Resp('{"ok": true}'))
    SkillModelClient(model="qwen/qwen3.5-9b", api_key="k", disable_thinking=False)._post([{"role": "user", "content": "x"}])
    assert "chat_template_kwargs" not in captured


def test_thinking_exhaustion_is_flagged_from_usage_even_when_reasoning_is_hidden(monkeypatch) -> None:
    # Observed live on SiliconFlow: reasoning.exclude strips message.reasoning but
    # completion_tokens_details.reasoning_tokens still shows the spend.
    _capture(monkeypatch, _Resp(content="", reasoning=None, finish="length", reasoning_tokens=253))
    c = SkillModelClient(model="qwen/qwen3.5-9b", api_key="k")
    assert c._post([{"role": "user", "content": "x"}]) == ""
    assert c.last_response_metadata["thinking_exhausted"] is True
    assert c.last_response_metadata["reasoning_tokens"] == 253


def test_provider_preference_is_forwarded_on_openrouter(monkeypatch) -> None:
    captured = _capture(monkeypatch, _Resp('{"ok": true}'))
    SkillModelClient(model="qwen/qwen3.5-9b", api_key="k", provider={"order": ["DeepInfra"], "allow_fallbacks": False})._post([{"role": "user", "content": "x"}])
    assert captured["provider"] == {"order": ["DeepInfra"], "allow_fallbacks": False}


def test_provider_preference_is_not_sent_to_local_endpoints(monkeypatch) -> None:
    captured = _capture(monkeypatch, _Resp('{"ok": true}'))
    SkillModelClient(model="Qwen/Qwen3.5-9B", api_base="http://127.0.0.1:1/v1/chat/completions", provider={"order": ["X"]})._post([{"role": "user", "content": "x"}])
    assert "provider" not in captured


def test_normal_completion_is_not_flagged(monkeypatch) -> None:
    _capture(monkeypatch, _Resp('{"ok": true}'))
    c = SkillModelClient(model="qwen/qwen3.5-9b", api_key="k")
    c._post([{"role": "user", "content": "x"}])
    assert c.last_response_metadata["thinking_exhausted"] is False
