"""Startup probe for the GRPO skill executor's model.

The trainer's default --skill-model (qwen/qwen3.5-9b) now returns empty content
on OpenRouter because the whole budget goes to hidden reasoning; skills fall
back to lexical rules with ok=True and training proceeds on them silently.
"""

from trainer.grpo.live_rollout import probe_skill_model


class _Client:
    model = "fake/model"
    def __init__(self, result, meta):
        self._result, self.last_response_metadata = result, meta
    def reason(self, prompt):
        return self._result


class _Raises(_Client):
    def reason(self, prompt):
        raise ConnectionError("boom")


def test_healthy_model_passes() -> None:
    assert probe_skill_model(_Client({"ok": True}, {"thinking_exhausted": False})) is None


def test_thinking_exhaustion_is_refused() -> None:
    reason = probe_skill_model(_Client({"parse_error": True}, {"thinking_exhausted": True, "reasoning_tokens": 253, "finish_reason": "length"}))
    assert reason and "hidden reasoning" in reason and "253" in reason


def test_unparseable_content_is_refused() -> None:
    reason = probe_skill_model(_Client({"parse_error": True, "raw_response": ""}, {"thinking_exhausted": False}))
    assert reason and "unparseable" in reason


def test_transport_failure_is_reported_not_raised() -> None:
    reason = probe_skill_model(_Raises(None, {}))
    assert reason and "ConnectionError" in reason
