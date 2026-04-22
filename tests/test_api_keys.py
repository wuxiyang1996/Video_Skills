"""Smoke tests for provisioned API keys (Anthropic direct + Vertex AI).

Usage
-----
    # Set keys, then run:
    export ANTHROPIC_API_KEY="sk-ant-api03-..."
    export VERTEX_AI_TOKEN="AQ.Ab8..."            # short-lived Vertex token
    export VERTEX_PROJECT="your-gcp-project-id"   # required for Vertex
    export VERTEX_REGION="us-east5"                # default: us-east5

    pytest tests/test_api_keys.py -v
"""

import os
import pytest

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VERTEX_TOKEN = os.environ.get("VERTEX_AI_TOKEN", "")
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")
VERTEX_REGION = os.environ.get("VERTEX_REGION", "us-east5")

SMOKE_PROMPT = "Reply with exactly one word: hello"


def _classify_key(key: str) -> str:
    """Return 'anthropic', 'vertex', or 'unknown' based on key prefix."""
    key = key.strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("AQ.") or key.startswith("ya29.") or key.startswith("eyJ"):
        return "vertex"
    return "unknown"


# ── unit: key classification ─────────────────────────────────────────────────

class TestKeyClassification:
    def test_anthropic_prefix(self):
        assert _classify_key("sk-ant-api03-abcdef") == "anthropic"

    def test_vertex_aq_prefix(self):
        assert _classify_key("AQ.Ab8RN6J0rm_example") == "vertex"

    def test_vertex_ya29_prefix(self):
        assert _classify_key("ya29.a0AfH6SMBx_example") == "vertex"

    def test_vertex_jwt_prefix(self):
        assert _classify_key("eyJhbGciOiJSUzI1NiIsInR5cCI6Ikp") == "vertex"

    def test_unknown_key(self):
        assert _classify_key("some-random-key-1234") == "unknown"

    def test_empty_key(self):
        assert _classify_key("") == "unknown"


# ── integration: Anthropic direct API ────────────────────────────────────────

@pytest.mark.skipif(not ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
class TestAnthropicDirect:
    def test_key_format(self):
        assert _classify_key(ANTHROPIC_KEY) == "anthropic", (
            f"ANTHROPIC_API_KEY doesn't look like an Anthropic key "
            f"(prefix: {ANTHROPIC_KEY[:10]}...)"
        )

    def test_claude_smoke(self):
        from anthropic import Anthropic

        client = Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            messages=[{"role": "user", "content": SMOKE_PROMPT}],
        )
        text = msg.content[0].text.strip().lower()
        assert "hello" in text, f"Unexpected response: {text}"
        print(f"  Anthropic direct OK — response: {text}")


# ── integration: Vertex AI (Claude on Vertex) ───────────────────────────────

@pytest.mark.skipif(
    not (VERTEX_TOKEN and VERTEX_PROJECT),
    reason="VERTEX_AI_TOKEN or VERTEX_PROJECT not set",
)
class TestVertexAI:
    def test_token_format(self):
        assert _classify_key(VERTEX_TOKEN) == "vertex", (
            f"VERTEX_AI_TOKEN doesn't look like a Vertex/GCP token "
            f"(prefix: {VERTEX_TOKEN[:10]}...)"
        )

    def test_vertex_claude_smoke(self):
        """Call Claude via Vertex AI using the anthropic SDK's Vertex support."""
        from anthropic import AnthropicVertex

        client = AnthropicVertex(
            project_id=VERTEX_PROJECT,
            region=VERTEX_REGION,
            access_token=VERTEX_TOKEN,
        )
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=32,
            messages=[{"role": "user", "content": SMOKE_PROMPT}],
        )
        text = msg.content[0].text.strip().lower()
        assert "hello" in text, f"Unexpected response: {text}"
        print(f"  Vertex AI (Claude) OK — response: {text}")
