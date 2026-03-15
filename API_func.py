# This file is to define the API calling functions for the agent, create a general function for each model
# By default, we use OpenRouter when open_router_api_key is set (cold-start, ask_model, etc.); else OpenAI.

import os
import openai
from anthropic import Anthropic
from google import genai
from api_keys import openai_api_key, claude_api_key, gemini_api_key, open_router_api_key

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# Cached vLLM reachability (None = not yet probed)
_vllm_reachable: bool | None = None


def _probe_vllm() -> bool:
    """One-shot TCP probe to check if the vLLM server is reachable.

    Result is cached so subsequent calls return instantly.
    """
    global _vllm_reachable
    if _vllm_reachable is not None:
        return _vllm_reachable

    import socket
    try:
        stripped = VLLM_BASE_URL.replace("http://", "").replace("https://", "").rstrip("/")
        host_port = stripped.split("/")[0]
        host, port_str = host_port.rsplit(":", 1)
        sock = socket.create_connection((host, int(port_str)), timeout=2)
        sock.close()
        _vllm_reachable = True
    except Exception:
        _vllm_reachable = False
        print(f"[API_func] vLLM at {VLLM_BASE_URL} unreachable — "
              "Qwen calls will be routed through OpenRouter.")
    return _vllm_reachable


def ask_openrouter(question, model="openai/gpt-4o-mini", temperature=0.7, max_tokens=2000):
    """
    Ask a question via OpenRouter (unified API for GPT, Claude, Gemini, etc.).
    Used by default for cold-start data gathering and ask_model when key is set.
    """
    if not (open_router_api_key and open_router_api_key.strip()):
        return f"Error: OpenRouter API key not set. Set OPENROUTER_API_KEY or add open_router_api_key in api_keys.py."
    try:
        client = openai.OpenAI(base_url=OPENROUTER_BASE, api_key=open_router_api_key.strip())
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error calling OpenRouter API: {str(e)}"


def ask_gpt(question, model="gpt-4o", temperature=0.7, max_tokens=2000):
    """
    Ask a question to GPT models. Uses OpenRouter when open_router_api_key is set (default in this repo).
    """
    if open_router_api_key and open_router_api_key.strip():
        # Prefer OpenRouter so one key is used for cold-start, etc.
        openrouter_model = model if "/" in model else f"openai/{model}"
        return ask_openrouter(question, model=openrouter_model, temperature=temperature, max_tokens=max_tokens)
    openai.api_key = openai_api_key
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling GPT API: {str(e)}"


def ask_claude(question, model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=2000):
    """
    Ask a question to Claude models using Anthropic API.
    
    Args:
        question (str): The question to ask
        model (str): The Claude model to use (default: "claude-3-5-sonnet-20241022")
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)
    
    Returns:
        str: The generated answer
    """
    try:
        client = Anthropic(api_key=claude_api_key)
        
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"


def ask_gemini(question, model="gemini-2.5-flash", temperature=0.7, max_tokens=2000):
    """
    Ask a question to Gemini models using Google Generative AI API.
    
    Args:
        question (str): The question to ask
        model (str): The Gemini model to use (default: "gemini-2.5-flash")
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)
    
    Returns:
        str: The generated answer
    """
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        response = client.models.generate_content(
            model=model,
            contents=question,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"


def _strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks (Qwen3, QwQ, etc.)."""
    import re
    if not text or "<think>" not in text:
        return text
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def ask_vllm(question, model="Qwen/Qwen3-14B", temperature=0.7, max_tokens=2000):
    """
    Ask a question via a vLLM-served model using its OpenAI-compatible endpoint.
    Configure the endpoint via VLLM_BASE_URL env var (default: http://localhost:8000/v1).

    Automatically strips ``<think>`` tags from reasoning models (Qwen3, QwQ, etc.).

    Falls back to OpenRouter when the vLLM server is unreachable and
    ``open_router_api_key`` is configured.
    """
    if not _probe_vllm():
        return _ask_qwen_via_openrouter(
            question, model=model, temperature=temperature, max_tokens=max_tokens,
        )

    try:
        client = openai.OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content or ""
        return _strip_think_tags(raw)
    except Exception as e:
        global _vllm_reachable
        _vllm_reachable = False
        fallback = _ask_qwen_via_openrouter(
            question, model=model, temperature=temperature, max_tokens=max_tokens,
        )
        if not fallback.startswith("Error"):
            return fallback
        return f"Error calling vLLM API at {VLLM_BASE_URL}: {str(e)}"


def _ask_qwen_via_openrouter(question, model="Qwen/Qwen3-14B", temperature=0.7, max_tokens=2000):
    """Route a Qwen model call through OpenRouter as a fallback.

    Handles Qwen3 reasoning-model quirks:
      - Appends ``/no_think`` if not already present so the full token
        budget goes to actual content rather than thinking.
      - Falls back to the ``reasoning`` response field when ``content``
        is empty (some OpenRouter providers put thinking there).
    """
    if not (open_router_api_key and open_router_api_key.strip()):
        return (f"Error: vLLM at {VLLM_BASE_URL} unreachable and no "
                "OpenRouter API key configured for Qwen fallback.")

    if "/no_think" not in question:
        question = question.rstrip() + "\n/no_think"

    or_model = model.lower()
    try:
        client = openai.OpenAI(
            base_url=OPENROUTER_BASE, api_key=open_router_api_key.strip(),
        )
        response = client.chat.completions.create(
            model=or_model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        if not content:
            reasoning = getattr(choice.message, "reasoning", None) or ""
            if reasoning:
                content = reasoning
        return _strip_think_tags(content)
    except Exception as e:
        return f"Error calling OpenRouter API (Qwen fallback): {str(e)}"


def ask_model(question, model=None, temperature=0.7, max_tokens=2000):
    """
    General function to ask any AI model a question.
    Automatically routes to the appropriate API based on the model name.
    
    Args:
        question (str): The question to ask
        model (str): The model to use. Can be:
            - GPT models: "gpt-4o", "gpt-4", "gpt-3.5-turbo", etc.
            - Claude models: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", etc.
            - Gemini models: "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash", etc.
            - If None, defaults to "gpt-4o"
        temperature (float): Sampling temperature (default: 0.7)
        max_tokens (int): Maximum tokens in response (default: 2000)
    
    Returns:
        str: The generated answer
    """
    # Default model if none specified (routed via OpenRouter when key is set)
    if model is None:
        model = "gpt-4o"
    model_lower = model.lower()

    # GPT-style models: use ask_gpt (which uses OpenRouter when open_router_api_key is set)
    if "gpt" in model_lower or model_lower.startswith("o1"):
        return ask_gpt(question, model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif "claude" in model_lower:
        # Anthropic Claude models
        return ask_claude(question, model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif "gemini" in model_lower:
        # Google Gemini models
        return ask_gemini(question, model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif "qwen" in model_lower or "vllm" in model_lower:
        return ask_vllm(question, model=model, temperature=temperature, max_tokens=max_tokens)

    else:
        return f"Error: Unknown model '{model}'. Please specify a GPT, Claude, Gemini, or Qwen model."