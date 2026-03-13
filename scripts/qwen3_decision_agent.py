#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-14B Decision Agent with Skill Bank Guidance.

Uses Qwen3-14B (served via vLLM) as the decision-making LLM to play games,
querying a pre-built skill bank (from GPT-5.4 extraction) to guide action
selection at each step.

Pipeline:
  1. Load skill bank from labeling/output/gpt54_skillbank/<game>/
  2. For each game episode:
     a. Get state summary (deterministic + LLM)
     b. Infer intention via Qwen3-14B
     c. Query skill bank for relevant skill guidance
     d. Inject skill protocol into LLM prompt
     e. Qwen3-14B selects action
  3. Save rollouts to test_rollout/decision_agent/<game>/<timestamp>/

Usage (from Game-AI-Agent root):

    export PYTHONPATH="$(pwd):$(pwd)/../GamingAgent:$PYTHONPATH"
    export VLLM_BASE_URL="http://localhost:8000/v1"

    # Single game, 3 episodes
    python -m scripts.qwen3_decision_agent --games twenty_forty_eight --episodes 3

    # One episode per game (run each game once), on GPU 0, verbose
    python -m scripts.qwen3_decision_agent --one_per_game --gpu 0 -v

    # All available games, 5 episodes each, verbose
    python -m scripts.qwen3_decision_agent --episodes 5 -v

    # Specific GPU(s)
    python -m scripts.qwen3_decision_agent --gpu 1 --games tetris --episodes 2
    python -m scripts.qwen3_decision_agent --gpu 0,1 --one_per_game

    # Without skill bank (baseline)
    python -m scripts.qwen3_decision_agent --no-bank --episodes 3

    # Custom skill bank path
    python -m scripts.qwen3_decision_agent --bank /path/to/bank --episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CODEBASE_ROOT = SCRIPT_DIR.parent
GAMINGAGENT_ROOT = CODEBASE_ROOT.parent / "GamingAgent"

for p in [str(CODEBASE_ROOT), str(GAMINGAGENT_ROOT)]:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from data_structure.experience import Experience, Episode, Episode_Buffer

from env_wrappers.gamingagent_nl_wrapper import GamingAgentNLWrapper
from evaluate_gamingagent.gym_like import make_gaming_env, list_games
from evaluate_gamingagent.game_configs import GAME_CONFIGS

from decision_agents.agent_helper import (
    get_state_summary,
    infer_intention,
    strip_think_tags,
    select_skill_from_bank,
    HARD_SUMMARY_CHAR_LIMIT,
)
from decision_agents.dummy_agent import (
    extract_action,
    GAME_GAMINGAGENT,
)

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

from skill_agents.skill_bank.bank import SkillBankMVP

try:
    from skill_agents.query import SkillQueryEngine
except ImportError:
    SkillQueryEngine = None

try:
    from env_wrappers.sokoban_nl_wrapper import SokobanNLWrapper
except ImportError:
    SokobanNLWrapper = None

# RAG embedding for semantic action matching
try:
    from rag import get_text_embedder
    from rag.embedding.base import TextEmbedderBase
except ImportError:
    get_text_embedder = None
    TextEmbedderBase = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen3-14B"
DEFAULT_BANK_DIR = CODEBASE_ROOT / "labeling" / "output" / "gpt54_skillbank"
DEFAULT_OUTPUT_DIR = CODEBASE_ROOT / "test_rollout" / "decision_agent"

LMGAME_BENCH_NAMES = {"twenty_forty_eight", "sokoban", "candy_crush", "tetris"}

# ---------------------------------------------------------------------------
# Skill bank loading
# ---------------------------------------------------------------------------

def load_skill_bank(
    bank_path: str,
    *,
    use_query_engine: bool = True,
) -> Tuple[Any, Any]:
    """Load a SkillBankMVP from a JSONL file or directory."""
    bp = Path(bank_path)
    if bp.is_dir():
        candidates = ["bank.jsonl", "skill_bank.jsonl"]
        jsonl = None
        for c in candidates:
            if (bp / c).exists():
                jsonl = bp / c
                break
        if jsonl is None:
            jsonls = sorted(bp.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
            jsonl = jsonls[0] if jsonls else None
        if jsonl is None:
            print(f"[load_skill_bank] WARNING: no .jsonl found in {bp}, using empty bank.")
            return None, None
        bp = jsonl

    bank = SkillBankMVP(path=str(bp))
    bank.load()
    print(f"[load_skill_bank] Loaded {len(bank)} skills from {bp}")

    engine = None
    if use_query_engine and SkillQueryEngine is not None and len(bank) > 0:
        try:
            engine = SkillQueryEngine(bank)
            print(f"[load_skill_bank] SkillQueryEngine initialised")
        except Exception as exc:
            print(f"[load_skill_bank] SkillQueryEngine init failed: {exc}")
    return bank, engine


def get_skill_guidance(
    skill_bank: Any,
    state_text: str,
    game_name: str = "",
    intention: str = "",
    structured_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Query the skill bank for guidance using expanded state context.

    When *structured_state* is provided it is passed through to
    ``select_skill_from_bank`` so the ``SkillQueryEngine.select()`` path
    can compute applicability scores (the full find-apply protocol).
    """
    if skill_bank is None:
        return None
    key_parts = []
    if game_name:
        key_parts.append(game_name)
    if intention:
        key_parts.append(intention)
    key_parts.append(state_text[:1500])
    key = " ".join(key_parts)

    state_for_scoring = None
    if structured_state and isinstance(structured_state, dict):
        state_for_scoring = {
            k: (float(v) if isinstance(v, (int, float, bool)) else 1.0)
            for k, v in structured_state.items()
            if v is not None and str(v).strip()
        }

    try:
        result = select_skill_from_bank(
            skill_bank, key,
            current_state=state_for_scoring,
            top_k=1,
        )
        if result and result.get("skill_id"):
            if not result.get("skill_name"):
                underlying = (
                    getattr(skill_bank, "_bank", None)
                    or getattr(skill_bank, "bank", None)
                    or skill_bank
                )
                if hasattr(underlying, "get_skill"):
                    skill_obj = underlying.get_skill(result["skill_id"])
                    if skill_obj:
                        result["skill_name"] = skill_obj.name or result["skill_id"]
                        if not result.get("execution_hint"):
                            result["execution_hint"] = skill_obj.strategic_description or ""
            return result
    except Exception:
        pass
    return None


def format_skill_guidance_for_prompt(
    guidance: Optional[Dict[str, Any]],
    protocol_step_idx: int = 0,
) -> str:
    """Format skill guidance as text to inject into LLM prompts.

    When *protocol_step_idx* > 0 the current step is highlighted with
    ``>>`` so the LLM knows where it is in the protocol sequence.
    """
    if guidance is None or not guidance.get("skill_id"):
        return ""

    parts = [f"\n--- Active Skill: {guidance.get('skill_name', guidance['skill_id'])} ---"]
    if guidance.get("execution_hint"):
        parts.append(f"  Strategy: {guidance['execution_hint'][:200]}")

    protocol = guidance.get("protocol", {})
    steps = protocol.get("steps", []) if isinstance(protocol, dict) else []
    if steps:
        parts.append(f"  Plan ({len(steps)} steps):")
        for i, step in enumerate(steps[:7], 1):
            marker = ">>" if (i - 1) == protocol_step_idx else "  "
            parts.append(f"  {marker} {i}. {step}")

    preconditions = protocol.get("preconditions", []) if isinstance(protocol, dict) else []
    if preconditions:
        parts.append(f"  Preconditions: {'; '.join(preconditions[:3])}")

    success = protocol.get("success_criteria", []) if isinstance(protocol, dict) else []
    if success:
        parts.append(f"  Done when: {'; '.join(success[:2])}")

    abort = protocol.get("abort_criteria", []) if isinstance(protocol, dict) else []
    if abort:
        parts.append(f"  Abort if: {'; '.join(abort[:2])}")

    parts.append("--- end skill ---\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert game-playing agent. "
    "You receive a game state and must choose exactly one action by its NUMBER.\n\n"
    "Rules:\n"
    "- Study the state carefully before choosing.\n"
    "- Consider which action makes the most progress toward winning.\n"
    "- NEVER repeat the same action more than 2 times in a row — try something different.\n"
    "- If recent actions got zero reward, change strategy.\n\n"
    "Output format (strict):\n"
    "REASONING: <1-2 sentences>\n"
    "ACTION: <number>\n"
)

USER_TEMPLATE = (
    "Game state:\n\n{state}\n\n"
    "{recent_context}"
    "Available actions (pick ONE by number):\n{actions}\n\n"
    "Choose the best action. Output REASONING then ACTION number."
)

MAX_REPEAT_ACTIONS = 2


# ---------------------------------------------------------------------------
# RAG-based action embedding matcher
# ---------------------------------------------------------------------------

class ActionEmbeddingMatcher:
    """Lazily-initialized semantic action matcher using RAG text embeddings.

    Embeds valid actions once per action-set change, then scores the LLM's
    raw output against them via cosine similarity.  Falls back gracefully
    to None when the embedder is unavailable.
    """

    def __init__(self):
        self._embedder = None
        self._init_attempted = False
        self._cached_actions: Optional[List[str]] = None
        self._cached_embeddings = None  # np.ndarray or None

    def _ensure_embedder(self):
        if self._init_attempted:
            return
        self._init_attempted = True
        if get_text_embedder is None:
            return
        try:
            self._embedder = get_text_embedder()
            print("[ActionEmbeddingMatcher] TextEmbedder loaded")
        except Exception as exc:
            print(f"[ActionEmbeddingMatcher] TextEmbedder unavailable: {exc}")

    def _embed_actions(self, actions: List[str]):
        action_key = tuple(actions)
        if self._cached_actions == action_key:
            return
        self._cached_actions = action_key
        try:
            import numpy as np
            self._cached_embeddings = self._embedder.encode(
                actions, prompt_name="passage",
            )
        except Exception:
            self._cached_embeddings = None

    def match(
        self,
        raw: str,
        valid_actions: List[str],
        threshold: float = 0.35,
    ) -> Optional[str]:
        """Return the best semantic match or None if below threshold."""
        self._ensure_embedder()
        if self._embedder is None or not valid_actions or not raw:
            return None
        try:
            import numpy as np
            self._embed_actions(valid_actions)
            if self._cached_embeddings is None:
                return None
            q_emb = self._embedder.encode(raw, prompt_name="query")
            q_emb = np.atleast_2d(q_emb).astype(np.float32)
            scores = (q_emb @ self._cached_embeddings.T).squeeze(0)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            if best_score >= threshold:
                return valid_actions[best_idx]
        except Exception:
            pass
        return None


_action_matcher = ActionEmbeddingMatcher()


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _fuzzy_match_action(raw: str, valid_actions: List[str]) -> Optional[str]:
    """Match *raw* against *valid_actions* with multi-strategy fallback.

    1. Exact match (case-insensitive)
    2. Numbered selection (e.g. "3" -> third action)
    3. Substring containment
    4. Edit distance (for short actions) or token overlap (for long ones)
    """
    raw = raw.strip()
    if not raw or not valid_actions:
        return None

    raw_lower = raw.lower().rstrip(".").strip()
    lower_map = {a.lower(): a for a in valid_actions}

    if raw_lower in lower_map:
        return lower_map[raw_lower]

    num_m = re.match(r"^(\d+)\.?\s*$", raw_lower)
    if num_m:
        idx = int(num_m.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]

    num_m2 = re.search(r"(?:^|\s)(\d+)\s*[.:\-]", raw_lower)
    if num_m2:
        idx = int(num_m2.group(1)) - 1
        if 0 <= idx < len(valid_actions):
            return valid_actions[idx]

    best_substr_match = None
    best_substr_len = 0
    for canon_lower, canon in lower_map.items():
        if len(canon_lower) < 3 and len(raw_lower) > 5:
            continue
        if canon_lower in raw_lower or raw_lower in canon_lower:
            if len(canon_lower) > best_substr_len:
                best_substr_len = len(canon_lower)
                best_substr_match = canon
    if best_substr_match:
        return best_substr_match

    raw_core = re.sub(r"[^a-z0-9_]", "", raw_lower)
    best_core_match = None
    best_core_len = 0
    for canon_lower, canon in lower_map.items():
        canon_core = re.sub(r"[^a-z0-9_]", "", canon_lower)
        if len(canon_core) < 3 and len(raw_core) > 5:
            continue
        if raw_core and canon_core and (raw_core in canon_core or canon_core in raw_core):
            if len(canon_core) > best_core_len:
                best_core_len = len(canon_core)
                best_core_match = canon
    if best_core_match:
        return best_core_match

    if len(valid_actions) <= 20 and all(len(a) < 30 for a in valid_actions):
        best_dist = float("inf")
        best_action = None
        for a in valid_actions:
            d = _edit_distance(raw_lower, a.lower())
            threshold = max(2, len(a) // 3)
            if d < best_dist and d <= threshold:
                best_dist = d
                best_action = a
        if best_action:
            return best_action

    raw_tokens = set(re.split(r"[\s,_\-()=]+", raw_lower))
    raw_tokens = {t for t in raw_tokens if len(t) >= 2}
    if raw_tokens:
        best_overlap = 0
        best_action = None
        for a in valid_actions:
            a_tokens = set(re.split(r"[\s,_\-()=]+", a.lower()))
            a_tokens = {t for t in a_tokens if len(t) >= 2}
            overlap = len(raw_tokens & a_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_action = a
        if best_action and best_overlap >= max(1, len(raw_tokens) // 3):
            return best_action

    # RAG embedding semantic match as final fallback
    rag_match = _action_matcher.match(raw, valid_actions)
    if rag_match:
        return rag_match

    return None


def parse_qwen_response(reply: str, valid_actions: List[str]) -> Tuple[str, Optional[str]]:
    """Parse Qwen response to extract action and reasoning with fuzzy matching."""
    if not reply:
        return (valid_actions[0] if valid_actions else "stay"), None

    cleaned = strip_think_tags(reply)
    if not cleaned:
        cleaned = reply

    reasoning = None
    reasoning_m = re.search(r"REASONING\s*:\s*(.+?)(?=\nACTION|\Z)", cleaned, re.DOTALL | re.IGNORECASE)
    if reasoning_m:
        reasoning = reasoning_m.group(1).strip()

    action_m = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if action_m:
        raw_action = action_m.group(1).strip()
        matched = _fuzzy_match_action(raw_action, valid_actions)
        if matched:
            return matched, reasoning

    for pattern in [
        r"(?:choose|select|pick|answer)\s*(?:is\s*)?:?\s*(.+?)(?:\n|$)",
        r"(?:I (?:will|choose|select|pick))\s+(.+?)(?:\n|\.|$)",
    ]:
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            matched = _fuzzy_match_action(m.group(1).strip(), valid_actions)
            if matched:
                return matched, reasoning

    extracted = extract_action(cleaned, GAME_GAMINGAGENT, "Valid actions: " + ", ".join(valid_actions))
    if extracted:
        return extracted, reasoning

    for a in valid_actions:
        if len(a) >= 3 and a.lower() in cleaned.lower():
            return a, reasoning

    # RAG embedding similarity as final semantic fallback
    action_m_text = re.search(r"ACTION\s*:\s*(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    raw_for_rag = action_m_text.group(1).strip() if action_m_text else cleaned[-200:]
    rag_match = _action_matcher.match(raw_for_rag, valid_actions)
    if rag_match:
        return rag_match, reasoning

    return (valid_actions[0] if valid_actions else "stay"), reasoning


def _format_numbered_actions(action_names: List[str]) -> str:
    """Format actions as a numbered list for the LLM prompt."""
    return "\n".join(f"  {i+1}. {a}" for i, a in enumerate(action_names))


def _build_recent_context(recent_actions: List[str], recent_rewards: List[float]) -> str:
    """Build a short context string about recent actions to prevent repetition."""
    if not recent_actions:
        return ""
    lines = ["Recent actions and rewards:"]
    for a, r in zip(recent_actions[-5:], recent_rewards[-5:]):
        lines.append(f"  {a} -> reward {r:.1f}")
    total = sum(recent_rewards[-5:])
    if total == 0 and len(recent_actions) >= 3:
        lines.append("WARNING: Recent actions got 0 reward. Try a DIFFERENT action!")
    lines.append("")
    return "\n".join(lines) + "\n"


def qwen3_action(
    state_nl: str,
    action_names: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    skill_guidance: Optional[Dict[str, Any]] = None,
    recent_actions: Optional[List[str]] = None,
    recent_rewards: Optional[List[float]] = None,
    protocol_step_idx: int = 0,
) -> Tuple[str, Optional[str]]:
    """Query Qwen3-14B via vLLM. Returns (action, reasoning)."""
    if ask_model is None:
        return (action_names[0] if action_names else "stay"), None

    recent_context = _build_recent_context(
        recent_actions or [], recent_rewards or [],
    )
    user_content = USER_TEMPLATE.format(
        state=state_nl[:4000],
        actions=_format_numbered_actions(action_names),
        recent_context=recent_context,
    )
    skill_text = format_skill_guidance_for_prompt(skill_guidance, protocol_step_idx)
    prompt = SYSTEM_PROMPT + skill_text + "\n" + user_content

    try:
        reply = ask_model(prompt, model=model, temperature=temperature, max_tokens=512)
        if reply and not reply.startswith("Error"):
            return parse_qwen_response(reply, action_names)
    except Exception as exc:
        print(f"    [WARN] Qwen3-14B call failed ({exc}), using fallback")

    return (action_names[0] if action_names else "stay"), None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class _SkillTracker:
    """Protocol-aware skill lifecycle tracker.

    Implements the *find-apply* protocol:
    - **Find**: triggered on first step and whenever ``should_reselect()``
      returns True (duration exceeded, zero-reward stall, or abort criteria
      matched in the current state).
    - **Apply**: tracks progress through protocol steps, checks
      success/abort criteria each step, and exposes the current protocol
      step index for richer prompt injection.
    """

    def __init__(self):
        self.active_skill_id: Optional[str] = None
        self.active_skill_name: str = ""
        self.steps_on_skill: int = 0
        self.reward_on_skill: float = 0.0
        self.max_skill_duration: int = 10
        self.skill_switches: int = 0

        self._protocol: Optional[Dict[str, Any]] = None
        self._protocol_step_idx: int = 0
        self._success_criteria: List[str] = []
        self._abort_criteria: List[str] = []
        self._reselect_reason: str = ""

    @property
    def reselect_reason(self) -> str:
        return self._reselect_reason

    @property
    def protocol_step_idx(self) -> int:
        return self._protocol_step_idx

    @property
    def total_protocol_steps(self) -> int:
        if self._protocol and isinstance(self._protocol, dict):
            return len(self._protocol.get("steps", []))
        return 0

    def should_reselect(
        self,
        guidance: Optional[Dict[str, Any]],
        state_text: str = "",
    ) -> bool:
        """Decide whether to force a skill re-selection.

        Checks (in order): no skill, duration exceeded, zero-reward stall,
        abort criteria keyword match in current state, success criteria match.
        """
        self._reselect_reason = ""
        if guidance is None or not guidance.get("skill_id"):
            self._reselect_reason = "no_skill"
            return True
        new_id = guidance["skill_id"]
        if new_id != self.active_skill_id:
            return False

        if self.steps_on_skill >= self.max_skill_duration:
            self._reselect_reason = "duration_exceeded"
            return True

        if self.steps_on_skill >= 4 and self.reward_on_skill <= 0:
            self._reselect_reason = "zero_reward_stall"
            return True

        state_lower = state_text.lower() if state_text else ""
        if state_lower and self._abort_criteria:
            for crit in self._abort_criteria:
                tokens = [t for t in crit.lower().split() if len(t) >= 3]
                if tokens and all(tok in state_lower for tok in tokens[:3]):
                    self._reselect_reason = f"abort:{crit[:40]}"
                    return True

        if state_lower and self._success_criteria and self.steps_on_skill >= 2:
            for crit in self._success_criteria:
                tokens = [t for t in crit.lower().split() if len(t) >= 3]
                if tokens and all(tok in state_lower for tok in tokens[:3]):
                    self._reselect_reason = f"success:{crit[:40]}"
                    return True

        return False

    def update(self, skill_id: Optional[str], skill_name: str, reward: float):
        if skill_id != self.active_skill_id:
            self.active_skill_id = skill_id
            self.active_skill_name = skill_name
            self.steps_on_skill = 1
            self.reward_on_skill = reward
            self.skill_switches += 1
            self._protocol_step_idx = 0
        else:
            self.steps_on_skill += 1
            self.reward_on_skill += reward
            n_steps = self.total_protocol_steps
            if n_steps > 0:
                self._protocol_step_idx = min(
                    self._protocol_step_idx + 1, n_steps - 1,
                )

    def set_protocol(self, protocol: Optional[Dict[str, Any]]):
        """Load protocol and extract criteria for lifecycle checks."""
        self._protocol = protocol
        self._protocol_step_idx = 0
        self._success_criteria = []
        self._abort_criteria = []
        if protocol and isinstance(protocol, dict):
            dur = protocol.get("expected_duration", 0)
            if isinstance(dur, (int, float)) and dur > 0:
                self.max_skill_duration = max(int(dur) + 3, 5)
            else:
                self.max_skill_duration = 10
            self._success_criteria = protocol.get("success_criteria", []) or []
            self._abort_criteria = protocol.get("abort_criteria", []) or []
        else:
            self.max_skill_duration = 10


def _apply_anti_repetition(
    action: str,
    valid_actions: List[str],
    recent_actions: List[str],
    recent_rewards: List[float],
) -> str:
    """If the same action was repeated MAX_REPEAT_ACTIONS times with no reward,
    pick a different action from valid_actions."""
    if len(recent_actions) < MAX_REPEAT_ACTIONS:
        return action
    tail = recent_actions[-MAX_REPEAT_ACTIONS:]
    tail_rewards = recent_rewards[-MAX_REPEAT_ACTIONS:]
    if all(a == action for a in tail) and sum(tail_rewards) <= 0:
        alternatives = [a for a in valid_actions if a != action]
        if alternatives:
            import random
            chosen = random.choice(alternatives)
            return chosen
    return action


def run_episode(
    game: str,
    max_steps: int,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    verbose: bool = False,
    skill_bank: Any = None,
) -> Tuple[Episode, Dict[str, Any]]:
    """Run one episode with Qwen3-14B decision agent + skill bank guidance."""

    game_cfg = GAME_CONFIGS.get(game)
    task = game_cfg.description if game_cfg else f"Play {game}"

    base_env = make_gaming_env(game=game, max_steps=max_steps)

    if game == "sokoban" and SokobanNLWrapper is not None:
        env = SokobanNLWrapper(base_env, reflect_every=3)
    else:
        env = GamingAgentNLWrapper(base_env)

    obs_nl, info = env.reset()
    action_names = info.get("action_names", [])
    structured_state = info.get("structured_state")

    experiences: List[Experience] = []
    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False
    last_state_summary = ""
    current_intention = ""

    recent_actions: List[str] = []
    recent_rewards: List[float] = []
    skill_tracker = _SkillTracker()
    last_guidance: Optional[Dict[str, Any]] = None

    while step_count < max_steps:
        step_actions = action_names if action_names else ["stay"]

        summary = get_state_summary(
            obs_nl,
            structured_state=structured_state,
            game=GAME_GAMINGAGENT,
            model=model,
        )
        if summary:
            last_state_summary = summary[:HARD_SUMMARY_CHAR_LIMIT]

        intention = infer_intention(
            last_state_summary or obs_nl,
            game=GAME_GAMINGAGENT,
            model=model,
            context={
                "last_actions": recent_actions[-5:],
                "task": task,
            },
        )
        if intention:
            current_intention = intention

        need_reselect = skill_tracker.should_reselect(
            last_guidance,
            state_text=last_state_summary or obs_nl,
        )
        if need_reselect or last_guidance is None:
            if verbose and skill_tracker.reselect_reason:
                print(f"    [skill-reselect] reason={skill_tracker.reselect_reason}")
            guidance = get_skill_guidance(
                skill_bank,
                last_state_summary or obs_nl,
                game_name=game,
                intention=current_intention,
                structured_state=structured_state,
            )
            if guidance and guidance.get("skill_id"):
                if need_reselect and last_guidance and guidance.get("skill_id") == last_guidance.get("skill_id"):
                    exclude_id = guidance["skill_id"]
                    alt = _try_alternate_skill(skill_bank, last_state_summary or obs_nl, game, current_intention, exclude_id)
                    if alt:
                        guidance = alt
                skill_tracker.set_protocol(guidance.get("protocol"))
            last_guidance = guidance
        else:
            guidance = last_guidance

        action, reasoning = qwen3_action(
            state_nl=obs_nl,
            action_names=step_actions,
            model=model,
            temperature=temperature,
            skill_guidance=guidance,
            recent_actions=recent_actions,
            recent_rewards=recent_rewards,
            protocol_step_idx=skill_tracker.protocol_step_idx,
        )

        action = _apply_anti_repetition(action, step_actions, recent_actions, recent_rewards)

        try:
            next_obs_nl, reward, terminated, truncated, next_info = env.step(action)
        except Exception as e:
            print(f"    [ERROR at step {step_count}] env.step failed: {e}")
            break

        done = terminated or truncated
        total_reward += reward
        next_action_names = next_info.get("action_names", action_names)
        next_structured_state = next_info.get("structured_state")

        recent_actions.append(str(action))
        recent_rewards.append(float(reward))

        skill_id = guidance.get("skill_id") if guidance else None
        skill_name = guidance.get("skill_name", "") if guidance else ""
        skill_tracker.update(skill_id, skill_name, float(reward))

        exp = Experience(
            state=obs_nl,
            action=str(action),
            reward=float(reward),
            next_state=next_obs_nl,
            done=done,
            intentions=current_intention if current_intention else None,
            tasks=task,
        )
        exp.idx = step_count
        exp.action_type = "primitive"
        exp.summary_state = last_state_summary if last_state_summary else None
        exp.available_actions = list(step_actions) if step_actions else None
        exp.interface = {"env_name": "gamingagent", "game_name": game}

        if guidance and guidance.get("skill_id"):
            exp.sub_tasks = guidance.get("skill_name", guidance["skill_id"])

        exp.summary = last_state_summary or None

        experiences.append(exp)

        if verbose:
            intent_short = (current_intention[:60] + "...") if len(current_intention) > 60 else current_intention
            reason_short = (reasoning[:60] + "...") if reasoning and len(reasoning) > 60 else reasoning
            skill_disp = skill_tracker.active_skill_name or "none"
            proto_total = skill_tracker.total_protocol_steps
            if proto_total > 0:
                skill_info = (
                    f"{skill_disp} (step {skill_tracker.steps_on_skill}/{skill_tracker.max_skill_duration}, "
                    f"proto {skill_tracker.protocol_step_idx+1}/{proto_total})"
                )
            else:
                skill_info = f"{skill_disp} (step {skill_tracker.steps_on_skill}/{skill_tracker.max_skill_duration})"
            print(
                f"  step {step_count}: action={action}, reward={reward:.2f}, "
                f"cum={total_reward:.2f}\n"
                f"    intention: {intent_short}\n"
                f"    skill:     {skill_info}\n"
                f"    reasoning: {reason_short}"
            )

        obs_nl = next_obs_nl
        action_names = next_action_names
        structured_state = next_structured_state
        step_count += 1

        if done:
            break

    env.close()

    episode = Episode(
        experiences=experiences,
        task=task,
        env_name="gamingagent",
        game_name=game,
        metadata={
            "done": terminated or truncated,
            "steps": step_count,
            "total_reward": total_reward,
            "model": model,
            "agent_type": "qwen3_14b_decision_with_skillbank",
            "final_intention": current_intention,
            "final_state_summary": last_state_summary,
            "skill_switches": skill_tracker.skill_switches,
            "unique_actions": len(set(recent_actions)),
        },
    )
    episode.set_outcome()

    stats = {
        "game": game,
        "steps": step_count,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "model": model,
        "agent_type": "qwen3_14b_decision_with_skillbank",
        "skill_switches": skill_tracker.skill_switches,
        "unique_actions": len(set(recent_actions)),
    }
    return episode, stats


def _try_alternate_skill(
    skill_bank: Any,
    state_text: str,
    game_name: str,
    intention: str,
    exclude_id: str,
) -> Optional[Dict[str, Any]]:
    """Try to find an alternative skill excluding *exclude_id*."""
    if skill_bank is None:
        return None
    bank = getattr(skill_bank, "bank", skill_bank) if not hasattr(skill_bank, "skill_ids") else skill_bank
    try:
        ids = list(bank.skill_ids)
    except AttributeError:
        return None
    alt_ids = [sid for sid in ids if sid != exclude_id]
    if not alt_ids:
        return None
    import random
    chosen = random.choice(alt_ids)
    has_get_skill = hasattr(bank, "get_skill")
    from decision_agents.agent_helper import _get_protocol_for_skill
    protocol = _get_protocol_for_skill(bank, chosen)
    skill_obj = bank.get_skill(chosen) if has_get_skill else None
    skill_name = ""
    execution_hint = ""
    if skill_obj:
        skill_name = skill_obj.name or chosen
        execution_hint = skill_obj.strategic_description or ""
    return {
        "skill_id": chosen,
        "skill_name": skill_name,
        "why_selected": f"alternate skill (prev exhausted)",
        "protocol": protocol,
        "execution_hint": execution_hint,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _sanitize_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_keys(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_keys(v) for v in obj]
    return obj


def run_game_rollouts(
    game_name: str,
    args: argparse.Namespace,
    game_dir: Path,
    skill_bank: Any = None,
) -> Dict[str, Any]:
    """Run all episodes for one game and save outputs."""
    game_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = game_dir / "rollouts.jsonl"

    game_cfg = GAME_CONFIGS.get(game_name)
    if args.max_steps:
        effective_max_steps = args.max_steps
    elif game_cfg:
        effective_max_steps = game_cfg.max_steps
    else:
        effective_max_steps = 50

    episode_buffer = Episode_Buffer(buffer_size=args.episodes + 10)
    all_stats: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx in range(args.episodes):
        print(f"\n  [{game_name}] Episode {ep_idx + 1}/{args.episodes}")

        try:
            episode, stats = run_episode(
                game=game_name,
                max_steps=effective_max_steps,
                model=args.model,
                temperature=args.temperature,
                verbose=args.verbose,
                skill_bank=skill_bank,
            )

            stats["episode_index"] = ep_idx
            print(f"    Steps: {stats['steps']}, Reward: {stats['total_reward']:.2f}")

            episode_buffer.add_episode(episode)
            all_stats.append(stats)

            ep_data = _sanitize_keys(episode.to_dict())
            ep_data["metadata"] = stats
            ep_path = game_dir / f"episode_{ep_idx:03d}.json"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(ep_data, f, indent=2, ensure_ascii=False, default=str)

            record = _sanitize_keys(episode.to_dict())
            record["rollout_metadata"] = stats
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        except Exception as e:
            print(f"    [ERROR] Episode {ep_idx + 1} failed: {e}")
            traceback.print_exc()
            all_stats.append({
                "game": game_name,
                "episode_index": ep_idx,
                "error": str(e),
                "steps": 0,
                "total_reward": 0.0,
            })

    elapsed = time.time() - t0

    if len(episode_buffer) == 0:
        if game_dir.exists():
            shutil.rmtree(game_dir)
        print(f"\n  [INVALID] {game_name}: 0 successful episodes, removed output dir")
        return {"game": game_name, "total_episodes": 0, "invalid": True}

    buffer_path = game_dir / "episode_buffer.json"
    episode_buffer.save_to_json(str(buffer_path))
    print(f"\n  Saved {len(episode_buffer)} episodes to {buffer_path}")

    summary: Dict[str, Any] = {
        "game": game_name,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "qwen3_14b_decision_with_skillbank",
        "total_episodes": len(all_stats),
        "elapsed_seconds": elapsed,
        "episode_stats": all_stats,
    }
    valid_stats = [s for s in all_stats if "error" not in s]
    if valid_stats:
        rewards = [s["total_reward"] for s in valid_stats]
        steps = [s["steps"] for s in valid_stats]
        summary["mean_reward"] = sum(rewards) / len(rewards)
        summary["mean_steps"] = sum(steps) / len(steps)
        summary["max_reward"] = max(rewards)
        summary["min_reward"] = min(rewards)

    summary_path = game_dir / "rollout_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-14B Decision Agent with Skill Bank guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=str, nargs="+", default=None,
                        help="Games to play (default: all available LMGame-Bench)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per game (default: 3)")
    parser.add_argument("--one_per_game", action="store_true",
                        help="Run exactly 1 episode per game (overrides --episodes)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (default: per-game config)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3)")
    parser.add_argument("--bank", type=str, default=None,
                        help=f"Skill bank path (default: {DEFAULT_BANK_DIR})")
    parser.add_argument("--no-bank", action="store_true",
                        help="Run without skill bank (baseline)")
    parser.add_argument("--no-query-engine", action="store_true",
                        help="Disable SkillQueryEngine")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device(s) to use, e.g. '0' or '0,1' (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print step-by-step details")

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.one_per_game:
        args.episodes = 1

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine available games
    try:
        all_games = list_games()
    except Exception:
        all_games = sorted(LMGAME_BENCH_NAMES)

    if args.games:
        requested_games = args.games
    else:
        requested_games = all_games

    available_games: List[str] = []
    for g in requested_games:
        cfg = GAME_CONFIGS.get(g)
        if cfg is None:
            print(f"[WARNING] Game '{g}' not found, skipping.")
            continue
        if not cfg.available:
            print(f"[WARNING] Game '{g}' not available, skipping.")
            continue
        available_games.append(g)

    if not available_games:
        print("[ERROR] No games available.")
        sys.exit(1)

    # Load skill bank
    skill_bank_obj = None
    bank_path_str = "none"

    if not args.no_bank:
        bank_base = Path(args.bank) if args.bank else DEFAULT_BANK_DIR

        per_game_banks: Dict[str, Any] = {}
        for game_name in available_games:
            game_bank_dir = bank_base / game_name
            if game_bank_dir.exists():
                bank, engine = load_skill_bank(
                    str(game_bank_dir),
                    use_query_engine=not args.no_query_engine,
                )
                per_game_banks[game_name] = engine if engine is not None else bank
            else:
                bank_obj, engine = load_skill_bank(
                    str(bank_base),
                    use_query_engine=not args.no_query_engine,
                )
                if bank_obj is not None:
                    per_game_banks[game_name] = engine if engine is not None else bank_obj
                    break

        if per_game_banks:
            bank_path_str = str(bank_base)
        else:
            print(f"[WARNING] No skill bank found at {bank_base}")

    vllm_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

    gpu_info = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

    print("=" * 78)
    print("  Qwen3-14B Decision Agent with Skill Bank")
    print("=" * 78)
    print(f"  Model:       {args.model}")
    print(f"  vLLM:        {vllm_url}")
    print(f"  GPU:         {gpu_info}")
    print(f"  Skill Bank:  {bank_path_str}")
    print(f"  Games:       {', '.join(available_games)}")
    print(f"  Episodes:    {args.episodes} per game")
    print(f"  Output:      {output_dir}")
    print("=" * 78)

    overall_t0 = time.time()
    game_summaries: List[Dict[str, Any]] = []

    for gi, game_name in enumerate(available_games, 1):
        print(f"\n{'━' * 78}")
        print(f"  GAME {gi}/{len(available_games)}: {game_name} ({args.episodes} episode(s))")
        print(f"{'━' * 78}")

        game_skill_bank = None
        if not args.no_bank:
            game_skill_bank = per_game_banks.get(game_name)
            if game_skill_bank is None:
                first_bank = next(iter(per_game_banks.values()), None) if per_game_banks else None
                game_skill_bank = first_bank
            bank_size = len(game_skill_bank) if game_skill_bank and hasattr(game_skill_bank, "__len__") else "?"
            print(f"  Skill bank: {bank_size} skills loaded")

        game_run_dir = output_dir / game_name / run_timestamp
        summary = run_game_rollouts(game_name, args, game_run_dir, skill_bank=game_skill_bank)
        game_summaries.append(summary)

        print(f"\n  ✓ {game_name} done — "
              f"{summary.get('total_episodes', 0)} ep, "
              f"mean_r={summary.get('mean_reward', 0):.2f}" if "mean_reward" in summary else
              f"\n  ✓ {game_name} done")

    overall_elapsed = time.time() - overall_t0

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "agent_type": "qwen3_14b_decision_with_skillbank",
        "skill_bank": bank_path_str,
        "episodes_per_game": args.episodes,
        "total_elapsed_seconds": overall_elapsed,
        "games": available_games,
        "results": game_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / f"eval_summary_{run_timestamp}.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 78}")
    print("  DECISION AGENT EVALUATION COMPLETE")
    print(f"{'=' * 78}")
    total_eps = sum(
        s.get("total_episodes", 0)
        for s in game_summaries
        if not s.get("invalid")
    )
    print(f"  Games:       {len(available_games)}")
    print(f"  Episodes:    {total_eps}")
    print(f"  Elapsed:     {overall_elapsed:.1f}s")
    print(f"  Output:      {output_dir}")
    print(f"  Summary:     {master_path}")

    successful = [s for s in game_summaries if not s.get("invalid") and "mean_reward" in s]
    if successful:
        avg_reward = sum(s["mean_reward"] for s in successful) / len(successful)
        avg_steps = sum(s["mean_steps"] for s in successful) / len(successful)
        print(f"  Avg reward:  {avg_reward:.2f}")
        print(f"  Avg steps:   {avg_steps:.1f}")

    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
