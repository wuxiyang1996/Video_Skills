# VLM decision-making agent: Observe → Decide → Act → Reward → Repeat.
# Uses tools: take_action, get_state_summary, get_intention, query_skill, query_memory, reward.
# Two-turn micro-loop: (1) take_action → (2) reward  per timestep.
# See .cursor/rules/vlm-decision-agent.mdc and VLM_AGENT_SPEC.md for the full spec.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

try:
    from API_func import ask_model
except ImportError:
    ask_model = None

from data_structure.experience import Experience, Episode

from .agent_helper import (
    get_state_summary as helper_get_state_summary,
    infer_intention as helper_infer_intention,
    skill_bank_to_text,
    query_skill_bank,
    EpisodicMemoryStore,
    HARD_SUMMARY_CHAR_LIMIT,
)
from .reward_func import RewardComputer, RewardConfig, RewardResult

# Reuse game detection and action extraction from dummy_agent
from .dummy_agent import (
    detect_game,
    extract_action,
    _default_action,
    GAME_OVERCOOKED,
    GAME_GAMINGAGENT,
    GAME_VIDEOGAMEBENCH,
    GAME_VIDEOGAMEBENCH_DOS,
    OVERCOOKED_VALID_ACTIONS,
    VIDEOGAMEBENCH_VALID_ACTIONS,
    VIDEOGAMEBENCH_DOS_VALID_KEYS,
)


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Internal state maintained by the VLM decision agent (per spec)."""
    current_intention: str = ""
    progress_notes: List[str] = field(default_factory=list)
    last_actions: List[Any] = field(default_factory=list)
    stuck_counter: int = 0
    steps_since_retrieval: int = 0
    active_skill_plan: Optional[List[Dict[str, Any]]] = None
    active_skill_id: Optional[str] = None
    skill_step_index: int = 0
    steps_without_progress: int = 0
    last_state_summary: str = ""
    last_reward: Optional[RewardResult] = None


# ---------------------------------------------------------------------------
# Tool names and defaults
# ---------------------------------------------------------------------------

TOOL_TAKE_ACTION = "take_action"
TOOL_GET_STATE_SUMMARY = "get_state_summary"
TOOL_GET_INTENTION = "get_intention"
TOOL_QUERY_SKILL = "query_skill"
TOOL_QUERY_MEMORY = "query_memory"
TOOL_REWARD = "reward"

TOOLS = [TOOL_TAKE_ACTION, TOOL_GET_STATE_SUMMARY, TOOL_GET_INTENTION, TOOL_QUERY_SKILL, TOOL_QUERY_MEMORY, TOOL_REWARD]

DEFAULT_RETRIEVAL_BUDGET_N = 10
DEFAULT_SKILL_ABORT_K = 5
MAX_LAST_ACTIONS = 5
MAX_PROGRESS_NOTES = 3


# ---------------------------------------------------------------------------
# VLM Decision Agent
# ---------------------------------------------------------------------------

class VLMDecisionAgent:
    """
    VLM decision-making agent that outputs exactly one tool call per step.
    Runner (e.g. run_episode_vlm_agent) executes the tool and passes result back.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        skill_bank: Any = None,
        memory: Optional[EpisodicMemoryStore] = None,
        retrieval_budget_n: int = DEFAULT_RETRIEVAL_BUDGET_N,
        skill_abort_k: int = DEFAULT_SKILL_ABORT_K,
        game: Optional[str] = None,
        reward_config: Optional[RewardConfig] = None,
        embedder: Any = None,
    ) -> None:
        self.model = model or "gpt-4o-mini"
        self.skill_bank = skill_bank
        if memory is not None:
            self.memory = memory
        else:
            emb = embedder
            if emb is None:
                try:
                    from rag import get_text_embedder
                    emb = get_text_embedder()
                except Exception:
                    emb = None
            self.memory = EpisodicMemoryStore(embedder=emb)
        self.retrieval_budget_n = retrieval_budget_n
        self.skill_abort_k = skill_abort_k
        self.game_hint = game
        self.reward_config = reward_config or RewardConfig()
        self.reward_computer = RewardComputer(self.reward_config)
        self.state = AgentState()

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self.state = AgentState()
        self.reward_computer.reset()

    def _valid_actions_for_game(self, game: str, observation: str = "") -> List[str]:
        """Return list of valid action names for prompt (for take_action)."""
        if game == GAME_OVERCOOKED:
            return list(OVERCOOKED_VALID_ACTIONS)
        if game == GAME_VIDEOGAMEBENCH:
            return list(VIDEOGAMEBENCH_VALID_ACTIONS)
        if game == GAME_VIDEOGAMEBENCH_DOS:
            return list(VIDEOGAMEBENCH_DOS_VALID_KEYS)
        if game == GAME_GAMINGAGENT:
            # Parse from observation "Valid actions: a, b, c"
            m = re.search(r"[Vv]alid\s+actions?\s*[:\-]\s*(.+?)(?:\n|\.|$)", observation or "")
            if m:
                raw = m.group(1).strip()
                return [a.strip() for a in re.split(r"[,;]", raw) if a.strip()]
            return ["no-op", "stay"]
        return ["stay", "no-op"]

    def _build_prompt(
        self,
        observation: str,
        info: Optional[Dict[str, Any]],
        last_tool_name: Optional[str],
        last_tool_result: Optional[str],
    ) -> str:
        game = self.game_hint or info.get("game") or detect_game(observation)
        valid_actions = self._valid_actions_for_game(game, observation)
        skill_text = skill_bank_to_text(self.skill_bank)
        s = self.state

        # Budget: can we call retrieval this step?
        can_retrieve = s.steps_since_retrieval >= self.retrieval_budget_n or s.stuck_counter >= 3
        # Per step: at most one non-action tool unless stuck
        allow_non_action = (last_tool_name == TOOL_TAKE_ACTION) or (s.stuck_counter >= 2)

        prompt = (
            "You are a VLM decision-making agent. Output exactly ONE tool call at a time.\n\n"
            "Tools:\n"
            "- take_action: execute one environment action (primitive or QUERY_MEM/QUERY_SKILL/CALL_SKILL). "
            "Args: {\"action\": \"<valid_action>\"}\n"
            "- reward: compute reward signals for the last transition (call ONCE right after take_action). Args: {}\n"
            "- get_state_summary: get a short text summary of the scene. Args: {}\n"
            "- get_intention: infer/refresh current objective. Args: {}\n"
            "- query_skill: retrieve a procedure from the skill bank. Args: {\"key\": \"<scene, objective, entities, failure_mode>\"}\n"
            "- query_memory: retrieve similar past experiences. Args: {\"key\": \"<scene, objective, entities>\"}\n\n"
            "Two-turn micro-loop per timestep:\n"
            "  1) Choose ONE action → take_action(...)\n"
            "  2) Immediately call reward() once for logging/training.\n"
            "You may call at most ONE non-action tool (get_state_summary/get_intention) BEFORE take_action.\n"
            "Never call query_skill and query_memory in the same timestep.\n"
            "Retrieval-as-action: QUERY_MEM/QUERY_SKILL have negative cost; use only when benefit > cost.\n"
            "CALL_SKILL has small overhead; avoid frequent switching.\n"
            "Retrieval allowed every " + str(self.retrieval_budget_n) + " steps or when stuck.\n"
        )
        if not can_retrieve:
            prompt += "Do NOT use query_skill or query_memory this step (budget).\n"
        if not allow_non_action and last_tool_name != TOOL_TAKE_ACTION:
            prompt += "You must use take_action this step (no extra non-action tools).\n"

        prompt += "\nValid actions for take_action: " + ", ".join(valid_actions) + "\n\n"
        prompt += "Skill bank:\n" + skill_text + "\n\n"

        prompt += "Current observation:\n" + (observation[:3000] if observation else "(none)") + "\n\n"

        prompt += "Your internal state:\n"
        prompt += "- intention: " + (s.current_intention or "(none)") + "\n"
        prompt += "- progress_notes: " + " | ".join(s.progress_notes[-MAX_PROGRESS_NOTES:]) or "(none)" + "\n"
        prompt += "- last_actions: " + ", ".join(str(a) for a in s.last_actions[-MAX_LAST_ACTIONS:]) or "(none)" + "\n"
        prompt += "- stuck_counter: " + str(s.stuck_counter) + "\n"
        prompt += "- active_skill: " + (s.active_skill_id or "(none)") + "\n"
        if s.last_reward:
            prompt += "- last_reward: " + repr(s.last_reward) + "\n"
        if s.last_state_summary:
            prompt += "- last_state_summary: " + s.last_state_summary[:400] + "\n"

        if last_tool_name:
            prompt += "\nLast tool called: " + last_tool_name + "\n"
            if last_tool_result:
                prompt += "Result: " + (str(last_tool_result)[:800]) + "\n"

        prompt += (
            "\nOutput format (strict): Either\n"
            "THOUGHT: <at most 2 sentences>\nTOOL: <tool_name>\nARGS: <json object>\n"
            "or just:\nTOOL: <tool_name>\nARGS: <json object>\n"
            "Example: TOOL: take_action\nARGS: {\"action\": \"north\"}\n"
            "Example: TOOL: reward\nARGS: {}\n"
        )
        return prompt

    def _parse_tool_response(self, reply: str, observation: str, game: str) -> Dict[str, Any]:
        """Parse LLM reply into {tool, args}. Fallback to take_action with default or extracted action."""
        tool = TOOL_TAKE_ACTION
        args: Dict[str, Any] = {}

        # Try TOOL: ... ARGS: ...
        tool_m = re.search(r"TOOL\s*:\s*(\w+)", reply, re.IGNORECASE)
        args_m = re.search(r"ARGS\s*:\s*(\{[\s\S]*?\})(?=\s*(?:TOOL|THOUGHT|$))", reply, re.IGNORECASE)
        if tool_m:
            raw_tool = tool_m.group(1).strip().lower()
            for t in TOOLS:
                if t == raw_tool or raw_tool in t:
                    tool = t
                    break
        if args_m:
            try:
                args = json.loads(args_m.group(1))
            except json.JSONDecodeError:
                pass

        if tool == TOOL_TAKE_ACTION and "action" not in args:
            # Extract action from reply text as fallback
            action = extract_action(reply, game, observation)
            if action is None:
                action = _default_action(game)
            args["action"] = action

        return {"tool": tool, "args": args or {}}

    def step(
        self,
        observation: str,
        info: Optional[Dict[str, Any]] = None,
        last_tool_name: Optional[str] = None,
        last_tool_result: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Decide the next tool call. Returns {"tool": str, "args": dict}.
        Runner executes the tool and calls update_from_tool_result then step again.
        """
        info = info or {}
        game = self.game_hint or info.get("game") or detect_game(observation or "")

        # If we have an active skill plan and not aborting, may output next skill step as take_action
        s = self.state
        if s.active_skill_plan and s.skill_step_index < len(s.active_skill_plan):
            step_spec = s.active_skill_plan[s.skill_step_index]
            action = step_spec.get("action")
            if action is not None:
                s.skill_step_index += 1
                return {"tool": TOOL_TAKE_ACTION, "args": {"action": action}}

        if ask_model is None:
            action = _default_action(game)
            return {"tool": TOOL_TAKE_ACTION, "args": {"action": action}}

        prompt = self._build_prompt(
            observation,
            info,
            last_tool_name,
            str(last_tool_result) if last_tool_result is not None else None,
        )
        reply = ask_model(prompt, model=self.model, temperature=0.3, max_tokens=400)
        out = self._parse_tool_response(reply or "", observation, game)
        return out

    def update_from_tool_result(
        self,
        tool_name: str,
        tool_result: Any,
        observation: Optional[str] = None,
        game: Optional[str] = None,
    ) -> None:
        """
        Update internal state after the runner executed a tool.
        Call this before the next step() so the agent sees the result.
        """
        s = self.state
        s.steps_since_retrieval += 1

        if tool_name == TOOL_GET_STATE_SUMMARY:
            s.last_state_summary = str(tool_result)[:HARD_SUMMARY_CHAR_LIMIT]

        elif tool_name == TOOL_GET_INTENTION:
            s.current_intention = str(tool_result)[:200] if tool_result else ""

        elif tool_name == TOOL_TAKE_ACTION:
            s.last_actions.append(tool_result if isinstance(tool_result, (str, list)) else str(tool_result))
            if len(s.last_actions) > MAX_LAST_ACTIONS:
                s.last_actions = s.last_actions[-MAX_LAST_ACTIONS:]
            s.steps_without_progress += 1
            # Runner can pass progress in result dict; if so, reset stuck_counter / steps_without_progress
            if isinstance(tool_result, dict) and tool_result.get("progress"):
                s.stuck_counter = 0
                s.steps_without_progress = 0
                s.progress_notes.append(tool_result.get("progress", "Progress")[:100])
            if len(s.progress_notes) > MAX_PROGRESS_NOTES:
                s.progress_notes = s.progress_notes[-MAX_PROGRESS_NOTES:]

        elif tool_name == TOOL_QUERY_SKILL:
            s.steps_since_retrieval = 0
            if isinstance(tool_result, dict) and tool_result.get("micro_plan"):
                s.active_skill_plan = tool_result.get("micro_plan")
                s.active_skill_id = tool_result.get("skill_id")
                s.skill_step_index = 0
            elif isinstance(tool_result, list):
                s.active_skill_plan = [{"action": a} if isinstance(a, str) else a for a in tool_result[:7]]
                s.skill_step_index = 0

        elif tool_name == TOOL_QUERY_MEMORY:
            s.steps_since_retrieval = 0

        elif tool_name == TOOL_REWARD:
            if isinstance(tool_result, RewardResult):
                s.last_reward = tool_result

        # Stuck detection: if we only took actions and no progress, increment stuck_counter
        if tool_name == TOOL_TAKE_ACTION and s.steps_without_progress >= self.skill_abort_k:
            s.stuck_counter += 1
            s.active_skill_plan = None
            s.active_skill_id = None
            s.skill_step_index = 0


# ---------------------------------------------------------------------------
# Tool execution (for use by runner)
# ---------------------------------------------------------------------------

def run_tool(
    tool_name: str,
    args: Dict[str, Any],
    agent: VLMDecisionAgent,
    observation: str,
    info: Optional[Dict[str, Any]] = None,
    env: Any = None,
) -> Any:
    """
    Execute one tool and return the result. Used by the episode runner.
    - take_action: env.step(args["action"]) and return (next_obs, reward, term, trunc, info) or action.
    - get_state_summary: return get_state_summary(observation, ...).
    - get_intention: return infer_intention(observation or agent.state.last_state_summary, ...).
    - query_skill: look up skill_bank by args.get("key") or skill_id, return contract or micro-plan.
    - query_memory: return memory.query(args.get("key", ""), k=3).
    """
    info = info or {}
    game = agent.game_hint or info.get("game") or detect_game(observation or "")

    if tool_name == TOOL_TAKE_ACTION:
        action = args.get("action")
        if env is not None and action is not None:
            result = env.step(action)
            return result  # (obs, reward, term, trunc, info) or multi-agent variant
        return action

    if tool_name == TOOL_GET_STATE_SUMMARY:
        structured = info.get("structured_state") if info else None
        return helper_get_state_summary(
            observation,
            structured_state=structured,
            game=game,
            model=agent.model,
        )

    if tool_name == TOOL_GET_INTENTION:
        summary = agent.state.last_state_summary or observation
        return helper_infer_intention(
            summary,
            game=game,
            model=agent.model,
            context={
                "last_actions": agent.state.last_actions,
                "progress_notes": agent.state.progress_notes,
                "task": info.get("task", ""),
            },
        )

    if tool_name == TOOL_QUERY_SKILL:
        key = args.get("key", "")
        return query_skill_bank(agent.skill_bank, key, top_k=1)

    if tool_name == TOOL_QUERY_MEMORY:
        key = args.get("key", "")
        return agent.memory.query(key, k=3)

    if tool_name == TOOL_REWARD:
        r_env = args.get("r_env", 0.0)
        action_type = args.get("action_type", "primitive")
        skill_id = agent.state.active_skill_id
        contract = None
        if skill_id and agent.skill_bank:
            try:
                contract = agent.skill_bank.get_contract(skill_id)
            except Exception:
                pass
        return agent.reward_computer.compute_reward(
            r_env=r_env,
            action_type=action_type,
            observation=observation,
            active_skill_id=skill_id,
            skill_contract=contract,
        )

    return None


# ---------------------------------------------------------------------------
# Episode runner: run one episode with the VLM agent
# ---------------------------------------------------------------------------

def run_episode_vlm_agent(
    env: Any,
    agent: Optional[VLMDecisionAgent] = None,
    *,
    model: Optional[str] = None,
    skill_bank: Any = None,
    memory: Optional[EpisodicMemoryStore] = None,
    reward_config: Optional[RewardConfig] = None,
    task: str = "",
    max_steps: int = 1000,
    verbose: bool = False,
) -> Episode:
    """
    Run one episode with the two-turn micro-loop per timestep:
      1) (optional) get_state_summary / get_intention
      2) take_action  (exactly one env action)
      3) reward       (compute r_env, r_follow, r_cost, r_total)

    Returns an Episode (data_structure format) with:
      - Experience objects per step, fully populated (state, action, reward,
        next_state, done, summary_state, intentions, sub_tasks, reward_details).
      - Episode.metadata with per-step reward_details, cumulative_reward,
        final agent_state, done flag, and step count — for callers that need
        the flat rollout information.
    """
    if agent is None:
        agent = VLMDecisionAgent(
            model=model, skill_bank=skill_bank, memory=memory,
            reward_config=reward_config,
        )
    agent.reset()

    obs, info = env.reset()
    is_multi = isinstance(obs, dict)
    if is_multi:
        obs = list(obs.values())[0] if obs else ""
    observation = str(obs) if obs else ""
    info = dict(info or {})
    info.setdefault("game", detect_game(observation))

    episode_task = task or info.get("task", "")

    last_tool_name: Optional[str] = None
    last_tool_result: Any = None
    step_count = 0
    done = False
    experiences: List[Experience] = []
    all_observations: List[str] = [observation]
    all_reward_details: List[Dict[str, float]] = []

    while step_count < max_steps:
        # ── Phase A: optional non-action tool (get_state_summary / get_intention) ──
        out = agent.step(observation, info, last_tool_name, last_tool_result)
        tool_name = out.get("tool", TOOL_TAKE_ACTION)
        tool_args = out.get("args") or {}

        if tool_name not in (TOOL_TAKE_ACTION, TOOL_REWARD):
            result = run_tool(tool_name, tool_args, agent, observation, info, env=None)
            agent.update_from_tool_result(tool_name, result, observation, info.get("game"))
            last_tool_name = tool_name
            last_tool_result = result
            if tool_name == TOOL_GET_STATE_SUMMARY:
                agent.state.last_state_summary = str(result)[:HARD_SUMMARY_CHAR_LIMIT]
            # Re-query agent for the actual action after the non-action tool.
            out = agent.step(observation, info, last_tool_name, last_tool_result)
            tool_name = out.get("tool", TOOL_TAKE_ACTION)
            tool_args = out.get("args") or {}
            if tool_name not in (TOOL_TAKE_ACTION,):
                tool_name = TOOL_TAKE_ACTION
                tool_args = {"action": _default_action(info.get("game") or detect_game(observation))}

        # Snapshot agent state BEFORE the action for this experience
        pre_action_summary = agent.state.last_state_summary or ""
        pre_action_intention = agent.state.current_intention or ""
        pre_action_skill = agent.state.active_skill_id

        # ── Phase B: take_action ──
        action = tool_args.get("action")
        if action is None:
            action = _default_action(info.get("game") or detect_game(observation))

        # Classify action type for reward cost computation.
        action_type = "primitive"
        action_str = str(action).upper() if action else ""
        if "QUERY_MEM" in action_str:
            action_type = "QUERY_MEM"
        elif "QUERY_SKILL" in action_str:
            action_type = "QUERY_SKILL"
        elif "CALL_SKILL" in action_str:
            action_type = "CALL_SKILL"

        if is_multi:
            active = info.get("active_players", [])
            actions = {pid: action for pid in active} if active else {0: action}
        else:
            actions = action

        next_obs, reward, terminated, truncated, next_info = env.step(actions)
        done = terminated or truncated

        progress = None
        if reward and float(reward) != 0:
            progress = "reward " + str(reward)
        agent.update_from_tool_result(
            TOOL_TAKE_ACTION,
            {"action": action, "progress": progress} if progress else action,
            observation,
            info.get("game"),
        )

        # ── Phase C: reward (two-turn micro-loop, step 2) ──
        env_reward = float(reward) if not isinstance(reward, dict) else sum(reward.values())
        rr = run_tool(
            TOOL_REWARD,
            {"r_env": env_reward, "action_type": action_type},
            agent,
            observation,
            info,
            env=None,
        )
        agent.update_from_tool_result(TOOL_REWARD, rr, observation, info.get("game"))
        rr_dict = rr.to_dict() if isinstance(rr, RewardResult) else {"r_env": env_reward}
        all_reward_details.append(rr_dict)

        # Resolve next observation
        if is_multi:
            next_observation = str(next_obs.get(list(next_obs.keys())[0], "")) if next_obs else ""
        else:
            next_observation = str(next_obs) if next_obs else ""

        # ── Build Experience with all fields populated ──
        exp = Experience(
            state=observation,
            action=str(action),
            reward=env_reward,
            next_state=next_observation,
            done=done,
            intentions=pre_action_intention if pre_action_intention else None,
            tasks=episode_task if episode_task else None,
            sub_tasks=pre_action_skill,
        )
        exp.idx = step_count
        exp.summary_state = pre_action_summary if pre_action_summary else None
        exp.reward_details = rr_dict
        experiences.append(exp)

        if verbose:
            print(f"  step {step_count}: action={action}  {rr}")

        last_tool_name = TOOL_REWARD
        last_tool_result = rr
        observation = next_observation
        info = dict(next_info or {})
        info.setdefault("game", detect_game(observation))
        all_observations.append(observation)
        step_count += 1

        if done:
            break

    cumulative = agent.reward_computer.cumulative

    episode = Episode(
        experiences=experiences,
        task=episode_task or "Unspecified task",
        metadata={
            "observations": all_observations,
            "actions": [exp.action for exp in experiences],
            "rewards": [exp.reward for exp in experiences],
            "reward_details": all_reward_details,
            "done": done,
            "steps": step_count,
            "agent_state": {
                "current_intention": agent.state.current_intention,
                "progress_notes": agent.state.progress_notes,
                "last_actions": agent.state.last_actions,
                "active_skill_id": agent.state.active_skill_id,
            },
            "cumulative_reward": cumulative.to_dict(),
        },
    )
    episode.set_outcome()
    return episode
