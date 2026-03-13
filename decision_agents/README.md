# decision_agents

VLM decision-making agent that plays video games step-by-step using a **two-turn micro-loop**: `take_action` → `reward` per timestep. Supports **skill-bank retrieval** (protocol-driven plans), **episodic memory**, **intention inference**, and **composite reward shaping**.

**Model-agnostic design:** The same code path is used for GPT, Qwen, and other backends. Pass `model="gpt-4o-mini"`, `model="Qwen/Qwen3-14B"`, or any supported name; `API_func.ask_model` routes to the correct API. Skill bank loading and querying (`load_skill_bank`, `select_skill_from_bank`, `skill_bank_to_text`) are identical for all models — both `scripts/run_inference.py` and `scripts/run_qwen3_14b_eval.py` use these same functions, with only the LLM backend differing. The default model when none is provided is `VLMDecisionAgent.DEFAULT_MODEL` (and `DEFAULT_LLM_MODEL` in helpers); callers (inference and training) should pass an explicit `model` for the intended backend.

## Files

| File | What it does |
|------|-------------|
| `agent.py` | `VLMDecisionAgent`, `run_tool()`, `run_episode_vlm_agent()`, tool handlers (e.g. `TOOL_QUERY_SKILL` → `active_skill_plan` from protocol steps) |
| `agent_helper.py` | `get_state_summary()`, `build_rag_summary()`, `extract_game_facts()`, `infer_intention()`, `EpisodicMemoryStore`, `skill_bank_to_text()`, `query_skill_bank()` / `select_skill_from_bank()`, `_get_protocol_for_skill()` |
| `reward_func.py` | `RewardConfig`, `RewardResult`, `RewardComputer`, `compute_reward()` (r_follow uses skill contract `eff_add`) |
| `dummy_agent.py` | Baseline `language_agent_action()` + game detection + action extraction |
| `__init__.py` | Re-exports the above |

---

## Quick start — run a full episode

`run_episode_vlm_agent()` returns an **`Episode`** object (from `data_structure.experience`) with fully-populated `Experience` objects per step.

```python
from decision_agents import VLMDecisionAgent, run_episode_vlm_agent, RewardConfig

# Wrap your env (must have reset() → (obs, info) and step(action) → (obs, r, term, trunc, info))
episode = run_episode_vlm_agent(
    env,
    model="gpt-4o-mini",       # or "Qwen/Qwen3-14B", "claude-...", "gemini-..."
    task="Complete level 1",
    max_steps=200,
    verbose=True,
)

# Episode structure
print(episode.get_length())                      # number of env steps taken
print([e.reward for e in episode.experiences])    # list of r_total per step
print([e.reward_details for e in episode.experiences])  # {r_env, r_follow, r_cost, r_total}
print(episode.metadata["cumulative_reward"])      # episode totals
print(episode.experiences[-1].done)               # whether episode terminated

# Each Experience has rich fields populated from agent state:
exp = episode.experiences[0]
print(exp.summary_state)   # key=value format, e.g. "game=tetris | phase=midgame | stack_h=14 | holes=32"
print(exp.intentions)      # [TAG] phrase, e.g. "[CLEAR] Reduce holes before stack overflows"
print(exp.sub_tasks)       # active skill ID being followed
print(exp.reward_details)  # full reward breakdown dict

# Serialize/deserialize
d = episode.to_dict()
from data_structure.experience import Episode
restored = Episode.from_dict(d)
```

### With a skill bank and custom reward config

The skill bank exposes a **protocol store** view to the decision agent: for each skill it can see `name`, `strategic_description`, `protocol` (steps, preconditions, success_criteria, expected_duration), and `confidence`. When the agent calls **query_skill**, the best-matching skill’s **protocol.steps** are used to fill `active_skill_plan`; reward shaping still uses the skill’s **contract** (e.g. `eff_add`) via `bank.get_contract(skill_id)`.

```python
from decision_agents import (
    VLMDecisionAgent,
    run_episode_vlm_agent,
    EpisodicMemoryStore,
    RewardConfig,
)
from skill_agents.skill_bank.bank import SkillBankMVP

bank = SkillBankMVP("path/to/bank.jsonl")
bank.load()

# RAG-powered episodic memory (auto-loads Qwen3-Embedding-0.6B)
from rag import get_text_embedder
memory = EpisodicMemoryStore(max_entries=500, embedder=get_text_embedder())
# Or let VLMDecisionAgent auto-load the embedder:
#   memory = None  # agent will create EpisodicMemoryStore with RAG embedder

reward_cfg = RewardConfig(
    w_follow=0.1,             # weight on skill-following shaping
    query_mem_cost=-0.05,     # cost per QUERY_MEM
    query_skill_cost=-0.05,   # cost per QUERY_SKILL
    call_skill_cost=-0.02,    # cost per CALL_SKILL
    skill_switch_cost=-0.10,  # penalty for changing active skill
)

agent = VLMDecisionAgent(
    model="gpt-4o-mini",       # or "Qwen/Qwen3-14B" for vLLM backend
    skill_bank=bank,
    memory=memory,
    reward_config=reward_cfg,
    retrieval_budget_n=10,    # allow retrieval every 10 steps
    skill_abort_k=5,          # abort skill after 5 steps without progress
)

episode = run_episode_vlm_agent(env, agent=agent, task="Deliver 3 soups", max_steps=500, verbose=True)

# Episode is ready for skill pipeline ingestion — no conversion needed:
from skill_agents.pipeline import SkillBankAgent
skill_agent = SkillBankAgent(bank_path="skills/bank.jsonl")
skill_agent.ingest_episodes([episode])
```

---

## Step-by-step control (manual loop)

If you need finer control, drive the agent one step at a time:

```python
from decision_agents import VLMDecisionAgent

agent = VLMDecisionAgent(model="gpt-4o-mini")  # or any supported model name
obs, info = env.reset()

last_tool_name = None
last_tool_result = None

for t in range(200):
    # Agent decides which tool to call
    decision = agent.step(str(obs), info, last_tool_name, last_tool_result)
    # decision = {"tool": "take_action", "args": {"action": "north"}}

    tool = decision["tool"]
    args = decision["args"]

    if tool == "take_action":
        obs, reward, term, trunc, info = env.step(args["action"])
        agent.update_from_tool_result("take_action", args["action"], str(obs))
        if term or trunc:
            break
    elif tool == "reward":
        # Normally auto-called by run_episode_vlm_agent;
        # in manual mode, call the reward computer directly:
        rr = agent.reward_computer.compute_reward(r_env=reward, action_type="primitive", observation=str(obs))
        agent.update_from_tool_result("reward", rr, str(obs))
    else:
        from decision_agents import run_tool
        result = run_tool(tool, args, agent, str(obs), info)
        agent.update_from_tool_result(tool, result, str(obs))

    last_tool_name = tool
    last_tool_result = decision.get("result")
```

---

## Skill bank: protocol store vs contract

The skill bank stores each skill as a **Skill** object with two logical parts (see `skill_agents.stage3_mvp.schemas`):

- **Protocol store** — What the decision agent sees: `name`, `strategic_description`, `tags`, `protocol` (steps, preconditions, success_criteria, abort_criteria, expected_duration), `confidence`. Used by `skill_bank_to_text()`, `query_skill_bank()`, and to set `active_skill_plan` from `protocol.steps`.
- **Contract** — Effects (`eff_add`, `eff_del`, `eff_event`) used for segmentation, verification, and **reward shaping**. The agent still gets the contract via `bank.get_contract(skill_id)` when computing r_follow.

So: the agent **plans** from protocols (when present) and is **rewarded** for making progress on the contract’s eff_add predicates.

---

## Helper functions

### `get_state_summary(observation, structured_state=None, *, max_chars=400, use_llm_fallback=False, llm_callable=None)`

Produces a compact `key=value` state summary optimised for LLM/VLM context
windows, retrieval, skill-bank indexing, and trajectory segmentation.
Summaries are **never** raw observation text and always ≤ 400 characters.

**Priority order:**
1. `structured_state` → `compact_structured_state()` (preferred; wrapper-produced dict)
2. `observation` → `compact_text_observation()` (deterministic boilerplate removal + clause compression)
3. LLM fallback (optional, disabled by default)

**Length policy:** All summaries obey a unified **400-character hard cap**
(`HARD_SUMMARY_CHAR_LIMIT`).  Prefer ~220–380 chars when possible.

```python
from decision_agents import get_state_summary

# With structured state from a wrapper (preferred)
summary = get_state_summary(
    obs_text,
    structured_state=info.get("structured_state"),
)
# → "game=overcooked | self=hold:onion pos:1,2 | ally=hold:dish | critical=pot:2ing:cooking | orders=onion_soup | time_left=47"

# Text-only fallback (boilerplate stripped, clauses compressed)
summary = get_state_summary(long_observation_text)
```

**Supported wrappers with `build_structured_state_summary()`:**

| Wrapper | Key fields | Example |
|---------|-----------|---------|
| Overcooked | game, self, ally, critical, orders, time_left | `game=overcooked \| self=hold:onion pos:1,2 \| ally=hold:dish \| critical=pot:2ing:cooking` |
| Avalon | game, phase, self, progress, critical, objective | `game=avalon \| phase=team_vote \| self=role:Percival(G) \| progress=quest:1/5 good:1 evil:0` |
| Diplomacy | game, phase, self, resources, critical, objective | `game=diplomacy \| phase=S1902M \| self=power:FRANCE centers:5 \| resources=units:A PAR,F BRE` |
| GamingAgent | game, step, self, objective, critical, affordance | `game=sokoban \| self=Player at (2,3) \| objective=push box \| critical=Box at (3,3)` |
| VideoGameBench | game, step, last_action, progress, critical, objective | `game=kirby \| objective=break_stall \| critical=repeat:RIGHTx8 \| progress=stall:8` |

Each wrapper sets `info["structured_state"]` on `reset()` and `step()`, which
the `run_tool(TOOL_GET_STATE_SUMMARY, ...)` call site automatically consumes.

**Helper functions** (all in `agent_helper.py`):

| Function | Purpose |
|----------|---------|
| `compact_structured_state(dict, max_chars)` | Dict → `key=value \| ...` string, priority-ordered |
| `compact_text_observation(obs, max_chars)` | Raw text → boilerplate-stripped, clause-compressed string |
| `_safe_str(x)` | Coerce any value to a short summary-safe string |
| `_remove_boilerplate(obs)` | Strip action-format instructions from raw observation |
| `_truncate_keep_important(text, max_chars)` | Truncate at clause boundary |
| `_join_kv(parts, max_chars)` | Join `(key, value)` pairs with budget control |

**Constants:**

| Constant | Value | Meaning |
|----------|-------|---------|
| `DEFAULT_LLM_MODEL` | `"gpt-4o-mini"` | Fallback when no `model` is passed to `get_state_summary` / `infer_intention`; callers should pass an explicit model for the intended backend. |
| `DEFAULT_SUMMARY_CHAR_BUDGET` | 400 | Default budget for all summaries |
| `HARD_SUMMARY_CHAR_LIMIT` | 400 | Absolute upper bound; never exceeded |

### `build_rag_summary(state, game_name, *, step_idx, total_steps, reward, max_chars)`

Fully deterministic (no LLM) `key=value` summary optimised for RAG embedding
retrieval. Combines game-aware fact extraction with phase estimation and reward.

```python
from decision_agents.agent_helper import build_rag_summary

summary = build_rag_summary(
    state_text,
    game_name="tetris",
    step_idx=50,
    total_steps=86,
    reward=1.0,
)
# → "game=tetris | phase=midgame | step=50/86 | stack_h=14 | holes=32 | next=T,Z,I,J | level=1 | reward=+1"
```

Uses `extract_game_facts()` internally — game-specific parsers for Tetris
(stack_h, holes, piece, next), 2048 (highest, empty, tiles, merges), Candy
Crush (score, moves, pairs), Sokoban (boxes, goals, worker), Super Mario
(x_pos, y_pos, coins, time), Avalon (phase, role, quest), and Diplomacy
(phase, power, centers, units).

### `infer_intention(summary_or_observation, game=None, model=None, context=None)`

Returns a `[TAG] subgoal phrase` (≤15 words) describing the agent's current
subgoal, using the same format as the labeling pipeline. The tag is one of 13
canonical categories used for skill segmentation and RAG retrieval:

```
SETUP | CLEAR | MERGE | ATTACK | DEFEND | NAVIGATE | POSITION |
COLLECT | BUILD | SURVIVE | OPTIMIZE | EXPLORE | EXECUTE
```

```python
from decision_agents import infer_intention

intention = infer_intention(
    summary,
    context={
        "last_actions": ["north", "interact"],
        "progress_notes": ["picked up onion"],
        "task": "deliver 3 soups",
    },
)
# e.g. "[NAVIGATE] Deliver onion to pot and start cooking"

# With game-specific context:
intention = infer_intention(
    "game=tetris | stack_h=15 | holes=42",
    game="gamingagent",
    model="gpt-5.4",
)
# e.g. "[SURVIVE] Place Z vertically to avoid overhangs and prevent topping out"
```

### `EpisodicMemoryStore`

RAG-embedding retrieval memory for the `query_memory` tool.  When an
embedder is supplied (or auto-loaded from `rag/`), memories are embedded
on `add` and queries use cosine similarity blended with keyword overlap.

Keyword scoring splits on `=|:,;/` in addition to whitespace, so
`key=value` summaries tokenise correctly (e.g. `game=tetris` matches
queries for both "game" and "tetris").

```python
from decision_agents import EpisodicMemoryStore
from rag import get_text_embedder

mem = EpisodicMemoryStore(
    max_entries=500,
    embedder=get_text_embedder(),   # RAG embedding (Qwen3-Embedding-0.6B)
    embedding_weight=0.7,           # 70% embedding, 30% keyword overlap
)

# Add entries with key=value summaries (automatically embedded)
mem.add_experience(
    state_summary="game=fps | hp=20 | pos=corridor | threat=sniper_balcony",
    action="take cover behind pillar",
    next_state_summary="game=fps | hp=20 | pos=behind_pillar | threat=sniper_balcony",
    done=False,
)

# Query — uses cosine similarity + keyword overlap
results = mem.query("game=fps | hp=low | threat=sniper", k=3)
# returns list of dicts: [{key, summary, action, outcome, ...}, ...]
```

### `skill_bank_to_text(skill_bank)` and `query_skill_bank(skill_bank, state, task, ...)`

**`skill_bank_to_text(skill_bank)`** — Formats the skill bank for agent prompts. When a skill has a **protocol** (name, strategic_description, steps), the string shows those; otherwise it falls back to effect counts (eff_add / eff_del). Output is tuned for the decision agent (no internal evidence or sub-episode pointers).

```python
from decision_agents import skill_bank_to_text

text = skill_bank_to_text(bank)
# "Available skill IDs: nav_corridor, merge_corner, ...\n  - merge_corner: Keep highest tile in corner (conf=0.85, ~12 steps)\n    when: ...; done: ...\n  ..."
```

**`query_skill_bank(skill_bank, state, task, ...)`** — Picks the best-matching skill for the current state/task and returns it with a **protocol** dict (steps, preconditions, success_criteria, expected_duration) when available. The agent uses `protocol["steps"]` as `active_skill_plan`; reward shaping still uses `bank.get_contract(skill_id)` for r_follow. Same entry point as `select_skill_from_bank`.

---

## Reward function

### Standalone usage

```python
from decision_agents import RewardComputer, RewardConfig

cfg = RewardConfig(w_follow=0.1, skill_switch_cost=-0.10)
rc = RewardComputer(cfg)

# After each env step:
rr = rc.compute_reward(
    r_env=1.0,                    # from env.step
    action_type="primitive",      # or "QUERY_MEM", "QUERY_SKILL", "CALL_SKILL"
    observation="checkpoint area",
    active_skill_id="nav_to_cp",  # or None
    skill_contract=contract,      # object with .eff_add set, or None
)
print(rr)
# RewardResult(r_env=1.0000, r_follow=0.0500, r_cost=0.0000, r_total=1.0050)

# Episode totals:
print(rc.cumulative)
# Per-step history:
print(rc.history)
```

### RewardConfig defaults

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `w_follow` | 0.1 | Weight on r_follow in r_total |
| `query_mem_cost` | -0.05 | Cost per QUERY_MEM action |
| `query_skill_cost` | -0.05 | Cost per QUERY_SKILL action |
| `call_skill_cost` | -0.02 | Cost per CALL_SKILL action |
| `skill_switch_cost` | -0.10 | Penalty when active skill changes |
| `follow_predicate_bonus` | 0.05 | Bonus per newly satisfied eff_add predicate |
| `follow_completion_bonus` | 0.20 | Bonus when all eff_add predicates satisfied |
| `follow_no_progress_penalty` | -0.01 | Penalty per step with no predicate progress |

### Reward components

- **r_env**: Raw environment reward passed through.
- **r_follow**: Skill-following shaping (termination-free). Checks how many `eff_add` predicates from the active skill's contract appear in the observation. Awards bonuses for newly satisfied predicates and a completion bonus when all are met.
- **r_cost**: Negative costs for queries, skill calls, and skill switching.
- **r_total**: `r_env + w_follow * r_follow + r_cost`.

### How reward is computed (detail)

- **r_follow (skill-following)**  
  The active skill’s contract has an **eff_add** set (e.g. `at_pot`, `onion_in_pot`). Each predicate is treated as satisfied if **all** of its tokens (after tokenizing, length ≥ 2) appear in the **current observation** text (case-insensitive). The computer is **stateful**: it tracks which predicates were already satisfied in previous steps. It only gives the **per-predicate bonus** for **newly** satisfied predicates; when the set of satisfied predicates equals the full **eff_add**, it adds the **completion bonus**. If no new predicate is satisfied on a step, it applies a **no-progress penalty**.

- **r_cost**  
  **query_mem_cost** (default -0.05) when the action is QUERY_MEM; **query_skill_cost** (-0.05) for QUERY_SKILL; **call_skill_cost** (-0.02) for CALL_SKILL; **skill_switch_cost** (-0.10) when the active skill id changes from the previous step.

- **Combined**  
  **r_total = r_env + w_follow × r_follow + r_cost**. Used for the two-turn micro-loop (take_action → reward) and for logging/training.

---

## Dummy agent (baseline)

The original single-call LLM agent, for comparison or simple use:

```python
from decision_agents import language_agent_action

action = language_agent_action(
    state_nl=observation_text,
    game="overcooked",          # or None for auto-detect
    model="gpt-4o-mini",        # or "Qwen/Qwen3-14B", etc.
)
# Returns: "north", "interact", etc.
```

---

## Per-step loop (protocol)

Every timestep the runner executes:

1. **`get_state_summary`** — required; runner computes it before action (returns `key=value` facts).
2. **(Optional)** At most one of **`query_skill`** or **`query_memory`** — better to have when retrieval is allowed (budget-limited).
3. **`take_action`** — required; exactly one environment action. Agent has intention (from previous step), fresh state summary, and any retrieved skill/memory in context.
4. **`get_intention`** — required; runner updates intention after observing action result (returns `[TAG] subgoal phrase`).
5. **`reward`** — required; compute `(r_env, r_follow, r_cost, r_total)` for logging/training.

Only **`query_skill`** and **`query_memory`** are optional (but better to have). The others are necessary each env step. Retrieval is budget-limited to once every N steps (default 10) unless the agent is stuck. Never call both retrieval tools in the same timestep.

### Format consistency

The agent prompt explicitly labels both formats so Qwen agents learn them:
- **Intention**: `"intention ([TAG] subgoal): [CLEAR] Reduce holes before stack overflows"`
- **State summary**: `"state_summary (key=value): game=tetris | phase=endgame | stack_h=15 | holes=42"`
- **Memory results**: returned as key=value summaries from `EpisodicMemoryStore`

This matches the labeling pipeline in `labeling/label_episodes_gpt54.py`, ensuring
cold-start data and runtime data share identical formats for RAG retrieval and
skill extraction.
