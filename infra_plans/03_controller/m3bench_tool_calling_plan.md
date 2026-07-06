# M3-Bench Multi-Turn Tool-Calling Plan

> **Sub-folder:** `Video_Skills/infra_plans/03_controller/`
>
> **Goal:** Specify how M3-Bench long-video QA is materialized as a **multi-turn tool-calling reasoning episode** that runs on the Video_Skills controller + harness, *without* introducing a parallel architecture. This document concretizes the controller's `[Think] → [Search] → [Think] → [Answer]` loop for M3-Bench by (a) enumerating the M3-Bench-specific tool surface, (b) mapping each tool to an existing Memory Procedure / frozen tool / pure-reasoning function, (c) mapping the four M3-Bench question types to expected tool chains, and (d) listing the supervision signals M3-Bench provides for trajectory-level SFT/GRPO.
>
> **Related plans:**
>
> - [Actors / Reasoning Model](actors_reasoning_model.md) — controller loop, retriever §2B, verifier §2C, abstention §2D, training signals §2E
> - [Atomic skills & hop refactor](../04_harness/atomic_skills_hop_refactor_execution_plan.md) — harness step contract and trace format
> - [Video Benchmarks & Grounding §5.2 / §6](../01_grounding/video_benchmarks_grounding.md) — M3-Bench adapter and benchmark-to-capability matrix
> - [Agentic Memory](../02_memory/agentic_memory_design.md) — episodic / semantic / state stores + evidence layer
> - [Runtime Contracts](../00_overview/runtime_contracts.md) — `GroundedWindow`, `HopGoal`, `EvidenceBundle`, `AtomicStepResult`, `ReasoningTrace`
> - [MCP terminology map](../04_harness/mcp_harness_terminology_map.md) — relation to external "tool" vocabulary
>
> **Non-goals.** This document does **not** propose a new orchestration layer, a new memory store, or a new wire protocol. It does not change `runtime_contracts.md`. It is a benchmark-specific concretization of the controller's existing tool surface plus a function-calling adapter.

---

## 0. Key insight — M3-Bench is already a multi-turn tool-calling task

[m3-agent's `control.py`](https://github.com/bytedance/m3-agent/blob/main/m3_agent/control.py) implements M3-Bench QA as a ReAct loop with **exactly one tool** (`search` over a `videograph` memory bank) and a textual `Action: [Search] | [Answer]\nContent: …` action format. Concretely:

- The controller LLM emits `[Search] <query>` or `[Answer] <text>` each round.
- A consumer matches the regex `Action: \[(.*)\].*Content: (.*)` and, on `[Search]`, calls `mmagent.retrieve.search(mem_node, query, …)` which returns `{CLIP_k: [memory snippets]}`.
- `before_clip` is honored via `mem_node.truncate_memory_by_clip(before_clip, False)` so the agent cannot read future memory.
- Termination is `[Answer]` or a hard cap (`processing_config["total_round"]`), with a final forced-answer round.

So the question is **not** "can M3-Bench be modeled as multi-turn tool calling?" — it already is. The question is: **what is the right tool surface for M3-Bench under the Video_Skills controller**, and how does it map to the modules this repo already exposes? This document answers both.

---

## 1. Mapping M3-Bench to the Video_Skills role split

[Actors / Reasoning Model §0.1](actors_reasoning_model.md) fixes three roles: **72B frozen specialists**, **8B controller (only trainable)**, **harness (deterministic)**. For M3-Bench:

| Role | M3-Bench responsibility |
|------|-------------------------|
| **8B controller** | Decomposes the question into hops; per hop emits a typed **tool call** (function-calling protocol, see §3); reads tool observations from the harness; decides answer vs continue vs abstain. The controller never touches `pkl` files, never reads raw frames, never sees character-ID tables directly — it only sees what the harness returns. |
| **Harness** | Dispatches the tool call to one of: a Memory Procedure ([`02_memory/`](../02_memory/)), the `SocialVideoGraph` retriever ([`01_grounding/` §2.5 + 2.7](../01_grounding/video_benchmarks_grounding.md)), or a frozen 72B/32B visual specialist ([`01_grounding/`](../01_grounding/)). Enforces `before_clip` truncation. Emits one `AtomicStepResult` per call. |
| **Frozen 72B/32B** | Only called via two tools: `view_clip(clip_id, ...)` and `transcribe_clip(clip_id, ...)` (see §2). Never orchestrates. |
| **M3-Bench adapter** | Per [`01_grounding/ §5.2`](../01_grounding/video_benchmarks_grounding.md#52-retrieval-mode-adapters), calls `back_translate` on the question and `translate` on the final answer. This wrap happens **outside** the controller; the controller works in character-ID space. |

This is a strict refinement of m3-agent's loop:

- The single `search(query)` action is split into a typed tool set (§2).
- The text `Action: [Search]` / `Action: [Answer]` protocol is replaced by **OpenAI / Qwen-style function calling** (§3).
- The opaque `mem_node.search` result is replaced by **`AtomicStepResult`** with evidence refs that flow into the same `ReasoningTrace` everything else uses ([`runtime_contracts.md §2.7`](../00_overview/runtime_contracts.md)).

---

## 2. M3-Bench tool surface

Eight tools cover all 1276 M3-Bench-robot questions and all 920 M3-Bench-web questions. Each tool is a thin adapter over an existing Memory Procedure, retriever method, or frozen 72B call — **no new state is introduced**. Schemas are given as JSON-schema-like Python signatures; the exact field set lives in `video_skills/contracts.py` (Phase-1 freeze line).

### 2.1 Retrieval tools (back the controller's `[Search]` action)

| Tool | Signature | Implementation | When the controller uses it |
|------|-----------|----------------|------------------------------|
| `search_memory` | `(query: str, topk: int = 5, before_clip: int \| None = None, episodic_only: bool = False) -> list[ClipMemoryHit]` | `mmagent.retrieve.search(mem_node, query, …)` (clip-wise mode) | First-pass evidence gathering; matches `prompt_generate_action_with_plan` in m3-agent. |
| `search_memory_node` | `(query: str, topk: int = 20, before_clip: int \| None = None) -> list[MemoryNodeHit]` | Same `search(…, mem_wise=True)` | Node-wise (finer-grained) retrieval used for **character-ID lookups** and high-precision facts. m3-agent's existing convention is to trigger this when the query mentions `"character id"`; we lift that to an explicit tool. |
| `resolve_character` | `(name_or_id: str) -> CharacterResolution` | Wraps `videograph.character_mappings` / `reverse_character_mappings` + the M3-Bench `translate` / `back_translate` round-trip from [`01_grounding/ §2.7`](../01_grounding/video_benchmarks_grounding.md#27-entity-resolution--re-identification) | Whenever the question or a retrieved snippet mentions a **name** (which must be back-translated to `character_N` before further retrieval) or vice versa. Removes the hardcoded "use 'What is the character id of {name}'" hint in m3-agent's system prompt. |
| `list_clips_with_entity` | `(entity_ref: str, before_clip: int \| None = None) -> list[int]` | `videograph.text_nodes_by_clip` filtered by `entity_ref` (uses `get_related_nodes` from `mmagent.retrieve`) | Entity-centric temporal scan for **Person Understanding** questions ("what does Lily do across the day?"). |
| `get_clip_memory` | `(clip_id: int, episodic_only: bool = True) -> ClipMemoryDump` | Translated dump of `videograph.text_nodes_by_clip[clip_id]` (same translation pass as `mmagent.retrieve.translate`) | White-box fallback when embedding-based retrieval is missing a known-relevant clip (typical for **Multi-Detail Reasoning** that needs all events in a small interval). |

### 2.2 Perception tools (back the controller's optional `[Ground]` action)

These are the only paths to frozen 72B/32B visual specialists. They are **not** invoked by default; the controller calls them when retrieval returns evidence with `confidence < threshold` or when the retrieved snippet mentions a visual attribute the memory graph does not record (color, exact pose, on-screen text).

| Tool | Signature | Implementation | When the controller uses it |
|------|-----------|----------------|------------------------------|
| `view_clip` | `(clip_id: int, question: str, target_entities: list[str] = []) -> GroundedWindow` | Calls Observer-A (72B grounding model) on the clip's frames + subtitle span, returns one `GroundedWindow` per [`runtime_contracts.md §2.1`](../00_overview/runtime_contracts.md#21-groundedwindow). | **Cross-Modal Reasoning** questions whose answer is visual (e.g. "what color is the drink Jack picks up?") and Multi-Detail questions that need a single high-precision regrounding. |
| `transcribe_clip` | `(clip_id: int) -> list[SubtitleSpan]` | Reads `videograph.nodes[…].metadata["asr_text"]` if present; falls back to the diarized subtitle store from [`01_grounding/ §2.7`](../01_grounding/video_benchmarks_grounding.md#27-entity-resolution--re-identification). | When the retrieved memory snippet is "summarized" but the question requires **literal wording** ("what exactly did Lily say about the project?"). |

### 2.3 Decision tools

| Tool | Signature | Implementation | When the controller uses it |
|------|-----------|----------------|------------------------------|
| `final_answer` | `(answer: str, supporting_evidence: list[EvidenceId]) -> AnswerOutcome` | Terminal action; harness packages a `ReasoningTrace` with the supplied evidence refs and runs the verifier (see [`actors_reasoning_model.md §2C`](actors_reasoning_model.md)) for a final `claim_evidence_alignment` + `evidence_sufficiency` check before returning. | Whenever the controller decides accumulated evidence is sufficient. The verifier may force one more retrieval if it fails (`abstain_threshold` / `support_threshold` logic in §2C). |
| `abstain` | `(reason: AbstainReason, supporting_evidence: list[EvidenceId] = []) -> AnswerOutcome` | Records an `AbstainDecision` per [`runtime_contracts.md §2.6`](../00_overview/runtime_contracts.md#26-abstaindecision); M3-Bench scoring counts abstention as incorrect, but the trace is still logged for training. | Mandatory after `max_retrieval_steps - 1` if the controller cannot construct an answer with `supporting_evidence`. (The forced-answer fallback used by m3-agent's last round is replaced by an explicit abstain plus a "best-effort guess" auxiliary field; see §5 anti-hacking.) |

### 2.4 Tools intentionally **not** in the M3-Bench surface

| Excluded tool | Why |
|---------------|-----|
| `write_memory` / `update_entity` | M3-Bench evaluates retrieval and reasoning over pre-built memory graphs. The graphs are immutable at query time. All writes are at graph-construction time and belong to [`01_grounding/`](../01_grounding/) / [`02_memory/`](../02_memory/) procedures. |
| `replan` / `decompose_question` | Hop planning is the controller's *internal* operation, not a tool. It does not return evidence and must not appear in the tool-call log. |
| `web_search` | M3-Bench is closed-corpus; any tool that escapes the video would be a benchmark violation. |

---

## 3. Multi-turn protocol — function calling, not text actions

The current m3-agent format

```
Action: [Search] or [Answer]
Content: {content}
```

is replaced by the OpenAI / Qwen function-calling convention. One round = one `assistant` turn with zero-or-more `tool_calls`, followed by one `tool` message per call. The full protocol:

```
system   : controller policy + tool spec + question + before_clip
user     : (empty initial turn — kept for chat-template compatibility)
loop until final_answer / abstain / max_steps:
  assistant: { reasoning: <think>...</think>,           # CoT, not exposed to scoring
               tool_calls: [ { name, arguments }, ... ] }
  tool x N : { tool_call_id, content: <AtomicStepResult JSON> }
```

Concretely, the per-tool `content` returned by the harness is the JSON form of an `AtomicStepResult` from [`runtime_contracts.md §2.4`](../00_overview/runtime_contracts.md#24-atomicstepresult), which includes:

- `produced_evidence: list[EvidenceRef]` — the `evidence_refs` that the controller must cite in `final_answer.supporting_evidence`,
- `summary_text: str` — the short natural-language observation the next `[Think]` turn reads,
- `verification_passed: bool` — local check (per [`actors_reasoning_model.md §2C`](actors_reasoning_model.md)),
- `cost: { wall_ms, tokens, frames_grounded }`.

### 3.1 Why function calling (and not the text Action format)

| Property | Text `Action: [Search]` | Function calling |
|----------|--------------------------|------------------|
| Structured argument set | one free-form string per call | typed args per tool, validated against schema |
| Multiple parallel calls per turn | partial (`multiple_queries` flag in `retrieve.py`) | first-class (`tool_calls: [...]`) |
| Native to GPT-4o / Gemini / Qwen2.5 / Qwen2.5-Omni | needs custom prompt | yes |
| SFT/GRPO trace format | regex-extracted | identical to `AtomicStepResult` log — no parser |
| Trace replay for reflection | brittle (regex breaks on minor format drift) | stable JSON |

### 3.2 Backwards compatibility

The Video_Skills harness exposes a **text-action shim**: a wrapper that accepts the m3-agent `Action: [Search] / [Answer]` format, converts to a function call, and emits the same `AtomicStepResult`. This lets the existing **M3-Agent-Control** checkpoint run on Video_Skills with no retraining. The shim lives in `video_skills/adapters/m3agent_text_action.py` (Phase-2 work; not required for the trainable 8B controller).

---

## 4. Question-type → expected tool chain

The four M3-Bench question types stress different sub-tools. The mapping below is a **lower bound** on what the controller must call to answer correctly; it doubles as the gold-trajectory template used by the SFT cold-start (see §5).

| Type (count on robot.json) | Expected hop pattern | Tools per hop |
|-----------------------------|----------------------|----------------|
| **Person Understanding** (548) | 1. resolve name ↔ ID; 2. entity-centric scan; 3. compose | `resolve_character` → `list_clips_with_entity` → `search_memory_node(entity-scoped)` → `final_answer` |
| **Cross-Modal Reasoning** (476) | 1. semantic search; 2. literal subtitle / visual verify; 3. compose | `search_memory` → `transcribe_clip` *or* `view_clip` → `final_answer` |
| **Multi-Detail Reasoning** (842) | 1. semantic search; 2. dump local clip(s); 3. aggregate; 4. answer | `search_memory(topk=high)` → `get_clip_memory` × N → `final_answer` |
| **Multi-Hop Reasoning** (85) | 1. retrieve fact A; 2. retrieve fact B with A's referent; 3. compose | `search_memory` → `resolve_character`? → `search_memory` (refined with bound entity) → `final_answer` |
| **General Knowledge Extraction** (327, overlaps with above) | reuses the chain of whichever co-occurring type dominates the question semantics | — |

Counts come directly from `m3-agent/data/annotations/robot.json` (verified via the type-distribution audit in the discussion that motivated this plan). Multi-Hop is the smallest class (~6.7%); the dataset's center of gravity is **long-range single-hop retrieval over a memory graph**, which the §2.1 retrieval tools already cover.

---

## 5. Supervision signals available from M3-Bench

Each `qa_list` entry in `robot.json` / `web.json` contains four fields the controller can supervise against, **beyond the final-answer judge** (`prompt_agent_verify_answer_referencing`, GPT-4o):

| Field | Use as | Where it plugs in |
|-------|--------|-------------------|
| `answer` | terminal reward | `final_answer.content` matched by GPT-4o judge; same as m3-agent's `eval_answer` (binary). |
| `reasoning` (zh) | **trajectory shaping** | Sub-claim decomposition for `claim_evidence_alignment` checks; can also be used as gold sub-question text for SFT cold-start. |
| `timestamp` | **retrieval recall@k** | Any tool call whose `produced_evidence` contains a `clip_id` covering `timestamp` is rewarded +1; aggregated per question. |
| `before_clip` | **temporal constraint** | Hard constraint (enforced by the harness) and `temporal_consistency` verifier check from [`actors_reasoning_model.md §2C`](actors_reasoning_model.md). Any attempt to retrieve evidence with `clip_id > before_clip` is a trace violation. |
| `type` | **routing supervision** | Per §4, the expected tool chain. Used as a coarse-grained SFT target ("hop_plan" field of `ReasoningTrace`). |

### 5.1 Anti-hacking constraints specific to M3-Bench

These extend the generic list in [`actors_reasoning_model.md §2E`](actors_reasoning_model.md). All are enforced by the harness, not by the controller's prompt.

| Constraint | Detection | Penalty |
|------------|-----------|---------|
| **Temporal leak** — controller retrieves evidence with `clip_id > before_clip` | harness intercept on tool dispatch | hard reject; tool returns `{error: "temporal_leak"}`; trajectory marked invalid for reward |
| **Name leak** — controller emits a character name without first calling `resolve_character` on it | trace lint pass (post-hoc) | trace-level penalty; trains the controller to thread through `resolve_character` rather than memorizing names |
| **Search padding** — controller emits >`max_retrieval_steps`/2 retrieval calls that return zero `produced_evidence` | counted by harness | per-step negative reward (see [`actors_reasoning_model.md §2E`](actors_reasoning_model.md) "no-progress hops") |
| **Verifier collusion** — controller cites `supporting_evidence` IDs not present in any prior `AtomicStepResult.produced_evidence` | trace lint pass | hard reject of `final_answer` |
| **Best-effort guess masquerading as confident answer** — `final_answer` issued with `supporting_evidence == []` and `before_clip` retrieval budget unspent | trace lint pass | re-routed to `abstain`; counts as wrong but avoids over-abstention training pressure |

---

## 6. Evaluation

M3-Bench evaluation under this plan adds three trajectory-level metrics on top of the existing GPT-4o answer judge (which is preserved unchanged from m3-agent):

| Metric | Definition | Source |
|--------|------------|--------|
| **Answer accuracy** | GPT-4o judge of `final_answer.content` against `answer`. | m3-agent `eval_answer` |
| **Evidence recall@k** | Fraction of questions where at least one `clip_id` in `final_answer.supporting_evidence` covers `qa.timestamp`. | M3-Bench `timestamp` |
| **Trajectory legality** | Fraction of trajectories with zero anti-hacking violations (§5.1). | harness lints |
| **Tool budget** | Mean number of retrieval tool calls per question (lower is better, given equal accuracy). | trace |

Trajectory legality and tool budget are the two metrics that distinguish "the controller learned the M3-Bench data distribution" from "the controller learned a general retrieval-and-reasoning policy". They directly feed the ablation matrix in [`07_evaluation/`](../07_evaluation/) (M3-Bench rows).

---

## 7. Minimal runnable skeleton

The Phase-1 reasoning runtime in [`video_skills/`](../../video_skills/) already provides a `loop.run_question(...)` function that consumes a `HopGoal` and emits a `ReasoningTrace`. The M3-Bench wiring is a thin adapter:

```python
# video_skills/adapters/m3bench_tool_calling.py  (Phase-2)
from video_skills import build_runtime, run_question
from video_skills.tools import register_tool
from mmagent.utils.general import load_video_graph

TOOLS = [
    "search_memory", "search_memory_node", "resolve_character",
    "list_clips_with_entity", "get_clip_memory",
    "view_clip", "transcribe_clip",
    "final_answer", "abstain",
]

def run_m3bench_question(qa, mem_path):
    mem_graph = load_video_graph(mem_path)
    rt = build_runtime()
    for name in TOOLS:
        register_tool(rt, name, mem_graph=mem_graph, before_clip=qa.get("before_clip"))
    # back-translate names → character_N (per 01_grounding §2.7)
    question = rt.tools.resolve_character.back_translate(qa["question"])
    trace = run_question(
        rt,
        question,
        target_entities=[],                 # filled in by the controller, not the adapter
        tool_set=TOOLS,
        max_steps=rt.config["max_retrieval_steps"],
    )
    answer = rt.tools.resolve_character.translate(trace.answer)
    return answer, trace
```

This file does **not** exist yet. Its construction is the Phase-2 task tracked in [`99_meta/plan_docs_implementation_checklist.md`](../99_meta/plan_docs_implementation_checklist.md) under "M3-Bench adapter (multi-turn tool calling)".

---

## 8. Open work

1. **Tool registry**. The Phase-1 runtime currently exposes retrieval / verifier / harness primitives; it does not yet expose them under a function-calling-shaped `register_tool` API. This is the one schema gap between this plan and `video_skills/contracts.py`. Resolving it does not require changes to the freeze-line types — only a new `ToolSpec` table that wraps existing callables.
2. **Function-calling controller**. The current rule-based v0 controller (`video_skills/controller.py`) emits hop plans, not OpenAI tool calls. The trainable 8B controller (Phase-2) must natively emit `tool_calls`. SFT cold-start should use the §4 expected hop patterns as gold targets, with `reasoning` strings as gold CoT.
3. **`before_clip` plumbing through the harness**. Currently `before_clip` is a per-question parameter; it must become a `HopGoal` field and be propagated to every retrieval tool dispatch. This is the only required change in [`runtime_contracts.md`](../00_overview/runtime_contracts.md).
4. **Verifier checks for M3-Bench**. The existing six checks ([`actors_reasoning_model.md §2C`](actors_reasoning_model.md)) cover everything except **name round-trip** consistency (the answer must be in name space; intermediate evidence must be in character-ID space). Add a seventh check `name_space_consistency` gated to the M3-Bench adapter.
5. **Re-using the m3-agent memorization pipeline**. Memory graphs come from [`01_grounding/grounding_pipeline_execution_plan.md`](../01_grounding/grounding_pipeline_execution_plan.md), which already vendors m3-agent's perception stack. No change required here; this plan strictly consumes the produced graphs.
