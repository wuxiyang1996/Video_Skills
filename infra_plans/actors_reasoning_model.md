# Actors (Reasoning Model) — Design Plan

> Goal: Define the **8B controller / orchestrator** that manages memory,
> perspective threads, reasoning loops, and prompt composition — the central
> "actor" that drives all video understanding tasks.
>
> **Related plans:**
> - [Atomic skills & hop refactor — execution checklist](atomic_skills_hop_refactor_execution_plan.md)
> - [Agentic Memory](agentic_memory_design.md) — three memory stores + evidence layer
> - [Video Benchmarks & Grounding](video_benchmarks_grounding.md) — benchmarks, memory graph, adapters
> - [Skill Extraction / Bank](skill_extraction_bank.md) — atomic/composite skills, hops, bank infrastructure
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — skill crafting, evolution, reflection

---

## 1. Motivation

Large vision-language models (72B+) achieve strong performance on short-video
QA, but fail systematically on long-video social reasoning — questions that
require tracking hidden mental states, shifting alliances, and deceptive
behavior across tens of minutes or hours. The core bottleneck is not
perception quality but **reasoning orchestration**: deciding what to remember,
what to retrieve, and how to chain evidence across time.

We propose a **trainable 8B controller** that manages hierarchical memory,
maintains per-character perspective threads, and operates a self-evolving bank
of social inference skills. Frozen large VLMs serve only as perception tools
called on demand.

| Problem | How the small model solves it |
|---------|-------------------------------|
| Large VLMs are expensive to run on every frame | 8B processes all frames; the 72B model only sees curated text + a few key frames |
| Long videos overflow context windows | The 8B model distills hours of video into a compact memory graph |
| Reasoning from scratch produces hallucinations | Pre-extracted skills give the 72B model step-by-step scaffolds grounded in visual evidence |
| No reuse across videos | The skill bank accumulates transferable reasoning patterns |
| Prompt engineering is manual and brittle | The 8B model dynamically composes prompts with retrieved memory + skills |

**Why 8B is sufficient.** The controller never processes raw pixels. It
operates over structured text — memory nodes, skill protocols, evidence
chains — a regime where 8B models are competitive with 72B on planning and
tool-use tasks. Training a small controller is also 10-20× cheaper than
fine-tuning a 72B model.

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    8B CONTROLLER (trainable)                          │
│                                                                      │
│  Responsibilities:                                                   │
│    • Build and update hierarchical memory from observer outputs      │
│    • Maintain per-character perspective threads                       │
│    • Decompose questions into multi-hop retrieval plans               │
│    • Select, compose, and invoke social inference skills              │
│    • Decide when evidence is sufficient to answer                    │
│    • Verify answers against memory; trigger reflection on failure    │
│    • Craft new skills and refine existing ones                       │
│    • Manage the skill bank: merge, split, retire, promote            │
│                                                                      │
│  Inputs:   structured observer outputs, question text                │
│  Outputs:  memory updates, skill invocations, evidence chains,       │
│            composed prompts for frozen reasoner, skill bank updates   │
└────────┬──────────────────────────────────────┬─────────────────────┘
         │ calls (frozen tools)                  │ reads/writes
         ▼                                       ▼
┌──────────────────────┐          ┌──────────────────────────────────┐
│  Frozen Large VLMs    │          │  Persistent Data Structures       │
│                       │          │                                   │
│  Observer-A (72B)     │          │  Hierarchical Memory Graph        │
│    social extraction  │          │    event / episode / arc layers   │
│                       │          │    perspective threads per char   │
│  Observer-B (72B)     │          │                                   │
│    spatial extraction │          │  Skill Bank                       │
│                       │          │    social inference operators     │
│  Reasoner (72B)       │          │    composition DAG                │
│    evidence→answer    │          │    performance tracking           │
│                       │          │                                   │
│  Embedders (0.6B/2B)  │          │  Reasoning Traces                 │
│    retrieval index    │          │    for reflection + GRPO training │
└──────────────────────┘          └──────────────────────────────────┘
```

### 2.2 Two-Phase Operation

```
┌──────────────────────────────────────────────────────────────────────┐
│                         OFFLINE PHASE                                │
│                  (Qwen3-VL-8B, runs once per video)                  │
│                                                                      │
│   Video ──► Frame Sampler ──► Qwen3-VL-8B ──┬──► SocialVideoGraph    │
│                                              └──► SkillBank          │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        ONLINE PHASE                                  │
│                 (Qwen3-VL-8B + large VLM 72B)                        │
│                                                                      │
│   Question ──► Qwen3-VL-8B ──┬──► Memory retrieval (graph.search)   │
│                               ├──► Skill retrieval (RAG)             │
│                               ├──► Keyframe selection                │
│                               └──► Prompt Composer ──► 72B ──► Ans  │
└──────────────────────────────────────────────────────────────────────┘
```

The offline artifact is the **`SocialVideoGraph`** index ([Video Benchmarks & Grounding](video_benchmarks_grounding.md)), backed logically by **episodic / semantic / state** stores ([Agentic Memory](agentic_memory_design.md)). Existing code may still use the name `SocialVideoGraph` as an alias.

### 2.3 What the 8B Controller Does at Each Stage

| Stage | Controller Action | Frozen Tool Used |
|-------|-------------------|------------------|
| **Memory construction** | Fuses observer outputs, resolves entities, builds graph edges, distills semantic summaries, constructs perspective threads | Observers (offline, one-time) |
| **Question decomposition** | Parses question into retrieval sub-goals, identifies target entities and temporal scope | None |
| **Skill selection** | Matches question to **atomic / composite** skills via embedding + trigger conditions; plans **hops** | Embedder (retrieval) |
| **Skill execution** | Runs a **hop** (atomic chain or composite expanded to atomics); traverses memory; updates evidence chain | None (graph ops are local); `search_memory` is a **primitive**, not a bank skill |
| **Evidence sufficiency** | Evaluates whether collected evidence answers the question | None |
| **Prompt composition** | Assembles evidence chain + skill protocol + keyframes into prompt for Reasoner | None |
| **Answer generation** | Delegates to frozen Reasoner | Reasoner (single call) |
| **Verification** | Checks answer against memory for grounding, consistency, perspective correctness | None |
| **Reflection** | On failure: classifies error type, updates skill or memory accordingly | None |
| **Skill evolution** | Extracts new skills from successful traces, refines/retires based on performance | None |

### 2.4 Frozen Tool Specifications

| Tool | Default Model | Role | When Called |
|------|--------------|------|------------|
| Observer-A | Qwen3-VL-72B | Extract social signals: faces, emotions, dialogue, gaze, ToM cues | Offline (once per video) |
| Observer-B | Qwen3-VL-72B | Extract spatial signals: objects, layout, trajectories, actions | Offline (once per video) |
| Reasoner | Qwen3-VL-72B | Produce evidence-grounded answer from curated prompt | Online (once per question, max 2 retries) |
| Text Embedder | Qwen3-Embedding-0.6B | Embed queries and memory nodes for retrieval | On demand |
| MM Embedder | Qwen3-VL-Embedding-2B | Embed multimodal content (frames + text) | On demand |

### 2.5 Skill execution granularity

The controller does **not** execute only monolithic skills. It plans and executes reasoning as a sequence of **hops**, where each hop is a short composition of **atomic skills** (see [Skill Extraction / Bank](skill_extraction_bank.md)). **Composite skills** may be invoked as macros but must remain **expandable** into an explicit atomic trace for diagnosis, localization, and bank updates ([Skill Synthetics Agents §4](skill_synthetics_agents.md)).

One reasoning hop may involve a few atomic skills rather than a single monolithic reasoning skill. Atomic skills are the minimal reusable reasoning operators, while composite skills are stable, reusable short chains of atomic skills that can still be expanded for diagnosis and revision.

The controller operates over: **atomic skills**, **composite skills**, **hop traces** (ordered atomic invocations with intermediate outputs), and **reflection outputs** (failure class, localized step, repair proposal).

### 2.6 Reasoning loop

1. Parse the question and determine reasoning type / task policy.  
2. Select the **next hop goal** (what this hop must establish).  
3. Choose an **atomic chain** or **composite** skill (composite expands to atomics in the trace).  
4. Execute step-by-step over memory, perspective threads, and evidence.  
5. **Verify** intermediate outputs (local checks aligned with each atomic’s `verification_rule`).  
6. Either: continue to the **next hop**, **retrieve more evidence** (via `search_memory` and related primitives), **answer**, or **abstain**.  
7. On failure, emit a **trace** for reflection and skill-bank update ([Skill Synthetics Agents](skill_synthetics_agents.md)).

Surface protocol to the frozen reasoner may still use `[Think]` / `[Search]` / `[Answer]`; the controller’s **authoritative log** for learning is the **atomic trace** per hop.

### 2.7 Controller outputs during reasoning

During online reasoning, the controller should expose (for logging, GRPO, and evolution):

- Chosen **hop goal**  
- Selected **skill(s)** (atomic IDs and/or composite ID with expansion)  
- **Atomic trace** (ordered steps with inputs/outputs)  
- **Intermediate claims**  
- **Supporting evidence** pointers  
- **Verification** result per hop / final  
- **Confidence** and **abstain** decision  

---

## 3. Reasoning Core — Think / Search / Answer Loop

The single module that all benchmarks share.

### Core function

```python
def reason(
    question: str,
    options: Optional[List[str]],
    video_context: Union[str, List[str], "SocialVideoGraph"],
    vlm_fn: Callable,
    max_iterations: int = 5,
    mode: str = "auto",  # "direct" | "retrieval" | "auto"
) -> ReasoningResult:
```

### Two execution paths

**Direct mode** (short video — Video-Holmes, SIV-Bench):

`video_context` is raw frames/transcript fed directly to the VLM. No graph,
no retrieval. Pure chain-of-thought reasoning.

```
VLM receives: [video frames] + [question + options]
         ├─ [Think] reason about what is observed
         ├─ [Think] connect cues / infer mental states / apply physics
         └─ [Answer] final answer
```

**Retrieval mode** (long video — VRBench, LongVideoBench, CG-Bench, M3-Bench):

`video_context` is a `SocialVideoGraph`. The VLM issues `[Search]` queries to
retrieve relevant memories.

```
VLM receives: [question] + [memory context so far]
         ├─ [Think] what do I need to know?
         ├─ [Search] "who was in the kitchen at 3:00?"
         │     → system retrieves from graph, injects results
         ├─ [Think] connect retrieved evidence
         ├─ [Search] "what did face_0 say about the cup?"
         │     → system retrieves, injects
         └─ [Answer] final answer
```

**Auto mode** (default): checks video duration against a configurable
threshold (e.g. 5 min). Below → direct. Above → retrieval.

### Iteration protocol

| Tag | Meaning | System action |
|---|---|---|
| `[Think] <reflection>` | Internal reasoning step | Append to chain-of-thought; continue |
| `[Search] <query>` | Request information from memory | Run `graph.search(query)`, inject results. **Retrieval mode only.** |
| `[Answer] <answer>` | Final answer | Stop; return answer + full reasoning chain |

### Output

```python
@dataclass
class ReasoningResult:
    answer: str
    thinking_chain: List[str]
    retrieved_contexts: List[str]   # empty in direct mode
    n_iterations: int
    mode_used: str                  # "direct" | "retrieval"
    confidence: Optional[float]
```

---

## 4. Hierarchical Memory Design

### 4.0 Agentic memory model (three stores + evidence)

The controller’s **persistent semantics** follow [Agentic Memory](agentic_memory_design.md):

1. **Episodic memory** — grounded timeline: clips, timestamps, characters, dialogue/actions, evidence links.  
2. **Semantic memory** — distilled long-horizon abstractions (traits, stable relationships, patterns).  
3. **State memory** — query-time world state with **social** and **spatial** subfields (not two top-level stores).

**Visual / subtitle / audio** material is an **evidence attachment and retrieval-feature layer** on episodic and state entries, not a separate top-level memory bank.

The subsections below detail **how** episodic material is organized (timescales, nodes, edges) and how **perspective threads** (§5) feed **state memory**’s social subfield.

### 4.1 Three timescale levels (within episodic memory)

```
Arc Level (minutes–hours)
  │  Relationship trajectories, alliances, plans, suspicion arcs
  │  "How did trust between A and B evolve?"
  │
  ├── Episode Level (tens of seconds–minutes)
  │     Conversations, conflicts, joint activities, subgoals
  │     "What happened during the dinner argument?"
  │
  └──── Event Level (seconds)
          Individual actions, utterances, expressions, object states
          "Did A see B pick up the envelope?"
```

| Level | Grain | Content | Created By |
|-------|-------|---------|------------|
| **Event** | 1–10 s | A single action, utterance, expression change, or object manipulation | Controller ingests observer JSON → one event node per detected action/dialogue/state change |
| **Episode** | 30 s – 5 min | A coherent interaction: a conversation, a conflict, a shared activity | Controller clusters temporally adjacent events with shared participants and causal links |
| **Arc** | minutes – full video | A long-range development: alliance formation, trust erosion, a plan unfolding | Controller distills episode sequences into arc summaries |

### 4.2 Memory node schema (episodic backbone + links to semantic/state)

`MemoryNode` primarily implements the **episodic** backbone (event / episode / arc timescales). Distilled **semantic** summaries are separate records (or typed nodes) populated from episodic clusters. **State memory** holds the rolling social+spatial snapshot; `SocialStateEntry` (§5.3) is part of the **state** social subfield, not a fifth memory product.

```python
@dataclass
class MemoryNode:
    node_id: str
    level: str              # "event" | "episode" | "arc"
    timestamp: Tuple[float, float]
    entity_ids: List[str]
    content: Dict[str, Any] # level-specific fields
    embedding: Optional[np.ndarray]
    confidence: float       # 0–1, decays if contradicted
    provenance: str         # "observed" | "inferred"
    source_ids: List[str]   # IDs of supporting lower-level nodes
```

**Event-level content:**
```python
{
    "type": "action" | "utterance" | "expression" | "object_state",
    "description": "A picks up envelope from table",
    "agent": "face_3",
    "target": "obj_12",
    "witnesses": ["face_5"],
    "dialogue": {"speaker": "face_3", "text": "...", "tone": "..."},
    "spatial_context": {"location": "kitchen", "layout": "..."},
}
```

**Episode-level content:**
```python
{
    "summary": "A confronts B about the missing money; B denies involvement",
    "interaction_type": "confrontation" | "cooperation" | "negotiation" | ...,
    "outcome": "unresolved — A remains suspicious",
    "key_events": ["evt_042", "evt_043", "evt_047"],
    "social_dynamics": {
        "trust_change": {"face_3→face_7": -0.3},
        "information_revealed": ["face_3 now knows B was in the room"],
        "deception_detected": false,
    },
}
```

**Arc-level content:**
```python
{
    "summary": "B systematically conceals evidence from A across 3 episodes",
    "arc_type": "deception" | "alliance" | "betrayal" | "investigation" | ...,
    "trajectory": [
        {"episode_id": "ep_005", "state": "B hides envelope"},
        {"episode_id": "ep_012", "state": "B deflects A's questions"},
        {"episode_id": "ep_019", "state": "A finds contradictory evidence"},
    ],
    "resolution": "pending" | "resolved" | "escalated",
}
```

### 4.3 Graph Edges

| Edge Type | Connects | Meaning |
|-----------|----------|---------|
| `contains` | Episode → Event, Arc → Episode | Hierarchical nesting |
| `precedes` | Node → Node (same level) | Temporal ordering |
| `causes` | Event → Event | Causal link |
| `supports` | Event → Episode/Arc | Evidence for a higher-level conclusion |
| `contradicts` | Event → Episode/Arc | Counter-evidence |
| `participates` | Entity → Event/Episode | Character involvement |
| `witnesses` | Entity → Event | Character was present and could observe |
| `trusts` / `suspects` | Entity → Entity | Social relation at a time point |
| `believes` | Entity → MemoryNode | Attributed belief (may differ from ground truth) |

---

## 5. Perspective-Aware Social Memory

Perspective threads and `SocialStateEntry` support **state memory** (social subfield): query-time beliefs, knowledge boundaries, and stance — always evidence-linked to episodic entries. They are not an additional top-level memory category beside episodic / semantic / state.

### 5.1 The perspective confusion problem

The most common failure in social reasoning is **perspective confusion**:
the system answers based on what the viewer knows, rather than what a
specific character knows.

> Q: "Does Alice know that Bob stole the key?"
> Correct: No — Alice was not in the room when Bob took it.
> Common error: Yes — the system saw Bob take it and projects that knowledge onto Alice.

### 5.2 Perspective Thread Schema

```python
@dataclass
class PerspectiveThread:
    entity_id: str
    entity_name: str
    observed_events: List[str]
    heard_dialogue: List[str]
    inferred_beliefs: List[SocialStateEntry]
    goals: List[str]
    knowledge_boundary: str   # summary of what this character does NOT know
    last_updated: float
    update_history: List[Dict]
```

### 5.3 Social-State Entry Schema

```python
@dataclass
class SocialStateEntry:
    entry_id: str
    entities: List[str]
    state_type: str    # "belief" | "intention" | "trust" | "suspicion" | "commitment" | "deception"
    description: str
    timestamp: Tuple[float, float]
    confidence: float

    provenance: str    # "directly_observed" | "inferred_from_dialogue" | "inferred_from_behavior" | "inferred_from_absence"
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    alternative_hypotheses: List[str]
    is_active: bool    # false if superseded by later state
```

### 5.4 Perspective Thread Construction

1. For each event, check `witnesses` to determine present characters.
2. Add event to `observed_events` of each witness.
3. For dialogue events, add to `heard_dialogue` of speaker and listeners.
4. After each episode, run a belief-update pass: infer what each character
   likely believes based on what they've observed and heard.
5. Flag cases where a character's perspective diverges from ground truth.

### 5.5 Perspective-Aware Retrieval

When the controller encounters "Does A know X?":
1. Retrieve A's perspective thread, not the global memory.
2. Check whether the relevant event is in A's `observed_events`.
3. If not, check whether A could have learned through dialogue or inference.
4. Return answer with explicit evidence trail.

---

## 6. Question Analyzer

When a question arrives, the 8B model decomposes it into structured
retrieval signals.

```python
@dataclass
class QuestionAnalysis:
    question_type: str          # "SR", "TCI", "MHR", "PAR", "CTI"
    target_entities: List[str]
    temporal_scope: str         # "full_video" | "segment" | "point_in_time"
    temporal_hint: Optional[Tuple[float, float]]
    required_evidence: List[str]
    reasoning_type: str         # "causal" | "temporal" | "relational" | "descriptive"
    retrieval_queries: List[str]
    skill_tags: List[str]
    difficulty_estimate: str    # "simple" | "moderate" | "hard"
```

### Adaptive Routing

| Difficulty | Route |
|-----------|-------|
| **simple** | 8B answers directly from memory graph |
| **moderate** | 8B composes prompt, 72B answers |
| **hard** | 8B composes prompt with extra skills + more memory, 72B answers with CoT |

This routing saves 72B inference cost on ~30-40% of questions.

---

## 7. Keyframe Selector

```python
def select_keyframes(
    question_analysis: QuestionAnalysis,
    memory_graph: SocialVideoGraph,
    retrieved_memories: List[MemoryNode],
    max_frames: int = 5,
) -> List[KeyFrame]:
```

1. **Temporal anchoring:** frames from timestamps of retrieved memory nodes
2. **Entity anchoring:** frames containing target entities
3. **Diversity sampling:** span different segments
4. **Saliency scoring:** 8B model scores candidate frames by relevance

| Scenario | Max frames | Rationale |
|----------|-----------|-----------|
| Simple factual | 1-2 | Verification only |
| Relationship / causal | 3-5 | Need multiple moments |
| Temporal ordering | 5-8 | Representative moments from each segment |

---

## 8. Prompt Composer

The 8B model assembles a structured prompt for the 72B model:

```
╔═══════════════════════════════════════════════════╗
║  SYSTEM                                            ║
║  Expert video analyst instructions                 ║
╠═══════════════════════════════════════════════════╣
║  REASONING SKILLS  (from skill bank, top-k)        ║
║  --- Skill 1: {skill.name} ---                     ║
║  Steps: {protocol.steps}                           ║
║  --- Skill 2: ... ---                              ║
╠═══════════════════════════════════════════════════╣
║  MEMORY CONTEXT  (from SocialVideoGraph, top-k)    ║
║  [Episodic] 00:12-00:24: ...                       ║
║  [Semantic] ...                                    ║
║  [Entity] <face_0>: ...                            ║
╠═══════════════════════════════════════════════════╣
║  VISUAL FRAMES  (keyframes from 8B selector)       ║
║  [Frame @ 00:15] <image_1>                         ║
╠═══════════════════════════════════════════════════╣
║  QUESTION + OPTIONS                                ║
╠═══════════════════════════════════════════════════╣
║  INSTRUCTIONS                                      ║
║  <think>reasoning</think> <answer>X</answer>       ║
╚═══════════════════════════════════════════════════╝
```

### Token Budget Management

| Section | Token budget | Priority |
|---------|-------------|----------|
| System + instructions | ~300 | Fixed |
| Skills (top-k) | ~800 (k=3) | High — reasoning scaffold |
| Memory context | ~1200 (top-5 nodes) | High — factual grounding |
| Visual frames | ~5000 (5 frames) | Medium — visual grounding |
| Question + options | ~200 | Fixed |
| **Total** | **~7500** | Fits within 32K context |

---

## 9. Self-Reflection Loop

After the 72B model produces an answer, the 8B model verifies it:

1. **Claim extraction:** 8B extracts factual claims from the 72B answer
2. **Memory grounding:** search memory graph for each claim; flag ungrounded claims
3. **Temporal consistency:** verify reasoning chain respects temporal order
4. **Entity consistency:** verify entity references match entity nodes

| Verification outcome | Action |
|---------------------|--------|
| All claims grounded | Accept answer |
| 1-2 ungrounded claims | Re-prompt 72B with corrections |
| Major contradiction | Re-compose prompt with additional memory and re-run |
| Temporal error | Re-compose with explicit timeline and re-run |

Max **2 iterations** to bound cost.

---

## 10. Training the 8B Controller

### 10.1 What Is Trained

**Only the 8B controller.** All other components are frozen.

| Component | Trainable? | Optimization Target |
|-----------|-----------|---------------------|
| **8B Controller** | **Yes** | Memory management, skill selection, retrieval planning, evidence sufficiency, verification, reflection |
| Observer-A/B (72B) | No | Frozen perception tools |
| Reasoner (72B) | No | Frozen answer generator |
| Embedders | No | Frozen retrieval index |
| Skill Bank | Evolves (not gradient-trained) | Updated by reflection logic |

### 10.2 LoRA Adapters

| Adapter | Input | Output | When Active |
|---------|-------|--------|-------------|
| `memory_builder` | Observer JSON / video frames | Memory graph update commands / structured observation JSON | Offline memory construction |
| `planner` | Question + memory state | Skill selection + retrieval plan | Online, per question |
| `question_analyzer` | Question text | QuestionAnalysis JSON | Online, per question |
| `prompt_composer` | Context bundle | Optimized prompt | Online, per question |
| `verifier` | Answer + evidence chain + memory | Accept / reject / retry decision | Online, post-answer |
| `reflector` | Failed trace + ground truth | Failure classification + skill update | Post-evaluation |
| `self_reflector` | Answer + memory | Verification judgement | Online, post-answer |

### 10.3 Dual-Thread Reward

Every question runs through two parallel threads: one with skill bank access,
one without.

| A Correct? | B Correct? | `r_outcome` | Meaning |
|-----------|-----------|-------------|---------|
| Yes | No | **+1.0** | Skills were the deciding factor |
| Yes | Yes | **+0.2** | Skills at least didn't hurt |
| No | No | **-0.3** | Skills failed to help |
| No | Yes | **-1.0** | Skills actively damaged reasoning |

### 10.4 Step-Level Reward

| Signal | Value | Condition |
|--------|-------|-----------|
| `r_evidence` | **+0.1** | Retrieved a new, relevant memory node |
| `r_grounding` | **+0.15** | Grounded a question entity in the graph |
| `r_progress` | **+0.2** | Closed an identified evidence gap |
| `r_novel_info` | **+0.1** | Found non-redundant information |
| `p_turn_cost` | **-0.05** | Per-turn fixed cost (encourages efficiency) |
| `p_redundant` | **-0.10** | Re-retrieved already-known information |
| `p_irrelevant` | **-0.15** | Retrieved low-relevance evidence |
| `p_wrong_skill` | **-0.20** | Invoked a skill with unmet preconditions |
| `p_hallucination` | **-0.30** | Claim not grounded in any memory node |

### 10.5 Composite Reward

```
R = 0.35 × r_outcome
  + 0.25 × (step_total_A - step_total_B)
  + 0.20 × (evidence_quality_A - evidence_quality_B)
  + 0.10 × efficiency_bonus
  + 0.10 × step_total_A
```

### 10.6 Training via GRPO

The composite reward feeds into Group Relative Policy Optimization. The
controller learns when to invoke skills, which skills to select, how to
compose multi-step chains, and when to stop gathering evidence.

| Training Phase | Data Source | Signal |
|---------------|------------|--------|
| **Cold start** | Video-Holmes + MA-EgoQA | Outcome reward only |
| **Skill evolution** | Same benchmarks, iterative | Dual-thread reward with step-level scoring |
| **Cross-benchmark transfer** | Train on one, evaluate on another | Skill bank portability signal |

---

## 11. Model Choices and Alternatives

### Recommended Small Models

| Model | Params | VRAM | Strength | Limitation |
|-------|--------|------|----------|------------|
| **Qwen3-VL-8B** (default) | 8B | ~16 GB | Strong structured output, native multi-image | Weaker on subtle social reasoning |
| Qwen2.5-VL-7B | 7B | ~14 GB | Proven locally | Older generation |
| InternVL3-8B | 8B | ~16 GB | Strong visual grounding | Different API |
| Phi-4-multimodal | 5.6B | ~12 GB | Very efficient | Smaller capacity |

### Recommended Large Models

| Model | Params | VRAM | When to use |
|-------|--------|------|-------------|
| **Qwen3-VL-72B-Instruct** (default) | 72B | ~144 GB (4×A100) | Best accuracy, same family |
| InternVL2.5-78B | 78B | ~156 GB | Alternative, strong on M3-Bench |
| Qwen3-VL-32B | 32B | ~64 GB (2×A100) | Budget option |
| GPT-5.4 / Claude 4.6 | API | API | When local GPU unavailable |

---

## 12. Latency Analysis

### 12.1 Online Phase — Per-Question Latency

```
Question ──► 8B calls (~3-5 s) ──► 72B call (~15-25 s) ──► Answer
```

| Step | Model | Latency |
|------|-------|---------|
| Question analysis | 8B | ~1.2 s |
| Memory retrieval | Embedding search (CPU) | ~50 ms |
| Skill retrieval | Embedding search (CPU) | ~50 ms |
| Keyframe selection | 8B | ~1.5 s |
| Prompt composition | CPU | ~10 ms |
| **72B inference** | 72B on 4×A100 | **~18 s** |
| Self-reflection (opt.) | 8B | ~2 s |
| **Total per question** | | **~23 s** |

### 12.2 Comparison: Orchestrator vs Raw 72B

| Approach | 72B input tokens | Total |
|----------|-----------------|-------|
| 72B on all 60 frames (raw) | ~60K | **40-50 s** |
| 72B on 15 sampled frames | ~15K | **22-30 s** |
| **Orchestrator (8B + 72B)** | **~7.5K** | **~20-25 s** |
| Orchestrator (simple Q, 8B only) | 0 | **~2-3 s** |

The orchestrator's online latency is **constant** regardless of video
length because the memory graph compresses the video into a fixed-size
text representation.

### 12.3 Deployment Scenarios

| Scenario | Config | Per-Q latency | Hardware |
|----------|--------|---------------|----------|
| **Research eval** | 8B + 72B + reflection | ~23 s | 1 + 4 A100s |
| **Fast eval** | 8B + 72B, no reflection | ~18 s | 1 + 4 A100s |
| **Budget** | 8B + 32B, no reflection | ~10-12 s | 1 + 2 A100s |
| **API** | 8B local + GPT-5.4 API | ~8-15 s | 1 A100 + API |
| **8B only** | 8B with routing | ~2-3 s | 1 A100 |

### 12.4 Batch Throughput

| Batch size | Per-question amortized | Questions/hour |
|-----------|----------------------|----------------|
| 1 (sequential) | ~23 s | ~156 |
| 4 (concurrent) | ~8-10 s | ~360-450 |
| 8 (concurrent, 8B batched) | ~6-8 s | ~450-600 |

---

## 13. Module Layout

```
Video_Skills/small_model_orchestrator/
├── __init__.py
├── config.py                     # Model paths, thresholds, prompt templates
├── orchestrator.py               # SmallModelOrchestrator — top-level controller
├── memory_builder.py             # Video → SocialVideoGraph (offline)
├── skill_crafter.py              # SocialVideoGraph → SkillBank (offline)
├── prompt_composer.py            # Question → composed prompt for large VLM
├── keyframe_selector.py          # Pick the N most question-relevant frames
├── question_analyzer.py          # Decompose question into retrieval signals
├── self_reflection.py            # 8B model verifies/refines the 72B answer
└── adapters/
    ├── video_holmes.py
    ├── m3_bench.py
    └── siv_bench.py
```

### Orchestrator API

```python
class SmallModelOrchestrator:
    def __init__(
        self,
        small_model: str = "Qwen/Qwen3-VL-8B",
        large_model: str = "Qwen/Qwen3-VL-72B-Instruct",
        embedder: str = "Qwen/Qwen3-Embedding-0.6B",
        mm_embedder: str = "Qwen/Qwen3-VL-Embedding-2B",
        skill_bank_path: Optional[str] = None,
        max_keyframes: int = 5,
        top_k_skills: int = 3,
        top_k_memories: int = 5,
        enable_self_reflection: bool = True,
    ):
        ...

    # --- Offline API ---
    def process_video(self, video_path, subtitle_path=None) -> SocialVideoGraph: ...
    def craft_skills(self, graph, existing_bank=None) -> SkillBank: ...

    # --- Online API ---
    def answer(self, question, options=None, video_path=None, graph=None) -> AnswerResult: ...
    def compose_prompt(self, question_analysis, graph, skill_bank) -> str: ...
```

---

## 14. Future Extensions

### A. Online Skill Learning
When the 72B model discovers a novel reasoning pattern, the 8B model
packages it as a new skill and adds it to the bank — creating a
co-evolution loop.

### B. Multi-Turn Dialogue
The 8B model maintains a conversation memory alongside the video memory
graph, allowing the prompt composer to include dialogue history.

### C. Real-Time Video Processing
The 8B model runs the observe pipeline in real time (1 fps), continuously
updating the memory graph.

### D. Hierarchical Multi-Model Cascade
For extremely long videos (10+ hours):
```
Qwen3-VL-3B (frame-level triage, ~6 GB)
    → Qwen3-VL-8B (segment-level reasoning, ~16 GB)
        → Qwen3-VL-72B (question answering, ~144 GB)
```

---

## 15. Related Work

| Work | Relationship |
|------|-------------|
| **M3-Agent** (M3-Bench) | Memory graph for video QA. We add hierarchical levels, perspective threads, trainable skill management. |
| **SCALAR** (arXiv:2603.09036) | LLM-proposed symbolic skills. We extend to social inference operators with evidence grounding and self-evolution. |
| **WorldMM** (arXiv:2512.02425) | World model for multimodal reasoning. We add perspective-aware social state and trainable orchestration. |
| **Video-Holmes** (arXiv:2505.21374) | Deep reasoning benchmark. Our primary evaluation target. |
| **arXiv:2603.24558** | Direct comparison. We add hierarchical memory, perspective threads, skill evolution, and evidence grounding. |
