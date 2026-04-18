# Skill Extraction & Skill Bank — Design Plan

> Goal: Define the **skill bank infrastructure** — how skills are represented,
> stored, composed, and managed. This covers universal video skills, social
> inference operators, the skill bank class, and composition patterns.
>
> **Related plans:**
> - [Agentic Memory](agentic_memory_design.md) — episodic / semantic / state stores + evidence layer
> - [Video Benchmarks & Grounding](video_benchmarks_grounding.md) — benchmarks, memory graph, adapters
> - [Actors / Reasoning Model](actors_reasoning_model.md) — 8B controller, reasoning core, orchestrator
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — skill crafting, evolution, quality control

---

## 1. Skills as Reusable Operators

Skills in this system are **not** generic retrieval routines. They are
**reusable inference operators** — each encodes a specific reasoning pattern
over the memory graph and perspective threads. They come in two flavors:

1. **Universal video skills** — applicable to every benchmark regardless of
   video length (always active)
2. **Social inference skills** — specialized reasoning patterns for social
   dynamics, perspective tracking, and theory-of-mind questions

---

## 2. Universal Video Skills

Always available, regardless of video length or memory layer status.

| skill_id | Name | What it does |
|---|---|---|
| `reason_chain` | Chain-of-Thought Reasoning | The core `[Think]/[Answer]` loop. Every benchmark uses this. |
| `temporal_reason` | Temporal Ordering | Reason about event ordering using visual evidence or timestamps |
| `causal_reason` | Causal / Physical Inference | Infer cause-effect; applies to Video-Holmes deduction, SIV-Bench counterfactuals |
| `social_reason` | Social State Inference | Infer unobservable mental states (emotions, intentions, attitudes, relationships) from observable cues |

### Long-video-only skills (activated when memory layer is active)

| skill_id | Name | What it does |
|---|---|---|
| `observe_segment` | Observe Video Segment | Run observation pipeline on one clip; produce episodic descriptions |
| `detect_entities` | Detect & Track Entities | Face/voice detection and entity graph updates |
| `build_episodic` | Build Episodic Memory | Store timestamped observations with entity links and evidence attachments |
| `build_semantic` | Build Semantic Memory | Distill episodic clusters into semantic summaries (long-horizon abstractions) |
| `update_state` | Maintain State Memory | Refresh **social + spatial** query-time state from new evidence (single store, two subfields; not separate “social” and “spatial” memories) |
| `search_memory` | Retrieve from Memory | Embed a query and return top-k matches from the graph / stores |

### Example skill definition

```python
Skill(
    skill_id="reason_chain",
    name="Chain-of-Thought Reasoning",
    strategic_description=(
        "Iterative Think/Answer reasoning over video evidence. "
        "Works in direct mode (raw video in context) or retrieval mode "
        "(search memory graph). Universal across all benchmarks."
    ),
    tags=["REASONING", "UNIVERSAL"],
    protocol=Protocol(
        preconditions=["question_received=true"],
        steps=[
            "Determine mode: direct (video in context) or retrieval (memory graph)",
            "Generate [Think] step: reason about available evidence",
            "If evidence insufficient and retrieval available: [Search] for more",
            "Repeat Think/Search until confident",
            "Generate [Answer] with final response",
        ],
        success_criteria=["answer_produced=true"],
        abort_criteria=["max_iterations_exceeded"],
        expected_duration=5,
    ),
    contract=SkillEffectsContract(
        skill_id="reason_chain",
        eff_add={"answer_produced", "reasoning_chain_generated"},
        eff_event={"reasoning_completed"},
    ),
)
```

### Composition patterns

**Direct mode** (Video-Holmes, SIV-Bench):
```
[raw video + question] → reason_chain
                           ├─ [Think] (perceive + reason)
                           ├─ [Think] (connect / infer)
                           └─ [Answer]
```

**Retrieval mode** (VRBench, LongVideoBench, CG-Bench, M3-Bench):
```
[offline] for each clip:
              detect_entities → observe_segment → build_episodic
          build_semantic (aggregate)
          update_state (social + spatial snapshot for query-time reasoning)

[online]  [question] → reason_chain
                         ├─ [Think]
                         ├─ [Search] → search_memory (+ entity translate if needed)
                         ├─ [Think]
                         └─ [Answer]
```

---

## 3. Core Social Inference Skills

Each social skill operates over the memory graph and perspective threads
maintained by the 8B controller.

| Skill | Purpose | Trigger | Execution Pattern | Expected Output |
|-------|---------|---------|-------------------|-----------------|
| `who_saw_what` | Determine which characters witnessed a specific event | "did X see/know about Y" | Locate event → check `witnesses` → cross-reference perspective threads | List of witnesses + non-witnesses with evidence |
| `infer_belief_update` | Track how a character's belief changed after an event | Belief change question | Get perspective thread → find belief before event → check if event in observed_events → infer post-event belief | Before/after belief pair with evidence |
| `track_commitment` | Track whether a character kept or broke a promise | Promise fulfillment question | Locate commitment event → find subsequent actions → compare against commitment | Kept/broken with timeline |
| `detect_intention_shift` | Identify when a character's goal changed | Motivation question | Query perspective thread goals across time → identify change points → retrieve causal events | Intention timeline with pivot events |
| `resolve_conflicting_testimony` | Reconcile contradictory claims by different characters | Multiple conflicting accounts | Retrieve each character's perspective → identify what each observed → determine consistency | Reconciliation with credibility scores |
| `identify_hidden_action` | Detect an action performed when others weren't present | Concealed behavior question | Find events where actor is sole witness → check if info leaked later | Hidden action + who does/doesn't know |
| `track_relationship_change` | Trace relationship evolution over time | Relationship trajectory question | Query arc-level memory → retrieve episodes → extract trust/stance at each point | Relationship timeline with evidence |
| `social_causal_attribution` | Determine why a social state changed | "Why did A become suspicious of B?" | Find target social-state entry → trace supporting_evidence → follow causal edges | Causal chain from trigger to outcome |
| `deception_hypothesis_check` | Test whether a character is deceiving another | Suspected deception | Compare deceiver's perspective thread with public statements → check information asymmetry → find hidden actions | Deception confirmed/denied with evidence |
| `multi_perspective_reconciliation` | Compare how multiple characters perceive the same event | Different viewpoints question | Retrieve perspective threads for all entities → extract each character's view → identify divergences | Per-character view table with divergences |

### Failure modes per skill

| Skill | Known Failure Modes |
|-------|--------------------|
| `who_saw_what` | Event not found; ambiguous presence |
| `infer_belief_update` | Character absent; belief not inferable |
| `track_commitment` | Commitment ambiguous; outcome unclear |
| `detect_intention_shift` | Subtle shift; insufficient evidence |
| `resolve_conflicting_testimony` | Both accounts plausible; no resolution |
| `identify_hidden_action` | Action not captured by observer |
| `track_relationship_change` | Sparse interactions; no clear arc |
| `social_causal_attribution` | Multiple causes; weak evidence |
| `deception_hypothesis_check` | Deception too subtle; missing data |
| `multi_perspective_reconciliation` | Perspectives too similar to distinguish |

---

## 4. Skill Representation

### 4.1 SocialSkill Schema

```python
@dataclass
class SocialSkill:
    skill_id: str
    name: str
    purpose: str                    # one sentence
    trigger_conditions: List[str]   # when to apply this skill
    required_inputs: List[str]      # what the controller must provide
    execution_steps: List[str]      # ordered steps over memory graph
    expected_outputs: List[str]     # what the skill produces
    failure_modes: List[str]        # known ways this skill can fail
    refinement_signals: List[str]   # what feedback improves this skill

    # Performance tracking
    n_invocations: int = 0
    success_rate: float = 0.0
    avg_evidence_quality: float = 0.0
    version: int = 1
    protocol_history: List[Dict] = field(default_factory=list)
```

### 4.2 COS-PLAY-Compatible Skill Schema

Skills also conform to the existing COS-PLAY `Skill` schema:

```python
Skill(
    skill_id: str,
    name: str,
    strategic_description: str,
    tags: List[str],           # e.g. ["REASONING", "SOCIAL", "PERSPECTIVE"]
    protocol: Protocol(
        preconditions: List[str],
        steps: List[str],
        success_criteria: List[str],
        abort_criteria: List[str],
        expected_duration: int,
    ),
    contract: SkillEffectsContract(
        skill_id: str,
        eff_add: Set[str],     # postconditions added
        eff_event: Set[str],   # events emitted
    ),
    sub_episodes: List[str],   # evidence pointers
    n_instances: int,
)
```

---

## 5. Skill Composition

### 5.1 Composition Example

```
Question: "Does Alice know that Bob stole the key?"

Strategy: who_saw_what → infer_belief_update → deception_hypothesis_check

  Step 1: who_saw_what(event="Bob takes key")
    → Bob was alone; Alice not present [E-7, 02:15]

  Step 2: infer_belief_update(entity="Alice", event="Bob takes key")
    → Alice has no direct knowledge; check for indirect learning
    → No dialogue informed Alice [perspective thread shows gap]

  Step 3: deception_hypothesis_check(deceiver="Bob", target="Alice")
    → Bob has not mentioned the key; Alice asked about it at 04:30
    → Bob deflected → consistent with concealment

  Answer: No, Alice does not know. Evidence: [E-7], [E-23], [perspective gap]
```

### 5.2 Prompt variant mapping

| Question type | Skill emphasis | Memory emphasis | Frame emphasis |
|---------------|---------------|-----------------|----------------|
| Social Relationship (SR) | Character analysis, interaction tracking | Entity nodes, interaction episodes | Frames with multiple people |
| Temporal Causal (TCI) | Causal reasoning, temporal tracking | Chronological episodic chain | Frames at cause and effect points |
| Hidden Reasoning (MHR) | Threat detection, pattern recognition | Semantic inferences, anomaly nodes | Frames with subtle cues |
| Temporal Arrangement (TA) | Temporal ordering | Timestamped episodic nodes | Spread across timeline |
| Core Theme (CTI) | Theme extraction, pattern recognition | Semantic summary nodes | Representative frames |

---

## 6. Skill Bank Infrastructure

### 6.1 SkillBank Class

```python
class SkillBank:
    skills: Dict[str, SocialSkill]
    composition_edges: Dict[str, List[str]]  # parent strategy → child skills
    performance: Dict[str, SkillPerformance]
    co_occurrence: Dict[Tuple[str, str], int]

    def select(self, question_analysis: Dict) -> List[SocialSkill]: ...
    def compose(self, skill_ids: List[str]) -> ComposedStrategy: ...
    def update_performance(self, skill_id: str, trace: ReasoningTrace) -> None: ...
    def craft_new_skill(self, failure_analysis: FailureAnalysis) -> Optional[SocialSkill]: ...
    def maintain(self) -> MaintenanceReport: ...  # merge, split, retire, promote
```

### 6.2 Skill DAG

Skills form a directed acyclic graph tracking composition and performance:

- **Composition edges** link parent strategies to child skills
- **Performance tracking** records success rate, evidence quality, invocation count
- **Co-occurrence matrix** identifies which skills frequently chain together,
  informing future composition suggestions

### 6.3 Skill Retrieval

Skill retrieval uses RAG scoring with three factors (from
`decision_agents/agent_helper.py`):

1. **Relevance** — embedding similarity between question and skill description
2. **Applicability** — trigger condition match against question analysis
3. **Pass rate** — historical success rate of the skill

### 6.4 Bank Storage

- `skill_bank.jsonl` — COS-PLAY-compatible `Skill` objects
- Each skill carries: `skill_id`, `name`, `strategic_description`, `tags`,
  `protocol`, `contract`, `sub_episodes` (evidence pointers), `n_instances`
- Compatible with `SkillBankMVP` from `skill_agents/skill_bank/bank.py`

---

## 7. Integration with Existing Components

| Existing component | How it connects |
|---|---|
| `skill_agents/stage3_mvp/schemas.py` → `Skill`, `Protocol`, `SkillEffectsContract` | All skills use these schemas directly |
| `skill_agents/skill_bank/bank.py` → `SkillBankMVP` | Crafted skill bank stored as SkillBankMVP-compatible JSONL |
| `skill_agents/skill_evaluation/` → LLM judge | Quality control reuses evaluation dimensions |
| `decision_agents/agent_helper.py` → `select_skill_from_bank()` | Prompt composer uses the same RAG scoring |
| `rag/retrieval.py` → `MemoryStore` | Skill bank retrieval uses `SkillQueryEngine` |
| `rag/embedding/` → embedders | Skill embeddings for retrieval |
| `data_structure/experience.py` → `Experience`, `Episode` | Each Q&A trace can be packaged as an `Episode` |
