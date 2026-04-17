# A Trainable 8B Controller for Self-Evolving Social Reasoning over Long Videos

> **Location:** `Video_Skills/infra_plans/self_evolving_skill_bank_design.md`
>
> **Key Words:** Trainable 8B Controller, Self-Evolving Skill Bank, Hierarchical
> Agentic Memory, Perspective-Aware Social Reasoning, Long-Video Understanding,
> Continual Learning, Multi-Hop Evidence Retrieval

---

## 1. Motivation

Large vision-language models (72B+) achieve strong performance on short-video
QA, but fail systematically on long-video social reasoning — questions that
require tracking hidden mental states, shifting alliances, and deceptive
behavior across tens of minutes or hours of video. The core bottleneck is not
perception quality but **reasoning orchestration**: deciding what to remember,
what to retrieve, and how to chain evidence across time.

We propose a **trainable 8B controller** that manages hierarchical memory,
maintains per-character perspective threads, and operates a self-evolving bank
of social inference skills. Frozen large VLMs serve only as perception tools
called on demand. The 8B controller is the sole trainable component and the
primary contribution.

**Why 8B is sufficient.** The controller never processes raw pixels. It
operates over structured text — memory nodes, skill protocols, evidence
chains — a regime where 8B models are competitive with 72B on planning and
tool-use tasks. Training a small controller is also 10-20× cheaper than
fine-tuning a 72B model, enabling rapid iteration on skill and memory design.

---

## 2. Problem Formulation

### 2.1 The Long-Video Social Reasoning Challenge

Long-video social reasoning differs from standard video QA in three ways:

**Multi-timescale tracking.** A single question ("Does Alice trust Bob by the
end?") may require evidence spanning the full video: an early promise, a
mid-video betrayal witnessed by a third party, a late reconciliation. The
system must track events at second-level, episode-level (conversations,
conflicts), and arc-level (relationship trajectories over the full video).

**Hidden mental state modeling.** Social reasoning requires inferring states
that are never directly shown:
- **Who saw what** — was character A present when event X occurred?
- **Who knows what** — did the information propagate through dialogue or observation?
- **Who believes what** — does A believe B's claim, or does A suspect deception?
- **Who intends what** — is A cooperating, competing, or deceiving?
- **When states changed** — at what moment did trust break, or suspicion begin?
- **What evidence supports each inference** — is the belief directly observed
  or inferred from indirect cues?

**Perspective asymmetry.** The system (and the viewer) has a god's-eye view,
but each character has a local, partial view. Confusing global truth with a
character's perspective is the most common failure mode in theory-of-mind
reasoning. The system must maintain separate perspective threads for each
important character and reason about what each character can and cannot know.

### 2.2 Scope and Non-Goals

| In Scope | Out of Scope |
|----------|-------------|
| Social reasoning (beliefs, intentions, trust, deception) | General video captioning or description |
| Long videos (10 min to 10+ hours) | Single-frame or few-second clips |
| Evidence-grounded answers with timestamps | Open-ended generation without grounding |
| Trainable 8B controller | Training or fine-tuning 72B models |
| Skill bank evolution over time | Static prompt engineering |
| Multi-hop retrieval with chain-of-evidence | Single-step retrieval |

---

## 3. System Overview

### 3.1 Architecture

The 8B controller is the central agent. All other components are either
frozen tools (large VLMs, embedders) or persistent data structures (memory
graph, skill bank) that the controller reads and writes.

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

### 3.2 What the 8B Controller Does at Each Stage

| Stage | Controller Action | Frozen Tool Used |
|-------|-------------------|------------------|
| **Memory construction** | Fuses observer outputs, resolves entities, builds graph edges, distills semantic summaries, constructs perspective threads | Observers (offline, one-time) |
| **Question decomposition** | Parses question into retrieval sub-goals, identifies target entities and temporal scope | None |
| **Skill selection** | Matches question to skill bank via embedding + trigger conditions | Embedder (retrieval) |
| **Skill execution** | Runs selected skill: traverses memory graph, retrieves evidence, updates evidence chain | None (graph ops are local) |
| **Evidence sufficiency** | Evaluates whether collected evidence answers the question | None |
| **Prompt composition** | Assembles evidence chain + skill protocol + keyframes into prompt for Reasoner | None |
| **Answer generation** | Delegates to frozen Reasoner | Reasoner (single call) |
| **Verification** | Checks answer against memory for grounding, consistency, perspective correctness | None |
| **Reflection** | On failure: classifies error type, updates skill or memory accordingly | None |
| **Skill evolution** | Extracts new skills from successful traces, refines/retires based on performance | None |

### 3.3 Frozen Tool Specifications

These are interchangeable; the design is model-agnostic.

| Tool | Default Model | Role | When Called |
|------|--------------|------|------------|
| Observer-A | Qwen3-VL-72B | Extract social signals: faces, emotions, dialogue, gaze, ToM cues | Offline (once per video) |
| Observer-B | Qwen3-VL-72B | Extract spatial signals: objects, layout, trajectories, actions | Offline (once per video) |
| Reasoner | Qwen3-VL-72B | Produce evidence-grounded answer from curated prompt | Online (once per question, max 2 retries) |
| Text Embedder | Qwen3-Embedding-0.6B | Embed queries and memory nodes for retrieval | On demand |
| MM Embedder | Qwen3-VL-Embedding-2B | Embed multimodal content (frames + text) | On demand |

---

## 4. Hierarchical Memory Design

### 4.1 Three Timescale Levels

The controller maintains memory at three levels of temporal granularity.
Each level serves different question types and retrieval patterns.

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
| **Event** | 1–10 s | A single action, utterance, expression change, or object manipulation. Directly grounded in observer output. | Controller ingests observer JSON → one event node per detected action/dialogue/state change |
| **Episode** | 30 s – 5 min | A coherent interaction: a conversation, a conflict, a shared activity. Groups related events. | Controller clusters temporally adjacent events with shared participants and causal links |
| **Arc** | minutes – full video | A long-range development: alliance formation, trust erosion, a plan unfolding, deception revealed. Links episodes into narrative threads. | Controller distills episode sequences into arc summaries (second pass) |

### 4.2 Memory Node Schema

All memory nodes share a base schema with level-specific extensions:

```python
@dataclass
class MemoryNode:
    node_id: str
    level: str              # "event" | "episode" | "arc"
    timestamp: Tuple[float, float]  # (start_sec, end_sec)
    entity_ids: List[str]   # participants
    content: Dict[str, Any] # level-specific fields (below)
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
    "witnesses": ["face_5"],        # who was present and could see this
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
    "key_events": ["evt_042", "evt_043", "evt_047"],  # constituent events
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
    "key_episodes": ["ep_005", "ep_012", "ep_019"],
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
| `sees` / `cannot_see` | Entity → Entity/Object | Perceptual access |
| `believes` | Entity → MemoryNode | Attributed belief (may differ from ground truth) |

### 4.4 Why Hierarchical?

- **Event-only** memory produces thousands of nodes for a 1-hour video,
  making retrieval noisy. Episode and arc levels provide pre-aggregated
  summaries for coarse-grained questions.
- **Arc-level** memory directly supports relationship-tracking questions
  without requiring the controller to re-derive long-range patterns at
  query time.
- **Cross-level retrieval**: the controller first retrieves at the arc or
  episode level (coarse), then drills into constituent events for evidence
  grounding (fine). This is faster and more precise than flat search.

---

## 5. Perspective-Aware Social Memory

### 5.1 The Perspective Confusion Problem

The most common failure in social reasoning is **perspective confusion**:
the system answers based on what the viewer (or the system) knows, rather
than what a specific character knows. Example:

> Q: "Does Alice know that Bob stole the key?"
> Correct: No — Alice was not in the room when Bob took it.
> Common error: Yes — the system saw Bob take it and projects that knowledge onto Alice.

To prevent this, the controller maintains a **perspective thread** for each
important character — a structured record of what that character has
observed, heard, and can reasonably infer.

### 5.2 Perspective Thread Schema

```python
@dataclass
class PerspectiveThread:
    entity_id: str
    entity_name: str

    observed_events: List[str]     # event node IDs this character witnessed
    heard_dialogue: List[str]      # dialogue event IDs where this character was listener/speaker
    inferred_beliefs: List[SocialStateEntry]  # what the controller infers this character believes
    goals: List[str]               # inferred goals/intentions
    knowledge_boundary: str        # summary of what this character does NOT know

    last_updated: float            # timestamp of most recent update
    update_history: List[Dict]     # log of belief changes with evidence
```

### 5.3 Social-State Entry Schema

Each inferred mental state is stored with full provenance:

```python
@dataclass
class SocialStateEntry:
    entry_id: str
    entities: List[str]           # who is involved
    state_type: str               # "belief" | "intention" | "trust" | "suspicion" | "commitment" | "deception"
    description: str              # "face_3 believes face_7 is hiding the envelope"
    timestamp: Tuple[float, float]
    confidence: float             # 0–1

    # Provenance
    provenance: str               # "directly_observed" | "inferred_from_dialogue" | "inferred_from_behavior" | "inferred_from_absence"
    supporting_evidence: List[str] # event/episode node IDs that support this inference
    contradicting_evidence: List[str]

    # Uncertainty
    alternative_hypotheses: List[str]  # competing interpretations if ambiguous
    is_active: bool               # false if superseded by later state
```

**Key separation:**
- **Observed facts** (`provenance="directly_observed"`): character was
  present and the event is grounded in observer output.
- **Inferred mental states** (`provenance="inferred_from_*"`): the
  controller's hypothesis about what a character believes, wants, or intends.
  Always accompanied by `supporting_evidence` and `confidence`.
- **Evidence provenance**: every social-state entry links back to the
  specific event nodes that justify it. The controller can trace any claim
  to its visual source.

### 5.4 Perspective Thread Construction

The controller builds perspective threads during memory construction:

1. For each event, check the `witnesses` field to determine which characters
   were present.
2. Add the event to the `observed_events` of each witness.
3. For dialogue events, add to `heard_dialogue` of speaker and listeners.
4. After each episode, run a belief-update pass: for each character, infer
   what they likely believe based on what they've observed and heard.
5. Flag any case where a character's perspective diverges from ground truth
   (e.g., character was absent during a key event).

### 5.5 Perspective-Aware Retrieval

When the controller encounters a question like "Does A know X?", it:
1. Retrieves A's perspective thread, not the global memory.
2. Checks whether the relevant event is in A's `observed_events`.
3. If not directly observed, checks whether A could have learned through
   dialogue (`heard_dialogue`) or inference.
4. Returns the answer with explicit evidence: "A does not know X because
   A was not present at event E-7 [02:15] and no subsequent dialogue
   informed A."

---

## 6. Skill Bank Design

### 6.1 Skills as Social Inference Operators

Skills are not generic retrieval routines. They are **reusable social
inference operators** — each encodes a specific reasoning pattern over the
memory graph and perspective threads.

### 6.2 Core Social Skills

Each skill has a compact representation:

| Skill | Purpose | Trigger | Required Inputs | Execution Pattern | Expected Output | Failure Modes |
|-------|---------|---------|-----------------|-------------------|-----------------|---------------|
| `who_saw_what` | Determine which characters witnessed a specific event | Question asks "did X see/know about Y" | event description, entity list | Locate event → check `witnesses` field → cross-reference perspective threads | List of witnesses + non-witnesses with evidence | Event not found; ambiguous presence |
| `infer_belief_update` | Track how a character's belief changed after an event | Question about belief change | entity_id, event_id | Get perspective thread → find belief state before event → check if event is in observed_events → infer post-event belief | Before/after belief pair with evidence | Character absent; belief not inferable |
| `track_commitment` | Track whether a character kept or broke a promise/plan | Question about promise fulfillment | commitment event, entity_id | Locate commitment event → find subsequent actions → compare against commitment | Kept/broken with timeline | Commitment ambiguous; outcome unclear |
| `detect_intention_shift` | Identify when a character's goal changed | Question about motivation | entity_id, time range | Query perspective thread goals across time → identify change points → retrieve causal events | Intention timeline with pivot events | Subtle shift; insufficient evidence |
| `resolve_conflicting_testimony` | Reconcile contradictory claims by different characters | Multiple characters give different accounts | entity_ids, claim descriptions | Retrieve each character's perspective → identify what each could have observed → determine which account is consistent with evidence | Reconciliation with credibility scores | Both accounts plausible; no resolution |
| `identify_hidden_action` | Detect an action performed when others weren't present | Question about concealed behavior | actor_id, time range | Find events where actor is sole witness → check if any observer saw it → check if information leaked later | Hidden action + who does/doesn't know | Action not captured by observer |
| `track_relationship_change` | Trace relationship evolution over time | Question about relationship trajectory | entity_id pair | Query arc-level memory → retrieve constituent episodes → extract trust/stance at each point | Relationship timeline with evidence | Sparse interactions; no clear arc |
| `social_causal_attribution` | Determine why a social state changed | "Why did A become suspicious of B?" | social state change, entities | Find the target social-state entry → trace `supporting_evidence` → follow causal edges → build causal chain | Causal chain from trigger to outcome | Multiple causes; weak evidence |
| `deception_hypothesis_check` | Test whether a character is deceiving another | Suspected deception | deceiver_id, target_id | Compare deceiver's perspective thread with their public statements → check for information asymmetry → find hidden actions | Deception confirmed/denied with evidence | Deception too subtle; missing data |
| `multi_perspective_reconciliation` | Compare how multiple characters perceive the same event | Question requires understanding different viewpoints | event_id, entity_ids | Retrieve perspective threads for all entities → extract each character's view of the event → identify divergences | Per-character view table with divergences | Perspectives too similar to distinguish |

### 6.3 Skill Representation

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

### 6.4 Skill Composition

Skills compose into multi-step strategies. The controller learns which
compositions work through the dual-thread reward signal (§8).

```
Example: "Does Alice know that Bob stole the key?"

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

### 6.5 Skill DAG

Skills form a directed acyclic graph tracking composition and performance:

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

---

## 7. Failure Reflection and Self-Evolution

### 7.1 Failure Taxonomy

When the controller produces a wrong answer, the failure is classified into
one of these types. Each type triggers a specific update to the system.

| Failure Type | Description | Example |
|-------------|-------------|---------|
| **Missed evidence** | Relevant memory node exists but was not retrieved | Question about a conversation; the retrieval query missed the right episode |
| **Wrong temporal linkage** | Events connected incorrectly in the reasoning chain | Cause and effect reversed; wrong temporal ordering |
| **Wrong entity grounding** | Confused two characters or misidentified an entity | "The woman" resolved to wrong face_id |
| **Perspective confusion** | Answered based on global truth instead of character's local view | System knows Bob stole the key; incorrectly claims Alice also knows |
| **False-belief reasoning error** | Failed to model that a character holds an incorrect belief | Character was told a lie; system treated the lie as truth for that character |
| **Overconfident inference** | Drew a strong conclusion from weak or ambiguous evidence | Single facial expression interpreted as definitive proof of deception |
| **Insufficient evidence, forced answer** | Not enough evidence existed but system answered anyway instead of abstaining | Memory graph lacked the relevant segment; system hallucinated |

### 7.2 Failure → Update Mapping

Each failure type triggers a targeted update. The controller does not
apply generic "try harder" fixes — it identifies the structural cause and
patches the specific component.

| Failure Type | Update Target | Specific Action |
|-------------|---------------|-----------------|
| Missed evidence | **Retrieval strategy** | Add alternative query patterns to the skill; increase retrieval breadth for this question type |
| Wrong temporal linkage | **Memory schema** | Strengthen `precedes`/`causes` edges; add temporal verification step to skill protocol |
| Wrong entity grounding | **Entity resolver** | Add disambiguation step; refine entity matching thresholds |
| Perspective confusion | **Perspective thread** | Add explicit "check perspective" step to social skills; create new `check_character_access` skill if none exists |
| False-belief reasoning error | **Skill refinement** | Refine `infer_belief_update` to handle lie propagation; add false-belief reasoning substep |
| Overconfident inference | **Confidence calibration** | Lower confidence thresholds; add "require N supporting events" constraint to the skill |
| Insufficient evidence | **Verifier rule** | Strengthen evidence sufficiency checker; add abstention option when confidence < threshold |

### 7.3 Skill Evolution Mechanisms

| Mechanism | Trigger | Action |
|-----------|---------|--------|
| **Reinforce** | Skill used in correct answer | Bump success rate, update average evidence quality |
| **Refine** | Skill used in wrong answer with identified fix | Modify execution steps; add/remove preconditions |
| **Split** | Skill has high variance (works for some subtypes, fails for others) | Create two specialized skills from the original |
| **Merge** | Two skills have >80% step overlap and similar performance | Combine into single skill with broader trigger conditions |
| **Craft new** | Failure type has no matching skill | 8B controller generates a new skill from the failure analysis + successful counter-example |
| **Retire** | Skill has <20% success rate over 20+ invocations | Remove from active bank; archive for reference |

### 7.4 Evolution Loop

```
For each evaluation batch:
  1. Run questions through controller (with skill bank)
  2. Score against ground truth
  3. For correct answers:
     → Reinforce skills used
     → Extract new skill patterns from novel compositions
  4. For wrong answers:
     → Classify failure type (§7.1)
     → Apply targeted update (§7.2)
     → If no existing skill addresses the failure pattern → craft new skill
  5. Every K batches:
     → Run bank maintenance (merge, split, retire)
     → Log skill bank statistics for analysis
```

---

## 8. Dual-Thread Reward for Controller Training

### 8.1 Mechanism

Every question runs through two parallel threads: one with skill bank access,
one without. The comparison provides a per-question, per-skill training signal.

| Thread | Skill Bank | Memory | Reasoner |
|--------|-----------|--------|----------|
| **A (with skills)** | Full access | Full graph | Frozen 72B |
| **B (without skills)** | Disabled | Full graph | Frozen 72B |

### 8.2 Outcome Reward

| A Correct? | B Correct? | `r_outcome` | Meaning |
|-----------|-----------|-------------|---------|
| Yes | No | **+1.0** | Skills were the deciding factor |
| Yes | Yes | **+0.2** | Skills at least didn't hurt |
| No | No | **-0.3** | Skills failed to help |
| No | Yes | **-1.0** | Skills actively damaged reasoning |

### 8.3 Step-Level Reward

Each reasoning turn receives dense reward:

| Signal | Value | Condition |
|--------|-------|-----------|
| `r_evidence` | **+0.1** | Retrieved a new, relevant memory node |
| `r_grounding` | **+0.15** | Grounded a question entity in the graph |
| `r_progress` | **+0.2** | Closed an identified evidence gap |
| `r_novel_info` | **+0.1** | Found non-redundant information |
| `p_turn_cost` | **-0.05** | Per-turn fixed cost (encourages efficiency) |
| `p_redundant` | **-0.10** | Re-retrieved already-known information |
| `p_irrelevant` | **-0.15** | Retrieved low-relevance evidence |
| `p_wrong_skill` | **-0.20** | Invoked a skill whose preconditions were unmet |
| `p_hallucination` | **-0.30** | Introduced a claim not grounded in any memory node |

### 8.4 Composite Reward

```
R = 0.35 × r_outcome
  + 0.25 × (step_total_A - step_total_B)   # relative step quality
  + 0.20 × (evidence_quality_A - evidence_quality_B)
  + 0.10 × efficiency_bonus                 # fewer turns for same result
  + 0.10 × step_total_A                     # absolute step quality
```

### 8.5 Training via GRPO

The composite reward feeds into Group Relative Policy Optimization over the
8B controller. The controller learns:
- **When** to invoke skills vs. raw retrieval
- **Which** skills to select for each question type
- **How** to compose multi-step skill chains
- **When** to stop gathering evidence and answer

---

## 9. Training Scope

### 9.1 What Is Trained

**Only the 8B controller is trained.** All other components are frozen.

| Component | Trainable? | Optimization Target |
|-----------|-----------|---------------------|
| **8B Controller** | **Yes** | Memory management, skill selection, retrieval planning, evidence sufficiency, verification, reflection |
| Observer-A (72B) | No | Frozen perception tool |
| Observer-B (72B) | No | Frozen perception tool |
| Reasoner (72B) | No | Frozen answer generator |
| Embedders | No | Frozen retrieval index |
| Skill Bank | Evolves (not gradient-trained) | Updated by controller's reflection logic |

### 9.2 Optimization Target

The goal is **not** to improve raw visual perception — the frozen 72B
observers already extract high-quality structured facts. The goal is to
improve **reasoning orchestration over long videos**:

- Better memory control: which events to promote to episodes, which episodes
  to link into arcs, when to update perspective threads.
- Better skill use: selecting the right social inference operator for the
  question type, composing multi-step strategies, avoiding unnecessary
  retrieval.
- Better long-horizon efficiency: answering in fewer turns while maintaining
  evidence quality.

### 9.3 Training Data

Training uses the dual-thread reward signal (§8) over benchmark questions.
No human annotations beyond the existing QA ground truth are required.

| Training Phase | Data Source | Signal |
|---------------|------------|--------|
| **Cold start** | Video-Holmes + MA-EgoQA (question-answer pairs) | Outcome reward only (no step-level until skills exist) |
| **Skill evolution** | Same benchmarks, iterative | Dual-thread reward with step-level scoring |
| **Cross-benchmark transfer** | Train on one, evaluate on another | Skill bank portability signal |

### 9.4 What the 8B Controller Learns (LoRA Adapters)

| Adapter | Input | Output | When Active |
|---------|-------|--------|-------------|
| `memory_builder` | Observer JSON | Memory graph update commands | Offline memory construction |
| `planner` | Question + memory state | Skill selection + retrieval plan | Online, per question |
| `verifier` | Answer + evidence chain + memory | Accept / reject / retry decision | Online, post-answer |
| `reflector` | Failed trace + ground truth | Failure classification + skill update | Post-evaluation |

---

## 10. Evaluation Plan

### 10.1 Benchmarks

| Benchmark | Focus | Why It Tests Our System |
|-----------|-------|------------------------|
| **Video-Holmes** | Deep causal/temporal/social reasoning (short films) | Reasoning depth over social dynamics |
| **MA-EgoQA** | Multi-agent social interaction, ToM, task coordination | Perspective-aware reasoning across multiple agents |
| **EgoLife** | Long-term ego-centric daily life | Long-range memory + social tracking at scale |
| **LongVidSearch** | Multi-hop evidence retrieval | Evidence chaining quality |

### 10.2 Metrics

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Answer accuracy** | Final answer correctness | Standard MCQ / open-ended eval |
| **Evidence grounding quality** | Are cited timestamps and facts correct? | Compare cited evidence against ground truth |
| **Multi-hop retrieval quality** | Does the system chain evidence across distant segments? | Count correct cross-segment links in evidence chain |
| **Social consistency** | Are social-state inferences consistent across time? | Check for contradictions in perspective threads |
| **Memory efficiency** | How compact is the memory graph relative to video length? | Nodes per minute of video; retrieval precision |
| **Tool call count** | How many frozen-model calls per question? | Count observer/reasoner invocations |
| **Skill reuse rate** | Are skills being reused across questions? | Unique skills / total skill invocations |
| **Skill refinement rate** | How often does reflection improve skills? | Track success rate before/after refinement |
| **Long-range dependency handling** | Can the system answer questions requiring evidence from >5 min apart? | Stratify accuracy by temporal span of required evidence |

### 10.3 Ablation Design

| Variant | What's Removed | What It Tests |
|---------|---------------|---------------|
| **No social-state memory** | Remove all social-state entries | Value of tracking beliefs/intentions |
| **No perspective threads** | Remove per-character perspective tracking | Value of perspective-aware reasoning |
| **No hierarchical memory** | Flat event-level only (no episodes/arcs) | Value of multi-timescale organization |
| **No uncertainty fields** | Remove confidence, alternative hypotheses, provenance from social states | Value of calibrated inference |
| **No skill evolution** | Freeze skill bank after cold start | Value of continual improvement |
| **Static skill bank** | Use hand-written skills only, no learning | Value of learned vs. designed skills |
| **No 8B controller** | Direct 72B QA on raw video (no memory, no skills) | Full system contribution vs. raw VLM |
| **8B without skills** | Controller manages memory but has no skill bank | Value of skill bank specifically |
| **Single observer** | One 72B observer instead of two specialized | Value of observer specialization |
| **7B observer** | Replace 72B observers with 7B | Observer capacity vs. controller quality |

---

## 11. Expected Contributions

1. **A trainable 8B controller for long-video social reasoning** that
   manages hierarchical memory, perspective-specific social-state threads,
   and a self-evolving bank of social inference skills — while using frozen
   large VLMs only as perception and answer-generation tools.

2. **Perspective-aware social memory** with per-character threads that
   separate observed facts from inferred mental states, with explicit
   evidence provenance and uncertainty, preventing the most common failure
   mode in theory-of-mind reasoning.

3. **Social inference skills as reusable operators** — not retrieval
   routines but structured reasoning patterns (`who_saw_what`,
   `infer_belief_update`, `deception_hypothesis_check`, etc.) that compose
   into multi-step strategies and evolve through failure-driven reflection.

4. **Hierarchical memory over event, episode, and arc timescales** enabling
   efficient retrieval at the appropriate granularity for each question
   type, from "what happened at 2:15?" to "how did the alliance evolve?"

5. **Dual-thread reward framework** that produces per-question, per-skill
   training signal by comparing skill-augmented and skill-free reasoning
   on identical inputs.

---

## 12. Related Work

| Work | Relationship |
|------|-------------|
| **M3-Agent** (M3-Bench) | Memory graph for video QA. We add hierarchical levels, perspective threads, and trainable skill management. |
| **SCALAR** (arXiv:2603.09036) | LLM-proposed symbolic skills. We extend to social inference operators with evidence grounding and self-evolution. |
| **X-Skill** | Cross-embodiment skill discovery. We adapt skill decomposition from robotics to social video reasoning. |
| **WorldMM** (arXiv:2512.02425) | World model for multimodal reasoning. We add perspective-aware social state and trainable orchestration. |
| **Video-Holmes** (arXiv:2505.21374) | Deep reasoning benchmark. Our primary evaluation target. |
| **Multi-hop reasoning** (arXiv:2502.12442) | Multi-hop retrieval. Our planner implements skill-guided multi-hop with evidence sufficiency checking. |
| **arXiv:2603.24558** | Direct comparison. We add hierarchical memory, perspective threads, skill evolution, and evidence grounding. |
