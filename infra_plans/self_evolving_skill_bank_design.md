# Self-Evolving Visual Skill Bank for Evidence-Grounded Agentic Memory in Long Videos

> **Location:** `Video_Skills/infra_plans/self_evolving_skill_bank_design.md`
>
> **Key Words:** Self-evolving Visual Skill Bank, Agentic Memory for Video,
> Social / Spatial Reasoning for Long-term Videos, Continual Learning, Multi-hop Reasoning
>
> **Date:** 2025-04-16

---

## 0. Problem Statement

We seek to build a **self-evolving multimodal agent** for long-term video
reasoning, where answers to socially complex questions require multi-hop
retrieval over long-range visual, audio, and interaction memories. Unlike
standard long-video QA, our goal is not only to answer correctly, but to
support each conclusion with an **explicit chain of retrieved evidence**.

This requires an agent that can:

1. **Manage and update structured visual memory** — textual memory loses key
   spatial, appearance, and relational information that only visual
   representations can preserve.
2. **Perform multi-hop retrieval planning** over temporally distributed
   evidence — a single retrieval step is insufficient for questions that
   require chaining facts across distant video segments.
3. **Continually improve** by extracting and refining reusable
   visual-reasoning skills from prior rollouts — starting from atomized
   skills (memory query, simple VQA, grounding) and aggregating them into
   complex, reusable reasoning strategies.
4. **Learn from failure** — by incorporating failure reflection, the agent
   transforms unsuccessful reasoning traces into new skills, better
   retrieval strategies, and improved memory usage, enabling continual
   adaptation to increasingly complex social reasoning scenarios.

---

## 1. Architecture Overview

### 1.1 Design Philosophy: Separate Observer from Reasoner

The core architectural insight is **separating visual observation from
abstract reasoning** via an explicit context interface:

- **Observers (2× 72B VLMs):** Process raw video frames and produce
  structured, explicit context representations. They see pixels; they output
  structured facts.
- **Orchestrator / Skill Manager (8B LM):** Manages memory, decomposes
  skills, crafts new skills from reflection, plans multi-hop reasoning, and
  decides when to invoke Observers. Never sees raw pixels.
- **Reasoner (72B VLM):** Receives curated explicit context (structured
  memory + retrieved evidence + skill protocols + selected keyframes) and
  produces evidence-grounded answers. Sees only a small number of curated
  frames, not the full video.

This separation means the Reasoner operates on a **compact, information-dense
representation** rather than thousands of raw frames, dramatically reducing
hallucination and improving faithfulness.

### 1.2 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE PHASE                                      │
│              (Observers + 8B Orchestrator, runs once per video)               │
│                                                                              │
│  Video ──► Frame Sampler ──┬──► Observer-A (72B-α)                           │
│                            │     Social / Dialogue / ToM extraction          │
│                            │     → Social-State Memory nodes                 │
│                            │     → Entity profiles, relationships            │
│                            │                                                 │
│                            └──► Observer-B (72B-β)                           │
│                                  Spatial / Object / Trajectory extraction    │
│                                  → Spatial-State Memory nodes                │
│                                  → Visual Memory (keyframes, regions, layout)│
│                                  → Episodic Memory (events, clips)           │
│                                                                              │
│  Observer outputs ──► 8B Orchestrator ──┬──► Agentic Memory Graph            │
│                                         │    (episodic + semantic + visual    │
│                                         │     + social-state + spatial-state) │
│                                         │                                    │
│                                         └──► Visual Skill Bank               │
│                                              (atomized + composed skills)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE PHASE                                       │
│                (8B Orchestrator + Reasoner 72B, per question)                 │
│                                                                              │
│  Question ──► 8B Orchestrator ──┬──► Planner (multi-hop retrieval plan)      │
│                                  │                                           │
│   ┌──────────────────────────────┘                                           │
│   │  For each reasoning turn:                                                │
│   │    ├─ Select atomized skill(s) from Skill Bank                           │
│   │    ├─ Execute skill: query memory graph, ground entities, etc.           │
│   │    ├─ Accumulate evidence chain                                          │
│   │    ├─ Decide: terminate / gather more / invoke Observer on-demand        │
│   │    └─ Verify: is current evidence sufficient?                            │
│   │                                                                          │
│   └──► Evidence-First Prompt Composer                                        │
│          │  Curated context:                                                 │
│          │    • Retrieved memory nodes with timestamps                       │
│          │    • Skill protocols for reasoning scaffold                       │
│          │    • Selected keyframes (3-8 frames)                              │
│          │    • Explicit evidence chain                                      │
│          │                                                                   │
│          └──► Reasoner (72B) ──► Answer + Evidence + Timestamps              │
│                    │                                                         │
│                    └──► 8B Verifier ──► Accept / Retry / Reflect             │
│                              │                                               │
│                              └──► Skill Evolution (on failure)               │
│                                   • Failure analysis → new skills            │
│                                   • Refine existing skill protocols          │
│                                   • Update retrieval strategies              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Why Two 72B Observers?

Social and spatial reasoning require fundamentally different visual
attention patterns. A single model trying to do both tends to compromise:

| Observer | Focus | Output Memory Types | Why Separate |
|----------|-------|---------------------|--------------|
| **Observer-A (Social)** | Faces, expressions, dialogue, gestures, gaze direction, body language, theory-of-mind cues | Social-State, Episodic (dialogue), Entity profiles | Social reasoning requires fine-grained facial/gestural analysis and tracking who-knows-what across the video |
| **Observer-B (Spatial)** | Object positions, scene layout, trajectories, spatial relations, object states | Spatial-State, Visual Memory, Episodic (actions/events) | Spatial reasoning requires tracking object permanence, viewpoint changes, and layout evolution |

Both Observers share the same frame sampling schedule but apply different
prompting strategies and produce complementary structured outputs that are
fused into the unified memory graph by the 8B Orchestrator.

**Practical benefit:** Each Observer can run on a separate 4×A100 node in
parallel, halving the wall-clock time for the offline phase.

---

## 2. Agentic Memory Graph

### 2.1 Memory Types

The memory system comprises five interconnected memory types, stored as
nodes in a unified graph structure:

| Memory Type | Content | Node Schema | Source |
|-------------|---------|-------------|--------|
| **Episodic** | Events, clips, timestamps, character and dialogue segments | `{text, clip_id, timestamp, entity_ids, frames, audio_transcript}` | Both Observers |
| **Semantic** | Long-term knowledge, character attributes, relationship summaries, behavioral patterns | `{text, entity_ids, confidence, support_count, source_episodes}` | 8B distillation from episodic |
| **Visual** | Keyframes, regions-of-interest, appearances, spatial layouts, trajectories | `{frame_path, bbox, appearance_embedding, layout_description, trajectory}` | Observer-B |
| **Social-State** | Who knows what, who believes what, trust/stance dynamics, deception cues | `{entity_id, belief_state, knowledge_set, stance_toward, hiding/probing/avoiding}` | Observer-A |
| **Spatial-State** | Object locations, movement trajectories, visibility (who can see what), scene layout changes | `{entity_or_object_id, position, visibility_from, scene_region, movement_vector}` | Observer-B |

### 2.2 Graph Structure

**Yes, we use a graph.** The memory is represented as a heterogeneous
knowledge graph with typed nodes and typed edges:

```
                    ┌─────────────┐
                    │  Entity     │ (person, object)
                    │  Node       │
                    └──┬──┬──┬───┘
             has_appearance │  │  participates_in
                   │       │  │
            ┌──────┘   trusts/  └──────────────┐
            ▼          suspects                 ▼
     ┌─────────────┐    │              ┌─────────────────┐
     │  Visual     │    ▼              │  Episodic       │
     │  Memory     │  ┌──────────┐    │  Memory         │
     │  Node       │  │ Social-  │    │  Node           │
     └─────────────┘  │ State    │    └────────┬────────┘
                      │ Node     │             │ supports
                      └──────────┘             ▼
                                       ┌─────────────────┐
              ┌─────────────┐          │  Semantic        │
              │  Spatial-   │ near/    │  Memory          │
              │  State      │─────────►│  Node            │
              │  Node       │ seen_by  └─────────────────┘
              └─────────────┘
```

#### Node Types
- **Entity nodes:** Persons and objects with stable identities across the
  video (via face/voice/appearance embedding matching)
- **Memory nodes:** Episodic, semantic, visual, social-state, spatial-state
  (as defined above)

#### Edge Types
| Edge Type | From → To | Meaning |
|-----------|-----------|---------|
| `participates_in` | Entity → Episodic | Character appears in this event |
| `has_appearance` | Entity → Visual | Visual appearance record |
| `has_social_state` | Entity → Social-State | Current belief/stance state |
| `has_spatial_state` | Entity → Spatial-State | Current location/visibility |
| `supports` | Episodic → Semantic | Evidence for a semantic conclusion |
| `contradicts` | Episodic → Semantic | Counter-evidence |
| `causes` | Episodic → Episodic | Causal link between events |
| `precedes` | Episodic → Episodic | Temporal ordering |
| `near` | Entity → Entity | Spatial proximity at a time point |
| `trusts` / `suspects` | Entity → Entity | Social relationship |
| `sees` / `cannot_see` | Entity → Entity/Object | Visibility relation |
| `co_occurs` | Memory → Memory | Same temporal window |

#### Why a Graph?

1. **Multi-hop retrieval becomes graph traversal:** "What does Person A
   believe about Person B's location?" requires traversing
   `A → has_social_state → belief → about → B → has_spatial_state`.
2. **Relationship queries are natural:** "Who trusts whom?" is a direct
   edge query, not an embedding similarity search.
3. **Temporal chains are first-class:** `precedes` and `causes` edges
   support temporal reasoning without re-deriving order from timestamps.
4. **Skill-memory alignment:** Skills can specify which graph traversal
   patterns they implement (e.g., a "track_trust_change" skill traverses
   `trusts` edges across time).

### 2.3 Graph Implementation

```python
@dataclass
class MemoryNode:
    node_id: str
    node_type: str  # "episodic" | "semantic" | "visual" | "social_state" | "spatial_state" | "entity"
    content: Dict[str, Any]  # type-specific structured content
    timestamp: Optional[Tuple[float, float]]  # (start_sec, end_sec)
    embedding: Optional[np.ndarray]  # for similarity search
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryEdge:
    edge_type: str  # from the edge type table
    source_id: str
    target_id: str
    weight: float = 1.0
    timestamp: Optional[float] = None  # when this relation was observed
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgenticMemoryGraph:
    """Heterogeneous knowledge graph over video memories."""

    nodes: Dict[str, MemoryNode]
    edges: Dict[Tuple[str, str], List[MemoryEdge]]  # (src, tgt) → edges
    entity_index: Dict[str, List[str]]  # entity_id → connected node_ids
    temporal_index: SortedList  # nodes sorted by timestamp
    embedding_store: MemoryStore  # for similarity-based retrieval

    # --- Core Operations ---
    def add_node(self, node: MemoryNode) -> None: ...
    def add_edge(self, edge: MemoryEdge) -> None: ...
    def search_by_embedding(self, query: str, top_k: int, node_type: Optional[str]) -> List[MemoryNode]: ...
    def traverse(self, start_id: str, edge_types: List[str], max_hops: int) -> List[List[MemoryNode]]: ...
    def get_temporal_window(self, t_start: float, t_end: float) -> List[MemoryNode]: ...
    def get_entity_subgraph(self, entity_id: str) -> SubGraph: ...

    # --- Update Operations ---
    def reinforce(self, node_id: str, delta: float) -> None: ...  # confidence update
    def decay(self, threshold: float) -> List[str]: ...  # remove low-confidence nodes
    def merge_entities(self, id_a: str, id_b: str) -> None: ...  # union-find merge
```

### 2.4 Graph Construction Pipeline

```
Video (N frames)
  │
  ├─ 1. Adaptive Frame Sampling
  │     • Base rate: 1 fps (short), 0.2 fps (long)
  │     • + scene-change keyframes (pixel/feature delta)
  │     • + subtitle-aligned frames
  │
  ├─ 2. Parallel Observer Processing
  │     ┌─────────────────────────────────────────────────┐
  │     │ Observer-A (Social, 72B-α)                      │
  │     │   Per 30s clip batch:                           │
  │     │   • Face detection + recognition → entity nodes │
  │     │   • Emotion/expression classification           │
  │     │   • Dialogue transcription + speaker ID         │
  │     │   • Gaze direction, body language               │
  │     │   • Social interaction classification           │
  │     │   • Theory-of-mind state estimation             │
  │     │   Output: social observations + entity profiles │
  │     └─────────────────────────────────────────────────┘
  │     ┌─────────────────────────────────────────────────┐
  │     │ Observer-B (Spatial, 72B-β)                     │
  │     │   Per 30s clip batch:                           │
  │     │   • Object detection + tracking                 │
  │     │   • Scene layout description                    │
  │     │   • Spatial relation extraction                 │
  │     │   • Trajectory estimation                       │
  │     │   • Action recognition                          │
  │     │   • Event boundary detection                    │
  │     │   Output: spatial observations + event timeline │
  │     └─────────────────────────────────────────────────┘
  │
  ├─ 3. 8B Orchestrator: Memory Graph Construction
  │     • Fuse Observer-A and Observer-B outputs
  │     • Entity resolution across clips (face/voice union-find)
  │     • Construct episodic nodes from fused observations
  │     • Infer and add edges (temporal, causal, spatial, social)
  │     • Build embedding index for similarity search
  │
  ├─ 4. 8B Orchestrator: Semantic Distillation (second pass)
  │     • Cluster related episodic nodes
  │     • Generate semantic summaries (character arcs, relationship
  │       evolution, recurring patterns)
  │     • Add semantic nodes with `supports` edges to episodic evidence
  │
  └─ 5. 8B Orchestrator: Social/Spatial State Tracking
        • For each entity, construct time-series of social states
          (belief updates, trust changes, knowledge gains)
        • For each entity/object, construct spatial state timeline
          (position changes, visibility changes, layout evolution)
        • Add `has_social_state` and `has_spatial_state` edges
```

### 2.5 Observer Output Format (Explicit Context)

Rather than passing raw embeddings or free-text paragraphs, Observers
produce **structured JSON** that the 8B Orchestrator can directly consume
and index:

```json
// Observer-A output for a 30s clip
{
  "clip_id": "clip_042",
  "timestamp": [1260.0, 1290.0],
  "entities": [
    {
      "entity_id": "face_3",
      "name_hint": "woman in red",
      "emotion": "anxious",
      "gaze_target": "face_7",
      "body_language": "crossed arms, leaning away"
    }
  ],
  "dialogues": [
    {
      "speaker": "face_3",
      "text": "I don't think he's telling the truth.",
      "listener": "face_5",
      "tone": "suspicious"
    }
  ],
  "social_states": [
    {
      "entity": "face_3",
      "believes": "face_7 is hiding something",
      "trust_toward_face_7": 0.2,
      "stance": "probing"
    }
  ],
  "interactions": [
    {
      "type": "confrontation",
      "participants": ["face_3", "face_7"],
      "description": "face_3 questions face_7 about the missing object"
    }
  ]
}
```

```json
// Observer-B output for the same clip
{
  "clip_id": "clip_042",
  "timestamp": [1260.0, 1290.0],
  "scene": {
    "location": "living room",
    "layout": "couch center, table left, door back-right",
    "lighting": "dim evening"
  },
  "objects": [
    {
      "object_id": "obj_12",
      "label": "envelope",
      "position": "under couch cushion",
      "visibility": {"face_3": false, "face_7": true},
      "state": "hidden"
    }
  ],
  "spatial_relations": [
    {"subject": "face_3", "relation": "facing", "object": "face_7"},
    {"subject": "face_7", "relation": "near", "object": "obj_12"}
  ],
  "events": [
    {
      "type": "action",
      "agent": "face_7",
      "action": "subtly pushes envelope further under cushion",
      "timestamp": 1275.3
    }
  ],
  "keyframes": ["frame_1260.jpg", "frame_1275.jpg", "frame_1290.jpg"]
}
```

This explicit context interface means:
- The 8B Orchestrator never needs to process raw pixels
- The Reasoner receives precisely the information it needs
- Memory nodes are constructed from structured fields, not parsed from free text
- The graph edges can be inferred deterministically from the structured output

---

## 3. Visual Skill Bank

### 3.1 Skill Taxonomy

Skills are organized in a hierarchy from **atomic** (single-step, reusable
primitives) to **composed** (multi-step reasoning strategies):

#### Level 0: Atomic Skills

These are the building blocks — each performs a single, well-defined
operation on the memory graph or visual input.

| Skill ID | Name | Input | Output | Category |
|----------|------|-------|--------|----------|
| `query_episodic` | Query Episodic Memory | text query | top-k episodic nodes | Memory |
| `query_semantic` | Query Semantic Memory | text query | top-k semantic nodes | Memory |
| `query_visual` | Query Visual Memory | text/image query | top-k visual nodes | Memory |
| `query_social_state` | Query Social State | entity_id, time | social state snapshot | Memory |
| `query_spatial_state` | Query Spatial State | entity_id/object_id, time | spatial state snapshot | Memory |
| `traverse_graph` | Graph Traversal | start_node, edge_types, hops | paths | Memory |
| `ground_entity` | Ground Entity | entity description | entity_id, evidence | Grounding |
| `ground_event` | Ground Event | event description | episodic node(s) | Grounding |
| `ground_dynamics` | Ground Dynamics | description of change | before/after states | Grounding |
| `locate_temporal` | Temporal Localization | event description | timestamp range | Grounding |
| `compare_states` | Compare States | entity, time_a, time_b | delta description | Reasoning |
| `basic_vqa` | Basic Visual QA | question, frame(s) | answer | Reasoning |
| `verify_claim` | Verify Claim | claim text, memory graph | supported/unsupported + evidence | Reasoning |
| `count_entities` | Count Entities | entity type, time range | count + list | Reasoning |

#### Level 1: Composed Skills (Multi-Step Strategies)

These combine 2-4 atomic skills into reusable multi-step patterns:

| Skill ID | Name | Composition | Use Case |
|----------|------|-------------|----------|
| `track_relationship` | Track Relationship Change | `ground_entity` × 2 → `query_social_state` → `compare_states` | "How did trust between A and B change?" |
| `trace_object` | Trace Object Movement | `ground_entity` → `query_spatial_state` (multi-time) → `traverse_graph(precedes)` | "Where did the envelope go?" |
| `reconstruct_timeline` | Reconstruct Event Timeline | `query_episodic` → `traverse_graph(precedes,causes)` → `locate_temporal` | "What happened between time X and Y?" |
| `infer_belief` | Infer Character Belief | `query_social_state` → `query_episodic` (evidence) → `verify_claim` | "Does A know that B took the key?" |
| `identify_deception` | Identify Deception | `query_social_state` → `query_spatial_state` → `compare_states` (what entity says vs. what is true) | "Is person X lying about Y?" |
| `multi_perspective` | Multi-Perspective Analysis | `query_social_state` × N entities → `compare_states` | "How do different characters view this event?" |

#### Level 2: Reasoning Strategies (Learned from Rollouts)

These are high-level plans that chain Level 0+1 skills and are
**discovered/refined through self-evolution:**

| Strategy ID | Name | Pattern | Evolved From |
|-------------|------|---------|--------------|
| `social_chain_reasoning` | Social Chain Reasoning | `ground_entity → track_relationship → infer_belief → verify_claim → answer` | Successful social QA rollouts |
| `spatial_tracking` | Spatial Object Tracking | `ground_entity → trace_object → reconstruct_timeline → verify_claim → answer` | Successful spatial QA rollouts |
| `temporal_causal` | Temporal Causal Analysis | `reconstruct_timeline → compare_states → identify cause → verify_claim → answer` | Video-Holmes TCI questions |
| `deception_detection` | Deception Detection Strategy | `identify_deception → multi_perspective → gather counter-evidence → answer` | Failed then refined on ToM questions |

### 3.2 Skill Schema (Extending COS-PLAY)

Each skill uses the existing `Skill` schema from
`skill_agents/stage3_mvp/schemas.py` with video-specific extensions:

```python
Skill(
    skill_id="track_relationship",
    name="Track Relationship Change Over Time",
    strategic_description=(
        "Track how the relationship (trust, stance, knowledge sharing) "
        "between two entities evolves across the video by querying "
        "social-state memory at multiple time points."
    ),
    tags=["SOCIAL", "TEMPORAL", "MULTI_HOP", "RELATIONSHIP"],
    protocol=Protocol(
        preconditions=[
            "two_entities_identified=true",
            "social_state_memory_populated=true",
        ],
        steps=[
            "1. Ground both entities in the memory graph",
            "2. Query social-state for entity-A's stance toward entity-B at key time points",
            "3. Query social-state for entity-B's stance toward entity-A at the same time points",
            "4. Compare states across time to identify change points",
            "5. Retrieve episodic evidence for each change point",
            "6. Synthesize: describe the relationship trajectory with evidence",
        ],
        success_criteria=[
            "relationship_trajectory_extracted=true",
            "evidence_grounded=true",
            "temporal_order_correct=true",
        ],
        abort_criteria=[
            "entity_not_found",
            "social_state_empty",
            "no_interaction_between_entities",
        ],
        expected_duration=4,
    ),
    contract=SkillEffectsContract(
        skill_id="track_relationship",
        eff_add={"relationship_trajectory_known", "evidence_collected"},
        eff_event={"social_state_queried", "temporal_comparison_done"},
    ),
)
```

### 3.3 Skill Graph for Management

**Yes, we also use a graph for skill management.** Skills form a directed
acyclic graph (DAG) based on composition relationships:

```
Level 2 Strategies
    │ composes
    ▼
Level 1 Composed Skills
    │ composes
    ▼
Level 0 Atomic Skills
```

The skill graph tracks:
- **Composition edges:** which atomic skills a composed skill uses
- **Performance metrics:** success rate, avg turns, evidence quality per
  skill, per benchmark
- **Usage patterns:** which skills tend to co-occur in successful rollouts
- **Failure modes:** common failure patterns per skill (for reflection)

```python
class SkillGraph:
    """DAG of skills with composition and performance tracking."""

    skills: Dict[str, Skill]  # skill_id → Skill
    composition_edges: Dict[str, List[str]]  # parent → children
    performance: Dict[str, SkillPerformance]  # per-skill metrics
    co_occurrence: Dict[Tuple[str, str], int]  # (skill_a, skill_b) → count in successful traces
    failure_modes: Dict[str, List[FailureRecord]]  # per-skill failure patterns

    def get_skill(self, skill_id: str) -> Skill: ...
    def get_composed_from(self, skill_id: str) -> List[Skill]: ...
    def suggest_composition(self, question_type: str) -> List[Skill]: ...
    def update_performance(self, skill_id: str, outcome: str, trace: ReasoningTrace) -> None: ...
    def find_similar_skills(self, description: str, top_k: int) -> List[Skill]: ...
```

---

## 4. Planner: Multi-Turn Reasoning Orchestration

### 4.1 Role of the Planner

The Planner (implemented as a component of the 8B Orchestrator) decides,
at each reasoning turn:

1. **Which atomized skill(s) to invoke** — based on the current evidence
   state and the question requirements.
2. **Whether to terminate** — the reasoning has gathered sufficient evidence
   and the answer is well-supported.
3. **Whether the current answer is satisfactory** — or if more evidence is
   needed, a different skill should be tried, or an Observer should be
   re-invoked for a specific clip.

### 4.2 Planner State Machine

```
                         ┌──────────────────┐
                         │   PLAN            │
              ┌─────────►│   Analyze question│
              │          │   Select skill(s) │
              │          └────────┬──────────┘
              │                   │
              │                   ▼
              │          ┌──────────────────┐
              │          │   EXECUTE         │
              │          │   Run selected    │
              │          │   skill(s)        │
              │          └────────┬──────────┘
              │                   │
              │                   ▼
              │          ┌──────────────────┐
              │          │   EVALUATE        │
      more    │          │   Check evidence  │
    evidence  │          │   sufficiency     │
    needed    │          └────────┬──────────┘
              │                   │
              │           ┌───────┴──────┐
              │           │              │
              │      insufficient   sufficient
              │           │              │
              └───────────┘              ▼
                                 ┌──────────────────┐
                                 │   ANSWER          │
                                 │   Compose evidence│
                                 │   + answer        │
                                 └──────────────────┘
```

### 4.3 Planner Prompt (8B Model)

```
You are a reasoning planner for video question answering. You have access
to a memory graph and a bank of visual reasoning skills.

Current state:
- Question: {question}
- Evidence collected so far: {evidence_chain}
- Skills used: {skills_used}
- Turns remaining: {max_turns - current_turn}

Available skills: {skill_bank_summary}

Decide your next action. Output JSON:
{
  "action": "execute_skill" | "answer" | "request_observer",
  "skill_id": "...",       // if action=execute_skill
  "skill_args": {...},     // arguments for the skill
  "reasoning": "...",      // why this action
  "confidence": 0.0-1.0,  // confidence that current evidence is sufficient
  "evidence_gaps": [...]   // what information is still missing
}
```

### 4.4 Evidence Sufficiency Checker

The Planner uses a lightweight evidence sufficiency check after each turn:

```python
@dataclass
class EvidenceSufficiency:
    is_sufficient: bool
    confidence: float
    missing_evidence: List[str]
    contradictions: List[str]
    temporal_coverage: float  # fraction of question's time range covered
    entity_coverage: float    # fraction of mentioned entities grounded
```

The 8B model evaluates sufficiency by checking:
- All entities mentioned in the question are grounded in memory
- The temporal span of the question is covered by retrieved evidence
- No contradictory evidence remains unresolved
- The evidence chain forms a complete reasoning path from question to answer

---

## 5. Evidence-First Answerer / Verifier

### 5.1 Prompt Structure for the Reasoner (72B)

The Reasoner receives a carefully composed prompt with explicit evidence:

```
╔═══════════════════════════════════════════════════════════════╗
║  SYSTEM                                                       ║
║  You are an evidence-grounded video analyst. Answer the        ║
║  question using ONLY the provided evidence. Every claim must   ║
║  cite a specific evidence entry by [E-N]. If insufficient     ║
║  evidence exists, say so rather than speculate.               ║
╠═══════════════════════════════════════════════════════════════╣
║  REASONING PROTOCOL                                           ║
║  (from skill bank — the strategy to follow)                   ║
║                                                               ║
║  Strategy: {strategy_name}                                    ║
║  Steps:                                                       ║
║    1. {step_1}                                                ║
║    2. {step_2}                                                ║
║    ...                                                        ║
╠═══════════════════════════════════════════════════════════════╣
║  EVIDENCE CHAIN  (retrieved by the planner)                   ║
║                                                               ║
║  [E-1] [Episodic] 00:12-00:24: Woman enters the room and     ║
║    notices the envelope on the table. She looks surprised.     ║
║    Entities: face_3 (woman), obj_12 (envelope)                ║
║                                                               ║
║  [E-2] [Social-State] face_3 at 00:30: Believes face_7 has   ║
║    hidden something. Trust toward face_7: 0.2. Stance:        ║
║    suspicious, probing.                                       ║
║                                                               ║
║  [E-3] [Spatial-State] obj_12 at 00:28: Position changed      ║
║    from "on table" to "under couch cushion". Moved by face_7.  ║
║    Visibility: face_3=false, face_7=true.                     ║
║                                                               ║
║  [E-4] [Episodic] 00:45-01:02: face_3 confronts face_7 about ║
║    the missing envelope. face_7 denies knowledge.              ║
║                                                               ║
║  [E-5] [Semantic] face_7 has a pattern of concealing objects   ║
║    from face_3 (3 prior instances). Confidence: 0.91.         ║
╠═══════════════════════════════════════════════════════════════╣
║  VISUAL FRAMES  (selected keyframes with annotations)         ║
║                                                               ║
║  [Frame @ 00:15] <image_1> — Woman notices envelope           ║
║  [Frame @ 00:28] <image_2> — Man pushes envelope under cushion║
║  [Frame @ 00:50] <image_3> — Confrontation scene              ║
╠═══════════════════════════════════════════════════════════════╣
║  QUESTION                                                     ║
║                                                               ║
║  Does the woman know that the man hid the envelope?           ║
║  Options: (A) Yes, she saw him. (B) No, she suspects but      ║
║  doesn't know. (C) No, she has no idea. (D) Yes, someone      ║
║  told her.                                                     ║
╠═══════════════════════════════════════════════════════════════╣
║  OUTPUT FORMAT                                                ║
║  <evidence_chain>                                             ║
║    Step 1: [cite E-N] reasoning...                            ║
║    Step 2: [cite E-N] reasoning...                            ║
║    ...                                                        ║
║  </evidence_chain>                                            ║
║  <answer>X</answer>                                           ║
║  <confidence>0.0-1.0</confidence>                              ║
╚═══════════════════════════════════════════════════════════════╝
```

### 5.2 Verifier (8B Model)

After the Reasoner produces an answer, the 8B Verifier checks:

| Check | Method | Action on Failure |
|-------|--------|-------------------|
| **Evidence grounding** | Every [E-N] citation in the reasoning chain is verified against the actual evidence entries | Flag ungrounded claims |
| **Temporal consistency** | Check that the reasoning chain respects the temporal order of cited evidence | Re-order or request more evidence |
| **Entity consistency** | Verify entity references match the memory graph | Correct entity IDs |
| **Answer-evidence alignment** | The final answer logically follows from the cited evidence chain | Re-prompt with stronger evidence requirements |
| **Contradiction check** | No two cited evidence entries contradict each other without acknowledgment | Request resolution |

If verification fails, the Verifier either:
- Requests additional evidence from the Planner (up to 2 retry iterations)
- Flags the answer as low-confidence with specific failure reasons

---

## 6. Self-Evolving Visual Skills

### 6.1 Skill Evolution Pipeline

The skill bank evolves through three mechanisms:

```
                    Successful Rollouts
                           │
                    ┌──────┴──────┐
                    │ Extract     │
                    │ skill from  │
                    │ trace       │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌──────────┐ ┌───────────────┐
     │ Reinforce    │ │ Compose  │ │ Generalize    │
     │ existing     │ │ new L1   │ │ across        │
     │ skill        │ │ skill    │ │ benchmarks    │
     └──────────────┘ └──────────┘ └───────────────┘


                    Failed Rollouts
                           │
                    ┌──────┴──────┐
                    │ Analyze     │
                    │ failure     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌──────────┐ ┌───────────────┐
     │ Refine skill │ │ Craft    │ │ Update        │
     │ protocol     │ │ new      │ │ retrieval     │
     │ (fix steps)  │ │ skill    │ │ strategy      │
     └──────────────┘ └──────────┘ └───────────────┘
```

### 6.2 Skill Extraction from Successful Rollouts

When a question is answered correctly with an evidence chain:

1. **Trace analysis (8B):** Identify the skill sequence that led to success
2. **Pattern detection:** Compare with existing skill traces — is this a
   known pattern or a new composition?
3. **If new composition:** Create a Level 1 composed skill from the trace
   - Extract the atomic skill sequence
   - Generalize: replace specific entity names with role placeholders
   - Generate protocol steps from the trace
   - Compute initial confidence from the evidence quality
4. **If existing skill:** Reinforce — bump instance count, update success
   rate, refine protocol if this trace was more efficient

### 6.3 Failure Reflection and Skill Crafting

When a question is answered incorrectly:

```python
@dataclass
class FailureAnalysis:
    question: str
    predicted_answer: str
    correct_answer: str
    reasoning_trace: List[SkillInvocation]
    failure_point: str  # which step in the trace went wrong
    failure_type: str   # "wrong_retrieval" | "wrong_reasoning" | "missing_evidence" |
                        #  "wrong_skill" | "insufficient_hops" | "hallucination"
    root_cause: str     # 8B model's analysis of why it failed
    suggested_fix: str  # what should have been done differently
```

**Failure-to-Skill Transformation (8B model):**

```
Failure Analysis
  │
  ├─ failure_type == "wrong_retrieval"
  │   → Craft new retrieval skill with better query decomposition
  │   → Update skill protocol: add "verify retrieval relevance" step
  │
  ├─ failure_type == "insufficient_hops"
  │   → Craft new multi-hop skill with more traversal steps
  │   → Increase max_hops in existing traversal skills
  │
  ├─ failure_type == "wrong_skill"
  │   → Update skill selection heuristics in the Planner
  │   → Add negative example to skill's abort_criteria
  │
  ├─ failure_type == "missing_evidence"
  │   → Add "request Observer re-analysis" step to the protocol
  │   → Create targeted observation skill for this evidence type
  │
  └─ failure_type == "hallucination"
      → Strengthen verification steps in the protocol
      → Add "check against memory before claiming" step
```

### 6.4 Cross-Benchmark Skill Transfer

Skills learned from one benchmark can transfer to others:

| Source Benchmark | Skill Learned | Transfer Target | Adaptation |
|------------------|---------------|-----------------|------------|
| Video-Holmes (TCI) | Temporal causal chain skill | MA-EgoQA temporal reasoning | Replace character analysis with ego-centric spatial analysis |
| MA-EgoQA (ToM) | Theory-of-mind inference skill | Video-Holmes (MHR) | Add hidden reasoning detection step |
| EgoLife | Long-term tracking skill | LongVidSearch | Reduce entity complexity, increase temporal range |
| LongVidSearch | Multi-hop evidence chain | Video-Holmes | Add visual grounding step |

The 8B Orchestrator manages skill transfer by:
1. Matching skill tags and protocol patterns across benchmarks
2. Adapting preconditions and success criteria for the target domain
3. Running lightweight validation (does the transferred skill improve
   accuracy on a held-out set?)

### 6.5 Continual Learning Loop

```
┌──────────────────────────────────────────────────────────┐
│                  CONTINUAL LEARNING CYCLE                  │
│                                                            │
│  1. Process new videos → build memory graphs               │
│     (Observers + 8B)                                       │
│                                                            │
│  2. Answer questions using current skill bank               │
│     (Planner + Reasoner)                                   │
│                                                            │
│  3. Evaluate answers against ground truth                   │
│                                                            │
│  4a. Successful: extract/reinforce skills                   │
│  4b. Failed: reflect → craft new skills / refine existing   │
│                                                            │
│  5. Run bank maintenance:                                   │
│     • Merge similar skills (embedding + contract overlap)   │
│     • Split overly general skills (high variance in perf)   │
│     • Retire consistently failing skills                    │
│     • Promote proto-skills with sufficient evidence         │
│                                                            │
│  6. (Optional) Fine-tune 8B Orchestrator with LoRA          │
│     on successful reasoning traces                          │
│                                                            │
│  7. Repeat from step 2 with evolved skill bank              │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Dual-Thread Reward for Skill Agent Training

### 7.1 Core Idea: Parallel With-Skill / Without-Skill Threads

To produce a grounded reward signal for whether skills actually help — and
*which* skills help — we run **two parallel reasoning threads** on every
question during training:

```
                         Same Question
                              │
                 ┌────────────┴────────────┐
                 ▼                          ▼
    ┌────────────────────┐     ┌────────────────────┐
    │  Thread A:          │     │  Thread B:          │
    │  WITH Skills        │     │  WITHOUT Skills     │
    │                     │     │                     │
    │  Planner selects    │     │  Same Planner,      │
    │  skills from bank,  │     │  same memory graph,  │
    │  follows protocols  │     │  but NO skill bank   │
    │  as reasoning       │     │  access — must       │
    │  scaffolds          │     │  reason from scratch  │
    │                     │     │  using raw memory     │
    │  → Answer_A         │     │  retrieval only       │
    │  → Evidence_A       │     │                     │
    │  → Trace_A          │     │  → Answer_B         │
    └────────┬───────────┘     │  → Evidence_B       │
             │                  │  → Trace_B          │
             │                  └────────┬───────────┘
             │                           │
             └──────────┬────────────────┘
                        ▼
              ┌──────────────────┐
              │  COMPARE          │
              │                   │
              │  Δ_correct:       │
              │    A correct,     │
              │    B wrong        │
              │    → skill helped │
              │                   │
              │  Δ_evidence:      │
              │    A's evidence   │
              │    more complete  │
              │    → skill helped │
              │                   │
              │  Δ_efficiency:    │
              │    A used fewer   │
              │    turns          │
              │    → skill helped │
              │                   │
              │  Δ_negative:      │
              │    A wrong,       │
              │    B correct      │
              │    → skill hurt   │
              └──────────────────┘
```

### 7.2 Reward Formulation

The reward operates at **two granularities**: a per-question outcome reward
from the dual-thread comparison, and a per-step reward within each thread's
reasoning trace. Both include explicit penalties.

#### 7.2.1 Outcome Reward (Per-Question, Dual-Thread)

The dual-thread comparison produces a reward that captures whether the
skill-augmented thread outperformed the skill-free baseline:

| Outcome | Thread A (w/ skill) | Thread B (no skill) | `r_outcome` | Interpretation |
|---------|--------------------|--------------------|-------------|----------------|
| **Skill helped** | Correct | Wrong | **+1.0** | Skills were the difference-maker |
| **Both correct** | Correct | Correct | **+0.2** | Skills didn't hurt (slight bonus for using them) |
| **Both wrong** | Wrong | Wrong | **-0.3** | Skills failed to rescue; mild penalty |
| **Skill hurt** | Wrong | Correct | **-1.0** | Skills actively damaged reasoning |

```python
def outcome_reward(a_correct: bool, b_correct: bool) -> float:
    if a_correct and not b_correct:
        return +1.0   # skill was the deciding factor
    if a_correct and b_correct:
        return +0.2   # skill didn't hurt, small positive signal
    if not a_correct and not b_correct:
        return -0.3   # neither worked, mild penalty to discourage wasted skill calls
    return -1.0        # skill actively hurt — strong negative signal
```

#### 7.2.2 Step-Level Reward (Per-Turn, Within Each Thread)

Each reasoning turn within a thread receives a **step reward** based on
whether that step made measurable progress toward the answer. This gives
dense, per-action signal instead of only sparse end-of-episode reward.

```python
@dataclass
class StepReward:
    """Reward for a single reasoning step (skill invocation or retrieval)."""

    # --- Positive signals ---
    r_evidence: float       # +0.1 per new, relevant evidence node retrieved
    r_grounding: float      # +0.15 if step grounded an entity/event the question asks about
    r_progress: float       # +0.2 if step reduced an identified evidence gap
    r_novel_info: float     # +0.1 if step found information not in prior steps (non-redundant)

    # --- Penalties ---
    p_redundant: float      # -0.1 if step retrieved information already in the evidence chain
    p_irrelevant: float     # -0.15 if retrieved evidence has low relevance to the question
    p_wrong_skill: float    # -0.2 if skill preconditions were not met (skill misapplied)
    p_hallucination: float  # -0.3 if step introduced a claim not grounded in any memory node
    p_turn_cost: float      # -0.05 fixed per-turn cost (encourages efficiency)

    @property
    def total(self) -> float:
        return (
            self.r_evidence + self.r_grounding + self.r_progress + self.r_novel_info
            + self.p_redundant + self.p_irrelevant + self.p_wrong_skill
            + self.p_hallucination + self.p_turn_cost
        )
```

**How step rewards are computed (by the 8B Verifier):**

```
For each step t in the reasoning trace:

  1. Evidence check:
     retrieved_nodes = step.retrieved_memory_nodes
     prior_nodes = union of all nodes from steps 0..t-1

     new_relevant = [n for n in retrieved_nodes
                     if n not in prior_nodes
                     and similarity(n, question) > τ_relevant]

     r_evidence  = 0.1 × len(new_relevant)
     p_redundant = -0.1 × len(retrieved_nodes ∩ prior_nodes)
     p_irrelevant = -0.15 × len([n for n in retrieved_nodes
                                  if similarity(n, question) < τ_irrelevant])

  2. Grounding check:
     target_entities = extract_entities(question)
     grounded_this_step = [e for e in target_entities
                           if e newly grounded by this step]
     r_grounding = 0.15 × len(grounded_this_step)

  3. Progress check:
     gaps_before = evidence_gaps(question, steps[0..t-1])
     gaps_after  = evidence_gaps(question, steps[0..t])
     r_progress  = 0.2 × max(0, len(gaps_before) - len(gaps_after))

  4. Skill-specific checks (Thread A only):
     if step used a skill:
       if skill.preconditions not satisfied:
         p_wrong_skill = -0.2
       if step generated claim not in any memory node:
         p_hallucination = -0.3

  5. Turn cost:
     p_turn_cost = -0.05  (always)
```

#### 7.2.3 Composite Per-Question Reward

The final reward for a question combines the outcome reward with the
sum of step rewards, enabling both sparse (did you get it right?) and
dense (did each step make sense?) training signal:

```python
@dataclass
class DualThreadReward:
    question_id: str

    # Outcome
    answer_a_correct: bool
    answer_b_correct: bool
    r_outcome: float          # from §7.2.1

    # Step-level (Thread A)
    step_rewards_a: List[StepReward]
    turns_used_a: int

    # Step-level (Thread B)
    step_rewards_b: List[StepReward]
    turns_used_b: int

    # Evidence quality (continuous, 0-1, from Verifier)
    evidence_quality_a: float
    evidence_quality_b: float

    @property
    def step_total_a(self) -> float:
        """Sum of all step rewards in Thread A."""
        return sum(s.total for s in self.step_rewards_a)

    @property
    def step_total_b(self) -> float:
        return sum(s.total for s in self.step_rewards_b)

    @property
    def efficiency_bonus(self) -> float:
        """Bonus if Thread A used fewer turns than Thread B for same/better result."""
        if self.turns_used_b == 0:
            return 0.0
        saved_fraction = (self.turns_used_b - self.turns_used_a) / self.turns_used_b
        return 0.2 * max(0.0, saved_fraction)  # only reward saving, don't penalize more turns

    @property
    def evidence_delta(self) -> float:
        """Reward for Thread A producing better evidence."""
        return 0.3 * (self.evidence_quality_a - self.evidence_quality_b)

    @property
    def composite_reward(self) -> float:
        """Final reward combining all signals.

        Weights:
          outcome    (0.35) — did skills change correctness?
          step_delta (0.25) — did skill thread make better per-step progress?
          evidence   (0.20) — did skill thread produce better evidence?
          efficiency (0.10) — did skill thread use fewer turns?
          step_abs   (0.10) — absolute step quality of the skill thread
        """
        r_step_delta = 0.25 * (self.step_total_a - self.step_total_b)
        r_step_abs = 0.10 * self.step_total_a
        return (
            0.35 * self.r_outcome
            + r_step_delta
            + self.evidence_delta
            + self.efficiency_bonus
            + r_step_abs
        )
```

#### 7.2.4 Penalty Summary Table

| Penalty | Value | When Applied | Why |
|---------|-------|-------------|-----|
| `p_turn_cost` | **-0.05** | Every reasoning turn | Discourages unnecessary steps; teaches efficiency |
| `p_redundant` | **-0.10** | Step retrieves already-known information | Teaches the planner to track what's been found |
| `p_irrelevant` | **-0.15** | Retrieved evidence has low question relevance | Teaches better query formulation |
| `p_wrong_skill` | **-0.20** | Skill preconditions not met when invoked | Teaches when to (not) apply each skill |
| `p_hallucination` | **-0.30** | Step makes a claim not in any memory node | Teaches grounded reasoning; heaviest single-step penalty |
| `r_outcome = -0.30` | **-0.30** | Both threads wrong | Mild discouragement — skills didn't help but weren't the cause |
| `r_outcome = -1.0` | **-1.00** | Skill thread wrong, baseline correct | Strongest signal — skills actively damaged reasoning |

### 7.3 What Gets Rewarded — Reward Flow

The multi-granularity reward feeds into three training targets at different
levels:

```
                     ┌─────────────────────────────────────────┐
                     │           Reward Signals                 │
                     │                                          │
                     │  r_outcome ─────────────────┐            │
                     │  evidence_delta ─────────┐  │            │
                     │  efficiency_bonus ─────┐ │  │            │
                     │  step_rewards[] ─────┐ │ │  │            │
                     │                      │ │ │  │            │
                     └──────────────────────┼─┼─┼──┼───────────┘
                                            │ │ │  │
                     ┌──────────────────────┼─┼─┼──┼───────────┐
                     │                      ▼ ▼ ▼  ▼            │
                     │  ┌──────────────────────────────────┐    │
                     │  │  Planner (8B)                     │    │
                     │  │  Learns: which skill to pick,     │    │
                     │  │  when to stop, when to skip skills│    │
                     │  │  Signal: step_rewards + r_outcome │    │
                     │  └──────────────────────────────────┘    │
                     │                                          │
                     │  ┌──────────────────────────────────┐    │
                     │  │  Skill Bank                       │    │
                     │  │  Learns: which skills are good,   │    │
                     │  │  which to retire/refine            │    │
                     │  │  Signal: per-skill attributed      │    │
                     │  │  composite_reward                  │    │
                     │  └──────────────────────────────────┘    │
                     │                                          │
                     │  ┌──────────────────────────────────┐    │
                     │  │  Skill Crafter (8B)               │    │
                     │  │  Learns: which compositions work,  │    │
                     │  │  what new skills to synthesize     │    │
                     │  │  Signal: composite_reward for new  │    │
                     │  │  vs. existing skill comparisons    │    │
                     │  └──────────────────────────────────┘    │
                     └──────────────────────────────────────────┘
```

#### Per-Skill Attribution

When Thread A uses multiple skills, we attribute the reward to each skill.
The step-level rewards make this more precise than outcome-only attribution:

```python
def attribute_reward_to_skills(
    trace_a: ReasoningTrace,
    dual_reward: DualThreadReward,
) -> Dict[str, float]:
    """Attribute composite reward to individual skills using step-level signals.

    Each skill gets:
    - Direct step rewards from the turns where it was active
    - A share of the outcome reward proportional to its evidence contribution
    """
    skill_rewards: Dict[str, float] = {}
    total_evidence_contribution: Dict[str, float] = {}

    for step in trace_a.steps:
        if step.skill_id:
            # Direct: the step reward for turns where this skill was used
            skill_rewards.setdefault(step.skill_id, 0.0)
            skill_rewards[step.skill_id] += step.reward.total

            # Track how much evidence each skill contributed
            total_evidence_contribution.setdefault(step.skill_id, 0.0)
            total_evidence_contribution[step.skill_id] += step.reward.r_evidence

    # Distribute outcome reward proportional to evidence contribution
    total_evi = sum(total_evidence_contribution.values()) or 1.0
    for skill_id, evi in total_evidence_contribution.items():
        outcome_share = dual_reward.r_outcome * (evi / total_evi)
        skill_rewards[skill_id] += outcome_share

    return skill_rewards
```

### 7.4 Training the 8B Orchestrator with Dual-Thread Signal

The dual-thread reward can be used to fine-tune the 8B Orchestrator via
**GRPO (Group Relative Policy Optimization)** — the same framework already
in `skill_agents/grpo/`:

```
For each question Q:
  1. Generate N reasoning traces with skills     → Thread A samples
  2. Generate N reasoning traces without skills   → Thread B samples
  3. Compute per-trace reward from dual comparison
  4. Apply GRPO update: reinforce traces where skills helped,
     penalize traces where skills hurt

Loss = GRPO_loss(
    policy=8B_orchestrator,
    traces=Thread_A_traces + Thread_B_traces,
    rewards=[composite_reward for each trace],
    reference_policy=8B_base,
)
```

The 8B model learns three things simultaneously:
- **When to use skills** (routing: some questions are better without skills)
- **Which skills to select** (skill selection improves over time)
- **How to compose skill invocations** (multi-turn planning improves)

### 7.5 Dual-Thread Deployment During Training

```
┌──────────────────────────────────────────────────────────────┐
│  Training Loop (per batch of questions)                       │
│                                                               │
│  1. Sample batch of questions from benchmark                  │
│                                                               │
│  2. For each question, run BOTH threads in parallel:          │
│     ┌─────────────────────┐  ┌─────────────────────┐        │
│     │ Thread A (w/ skill)  │  │ Thread B (no skill)  │        │
│     │ 8B plans + selects   │  │ 8B plans + retrieves │        │
│     │ skills → Reasoner    │  │ directly → Reasoner  │        │
│     └─────────────────────┘  └─────────────────────┘        │
│                                                               │
│  3. Score both threads against ground truth                   │
│                                                               │
│  4. Compute dual-thread rewards                               │
│                                                               │
│  5. Update skill bank:                                        │
│     • Skills with positive attribution → reinforce            │
│     • Skills with negative attribution → flag for reflection  │
│     • New skill patterns discovered → craft proto-skills      │
│                                                               │
│  6. (Optional) Update 8B Orchestrator via GRPO                │
│                                                               │
│  7. Run bank maintenance every K iterations:                  │
│     • Merge, split, retire, promote                           │
│     • Retire skills with consistently negative rewards        │
│     • Promote proto-skills with consistently positive rewards │
└──────────────────────────────────────────────────────────────┘
```

### 7.6 Cost Analysis: Is Running Two Threads Affordable?

| Component | Thread A Cost | Thread B Cost | Total Overhead |
|-----------|--------------|--------------|----------------|
| 8B Planner | ~2s | ~1.5s (simpler without skill selection) | ~3.5s |
| Memory retrieval | ~50ms | ~50ms | ~100ms |
| 72B Reasoner | ~18s | ~18s | ~36s |
| 8B Verifier | ~2s | ~2s | ~4s |
| **Per question** | **~22s** | **~22s** | **~44s total** |

The main cost is the **second 72B Reasoner call**. Mitigation strategies:

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| **Share 72B calls** — batch both prompts in a single vLLM request | ~30% latency reduction | Requires same-length prompts |
| **Sample Thread B** — only run Thread B on a random 30-50% subset | ~50% total cost | Noisier reward signal |
| **Skip Thread B for easy questions** — if 8B confidence > 0.95, assume skill-free would also succeed | ~30% total cost | Slight reward bias |
| **Use 32B for Thread B** — cheaper model for the no-skill baseline | ~40% 72B cost | Slightly unfair comparison (weaker baseline) |

**Recommended:** Sample Thread B at 50% during early training (skill bank
is changing rapidly, need frequent signal) and reduce to 20% once the bank
stabilizes.

### 7.7 What the Dual-Thread Design Answers

This directly addresses the experimental question from the project brief:

> *"Skill-augment actually help? General VLM vs. Skill-augment?"*

Rather than answering this as a one-shot ablation, the dual-thread design
produces a **continuous, per-question, per-skill** answer throughout
training. It tells us:

- **Which question types** benefit from skills (social reasoning: yes;
  simple factual: no)
- **Which specific skills** contribute positively vs. negatively
- **When skill composition helps** vs. when a single atomic skill suffices
- **Whether skill benefit increases** as the bank evolves (continual
  learning signal)
- **The ceiling of the skill-free baseline** — how good the Reasoner is
  with just memory and no skill scaffolding

---

## 8. Model Configuration and Deployment

### 7.1 Model Assignments

| Role | Model | VRAM | GPUs | Deployment |
|------|-------|------|------|------------|
| **Observer-A (Social)** | Qwen3-VL-72B-Instruct | ~144 GB | 4× A100 (80GB) | vLLM, TP=4, offline |
| **Observer-B (Spatial)** | Qwen3-VL-72B-Instruct | ~144 GB | 4× A100 (80GB) | vLLM, TP=4, offline |
| **Reasoner** | Qwen3-VL-72B-Instruct | ~144 GB | 4× A100 (80GB) | vLLM, TP=4, online (shares with one Observer) |
| **8B Orchestrator** | Qwen3-VL-8B or Qwen3-8B (text-only sufficient) | ~16 GB | 1× A100 | vLLM, TP=1, always-on |
| **Text Embedder** | Qwen3-Embedding-0.6B | ~2 GB | Shared GPU | Transformers |
| **MM Embedder** | Qwen3-VL-Embedding-2B | ~4 GB | Shared GPU | Transformers |

### 7.2 GPU Allocation (2× 8-GPU Nodes)

```
Node 1 (8× A100-80GB):
  GPU 0-3: Observer-A (72B, TP=4) [offline]
  GPU 4-7: Observer-B (72B, TP=4) [offline]
  → During online phase: GPU 0-3 become Reasoner

Node 2 (or same node, time-shared):
  GPU 0: 8B Orchestrator
  GPU 1: Embedders (0.6B + 2B)
  GPU 2-7: Available for parallel batch processing
```

### 7.3 Serving Configuration

```yaml
observer_a:
  model: Qwen/Qwen3-VL-72B-Instruct
  backend: vllm
  tensor_parallel: 4
  gpu_memory_utilization: 0.90
  max_model_len: 32768
  system_prompt: "social_observer"  # specialized for social extraction

observer_b:
  model: Qwen/Qwen3-VL-72B-Instruct
  backend: vllm
  tensor_parallel: 4
  gpu_memory_utilization: 0.90
  max_model_len: 32768
  system_prompt: "spatial_observer"  # specialized for spatial extraction

reasoner:
  model: Qwen/Qwen3-VL-72B-Instruct
  backend: vllm
  tensor_parallel: 4
  gpu_memory_utilization: 0.90
  max_model_len: 32768

orchestrator:
  model: Qwen/Qwen3-VL-8B  # or Qwen3-8B text-only if sufficient
  backend: vllm
  tensor_parallel: 1
  gpu_memory_utilization: 0.85
  max_model_len: 32768
```

---

## 9. Benchmarks and Evaluation

### 8.1 Primary Benchmarks

| Benchmark | Type | What It Tests | Metrics |
|-----------|------|---------------|---------|
| **Video-Holmes** | 270 short films, 1,837 QA | Deep causal/temporal/social reasoning | Accuracy, reasoning chain quality |
| **MA-EgoQA** | Multi-agent ego-centric | Social interaction, task coordination, ToM, temporal reasoning | Per-category accuracy |
| **EgoLife** | Long-term ego-centric | Long-term social/spatial tracking | Accuracy, temporal grounding |
| **LongVidSearch** | Multi-hop evidence | Multi-hop evidence chaining | Evidence recall, answer accuracy |

### 8.2 Evaluation Dimensions

| Dimension | What We Measure | How |
|-----------|-----------------|-----|
| **Answer accuracy** | Final answer correctness | Standard MCQ / open-ended eval |
| **Evidence quality** | Are cited timestamps and facts correct? | Compare against annotated ground truth |
| **Reasoning faithfulness** | Does the reasoning chain actually support the answer? | Human eval + automated consistency check |
| **Skill contribution** | Does skill augmentation actually help? | Ablation: with/without skill bank |
| **Observer separation** | Does dual-observer outperform single? | Ablation: single 72B vs. dual |
| **Continual improvement** | Do skills improve over time? | Track accuracy across evolution iterations |

### 8.3 Ablation Design

| Variant | Observer | Memory | Skills | Evidence | Tests |
|---------|----------|--------|--------|----------|-------|
| **Full system** | Dual 72B | Graph (5 types) | Full bank | Yes | Main result |
| Single observer | Single 72B | Graph (5 types) | Full bank | Yes | Observer separation value |
| No skills | Dual 72B | Graph (5 types) | None | Yes | Skill augmentation value |
| No graph (flat memory) | Dual 72B | Flat text | Full bank | Yes | Graph structure value |
| No visual memory | Dual 72B | Graph (4 types, no visual) | Full bank | Yes | Visual memory value |
| No social-state | Dual 72B | Graph (4 types, no social) | Full bank | Yes | Social state value |
| No spatial-state | Dual 72B | Graph (4 types, no spatial) | Full bank | Yes | Spatial state value |
| No evolution | Dual 72B | Graph (5 types) | Frozen bank | Yes | Self-evolution value |
| General VLM baseline | Single 72B raw | None | None | No | Baseline |
| 8B only | None | Graph (5 types) | Full bank | Yes | Small model ceiling |

### 8.4 Key Comparison

Against the comparison baseline (arXiv:2603.24558):

| Axis | Them | Us |
|------|------|-----|
| Memory | Textual / flat | Graph-structured with 5 memory types |
| Visual | Discarded after captioning | Preserved as visual memory nodes + keyframes |
| Skills | Fixed prompts | Self-evolving, evidence-grounded skill bank |
| Multi-hop | Single retrieval | Planned multi-turn with evidence sufficiency |
| Evidence | Implicit | Explicit evidence chain with timestamps |
| Continual | No | Yes — failure reflection → new skills |

---

## 10. Module Layout

```
Video_Skills/
├── infra_plans/
│   └── self_evolving_skill_bank_design.md   # ← this document
│
├── agentic_memory/                          # NEW — Memory Graph
│   ├── __init__.py
│   ├── graph.py                             # AgenticMemoryGraph (§2.3)
│   ├── nodes.py                             # MemoryNode, MemoryEdge schemas
│   ├── builders/
│   │   ├── __init__.py
│   │   ├── social_builder.py                # Observer-A output → social/episodic nodes
│   │   ├── spatial_builder.py               # Observer-B output → spatial/visual nodes
│   │   ├── semantic_distiller.py            # Episodic → Semantic (8B)
│   │   └── entity_resolver.py               # Cross-clip entity resolution
│   ├── retrieval.py                         # Graph-aware retrieval (traverse + embed)
│   └── serialization.py                     # Save/load graph (JSON/pickle)
│
├── observers/                               # NEW — Dual Observer System
│   ├── __init__.py
│   ├── base.py                              # ObserverBase ABC
│   ├── social_observer.py                   # Observer-A: social extraction
│   ├── spatial_observer.py                  # Observer-B: spatial extraction
│   ├── frame_sampler.py                     # Adaptive frame sampling
│   └── prompts/
│       ├── social_system.txt                # Social observer system prompt
│       └── spatial_system.txt               # Spatial observer system prompt
│
├── skill_bank_v2/                           # NEW — Visual Skill Bank
│   ├── __init__.py
│   ├── skill_graph.py                       # SkillGraph (§3.3)
│   ├── atomic_skills.py                     # Level 0 skill definitions (§3.1)
│   ├── composed_skills.py                   # Level 1 skill definitions
│   ├── skill_executor.py                    # Execute a skill against the memory graph
│   ├── skill_crafter.py                     # Craft new skills from traces (§6.2)
│   ├── failure_reflector.py                 # Failure analysis → new skills (§6.3)
│   └── bank_maintenance.py                  # Merge/split/retire/promote
│
├── planner/                                 # NEW — Multi-Turn Planner
│   ├── __init__.py
│   ├── planner.py                           # ReasoningPlanner (§4)
│   ├── evidence_tracker.py                  # Track evidence sufficiency
│   └── prompts/
│       └── planner_system.txt               # Planner prompt template
│
├── answerer/                                # NEW — Evidence-First Answerer
│   ├── __init__.py
│   ├── prompt_composer.py                   # Compose evidence-rich prompt for Reasoner
│   ├── verifier.py                          # 8B verification (§5.2)
│   └── prompts/
│       └── reasoner_system.txt              # Reasoner prompt template
│
├── evolution/                               # NEW — Continual Learning
│   ├── __init__.py
│   ├── skill_extractor.py                   # Extract skills from successful traces
│   ├── skill_transfer.py                    # Cross-benchmark skill transfer
│   └── evolution_loop.py                    # Main continual learning loop (§6.5)
│
├── orchestrator/                            # NEW — Top-Level Controller
│   ├── __init__.py
│   ├── orchestrator.py                      # Main entry point
│   ├── config.py                            # Model paths, thresholds
│   └── adapters/
│       ├── __init__.py
│       ├── video_holmes.py                  # Video-Holmes benchmark adapter
│       ├── ma_egoqa.py                      # MA-EgoQA adapter
│       ├── egolife.py                       # EgoLife adapter
│       └── longvidsearch.py                 # LongVidSearch adapter
│
├── memory_manage/                           # EXISTING — to be superseded by agentic_memory/
│   └── ...
│
├── skill_agents/                            # EXISTING — schemas reused
│   └── stage3_mvp/schemas.py               # Skill, Protocol, Contract, etc.
│
└── rag/                                     # EXISTING — MemoryStore reused
    └── retrieval.py                         # Embedding-based retrieval
```

---

## 11. Implementation Plan

### Phase 0: Infrastructure (Week 1)
- [ ] `orchestrator/config.py` — model paths, thresholds, prompt templates
- [ ] vLLM serving scripts for dual 72B + 8B
- [ ] Frame sampler (`observers/frame_sampler.py`)

### Phase 1: Memory Graph (Weeks 1-2)
- [ ] `agentic_memory/nodes.py` — MemoryNode, MemoryEdge schemas
- [ ] `agentic_memory/graph.py` — AgenticMemoryGraph core
- [ ] `agentic_memory/retrieval.py` — graph traversal + embedding search
- [ ] `agentic_memory/serialization.py` — save/load
- [ ] Integration tests with `rag/retrieval.py` MemoryStore

### Phase 2: Observers (Weeks 2-3)
- [ ] `observers/social_observer.py` — Observer-A prompts and output parsing
- [ ] `observers/spatial_observer.py` — Observer-B prompts and output parsing
- [ ] `agentic_memory/builders/` — all four builders
- [ ] End-to-end test: video → dual observer → memory graph

### Phase 3: Atomic Skills + Planner (Weeks 3-4)
- [ ] `skill_bank_v2/atomic_skills.py` — all Level 0 skills
- [ ] `skill_bank_v2/skill_executor.py` — execute skill against memory graph
- [ ] `planner/planner.py` — multi-turn reasoning planner
- [ ] `planner/evidence_tracker.py` — evidence sufficiency
- [ ] Test: question → planner → atomic skills → evidence chain

### Phase 4: Answerer + Verifier (Week 4)
- [ ] `answerer/prompt_composer.py` — evidence-rich prompt composition
- [ ] `answerer/verifier.py` — 8B verification pipeline
- [ ] End-to-end test: question → plan → evidence → answer → verify

**Milestone: Memory-Only Baseline** — Can answer questions using memory
graph + atomic skills + planner, without composed skills or evolution.
Run on Video-Holmes to establish baseline accuracy.

### Phase 5: Composed Skills + Skill Graph (Weeks 5-6)
- [ ] `skill_bank_v2/composed_skills.py` — Level 1 skills
- [ ] `skill_bank_v2/skill_graph.py` — skill DAG with performance tracking
- [ ] Update planner to use composed skills
- [ ] Evaluate: does skill composition improve over atomic-only?

### Phase 6: Self-Evolution (Weeks 6-7)
- [ ] `skill_bank_v2/skill_crafter.py` — craft skills from successful traces
- [ ] `skill_bank_v2/failure_reflector.py` — failure analysis → new skills
- [ ] `evolution/skill_extractor.py` — trace → skill extraction
- [ ] `evolution/evolution_loop.py` — continual learning cycle
- [ ] `skill_bank_v2/bank_maintenance.py` — merge/split/retire

### Phase 7: Benchmark Adapters (Week 7)
- [ ] `orchestrator/adapters/video_holmes.py`
- [ ] `orchestrator/adapters/ma_egoqa.py`
- [ ] `orchestrator/adapters/egolife.py`
- [ ] `orchestrator/adapters/longvidsearch.py`

### Phase 8: Full Evaluation + Ablations (Weeks 8-10)
- [ ] Run full system on all benchmarks
- [ ] Run all ablation variants (§9.3)
- [ ] Run continual learning experiments (3-5 evolution iterations)
- [ ] Run Observer comparison (7B vs 72B observers, same Reasoner)
- [ ] Analysis: which skills transferred, which were created by reflection

---

## 12. Open Design Questions

### Q1: Should the 8B model be text-only or VLM?

**Current position:** Use Qwen3-VL-8B (vision-capable) so it can:
- Verify keyframe relevance by looking at frames
- Perform lightweight visual checks during verification
- Generate better skill protocols by understanding what the frames contain

**Alternative:** Use Qwen3-8B (text-only) and rely entirely on the Observer
outputs. This is cheaper and faster but loses the ability to do visual
verification.

**Experiment:** Compare both configurations on Video-Holmes verification
accuracy.

### Q2: Graph database vs. in-memory graph?

**Current position:** In-memory Python graph (dict-based). For videos
up to ~2 hours, the graph fits comfortably in RAM (~100-500 nodes,
~1000-5000 edges).

**When to switch:** If processing 10+ hour videos (EgoLife), consider
Neo4j or similar for persistent, indexed graph storage.

### Q3: How many evolution iterations are needed?

**Hypothesis:** 3-5 iterations should show meaningful improvement on
reasoning-heavy benchmarks (Video-Holmes TCI/MHR). After that, returns
diminish as the skill bank saturates.

**Experiment:** Track accuracy vs. evolution iteration number across
all benchmarks.

### Q4: Skill transfer across benchmarks — automatic or curated?

**Current position:** Automatic transfer via embedding similarity +
contract overlap, with lightweight validation. The 8B model decides
whether a transferred skill is applicable.

**Risk:** Negative transfer — a skill that works well on short-video
reasoning (Video-Holmes) might hurt on long-video retrieval (EgoLife).

**Mitigation:** Per-benchmark performance tracking in the skill graph;
skills can have benchmark-specific performance gates.

---

## 13. Related Work and Positioning

| Work | Relationship to Our Design |
|------|----------------------------|
| **SCALAR** (arXiv:2603.09036) | LLM-proposed symbolic skills + environment grounding. We extend this to *visual* skills with evidence grounding and self-evolution. |
| **X-Skill** | Cross-embodiment skill discovery. We adapt skill discovery from robotics to video reasoning, using memory graphs instead of embodied trajectories. |
| **M3-Agent** (M3-Bench) | Memory graph for video QA. We extend their VideoGraph with social-state and spatial-state memory, and add self-evolving skills on top. |
| **WorldMM** (arXiv:2512.02425) | World model for multimodal reasoning. Our Observers produce similar structured world representations, but we add agentic skill management. |
| **Video-Holmes** (arXiv:2505.21374) | Benchmark for deep reasoning. Our primary evaluation target for reasoning depth. |
| **Multi-hop reasoning** (arXiv:2502.12442) | Multi-hop retrieval planning. Our Planner implements similar iterative retrieval but with skill-guided execution. |
| **arXiv:2603.24558** | Direct comparison target. We add graph memory, visual memory preservation, skill evolution, and evidence-grounded answering. |
| **Web-agent video** (arXiv:2410.19100) | Agentic video understanding. Different domain (web) but similar architecture of planning + tool use. |
| **arXiv:2408.14469** | Long video understanding. Complementary approach; we focus on structured memory + skills rather than compression. |
