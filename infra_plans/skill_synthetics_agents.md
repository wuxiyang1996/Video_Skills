# Skill Synthetics Agents — Design Plan

> Goal: Define how skills are **synthesized, crafted, evaluated, and evolved**
> over time. This covers the skill crafter pipeline, quality control (8B as
> judge), cross-video accumulation, failure-driven self-evolution, and the
> continuous skill maintenance loop.
>
> **Related plans:**
> - [Video Benchmarks & Grounding](video_benchmarks_grounding.md) — benchmarks, memory graph, adapters
> - [Actors / Reasoning Model](actors_reasoning_model.md) — 8B controller, reasoning core, orchestrator
> - [Skill Extraction / Bank](skill_extraction_bank.md) — skill definitions and bank infrastructure

---

## 1. Skill Crafter Pipeline

After a memory graph is built (see [Actors plan §3](actors_reasoning_model.md)),
the 8B model analyzes the graph to discover reusable reasoning skills. This
follows the same three-stage pipeline as the COS-PLAY video skill pipeline but
is driven entirely by the small model over the memory graph.

### Pipeline

```
VideoMemoryGraph
  │
  ├─ 1. Temporal segmentation
  │     • Boundary detection via predicate delta between consecutive
  │       episodic nodes (reuses COS-PLAY's ScoredBoundary scoring)
  │     • Intention tagging: Qwen3-VL-8B classifies each segment
  │       with a [TAG] from the video intention taxonomy:
  │       OBSERVE | INTERACT | NAVIGATE | COMMUNICATE | MANIPULATE |
  │       INVESTIGATE | REACT | WAIT | APPROACH | RETREAT | DELIVER | RECEIVE
  │
  ├─ 2. Contract extraction  (Qwen3-VL-8B)
  │     • For each segment: compute eff_add / eff_del from predicate
  │       changes across the segment boundary
  │     • Aggregate across similar segments (same intention + high
  │       predicate overlap) to build robust contracts
  │
  └─ 3. Protocol generation  (Qwen3-VL-8B)
        • For each skill cluster, synthesize a step-by-step Protocol
        • Prompt: "You are creating a reusable reasoning skill from
          these video segments. Write a protocol that a reasoning agent
          can follow to analyze similar scenes in new videos."
        • Output: preconditions, steps[], success_criteria, abort_criteria
```

### Output

- `skill_bank.jsonl` — COS-PLAY-compatible `Skill` objects
- Each skill carries: `skill_id`, `name`, `strategic_description`, `tags`,
  `protocol`, `contract`, `sub_episodes` (evidence pointers), `n_instances`

---

## 2. Skill Quality Control (8B as Judge)

The 8B model evaluates crafted skills on six dimensions (reusing
`skill_agents/skill_evaluation/`):

| Dimension | 8B Model Check |
|-----------|----------------|
| **Coherence** | "Does this skill's protocol make logical sense for its intention tag?" |
| **Discriminability** | "Is this skill distinct from existing skills in the bank?" |
| **Composability** | "Can this skill chain with other skills in a reasoning plan?" |
| **Generalization** | "Would this skill apply to videos beyond the source?" |
| **Utility** | "Would following this protocol help answer a question?" |
| **Granularity** | "Is this skill at the right level of abstraction?" |

Skills scoring below threshold on any dimension are sent back for
refinement or merged with higher-quality neighbors.

---

## 3. Cross-Video Skill Accumulation

When processing multiple videos, the skill crafter:

1. Loads the existing skill bank
2. Attempts to **merge** new segments into existing skills (by embedding
   similarity + contract overlap)
3. Creates **new** skills only when no existing skill covers the pattern
4. **Retires** skills that lose all supporting evidence

This mirrors COS-PLAY's `bank_maintenance` (split/merge/refine/retire)
but adapted for video reasoning skills rather than game strategies.

---

## 4. Failure Taxonomy

When the controller produces a wrong answer, the failure is classified to
drive targeted skill updates.

| Failure Type | Description | Example |
|-------------|-------------|---------|
| **Missed evidence** | Relevant memory node exists but was not retrieved | Question about a conversation; retrieval query missed the right episode |
| **Wrong temporal linkage** | Events connected incorrectly in the reasoning chain | Cause and effect reversed; wrong temporal ordering |
| **Wrong entity grounding** | Confused two characters or misidentified an entity | "The woman" resolved to wrong face_id |
| **Perspective confusion** | Answered based on global truth instead of character's local view | System knows Bob stole the key; incorrectly claims Alice also knows |
| **False-belief reasoning error** | Failed to model that a character holds an incorrect belief | Character was told a lie; system treated the lie as truth |
| **Overconfident inference** | Drew strong conclusion from weak/ambiguous evidence | Single facial expression interpreted as definitive proof of deception |
| **Insufficient evidence, forced answer** | Not enough evidence existed but system answered anyway | Memory graph lacked the relevant segment; system hallucinated |

---

## 5. Failure → Update Mapping

Each failure type triggers a targeted update. The controller identifies the
structural cause and patches the specific component.

| Failure Type | Update Target | Specific Action |
|-------------|---------------|-----------------|
| Missed evidence | **Retrieval strategy** | Add alternative query patterns to the skill; increase retrieval breadth |
| Wrong temporal linkage | **Memory schema** | Strengthen `precedes`/`causes` edges; add temporal verification step |
| Wrong entity grounding | **Entity resolver** | Add disambiguation step; refine entity matching thresholds |
| Perspective confusion | **Perspective thread** | Add explicit "check perspective" step to social skills; create `check_character_access` skill if none exists |
| False-belief reasoning error | **Skill refinement** | Refine `infer_belief_update` to handle lie propagation; add false-belief substep |
| Overconfident inference | **Confidence calibration** | Lower confidence thresholds; add "require N supporting events" constraint |
| Insufficient evidence | **Verifier rule** | Strengthen evidence sufficiency checker; add abstention option |

---

## 6. Skill Evolution Mechanisms

| Mechanism | Trigger | Action |
|-----------|---------|--------|
| **Reinforce** | Skill used in correct answer | Bump success rate, update average evidence quality |
| **Refine** | Skill used in wrong answer with identified fix | Modify execution steps; add/remove preconditions |
| **Split** | Skill has high variance (works for some subtypes, fails for others) | Create two specialized skills from the original |
| **Merge** | Two skills have >80% step overlap and similar performance | Combine into single skill with broader trigger conditions |
| **Craft new** | Failure type has no matching skill | 8B controller generates a new skill from failure analysis + successful counter-example |
| **Retire** | Skill has <20% success rate over 20+ invocations | Remove from active bank; archive for reference |

---

## 7. Evolution Loop

```
For each evaluation batch:
  1. Run questions through controller (with skill bank)
  2. Score against ground truth
  3. For correct answers:
     → Reinforce skills used
     → Extract new skill patterns from novel compositions
  4. For wrong answers:
     → Classify failure type (§4)
     → Apply targeted update (§5)
     → If no existing skill addresses the failure pattern → craft new skill
  5. Every K batches:
     → Run bank maintenance (merge, split, retire)
     → Log skill bank statistics for analysis
```

### Evolution metrics

| Metric | What It Measures |
|--------|-----------------|
| **Skill reuse rate** | Unique skills / total skill invocations |
| **Skill refinement rate** | Success rate before/after refinement |
| **Bank growth rate** | New skills per evaluation batch |
| **Retirement rate** | Skills retired per K batches |
| **Composition stability** | How often top compositions change across batches |

---

## 8. Integration with Existing Components

| Existing component | How it connects |
|---|---|
| `skill_agents/stage3_mvp/schemas.py` → `Skill`, `Protocol`, `SkillEffectsContract` | Crafted skills use these schemas |
| `skill_agents/skill_bank/bank.py` → `SkillBankMVP` | Bank stored as SkillBankMVP-compatible JSONL |
| `skill_agents/skill_evaluation/` → LLM judge | Quality control reuses evaluation dimensions |
| `dataset_examples/video_skill_pipeline_design.md` | Offline pipeline extends that design with memory-graph-driven discovery |
| `data_structure/experience.py` → `Experience`, `Episode` | Reasoning traces stored as Episodes for replay/analysis |

---

## 9. Implementation Notes

### Skill Crafter Module (`skill_crafter.py`)

Lives in `Video_Skills/small_model_orchestrator/skill_crafter.py`.

```python
class SkillCrafter:
    def __init__(self, vlm_fn, embedder, existing_bank_path=None): ...
    def craft_from_graph(self, graph: VideoMemoryGraph) -> List[Skill]: ...
    def evaluate_skill(self, skill: Skill) -> SkillEvaluation: ...
    def merge_into_bank(self, new_skills: List[Skill], bank: SkillBank) -> SkillBank: ...
```

### Expected effort

| Phase | Task | Effort |
|-------|------|--------|
| Crafter pipeline | Temporal segmentation + contract extraction + protocol gen | 3-4 days |
| Quality control | 8B judge + evaluation dimensions | 1-2 days |
| Cross-video accumulation | Merge/dedup logic | 1-2 days |
| Evolution loop | Failure classification + targeted updates | 2-3 days |
| Bank maintenance | Split/merge/retire automation | 1-2 days |
| **Total** | | **~8-13 days** |
