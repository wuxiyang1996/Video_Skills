# Video Visual Grounding to Skill Bank Pipeline

Design document for extracting reusable **inference-time skills** from Video-Holmes and M3-Bench videos using visual grounding (Qwen2.5-VL-7B), formulated in the [COS-PLAY](https://github.com/wuxiyang1996/COS-PLAY) skill bank format.

---

## Purpose

The goal is to build a **skill bank that augments large VLMs (72B+) at inference time** for video understanding tasks. Rather than requiring the VLM to reason from scratch over raw video, we pre-extract structured reasoning skills from training/reference videos and **retrieve relevant skills into the VLM's prompt** when it encounters a new question.

This is analogous to how COS-PLAY's decision agent retrieves skills from a skill bank to guide game actions — here, a large VLM (e.g., Qwen2.5-VL-72B, InternVL-78B) retrieves video-reasoning skills to guide its answer to complex video understanding questions.

### Inference-Time Flow

```
   New video + question
          |
          v
   +-----------------+      +------------------+
   | Visual Grounding|      |   Skill Bank     |
   | (quick, on the  | ---> |  (pre-built,     |
   |  new video)     |      |   from training  |
   +-----------------+      |   videos)        |
          |                 +--------+---------+
          |  query: predicates,             |
          |  scene type, intention     retrieve top-k
          |                                 |
          v                                 v
   +----------------------------------------------+
   |  Large VLM Prompt (72B)                       |
   |                                               |
   |  [System] You are a video reasoning agent.    |
   |                                               |
   |  [Retrieved Skills]                           |
   |  Skill: "Track Character Relationship"        |
   |    When: person_count>=2, social_cue=true     |
   |    Steps: 1. Identify all characters...       |
   |           2. Note interaction patterns...     |
   |           3. Infer relationship type...       |
   |    Look for: body language, dialogue, ...     |
   |                                               |
   |  [Video Context] <frames + descriptions>      |
   |  [Question] What is the relationship between  |
   |             the man and Benjamin?              |
   +----------------------------------------------+
          |
          v
      VLM Answer (skill-guided reasoning)
```

### Why Skills Help Large VLMs

1. **Structured reasoning scaffolding**: Skills provide step-by-step protocols that decompose hard questions (e.g., "What caused X?") into visual sub-tasks the VLM can follow
2. **Predicate-based attention guidance**: Skill preconditions tell the VLM which visual predicates to attend to (e.g., "look for threat_present, confined_space")
3. **Cross-video transfer**: Skills learned from one suspense film's threat-detection pattern transfer to new films with similar setups
4. **Reduced hallucination**: Contracts (eff_add/eff_del) ground the VLM's reasoning in observable state changes rather than fabricated narratives

## Skill Bank Construction Pipeline

A smaller model (**Qwen2.5-VL-7B**) processes training/reference videos offline to build the skill bank. At inference time, a larger model (**72B**) retrieves and uses these skills.

```
  OFFLINE (skill bank construction, Qwen2.5-VL-7B)
  ================================================

   Video-Holmes (270 films) / M3-Bench (100 robot + 920 web)
                              |
                    +---------v----------+
                    | Stage 1: Grounding |    Qwen2.5-VL-7B
                    |  adaptive frame    |    per-frame structured
                    |  sampling + VLM    |    descriptions + predicates
                    +---------+----------+
                              |
                    +---------v--------------+
                    | Stage 2: Segmentation  |   boundary detection via
                    |  temporal sub-episodes |   predicate deltas, scene
                    |  + intention tagging   |   change, action transitions
                    +---------+--------------+
                              |
                    +---------v-----------+
                    | Stage 3: Skills     |   contract extraction,
                    |  contracts +        |   protocol generation,
                    |  protocols + bank   |   clustering + bank output
                    +---------+-----------+
                              |
                       skill_bank.jsonl


  ONLINE (inference, large VLM 72B)
  ==================================

   New video + question
          |
    quick grounding (predicates, scene type)
          |
    RAG retrieval from skill_bank.jsonl
          |
    inject top-k skill protocols into VLM prompt
          |
    large VLM generates skill-guided answer
```

---

## Stage 1: Frame Sampling + Visual Grounding

### Frame Sampling

Different sampling rates per dataset to balance coverage and cost:

| Dataset | Video Length | Sampling Rate | Frames per Video |
|---------|------------|---------------|-----------------|
| Video-Holmes | 1-5 min | 1 frame / 2 sec | ~30-150 |
| M3-Bench robot | ~30 min | 1 frame / 5 sec | ~360 |
| M3-Bench web | varies | 1 frame / 3 sec | varies |

Additional keyframes are added at scene-change boundaries detected via pixel-difference thresholds between consecutive frames.

For M3-Bench robot, subtitle timestamps from the `.srt` files are used to add keyframes aligned with dialogue turns.

### VLM Grounding

Each sampled frame is sent to **Qwen2.5-VL-7B-Instruct** (already available at `transformers/models/Qwen/Qwen2.5-VL-7B-Instruct`) with a structured prompt:

```
Analyze this video frame and return a JSON object:
{
  "objects": ["list of visible objects/people with descriptions"],
  "actions": ["what actions are being performed"],
  "spatial": ["spatial relationships between key entities"],
  "scene": "environment/setting description",
  "emotions": ["visible emotional states or social dynamics"],
  "predicates": {"person_standing": true, "door_open": false, ...}
}
```

### Output: VisualExperience

Each frame produces a `VisualExperience` (extending COS-PLAY's `Experience`):

```json
{
  "frame_idx": 15,
  "timestamp": 30.0,
  "state": "A woman stands in an old elevator looking frightened. A person with a plastic bag on their head is visible in the corridor.",
  "summary_state": "scene=elevator | person_count=2 | woman_emotion=fear | threat_present=true | door_state=open",
  "objects": ["woman", "elevator", "person with plastic bag", "corridor"],
  "actions": ["standing", "looking"],
  "predicates": {
    "person_in_elevator": true,
    "threat_present": true,
    "door_open": true,
    "woman_afraid": true
  },
  "action": "OBSERVE",
  "reward": 0.0,
  "next_state": null,
  "done": false
}
```

### State Delta

Between consecutive frames, a `visual_state_delta` is computed (analogous to COS-PLAY's `_compute_state_delta`):

```
Frame 14: threat_present=false, woman_afraid=false, door_open=true
Frame 15: threat_present=true,  woman_afraid=true,  door_open=true
Delta:    threat_present:false->true, woman_afraid:false->true
```

This delta drives both segmentation (Stage 2) and the strategic notes in summaries.

---

## Stage 2: Temporal Segmentation

### Boundary Detection

Multiple signals are combined to score candidate segment boundaries (analogous to COS-PLAY's `ScoredBoundary`):

| Signal | Weight | Description |
|--------|--------|-------------|
| Predicate change rate | 0.35 | Fraction of boolean predicates that flip between frames |
| Scene change | 0.25 | Pixel-level or embedding-level visual difference |
| Action transition | 0.20 | Dominant action in window changes |
| Subtitle boundary | 0.20 | Speaker turn or topic shift (M3-Bench only) |

Constraints: minimum segment length = 3 frames, maximum = 30 frames.

### Intention Tagging

Each segment is tagged with an intention using Qwen2.5-VL, given the segment's keyframes. The tag set is adapted from COS-PLAY's game tags to video understanding:

| Tag | Meaning | Example |
|-----|---------|---------|
| OBSERVE | Watching/scanning environment | Character surveys a room |
| INTERACT | Social interaction between entities | Two people talking |
| NAVIGATE | Movement through space | Walking down a corridor |
| COMMUNICATE | Verbal/text exchange | Robot answering a question |
| MANIPULATE | Physical object manipulation | Picking up a cup |
| INVESTIGATE | Examining clues or details | Looking at a suspicious object |
| REACT | Emotional/physical reaction | Showing fear or surprise |
| WAIT | Idle / anticipating | Standing still, waiting |
| APPROACH | Moving toward a target | Walking toward a person |
| RETREAT | Moving away from threat | Backing away |
| DELIVER | Providing item/service | Robot serving coffee |
| RECEIVE | Accepting item/information | Taking a delivered object |

### Output: VisualSubEpisode

Each segment maps to COS-PLAY's `SubTask_Experience`:

```json
{
  "sub_task": "[REACT] Woman reacts to threatening figure in elevator",
  "final_goal": "Understand the suspense narrative",
  "segment_frames": [12, 13, 14, 15, 16, 17],
  "timestamp_range": [24.0, 34.0],
  "intention_tag": "REACT",
  "predicates_start": {"threat_present": false, "woman_afraid": false},
  "predicates_end": {"threat_present": true, "woman_afraid": true},
  "cumulative_reward": 0.0,
  "outcome_classification": "partial"
}
```

### Validation Against Ground Truth

For Video-Holmes, annotations provide ground-truth `Segment Description` entries with `TimeRange` fields (e.g., `"00:01-00:24"`). We compare our auto-detected boundaries against these to measure segmentation quality:

```
Ground truth segments for 0at001QMutY:
  Segment 1: 00:01 - 00:24  (woman enters elevator, sees plastic bag person)
  Segment 2: 00:24 - 01:34  (elevator descends, woman disappears)

Our detected segments:
  Segment A: 00:01 - 00:12  (woman enters elevator)
  Segment B: 00:12 - 00:24  (woman notices threat)
  Segment C: 00:24 - 00:58  (elevator descending, fear reaction)
  Segment D: 00:58 - 01:34  (empty elevator revelation)

Overlap IoU with ground truth: measured per-segment
```

---

## Stage 3: Skill Formulation

The key design principle: each skill must be **self-contained and prompt-injectable** — a large VLM (72B) should be able to read a retrieved skill and immediately know (a) when to apply it, (b) what visual evidence to look for, and (c) how to reason step-by-step.

### Skill Clustering

Sub-episodes with similar intention tags and predicate signatures are grouped into skills. Similarity is computed via:

1. Intention tag match (same primary tag)
2. Predicate overlap (Jaccard similarity on `eff_add` / `eff_del` sets)
3. Description embedding cosine similarity (using Qwen2.5-VL text embeddings)

Each cluster becomes a `Skill` with multiple `SubEpisodeRef` evidence pointers.

### Contract Extraction (SkillEffectsContract)

For each skill, aggregate predicate changes across all instances:

```json
{
  "skill_id": "vh_react_threat_001",
  "eff_add": ["woman_afraid", "threat_acknowledged", "escape_attempted"],
  "eff_del": ["environment_safe", "person_calm"],
  "eff_event": ["emotional_shift", "threat_detection"],
  "support": {"woman_afraid": 45, "threat_acknowledged": 42},
  "n_instances": 50
}
```

- `eff_add`: predicates consistently becoming TRUE across skill instances
- `eff_del`: predicates consistently becoming FALSE
- `eff_event`: categorical events observed during the skill
- Only predicates with support > 50% of instances are retained in the contract

At inference time, the contract tells the large VLM **what state changes to expect** — grounding its reasoning in observable visual evidence rather than hallucinated narratives.

### Protocol Generation

For each skill, Qwen2.5-VL synthesizes a `Protocol` from its best sub-episodes. The protocol is the primary artifact consumed by the large VLM at inference time — it provides a **step-by-step reasoning scaffold**:

```json
{
  "preconditions": ["person_present=true", "threat_present=true"],
  "steps": [
    "Identify all entities in the scene and note any unusual appearances",
    "Track how the threatening entity moves relative to the main character",
    "Observe the main character's emotional reaction for fear/surprise cues",
    "Check whether the character attempts escape or is trapped"
  ],
  "success_criteria": ["threat_acknowledged=true", "escape_attempted=true"],
  "abort_criteria": ["scene_transition=true", "threat_present=false"],
  "expected_duration": 8
}
```

### Dataset-Specific Skill Categories

**Video-Holmes** — reasoning skills that help the VLM answer complex narrative questions:

| Category | Example Skills | Helps With (Question Types) |
|----------|---------------|-----------------------------|
| Character Analysis | Identify relationship, Track emotional arc | SR (Social Relationship) |
| Threat Detection | Observe suspicious behavior, Detect hidden danger | MHR (Hidden Reasoning) |
| Causal Reasoning | Connect cause to effect, Identify turning point | TCI (Temporal Causal Inference), IMC (Implicit Causal) |
| Temporal Tracking | Order events chronologically, Detect time gaps | TA (Temporal Arrangement) |
| Pattern Recognition | Identify recurring motifs, Recognize cycles | PAR (Pattern Recognition) |
| Theme Extraction | Identify moral lesson, Recognize narrative pattern | CTI (Core Theme Identification) |

Enriched by annotations: `Key Relationships` feed into character-analysis contracts, `Inference Shots` feed into causal-reasoning protocols.

**M3-Bench Robot** — interaction skills that help the VLM reason about long-form human-robot dialogue:

| Category | Example Skills | Helps With |
|----------|---------------|------------|
| Service Delivery | Prepare and serve requested item | Task completion QA |
| Dialogue Management | Answer question, Clarify preference | Dialogue understanding |
| Navigation | Move to target location, Avoid obstacles | Spatial reasoning |
| Object Manipulation | Pick up item, Place on surface | Action recognition |
| Memory Recall | Remember previous request, Track preferences | Long-context memory QA |

Enriched by subtitles: dialogue turns anchor `COMMUNICATE` and `DELIVER` skill boundaries.

---

## Output: skill_bank.jsonl

Each line is a complete `Skill` object (COS-PLAY-compatible). The `protocol` and `strategic_description` fields are the primary content injected into the large VLM's prompt at inference time:

```json
{
  "skill_id": "vh_observe_suspicious_001",
  "version": 1,
  "name": "Observe Suspicious Behavior",
  "strategic_description": "Detect and track threatening entities in confined spaces by scanning the environment and noting unusual appearances or movements. Look for asymmetric power dynamics, concealed identity, and restricted exit routes.",
  "tags": ["OBSERVE", "INVESTIGATE"],
  "protocol": {
    "preconditions": ["person_present=true", "confined_space=true"],
    "steps": [
      "Scan environment for entities with unusual appearance or concealed identity",
      "Track entity movement and positioning relative to the main character",
      "Note behavioral cues: body language, approach patterns, blocking exits",
      "Assess whether the main character recognizes the threat (fear cues, avoidance)"
    ],
    "success_criteria": ["threat_identified=true"],
    "abort_criteria": ["scene_transition=true"],
    "expected_duration": 8
  },
  "contract": {
    "skill_id": "vh_observe_suspicious_001",
    "eff_add": ["threat_identified", "entity_tracked"],
    "eff_del": ["environment_safe"],
    "eff_event": ["suspicious_movement_detected"],
    "n_instances": 12
  },
  "sub_episodes": [
    {
      "episode_id": "0at001QMutY",
      "seg_start": 0,
      "seg_end": 12,
      "summary": "Woman in elevator notices person with plastic bag on head in corridor",
      "intention_tags": ["OBSERVE", "REACT"],
      "outcome": "success",
      "quality_score": 0.85
    }
  ],
  "n_instances": 12,
  "retired": false
}
```

---

## Inference-Time Skill Retrieval

At inference time, the large VLM (72B) receives a new video and a question. The retrieval process:

### Step 1: Quick Grounding of the New Video

Run the same Stage 1 grounding (or a lighter version) on the new video to extract a summary set of predicates and scene descriptors. This can use a smaller model or even the 72B model's own first pass.

### Step 2: RAG Retrieval from Skill Bank

Query the skill bank using the grounded predicates + question keywords:

```python
query_predicates = {"confined_space": True, "person_count": 2, "threat_present": True}
query_question_type = "SR"  # Social Relationship question
query_text = "relationship between the woman and the person with plastic bag"

retrieved_skills = skill_bank.retrieve(
    predicates=query_predicates,
    tags=["OBSERVE", "INVESTIGATE", "INTERACT"],
    question_type=query_question_type,
    top_k=3
)
```

Retrieval uses a combination of:
- **Predicate matching**: skill preconditions vs. current video predicates
- **Tag matching**: skill tags vs. inferred intention from the question
- **Text similarity**: skill descriptions vs. question text (embedding cosine similarity)

### Step 3: Prompt Construction

Inject retrieved skills into the VLM prompt as structured reasoning guidance:

```
You are analyzing a video to answer a question. Use the following
reasoning skills to guide your analysis:

--- Skill 1: Observe Suspicious Behavior ---
When to apply: person_present=true AND confined_space=true
Steps:
  1. Scan for entities with unusual appearance or concealed identity
  2. Track entity movement relative to the main character
  3. Note behavioral cues: body language, approach patterns, exits
  4. Assess whether the main character recognizes the threat
Expect: threat_identified -> true, environment_safe -> false

--- Skill 2: Track Character Relationship ---
When to apply: person_count >= 2 AND social_cue=true
Steps:
  1. Identify all characters and their visual attributes
  2. Note interaction patterns: dialogue, proximity, body language
  3. Track power dynamics and emotional responses between characters
  4. Infer relationship type from accumulated evidence
Expect: relationship_identified -> true

Now analyze the video and answer:
Q: What is the relationship between the woman and the person
   wearing a plastic bag on their head?
```

The skills act as **structured chain-of-thought scaffolds** — the 72B model follows the steps and grounds its reasoning in the visual evidence the skills direct it to look for.

---

## Usage

```bash
# === OFFLINE: Build skill bank ===

# Process Video-Holmes (all 270 videos)
python video_skill_pipeline/run_pipeline.py \
    --dataset video_holmes \
    --output output/vh_skill_bank.jsonl

# Process M3-Bench robot domain (100 videos)
python video_skill_pipeline/run_pipeline.py \
    --dataset m3_bench_robot \
    --output output/m3_skill_bank.jsonl

# Quick test on one sample video
python video_skill_pipeline/run_pipeline.py \
    --dataset video_holmes \
    --max_videos 1 \
    --verbose

# === ONLINE: Inference with skill retrieval ===

# Answer Video-Holmes questions with skill-augmented 72B VLM
python video_skill_pipeline/inference.py \
    --skill_bank output/vh_skill_bank.jsonl \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --questions Video-Holmes/Benchmark/test_Video-Holmes.json \
    --top_k 3
```

---

## File Structure

```
video_skill_pipeline/
    config.py                 # Paths, model config, hyperparams
    data_structures.py        # VisualExperience, VisualSubEpisode, Skill mappings
    stage1_grounding.py       # Frame sampling + Qwen2.5-VL visual grounding
    stage2_segmentation.py    # Temporal segmentation + intention tagging
    stage3_skills.py          # Contract extraction + protocol generation + bank output
    run_pipeline.py           # Offline CLI: build skill bank from videos
    retrieval.py              # Online: RAG retrieval from skill bank
    inference.py              # Online: skill-augmented VLM inference
    prompt_builder.py         # Format retrieved skills into VLM prompts
    utils.py                  # Video I/O, SRT parsing, time helpers
    requirements.txt          # torch, transformers, qwen-vl-utils, decord, etc.
```

## Dependencies

**Offline (skill bank construction)**:
- **Model**: Qwen2.5-VL-7B-Instruct (local, at `transformers/models/Qwen/Qwen2.5-VL-7B-Instruct`)
- **Inference code reference**: `Qwen-VL/eval_code_qwen/inference_mass_qwen_25.py`
- **Python**: torch, transformers, qwen-vl-utils, decord (video decoding), Pillow
- **GPU**: 1x A100/H100 (24+ GB VRAM) for Qwen2.5-VL-7B inference

**Online (inference with skill retrieval)**:
- **Model**: Large VLM (72B) — Qwen2.5-VL-72B-Instruct, InternVL2.5-78B, or similar
- **Retrieval**: Embedding-based similarity search (sentence-transformers or Qwen3-Embedding)
- **GPU**: Multi-GPU setup for 72B model inference

## Relationship to COS-PLAY

| COS-PLAY Component | Video Pipeline Equivalent | Role at Inference |
|---------------------|--------------------------|-------------------|
| Game text state | VLM frame description (JSON) | Quick grounding of new video |
| `Experience` | `VisualExperience` (frame-level) | Frame-level context |
| `summary_state` (key=value) | Visual predicates (key=bool) | Retrieval query features |
| `_compute_state_delta` | `visual_state_delta` (predicate flips) | Segment boundary signals |
| `SubTask_Experience` | `VisualSubEpisode` (temporal segment) | Evidence backing a skill |
| `ScoredBoundary` | Multi-signal boundary score | Offline segmentation only |
| Intention `[TAG] phrase` | Video-adapted `[TAG] phrase` | Retrieval tag matching |
| `SkillEffectsContract` | Same schema, visual predicates | Expected state changes in prompt |
| `Protocol` | Same schema, visual steps | **Main prompt injection: reasoning scaffold** |
| `Skill` | Same schema, video evidence | Retrieved unit |
| `skill_bank.jsonl` | Same format, video-derived skills | Retrieval index |
| Decision agent queries skill bank | Large VLM queries skill bank | RAG retrieval + prompt augmentation |
