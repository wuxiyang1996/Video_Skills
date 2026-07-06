# Dataset Examples

Sample data from the **Video-Holmes** and **M3-Bench** datasets used in this project.

---

## Video-Holmes

**Paper:** [Video-Holmes: Can MLLM Think Like Holmes for Complex Video Reasoning?](http://arxiv.org/abs/2505.21374)
**Source:** [HuggingFace — TencentARC/Video-Holmes](https://huggingface.co/datasets/TencentARC/Video-Holmes)

A benchmark for evaluating complex video reasoning in multimodal LLMs. It contains **1,837 questions** across **270 suspense short films** (1–5 min each), spanning 7 task types that require models to locate and connect visual clues scattered across video segments.

### Question Types

| Code | Task | Count |
|------|------|-------|
| SR | Social Relationship | 292 |
| IMC | Implicit Mental State & Causal Reasoning | 276 |
| TCI | Temporal Causal Inference | 273 |
| TA | Temporal Arrangement | 200 |
| MHR | Multimodal Hidden Reasoning | 332 |
| PAR | Pattern Recognition | 194 |
| CTI | Core Theme Identification | 270 |

### Example: `0at001QMutY`

A suspense short film about an encounter in an elevator. The included files are:

- **`0at001QMutY.mp4`** — the video clip
- **`0at001QMutY.json`** — human annotation with segment descriptions, key relationships, inference shots, and core theme
- **`0at001QMutY_questions.json`** — 6 multiple-choice questions (one per task type except PAR), each with answer and explanation

Sample question (TCI — Temporal Causal Inference):
> *"What is the direct reason why the elevator is ultimately empty?"*
> Answer: **F** — "The plastic bag man killed the woman."

### Data Format

Each question entry:
```json
{
  "video ID": "0at001QMutY",
  "Question ID": 37,
  "Question Type": "TCI",
  "Question": "What is the direct reason why the elevator is ultimately empty?",
  "Options": { "A": "...", "B": "...", ... },
  "Answer": "F",
  "Explanation": "When the elevator returned to the first floor and there was no one inside, ..."
}
```

Each annotation entry contains: `Segment Description`, `Key Relationships`, `Inference Shots`, `Supernatural Elements`, and `Core Theme`.

---

## M3-Bench

**Paper:** [arxiv.org/abs/2508.09736](https://arxiv.org/abs/2508.09736)

A benchmark for memory-augmented multimodal understanding with two domains:

| Domain | Videos | Memory Graphs | Subtitles |
|--------|--------|---------------|-----------|
| **Robot** | 100 long-form videos | 100 `.pkl` files | 100 `.srt` files |
| **Web** | YouTube videos (by ID) | 920 `.pkl` files | — |

Robot videos cover 7 room types: `bedroom`, `gym`, `kitchen`, `living_room`, `meeting_room`, `office`, `study`.

### Example: Robot — `bedroom_01`

- **`bedroom_01.mp4`** — long-form robot interaction video (symlink)
- **`bedroom_01.srt`** — subtitle/dialogue transcript (symlink)

Sample dialogue from the subtitle:
```
Lily: What have you prepared for me for my afternoon tea?
Robot: There are coffee, fries and cake with two types of cake: strawberry and banana.
Lily: Emm well it is ten to four now, please give me a mocha first.
```

### Example: Web — `02I8Ad7qkjQ`

- **`02I8Ad7qkjQ.pkl`** — memory graph (symlink, requires `videograph` module to load)

Web domain videos are sourced from YouTube and referenced by video ID.

---

## Directory Structure

```
dataset_examples/
├── README.md
├── video_holmes/
│   ├── 0at001QMutY.mp4              # sample video
│   ├── 0at001QMutY.json             # human annotation
│   └── 0at001QMutY_questions.json   # 6 multiple-choice questions
└── m3_bench/
    ├── robot/
    │   ├── bedroom_01.mp4 -> ...    # symlink to video
    │   └── bedroom_01.srt -> ...    # symlink to subtitle
    └── web/
        └── 02I8Ad7qkjQ.pkl -> ...   # symlink to memory graph
```

> **Note:** M3-Bench files are symlinks to the original dataset to avoid duplicating large video files.
