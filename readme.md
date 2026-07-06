# Video_Skills

> **Evidence-grounded multi-hop video reasoning, orchestrated by a small trainable controller over a stable structured memory of grounded perception.**

Video_Skills is a research codebase for long-horizon, evidence-grounded video understanding. The system answers questions about long, multi-character, perspective-bearing videos by:

1. **Grounding** the video into a structured `SocialVideoGraph` of entities, interactions, events, and social/spatial state — using **frozen large VLMs (72B / 32B)** as visual specialists,
2. **Storing** that grounded perception in a fixed three-store memory (episodic / semantic / state) with an evidence layer,
3. **Reasoning** in short, verifiable hops orchestrated by a **trainable 8B controller** that decomposes the question, plans hops, picks reasoning skills from a curated bank, calls a retriever and a verifier, and decides answer-vs-abstain — emitting a fully typed `ReasoningTrace` along the way.

The complete design lives in [`infra_plans/`](infra_plans/README.md). This README is the **product-level entry point**; that folder is the **normative source of truth**.

---

## Design principle (the one paragraph that drives everything else)

**Stable memory, evolving reasoning.** Large VLMs are frozen specialists, not orchestrators. Memory is a fixed procedural substrate, not a learned policy. The 8B controller is the only trainable component. The "evolving" layer is restricted to **reasoning skills**, and even there v1 is curated, not free-form. The first milestone is to prove this stack beats direct large-VLM QA and naive retrieval baselines on long, multi-hop, perspective-bearing video reasoning — *before* any self-evolution is enabled.

Detailed statement: [`infra_plans/00_overview/system_overview.md`](infra_plans/00_overview/system_overview.md), [`infra_plans/00_overview/mvp_build_order.md`](infra_plans/00_overview/mvp_build_order.md).

---

## Architecture at a glance

```
                       ┌──────────────────────────────────────┐
                       │  question + video                    │
                       └────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌──────────────────────────────────────────┐
                  │  8B Controller  (the only trainable)     │
                  │  decompose → plan hops → pick skill      │
                  │  → call retriever → judge sufficiency    │
                  │  → call verifier → answer / abstain      │
                  └──┬───────────┬───────────┬──────────────┘
                     │           │           │
                     ▼           ▼           ▼
        ┌───────────────┐ ┌───────────┐ ┌──────────────────┐
        │  Retriever    │ │ Verifier  │ │  Reasoning       │
        │  (in 03_      │ │ (in 03_   │ │  Skill Bank      │
        │  controller)  │ │ controller)│ │  (curated v1)    │
        └──────┬────────┘ └─────┬─────┘ └────────┬─────────┘
               │                │                │
               ▼                ▼                ▼
        ┌──────────────────────────────────────────────┐
        │  Harness  (deterministic interpreter)        │
        │  expand composites → bind slots → call       │
        │  primitives → log AtomicStepResult /         │
        │  HopRecord → emit ReasoningTrace             │
        └──┬─────────────┬───────────────────┬─────────┘
           │             │                   │
           ▼             ▼                   ▼
  ┌──────────────┐ ┌────────────────┐ ┌──────────────────┐
  │  Memory      │ │ Frozen 72B/32B │ │ Pure-reasoning   │
  │  Procedure   │ │ visual         │ │ functions from   │
  │  Registry    │ │ specialists    │ │ atomic skills    │
  │  (fixed v1)  │ │ (Observer-A/B, │ │                  │
  │              │ │  Reasoner)     │ │                  │
  └──────┬───────┘ └────────┬───────┘ └──────────────────┘
         │                  │
         ▼                  ▼
  ┌──────────────┐ ┌────────────────┐
  │ SocialVideo- │ │ Grounded       │
  │ Graph +      │ │ perception     │
  │ episodic /   │ │ (face / voice  │
  │ semantic /   │ │  / scene /     │
  │ state +      │ │  subtitle /    │
  │ evidence     │ │  entity        │
  │              │ │  resolution)   │
  └──────────────┘ └────────────────┘
```

Every cross-module hand-off uses one of seven **canonical typed objects** (`GroundedWindow`, `EvidenceBundle`, `HopGoal`, `AtomicStepResult`, `VerificationResult`, `AbstainDecision`, `ReasoningTrace`) — defined in [`infra_plans/00_overview/runtime_contracts.md`](infra_plans/00_overview/runtime_contracts.md).

---

## What's implemented today

The **Phase-1 substrate** of the reasoning runtime now exists as runnable Python in [`video_skills/`](video_skills/README.md) — canonical contracts, memory stores + procedures, skill bank + 14 atomics, retriever, verifier, harness, rule-based v0 controller, and the online serving loop. End-to-end smoke tests pass on synthetic windows (`pytest tests/video_skills` → 58/58 green). The trainable 8B controller, live 72B/32B tool calls, learned retrieval, and replay-based skill mining remain Phase-2 work, intentionally targeted at the freeze-line in `video_skills/contracts.py`.

| Layer | Plan | Code | Status |
|-------|------|------|--------|
| **Visual grounding** | [`infra_plans/01_grounding/`](infra_plans/01_grounding/README.md) | [`visual_grounding/`](visual_grounding/README.md) — schemas, segmenter, perception, local grounder, social video graph, benchmark adapters | Schema-only smoke test (`out/claude_grounding/`, `out/gpt4o_grounding/`); m3-agent–based pipeline plan in place; Phase 0 → 6 execution pending |
| **Canonical runtime contracts** | [`infra_plans/00_overview/runtime_contracts.md`](infra_plans/00_overview/runtime_contracts.md) | [`video_skills/contracts.py`](video_skills/contracts.py) — all 7 typed objects + supporting refs | **Implemented** (Phase-1 freeze-line) |
| **Agentic memory** | [`infra_plans/02_memory/`](infra_plans/02_memory/README.md) | [`video_skills/memory/`](video_skills/memory/) — Episodic / Semantic / State (Belief + Spatial) / Evidence stores, `EntityProfileRegistry` (union-find aliases), `MemoryProcedureRegistry` (9 fixed procedures + audit log) | **Implemented** (in-memory v1; full eviction / semantic-refresh policies still open) |
| **Reasoning skill bank** | [`infra_plans/05_skills/`](infra_plans/05_skills/README.md) | [`video_skills/skills/`](video_skills/skills/) — `SkillRecord` schema, `ReasoningSkillBank`, 14 curated v1 atomics across grounding / temporal / belief / perspective / evidence / decision | **Implemented** (atomics only; composite promotion still open) |
| **Retriever** | [`infra_plans/03_controller/`](infra_plans/03_controller/README.md) | [`video_skills/retriever.py`](video_skills/retriever.py) — rewrite, entity / time / perspective filters, dedup, broaden ladder, counter-retrieval | **Implemented** (lexical v1; learned reranker is Phase-2) |
| **Verifier** | [`infra_plans/03_controller/`](infra_plans/03_controller/README.md) | [`video_skills/verifier.py`](video_skills/verifier.py) — 6 named checks (`claim_evidence_alignment`, `evidence_sufficiency`, `counterevidence`, `temporal_consistency`, `perspective_consistency`, `entity_consistency`) with `support_threshold` / `abstain_threshold` gates | **Implemented** |
| **Harness** | [`infra_plans/04_harness/`](infra_plans/04_harness/README.md) | [`video_skills/harness.py`](video_skills/harness.py) — deterministic hop interpreter, atomic-step iteration, evidence binding, all writes routed through the procedure registry, trace logging | **Implemented** |
| **8B controller** | [`infra_plans/03_controller/`](infra_plans/03_controller/README.md) | [`video_skills/controller.py`](video_skills/controller.py) — rule-based v0 (question analysis, hop planning, skill routing, answer composition) | **Stubbed** (deterministic placeholder; trainable 8B is Phase-2) |
| **Online serving loop** | [`infra_plans/03_controller/actors_reasoning_model.md`](infra_plans/03_controller/actors_reasoning_model.md) §2D | [`video_skills/loop.py`](video_skills/loop.py) — wires controller + harness + verifier + retriever; cross-hop carryover; produces a `ReasoningTrace` | **Implemented** |
| **Training (8B SFT → GRPO)** | [`infra_plans/06_training/`](infra_plans/06_training/README.md) | — | Staged plan in place; reward computation against `ReasoningTrace` pending |
| **Evaluation & ablation** | [`infra_plans/07_evaluation/`](infra_plans/07_evaluation/README.md) | [`tests/video_skills/`](tests/video_skills/) (per-module + end-to-end smoke) and [`tests/visual_grounding/`](tests/visual_grounding/) | Eval matrix and baselines defined; first end-to-end A0 vs B0 / B1 report pending |

The detailed open-work backlog and a full doc-vs-code traceability matrix live in [`infra_plans/99_meta/plan_docs_implementation_checklist.md`](infra_plans/99_meta/plan_docs_implementation_checklist.md); higher-level "middle-layer glue" vs "later-phase capability" grouping is in [`infra_plans/00_overview/system_overview.md`](infra_plans/00_overview/system_overview.md).

---

## Repository layout

```
Video_Skills/
├── readme.md                       # this file
├── infra_plans/                    # design plans (normative; read this folder)
│   ├── README.md                   # plan index + reading order
│   ├── 00_overview/                # system overview, MVP build order, runtime contracts
│   ├── 01_grounding/               # benchmark landscape + grounding pipeline plan
│   ├── 02_memory/                  # three-store memory + fixed Memory Procedure Registry
│   ├── 03_controller/              # 8B controller + retriever + verifier + training signals
│   ├── 04_harness/                 # execution runtime spec + MCP terminology map
│   ├── 05_skills/                  # reasoning skill bank + atomic / composite design + synthesis
│   ├── 06_training/                # staged SFT → GRPO plan for the 8B controller
│   ├── 07_evaluation/              # eval matrix + ablations + MVP eval priority
│   └── 99_meta/                    # implementation checklist
├── video_skills/                   # Phase-1 reasoning runtime (contracts, memory, skills,
│                                   #   retriever, verifier, harness, controller, loop)
├── visual_grounding/               # grounding-layer code (in progress)
├── out/                            # grounded outputs (claude_grounding, gpt4o_grounding)
├── tests/                          # pytest suite (video_skills + visual_grounding)
├── configs/                        # YAML configs
├── scripts/                        # entry-point scripts
├── install/                        # environment setup
├── pyproject.toml
├── requirements.txt
├── INSTALL.md
└── LICENSE
```

---

## MVP success criterion

> Outperform **direct large-VLM QA** (B0) and **naive retrieval over raw clips** (B1) on **evidence-grounded multi-hop video reasoning**, on long-video, perspective-bearing benchmarks (Video-Holmes, SIV-Bench, VRBench, LongVideoBench, CG-Bench, M3-Bench), using the 8B controller over structured memory and grounded evidence.

Defined in [`infra_plans/00_overview/mvp_build_order.md`](infra_plans/00_overview/mvp_build_order.md#mvp-success-criterion). Measured by the matrix in [`infra_plans/07_evaluation/evaluation_ablation_plan.md`](infra_plans/07_evaluation/evaluation_ablation_plan.md).

The MVP is **not** measured by skill-bank size, **not** by self-evolution metrics, **not** by per-component cleverness. The system either beats the baselines on grounded multi-hop reasoning, or it does not.

---

## Build order (3 phases)

From [`infra_plans/00_overview/mvp_build_order.md`](infra_plans/00_overview/mvp_build_order.md):

1. **Phase 1 — Stable Substrate.** Canonical runtime schemas → visual grounding layer → fixed memory procedures → retriever / verifier → harness runtime → curated atomic reasoning skill set → 8B controller training (SFT then GRPO). No free bank evolution. No adaptive memory. No aggressive synthesis.
2. **Phase 2 — Limited Reuse.** Conservative composite reasoning skills, limited promotion from repeated successful atomic chains, stronger verification / abstention, better trace export and failure localization.
3. **Phase 3 — Controlled Evolution.** Trace-based synthesis behind a feature flag, patch / split / retire policies, bank versioning and rollback, broader cross-benchmark transfer.

Phase 2 only begins when Phase 1 evaluation is reproducible. Phase 3 only begins when the [synthesis preconditions](infra_plans/05_skills/skill_synthetics_agents.md#03-preconditions-for-later-self-evolution) are demonstrably in place.

---

## Quick start

> The Phase-1 reasoning substrate (`video_skills/`) is runnable end-to-end on synthetic windows. The grounding layer (`visual_grounding/`) is the most mature perception component. Live integration of the two and the trainable 8B controller follow the build order above.

### Install

```bash
cd Video_Skills
pip install -e .
# or:
pip install -r requirements.txt
```

See [`INSTALL.md`](INSTALL.md) for full setup including conda environments and CUDA notes.

### Run the Phase-1 reasoning runtime

```python
from video_skills import build_runtime, run_question
from tests.video_skills.synthetic import make_alice_bob_key_window

rt = build_runtime()

rt.memory_procedures.update_entity_profile(
    entity_id="alice", canonical_name="Alice", seen_at=10.0,
)
rt.memory_procedures.update_entity_profile(
    entity_id="bob", canonical_name="Bob", seen_at=10.0,
)
rt.memory_procedures.append_grounded_event(window=make_alice_bob_key_window())

trace = run_question(
    rt,
    "Did Alice pick up the key before Bob entered?",
    target_entities=["alice", "bob"],
)
print(trace.answer)                     # "Order: a_before_b."
print(trace.final_verification.passed)  # True
print(trace.bank_skill_ids_used)        # ['atom.ground_event_span', 'atom.order_two_events']
```

Every cross-module hand-off in this loop is one of the seven canonical typed objects in `video_skills/contracts.py`. See [`video_skills/README.md`](video_skills/README.md) for the full subsystem map.

### Run the visual grounding layer

```python
from visual_grounding import build_grounded_context

# Short video — grounded state stays in-context
ctx = build_grounded_context(
    "clip.mp4",
    mode="auto",
    vlm_fn=my_vlm_callable,           # any 32B+ VLM that can describe frames
)
print(ctx.as_reasoner_text())          # consumed by the [Think]/[Answer] loop

# Long video — grounded state goes into a SocialVideoGraph for retrieval
graph = build_grounded_context(
    "movie.mp4",
    subtitle_path="movie.srt",
    mode="auto",
    vlm_fn=my_vlm_callable,
    embedder=my_text_embedder,
)
for node, score in graph.search("who left the room first?"):
    print(score, node.node_type, node.text)
```

Six benchmark adapters (Video-Holmes, SIV-Bench, VRBench, LongVideoBench, CG-Bench, M3-Bench) wrap this with benchmark-specific defaults — see [`visual_grounding/README.md`](visual_grounding/README.md).

### Tests

```bash
pytest tests/                  # everything
pytest tests/video_skills -q   # Phase-1 runtime only (58 tests, ~0.1s)
```

---

## Where to read next

| If you want to … | Read |
|------------------|------|
| Understand the whole system in one sitting | [`infra_plans/00_overview/system_overview.md`](infra_plans/00_overview/system_overview.md) |
| Know what gets built first | [`infra_plans/00_overview/mvp_build_order.md`](infra_plans/00_overview/mvp_build_order.md) |
| Implement any subsystem (need wire format) | [`infra_plans/00_overview/runtime_contracts.md`](infra_plans/00_overview/runtime_contracts.md) |
| Run the Phase-1 reasoning runtime end-to-end | [`video_skills/README.md`](video_skills/README.md) |
| Work on the grounding layer | [`infra_plans/01_grounding/`](infra_plans/01_grounding/README.md) |
| Work on memory | [`infra_plans/02_memory/`](infra_plans/02_memory/README.md) |
| Work on the 8B controller / retriever / verifier | [`infra_plans/03_controller/`](infra_plans/03_controller/README.md) |
| Work on the harness / execution runtime | [`infra_plans/04_harness/`](infra_plans/04_harness/README.md) |
| Work on reasoning skills | [`infra_plans/05_skills/`](infra_plans/05_skills/README.md) |
| Train the 8B controller | [`infra_plans/06_training/`](infra_plans/06_training/README.md) |
| Set up evaluation / ablations | [`infra_plans/07_evaluation/`](infra_plans/07_evaluation/README.md) |
| Triage open work | [`infra_plans/99_meta/plan_docs_implementation_checklist.md`](infra_plans/99_meta/plan_docs_implementation_checklist.md) |

---

## License

MIT — see [`LICENSE`](LICENSE).
