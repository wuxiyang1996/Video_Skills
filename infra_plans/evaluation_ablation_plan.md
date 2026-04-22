# Evaluation & Ablation — Plan

> Goal: Define an implementation-oriented evaluation protocol that proves the **value of each subsystem** in the Video_Skills stack — memory, grounding, retrieval, verifier, abstention, skill bank, and the atomic/composite skill structure — across short- and long-video benchmarks.
>
> **Related plans:**
> - [Actors / Reasoning Model](actors_reasoning_model.md) — controller, canonical runtime contracts, retriever, verifier, training signals
> - [Agentic Memory](agentic_memory_design.md) — three stores + lifecycle policies
> - [Atomic Skills & Hop Refactor — Harness](atomic_skills_hop_refactor_execution_plan.md) — harness, hop, trace logging
> - [Skill Extraction / Bank](skill_extraction_bank.md) — `SkillRecord`, starter inventory
> - [Skill Synthetics Agents](skill_synthetics_agents.md) — trace-first synthesis, promotion thresholds, bank versioning
> - [Video Benchmarks & Grounding](video_benchmarks_grounding.md) — benchmark landscape

---

## 1. Evaluation Goals

Evaluation must move beyond a single end-to-end QA accuracy number. Each subsystem in the design carries a hypothesis; the protocol below is built to falsify or confirm those hypotheses individually and together.

| Subsystem | Hypothesis to test | What evaluation must show |
|---|---|---|
| **Memory** (episodic / semantic / state) | Compact 3-store memory beats a flat episodic-only buffer for long videos | Long-video QA accuracy with each store ablated |
| **Grounding** | Structured grounded windows beat raw frame dumps for both short and long video | QA + grounding-error rate across both regimes |
| **Retrieval** | First-class retriever (rewriting, perspective, counter-evidence) outperforms a thin `search_memory` wrapper | Retrieval recall / precision and downstream QA |
| **Verifier** | Local + final verification reduces unsupported answers and improves abstention | Unsupported-answer rate, abstention F1 |
| **Abstention** | Trained abstention beats forced answering on unanswerable / low-evidence questions | Abstention F1; QA on answerable subset preserved |
| **Skill bank** | A versioned bank of reasoning operators improves QA over an LLM-only baseline | QA delta from bank on/off, with cost held constant |
| **Atomic / composite structure** | Composites built from successful atomic chains generalize across tasks | Atomic-only vs composite-only vs joint comparisons; transfer-rate per composite |

Every reported number must be paired with cost (§7) so accuracy gains from extra hops are not double-counted.

---

## 2. Core Ablations

The eight ablation arms below are run with the same model checkpoints and the same data splits. Each arm produces a `ReasoningTrace` per question, so all subsystem metrics (§3) are reproducible per-arm.

| Arm | Configuration | What is removed | Expected pattern |
|---|---|---|---|
| **A0 — Full system** | Full controller + memory + grounding + retriever + verifier + abstention + bank (atomic + composite) | nothing | Baseline; best on all metrics |
| **A1 — No memory** | Disable episodic / semantic / state writes; controller has only the in-context window | All persistent memory | Sharp drop on long-video benchmarks; minimal effect on short-video |
| **A2 — No state memory** | Episodic + semantic kept; state subfields disabled | `state.social`, `state.spatial` | Drop on perspective / belief tracking benchmarks |
| **A3 — No entity resolver** | `EntityProfile` index disabled; no alias / face / voice unification | Entity-centric indexing | Drop on entity resolution and any multi-character question |
| **A4 — No verifier** | Verifier returns `passed=True, next_action="continue"` for all checks | All local + final verification | Higher unsupported-answer rate; near-zero abstention |
| **A5 — No abstention** | Force answering even when verifier requests abstain | `decide_answer_or_abstain` clamped to "answer" | Higher false-confident answers on unanswerable subset |
| **A6 — No skill bank** | Controller calls atomic primitives directly via free-form prompting; no composite skills | `SkillBank` (both atomic and composite records) | Loss of stable hop structure; higher hop count and latency |
| **A7 — Atomic-only** | Bank with atomic skills only; no composites loaded | All composites | Tests whether composites add value over equivalent atomic chains |
| **A8 — Composite-only** | Bank loaded with composites; atomic selection blocked | Direct atomic invocation | Tests degradation when composites must cover all cases without atomic fallback |

Each arm logs the same `ReasoningTrace` structure so per-arm metrics are computed by the same evaluator.

---

## 3. Subsystem Metrics

Six **subsystem metrics** are computed from `ReasoningTrace` plus benchmark gold annotations. They are reported per arm and per benchmark family (§4).

| Metric | Definition | Source fields |
|---|---|---|
| **Retrieval recall** | `|EvidenceBundle.refs ∩ gold_evidence| / |gold_evidence|` averaged over hops | per-hop `EvidenceBundle.refs`, gold evidence ids |
| **Evidence precision** | `|EvidenceBundle.refs ∩ gold_evidence| / |EvidenceBundle.refs|` averaged over hops | same as above |
| **Entity resolution accuracy** | Fraction of `character_id` resolutions that match gold identity (per detected entity mention) | `EntityProfile` reads logged in `AtomicStepResult.inputs` |
| **Perspective accuracy** | For perspective-bound questions: fraction where `perspective_consistency` check passed AND final answer respected the anchored character's view | `VerificationResult.checks`, `final_v` |
| **Abstention F1** | Standard F1 on the binary "should-abstain" label (gold) vs `AbstainDecision.abstain` | `ReasoningTrace.abstain` |
| **Final QA accuracy** | Benchmark-native metric (EM / MC accuracy / judge-equivalent) | `ReasoningTrace.answer` |

Reporting requirements:

- Recall and precision are computed only over hops whose gold evidence set is annotated; for benchmarks without gold evidence, these metrics are reported as "n/a" rather than approximated.
- Abstention F1 requires a binary "answerable" annotation per question; benchmarks without it are reported only on the abstention-rate side, not F1.
- All metrics are also reported as **delta vs A0** for ablation arms A1–A8 to make subsystem contribution explicit.

---

## 4. Benchmark-to-Metric Mapping

Different benchmarks stress different subsystems. The table below maps each benchmark family to the subsystem stresses it primarily tests; metrics from §3 are weighted accordingly when comparing arms.

| Benchmark family | Primary stresses | Key metrics |
|---|---|---|
| **Local grounding** (Video-Holmes, SIV-Bench short clips) | Grounding, in-context reasoning, perspective on small casts | Final QA, evidence precision, perspective accuracy |
| **Temporal ordering** (CG-Bench temporal subset, VRBench narrative ordering) | Temporal links, episodic chain, ordering atomics | Final QA, retrieval recall on temporal evidence, hop count |
| **Retrieval** (LongVideoBench QA, M3-Bench retrieval-heavy questions) | Retriever (rewriting, broaden, fusion), semantic memory | Retrieval recall + precision, retrieval cost |
| **Evidence attribution** (VRBench evidence-cited questions) | Verifier (`claim_evidence_alignment`), evidence-bearing outputs | Evidence precision, unsupported-answer rate |
| **Perspective tracking** (SIV-Bench ToM subset, MA-EgoQA) | State memory (social), perspective threads, belief skills | Perspective accuracy, final QA on perspective-bound subset |
| **Entity resolution** (M3-Bench multi-character, MA-EgoQA cross-clip identity) | EntityProfile, alias resolution, identity persistence | Entity resolution accuracy, retrieval recall |
| **Long-horizon memory** (M3-Bench long, LongVideoBench summarization) | Semantic memory refresh, compression, eviction | Final QA, retrieval recall on aged evidence, memory footprint |

This mapping is also the **selection rule for ablation reporting**: when reporting A1 (no memory), highlight long-horizon memory + retrieval benchmarks; when reporting A4 (no verifier), highlight evidence-attribution + abstention metrics.

---

## 5. Error Buckets

Failure analysis uses the same taxonomy as the harness's *Failure Localization Protocol* and the synthesizer's *Trace Localization Procedure*, kept aligned across the system.

| Bucket | Definition | Detected by |
|---|---|---|
| **Grounding error** | Cited evidence comes from a `GroundedWindow` whose `confidence < τ_grounding` or whose perception was wrong by gold | `EvidenceRef.provenance`, gold evidence labels |
| **Retrieval error** | Gold evidence existed in memory but `EvidenceBundle.refs` missed it; or empty after broaden | `EvidenceBundle.refs` vs gold; `failure_mode="empty_evidence"` |
| **Perspective error** | Final answer used global state where the question was perspective-bound; or `perspective_consistency` failed | `VerificationResult.checks`, gold perspective annotation |
| **Reasoning-chain error** | Per-hop `verification_failed` with non-empty evidence; correct evidence misused | `AtomicStepResult.failure_mode`, `VerificationResult` |
| **Unsupported answer** | Final answer's `claim_evidence_alignment` failed; or `final_evidence.refs` empty | `verify_final` output |
| **Abstention failure** | Either (a) abstained on an answerable question, or (b) answered an unanswerable question | `AbstainDecision.abstain` vs gold answerability |

Per-arm error bucket distributions are reported alongside QA accuracy; an arm that wins on accuracy but shifts errors into "unsupported answer" or "abstention failure" is flagged.

---

## 6. Cost and Latency Reporting

Every evaluation run reports cost alongside accuracy. This prevents the system from claiming gains that come from spending more compute and is required to keep the *Anti-Hacking Constraints* in [Actors §2E.2](actors_reasoning_model.md#2e2-anti-hacking-constraints) honest.

| Metric | Source | Reported as |
|---|---|---|
| **Retrieval cost** | `cost.retrieval_calls` per question | mean, p50, p95 |
| **Hop count** | `cost.hops` per question | mean, p50, p95 |
| **Atomic step count** | sum of `len(HopRecord.steps)` per question | mean, p50, p95 |
| **Latency** | `cost.latency_ms` per question (offline + online split) | mean, p50, p95 |
| **Token usage** | `cost.tokens` per question (controller + reasoner split) | mean, total per benchmark |
| **Memory footprint** | size of episodic / semantic / state stores per video at end-of-offline | per-store size, percentiles |

Cost-aware reporting rules:

- Each subsystem-level result table includes the matching cost column; for example, "Retrieval recall vs retrieval cost" is reported as a pair.
- Pareto curves (accuracy vs latency, accuracy vs retrieval cost) are reported on long-video benchmarks where cost varies most.
- For ablations that change cost (e.g., A6 No skill bank usually raises hop count), the cost-normalized accuracy delta is reported alongside the raw delta, so subsystem credit is not confounded with extra compute.

---

## 7. Reporting template

For each (arm, benchmark) cell, the evaluator emits a row with:

```
arm | benchmark | n | qa_acc | retrieval_recall | evidence_precision |
entity_acc | perspective_acc | abstention_f1 |
err_grounding | err_retrieval | err_perspective | err_reasoning |
err_unsupported | err_abstention |
cost_retrieval | cost_hops | cost_atomic_steps | latency_p50 | latency_p95 |
tokens | mem_footprint
```

This is the canonical row format; downstream tooling (notebooks, leaderboards) reads from it without further normalization.
