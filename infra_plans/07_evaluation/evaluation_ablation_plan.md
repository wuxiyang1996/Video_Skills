# Evaluation & Ablation — Plan

> Goal: Define an implementation-oriented evaluation protocol that proves the **value of each subsystem** in the Video_Skills stack — memory, grounding, retrieval, verifier, abstention, skill bank, and the atomic/composite skill structure — across short- and long-video benchmarks.
>
> **Related plans:**
> - [Actors / Reasoning Model](../03_controller/actors_reasoning_model.md) — controller, canonical runtime contracts, retriever, verifier, training signals
> - [Agentic Memory](../02_memory/agentic_memory_design.md) — three stores + lifecycle policies
> - [Atomic Skills & Hop Refactor — Harness](../04_harness/atomic_skills_hop_refactor_execution_plan.md) — harness, hop, trace logging
> - [Skill Extraction / Bank](../05_skills/skill_extraction_bank.md) — `SkillRecord`, starter inventory
> - [Skill Synthetics Agents](../05_skills/skill_synthetics_agents.md) — trace-first synthesis, promotion thresholds, bank versioning
> - [Video Benchmarks & Grounding](../01_grounding/video_benchmarks_grounding.md) — benchmark landscape
> - [MVP Build Order](../00_overview/mvp_build_order.md) — phase 1 / 2 / 3 sequencing this plan validates
> - [Runtime Contracts](../00_overview/runtime_contracts.md) — canonical objects every metric is computed over
> - [Training Plan: SFT to GRPO](../06_training/training_plan_sft_grpo.md) — training stages this plan provides eval slices for

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
| **A9 — Composite-disabled** | Atomic selection enabled; composite expansion blocked at the harness | Composite skill use (atomics only at runtime) | Isolates the runtime-time benefit of composite expansion vs the offline benefit of composite mining |

### 2.1 External baselines (non-ablation arms)

These are not ablations of the system — they are the **external baselines** the [MVP success criterion](../00_overview/mvp_build_order.md#mvp-success-criterion) is defined against. They run on the same benchmarks and are reported alongside A0–A9.

| Arm | Configuration | What it represents |
|---|---|---|
| **B0 — Direct large-VLM QA** | Frozen 72B reasoner answers from raw frames + question; no controller, no memory, no retriever, no bank | The "just throw the big model at it" baseline |
| **B1 — Naive retrieval + 72B answer** | Single dense retrieval pass over a flat episodic buffer → 72B reasoner answers; no hop planning, no perspective handling, no verifier-driven abstention | The "RAG without orchestration" baseline |

A0 must beat both B0 and B1 on evidence-grounded multi-hop benchmarks for the MVP success criterion to be considered met (§8).

Each arm logs the same `ReasoningTrace` structure so per-arm metrics are computed by the same evaluator.

---

## 3. Subsystem Metrics

Eight **subsystem metrics** are computed from `ReasoningTrace` plus benchmark gold annotations. They are reported per arm and per benchmark family (§4).

| Metric | Definition | Source fields |
|---|---|---|
| **Retrieval recall** | `|EvidenceBundle.refs ∩ gold_evidence| / |gold_evidence|` averaged over hops | per-hop `EvidenceBundle.refs`, gold evidence ids |
| **Evidence precision** | `|EvidenceBundle.refs ∩ gold_evidence| / |EvidenceBundle.refs|` averaged over hops | same as above |
| **Entity resolution accuracy** | Fraction of `character_id` resolutions that match gold identity (per detected entity mention) | `EntityProfile` reads logged in `AtomicStepResult.inputs` |
| **Perspective accuracy** | For perspective-bound questions: fraction where `perspective_consistency` check passed AND final answer respected the anchored character's view | `VerificationResult.checks`, `final_v` |
| **Abstention F1** | Standard F1 on the binary "should-abstain" label (gold) vs `AbstainDecision.abstain` | `ReasoningTrace.abstain` |
| **Hop efficiency** | Fraction of hops where `hop_verification.passed=True` AND `score` improved over the previous hop by ≥ `δ_hop`; complement is the "no-progress hop rate" | `ReasoningTrace.hops`, per-hop `VerificationResult.score` |
| **Trace success rate** | Fraction of questions where the trace closed via `compose_answer` with `final_verification.passed=True` (vs `blocked` or `abstain` on answerable questions) | `ReasoningTrace.final_verification`, `HopRecord.outcome` |
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
| **Verifier error** | Verifier itself was wrong: passed a check that gold marks failing, or failed a check that gold marks passing (false-pass / false-fail) | `VerificationResult.checks` vs gold check labels (when annotated) |
| **Unsupported answer** | Final answer's `claim_evidence_alignment` failed; or `final_evidence.refs` empty | `verify_final` output |
| **Abstention failure** | Either (a) abstained on an answerable question, or (b) answered an unanswerable question | `AbstainDecision.abstain` vs gold answerability |

Per-arm error bucket distributions are reported alongside QA accuracy; an arm that wins on accuracy but shifts errors into "unsupported answer" or "abstention failure" is flagged.

---

## 6. Cost and Latency Reporting

Every evaluation run reports cost alongside accuracy. This prevents the system from claiming gains that come from spending more compute and is required to keep the *Anti-Hacking Constraints* in [Actors §2E.2](../03_controller/actors_reasoning_model.md#2e2-anti-hacking-constraints) honest.

| Metric | Source | Reported as |
|---|---|---|
| **Retrieval cost** | `cost.retrieval_calls` per question | mean, p50, p95 |
| **Hop count** | `cost.hops` per question | mean, p50, p95 |
| **Atomic step count** | sum of `len(HopRecord.steps)` per question | mean, p50, p95 |
| **Large-model calls** | count of 72B / 32B invocations per question (grounding + reasoner combined, with split reported) | mean, p50, p95 |
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
entity_acc | perspective_acc | abstention_f1 | hop_efficiency | trace_success_rate |
err_grounding | err_retrieval | err_perspective | err_reasoning | err_verifier |
err_unsupported | err_abstention |
cost_retrieval | cost_hops | cost_atomic_steps | cost_large_model_calls |
latency_p50 | latency_p95 | tokens | mem_footprint
```

This is the canonical row format; downstream tooling (notebooks, leaderboards) reads from it without further normalization. Both ablation arms (A0–A9) and external baselines (B0, B1) use this row shape so a single comparison table covers all configurations.

---

## 8. MVP Evaluation Priority

Evaluation is staged to mirror [MVP Build Order](../00_overview/mvp_build_order.md): the first thing the protocol must establish is the MVP success criterion, before any of the more elaborate ablations are interpreted.

### 8.1 First priority — prove the MVP success criterion

The first eval pass must show that **A0 (full system)** beats both **B0 (direct large-VLM QA)** and **B1 (naive retrieval + 72B answer)** on evidence-grounded multi-hop video reasoning, with reviewable traces. Concretely:

- Run A0, B0, B1 on the long-video, multi-hop, perspective-bearing benchmarks listed in [Video Benchmarks & Grounding](../01_grounding/video_benchmarks_grounding.md).
- Report `qa_acc`, `evidence_precision`, `perspective_acc`, `abstention_f1`, `trace_success_rate`, plus `cost_large_model_calls` and `latency_p50` for each.
- The MVP success criterion is met when A0 wins on `qa_acc` and on at least one of (`evidence_precision`, `perspective_acc`) **without** spending strictly more `cost_large_model_calls` than B0.

Until this is demonstrated, the more granular ablations below are diagnostic, not headline results.

### 8.2 Second priority — subsystem ablations (A1–A9)

Once the MVP criterion is met, run A1–A9 to establish per-subsystem credit:

- **memory** (A1, A2, A3) — long-video and perspective benchmarks
- **verifier / abstention** (A4, A5) — evidence-attribution and abstention-bearing benchmarks
- **bank structure** (A6, A7, A8, A9) — full benchmark sweep with cost-normalized deltas

Report each subsystem's contribution as the (arm vs A0) delta on its primary stress benchmarks (§4 mapping), paired with cost deltas (§6).

### 8.3 Third priority — controlled-evolution evaluation (later phases)

Only after Stages 8.1 and 8.2 are stable and reproducible should evaluation cover the [Phase 3 controlled-evolution capabilities](../00_overview/mvp_build_order.md#phase-3-controlled-evolution): trace-based synthesis, patch / split / retire policies, bank versioning / rollback, and broader cross-benchmark transfer testing. Those experiments are out of scope for the MVP eval pass and have their own success criteria defined alongside the synthesizer rollout ([Skill Synthetics §0.3](../05_skills/skill_synthetics_agents.md#03-preconditions-for-later-self-evolution)).
