# 07_evaluation — Evaluation & Ablation Plan

> **Layer purpose:** Define how the system is **measured**, so that every claim about its working pieces can be isolated. The matrix here is what makes the [MVP success criterion](../00_overview/mvp_build_order.md#mvp-success-criterion) testable: external baselines, subsystem ablations, error buckets, cost / latency reporting, and the staged eval priority that mirrors the build order.

---

## Files in this folder

| File | What it is | When you read it |
|------|------------|------------------|
| [`evaluation_ablation_plan.md`](evaluation_ablation_plan.md) | The full **eval / ablation matrix**: external baselines (B0 direct large-VLM QA, B1 naive retrieval), core ablations (A0 full system through A9 composite-disabled), per-subsystem metrics (`hop_efficiency`, `trace_success_rate`, retrieval recall, evidence precision, perspective accuracy, abstention F1, final QA accuracy), error taxonomy (including `verifier_error`), cost / latency reporting (including `large_model_calls`), reporting template, and the **MVP Evaluation Priority** section. | When you run any evaluation; when you propose a new ablation; when you need to know what the reporting format must contain. |

---

## What this layer answers

| Question | Where in the plan |
|----------|-------------------|
| Does the full system beat external baselines? | Stage 8.1 — A0 vs B0 / B1 |
| Does each subsystem matter? | Stage 8.2 — A0 vs A1 (no memory), A2 (no state), …, A8 (no bank), A9 (no composites) |
| Where does the system fail? | §5 Error buckets — entity, perspective, retrieval, verifier, abstention, etc. |
| What does it cost? | §6 Cost / latency reporting — large-model calls, latency, token budget |
| What's the MVP eval cut? | §8 MVP Evaluation Priority |

## How this layer relates to the others

- **Inputs come from [`../03_controller/`](../03_controller/README.md)**: every metric is computed over `ReasoningTrace` objects ([`../00_overview/runtime_contracts.md`](../00_overview/runtime_contracts.md)).
- **Subjects of measurement come from every other layer**: grounding quality (01), memory write/read correctness (02), controller decisions (03), harness trace fidelity (04), skill applicability (05), training improvements (06).
- **Anti-hacking cross-link**: cost / latency reporting is required so reward gains in [`../06_training/`](../06_training/README.md) are not just "spent more compute" gains.

## What this layer does *not* do

- It does **not** define what to *do* on failure — that's the controller / harness.
- It does **not** propose new subsystems — it isolates the contribution of existing ones.
- It does **not** rank skills — that's the bank's job ([`../05_skills/`](../05_skills/README.md) usage stats and synthesizer's promotion thresholds).

## Open work

Lock the eval matrix per benchmark family, integrate the metric computation into the harness's `ReasoningTrace` consumer, and produce the first end-to-end report comparing A0 vs B0 / B1 on the long-video, multi-hop, perspective-bearing benchmarks listed in [`../01_grounding/video_benchmarks_grounding.md`](../01_grounding/video_benchmarks_grounding.md).
