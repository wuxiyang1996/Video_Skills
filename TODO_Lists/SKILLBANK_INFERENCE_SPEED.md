# Skill Bank Inference Speed & Parallelism

**Created:** 2026-03-14  
**Status:** Open  
**Related:** `SKILLBANK_GRPO_PLAN.md`

---

## Problem

With GRPO on Stages 1–3 and a tool-calling agent on Stage 4, the skill bank pipeline makes **~572 LLM calls per EM step**. The current `MultiLoraSkillBankLLM.generate()` is single-request (batch size 1, HuggingFace `.generate()`), and the extraction script hard-codes `max_concurrent_llm_calls=1`. Everything is serial.

At ~50 tokens/sec output (Qwen3-14B, single request), one EM step takes **~47 minutes**. With 3 EM iterations per co-evolution step, that's over 2 hours just for skill bank updates. Unacceptable.

---

## Root Cause: Batch-1 Serving

```python
# Current: one prompt at a time, GPU idles between requests
inputs = self._tokenizer(prompt, return_tensors="pt")
output_ids = self._model.generate(input_ids=input_ids, ...)
```

The GPU can process hundreds of tokens concurrently across multiple requests, but HuggingFace `.generate()` only handles one at a time. The GPU utilization during inference is ~15-25%.

---

## Fix: vLLM Serving with Multi-LoRA + Async Batching

### Step 1: Serve LoRA adapters through vLLM

vLLM natively supports serving multiple LoRA adapters on the same base model. All adapters share base weights in GPU memory. Requests targeting different adapters get batched together.

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --enable-lora \
    --lora-modules \
        boundary=runs/lora_adapters/boundary \
        segment=runs/lora_adapters/segment \
        contract=runs/lora_adapters/contract \
        retrieval=runs/lora_adapters/retrieval \
    --max-loras 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --dtype auto \
    --trust-remote-code
```

Client specifies which adapter per request via `model="boundary"` in the OpenAI API call.

### Step 2: Async client to send requests concurrently

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

async def generate_batch(
    prompts: list[str],
    adapter: str,
    max_retries: int = 3,
    **kwargs,
) -> list[str]:
    """Fire all prompts concurrently and return responses in prompt order.

    Each prompt becomes an independent HTTP request to vLLM. asyncio.gather
    preserves the order of its input tasks, so responses[i] corresponds to
    prompts[i] regardless of which request finishes first on the server.
    """
    async def _single(prompt: str) -> str:
        for attempt in range(max_retries):
            try:
                r = await client.completions.create(
                    model=adapter,
                    prompt=prompt,
                    max_tokens=kwargs.get("max_tokens", 300),
                    temperature=kwargs.get("temperature", 0.7),
                )
                return r.choices[0].text
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (2 ** attempt))

    responses = await asyncio.gather(*[_single(p) for p in prompts])
    return list(responses)
```

`asyncio.gather` guarantees: `responses[i]` = result of `tasks[i]`, even if task 5 finishes before task 0 on the server. The ordering is tracked by the Python event loop, not by vLLM.

### Step 3: Remove `max_concurrent_llm_calls=1` cap

In `scripts/qwen3_skillbank_agent.py` line 891:
```python
# BEFORE:
max_concurrent_llm_calls=1,

# AFTER:
max_concurrent_llm_calls=None,  # let vLLM handle batching
```

---

## Per-Stage Parallelism Analysis

### Stage 1 — Boundary Proposal (GRPO)

| Item | Count | Independent? | Parallelizable? |
|------|-------|-------------|-----------------|
| Boundary prompts | 32 trajs × G=4 = 128 | Yes — each proposal is independent | All 128 fire concurrently |
| Reward evaluation (decode each proposal) | 128 decode runs | Yes — each DP decode is independent | `ProcessPoolExecutor` on CPU |
| GRPO parameter update | 1 | No — needs all rewards first | Sequential after eval |

**Projected time (vLLM):** 128 × 200 output tokens / 1000 tok/s ≈ **26s** LLM + 6s CPU = **~30s**

### Stage 2 — Decode (GRPO)

| Item | Count | Independent? | Parallelizable? |
|------|-------|-------------|-----------------|
| Decode prompts (re-ranking) | 32 trajs × G=8 = 256 | Yes | All 256 fire concurrently |
| Reward computation | 256 | Yes — CPU set operations | `ProcessPoolExecutor` |
| GRPO parameter update | 1 | No | Sequential |

**Projected time (vLLM):** 256 × 300 output tokens / 1000 tok/s ≈ **77s** LLM + 3s CPU = **~80s**

### Stage 3 — Contract Learning (GRPO)

| Item | Count | Independent? | Parallelizable? |
|------|-------|-------------|-----------------|
| Contract prompts | 42 skills × G=4 = 168 | Yes | All 168 fire concurrently |
| Holdout verification | 42 skills | Yes — pure set operations | Trivial CPU parallelism |
| GRPO parameter update | 1 | No | Sequential |

**Projected time (vLLM):** 168 × 200 output tokens / 1000 tok/s ≈ **34s** LLM + 1s CPU = **~35s**

### Stage 4 — Tool-Calling Agent

| Item | Count | Independent? | Parallelizable? |
|------|-------|-------------|-----------------|
| Chat turns | 10–20 | **No** — each turn depends on previous tool result | Sequential |
| Tool execution (trial merge/split) | 10–20 | No — sandboxed bank is shared state | Sequential |

**Projected time:** ~20 turns × (0.2s LLM + 2s tool execution) ≈ **~44s**

Stage 4 is inherently sequential but is also the cheapest stage. The tool execution (decode + contract on CPU) dominates, not LLM latency.

---

## Projected Timeline

### Per EM Iteration

| Stage | Serial (current) | vLLM batched (target) | Speedup |
|-------|-------------------|-----------------------|---------|
| Stage 1 GRPO | 512s | 30s | 17x |
| Stage 2 GRPO | 1,536s | 80s | 19x |
| Stage 3 GRPO | 672s | 35s | 19x |
| Stage 4 agent | 80s | 44s | 2x |
| GRPO updates | 45s | 45s | 1x |
| **Total** | **2,845s (47 min)** | **234s (~4 min)** | **~12x** |

### Per EM Step (3 iterations)

| | Serial | Batched |
|-|--------|---------|
| 3 iterations | 141 min | **~12 min** |

### Extraction Pipeline (non-training)

For skill extraction (`run_qwen3_skillbank_agent.sh`), the speedup is similar. Currently processes 8 games × 1 episode each, serial. With batching:

| | Serial | Batched + pipelined |
|-|--------|---------------------|
| 8 games, 1 episode each | ~20 min | **~3 min** |
| 8 games, 5 episodes each | ~100 min | **~12 min** |

---

## Cross-Stage Pipeline Parallelism

### The Idea

Instead of processing all 32 trajectories through Stage 1, waiting, then all through Stage 2, waiting, then all through Stage 3 — split into micro-batches and pipeline them through the stages. Since each stage uses a different LoRA adapter and vLLM serves all adapters concurrently, Stage 1 requests (boundary adapter) and Stage 2 requests (segment adapter) from different micro-batches land in the **same GPU batch**.

### Without Pipeline (stage-at-a-time)

```
GPU:  [====== Stage 1: 128 boundary calls ======][========== Stage 2: 256 segment calls ==========][===== Stage 3: 168 contract calls =====]
Time: |<------------- 26s ---------------------->|<------------------ 77s ----------------------->|<------------- 34s ------------------>|
                                                                                                                            Total: 137s
```

Between stages, the GPU waits while CPU computes next stage's inputs. Each stage fully drains before the next starts.

### With Pipeline (micro-batch overlap)

Split 32 trajectories into 4 micro-batches of 8. Each micro-batch flows through Stage 1 → 2 → 3 independently:

```
           t=0        t=7s       t=14s      t=21s      t=28s      t=35s      t=42s      t=55s
           ├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
GPU:       │ S1:MB1   │ S1:MB2   │ S1:MB3   │ S1:MB4   │          │          │          │
(boundary) │ 32 calls │ 32 calls │ 32 calls │ 32 calls │          │          │          │
           │          │          │          │          │          │          │          │
GPU:       │          │ S2:MB1   │ S2:MB2   │ S2:MB3   │ S2:MB4   │          │          │
(segment)  │          │ 64 calls │ 64 calls │ 64 calls │ 64 calls │          │          │
           │          │          │          │          │          │          │          │
GPU:       │          │          │          │ S3:MB1   │ S3:MB2   │ S3:MB3   │ S3:MB4   │
(contract) │          │          │          │ 42 calls │ 42 calls │ 42 calls │ 42 calls │
           ├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤

vLLM batch at t=14s:  S1:MB3 (32 boundary) + S2:MB1 (64 segment) = 96 concurrent requests
vLLM batch at t=21s:  S1:MB4 (32 boundary) + S2:MB2 (64 segment) + S3:MB1 (42 contract) = 138 concurrent requests ← peak
```

At peak, the GPU processes boundary, segment, and contract adapter requests **in the same forward pass**. vLLM's continuous batching handles this transparently — it doesn't care which LoRA adapter each request uses.

### Why This Works

1. **Different LoRA adapters coexist in vLLM**: with `--enable-lora --max-loras 4`, all four adapters are loaded simultaneously. A single vLLM batch can contain requests targeting different adapters. The base model forward pass is shared; only the LoRA delta differs per request.

2. **Higher GPU utilization**: in the stage-at-a-time approach, the GPU briefly idles between stages while CPU prepares the next stage's inputs. In the pipeline, those idle slots are filled by requests from the next micro-batch's earlier stage.

3. **CPU work overlaps with GPU work**: while vLLM generates Stage 2 outputs for micro-batch 1, CPU computes Stage 1's reward evaluation (DP decode) for micro-batch 1 AND formats Stage 1 prompts for micro-batch 2, all concurrently.

### Pipeline Speedup

| Approach | Total generation time | GPU idle between stages |
|----------|----------------------|------------------------|
| Stage-at-a-time | 137s | ~5s per boundary (3 gaps = 15s idle) |
| Pipeline (4 micro-batches) | **~95s** | Near-zero (stages overlap) |
| Pipeline (8 micro-batches) | **~85s** | Near-zero, better batch fill |

The pipeline doesn't reduce the total tokens processed — it reduces dead time by keeping the GPU busy with a mixed workload of adapters.

### Implementation: Async Pipeline Executor

```python
import asyncio
from typing import List, Any

class PipelinedEMExecutor:
    """Run Stages 1→2→3 with micro-batch pipeline parallelism.

    Ordering guarantee: within each micro-batch, Stage N+1 only starts
    after Stage N completes (enforced by await). Across micro-batches,
    all requests land in the same vLLM server and are continuously batched.
    """

    def __init__(self, vllm_client, bank, micro_batch_size: int = 8,
                 stagger_delay: float = 0.0):
        self.client = vllm_client
        self.bank = bank
        self.mbs = micro_batch_size
        self.stagger_delay = stagger_delay

    async def run_pipeline(self, trajectories: List[Any]) -> dict:
        micro_batches = [
            trajectories[i:i + self.mbs]
            for i in range(0, len(trajectories), self.mbs)
        ]

        all_s1_results = {}
        all_s2_results = {}
        all_s3_results = {}

        async def process_micro_batch(mb_idx, mb_trajs):
            # Stagger launch so micro-batches hit different stages at
            # different times → true multi-adapter overlap in vLLM
            if self.stagger_delay > 0 and mb_idx > 0:
                await asyncio.sleep(self.stagger_delay * mb_idx)

            # Stage 1: boundary proposals (G=4 per trajectory)
            s1_prompts = []
            for t in mb_trajs:
                for g in range(4):
                    s1_prompts.append(format_boundary_prompt(t, g))
            s1_outputs = await self.client.generate_batch(
                s1_prompts, adapter="boundary"
            )
            # ^^^ This await is the ordering barrier for THIS micro-batch.
            # MB1 cannot proceed to Stage 2 until all its S1 responses arrive.
            # Meanwhile, MB2/MB3/MB4 Stage 1 requests (or later-stage requests
            # from earlier MBs) are being served by vLLM concurrently.
            s1_parsed = parse_and_evaluate_boundaries(s1_outputs, mb_trajs)
            all_s1_results[mb_idx] = s1_parsed

            # Stage 2: segment decoding (G=8 per trajectory)
            s2_prompts = []
            for t, cuts in zip(mb_trajs, s1_parsed.best_cuts):
                candidates = dp_prefilter(t, cuts, self.bank)
                for g in range(8):
                    s2_prompts.append(format_decode_prompt(t, candidates, g))
            s2_outputs = await self.client.generate_batch(
                s2_prompts, adapter="segment"
            )
            s2_parsed = parse_and_evaluate_decode(s2_outputs, mb_trajs)
            all_s2_results[mb_idx] = s2_parsed

            # Stage 3: contract learning (G=4 per skill)
            skills_in_mb = collect_skills_from_decode(s2_parsed)
            s3_prompts = []
            for skill_id, instances in skills_in_mb.items():
                for g in range(4):
                    s3_prompts.append(format_contract_prompt(skill_id, instances, g))
            if s3_prompts:
                s3_outputs = await self.client.generate_batch(
                    s3_prompts, adapter="contract"
                )
                s3_parsed = parse_and_verify_contracts(s3_outputs, skills_in_mb)
                all_s3_results[mb_idx] = s3_parsed

        await asyncio.gather(*[
            process_micro_batch(i, mb)
            for i, mb in enumerate(micro_batches)
        ])

        return {
            "stage1": all_s1_results,
            "stage2": all_s2_results,
            "stage3": all_s3_results,
        }
```

**How vLLM continuous batching makes this work under the hood:**

When `generate_batch()` sends N HTTP requests to vLLM, they enter vLLM's **waiting queue**. vLLM's scheduler runs a tight loop:

```
while True:
    # 1. Check for new requests in the waiting queue
    new_reqs = dequeue_waiting(max=remaining_kv_capacity)
    running.extend(new_reqs)

    # 2. Run one forward pass on ALL running requests
    #    (different LoRA adapters in the same batch — base weights shared)
    logits = model.forward(running)

    # 3. Sample next token for each request
    for req in running:
        token = sample(logits[req], temperature=req.temperature)
        req.append(token)
        if token == EOS or len(req.output) >= max_tokens:
            req.finish()
            running.remove(req)
```

Each iteration processes one token for EVERY in-flight request. New requests join mid-iteration — this is "continuous batching." A batch at any moment might contain:
- 32 boundary-adapter requests (MB1:S1) at token position 45
- 64 segment-adapter requests (MB0:S2) at token position 120
- 42 contract-adapter requests (MB0:S3) still prefilling

All sharing the same base Qwen3-14B weights, with only the LoRA delta differing per request. This is maximally GPU-efficient.

**Why order is preserved:** vLLM's scheduling is **per-request, not per-batch**. Each HTTP request gets its own response when its generation is done. There is no global "batch N done" signal. `generate_batch()` tracks individual request IDs and returns when ALL requests in its group have responses. The grouping is application-level (Python), not vLLM-level.

### Why the Order Can Never Be Messed Up

There is no "vLLM pipeline" feature — vLLM is a stateless request server. The pipeline is an **application-level pattern** built on two orthogonal mechanisms:

| Layer | Mechanism | What it guarantees |
|-------|-----------|-------------------|
| **Python (asyncio)** | `await` per stage in each micro-batch coroutine | Stage N+1 only starts after Stage N returns, per micro-batch |
| **vLLM (continuous batching)** | Dynamically inserts arriving requests into the current GPU batch | Requests from different stages/micro-batches share GPU forward passes |

Neither layer knows about the other. Python doesn't know about GPU batching. vLLM doesn't know about stage dependencies. Together they produce correct pipelining.

**Concrete trace of a 2-micro-batch pipeline:**

```
t=0.0s  Python: MB1 sends 32 S1 requests (boundary adapter) → vLLM queue
        Python: MB2 is at "await S1" — its coroutine hasn't started yet? No, it HAS started
                but it's also awaiting its own S1 generate_batch
        vLLM:   batch contains [MB1:S1 x32] + [MB2:S1 x32] = 64 boundary requests
                (both micro-batches' Stage 1 requests land in the SAME GPU batch)

t=6.5s  vLLM:   all 64 S1 requests complete, responses returned via HTTP
        Python: MB1 coroutine resumes — parses S1 output, builds S2 prompts
        Python: MB2 coroutine resumes — parses S1 output, builds S2 prompts
        (both happen concurrently on the CPU — asyncio event loop)

t=6.8s  Python: MB1 sends 64 S2 requests (segment adapter) → vLLM queue
        Python: MB2 sends 64 S2 requests (segment adapter) → vLLM queue
        vLLM:   batch contains [MB1:S2 x64] + [MB2:S2 x64] = 128 segment requests

t=26s   vLLM:   all 128 S2 requests complete
        Python: MB1 builds S3 prompts from its S2 output
        Python: MB2 builds S3 prompts from its S2 output
        (same pattern continues)
```

Notice: with 2 micro-batches, the pipeline is "wide" rather than "staggered" — both MBs hit the same stage at roughly the same time because Stage 1 runs equally fast for both. The staggered pattern from the diagram above only emerges when micro-batches have **unequal processing times** (e.g., different trajectory lengths) or when we deliberately stagger their launch with `asyncio.sleep`:

```python
async def process_micro_batch(mb_idx, mb_trajs, stagger_delay: float = 0.0):
    await asyncio.sleep(stagger_delay * mb_idx)  # stagger launch
    s1 = await generate_batch(s1_prompts, adapter="boundary")
    s2 = await generate_batch(build_s2(s1), adapter="segment")
    s3 = await generate_batch(build_s3(s2), adapter="contract")
    return s1, s2, s3

# With stagger_delay=5.0, MB2 launches 5s after MB1
# → MB1 is in Stage 2 while MB2 is in Stage 1 → true overlap
await asyncio.gather(*[
    process_micro_batch(i, mb, stagger_delay=5.0)
    for i, mb in enumerate(micro_batches)
])
```

**When does staggering actually help?** Only when multi-LoRA mixing creates a more efficient GPU batch than single-adapter batching. With vLLM's continuous batching, the overhead of mixing adapters is minimal (shared base weights, only LoRA delta differs). So staggering helps when:

- The total concurrent requests from one stage **exceed vLLM's batch capacity** (KV cache limited to ~60-80 concurrent requests on 80GB A100). Staggering spreads the peak across time.
- You want to hide CPU-side work (reward computation, prompt formatting) behind GPU generation time.

For our workload (128–256 requests per stage), the batch capacity is the bottleneck — vLLM processes them in 2-4 waves regardless. Staggering helps smooth those waves.

**What CAN go wrong (and the safeguards):**

| Failure mode | Safeguard |
|-------------|-----------|
| vLLM returns an error for some requests | `generate_batch()` retries failed requests up to 3 times; if still failing, the micro-batch raises and `asyncio.gather` propagates the exception |
| vLLM reorders responses | Impossible — each HTTP request gets its own response. We track prompt→response by request ID, not by position. |
| A micro-batch's S1 output is corrupt / empty | Parse-and-validate before building S2 prompts. If invalid, skip this micro-batch (log warning) rather than feeding garbage to S2. |
| LoRA adapter weights change mid-pipeline (during GRPO training) | GRPO weight updates happen in a separate phase AFTER all generation completes. Generation and training never overlap. |

### GRPO Training Constraint

For GRPO, the pipeline applies to the **generation + evaluation phase only**. The parameter update still needs all rewards:

```
Phase 1 (pipelined):   Generate + evaluate all micro-batches through S1→S2→S3
Phase 2 (sequential):  Compute GRPO advantages across all micro-batches
Phase 3 (sequential):  Update LoRA adapter weights (one stage at a time)
```

Reward dependencies across stages:
- Stage 1 reward needs Stage 2 decode quality → available after Phase 1 completes
- Stage 2 reward needs Stage 3 contract pass rate → available after Phase 1 completes
- Stage 3 reward needs holdout verification → computed in Phase 1

So all rewards are available at the end of Phase 1. Phase 2+3 are the same as before.

### Extraction Pipeline (non-training)

For extraction there are no GRPO updates, so the pipeline is **fully unconstrained**. Each episode flows through all stages independently:

```python
# Multi-episode pipelined extraction
async def extract_all(episodes):
    async def extract_one(ep):
        s1 = await stage1_boundary(ep)     # boundary adapter
        s2 = await stage2_decode(ep, s1)   # segment adapter
        s3 = await stage3_contract(ep, s2) # contract adapter
        return s1, s2, s3

    return await asyncio.gather(*[extract_one(ep) for ep in episodes])
```

All episodes' LLM calls from all stages interleave freely in vLLM.

---

## Updated Projected Timeline (with Pipeline)

### Per EM Iteration

| Stage | Stage-at-a-time (batched) | Pipelined (micro-batch) | Improvement |
|-------|---------------------------|-------------------------|-------------|
| Stages 1+2+3 generation | 137s | ~85-95s | ~1.5x |
| Stage 4 agent | 44s | 44s (sequential) | — |
| GRPO updates | 45s | 45s | — |
| CPU reward eval | ~10s | overlapped with GPU | ~0s visible |
| **Total** | **~236s (4 min)** | **~180s (3 min)** | **1.3x** |

The pipeline saves ~50s per EM iteration by eliminating inter-stage idle time and improving GPU batch fill rate. Combined with vLLM batching (12x over serial), the total speedup over the current serial approach is **~16x**.

### Extraction (8 games, 5 episodes each)

| Approach | Wall time |
|----------|-----------|
| Serial (current) | ~100 min |
| vLLM batched, stage-at-a-time | ~12 min |
| vLLM batched + pipeline | **~7 min** |

---

## What Cannot Be Parallelized

| Component | Why sequential |
|-----------|---------------|
| Stage 4 tool-calling turns | Each turn depends on previous tool result |
| GRPO parameter updates | Gradient computation needs all group rewards |
| SkillEval gating | Must see final bank state after all 4 stages |
| Cross-iteration EM (iter 1 → 2 → 3) | Next iteration uses the bank from the previous |
| LoRA hot-reload after GRPO update | vLLM must reload updated adapter weights |

### LoRA Hot-Reload Overhead

After each GRPO update modifies a LoRA adapter's weights, vLLM must reload the adapter. vLLM supports this via its LoRA manager without restarting the server, but it incurs a brief pause (~1-2s per adapter). For 3 adapters updated per EM iteration, that's ~4-6s. Negligible.

---

## Implementation Checklist

### P0 — Critical (20x throughput)

- [ ] **Update vLLM launch script** — add `--enable-lora --lora-modules ... --max-loras 4` to `run_qwen3_skillbank_agent.sh`
- [ ] **Create async vLLM client wrapper** — `skill_agents/lora/vllm_client.py` with `generate_batch()` and `chat_with_tools()` methods
- [ ] **Replace `MultiLoraSkillBankLLM.generate()`** — route through vLLM API instead of HuggingFace `.generate()`. Keep the same public interface (`generate(function, prompt)`) but internally use HTTP calls to vLLM.
- [ ] **Remove `max_concurrent_llm_calls=1`** in `scripts/qwen3_skillbank_agent.py`

### P1 — Important (further 1.5-2x)

- [ ] **Cross-stage pipeline executor** — `trainer/skillbank/grpo/pipeline_executor.py` implementing `PipelinedEMExecutor` with micro-batch overlap across stages. Split trajectories into micro-batches, run each through S1→S2→S3 as async coroutines so requests from different stages/micro-batches hit vLLM concurrently.
- [ ] **Batch GRPO generation** — in each stage's GRPO loop, collect all prompts first, then call `generate_batch()` once instead of a for-loop
- [ ] **CPU multiprocessing for reward evaluation** — `ProcessPoolExecutor` for `decode_trajectory()` calls during Stage 1/2 reward computation
- [ ] **Multi-episode pipelined extraction** — `asyncio.gather` over episodes in `extract_skills_for_game()` so each episode's S1→S2→S3 requests interleave in vLLM

### P2 — Nice to have

- [ ] **Cross-game parallelism** — process multiple games concurrently in the extraction script (currently sequential `for game in games`)
- [ ] **Streaming tool execution** in Stage 4 — start computing next tool call's CPU work while vLLM generates the response
- [ ] **Adaptive batch sizing** — monitor vLLM queue depth, throttle if queue gets too deep (backpressure)

---

## GPU Memory Budget

All components share one Qwen3-14B on one A100-80GB:

| Component | Memory |
|-----------|--------|
| Qwen3-14B base weights (bf16) | ~28 GB |
| 4 LoRA adapters (rank 16 each) | ~0.8 GB total |
| vLLM KV cache (at 0.85 utilization) | ~39 GB |
| **Total** | ~68 GB / 80 GB |

The KV cache size determines how many requests can be batched simultaneously. With ~39 GB for KV cache and 8192 max sequence length, vLLM can hold ~60-80 concurrent requests in-flight. This is more than enough for our peak of 256 concurrent requests (Stage 2 GRPO) — vLLM will process them in 3-4 waves.

### During GRPO Training

GRPO LoRA updates need optimizer states + gradients for adapter params. With rank-16 LoRA on 14B, this is ~1-2 GB per adapter. Options:

**Option A (time-slice):** Pause vLLM during GRPO updates, run LoRA training on the same GPU, resume vLLM. Each GRPO update takes ~15s. Total pause per EM iteration: ~45s across 3 adapters. vLLM restart overhead: ~5s per pause. Total: ~60s added.

**Option B (separate GPU):** LoRA training on GPU 1, vLLM serving on GPU 0. No pauses, but requires 2 GPUs. After training, updated adapter files are hot-reloaded into vLLM.

**Recommendation:** Start with Option A (single GPU, time-slicing). Move to Option B if the 60s overhead matters.

---

## Verification Metrics

After implementing, measure:

| Metric | Target | How to measure |
|--------|--------|----------------|
| vLLM throughput | ≥ 800 tok/s | `vllm.metrics` endpoint |
| Stage 1 GRPO wall time | ≤ 30s | Timer in `stage1_grpo.py` |
| Stage 2 GRPO wall time | ≤ 80s | Timer in `stage2_grpo.py` |
| Stage 3 GRPO wall time | ≤ 35s | Timer in `stage3_grpo.py` |
| Stage 4 agent wall time | ≤ 60s | Timer in `stage4_agent.py` |
| Full EM iteration | ≤ 5 min | Timer in `em_trainer.py` |
| Extraction (8 games, 1 ep each) | ≤ 5 min | Timer in shell script |
| GPU utilization during GRPO stages | ≥ 80% | `nvidia-smi` |
