# Self-Reflection & Repair for Small (7-9B) Models

## Summary

9B models **can** perform failure reflection and self-repair, but **not via naive prompting**. They require structured training (SFT on repair trajectories + RL). Multiple 2025-2026 works demonstrate 7-9B models matching or exceeding 72B models on multi-hop reasoning with self-correction.

---

## Key Prior Works

### 1. Agent0-VL (2025, AIMING Lab)

**Paper**: "Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning"
**Code**: https://github.com/aiming-lab/Agent0

| Model | Mechanism | Key Result |
|-------|-----------|------------|
| Agent0-VL-7B | Solver + Verifier dual-role, single model | +12.5% over base |
| Agent0-VL-8B | Self-Evolving Reasoning Cycle (SERC) | **Surpasses 72B open-source models** |

Core design:
- **Solver**: multi-turn tool-integrated reasoning
- **Verifier**: tool-grounded critique → structured feedback → self-repair
- No external reward model — self-generated rewards
- Training: RL (not pure SFT)
- Tool usage in reasoning + evaluation + repair

Key insight: **Tool-grounded verification enables reliable self-repair** — the model uses tools to check its own reasoning instead of relying on internal self-judgment (which fails for small models).

---

### 2. OmniAgent (ICML 2026)

**Paper**: "OmniAgent: the first native omni-modal agent for active video perception"
**Code**: https://github.com/HarryHsing/OmniAgent

| Model | Task | Result |
|-------|------|--------|
| Qwen2.5-Omni-7B (trained) | Video QA (POMDP) | **Beats Qwen2.5-VL-72B, 73% fewer frames** |

Core design:
- **Observation-Thought-Action (OTA) loop** — iterative perception
- Cold-start SFT (58K trajectories) with **self-correction** built into trajectory synthesis
- RL with TAURA (turn-level credit assignment)
- Repair via **best-of-N exploration + self-correction** during training data synthesis
- Memory consolidation: raw media → text summary → reasoning

Critical for us: **7B single model doing active video reasoning with repair — exactly our target architecture.**

---

### 3. SenseNova-MARS (2025-2026)

**Paper**: "Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning"

Models: SenseNova-MARS-7B, 8B, 32B (based on Qwen2.5-VL / Qwen3-VL)

Core design:
- **3,000 sample cold-start SFT** → learns basic tool-use patterns
- RL stage (BN-GSPO) → learns multi-tool collaboration + reasoning repair
- Key finding: cold-start needs **very few samples** (~3K), but RL stage is critical

Relevance: Shows the minimal data requirement for cold-start (our expert demo generation with gpt-oss provides this).

---

### 4. ReflectEvo (ACL 2025 Findings)

**Paper**: "Improving Meta Introspection of Small LLMs by Learning Self-Reflection"

| Model | Before | After ReflectEvo | Method |
|-------|--------|------------------|--------|
| Llama-3-8B | 52.4% | **71.2%** | SFT + DPO on self-generated reflections |
| Mistral-7B | 44.4% | **71.1%** | Same |
| Gemma-2-9B | Similar gains | | |

Key findings:
- 9B models **can learn error localization + correction**
- No distillation from larger models needed (self-generated reflection data suffices)
- **First 2 repair rounds capture 76-95% of fixable errors**
- Quality of self-generated reflections matters more than quantity

---

### 5. Entrospect (ACL 2025 Findings)

**Paper**: "Information-Theoretic Self-Reflection Elicits Better Response Refinement of Small Language Models"

**Critical negative result**: Standard Self-Refine (prompt-based reflection) **fails** for models ≤10B. Even worse than Chain-of-Thought in many cases.

**Solution**: Entropy-aware introspection — redesigned prompting that is information-theoretically grounded. Achieves +36.2 reasoning accuracy improvement with 10x efficiency gain.

**Lesson for us**: Do NOT rely on naive "reflect on your error" prompts for 9B. Must use either:
1. Trained reflection (SFT/DPO on repair trajectories), or
2. Tool-grounded verification (external signals, not self-judgment)

---

### 6. POLARIS (ICLR 2026 Workshop RSI)

**Paper**: "A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair"

- 7B model inspects its own policy → finds bugs → generates minimal code patches
- No full retraining needed
- Works for reasoning + problem-solving tasks

---

## Synthesis: When Does 9B Self-Repair Work?

| Condition | Works? | Evidence |
|-----------|--------|----------|
| Prompt-based "reflect on error" | ❌ **No** | Entrospect shows failure ≤10B |
| SFT on repair trajectory data | ✅ **Yes** | ReflectEvo: 52% → 71% |
| RL (GRPO/TAURA) after cold-start | ✅ **Yes** | Agent0-VL, OmniAgent, MARS |
| Tool-grounded verification as reward | ✅ **Yes** | Agent0-VL (tool-integrated critique) |
| Single model Solver + Verifier | ✅ **Yes** | Agent0-VL 8B surpasses 72B |
| Active video perception + repair | ✅ **Yes** | OmniAgent 7B beats 72B |

### Critical Success Factors

1. **Do NOT rely on self-judgment** — use external verification (tools, runtime verifier, ground truth)
2. **Train on repair trajectories** — SFT first, then RL
3. **First 1-2 repair rounds** capture most of the gain (diminishing returns after)
4. **Cold-start data can be small** (~3K-58K trajectories)
5. **RL is essential** for generalization (SFT alone overfits to seen failure patterns)

---

## Implications for Our System

### Current Architecture (Stage 0.5)

```
gpt-oss-120b (teacher) → expert plan → 9B executes → failures recorded
                                                            ↓
                                              rule-based fault_repair.py
                                                            ↓
                                              repair trace (training data)
```

### Target Architecture (informed by prior work)

```
Stage 0.5 (NOW):
  rule-based repair → generates repair trajectory data
  runtime_verifier provides tool-grounded verification signals

Stage 1 (NEXT):
  SFT Qwen3.5-9B on collected repair trajectories (~3K-58K)
  9B learns: given failure trace → localize fault → select repair strategy

Stage 2 (FINAL):
  RL (GRPO/TAURA) with:
    - runtime verifier pass/fail as reward
    - answer correctness as reward
    - per-step protocol compliance as intrinsic reward
  9B autonomously: perceive → reason → verify → repair → commit

  Architecture (from Agent0-VL):
    Same 9B model plays both Solver (execute skills) and Verifier (critique + repair)
    Tool-grounded verification via our runtime_verifier.py
```

### Training Data Strategy

| Source | Samples | Purpose |
|--------|---------|---------|
| Expert trajectories (gpt-oss) | ~200-1000 per dataset | Cold-start, correct execution patterns |
| Repair trajectories (fault_repair.py) | ~3K+ | Fault localize + repair patterns |
| Self-rollout (9B after SFT) | ~10K+ | Self-generated for RL |
| Runtime verifier signals | Per-step | Dense reward for RL |

### Key Design Decisions

1. **Solver-Verifier in same model** (Agent0-VL approach)
   - Our 9B plays both roles
   - Verification uses runtime_verifier (tool-grounded, not self-judgment)

2. **Repair as part of skill loop** (not offline post-processing)
   - After each failed skill step, attempt 1 repair round
   - First round captures 76-95% of fixable errors (ReflectEvo)
   - Budget: max 1-2 repair attempts per failed step

3. **Training curriculum**
   - Phase 1: SFT on teacher trajectories (correct execution)
   - Phase 2: SFT on repair trajectories (error → diagnosis → fix)
   - Phase 3: RL with verifier rewards (autonomous improvement)

---

## References

1. Agent0-VL. "Exploring Self-Evolving Agent for Tool-Integrated Vision-Language Reasoning." arXiv:2511.19900, Nov 2025.
2. OmniAgent. "The first native omni-modal agent for active video perception." ICML 2026.
3. SenseNova-MARS. "Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning." arXiv:2512.24330, Dec 2025.
4. ReflectEvo. "Improving Meta Introspection of Small LLMs by Learning Self-Reflection." ACL 2025 Findings.
5. Entrospect. "Information-Theoretic Self-Reflection Elicits Better Response Refinement of Small Language Models." ACL 2025 Findings.
6. POLARIS. "A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair." ICLR 2026 Workshop RSI.
