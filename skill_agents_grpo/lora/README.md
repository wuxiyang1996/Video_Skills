# Multi-LoRA Skill Bank Agent

## Why 4 function-specific LoRAs?

The skill bank agent performs four distinct LLM-powered functions:

| Function     | What it does                                          | Adapter |
|-------------|-------------------------------------------------------|---------|
| **boundary** | Extract state predicates and detect skill boundaries  | `boundary` |
| **segment**  | Rank/label skill candidates for trajectory segments   | `segment` |
| **contract** | Summarize skill effects and contracts                 | `contract` |
| **retrieval**| Rewrite queries and rerank skill search results       | `retrieval` |

Each function has different input/output patterns, prompt styles, and desired behaviors. A single monolithic fine-tune would conflate these patterns. Instead, we keep one shared Qwen3-8B backbone and attach a small LoRA adapter per function. This gives:

- **Specialization** — each adapter learns function-specific patterns without interference.
- **Efficiency** — the base model is loaded once; switching adapters is a lightweight `set_adapter()` call.
- **Independent training** — each adapter can be updated on its own data/schedule.
- **Modularity** — adding a 5th function later only requires one more adapter.

## Architecture

```
┌─────────────────────────────────┐
│     Qwen3-8B  (shared base)     │
│                                 │
│  ┌──────────┐  ┌──────────┐    │
│  │ boundary  │  │ segment  │    │
│  │  LoRA     │  │  LoRA    │    │
│  └──────────┘  └──────────┘    │
│  ┌──────────┐  ┌──────────┐    │
│  │ contract  │  │ retrieval│    │
│  │  LoRA     │  │  LoRA    │    │
│  └──────────┘  └──────────┘    │
└─────────────────────────────────┘
          │
    SkillFunction enum selects
    which adapter is active
```

## How adapter routing works

1. `SkillFunction` enum defines the 4 functions (`BOUNDARY`, `SEGMENT`, `CONTRACT`, `RETRIEVAL`).
2. `MultiLoraSkillBankLLM` loads the base model + adapters.
3. Calling `llm.generate(SkillFunction.BOUNDARY, prompt)` does:
   - `model.set_adapter("boundary")` — activates the correct LoRA
   - Runs inference on the shared backbone + boundary LoRA
   - Returns the generated text
4. Existing call sites (`llm_extractor.py`, `llm_teacher.py`) auto-discover the shared instance via `get_shared_instance()` and route to the correct adapter.
5. If no LoRA is configured, they fall back to the API-based `ask_model`.

## Quick start

### Inference with all 4 adapters

```python
from skill_agents.lora import MultiLoraSkillBankLLM, MultiLoraConfig, SkillFunction

cfg = MultiLoraConfig(
    base_model_name_or_path="Qwen/Qwen3-8B",
    adapter_paths={
        "boundary":  "runs/lora_adapters/boundary",
        "segment":   "runs/lora_adapters/segment",
        "contract":  "runs/lora_adapters/contract",
        "retrieval": "runs/lora_adapters/retrieval",
    },
)

llm = MultiLoraSkillBankLLM(cfg)
llm.load()

# Register so existing code auto-discovers it
MultiLoraSkillBankLLM.set_shared_instance(llm)

# Generate with a specific adapter
out = llm.generate(SkillFunction.BOUNDARY, "Extract predicates from ...")
```

### From YAML config

```python
import yaml
from skill_agents.lora import MultiLoraSkillBankLLM, MultiLoraConfig

with open("configs/skillbank_lora.yaml") as f:
    raw = yaml.safe_load(f)

cfg = MultiLoraConfig.from_dict(raw["lora"])
llm = MultiLoraSkillBankLLM(cfg)
llm.load()
MultiLoraSkillBankLLM.set_shared_instance(llm)
```

### Backward compatibility (no LoRA)

If you don't set a shared instance, all LLM calls fall through to the existing `API_func.ask_model` with no changes needed.

## Training adapters

### Unified script

```bash
# Train boundary adapter
python -m trainer.skillbank.lora.train_lora \
    --skill_function boundary \
    --data_path runs/datasets/boundary_train.jsonl \
    --output_dir runs/lora_adapters/boundary \
    --base_model Qwen/Qwen3-8B \
    --lora_r 16 --lora_alpha 32 \
    --epochs 3 --lr 2e-4

# Train segment adapter
python -m trainer.skillbank.lora.train_lora \
    --skill_function segment \
    --data_path runs/datasets/segment_train.jsonl \
    --output_dir runs/lora_adapters/segment

# Train contract adapter
python -m trainer.skillbank.lora.train_lora \
    --skill_function contract \
    --data_path runs/datasets/contract_train.jsonl \
    --output_dir runs/lora_adapters/contract

# Train retrieval adapter
python -m trainer.skillbank.lora.train_lora \
    --skill_function retrieval \
    --data_path runs/datasets/retrieval_train.jsonl \
    --output_dir runs/lora_adapters/retrieval
```

### Convenience wrappers

```bash
python -m trainer.skillbank.lora.train_boundary_lora  --data_path runs/datasets/boundary_train.jsonl
python -m trainer.skillbank.lora.train_segment_lora   --data_path runs/datasets/segment_train.jsonl
python -m trainer.skillbank.lora.train_contract_lora  --data_path runs/datasets/contract_train.jsonl
python -m trainer.skillbank.lora.train_retrieval_lora --data_path runs/datasets/retrieval_train.jsonl
```

### Building training data

```python
from trainer.skillbank.lora.data_builder import (
    build_boundary_dataset,
    build_segment_dataset,
    build_contract_dataset,
    build_retrieval_dataset,
)

# Each builder produces [{"prompt": ..., "completion": ...}, ...]
boundary_data = build_boundary_dataset(trajectories, output_path="runs/datasets/boundary_train.jsonl")
segment_data  = build_segment_dataset(trajectories, decode_results, skill_names, output_path="runs/datasets/segment_train.jsonl")
```

## Adding a new function

1. Add a new value to `SkillFunction` in `skill_agents/lora/skill_function.py`.
2. Add its adapter path to `MultiLoraConfig.adapter_paths`.
3. Create a dataset builder in `trainer/skillbank/lora/data_builder.py`.
4. Create a thin training wrapper script.
5. In the code that makes LLM calls for this function, use `llm.as_ask_fn(SkillFunction.NEW_FUNCTION)`.

## File layout

```
skill_agents/lora/
├── __init__.py          # Exports: SkillFunction, MultiLoraConfig, MultiLoraSkillBankLLM
├── skill_function.py    # SkillFunction enum (routing key)
├── config.py            # MultiLoraConfig, LoraTrainingConfig
├── model.py             # MultiLoraSkillBankLLM (shared model + adapter switching)
└── README.md            # This file

trainer/skillbank/lora/
├── __init__.py
├── train_lora.py            # Unified training (--skill_function flag)
├── train_boundary_lora.py   # Wrapper for boundary
├── train_segment_lora.py    # Wrapper for segment
├── train_contract_lora.py   # Wrapper for contract
├── train_retrieval_lora.py  # Wrapper for retrieval
└── data_builder.py          # Dataset builders per function

configs/
└── skillbank_lora.yaml      # Example inference + training config

tests/
└── test_lora_dispatch.py    # Adapter dispatch + config + fallback tests
```

## Tests

```bash
cd Game-AI-Agent
python -m pytest tests/test_lora_dispatch.py -v
```
