# Scripts

Entry points and **declared fatal hyperparameters** / **game envs** for [trainer](../trainer/) and [inference](../inference/). Use the **`.sh`** scripts; they declare all fatal vars and game envs at the top.

## Fatal hyperparameters and game envs

| Script | Hyperparameters | Game envs |
|--------|-----------------|-----------|
| **Trainer** | [run_trainer.sh](run_trainer.sh) — `TRAINER_*` env vars (model, GRPO, rollout, replay, costs, schedule) | Comment at top: overcooked, avalon, diplomacy, gamingagent, videogamebench, videogamebench_dos |
| **Inference** | [run_inference.sh](run_inference.sh) — `INFERENCE_*` env vars (game, task, max_steps, model, save_path, buffer sizes) | Comment at top: same list |

- **Trainer**: GRPO (group_size, lr, clip_ratio, …), rollout (max_steps, batch_size), replay (capacity, priority_*), reward costs, schedule. Config YAML paths: `TRAINER_DECISION_CONFIG`, `TRAINER_SKILLBANK_CONFIG`.
- **Inference**: max_steps, task, model, episode_buffer_size, experience_buffer_size, save_path. Override via env or CLI.

## Trainer (.sh)

Run from repo root:

```bash
chmod +x scripts/run_trainer.sh

# Decision-only GRPO
./scripts/run_trainer.sh

# Co-evolution (Decision + SkillBank)
./scripts/run_trainer.sh --coevolution

# Override config paths
./scripts/run_trainer.sh --config trainer/common/configs/decision_grpo.yaml
./scripts/run_trainer.sh --coevolution --decision-config trainer/common/configs/decision_grpo.yaml --skillbank-config trainer/common/configs/skillbank_em.yaml

# Override fatal hyperparameters via env
TRAINER_MAX_STEPS=300 TRAINER_BATCH_SIZE=16 ./scripts/run_trainer.sh

# Print declared game envs
./scripts/run_trainer.sh --print-envs
```

## Inference (.sh)

```bash
chmod +x scripts/run_inference.sh

# Default game from INFERENCE_GAME (default: gamingagent)
./scripts/run_inference.sh --game gamingagent

# With task and save path
./scripts/run_inference.sh --game gamingagent --task "Complete level" --max-steps 500 --save-path rollouts/episodes.jsonl --verbose

# Override via env
INFERENCE_GAME=videogamebench_dos INFERENCE_MAX_STEPS=500 ./scripts/run_inference.sh --game videogamebench_dos

# Print declared game envs and defaults
./scripts/run_inference.sh --print-envs
./scripts/run_inference.sh --print-defaults
```

If `--game` is not supported for auto env construction, use `inference.run_inference(env=...)` from your own code with an env from [env_wrappers](../env_wrappers/) or the `evaluate_*` folders.

Python defaults modules ([trainer_defaults.py](trainer_defaults.py), [inference_defaults.py](inference_defaults.py)) remain the source of values used by `scripts.run_inference` when invoked by the .sh scripts.
