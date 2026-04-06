# Baselines

LLM API baselines for evaluating frontier models on the 6 game environments (Table 1 in the paper). Each game has a single script that accepts a `--model` flag.

## Supported Models

| Model | Flag |
|-------|------|
| GPT-5.4 | `--model gpt-5.4` |
| GPT-OSS 120B | `--model openai/gpt-oss-120b` |
| Gemini 3.1 Pro | `--model google/gemini-3.1-pro-preview` |
| Claude 4.6 Sonnet | `--model anthropic/claude-4.6-sonnet-20260217` |

## Usage

```bash
# Single-player games (no vLLM server needed, uses OpenRouter API)
bash baselines/run_tetris_baseline.sh                                      # GPT-5.4 default
bash baselines/run_tetris_baseline.sh --model openai/gpt-oss-120b
bash baselines/run_2048_baseline.sh --model google/gemini-3.1-pro-preview
bash baselines/run_candy_crush_baseline.sh --model anthropic/claude-4.6-sonnet-20260217

# Multi-agent games (controlled model vs GPT-5.4 opponents)
bash baselines/run_avalon_baseline.sh --model openai/gpt-oss-120b          # 40 eps (8/player × 5)
bash baselines/run_diplomacy_baseline.sh --model google/gemini-3.1-pro-preview  # 56 eps (8/power × 7)

# Super Mario (requires orak-mario conda env for NES emulator)
bash baselines/run_super_mario_baseline.sh --model openai/gpt-oss-120b

# Override episodes / temperature
EPISODES=16 bash baselines/run_tetris_baseline.sh --model gpt-5.4
EPISODES_PER_POWER=4 bash baselines/run_diplomacy_baseline.sh --model gpt-5.4
```

## Analysis

After collecting results, compute win rates and confidence intervals:

```bash
python baselines/analyze_baselines.py
```

## Files

| File | Purpose |
|------|---------|
| `run_tetris_baseline.sh` | Tetris baseline via macro-action wrapper |
| `run_2048_baseline.sh` | 2048 baseline |
| `run_candy_crush_baseline.sh` | Candy Crush baseline |
| `run_avalon_baseline.sh` | Avalon baseline (per-role cycling vs GPT-5.4) |
| `run_diplomacy_baseline.sh` | Diplomacy baseline (per-power cycling vs GPT-5.4) |
| `run_super_mario_baseline.sh` | Super Mario baseline (Xvfb + orak-mario env) |
| `run_gpt54_tetris_macro.py` | Python backend for Tetris (shared by all models) |
| `analyze_baselines.py` | Post-hoc analysis: win rates, CIs |
