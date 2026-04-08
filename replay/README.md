# Replay Animation Generator

Generate animated GIF replays from best-performing episodes, using official game environment renderers for high-fidelity visualization.

## Supported Games & Renderers

| Game | Renderer | Source |
|------|----------|--------|
| Tetris | `TetrisEnv` (cv2 rgb_array) | GamingAgent |
| 2048 | `TwentyFortyEightEnv` (pygame rgb_array) | GamingAgent |
| Sokoban | `SokobanEnv` (rgb_array + PIL fallback) | GamingAgent |
| Candy Crush | `tile_match_gym.Renderer` | GamingAgent |
| Super Mario | NES emulator (`gym_super_mario_bros`) | Orak |
| Avalon | Playwright + HTML template (AgentEvolver pixel-art style) | AgentEvolver |
| Diplomacy | `diplomacy.engine.renderer.Renderer` (SVG map) | diplomacy package |

## Quick Start

```bash
cd /workspace/COS-PLAY

export PYTHONPATH=/workspace/COS-PLAY:/workspace/AgentEvolver:/workspace/GamingAgent:$PYTHONPATH

# Use default output/ directory
python replay/generate_replay_gifs.py

# Or point at a specific episode data directory
python replay/generate_replay_gifs.py --output-dir /path/to/output
```

GIFs are saved to `replay/replays/`.

## Prerequisites

**Python packages** (install into the active conda env):

```bash
pip install pillow numpy imageio playwright
playwright install chromium
playwright install-deps chromium
```

**Environment renderers** (for full game coverage):

- **Tetris / 2048 / Sokoban / Candy Crush**: Requires [GamingAgent](https://github.com/) on `PYTHONPATH` with `opencv-python-headless`, `pygame`, `gymnasium`, `tile_match_gym`.
- **Avalon**: Uses Playwright (headless Chromium) to render an HTML template matching the [AgentEvolver](https://github.com/modelscope/AgentEvolver) web UI pixel-art style. No game server needed.
- **Diplomacy**: Requires the `diplomacy` Python package and `librsvg2-bin` (`apt install librsvg2-bin`) for SVG-to-PNG conversion.
- **Super Mario**: Requires a separate conda env (`orak-mario`) with `gym_super_mario_bros` and `nes_py`. The main script automatically calls `generate_mario_replay.py` in that env.

## How It Works

1. **Episode discovery**: Scans `output/**/rollout_summary.json` for the best episode per game (highest `total_reward`).
2. **State parsing**: Each episode step has a text-based state observation. Game-specific parsers extract board grids, phase info, player data, etc.
3. **Frame rendering**: The parsed state is injected into the official game environment (or HTML template for Avalon), and `render()` / screenshot captures an RGB frame.
4. **GIF assembly**: Frames are uniformly sized and saved as animated GIFs with game-appropriate frame rates (1.5-3.0 fps).

## Files

| File | Description |
|------|-------------|
| `generate_replay_gifs.py` | Main script: episode discovery, all renderers, GIF assembly |
| `generate_mario_replay.py` | Standalone Super Mario replay via NES emulator (runs in `orak-mario` env) |
| `avalon_template.html` | Self-contained HTML template replicating AgentEvolver's Avalon pixel-art UI |
| `replays/` | Output directory for generated GIFs |

## Episode Data Format

The script expects episode JSON files with this structure:

```json
{
  "experiences": [
    {
      "state": "<game-specific text observation>",
      "action": "<action taken>",
      "reward": 0.0,
      "done": false
    }
  ]
}
```

Each game's state format differs (board grids for puzzle games, JSON dicts of player observations for Avalon/Diplomacy, NL descriptions for Mario).
