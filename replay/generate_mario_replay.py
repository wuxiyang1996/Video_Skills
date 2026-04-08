#!/usr/bin/env python3
"""
Generate Super Mario replay GIF by replaying recorded actions through
the actual NES emulator (gym_super_mario_bros).

Requires the orak-mario conda env with gym_super_mario_bros + nes_py.

Usage:
    /workspace/anaconda3/envs/orak-mario/bin/python generate_mario_replay.py
"""
import json
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

OUTPUT_DIR = Path(__file__).parent / "output"
REPLAY_DIR = Path(__file__).parent / "replays"


def _pil_font(size: int):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def find_best_mario_episode():
    best = None
    for sp in glob.glob(str(OUTPUT_DIR / "**" / "rollout_summary.json"), recursive=True):
        with open(sp) as f:
            data = json.load(f)
        if data.get("game") != "super_mario":
            continue
        stats = data.get("episode_stats", [])
        if not stats:
            continue
        top = max(stats, key=lambda s: s["total_reward"])
        if best is None or top["total_reward"] > best["reward"]:
            ep_dir = Path(sp).parent
            ep_file = ep_dir / f"episode_{top['episode_index']:03d}.json"
            if ep_file.exists():
                best = {
                    "episode_index": top["episode_index"],
                    "reward": top["total_reward"],
                    "steps": top["steps"],
                    "episode_path": str(ep_file),
                }
    return best


def parse_jump_level(action_text: str) -> int:
    m = re.search(r"Jump Level:\s*(\d+)", action_text)
    if m:
        return int(m.group(1))
    return 0


def replay_mario(experiences: list, fps: float = 4.0) -> List[Image.Image]:
    """
    Replay actions through the NES emulator capturing rgb_array frames.
    The Orak env uses:
      - JoypadSpace with [['right'], ['right', 'A']]
      - SkipFrame(skip=4)
      - Jump Level N → N frames of action=1 (right+jump), then wait until landed
    """
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v1",
        render_mode="rgb_array",
        apply_api_compatibility=True,
    )
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    state, info = env.reset()
    for _ in range(25):
        state, reward, done, trunc, info = env.step(0)
        if done:
            state, info = env.reset()

    frames: List[Image.Image] = []
    cumulative_reward = 0.0
    fnt = _pil_font(12)
    fnt_title = _pil_font(14)

    for step_i, exp in enumerate(experiences):
        action_text = exp.get("action", "Jump Level: 0")
        step_reward = exp.get("reward", 0.0)
        cumulative_reward += step_reward
        jump_level = parse_jump_level(action_text)

        if jump_level == 0:
            for _ in range(4):
                state, reward, done, trunc, info = env.step(0)
                if done:
                    break
        else:
            for _ in range(jump_level):
                for _ in range(4):
                    state, reward, done, trunc, info = env.step(1)
                    if done:
                        break
                if done:
                    break
            if not done:
                y_hist = []
                for _ in range(120):
                    for _ in range(4):
                        state, reward, done, trunc, info = env.step(0)
                        if done:
                            break
                    if done:
                        break
                    y_hist.append(info.get("y_pos", 0))
                    if len(y_hist) >= 3 and y_hist[-3] == y_hist[-2] == y_hist[-1]:
                        break

        raw_frame = env.render()
        if raw_frame is None:
            raw_frame = state
        if isinstance(raw_frame, np.ndarray):
            if raw_frame.dtype != np.uint8:
                raw_frame = (raw_frame * 255).clip(0, 255).astype(np.uint8)
            if raw_frame.ndim == 4:
                raw_frame = raw_frame[0]

        H, W = raw_frame.shape[:2]
        info_h = 36
        img = Image.new("RGB", (W, H + info_h), (0, 0, 0))
        img.paste(Image.fromarray(raw_frame), (0, info_h))
        draw = ImageDraw.Draw(img)

        draw.text((6, 2), f"SUPER MARIO BROS  Step {step_i}", fill=(255, 255, 255), font=fnt_title)
        draw.text((6, 19),
                  f"Action: {action_text[:30]}  Score: {step_reward:.0f}  "
                  f"x={info.get('x_pos', '?')}",
                  fill=(255, 220, 100), font=fnt)
        frames.append(img)

        if done:
            break

    env.close()
    return frames


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    args = parser.parse_args()

    REPLAY_DIR.mkdir(exist_ok=True)
    best = find_best_mario_episode()
    if best is None:
        print("No Super Mario episodes found.")
        return

    print(f"Best Super Mario episode: ep {best['episode_index']}, "
          f"reward={best['reward']}, steps={best['steps']}")

    with open(best["episode_path"]) as f:
        experiences = json.load(f)["experiences"]

    print(f"Replaying {len(experiences)} actions through NES emulator ...")
    frames = replay_mario(experiences)

    if not frames:
        print("No frames captured!")
        return

    ext = args.format
    out_path = REPLAY_DIR / f"best_super_mario.{ext}"
    fps = 4.0

    if ext == "gif":
        duration_ms = int(1000 / fps)
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=duration_ms, loop=0)
    else:
        import imageio
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                    quality=8, pixelformat="yuv420p",
                                    macro_block_size=2)
        for f in frames:
            writer.append_data(np.array(f))
        writer.close()

    print(f"  -> {out_path.name}  ({len(frames)} frames, {fps} fps, "
          f"{out_path.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
