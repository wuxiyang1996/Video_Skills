#!/usr/bin/env python3
"""
Generate replay animation GIFs from best-performing episodes using the
official game environment renderers (cv2 for Tetris, pygame for 2048,
native for Sokoban, Playwright for Avalon, diplomacy package for Diplomacy).

Usage:
    cd /workspace/COS-PLAY
    export PYTHONPATH=/workspace/COS-PLAY:/workspace/AgentEvolver:/workspace/GamingAgent:$PYTHONPATH
    python replay/generate_replay_gifs.py [--output-dir /path/to/output]

If --output-dir is not given, defaults to <project_root>/output/.
GIFs are saved to replay/replays/.
"""

import json
import glob
import io
import os
import re
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGLET_HEADLESS", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
REPLAY_DIR = Path(__file__).resolve().parent / "replays"

# ═══════════════════════════════════════════════════════════════════════════
# Board state parsers  (text observation → numpy array the env expects)
# ═══════════════════════════════════════════════════════════════════════════

TETRIS_SYM_TO_ID = {".": 0, "#": 1, "I": 2, "O": 3, "T": 4, "S": 5, "Z": 6, "J": 7, "L": 8}
TETRIS_W, TETRIS_H = 10, 20


def parse_tetris_board(state_text: str) -> Optional[np.ndarray]:
    rows = []
    for line in state_text.split("\n"):
        s = line.strip()
        if s and len(s) == TETRIS_W and all(c in ".IOTSZJL" for c in s):
            rows.append([TETRIS_SYM_TO_ID[c] for c in s])
    if len(rows) < TETRIS_H:
        return None
    return np.array(rows[:TETRIS_H], dtype=np.uint8)


def parse_tetris_meta(state_text: str) -> dict:
    meta = {}
    m = re.search(r"Next Pieces:\s*([A-Z,]+)", state_text)
    if m:
        meta["next_symbols"] = m.group(1).split(",")
    for key, pat in [("perf", r"PerfS:(\S+)"), ("score", r" S:(\S+)"),
                     ("lines", r" L:(\d+)"), ("level", r"Lv:(\d+)")]:
        m = re.search(pat, state_text)
        if m:
            meta[key] = m.group(1)
    return meta


def parse_2048_board(state_text: str) -> Optional[np.ndarray]:
    m = re.search(r"'board':\s*\[(\[.*?\])\]", state_text, re.DOTALL)
    if not m:
        return None
    raw = re.sub(r"np\.\w+\((\d+)\)", r"\1", m.group(1))
    rows_raw = re.findall(r"\[([^\]]+)\]", raw)
    board = []
    for row in rows_raw:
        board.append([int(x.strip()) for x in row.split(",")])
    if not board:
        return None
    vals = np.array(board, dtype=np.int64)
    powers = np.zeros_like(vals, dtype=np.uint8)
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            v = vals[r, c]
            if v > 0:
                powers[r, c] = int(np.log2(v))
    return powers


SOKOBAN_ITEM_TO_STATE = {
    "Wall": 0, "Empty": 1, "Dock": 2, "Box on Dock": 3,
    "Box": 4, "Worker": 5, "Worker on Dock": 6,
}
SOKOBAN_STATE_COLORS = {
    0: (80, 80, 100),    # Wall   – dark blue-grey
    1: (220, 210, 190),  # Floor  – warm beige
    2: (200, 80, 80),    # Dock   – red target
    3: (180, 140, 50),   # Box on target – gold
    4: (160, 120, 60),   # Box    – brown
    5: (60, 120, 220),   # Worker – blue
    6: (100, 60, 200),   # Worker on target – purple
}


def parse_sokoban_table(state_text: str) -> Optional[np.ndarray]:
    items = []
    max_col, max_row = 0, 0
    for line in state_text.split("\n"):
        m = re.match(r"\s*\d+\s*\|\s*(\S[\w ]*\S|\S)\s*\|\s*\((\d+),\s*(\d+)\)", line)
        if not m:
            continue
        item_type = m.group(1).strip()
        col, row = int(m.group(2)), int(m.group(3))
        state_val = SOKOBAN_ITEM_TO_STATE.get(item_type)
        if state_val is None:
            continue
        items.append((row, col, state_val))
        max_row = max(max_row, row)
        max_col = max(max_col, col)
    if not items:
        return None
    grid = np.ones((max_row + 1, max_col + 1), dtype=np.uint8)  # default floor
    for r, c, v in items:
        grid[r, c] = v
    return grid


CANDY_CHAR_TO_IDX = {"R": 4, "C": 2, "G": 1, "P": 3, "Y": 5, "B": 6, "O": 7, " ": 0}


def parse_candy_board(state_text: str) -> Optional[np.ndarray]:
    rows = []
    for line in state_text.split("\n"):
        m = re.match(r"\d+\|\s*(.*)", line.strip())
        if m:
            chars = m.group(1).strip().split()
            rows.append([CANDY_CHAR_TO_IDX.get(ch, 0) for ch in chars])
    if not rows:
        return None
    return np.array(rows, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# Environment-based renderers
# ═══════════════════════════════════════════════════════════════════════════

_tetris_env = None
_2048_env = None
_sokoban_env = None
_candy_env = None
_candy_renderer = None


def _get_tetris_env():
    global _tetris_env
    if _tetris_env is None:
        from gamingagent.envs.custom_04_tetris.tetrisEnv import TetrisEnv
        _tetris_env = TetrisEnv(
            render_mode="rgb_array",
            observation_mode_for_adapter="text",
            agent_cache_dir_for_adapter=tempfile.mkdtemp(prefix="replay_tetris_"),
        )
        _tetris_env.reset(seed=0)
    return _tetris_env


def _get_2048_env():
    global _2048_env
    if _2048_env is None:
        from gamingagent.envs.custom_01_2048.twentyFortyEightEnv import TwentyFortyEightEnv
        _2048_env = TwentyFortyEightEnv(
            render_mode="rgb_array",
            observation_mode_for_adapter="text",
            agent_cache_dir_for_adapter=tempfile.mkdtemp(prefix="replay_2048_"),
        )
        _2048_env.reset(seed=0)
    return _2048_env


def _get_sokoban_env():
    global _sokoban_env
    if _sokoban_env is None:
        from gamingagent.envs.custom_02_sokoban.sokobanEnv import SokobanEnv
        _sokoban_env = SokobanEnv(
            render_mode="rgb_array",
            observation_mode_for_adapter="text",
            agent_cache_dir_for_adapter=tempfile.mkdtemp(prefix="replay_sokoban_"),
        )
        _sokoban_env.reset(seed=0)
    return _sokoban_env


def _get_candy_env():
    global _candy_env, _candy_renderer
    if _candy_env is None:
        from gamingagent.envs.custom_03_candy_crush.candyCrushEnv import CandyCrushEnv
        from tile_match_gym.renderer import Renderer
        _candy_env = CandyCrushEnv(
            observation_mode_for_adapter="text",
            agent_cache_dir_for_adapter=tempfile.mkdtemp(prefix="replay_candy_"),
        )
        _candy_renderer = Renderer(
            _candy_env.num_rows, _candy_env.num_cols,
            _candy_env.num_colours, _candy_env.num_moves,
            render_fps=2, render_mode="rgb_array",
        )
        _candy_env.renderer = _candy_renderer
        _candy_env.internal_render_mode = "rgb_array"
        _candy_env.reset(seed=0)
    return _candy_env


def render_tetris_env(state_text: str, step: int, reward: float,
                      total_reward: float, action: str) -> Optional[Image.Image]:
    board_ids = parse_tetris_board(state_text)
    if board_ids is None:
        return None
    env = _get_tetris_env()
    pad = env.padding
    env.board[0:TETRIS_H, pad:pad + TETRIS_W] = board_ids
    env.active_tetromino = None

    meta = parse_tetris_meta(state_text)
    env.current_score = float(meta.get("score", 0))
    env.lines_cleared_total = int(meta.get("lines", 0))
    env.level = int(meta.get("level", 1))
    env.total_perf_score_episode = float(meta.get("perf", 0))

    sym_to_idx = {"I": 0, "O": 1, "T": 2, "S": 3, "Z": 4, "J": 5, "L": 6}
    next_syms = meta.get("next_symbols", [])
    next_ids = [env.tetrominoes[sym_to_idx[s]].id for s in next_syms if s in sym_to_idx]
    while len(next_ids) < env.queue_size:
        next_ids.append(0)

    env.current_raw_obs_dict = {"board": env._get_raw_board_obs_for_render()}
    env.current_info_dict = {
        "score": env.current_score,
        "total_perf_score_episode": env.total_perf_score_episode,
        "lines": env.lines_cleared_total,
        "level": env.level,
        "next_piece_ids": next_ids,
    }

    frame = env.render()
    if frame is None:
        return None

    info_w = 200
    H, W = frame.shape[:2]
    canvas = np.zeros((H, W + info_w, 3), dtype=np.uint8)
    canvas[:, :W] = frame

    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    ix, iy = W + 8, 28
    cv2.putText(canvas, "TETRIS REPLAY", (ix, iy), font, 0.55, (255, 255, 255), 1)
    iy += 26
    cv2.putText(canvas, f"Step: {step}", (ix, iy), font, 0.45, (200, 200, 200), 1)
    iy += 22
    act_disp = action[:18] if len(action) > 18 else action
    cv2.putText(canvas, f"Act: {act_disp}", (ix, iy), font, 0.4, (180, 220, 255), 1)
    iy += 22
    cv2.putText(canvas, f"Reward: {reward:+.0f}", (ix, iy), font, 0.45, (255, 220, 100), 1)
    iy += 22
    cv2.putText(canvas, f"Total: {total_reward:.0f}", (ix, iy), font, 0.45, (100, 255, 100), 1)

    return Image.fromarray(canvas)


def render_2048_env(state_text: str, step: int, reward: float,
                    total_reward: float, action: str) -> Optional[Image.Image]:
    board_powers = parse_2048_board(state_text)
    if board_powers is None:
        return None
    env = _get_2048_env()
    env.board = board_powers
    env.total_score = int(total_reward)

    frame = env.render()
    if frame is None:
        return None

    img = Image.fromarray(frame.astype(np.uint8))

    fnt = _pil_font(14)
    info_h = 30
    new_img = Image.new("RGB", (img.width, img.height + info_h), (250, 248, 238))
    new_img.paste(img, (0, info_h))
    draw = ImageDraw.Draw(new_img)
    draw.text((10, 6), f"Step {step}  Action: {action}  Reward: {reward:+.0f}  Total: {total_reward:.0f}",
              fill=(119, 110, 101), font=fnt)
    return new_img


def render_sokoban_env(state_text: str, step: int, reward: float,
                       total_reward: float, action: str) -> Optional[Image.Image]:
    grid = parse_sokoban_table(state_text)
    if grid is None:
        return None

    env = _get_sokoban_env()
    env.room_state = grid
    env.dim_room = grid.shape
    player_pos = np.argwhere(np.isin(grid, [5, 6]))
    if player_pos.size:
        env.player_position = player_pos[0]

    frame = env.render()
    if frame is not None and hasattr(frame, "shape"):
        img = Image.fromarray(frame.astype(np.uint8))
    else:
        tile = 36
        rows, cols = grid.shape
        img = Image.new("RGB", (cols * tile, rows * tile), (40, 40, 50))
        draw = ImageDraw.Draw(img)
        fnt_t = _pil_font(14)
        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * tile, r * tile
                color = SOKOBAN_STATE_COLORS.get(int(grid[r, c]), (100, 100, 100))
                draw.rounded_rectangle([x0 + 1, y0 + 1, x0 + tile - 2, y0 + tile - 2],
                                       radius=4, fill=color)
                char = {0: "#", 1: "", 2: ".", 3: "*", 4: "$", 5: "@", 6: "+"}.get(int(grid[r, c]), "?")
                if char:
                    tw, th = _text_size(draw, char, fnt_t)
                    draw.text((x0 + (tile - tw) // 2, y0 + (tile - th) // 2),
                              char, fill=(255, 255, 255), font=fnt_t)

    fnt = _pil_font(13)
    info_h = 40
    new_img = Image.new("RGB", (img.width, img.height + info_h), (30, 30, 40))
    new_img.paste(img, (0, info_h))
    draw = ImageDraw.Draw(new_img)
    draw.text((6, 4), f"SOKOBAN  Step {step}  Action: {action}", fill=(255, 255, 255), font=fnt)
    draw.text((6, 22), f"Reward: {reward:+.1f}  Total: {total_reward:.1f}", fill=(255, 220, 100), font=fnt)
    return new_img


# ═══════════════════════════════════════════════════════════════════════════
# PIL fallback renderers  (Candy Crush, Super Mario, Avalon, Diplomacy)
# ═══════════════════════════════════════════════════════════════════════════

def render_candy_env(state_text: str, step: int, reward: float,
                     total_reward: float, action: str) -> Optional[Image.Image]:
    board_colors = parse_candy_board(state_text)
    if board_colors is None:
        return None
    env = _get_candy_env()
    rows, cols = board_colors.shape
    env.board.board[0, :rows, :cols] = board_colors
    moves_m = re.search(r"Moves Left:\s*(\d+)", state_text)
    moves_left = int(moves_m.group(1)) if moves_m else max(0, env.num_moves - step)
    env.timer = env.num_moves - moves_left

    frame = env.renderer.render(env.board.board, moves_left)
    if frame is None or not isinstance(frame, np.ndarray):
        return None
    img = Image.fromarray(frame.astype(np.uint8))

    fnt = _pil_font(13)
    info_h = 36
    new_img = Image.new("RGB", (img.width, img.height + info_h), (40, 30, 50))
    new_img.paste(img, (0, info_h))
    draw = ImageDraw.Draw(new_img)
    draw.text((8, 2), f"CANDY CRUSH   Step {step}", fill=(255, 255, 255), font=_pil_font(14))
    score_m = re.search(r"Score:\s*(\d+)", state_text)
    score_str = score_m.group(1) if score_m else f"{int(total_reward)}"
    draw.text((8, 19),
              f"Action: {action[:30]}  Score: {score_str}  Reward: {reward:+.0f}",
              fill=(255, 220, 100), font=fnt)
    return new_img


CANDY_COLORS = {
    "R": (220, 40, 40), "C": (60, 160, 240), "G": (40, 200, 60),
    "P": (180, 60, 200), "Y": (240, 200, 40), "O": (240, 140, 40),
}

AVALON_GOOD_ROLES = {"Merlin", "Percival", "Servant"}
AVALON_EVIL_ROLES = {"Assassin", "Morgana", "Mordred", "Oberon", "Minion"}

_avalon_pw = None
_avalon_browser = None
_avalon_page = None

DIPLO_POWER_COLORS = {
    "AUSTRIA": (200, 50, 50),
    "ENGLAND": (50, 80, 180),
    "FRANCE":  (80, 160, 220),
    "GERMANY": (100, 100, 100),
    "ITALY":   (60, 180, 80),
    "RUSSIA":  (180, 180, 180),
    "TURKEY":  (220, 180, 50),
}


def _pil_font(size: int):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _pil_font_regular(size: int):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ── Avalon renderer (AgentEvolver pixel-art style via Playwright) ─────────

def _get_avalon_page():
    global _avalon_pw, _avalon_browser, _avalon_page
    if _avalon_page is None:
        from playwright.sync_api import sync_playwright
        _avalon_pw = sync_playwright().start()
        _avalon_browser = _avalon_pw.chromium.launch(headless=True)
        _avalon_page = _avalon_browser.new_page(viewport={"width": 960, "height": 540})
        template_path = Path(__file__).parent / "avalon_template.html"
        _avalon_page.goto(f"file://{template_path.resolve()}")
        _avalon_page.wait_for_load_state("networkidle")
    return _avalon_page


def _parse_avalon_state(state_text: str):
    try:
        state_dict = json.loads(state_text)
    except (json.JSONDecodeError, TypeError):
        state_dict = {"0": state_text}

    ref = state_dict.get("0", list(state_dict.values())[0] if state_dict else "")

    phase_m = re.search(r"Avalon Game\s*[—–-]\s*([\w ]+)\s*===", ref)
    phase = phase_m.group(1).strip() if phase_m else "Unknown"

    quest_m = re.search(r"Current quest:\s*(\d+)\s*of\s*(\d+)", ref)
    cur_quest = int(quest_m.group(1)) if quest_m else 1

    round_m = re.search(r"Current round:\s*(\d+)\s*of\s*(\d+)", ref)
    cur_round = int(round_m.group(1)) if round_m else 1

    results_m = re.search(r"Quest results so far:\s*\[([^\]]*)\]", ref)
    quest_results = []
    if results_m:
        for tok in results_m.group(1).split(","):
            t = tok.strip().lower()
            if "pass" in t or "success" in t:
                quest_results.append("pass")
            elif "fail" in t:
                quest_results.append("fail")

    leader_m = re.search(r"Quest leader:\s*Player\s*(\d+)", ref)
    leader = int(leader_m.group(1)) if leader_m else None

    team_m = re.search(r"Proposed team:\s*\[([^\]]*)\]", ref)
    proposed_team = []
    if team_m:
        for tok in team_m.group(1).split(","):
            m = re.search(r"Player\s*(\d+)", tok.strip())
            if m:
                proposed_team.append(int(m.group(1)))

    num_players = len(state_dict)
    roles = []
    for pid in range(num_players):
        obs = state_dict.get(str(pid), "")
        rm = re.search(r"Your role:\s*(\w[\w ]*)", obs)
        if rm:
            name = rm.group(1).strip()
            is_good = name in AVALON_GOOD_ROLES
            roles.append({"name": name, "is_good": is_good})
        else:
            roles.append(None)

    return {
        "phase": phase,
        "quest": cur_quest,
        "round": cur_round,
        "quest_results": quest_results,
        "leader": leader,
        "proposed_team": proposed_team,
        "num_players": num_players if num_players >= 5 else 5,
        "roles": roles,
    }


def render_avalon_frame(state_text, step, reward, total_reward, action):
    parsed = _parse_avalon_state(state_text)

    actions = {}
    try:
        action_dict = json.loads(action)
        if isinstance(action_dict, dict):
            for pid, act in action_dict.items():
                actions[str(pid)] = str(act).strip()[:200]
    except (json.JSONDecodeError, TypeError):
        actions["0"] = str(action)[:200]

    frame_data = {
        **parsed,
        "actions": actions,
        "step": step,
        "reward": float(reward),
        "total_reward": float(total_reward),
    }

    page = _get_avalon_page()
    page.evaluate(f"renderFrame({json.dumps(frame_data)})")
    screenshot_bytes = page.screenshot()
    return Image.open(io.BytesIO(screenshot_bytes))


# ── Diplomacy renderer (official diplomacy package map) ──────────────────

def _parse_diplomacy_powers(state_text: str) -> tuple:
    try:
        state_dict = json.loads(state_text)
    except (json.JSONDecodeError, TypeError):
        return "?", {}

    ref = list(state_dict.values())[0] if state_dict else ""
    phase_m = re.search(r"Phase:\s*(\S+)", ref)
    phase = phase_m.group(1) if phase_m else "?"

    powers_info = {}
    for pname in ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]:
        ptext = state_dict.get(pname, ref)
        u_m = re.search(rf"{pname}[^:]*:\s*\d+\s*centers,\s*units=\[([^\]]*)\]", ref)
        units = []
        if u_m and u_m.group(1).strip():
            units = [u.strip().strip("'\"") for u in u_m.group(1).split(",") if u.strip()]
        sc_m = re.search(r"Your supply centers:\s*\[([^\]]*)\]", ptext)
        centers = []
        if sc_m and sc_m.group(1).strip():
            centers = [c.strip().strip("'\"") for c in sc_m.group(1).split(",") if c.strip()]
        powers_info[pname] = {"units": units, "centers": centers}
    return phase, powers_info


def render_diplomacy_map(state_text, step, reward, total_reward, action):
    from diplomacy import Game
    from diplomacy.engine.renderer import Renderer as DipRenderer

    phase, powers_info = _parse_diplomacy_powers(state_text)

    game = Game()
    for pname in game.powers:
        game.powers[pname].units = []
        game.powers[pname].centers = []
        game.powers[pname].retreats = {}
    for pname, info in powers_info.items():
        if pname in game.powers:
            game.powers[pname].units = info["units"]
            game.powers[pname].centers = info["centers"]

    renderer = DipRenderer(game)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        renderer.render(output_path=f.name, incl_abbrev=True)
        svg_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        png_path = f.name

    try:
        subprocess.run(
            ["rsvg-convert", "-w", "900", svg_path, "-o", png_path],
            check=True, capture_output=True,
        )
        map_img = Image.open(png_path).convert("RGB")
    except Exception:
        os.unlink(svg_path)
        os.unlink(png_path)
        return None
    finally:
        for p in (svg_path, png_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    fnt = _pil_font(12)
    info_h = 32
    out = Image.new("RGB", (map_img.width, map_img.height + info_h), (20, 25, 40))
    out.paste(map_img, (0, info_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 2), f"DIPLOMACY  {phase}  Step {step}", fill=(255, 255, 255), font=_pil_font(14))
    draw.text((8, 18), f"Reward: {reward:+.2f}  Total: {total_reward:.1f}",
              fill=(255, 220, 100), font=fnt)

    orders_str = ""
    try:
        a = json.loads(action) if isinstance(action, str) else action
        if isinstance(a, dict):
            parts = []
            for k, v in a.items():
                if v:
                    ords = "; ".join(str(o) for o in v) if isinstance(v, list) else str(v)
                    parts.append(f"{k[:3]}: {ords[:40]}")
            orders_str = "  |  ".join(parts[:3])
    except (json.JSONDecodeError, TypeError):
        pass
    if orders_str:
        draw.text((280, 18), f"Orders: {orders_str[:60]}",
                  fill=(180, 200, 255), font=_pil_font_regular(10))
    return out


# ── Candy Crush renderer ─────────────────────────────────────────────────

def render_candy_frame(state_text, step, reward, total_reward, action):
    board = []
    for line in state_text.split("\n"):
        m = re.match(r"\d+\|\s*(.*)", line.strip())
        if m:
            board.append(m.group(1).strip().split())
    if not board:
        return None
    rows, cols = len(board), len(board[0])
    cell = 52
    pad = 12
    info_h = 80
    W = pad + cols * cell + pad
    H = info_h + pad + rows * cell + pad
    img = Image.new("RGB", (W, H), (40, 30, 50))
    draw = ImageDraw.Draw(img)
    fnt, fnt_cell = _pil_font(15), _pil_font(20)
    draw.text((pad, 8), f"CANDY CRUSH   Step {step}", fill=(255, 255, 255), font=fnt)
    draw.text((pad, 30), f"Action: {action}", fill=(200, 180, 255), font=fnt)
    score_m = re.search(r"Score:\s*(\d+)", state_text)
    moves_m = re.search(r"Moves Left:\s*(\d+)", state_text)
    draw.text((pad, 52),
              f"Score: {score_m.group(1) if score_m else int(total_reward)}  "
              f"Moves: {moves_m.group(1) if moves_m else '?'}  Reward: {reward:+.0f}",
              fill=(255, 220, 100), font=fnt)
    for r, row in enumerate(board):
        for c, ch in enumerate(row):
            x0 = pad + c * cell
            y0 = info_h + pad + r * cell
            color = CANDY_COLORS.get(ch, (120, 120, 120))
            draw.rounded_rectangle([x0 + 2, y0 + 2, x0 + cell - 3, y0 + cell - 3], radius=8, fill=color)
            tw, th = _text_size(draw, ch, fnt_cell)
            draw.text((x0 + (cell - tw) // 2, y0 + (cell - th) // 2), ch, fill=(255, 255, 255), font=fnt_cell)
    return img


# ── Super Mario fallback renderer (used when NES env unavailable) ────────

MARIO_ELEM_COLORS = {
    "Mario": (228, 50, 50), "Bricks": (160, 100, 40),
    "Question Blocks": (255, 200, 50), "Goomba": (140, 80, 40),
    "Koopas": (40, 160, 40), "Warp Pipe": (40, 180, 40),
    "Mushrooms": (200, 60, 60), "Stair Blocks": (120, 120, 120),
    "Flag": (255, 255, 100),
}


def render_mario_frame(state_text, step, reward, total_reward, action):
    elements = {}
    mario = re.search(r"Position of Mario:\s*\((\d+),\s*(\d+)\)", state_text)
    if mario:
        elements["Mario"] = [(int(mario.group(1)), int(mario.group(2)))]
    for label in ["Bricks", "Question Blocks", "Inactivated Blocks", "Monster Goomba",
                  "Monster Koopas", "Warp Pipe", "Item Mushrooms", "Stair Blocks", "Flag"]:
        m = re.search(rf"- {label}:\s*(.*)", state_text)
        if m and "None" not in m.group(1):
            coords = re.findall(r"\((\d+),\s*(\d+)(?:,\s*\d+)?\)", m.group(1))
            if coords:
                short = label.replace("Monster ", "").replace("Item ", "")
                elements[short] = [(int(x), int(y)) for x, y in coords]
    if not elements:
        return None
    W, H, ground_y = 560, 300, 210
    img = Image.new("RGB", (W, H), (107, 140, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, ground_y, W, H], fill=(139, 90, 43))
    draw.rectangle([0, ground_y, W, ground_y + 4], fill=(80, 180, 80))
    fnt, fnt_sm = _pil_font(13), _pil_font(11)
    draw.text((8, 6), f"SUPER MARIO  Step {step}", fill=(255, 255, 255), font=_pil_font(15))
    draw.text((8, 28), f"Action: {action}  Reward: {reward:+.0f}  Total: {total_reward:.0f}",
              fill=(255, 220, 100), font=fnt)
    scale, ox = 1.2, 20
    for label, coords in elements.items():
        color = MARIO_ELEM_COLORS.get(label, (200, 200, 200))
        for (x, y) in coords:
            sx, sy = int(x * scale) + ox, ground_y - int(y * scale)
            sz = 20 if label == "Mario" else 14
            if label == "Mario":
                draw.ellipse([sx - sz // 2, sy - sz, sx + sz // 2, sy], fill=color)
                draw.text((sx - 4, sy - sz - 12), "M", fill=(255, 255, 255), font=fnt_sm)
            elif label == "Flag":
                draw.rectangle([sx, sy - 40, sx + 4, sy], fill=(200, 200, 200))
                draw.polygon([(sx + 4, sy - 40), (sx + 24, sy - 30), (sx + 4, sy - 20)], fill=(0, 200, 0))
            else:
                draw.rectangle([sx - sz // 2, sy - sz, sx + sz // 2, sy], fill=color)
    return img


# ── Generic text fallback renderer ───────────────────────────────────────

def render_text_frame(game, state_text, step, reward, total_reward, action):
    W, H = 700, 500
    img = Image.new("RGB", (W, H), (25, 25, 35))
    draw = ImageDraw.Draw(img)
    fnt, fnt_sm = _pil_font(14), _pil_font(11)
    draw.text((12, 8), f"{game.upper()}  Step {step}  Reward: {reward:+.1f}  Total: {total_reward:.1f}",
              fill=(255, 255, 255), font=fnt)
    draw.text((12, 30), f"Action: {action[:80]}", fill=(180, 220, 255), font=fnt_sm)
    lines = textwrap.fill(state_text[:2000], width=90).split("\n")[:32]
    y = 56
    for line in lines:
        draw.text((12, y), line[:100], fill=(180, 180, 180), font=fnt_sm)
        y += 14
    return img


# ═══════════════════════════════════════════════════════════════════════════
# Renderer dispatch
# ═══════════════════════════════════════════════════════════════════════════

RENDERERS = {
    "tetris":             render_tetris_env,
    "twenty_forty_eight": render_2048_env,
    "sokoban":            render_sokoban_env,
    "candy_crush":        render_candy_env,
    "super_mario":        render_mario_frame,
    "avalon":             render_avalon_frame,
    "diplomacy":          render_diplomacy_map,
}


# ═══════════════════════════════════════════════════════════════════════════
# Episode discovery + GIF assembly
# ═══════════════════════════════════════════════════════════════════════════

def find_best_episodes() -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for sp in glob.glob(str(OUTPUT_DIR / "**" / "rollout_summary.json"), recursive=True):
        with open(sp) as f:
            data = json.load(f)
        game = data.get("game", "unknown")
        stats = data.get("episode_stats", [])
        if not stats:
            continue
        top = max(stats, key=lambda s: s["total_reward"])
        if game not in best or top["total_reward"] > best[game]["reward"]:
            ep_dir = Path(sp).parent
            ep_file = ep_dir / f"episode_{top['episode_index']:03d}.json"
            if ep_file.exists():
                best[game] = {
                    "run": str(Path(sp).relative_to(OUTPUT_DIR)),
                    "episode_index": top["episode_index"],
                    "reward": top["total_reward"],
                    "steps": top["steps"],
                    "episode_path": str(ep_file),
                }
    return best


def load_episode(path: str):
    with open(path) as f:
        return json.load(f)["experiences"]


def generate_gif(game: str, experiences: list, out_path: Path,
                 fps: float = 2.0, max_frames: int = 200):
    frames: List[Image.Image] = []
    cumulative = 0.0
    for i, exp in enumerate(experiences[:max_frames]):
        state = exp.get("state", "")
        action = exp.get("action", "?")
        reward = exp.get("reward", 0.0)
        cumulative += reward

        renderer = RENDERERS.get(game)
        if renderer:
            frame = renderer(state, i, reward, cumulative, action)
        else:
            frame = render_text_frame(game, state, i, reward, cumulative, action)

        if frame is not None:
            frames.append(frame)

    if not frames:
        print(f"  [!] No frames rendered for {game}")
        return

    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    uniform = []
    for f in frames:
        if f.size != (max_w, max_h):
            new = Image.new("RGB", (max_w, max_h), (0, 0, 0))
            new.paste(f, (0, 0))
            uniform.append(new)
        else:
            uniform.append(f)

    duration_ms = int(1000 / fps)
    uniform[0].save(out_path, save_all=True, append_images=uniform[1:],
                    duration=duration_ms, loop=0)
    print(f"  -> {out_path.name}  ({len(uniform)} frames, {fps} fps, {out_path.stat().st_size // 1024}KB)")


def main():
    global OUTPUT_DIR
    import argparse
    parser = argparse.ArgumentParser(description="Generate replay GIFs from episode data")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to output/ directory containing rollout data")
    args = parser.parse_args()
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    REPLAY_DIR.mkdir(exist_ok=True)
    best = find_best_episodes()

    print("=" * 72)
    print("BEST EPISODES PER GAME")
    print("=" * 72)
    for game in sorted(best):
        info = best[game]
        print(f"  {game:22s}  ep {info['episode_index']:3d}  "
              f"reward {info['reward']:10.1f}  steps {info['steps']:4d}")
    print()

    for game in sorted(best):
        info = best[game]
        print(f"Rendering {game} ...")
        try:
            experiences = load_episode(info["episode_path"])
            out = REPLAY_DIR / f"best_{game}.gif"
            fps = 3.0 if game in ("tetris", "candy_crush", "twenty_forty_eight") else 1.5
            generate_gif(game, experiences, out, fps=fps)
        except Exception as exc:
            print(f"  [!] {game} failed: {exc}")

    # Clean up env adapters
    for env in [_tetris_env, _2048_env, _sokoban_env, _candy_env]:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    # Clean up Playwright browser
    if _avalon_browser is not None:
        try:
            _avalon_browser.close()
        except Exception:
            pass
    if _avalon_pw is not None:
        try:
            _avalon_pw.stop()
        except Exception:
            pass

    # Try NES emulator replay for Super Mario (needs orak-mario env)
    mario_script = Path(__file__).parent / "generate_mario_replay.py"
    orak_python = Path("/workspace/anaconda3/envs/orak-mario/bin/python")
    if "super_mario" in best and mario_script.exists() and orak_python.exists():
        print("\nRunning NES emulator replay for Super Mario ...")
        import subprocess
        result = subprocess.run(
            [str(orak_python), str(mario_script)],
            capture_output=True, text=True,
            env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  [!] NES replay failed: {result.stderr[-300:]}")

    print(f"\nAll replays saved to {REPLAY_DIR}/")


if __name__ == "__main__":
    main()
