# Credits to https://github.com/Quentin18/gymnasium-2048/tree/main for the original 2048 game implementation.
# We thank the author for their work, which serves as an excellent testbed for our agent.

from collections import deque
from typing import Any, Dict, Tuple, Optional, List
import math
import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame, SupportsFloat

from gamingagent.envs.gym_env_adapter import GymEnvAdapter
from gamingagent.modules.core_module import Observation
from gamingagent.envs.env_utils import create_board_image_2048

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
WINDOW_SCORE_HEIGHT = 60
WINDOW_BG_COLOR = (250, 248, 238)

BOARD_PADDING = 20
BOARD_BG_COLOR = (186, 172, 160)
TILE_PADDING = 5
TILE_COLOR_MAP = {
    0: (204, 193, 178),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
TILE_COLOR_DEFAULT = (60, 58, 50)
BORDER_RADIUS = 4

FONT_NAME = "Comic Sans MS"
FONT_DARK_COLOR = (119, 110, 101)
FONT_LIGHT_COLOR = (249, 246, 242)
FONT_SCORE_COLOR = (0, 0, 0)
FONT_SIZE = 40

ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}
ACTION_FROM_STR = {"up": 0, "right": 1, "down": 2, "left": 3}
MAX_REPEAT_ZERO_REWARD = 2

# ---------------------------------------------------------------------------
# Phase detection  (from Game-AI-Agent phase_detector._extract_2048_phases)
# ---------------------------------------------------------------------------

def detect_phase(highest: int, empty: int) -> str:
    occupancy = 1.0 - (empty / 16.0)
    if occupancy < 0.35 and highest <= 32:
        return "opening"
    if occupancy > 0.7 or highest >= 256:
        return "endgame"
    return "midgame"


# ---------------------------------------------------------------------------
# Heuristic board scoring
# ---------------------------------------------------------------------------

def _monotonicity(board_vals: list) -> float:
    total = 0.0
    for row in board_vals:
        inc = dec = 0.0
        for i in range(len(row) - 1):
            a = math.log2(row[i]) if row[i] > 0 else 0
            b = math.log2(row[i + 1]) if row[i + 1] > 0 else 0
            if a > b:
                dec += b - a
            elif b > a:
                inc += a - b
        total += max(inc, dec)
    cols = list(zip(*board_vals))
    for col in cols:
        inc = dec = 0.0
        for i in range(len(col) - 1):
            a = math.log2(col[i]) if col[i] > 0 else 0
            b = math.log2(col[i + 1]) if col[i + 1] > 0 else 0
            if a > b:
                dec += b - a
            elif b > a:
                inc += a - b
        total += max(inc, dec)
    return total


def _smoothness(board_vals: list) -> float:
    score = 0.0
    for r in range(len(board_vals)):
        for c in range(len(board_vals[r])):
            v = board_vals[r][c]
            if v == 0:
                continue
            lv = math.log2(v)
            if c + 1 < len(board_vals[r]) and board_vals[r][c + 1] > 0:
                score -= abs(lv - math.log2(board_vals[r][c + 1]))
            if r + 1 < len(board_vals) and board_vals[r + 1][c] > 0:
                score -= abs(lv - math.log2(board_vals[r + 1][c]))
    return score


def _corner_proximity(board_vals: list) -> str:
    max_val = 0
    max_pos = (0, 0)
    for r in range(len(board_vals)):
        for c in range(len(board_vals[r])):
            if board_vals[r][c] > max_val:
                max_val = board_vals[r][c]
                max_pos = (r, c)
    corners = {
        (0, 0): "top-left", (0, 3): "top-right",
        (3, 0): "bottom-left", (3, 3): "bottom-right",
    }
    best = min(corners, key=lambda c: abs(c[0] - max_pos[0]) + abs(c[1] - max_pos[1]))
    return corners[best]


def _count_merges(board_vals: list) -> int:
    count = 0
    for r in range(len(board_vals)):
        for c in range(len(board_vals[r])):
            v = board_vals[r][c]
            if v == 0:
                continue
            if c + 1 < len(board_vals[r]) and board_vals[r][c + 1] == v:
                count += 1
            if r + 1 < len(board_vals) and board_vals[r + 1][c] == v:
                count += 1
    return count


def _board_to_vals(board_powers: np.ndarray) -> list:
    return [[2 ** int(e) if e != 0 else 0 for e in row] for row in board_powers]


# ---------------------------------------------------------------------------
# Compound skill protocols  (from Game-AI-Agent skill seeds + enrichment)
# ---------------------------------------------------------------------------

SKILL_PROTOCOLS = {
    "opening": {
        "active_skill": "opening:MERGE+SETUP",
        "strategy": "Build a monotonic foundation with the max tile anchored in a corner.",
        "plan": [
            "1. Pick a corner (prefer bottom-left or bottom-right) and keep the largest tile there.",
            "2. Merge small tiles (2s, 4s) aggressively — speed matters more than precision.",
            "3. Build a decreasing row/column from the corner outward.",
        ],
        "done_when": "Max tile >= 64 and board has a clear monotonic pattern.",
        "abort_if": "Max tile dislodged from corner.",
    },
    "midgame": {
        "active_skill": "midgame:POSITION+OPTIMIZE",
        "strategy": "Maintain strict snake/zig-zag ordering. Set up chain merges.",
        "plan": [
            "1. Keep the max tile locked in its corner — NEVER dislodge it.",
            "2. Build descending chains along the edge (e.g. 256→128→64→32 along bottom row).",
            "3. Set up 2-step chain merges before executing them.",
            "4. Prefer moves that improve both monotonicity AND empty cells.",
        ],
        "done_when": "Max tile >= 512 or board well-organized with 5+ empty cells.",
        "abort_if": "Corner anchor lost OR empty < 4.",
    },
    "endgame": {
        "active_skill": "endgame:SURVIVE+MERGE",
        "strategy": "Every move matters. Maximize empty cells. Survive at all costs.",
        "plan": [
            "1. ONLY make moves that create merges — never make a move with 0 merges unless forced.",
            "2. Prioritize moves that free the MOST empty cells.",
            "3. If all moves are bad, pick the one that preserves the corner anchor.",
            "4. Watch for chain merge opportunities — a single good move can clear multiple cells.",
        ],
        "done_when": "Reach 2048 tile or recover to 5+ empty cells.",
        "abort_if": "Game over (no valid moves).",
    },
}


# ---------------------------------------------------------------------------
# Enhanced text representation builder
# ---------------------------------------------------------------------------

def build_enhanced_text_repr(
    board_powers: np.ndarray,
    total_score: int,
    step_num: int,
    recent_actions: List[str],
    recent_rewards: List[float],
    prev_board_powers: Optional[np.ndarray] = None,
) -> str:
    board_vals = _board_to_vals(board_powers)
    flat = [v for row in board_vals for v in row]
    highest = max(flat) if flat else 0
    empty = flat.count(0)
    merges = _count_merges(board_vals)
    phase = detect_phase(highest, empty)
    mono = _monotonicity(board_vals)
    smooth = _smoothness(board_vals)
    corner = _corner_proximity(board_vals)

    lines: List[str] = []

    # ── Structured state summary (Game-AI-Agent format) ──
    lines.append(
        f"game=2048 | step={step_num} | phase={phase} | score={total_score} "
        f"| max_tile={highest} | empty={empty}/16 | merges={merges} "
        f"| corner={corner} | mono={mono:.1f} | smooth={smooth:.1f}"
    )

    # ── Board ──
    lines.append("\n=== BOARD ===")
    for row in board_vals:
        lines.append("  " + "\t".join(str(v) if v > 0 else "." for v in row))

    # ── State delta (what changed since last move) ──
    if prev_board_powers is not None:
        prev_vals = _board_to_vals(prev_board_powers)
        prev_flat = [v for row in prev_vals for v in row]
        prev_highest = max(prev_flat) if prev_flat else 0
        prev_empty = prev_flat.count(0)
        delta_parts = []
        if highest != prev_highest:
            delta_parts.append(f"max_tile {prev_highest}→{highest}")
        if empty != prev_empty:
            delta_parts.append(f"empty {prev_empty}→{empty}")
        if delta_parts:
            lines.append(f"\nChanged: {', '.join(delta_parts)}")

    # ── Urgency detection (Game-AI-Agent _detect_urgency) ──
    if empty < 3:
        lines.append("\n!! URGENCY: board nearly full — must MERGE now !!")
    elif empty < 5 and phase == "endgame":
        lines.append("\n! WARNING: board getting tight — prioritize merges !")

    # ── Recent action/reward context (Game-AI-Agent _build_recent_context) ──
    if recent_actions:
        lines.append("\n=== RECENT ACTIONS ===")
        for a, r in zip(recent_actions[-5:], recent_rewards[-5:]):
            lines.append(f"  {a} → reward {r:.0f}")
        recent_total = sum(recent_rewards[-5:])
        if recent_total == 0 and len(recent_actions) >= 3:
            lines.append("  WARNING: Recent actions got 0 reward. Try a DIFFERENT strategy!")

    # ── Move lookahead ──
    lines.append("\n=== MOVE LOOKAHEAD ===")
    for action_idx in range(4):
        name = ACTION_NAMES[action_idx]
        next_board, score, is_legal = TwentyFortyEightEnv.apply_action(board_powers, action_idx)
        if not is_legal:
            lines.append(f"  {name.upper()}: ILLEGAL (no tiles move)")
            continue
        nv = _board_to_vals(next_board)
        nf = [v for row in nv for v in row]
        n_empty = nf.count(0)
        n_merges = _count_merges(nv)
        n_highest = max(nf)
        n_mono = _monotonicity(nv)
        n_smooth = _smoothness(nv)
        board_str = " / ".join(",".join(str(v) if v > 0 else "." for v in row) for row in nv)
        lines.append(f"  {name.upper()}: +{score}pts | empty={n_empty} merges={n_merges} max={n_highest} mono={n_mono:.1f} smooth={n_smooth:.1f}")
        lines.append(f"    board=[{board_str}]")

    # ── Active skill protocol (Game-AI-Agent compound skills) ──
    skill = SKILL_PROTOCOLS[phase]
    lines.append(f"\n=== ACTIVE SKILL: {skill['active_skill']} ===")
    lines.append(f"Strategy: {skill['strategy']}")
    lines.append("Plan:")
    for step in skill["plan"]:
        lines.append(f"  {step}")
    lines.append(f"Done when: {skill['done_when']}")
    lines.append(f"Abort if: {skill['abort_if']}")

    return "\n".join(lines)


class TwentyFortyEightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        size: int = 4,
        max_pow: int = 16,
        game_name_for_adapter: str = "twenty_forty_eight",
        observation_mode_for_adapter: str = "vision",
        agent_cache_dir_for_adapter: str = "cache/twenty_forty_eight/default_run",
        game_specific_config_path_for_adapter: str = "gamingagent/envs/gym_01_2048/game_env_config.json",
        max_stuck_steps_for_adapter: Optional[int] = 10,
    ) -> None:
        assert size >= 2, "size must be greater of equal than 2"

        self.observation_space = spaces.Box(
            low=0,
            high=max_pow - 1,
            shape=(size, size),
            dtype=np.uint8,
        )

        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.font = None

        self.board_size = size

        self.adapter = GymEnvAdapter(
            game_name=game_name_for_adapter,
            observation_mode=observation_mode_for_adapter,
            agent_cache_dir=agent_cache_dir_for_adapter,
            game_specific_config_path=game_specific_config_path_for_adapter,
            max_steps_for_stuck=max_stuck_steps_for_adapter
        )
        self.current_raw_board: Optional[np.ndarray] = None
        self.current_info_dict: Dict[str, Any] = {}

        # Game-AI-Agent inspired tracking state
        self._recent_actions: List[str] = []
        self._recent_rewards: List[float] = []
        self._prev_board: Optional[np.ndarray] = None
        self._step_num: int = 0

    def _get_raw_board_obs(self) -> np.ndarray:
        return self.board.copy()

    def _get_info(self) -> dict[str, Any]:
        return {
            "board": self.board.copy(),
            "step_score": self.step_score,
            "total_score": self.total_score,
            "max_tile_power": np.max(self.board),
            "is_legal_move": self.is_legal_move,
            "illegal_move_count": self.illegal_move_count,
        }

    def _spawn_tile(self) -> None:
        rows, cols = np.where(self.board == 0)
        if not len(rows):
            return
        index = self.np_random.choice(len(rows))
        value = 1 if self.np_random.random() > 0.1 else 2
        self.board[rows[index], cols[index]] = value

    def _build_text_repr(self) -> str:
        return build_enhanced_text_repr(
            board_powers=self.current_raw_board,
            total_score=self.total_score,
            step_num=self._step_num,
            recent_actions=self._recent_actions,
            recent_rewards=self._recent_rewards,
            prev_board_powers=self._prev_board,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        max_memory: Optional[int] = 10,
        episode_id: int = 1
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        self.board = np.zeros(
            (self.board_size, self.board_size),
            dtype=np.uint8,
        )
        self.step_score = 0
        self.total_score = 0
        self.is_legal_move = True
        self.illegal_move_count = 0

        # Reset Game-AI-Agent tracking state
        self._recent_actions = []
        self._recent_rewards = []
        self._prev_board = None
        self._step_num = 0

        self._spawn_tile()
        self._spawn_tile()

        self.adapter.reset_episode(episode_id)
        self.current_raw_board = self._get_raw_board_obs()
        self.current_info_dict = self._get_info()

        img_path_for_adapter = None
        text_representation_for_adapter = None
        initial_perf_score = self.adapter.calculate_perf_score(0, self.current_info_dict)

        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            create_board_image_2048(self.current_raw_board, img_path_for_adapter, perf_score=initial_perf_score)

        if self.adapter.observation_mode in ["text", "both"]:
            text_representation_for_adapter = self._build_text_repr()

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter,
            max_memory=max_memory
        )

        if self.render_mode == "human":
            self._render_frame()

        return agent_observation, self.current_info_dict

    @staticmethod
    def _transpose(board: np.ndarray) -> np.ndarray:
        return np.transpose(board)

    @staticmethod
    def _reverse(board: np.ndarray) -> np.ndarray:
        return np.flipud(board)

    @staticmethod
    def _cover_up(board: np.ndarray) -> np.ndarray:
        cover_board = np.zeros_like(board)
        for col in range(board.shape[1]):
            up = 0
            for row in range(board.shape[0]):
                if board[row, col] != 0:
                    cover_board[up, col] = board[row, col]
                    up += 1
        return cover_board

    @staticmethod
    def _merge(board: np.ndarray) -> tuple[np.ndarray, int]:
        score = 0
        for row in range(1, board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] != 0 and board[row, col] == board[row - 1, col]:
                    score += 2 ** (board[row, col] + 1)
                    board[row - 1, col] = board[row - 1, col] + 1
                    board[row, col] = 0
        return board, score

    @classmethod
    def _up(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._cover_up(board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        return next_board, score

    @classmethod
    def _right(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._reverse(cls._transpose(board))
        next_board = cls._cover_up(next_board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        next_board = cls._transpose(cls._reverse(next_board))
        return next_board, score

    @classmethod
    def _down(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._reverse(board)
        next_board = cls._cover_up(next_board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        next_board = cls._reverse(next_board)
        return next_board, score

    @classmethod
    def _left(cls, board: np.ndarray) -> tuple[np.ndarray, int]:
        next_board = cls._transpose(board)
        next_board = cls._cover_up(next_board)
        next_board, score = cls._merge(next_board)
        next_board = cls._cover_up(next_board)
        next_board = cls._transpose(next_board)
        return next_board, score

    @classmethod
    def apply_action(
        cls,
        board: np.ndarray,
        action: ActType,
    ) -> tuple[np.ndarray, int, bool]:
        action_func = (cls._up, cls._right, cls._down, cls._left)
        next_board, score = action_func[action](board.copy())
        is_legal = not np.array_equal(board, next_board)
        return next_board, score, is_legal

    @staticmethod
    def is_terminated(board: np.ndarray) -> bool:
        if (board == 0).any(): return False
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if c + 1 < board.shape[1] and board[r, c] == board[r, c+1]: return False
                if r + 1 < board.shape[0] and board[r, c] == board[r+1, c]: return False
        return True

    def _apply_anti_repetition(self, action_str: Optional[str]) -> Optional[str]:
        """Force a different action if same one repeated with zero reward.
        Borrowed from Game-AI-Agent episode_runner._apply_anti_repetition.
        """
        if action_str is None or len(self._recent_actions) < MAX_REPEAT_ZERO_REWARD:
            return action_str
        tail = self._recent_actions[-MAX_REPEAT_ZERO_REWARD:]
        tail_rewards = self._recent_rewards[-MAX_REPEAT_ZERO_REWARD:]
        if all(a == action_str for a in tail) and sum(tail_rewards) <= 0:
            alternatives = [n for n in ACTION_NAMES.values() if n != action_str]
            if alternatives:
                forced = random.choice(alternatives)
                print(f"[AntiRepetition] {action_str} repeated {MAX_REPEAT_ZERO_REWARD}x with 0 reward → forcing {forced}")
                return forced
        return action_str

    def step(
        self,
        agent_action_str: Optional[str],
        thought_process: str = "",
        time_taken_s: float = 0.0
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any], float]:

        self.adapter.increment_step()
        self._step_num += 1
        self._prev_board = self.current_raw_board.copy() if self.current_raw_board is not None else None

        # Anti-repetition guard (Game-AI-Agent)
        agent_action_str = self._apply_anti_repetition(agent_action_str)

        env_action_idx = self.adapter.map_agent_action_to_env_action(agent_action_str)

        reward = 0.0
        terminated = False
        truncated = False
        self.is_legal_move = False
        self.step_score = 0

        if env_action_idx is not None and self.action_space.contains(env_action_idx):
            next_board_state, current_step_score, self.is_legal_move = self.apply_action(
                board=self.board,
                action=env_action_idx
            )
            self.step_score = current_step_score
            self.total_score += self.step_score
            reward = float(self.step_score)

            if self.is_legal_move:
                self.board = next_board_state
                self._spawn_tile()
            else:
                self.illegal_move_count += 1

            terminated = self.is_terminated(board=self.board)
        else:
            print(f"[TwentyFortyEightEnv] Action '{agent_action_str}' is skip/invalid. Gym env not stepped.")
            terminated = self.is_terminated(board=self.board)

        # Track recent actions/rewards (Game-AI-Agent)
        self._recent_actions.append(str(agent_action_str or "none"))
        self._recent_rewards.append(reward)

        self.current_raw_board = self._get_raw_board_obs()
        self.current_info_dict = self._get_info()

        current_perf_score = self.adapter.calculate_perf_score(reward, self.current_info_dict)

        img_path_for_adapter = None
        text_representation_for_adapter = None

        if self.adapter.observation_mode in ["vision", "both"]:
            img_path_for_adapter = self.adapter._create_agent_observation_path(
                self.adapter.current_episode_id, self.adapter.current_step_num
            )
            create_board_image_2048(self.current_raw_board, img_path_for_adapter, perf_score=current_perf_score)

        if self.adapter.observation_mode in ["text", "both"]:
            text_representation_for_adapter = self._build_text_repr()

        agent_observation = self.adapter.create_agent_observation(
            img_path=img_path_for_adapter,
            text_representation=text_representation_for_adapter
        )

        final_terminated, final_truncated = self.adapter.verify_termination(agent_observation, terminated, truncated)

        self.adapter.log_step_data(
            agent_action_str=agent_action_str,
            thought_process=thought_process,
            reward=reward,
            info=self.current_info_dict,
            terminated=final_terminated,
            truncated=final_truncated,
            time_taken_s=time_taken_s,
            perf_score=current_perf_score,
            agent_observation=agent_observation
        )

        if self.render_mode == "human":
            self._render_frame()

        return agent_observation, reward, final_terminated, final_truncated, self.current_info_dict, current_perf_score

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _get_value(self, row: int, col: int) -> int:
        return 2 ** self.board[row, col] if self.board[row, col] > 0 else 0

    @staticmethod
    def _get_background_color(value: int) -> tuple[int, int, int]:
        return TILE_COLOR_MAP.get(value, TILE_COLOR_DEFAULT)

    @staticmethod
    def _get_text_color(value: int) -> tuple[int, int, int]:
        return FONT_DARK_COLOR if value < 8 else FONT_LIGHT_COLOR

    def _draw_board(self, canvas: pygame.Surface) -> None:
        board_left = BOARD_PADDING
        board_top = BOARD_PADDING
        board_width = WINDOW_WIDTH - 2 * BOARD_PADDING
        board_height = WINDOW_HEIGHT - 2 * BOARD_PADDING
        tile_width = (board_width - TILE_PADDING * (self.board_size + 1)) // self.board_size
        tile_height = (board_height - TILE_PADDING * (self.board_size + 1)) // self.board_size

        pygame.draw.rect(
            surface=canvas,
            color=BOARD_BG_COLOR,
            rect=(board_left, board_top, board_width, board_height),
            border_radius=BORDER_RADIUS,
        )
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                value = self._get_value(row=row, col=col)
                rect_x = board_left + TILE_PADDING * (col + 1) + col * tile_width
                rect_y = board_top + TILE_PADDING * (row + 1) + row * tile_height

                tile_rect = pygame.Rect(rect_x, rect_y, tile_width, tile_height)
                pygame.draw.rect(
                    surface=canvas,
                    color=self._get_background_color(value=value),
                    rect=tile_rect,
                    border_radius=BORDER_RADIUS,
                )
                if value == 0:
                    continue

                str_value = str(value)
                current_font_size = FONT_SIZE
                if len(str_value) > 2: current_font_size = int(FONT_SIZE * 0.7)
                if len(str_value) > 3: current_font_size = int(FONT_SIZE * 0.5)
                dynamic_font = pygame.font.SysFont(FONT_NAME, current_font_size)

                text_surface = dynamic_font.render(
                    str_value,
                    True,
                    self._get_text_color(value=value),
                )
                text_rect = text_surface.get_rect(center=tile_rect.center)
                canvas.blit(source=text_surface, dest=text_rect)

    def _draw_score(self, canvas: pygame.Surface) -> None:
        board_width = WINDOW_WIDTH - 2 * BOARD_PADDING
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

        score_surface = self.font.render(
            f"Score: {self.total_score}",
            True,
            FONT_SCORE_COLOR,
        )
        score_height = self.font.get_height()
        score_rect = pygame.Rect(
            BOARD_PADDING,
            WINDOW_HEIGHT + (WINDOW_SCORE_HEIGHT - score_height) // 2,
            board_width,
            score_height,
        )
        canvas.blit(source=score_surface, dest=score_rect)

    def _render_frame(self) -> RenderFrame | list[RenderFrame]:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT + WINDOW_SCORE_HEIGHT)
            )
            pygame.display.set_caption("2048")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

        canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT + WINDOW_SCORE_HEIGHT))
        canvas.fill(WINDOW_BG_COLOR)

        self._draw_board(canvas=canvas)
        self._draw_score(canvas=canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None
        self.adapter.close_log_file()
