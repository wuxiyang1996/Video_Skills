"""
Diplomacy game env wrapper: state <-> natural language, gym-style multi-agent interface.

Wraps the diplomacy.Game engine into a synchronous reset()/step() loop where every
power receives a natural-language observation and returns natural-language orders.

Supports multi-agent (all 7 powers controlled externally) and single-agent (one power
controlled, the rest use a partner_policy callback).

Usage (multi-agent, from codebase root):

    from env_wrappers.diplomacy_nl_wrapper import DiplomacyNLWrapper

    env = DiplomacyNLWrapper()
    obs, info = env.reset()
    # obs: {power_name: nl_str}  e.g. {"FRANCE": "...", "ENGLAND": "...", ...}

    # Each step: pass orders for each power
    actions = {
        "FRANCE": ["A PAR - BUR", "F BRE - ENG", "A MAR H"],
        "ENGLAND": ["F LON - ENG", "F EDI - NTH", "A LVP - YOR"],
        # ... other powers ...
    }
    obs, rewards, terminated, truncated, info = env.step(actions)

Usage (single-agent):

    env = DiplomacyNLWrapper(controlled_power="FRANCE")
    obs, info = env.reset()           # obs: str (NL for FRANCE)
    obs, reward, term, trunc, info = env.step(["A PAR - BUR", "F BRE H", "A MAR - SPA"])

Negotiation mode:
    If negotiation_rounds > 0, each movement phase has a negotiation step before orders.
    Use step_negotiate() to exchange messages, then step() to submit orders.
    Or set auto_negotiate=True to skip negotiation (default).
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Prefer local AI_Diplomacy/diplomacy package so "from diplomacy import Game" finds it
import sys
from pathlib import Path as _Path
_diplomacy_wrapper_dir = _Path(__file__).resolve().parent
_codebase_root = _diplomacy_wrapper_dir.parent
_ai_diplomacy_dir = _codebase_root / "AI_Diplomacy"
if _ai_diplomacy_dir.exists() and str(_ai_diplomacy_dir) not in sys.path:
    sys.path.insert(0, str(_ai_diplomacy_dir))

try:
    from diplomacy import Game
except ImportError as _e:
    # If still failing, try pip package or re-raise with a helpful message
    Game = None  # type: ignore
    _dip_import_error = _e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

_ORDER_TYPE_HELP = {
    "M": (
        "Movement phase. Issue orders for each unit:\n"
        "  Hold:         A PAR H\n"
        "  Move:         A PAR - BUR\n"
        "  Support hold: A MAR S A PAR\n"
        "  Support move: A MAR S A PAR - BUR\n"
        "  Convoy:       F ENG C A LON - BRE"
    ),
    "R": (
        "Retreat phase. Dislodged units must retreat or disband:\n"
        "  Retreat: A PAR R MAR\n"
        "  Disband: A PAR D"
    ),
    "A": (
        "Adjustment phase. Build or disband units:\n"
        "  Build:   A PAR B  or  F BRE B\n"
        "  Disband: A PAR D"
    ),
}


# ---------------------------------------------------------------------------
# Compact structured state summary (for agent context / retrieval)
# ---------------------------------------------------------------------------

def build_structured_state_summary(
    game: "Game",
    power_name: str,
) -> dict:
    """Build a compact structured dict for one Diplomacy power.

    Designed to be fed into ``compact_structured_state()`` from
    ``decision_agents.agent_helper``.

    Returns:
        Dict with short key=value-friendly fields.  Example::

            {"game": "diplomacy", "phase": "S1902M",
             "self": "power:France centers:5",
             "critical": "fronts:ENG,BUR",
             "resources": "units:A PAR,F BRE",
             "objective": "issue_orders"}
    """
    current_phase = game.get_current_phase()
    phase_type = current_phase[-1] if current_phase and current_phase not in ("FORMING", "COMPLETED") else ""
    power = game.powers.get(power_name)

    if power is None or power.is_eliminated():
        return {
            "game": "diplomacy",
            "phase": current_phase,
            "self": f"power:{power_name} ELIMINATED",
        }

    units = list(power.units) if power.units else []
    centers = list(power.centers) if power.centers else []
    units_short = ",".join(units[:6])
    if len(units) > 6:
        units_short += f"..+{len(units) - 6}"

    # Determine orderable locations to hint at action fronts
    orderable_locs = game.get_orderable_locations(power_name) or []
    fronts_short = ",".join(orderable_locs[:5])
    if len(orderable_locs) > 5:
        fronts_short += f"..+{len(orderable_locs) - 5}"

    objective = {
        "M": "issue_move_orders",
        "R": "issue_retreat_orders",
        "A": "build_or_disband",
    }.get(phase_type, "wait")

    summary: dict = {
        "game": "diplomacy",
        "phase": current_phase,
        "self": f"power:{power_name} centers:{len(centers)}",
        "resources": f"units:{units_short}" if units_short else "units:none",
        "objective": objective,
    }
    if fronts_short:
        summary["critical"] = f"locs:{fronts_short}"

    # Compact retreats if any
    if power.retreats:
        ret_units = ",".join(str(u) for u in list(power.retreats.keys())[:3])
        summary["critical"] = summary.get("critical", "") + f" retreat:{ret_units}"

    return summary


# ---------------------------------------------------------------------------
# State -> NL
# ---------------------------------------------------------------------------

def state_to_natural_language(
    game: "Game",
    power_name: str,
    negotiation_messages: Optional[List[str]] = None,
    phase_history: Optional[List[str]] = None,
) -> str:
    """
    Convert Diplomacy game state to natural-language description for one power.

    Args:
        game: diplomacy.Game instance.
        power_name: The power receiving this observation (e.g. "FRANCE").
        negotiation_messages: Recent negotiation messages (optional).
        phase_history: Summary of recent phase results (optional).

    Returns:
        Natural-language state description string.
    """
    current_phase = game.get_current_phase()
    phase_type = current_phase[-1] if current_phase and current_phase not in ("FORMING", "COMPLETED") else ""
    power = game.powers.get(power_name)

    lines = [f"=== Diplomacy — Phase: {current_phase} ==="]
    lines.append(f"You are: {power_name}.")
    lines.append("")

    # Your forces
    if power:
        lines.append(f"Your units: {power.units if power.units else 'None (eliminated)'}")
        lines.append(f"Your supply centers: {list(power.centers)} ({len(power.centers)} total)")
        lines.append(f"Your home centers: {list(power.homes) if power.homes else 'N/A'}")
        if power.retreats:
            lines.append(f"Your dislodged units needing retreat: {dict(power.retreats)}")
        lines.append("")

    # All powers summary
    lines.append("--- All Powers Status ---")
    for pname, pobj in game.powers.items():
        marker = " (you)" if pname == power_name else ""
        if pobj.is_eliminated():
            lines.append(f"  {pname}{marker}: ELIMINATED")
        else:
            lines.append(
                f"  {pname}{marker}: {len(pobj.centers)} centers, "
                f"units={pobj.units}"
            )
    lines.append("")

    # Possible orders for this power
    if power and not power.is_eliminated():
        orderable_locs = game.get_orderable_locations(power_name)
        possible_orders = game.get_all_possible_orders()

        if orderable_locs:
            lines.append("--- Your Possible Orders ---")
            for loc in orderable_locs:
                if loc in possible_orders and possible_orders[loc]:
                    orders_str = ", ".join(possible_orders[loc][:15])
                    more = f" ... (+{len(possible_orders[loc]) - 15} more)" if len(possible_orders[loc]) > 15 else ""
                    lines.append(f"  {loc}: {orders_str}{more}")
            lines.append("")
        else:
            lines.append("No units requiring orders this phase.")
            lines.append("")

    # Order format help
    if phase_type in _ORDER_TYPE_HELP:
        lines.append("--- Order Format ---")
        lines.append(_ORDER_TYPE_HELP[phase_type])
        lines.append("")

    # Phase action prompt
    if game.is_game_done:
        lines.append("Game is over.")
    elif power and power.is_eliminated():
        lines.append("You have been eliminated. No action needed.")
    else:
        lines.append("Submit your orders as a list of order strings.")
        lines.append("Example: [\"A PAR - BUR\", \"F BRE H\", \"A MAR S A PAR - BUR\"]")

    # Recent negotiation messages
    if negotiation_messages:
        lines.append("")
        lines.append("--- Recent Negotiation ---")
        for msg in negotiation_messages[-15:]:
            lines.append(msg)

    # Phase history
    if phase_history:
        lines.append("")
        lines.append("--- Recent History ---")
        for entry in phase_history[-10:]:
            lines.append(entry)

    return "\n".join(lines)


def state_to_natural_language_for_all(
    game: "Game",
    **kwargs,
) -> Dict[str, str]:
    """Return NL observation for every non-eliminated power."""
    return {
        power_name: state_to_natural_language(game, power_name, **kwargs)
        for power_name in game.powers
    }


# ---------------------------------------------------------------------------
# Order parsing
# ---------------------------------------------------------------------------

def parse_orders(text: Any, game: "Game", power_name: str) -> List[str]:
    """
    Parse order strings from various input formats.

    Accepts:
      - List[str]: ["A PAR - BUR", "F BRE H"]
      - str: "A PAR - BUR, F BRE H" or "A PAR - BUR\\nF BRE H" or JSON list
      - None/empty: returns empty list

    Validates against possible orders. Invalid orders are dropped.
    """
    if text is None:
        return []

    # Already a list
    if isinstance(text, (list, tuple)):
        raw_orders = [str(o).strip() for o in text if o]
    elif isinstance(text, str):
        text = text.strip()
        # Try JSON parse
        if text.startswith("["):
            import json
            try:
                raw_orders = json.loads(text)
                raw_orders = [str(o).strip() for o in raw_orders]
            except (json.JSONDecodeError, TypeError):
                raw_orders = []
        else:
            # Split by newlines or commas (but not commas inside order like "S A PAR - BUR")
            # Orders are generally separated by newlines or semicolons
            import re
            raw_orders = re.split(r"[;\n]+", text)
            raw_orders = [o.strip().strip(",").strip("'\"") for o in raw_orders if o.strip()]
    else:
        return []

    if not raw_orders:
        return []

    # Validate against possible orders
    possible_orders = game.get_all_possible_orders()
    orderable_locs = game.get_orderable_locations(power_name) or []
    valid_set = set()
    for loc in orderable_locs:
        if loc in possible_orders:
            valid_set.update(possible_orders[loc])

    validated = [o for o in raw_orders if o in valid_set]
    return validated


# ---------------------------------------------------------------------------
# Default partner policy (random orders)
# ---------------------------------------------------------------------------

def _random_partner_orders(game: "Game", power_name: str) -> List[str]:
    """Generate random valid orders for a power."""
    possible_orders = game.get_all_possible_orders()
    orderable_locs = game.get_orderable_locations(power_name) or []
    orders = []
    for loc in orderable_locs:
        if loc in possible_orders and possible_orders[loc]:
            orders.append(random.choice(possible_orders[loc]))
    return orders


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class DiplomacyNLWrapper:
    """
    Gym-style wrapper for the Diplomacy game engine with natural-language observations
    and actions.

    Multi-agent mode (default):
        - reset() -> (obs_dict, info)   where obs_dict = {power_name: nl_string}
        - step(actions) -> (obs_dict, rewards_dict, terminated, truncated, info)
          actions: {power_name: orders} where orders is List[str] or a single str

    Single-agent mode (controlled_power is set):
        - reset() -> (obs_str, info)
        - step(orders) -> (obs_str, reward, terminated, truncated, info)
          Other powers use partner_policy.

    Each call to step() processes one full game phase (submit orders for all powers,
    then process). The game advances through Spring Movement -> Spring Retreat ->
    Fall Movement -> Fall Retreat -> Fall Adjustment -> next year.
    """

    def __init__(
        self,
        controlled_power: Optional[str] = None,
        partner_policy: Optional[Callable] = None,
        map_name: str = "standard",
        max_phases: int = 100,
        seed: int = 42,
    ):
        """
        Args:
            controlled_power: If set (e.g. "FRANCE"), single-agent mode.
                If None, multi-agent mode (all 7 powers controlled externally).
            partner_policy: Callable(game, power_name) -> List[str] for non-controlled
                powers in single-agent mode. If None, uses random orders.
            map_name: Diplomacy map name (default "standard").
            max_phases: Maximum number of phases before truncation.
            seed: Random seed.
        """
        if Game is None:
            _msg = (
                "Cannot import diplomacy.Game. "
                "Either install the diplomacy package (pip install diplomacy>=1.1.2) or use the "
                "local AI_Diplomacy copy: run from codebase root so AI_Diplomacy is on PYTHONPATH, "
                "and install AI_Diplomacy dependencies (e.g. pip install coloredlogs; see AI_Diplomacy/)."
            )
            _err = globals().get("_dip_import_error")
            if _err is not None:
                _msg += f" Original error: {_err}"
            raise ImportError(_msg)
        self._controlled_power = controlled_power.upper() if controlled_power else None
        self._multi_agent = controlled_power is None
        self._partner_policy = partner_policy or _random_partner_orders
        self._map_name = map_name
        self._max_phases = max_phases
        self._seed = seed

        self.game: Optional[Game] = None
        self._phases_processed: int = 0
        self._phase_history: List[str] = []
        self._negotiation_log: List[str] = []

    # ---- Gym-like interface ----

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Any, dict]:
        """
        Reset the Diplomacy game.

        Returns:
            Multi-agent:  ({power_name: nl_str, ...}, info)
            Single-agent: (nl_str, info)
        """
        if seed is not None:
            self._seed = seed

        self.game = Game(map_name=self._map_name, seed=self._seed)
        self._phases_processed = 0
        self._phase_history = []
        self._negotiation_log = []

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(
        self,
        action: Union[Dict[str, Any], List[str], str, None] = None,
    ) -> Tuple[Any, Any, bool, bool, dict]:
        """
        Submit orders for the current phase and process it.

        Args:
            Multi-agent:  action = {power_name: orders} where orders is List[str] or str.
            Single-agent: action = List[str] or str (orders for controlled power).

        Returns:
            Multi-agent:  (obs_dict, rewards_dict, terminated, truncated, info)
            Single-agent: (obs_str, reward_float, terminated, truncated, info)
        """
        if self.game is None or self.game.is_game_done:
            raise RuntimeError("Game is not running. Call reset() first.")

        current_phase = self.game.get_current_phase()

        if self._multi_agent:
            actions = action if isinstance(action, dict) else {}
            self._submit_orders_multi(actions)
        else:
            self._submit_orders_single(action)

        # Record phase info before processing
        pre_phase = current_phase

        # Process the phase
        self.game.process()
        self._phases_processed += 1

        # Record what happened
        self._record_phase_result(pre_phase)

        terminated = self.game.is_game_done
        truncated = self._phases_processed >= self._max_phases

        obs = self._build_obs()
        info = self._build_info()
        rewards = self._build_rewards()

        if self._multi_agent:
            return obs, rewards, terminated, truncated, info
        else:
            cp = self._controlled_power
            return obs, rewards.get(cp, 0.0), terminated, truncated, info

    def step_negotiate(
        self,
        messages: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ) -> Tuple[Any, dict]:
        """
        Exchange negotiation messages (optional, before submitting orders).

        This is a separate step that does NOT advance the game phase. It only
        records messages that will appear in subsequent observations.

        Args:
            messages: {sender_power: [{"recipient": power_or_"GLOBAL", "content": str}, ...]}

        Returns:
            (obs, info) with updated observations including the new messages.
        """
        if messages:
            for sender, msg_list in messages.items():
                for msg in msg_list:
                    recipient = msg.get("recipient", "GLOBAL")
                    content = msg.get("content", "")
                    self._negotiation_log.append(f"{sender} -> {recipient}: {content}")

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    # ---- Order submission ----

    def _submit_orders_multi(self, actions: Dict[str, Any]) -> None:
        """Submit orders for all powers from the actions dict."""
        for power_name, power in self.game.powers.items():
            if power.is_eliminated():
                continue
            raw_orders = actions.get(power_name, None)
            if raw_orders is not None:
                orders = parse_orders(raw_orders, self.game, power_name)
            else:
                # No orders provided -> random fallback
                orders = _random_partner_orders(self.game, power_name)
            if orders:
                self.game.set_orders(power_name, orders)

    def _submit_orders_single(self, action: Any) -> None:
        """Submit orders for controlled power + partner policy for others."""
        for power_name, power in self.game.powers.items():
            if power.is_eliminated():
                continue
            if power_name == self._controlled_power:
                orders = parse_orders(action, self.game, power_name)
            else:
                orders = self._partner_policy(self.game, power_name)
            if orders:
                self.game.set_orders(power_name, orders)

    # ---- Phase history ----

    def _record_phase_result(self, phase_name: str) -> None:
        """Record a summary of what happened after processing a phase."""
        sc_summary = []
        for pname, pobj in self.game.powers.items():
            if not pobj.is_eliminated():
                sc_summary.append(f"{pname}:{len(pobj.centers)}")
        self._phase_history.append(
            f"[{phase_name}] Processed. Supply centers: {', '.join(sc_summary)}"
        )
        # Clear negotiation log for next phase
        self._negotiation_log = []

    # ---- Observations ----

    def _build_obs(self) -> Any:
        """Build observation(s) for the current state."""
        if self.game is None:
            return {} if self._multi_agent else ""
        if self._multi_agent:
            return state_to_natural_language_for_all(
                self.game,
                negotiation_messages=self._negotiation_log,
                phase_history=self._phase_history,
            )
        else:
            return state_to_natural_language(
                self.game,
                self._controlled_power,
                negotiation_messages=self._negotiation_log,
                phase_history=self._phase_history,
            )

    # ---- Rewards ----

    def _build_rewards(self) -> Dict[str, float]:
        """
        Rewards for all powers. Based on supply center count normalized by 18
        (solo victory threshold). Eliminated powers get -1.0.
        """
        rewards = {}
        for power_name, power in self.game.powers.items():
            if power.is_eliminated():
                rewards[power_name] = -1.0
            else:
                rewards[power_name] = len(power.centers) / 18.0
        return rewards

    # ---- Info ----

    def _build_info(self) -> dict:
        """Build info dict with game metadata."""
        if self.game is None:
            return {}

        current_phase = self.game.get_current_phase()
        phase_type = current_phase[-1] if current_phase and current_phase not in ("FORMING", "COMPLETED") else ""

        powers_info = {}
        for pname, pobj in self.game.powers.items():
            powers_info[pname] = {
                "units": list(pobj.units),
                "centers": list(pobj.centers),
                "homes": list(pobj.homes) if pobj.homes else [],
                "num_centers": len(pobj.centers),
                "eliminated": pobj.is_eliminated(),
                "retreats": dict(pobj.retreats) if pobj.retreats else {},
            }

        # Orderable locations per power
        active_powers = {}
        for pname, pobj in self.game.powers.items():
            if not pobj.is_eliminated():
                locs = self.game.get_orderable_locations(pname) or []
                if locs:
                    active_powers[pname] = locs

        # Structured summary for the primary power
        primary_power = self._controlled_power or next(iter(self.game.powers), None)
        structured = (
            build_structured_state_summary(self.game, primary_power)
            if primary_power else {}
        )

        info = {
            "env_name": "diplomacy",
            "game_name": "diplomacy",
            "phase": current_phase,
            "phase_type": phase_type,
            "phases_processed": self._phases_processed,
            "max_phases": self._max_phases,
            "is_game_done": self.game.is_game_done,
            "powers": powers_info,
            "active_powers": active_powers,
            "phase_history": list(self._phase_history),
            "negotiation_log": list(self._negotiation_log),
            "order_history": dict(self.game.order_history) if self.game.order_history else {},
            "structured_state": structured,
        }

        # Add possible orders (can be large; include for active powers only)
        possible_orders = self.game.get_all_possible_orders()
        info["possible_orders"] = {}
        for pname, locs in active_powers.items():
            info["possible_orders"][pname] = {
                loc: possible_orders.get(loc, []) for loc in locs
            }

        return info

    def get_map_svg(self) -> Optional[str]:
        """
        Render the current game map to SVG for the web visualizer.
        Returns None if rendering fails (e.g. game not initialized or renderer unavailable).
        """
        if self.game is None:
            return None
        try:
            from diplomacy.engine.renderer import Renderer
            renderer = Renderer(self.game)
            svg_content = renderer.render(output_path=None, incl_abbrev=True)
            if svg_content:
                try:
                    from games.games.diplomacy.utils import add_legend_to_svg
                    svg_content = add_legend_to_svg(svg_content, renderer.metadata.get("color", {}))
                except Exception:
                    pass
            return svg_content
        except Exception:
            return None

    # ---- Properties ----

    @property
    def powers(self) -> List[str]:
        """List of power names in the game."""
        if self.game:
            return list(self.game.powers.keys())
        return DEFAULT_POWERS

    @property
    def num_agents(self) -> int:
        return len(self.powers)

    @property
    def multi_agent(self) -> bool:
        return self._multi_agent

    @property
    def done(self) -> bool:
        if self.game is None:
            return True
        return self.game.is_game_done or self._phases_processed >= self._max_phases

    @property
    def action_space(self):
        """Informal; actions are lists of order strings. See info['possible_orders']."""
        return None

    @property
    def observation_space(self):
        """Observations are NL strings; no formal gym Space."""
        return None
