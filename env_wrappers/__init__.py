"""
Env wrappers: convert environment state to/from natural language for language-model agents.

Available wrappers:
  - AvalonNLWrapper:       Avalon hidden-role deduction (5-10 agents)
  - DiplomacyNLWrapper:    Diplomacy strategic negotiation (7 agents / powers)
  - GamingAgentNLWrapper:  GamingAgent / LMGame-Bench (2048, Candy Crush, Tetris, Super Mario)
  - OrakNLWrapper:         Orak environments (Super Mario, StarCraft II)
  - TetrisMacroWrapper:    Tetris macro-action wrapper (placement-level actions)
"""

from env_wrappers.avalon_nl_wrapper import (
    AvalonNLWrapper,
    state_to_natural_language as avalon_state_to_nl,
    state_to_natural_language_for_all as avalon_state_to_nl_all,
    parse_vote as avalon_parse_vote,
    parse_team as avalon_parse_team,
    parse_target as avalon_parse_target,
)

from env_wrappers.diplomacy_nl_wrapper import (
    DiplomacyNLWrapper,
    state_to_natural_language as diplomacy_state_to_nl,
    state_to_natural_language_for_all as diplomacy_state_to_nl_all,
    parse_orders as diplomacy_parse_orders,
)

from env_wrappers.gamingagent_nl_wrapper import (
    GamingAgentNLWrapper,
    state_to_natural_language as gamingagent_state_to_nl,
)

from env_wrappers.orak_nl_wrapper import (
    ORAK_GAMES,
    OrakNLWrapper,
    make_orak_env,
)

from env_wrappers.tetris_macro_wrapper import TetrisMacroWrapper

__all__ = [
    # Avalon
    "AvalonNLWrapper",
    "avalon_state_to_nl",
    "avalon_state_to_nl_all",
    "avalon_parse_vote",
    "avalon_parse_team",
    "avalon_parse_target",
    # Diplomacy
    "DiplomacyNLWrapper",
    "diplomacy_state_to_nl",
    "diplomacy_state_to_nl_all",
    "diplomacy_parse_orders",
    # GamingAgent
    "GamingAgentNLWrapper",
    "gamingagent_state_to_nl",
    # Orak
    "ORAK_GAMES",
    "OrakNLWrapper",
    "make_orak_env",
    # Tetris Macro
    "TetrisMacroWrapper",
]
