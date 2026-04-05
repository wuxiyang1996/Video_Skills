"""
Walkthrough: how the hybrid (LLM + rule-based) extractor works.

This example simulates a 10-step Avalon trajectory and shows exactly
what each layer (LLM predicates vs rule-based events) contributes to the
final boundary candidates.

Run from repo root:
    python -m skill_agents_grpo.boundary_proposal.example_hybrid

No real LLM call is made — we mock ask_model to show the data flow.
"""

import json
import numpy as np


class FakeExperience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.summary = None
        self.summary_state = None
        self.sub_tasks = None
        self.sub_task_done = None
        self.idx = None


class FakeEpisode:
    def __init__(self, experiences, task):
        self.experiences = experiences
        self.task = task


experiences = [
    FakeExperience(
        "Round 1 Team Selection. Leader: Player 0. Players: 5. Must pick 2 for quest.",
        "propose [Player 0, Player 2]", 0.0,
        "Round 1 Team Vote. Proposed team: [Player 0, Player 2].", False),
    FakeExperience(
        "Round 1 Team Vote. Proposed team: [Player 0, Player 2].",
        "approve", 0.0,
        "Round 1 Team Approved. Team goes on quest.", False),
    FakeExperience(
        "Round 1 Quest. You are on the quest team. You are Merlin (Good).",
        "succeed", 1.0,
        "Round 1 Quest Result: SUCCESS (2 succeed, 0 fail).", False),
    FakeExperience(
        "Round 2 Team Selection. Leader: Player 1. Score: Good 1, Evil 0.",
        "propose [Player 1, Player 3]", 0.0,
        "Round 2 Team Vote. Proposed team: [Player 1, Player 3].", False),
    FakeExperience(
        "Round 2 Team Vote. Proposed team: [Player 1, Player 3].",
        "reject", 0.0,
        "Round 2 Team Rejected. Vote track: 1/5.", False),
    FakeExperience(
        "Round 2 Team Selection (attempt 2). Leader: Player 2.",
        "propose [Player 0, Player 2]", 0.0,
        "Round 2 Team Vote. Proposed team: [Player 0, Player 2].", False),
    FakeExperience(
        "Round 2 Team Vote. Proposed team: [Player 0, Player 2].",
        "approve", 0.0,
        "Round 2 Team Approved. Team goes on quest.", False),
    FakeExperience(
        "Round 2 Quest. You are on the quest team.",
        "succeed", 1.0,
        "Round 2 Quest Result: SUCCESS (2 succeed, 0 fail).", False),
    FakeExperience(
        "Round 3 Team Selection. Leader: Player 3. Score: Good 2, Evil 0.",
        "propose [Player 0, Player 2, Player 4]", 0.0,
        "Round 3 Team Vote. Proposed team: [Player 0, Player 2, Player 4].", False),
    FakeExperience(
        "Round 3 Team Vote. Proposed team: [Player 0, Player 2, Player 4].",
        "approve", 1.0,
        "Good wins 3-0! Assassination phase begins.", True),
]

episode = FakeEpisode(experiences, task="win 3 quests as Good team")


def mock_ask_model(prompt, **kwargs):
    """Simulate what the LLM returns for predicate extraction."""
    mock_predicates = [
        {"phase": "team_selection", "round": 1, "role": "leader",    "quest_score": "0-0"},
        {"phase": "team_vote",      "round": 1, "role": "voter",     "quest_score": "0-0"},
        {"phase": "quest",          "round": 1, "role": "quester",   "quest_score": "0-0"},
        {"phase": "team_selection", "round": 2, "role": "non_leader","quest_score": "1-0"},
        {"phase": "team_vote",      "round": 2, "role": "voter",     "quest_score": "1-0"},
        {"phase": "team_selection", "round": 2, "role": "non_leader","quest_score": "1-0"},
        {"phase": "team_vote",      "round": 2, "role": "voter",     "quest_score": "1-0"},
        {"phase": "quest",          "round": 2, "role": "quester",   "quest_score": "1-0"},
        {"phase": "team_selection", "round": 3, "role": "non_leader","quest_score": "2-0"},
        {"phase": "game_end",       "round": 3, "role": "voter",     "quest_score": "3-0"},
    ]
    return json.dumps(mock_predicates)


from skill_agents_grpo.boundary_proposal.signal_extractors import get_signal_extractor

hybrid = get_signal_extractor(
    "llm+avalon",
    ask_model_fn=mock_ask_model,
    model="mock",
    chunk_size=30,
)

predicates = hybrid.extract_predicates(experiences)
event_times = hybrid.extract_event_times(experiences)

print("=" * 70)
print("HYBRID EXTRACTOR WALKTHROUGH: Avalon")
print("=" * 70)

print("\n-- Layer 1: LLM-extracted predicates --")
print("(LLM reads NL states, returns structured facts)")
for t, p in enumerate(predicates):
    print(f"  t={t:2d}: {p}")

print("\n-- Layer 1: Predicate FLIPS detected --")
print("(Flip = any predicate key changed from t-1 to t)")
for t in range(1, len(predicates)):
    prev, curr = predicates[t - 1], predicates[t]
    changes = {k: (prev.get(k), curr.get(k)) for k in set(prev) | set(curr) if prev.get(k) != curr.get(k)}
    if changes:
        print(f"  t={t:2d}: {changes}")

print("\n-- Layer 2: Rule-based hard events --")
print("(Avalon rules: phase transitions + done flags)")
for t in event_times:
    reason = []
    if experiences[t].done:
        reason.append("done=True")
    r = experiences[t].reward
    if isinstance(r, (int, float)) and r > 0:
        reason.append(f"reward={r}")
    print(f"  t={t:2d}: {', '.join(reason)}")


from skill_agents_grpo.boundary_proposal import ProposalConfig
from skill_agents_grpo.boundary_proposal.proposal import propose_boundary_candidates, candidate_centers_only

config = ProposalConfig(merge_radius=1, window_half_width=1)

candidates = propose_boundary_candidates(
    T=len(experiences),
    predicates=predicates,
    event_times=event_times,
    config=config,
)

print("\n-- Combined: Boundary candidates --")
print("(Predicate flips + hard events, merged within radius=2)")
for c in candidates:
    print(f"  center={c.center:2d}  ±{c.half_window}  source={c.source}")

centers = candidate_centers_only(candidates)
print(f"\n  Final C = {centers}  (|C|={len(centers)} out of T={len(experiences)})")


print("\n-- Attribution: what caught each boundary --")
for c in candidates:
    t = c.center
    reasons = []
    if "predicate" in c.source:
        for tt in range(max(0, t - 1), min(len(predicates), t + 2)):
            if tt > 0:
                prev_p = predicates[tt - 1]
                curr_p = predicates[tt]
                changes = {k: f"{prev_p.get(k)} -> {curr_p.get(k)}"
                           for k in set(prev_p) | set(curr_p) if prev_p.get(k) != curr_p.get(k)}
                if changes:
                    reasons.append(f"LLM flip at t={tt}: {changes}")
    if "event" in c.source:
        reasons.append(f"Rule-based event (reward={experiences[t].reward}, done={experiences[t].done})")
    for r in reasons:
        print(f"  center={t}: {r}")


print("\n-- Interpretation --")
print("""
  t=1:  Team vote phase          (LLM: phase team_selection -> team_vote)
  t=2:  Quest phase              (LLM: phase team_vote -> quest; Rule: reward=1.0)
  t=3:  New round, team select   (LLM: round 1->2, quest_score 0-0->1-0)
  t=4:  Team rejected            (LLM: phase team_selection -> team_vote)
  t=5:  Re-proposal              (LLM: phase team_vote -> team_selection)
  t=7:  Quest phase again        (LLM: phase team_vote -> quest; Rule: reward=1.0)
  t=8:  Round 3 begins           (LLM: round 2->3, quest_score 1-0->2-0)
  t=9:  Game ends, Good wins     (LLM: phase -> game_end; Rule: done=True)

  The RULE-BASED layer catches quest completions and game end for FREE.
  The LLM layer catches SEMANTIC transitions (phase changes, round shifts)
  that require understanding the game state description.
  Together they give complete boundary coverage.
""")
