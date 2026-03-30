"""Extract first and last skill bank snapshots for a single Diplomacy run.

Picks Qwen3-8B_diplomacy_20260322_234548 (25 training steps, 7 powers).
- "first" = skills discovered during step 0 (the very first rollout)
- "last"  = complete final skill bank after all 25 steps
"""

import json, os
from pathlib import Path
from collections import defaultdict

RUN_DIR = Path(__file__).parent / "runs" / "Qwen3-8B_diplomacy_20260322_234548"
OUTPUT_DIR = Path(__file__).parent / "diplomacy_skillbank_snapshots"
OUTPUT_DIR.mkdir(exist_ok=True)

COUNTRIES = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

step0_mtime = os.path.getmtime(RUN_DIR / "rewards" / "step_0000.jsonl")
step1_mtime = os.path.getmtime(RUN_DIR / "rewards" / "step_0001.jsonl")
FIRST_CUTOFF = step1_mtime


def load_all_raw(run_dir: Path) -> dict[str, list[dict]]:
    result = {}
    for country in COUNTRIES:
        path = run_dir / "skillbank" / "diplomacy" / country / "skill_bank.jsonl"
        skills = []
        if path.exists():
            for line in path.read_text().strip().split("\n"):
                if line.strip():
                    skills.append(json.loads(line))
        result[country] = skills
    return result


def summarize_skill(raw: dict, include_first_episodes: bool = False) -> dict:
    sk = raw["skill"]
    rpt = raw["report"]
    sid = sk["skill_id"]
    phase, intention = sid.rsplit(":", 1)

    subs = sk.get("sub_episodes", [])
    n_subs = len(subs)
    outcomes = defaultdict(int)
    for ep in subs:
        outcomes[ep.get("outcome", "unknown")] += 1
    avg_reward = sum(ep.get("cumulative_reward", 0) for ep in subs) / max(1, n_subs)

    out = {
        "skill_id": sid,
        "phase": phase,
        "intention": intention,
        "name": sk.get("name", ""),
        "description": sk.get("strategic_description", ""),
        "version": sk.get("version", 1),
        "n_instances": sk.get("n_instances", 0),
        "n_sub_episodes": n_subs,
        "outcomes": dict(outcomes),
        "avg_cumulative_reward": round(avg_reward, 4),
        "overall_pass_rate": rpt.get("overall_pass_rate"),
        "failure_signatures": rpt.get("failure_signatures", {}),
        "protocol": {
            "preconditions": sk["protocol"].get("preconditions", []),
            "steps": sk["protocol"].get("steps", []),
            "success_criteria": sk["protocol"].get("success_criteria", []),
            "expected_duration": sk["protocol"].get("expected_duration"),
        },
        "contract_effects": {
            "eff_add": sk.get("contract", {}).get("eff_add", []),
            "eff_del": sk.get("contract", {}).get("eff_del", []),
        },
        "created_at": sk.get("created_at"),
        "updated_at": sk.get("updated_at"),
        "retired": sk.get("retired", False),
        "n_protocol_revisions": len(sk.get("protocol_history", [])),
    }
    if include_first_episodes:
        out["sample_episodes"] = [
            {
                "episode_id": ep["episode_id"],
                "outcome": ep["outcome"],
                "reward": round(ep["cumulative_reward"], 4),
                "n_steps": ep["seg_end"] - ep["seg_start"],
                "tags": ep["intention_tags"],
            }
            for ep in subs[:3]
        ]
    return out


def aggregate(bank: dict[str, list[dict]]) -> dict:
    flat = [s for skills in bank.values() for s in skills]
    phases = defaultdict(int)
    intentions = defaultdict(int)
    for s in flat:
        phases[s["phase"]] += 1
        intentions[s["intention"]] += 1
    total = len(flat)
    valid_pr = [s["overall_pass_rate"] for s in flat if s["overall_pass_rate"] is not None]
    return {
        "total_skills": total,
        "per_country": {c: len(sk) for c, sk in bank.items()},
        "by_phase": dict(sorted(phases.items())),
        "by_intention": dict(sorted(intentions.items())),
        "phase_pct": {k: round(v / total * 100, 1) for k, v in sorted(phases.items())} if total else {},
        "intention_pct": {k: round(v / total * 100, 1) for k, v in sorted(intentions.items())} if total else {},
        "avg_pass_rate": round(sum(valid_pr) / max(1, len(valid_pr)), 4),
        "avg_reward": round(sum(s["avg_cumulative_reward"] for s in flat) / max(1, total), 4),
    }


all_raw = load_all_raw(RUN_DIR)

first_bank = {}
last_bank = {}

for country in COUNTRIES:
    first_skills = []
    last_skills = []
    for raw in all_raw[country]:
        sk = raw["skill"]
        last_skills.append(summarize_skill(raw))
        if sk["created_at"] < FIRST_CUTOFF:
            first_skills.append(summarize_skill(raw, include_first_episodes=True))
    first_skills.sort(key=lambda s: s["created_at"])
    last_skills.sort(key=lambda s: s["created_at"])
    first_bank[country] = first_skills
    last_bank[country] = last_skills

first_out = {
    "_meta": {
        "run": "Qwen3-8B_diplomacy_20260322_234548",
        "snapshot": "first",
        "description": "Skills discovered during step 0 (initial rollout, 4 episodes per power)",
        "step": 0,
        "cutoff_timestamp": FIRST_CUTOFF,
    },
    "aggregate": aggregate(first_bank),
    "countries": first_bank,
}

last_out = {
    "_meta": {
        "run": "Qwen3-8B_diplomacy_20260322_234548",
        "snapshot": "last",
        "description": "Complete skill bank after 25 training steps",
        "step": 24,
    },
    "aggregate": aggregate(last_bank),
    "countries": last_bank,
}

first_path = OUTPUT_DIR / "first_skillbank.json"
last_path = OUTPUT_DIR / "last_skillbank.json"
with open(first_path, "w") as f:
    json.dump(first_out, f, indent=2)
with open(last_path, "w") as f:
    json.dump(last_out, f, indent=2)

print(f"Saved {first_path}")
print(f"Saved {last_path}")

# ── Pretty-print comparison ────────────────────────────────────────
def print_bank(label, output):
    a = output["aggregate"]
    bank = output["countries"]
    print(f"\n{'='*65}")
    print(f" {label}")
    print(f" {output['_meta']['description']}")
    print(f"{'='*65}")
    print(f"  Total skills : {a['total_skills']}")
    print(f"  Per country  : {a['per_country']}")
    print(f"  By phase     : {a['by_phase']}  →  {a['phase_pct']}")
    print(f"  By intention : {a['by_intention']}  →  {a['intention_pct']}")
    print(f"  Avg pass rate: {a['avg_pass_rate']}")
    print(f"  Avg reward   : {a['avg_reward']}")
    for country in COUNTRIES:
        skills = bank[country]
        print(f"\n  {country} ({len(skills)} skills):")
        for s in skills:
            new = ""
            if s["created_at"] >= FIRST_CUTOFF:
                new = " [NEW]"
            print(f"    {s['skill_id']:25s}  v{s['version']}  "
                  f"inst={s['n_instances']:3d}  pass={s['overall_pass_rate']:.2f}  "
                  f"r={s['avg_cumulative_reward']:+.3f}  rev={s['n_protocol_revisions']}{new}")


print_bank("FIRST SKILL BANK (step 0)", first_out)
print_bank("LAST SKILL BANK  (step 24)", last_out)

# ── Delta summary ──────────────────────────────────────────────────
fa = first_out["aggregate"]
la = last_out["aggregate"]
print(f"\n{'='*65}")
print(f" CHANGES: first → last")
print(f"{'='*65}")
print(f"  Skills      : {fa['total_skills']} → {la['total_skills']}  (+{la['total_skills']-fa['total_skills']})")
print(f"  Pass rate   : {fa['avg_pass_rate']:.3f} → {la['avg_pass_rate']:.3f}")
print(f"  Avg reward  : {fa['avg_reward']:.3f} → {la['avg_reward']:.3f}")

new_phases = set(la["by_phase"]) - set(fa["by_phase"])
new_intents = set(la["by_intention"]) - set(fa["by_intention"])
if new_phases:
    print(f"  New phases   : {new_phases}")
if new_intents:
    print(f"  New intentions: {new_intents}")

print(f"\n  Per-country changes:")
for country in COUNTRIES:
    f_n = fa["per_country"][country]
    l_n = la["per_country"][country]
    delta = l_n - f_n
    if delta:
        new_ids = set(s["skill_id"] for s in last_bank[country]) - set(s["skill_id"] for s in first_bank[country])
        print(f"    {country:8s}: {f_n:2d} → {l_n:2d}  (+{delta})  new: {', '.join(sorted(new_ids))}")
    else:
        print(f"    {country:8s}: {f_n:2d} → {l_n:2d}  (unchanged)")
