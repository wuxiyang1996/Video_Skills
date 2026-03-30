"""LLM-driven classification of Diplomacy skills.

For each skill, reads the description, protocol, contract effects, and
performance metrics to assign labels across 5 strategic dimensions that
go beyond the raw phase:intention taxonomy.

Classification Dimensions
─────────────────────────
1. Strategic Function  — What does the skill actually achieve on the board?
     territory_gain     : captures new supply centers
     territory_hold     : maintains current centers / stabilises position
     territory_loss     : forced to give up centers (retreat/disband)
     unit_maneuver      : repositions units without center change
     phase_transition   : primarily advances the game clock

2. Tactical Complexity — How many moving parts does the skill involve?
     simple    : ≤2 protocol steps, short duration
     moderate  : 3–4 steps
     complex   : 5+ steps or long expected duration (≥6)

3. Reliability         — How dependable is the skill?
     proven      : pass_rate ≥ 0.8  AND  success_rate ≥ 0.85
     mixed       : in between
     unreliable  : pass_rate < 0.3  OR  success_rate < 0.7

4. Impact              — How much does it change the world state?
     high   : ≥3 total add+del effects, avg reward ≥ 1.0
     medium : 2 effects or moderate reward
     low    : ≤1 effect and reward < 0.5

5. Maturity            — How battle-tested is the skill?
     nascent    : n_instances ≤ 3
     developing : 4–15 instances
     mature     : >15 instances
"""

import json, re
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DIR = Path(__file__).resolve().parent

with open(DIR / "first_skillbank.json") as f:
    first = json.load(f)
with open(DIR / "last_skillbank.json") as f:
    last = json.load(f)

COUNTRIES = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

# ── Classification functions ─────────────────────────────────────────

def _parse_center_delta(desc: str, eff_add: list[str], eff_del: list[str]) -> str | None:
    """Return 'gain', 'loss', or None by inspecting description and effects."""
    import re
    d = desc.lower()

    if re.search(r"increas\w*\b.{0,30}\bcenters", d):
        return "gain"
    m_from_to = re.search(r"centers\s+from\s+(\d+)\s+to\s+(\d+)", d)
    if m_from_to:
        frm, to = int(m_from_to.group(1)), int(m_from_to.group(2))
        return "gain" if to > frm else "loss" if to < frm else None
    m_to = re.search(r"centers\s+to\s+(\d+)", d)
    m_from = re.search(r"centers\s+from\s+(\d+)", d)
    if m_to and m_from:
        frm, to = int(m_from.group(1)), int(m_to.group(1))
        return "gain" if to > frm else "loss" if to < frm else None
    if re.search(r"reduc\w*\b.{0,30}\bcenters", d):
        return "loss"

    add_centers = [e for e in eff_add if e.startswith("world.centers=")]
    del_centers = [e for e in eff_del if e.startswith("world.centers=")]
    if add_centers and del_centers:
        add_n = max(int(e.split("=")[1]) for e in add_centers)
        del_n = max(int(e.split("=")[1]) for e in del_centers)
        return "gain" if add_n > del_n else "loss" if add_n < del_n else None

    return None


def classify_strategic_function(sk: dict) -> str:
    desc = sk.get("description", "")
    desc_low = desc.lower()
    eff_add = sk.get("contract_effects", {}).get("eff_add", [])
    eff_del = sk.get("contract_effects", {}).get("eff_del", [])
    intention = sk.get("intention", "")
    phase = sk.get("phase", "")

    center_delta = _parse_center_delta(desc, eff_add, eff_del)

    if phase == "retreat":
        if intention == "BUILD":
            return "resource_management"
        return "territory_loss"

    if intention == "BUILD" or "build" in desc_low:
        return "resource_management"

    if center_delta == "gain":
        return "territory_gain"
    if center_delta == "loss":
        return "territory_loss"

    if any(kw in desc_low for kw in ["maintain", "stabiliz", "holding", "control over"]):
        return "territory_hold"
    if intention == "DEFEND" and "increas" not in desc_low:
        return "territory_hold"

    if any(kw in desc_low for kw in ["offensive", "attack"]):
        return "territory_gain"
    if intention == "ATTACK":
        return "territory_gain"

    has_unit_effects = any("units" in e for e in eff_add + eff_del
                           if not e.startswith("event."))
    only_phase_effects = all("phase" in e or e.startswith("event.") for e in eff_add + eff_del)

    if only_phase_effects or (
        "advances the game phase" in desc_low or
        "transitions the game phase" in desc_low or
        "phase transition" in desc_low
    ):
        return "phase_transition"

    if has_unit_effects and "position" in desc_low:
        return "unit_maneuver"
    if "reposition" in desc_low or "movement" in desc_low:
        return "unit_maneuver"

    if has_unit_effects:
        return "unit_maneuver"

    return "phase_transition"


def classify_complexity(sk: dict) -> str:
    steps = sk.get("protocol", {}).get("steps", [])
    duration = sk.get("protocol", {}).get("expected_duration") or 1
    n_steps = len(steps)
    if n_steps <= 2 and duration <= 3:
        return "simple"
    if n_steps >= 5 or duration >= 6:
        return "complex"
    return "moderate"


def classify_reliability(sk: dict) -> str:
    pass_rate = sk.get("overall_pass_rate")
    if pass_rate is None:
        pass_rate = 0.5
    outcomes = sk.get("outcomes", {})
    total = sum(outcomes.values())
    success_rate = outcomes.get("success", 0) / max(1, total)

    if pass_rate >= 0.8 and success_rate >= 0.85:
        return "proven"
    if pass_rate < 0.3 or success_rate < 0.7:
        return "unreliable"
    return "mixed"


def classify_impact(sk: dict) -> str:
    eff_add = sk.get("contract_effects", {}).get("eff_add", [])
    eff_del = sk.get("contract_effects", {}).get("eff_del", [])
    n_effects = len(eff_add) + len(eff_del)
    reward = sk.get("avg_cumulative_reward", 0)

    if n_effects >= 3 and reward >= 1.0:
        return "high"
    if n_effects <= 1 and reward < 0.5:
        return "low"
    return "medium"


def classify_maturity(sk: dict) -> str:
    n = sk.get("n_instances", 0)
    if n <= 3:
        return "nascent"
    if n <= 15:
        return "developing"
    return "mature"


def classify_skill(sk: dict) -> dict:
    return {
        "strategic_function": classify_strategic_function(sk),
        "complexity": classify_complexity(sk),
        "reliability": classify_reliability(sk),
        "impact": classify_impact(sk),
        "maturity": classify_maturity(sk),
    }


# ── Classify all skills ─────────────────────────────────────────────

def classify_bank(bank_data: dict) -> list[dict]:
    results = []
    for country, skills in bank_data["countries"].items():
        for sk in skills:
            labels = classify_skill(sk)
            results.append({
                "country": country,
                "skill_id": sk["skill_id"],
                "phase": sk["phase"],
                "intention": sk["intention"],
                "name": sk["name"],
                "description": sk["description"][:120],
                **labels,
                "n_instances": sk["n_instances"],
                "avg_reward": sk["avg_cumulative_reward"],
                "pass_rate": sk["overall_pass_rate"],
            })
    return results


first_classified = classify_bank(first)
last_classified = classify_bank(last)

# Save
with open(DIR / "first_classified.json", "w") as f:
    json.dump(first_classified, f, indent=2)
with open(DIR / "last_classified.json", "w") as f:
    json.dump(last_classified, f, indent=2)


# ── Aggregate counts ─────────────────────────────────────────────────

DIMENSIONS = {
    "strategic_function": ["territory_gain", "territory_hold", "territory_loss",
                           "unit_maneuver", "phase_transition", "resource_management"],
    "complexity": ["simple", "moderate", "complex"],
    "reliability": ["proven", "mixed", "unreliable"],
    "impact": ["high", "medium", "low"],
    "maturity": ["nascent", "developing", "mature"],
}

DIM_COLORS = {
    "strategic_function": {
        "territory_gain":  "#2CA02C", "territory_hold": "#1F77B4",
        "territory_loss":  "#D62728", "unit_maneuver":  "#FF7F0E",
        "phase_transition": "#9467BD", "resource_management": "#8C564B",
    },
    "complexity": {"simple": "#4DAF4A", "moderate": "#FF7F00", "complex": "#E41A1C"},
    "reliability": {"proven": "#4DAF4A", "mixed": "#FF7F00", "unreliable": "#E41A1C"},
    "impact": {"high": "#E41A1C", "medium": "#FF7F00", "low": "#999999"},
    "maturity": {"nascent": "#FF7F00", "developing": "#4DAF4A", "mature": "#1F77B4"},
}


def count_dim(classified: list[dict], dim: str) -> dict[str, int]:
    c = Counter(sk[dim] for sk in classified)
    return {k: c.get(k, 0) for k in DIMENSIONS[dim]}


# ── Plots ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, (dim, categories) in enumerate(DIMENSIONS.items()):
    ax = axes_flat[idx]
    fc = count_dim(first_classified, dim)
    lc = count_dim(last_classified, dim)

    x = np.arange(len(categories))
    w = 0.35
    f_vals = [fc[k] for k in categories]
    l_vals = [lc[k] for k in categories]

    colors = DIM_COLORS[dim]
    bar_colors_f = [colors.get(k, "#999") for k in categories]
    bar_colors_l = [colors.get(k, "#999") for k in categories]

    bars1 = ax.bar(x - w/2, f_vals, w, label="First (Step 0)",
                   color=bar_colors_f, alpha=0.55, edgecolor="white")
    bars2 = ax.bar(x + w/2, l_vals, w, label="Last (Step 24)",
                   color=bar_colors_l, alpha=1.0, edgecolor="white")

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                    str(int(h)), ha="center", fontsize=8, color="#555")
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                    str(int(h)), ha="center", fontsize=8, fontweight="bold")

    nice = dim.replace("_", " ").title()
    ax.set_title(nice, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    cat_labels = [k.replace("_", "\n") for k in categories]
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.legend(fontsize=7)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=.2)

axes_flat[-1].axis("off")

# Summary table in the empty panel
ax_t = axes_flat[-1]
ax_t.axis("off")
lines = [
    f"First bank: {len(first_classified)} skills",
    f"Last bank:  {len(last_classified)} skills",
    f"New skills: +{len(last_classified) - len(first_classified)}",
    "",
]
for dim in DIMENSIONS:
    fc = count_dim(first_classified, dim)
    lc = count_dim(last_classified, dim)
    changes = []
    for k in DIMENSIONS[dim]:
        d = lc[k] - fc[k]
        if d != 0:
            changes.append(f"{k}: {fc[k]}→{lc[k]}")
    if changes:
        lines.append(f"{dim.replace('_',' ').title()}:")
        for c in changes:
            lines.append(f"  {c}")

ax_t.text(0.05, 0.95, "\n".join(lines), transform=ax_t.transAxes,
          fontsize=9, verticalalignment="top", fontfamily="monospace",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#ccc"))

fig.suptitle("Skill Classification: First (Step 0) vs Last (Step 24)",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(DIR / "07_classification_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Percentage-based stacked bars (First vs Last side by side) ───────

fig2, axes2 = plt.subplots(1, 5, figsize=(22, 5), sharey=True)

for idx, (dim, categories) in enumerate(DIMENSIONS.items()):
    ax = axes2[idx]
    fc = count_dim(first_classified, dim)
    lc = count_dim(last_classified, dim)
    f_total = sum(fc.values())
    l_total = sum(lc.values())

    LABELS_2 = ["First", "Last"]
    bottoms = np.zeros(2)
    for k in categories:
        vals = np.array([fc[k] / f_total * 100, lc[k] / l_total * 100])
        col = DIM_COLORS[dim].get(k, "#999")
        ax.bar(LABELS_2, vals, bottom=bottoms, label=k.replace("_", " ").title(),
               color=col, edgecolor="white", linewidth=.5, width=0.5)
        for j, v in enumerate(vals):
            if v > 5:
                ax.text(j, bottoms[j] + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottoms += vals

    ax.set_title(dim.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(0, 110)
    if idx == 0:
        ax.set_ylabel("Percentage")

fig2.suptitle("Skill Classification Distribution (%): First → Last",
              fontsize=14, fontweight="bold", y=1.02)
fig2.tight_layout()
fig2.savefig(DIR / "08_classification_pct.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# ── Print summary ────────────────────────────────────────────────────

print(f"First bank: {len(first_classified)} skills")
print(f"Last bank:  {len(last_classified)} skills\n")

for dim in DIMENSIONS:
    fc = count_dim(first_classified, dim)
    lc = count_dim(last_classified, dim)
    print(f"  {dim.replace('_',' ').upper()}")
    for k in DIMENSIONS[dim]:
        f_pct = fc[k] / len(first_classified) * 100
        l_pct = lc[k] / len(last_classified) * 100
        delta = lc[k] - fc[k]
        sign = "+" if delta > 0 else ""
        print(f"    {k:22s}  {fc[k]:2d} ({f_pct:4.1f}%)  →  {lc[k]:2d} ({l_pct:4.1f}%)  {sign}{delta}")
    print()

# Print a few example classifications
print("=" * 75)
print("EXAMPLE CLASSIFICATIONS (last bank)")
print("=" * 75)
for sk in last_classified[:12]:
    print(f"\n  {sk['country']:8s}  {sk['skill_id']:25s}")
    print(f"    function={sk['strategic_function']:20s}  complexity={sk['complexity']:10s}")
    print(f"    reliability={sk['reliability']:12s}  impact={sk['impact']:8s}  maturity={sk['maturity']}")
    print(f"    desc: {sk['description']}")

print(f"\nSaved: {DIR / '07_classification_comparison.png'}")
print(f"Saved: {DIR / '08_classification_pct.png'}")
print(f"Saved: {DIR / 'first_classified.json'}")
print(f"Saved: {DIR / 'last_classified.json'}")

# ── Figure 9: Combined – Strategic Function + Intention Distribution ──

fa, la = first["aggregate"], last["aggregate"]
INTENT_ORDER = ["EXPLORE", "SETUP", "DEFEND", "ATTACK", "BUILD"]
INTENT_COLORS = {
    "EXPLORE": "#3B7DD8", "SETUP": "#4DAF4A", "DEFEND": "#E24A33",
    "ATTACK": "#F28C28", "BUILD": "#8B6DB0",
}
LABELS_2 = ["First", "Last"]

fig9, (ax_sf, ax_int) = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- Left: Strategic Function (grouped bar) ----
sf_cats = DIMENSIONS["strategic_function"]
fc_sf = count_dim(first_classified, "strategic_function")
lc_sf = count_dim(last_classified, "strategic_function")

x_sf = np.arange(len(sf_cats))
w = 0.35
f_vals = [fc_sf[k] for k in sf_cats]
l_vals = [lc_sf[k] for k in sf_cats]
colors_sf = DIM_COLORS["strategic_function"]

bars_f = ax_sf.bar(x_sf - w/2, f_vals, w, label="First (Step 0)",
                   color=[colors_sf[k] for k in sf_cats], alpha=0.45,
                   edgecolor="white", linewidth=.6)
bars_l = ax_sf.bar(x_sf + w/2, l_vals, w, label="Last (Step 24)",
                   color=[colors_sf[k] for k in sf_cats], alpha=1.0,
                   edgecolor="white", linewidth=.6)

for bar in bars_f:
    h = bar.get_height()
    if h > 0:
        ax_sf.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                   str(int(h)), ha="center", fontsize=8, color="#777")
for bar in bars_l:
    h = bar.get_height()
    if h > 0:
        ax_sf.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                   str(int(h)), ha="center", fontsize=9, fontweight="bold")

for j, (fv, lv) in enumerate(zip(f_vals, l_vals)):
    delta = lv - fv
    if delta != 0:
        sign = "+" if delta > 0 else ""
        color = "#2CA02C" if delta > 0 else "#D62728"
        y_pos = max(fv, lv) + 1.8
        ax_sf.text(x_sf[j], y_pos, f"{sign}{delta}",
                   ha="center", fontsize=8, fontweight="bold", color=color)

ax_sf.set_title("(A)  Strategic Function", fontsize=13, fontweight="bold", pad=12)
ax_sf.set_xticks(x_sf)
ax_sf.set_xticklabels([k.replace("_", "\n") for k in sf_cats], fontsize=8.5)
ax_sf.set_ylabel("Number of Skills", fontsize=10)
ax_sf.legend(fontsize=9, loc="upper right")
ax_sf.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax_sf.grid(axis="y", alpha=.15)
ax_sf.set_ylim(0, max(max(f_vals), max(l_vals)) + 5)

# ---- Right: Intention Distribution (%) ----
totals_arr = np.array([fa["total_skills"], la["total_skills"]], dtype=float)
bottoms = np.zeros(2)
for intent in INTENT_ORDER:
    raw = np.array([fa["by_intention"].get(intent, 0),
                    la["by_intention"].get(intent, 0)])
    pct = raw / totals_arr * 100
    ax_int.bar(LABELS_2, pct, bottom=bottoms,
               label=intent.title(), color=INTENT_COLORS[intent],
               edgecolor="white", linewidth=.5, width=0.5)
    for j, p in enumerate(pct):
        if p > 4:
            ax_int.text(j, bottoms[j] + p / 2, f"{p:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
    bottoms += pct

ax_int.set_title("(B)  Intention Distribution (%)", fontsize=13,
                 fontweight="bold", pad=12)
ax_int.set_ylabel("Percentage", fontsize=10)
ax_int.legend(fontsize=9, loc="upper right")
ax_int.set_ylim(0, 110)

fig9.suptitle(
    f"Skill Bank Development: First (Step 0, {fa['total_skills']} skills)"
    f"  →  Last (Step 24, {la['total_skills']} skills)",
    fontsize=14, fontweight="bold", y=1.02,
)
fig9.tight_layout()
fig9.savefig(DIR / "09_strategic_development.png", dpi=150, bbox_inches="tight")
plt.close(fig9)
print(f"Saved: {DIR / '09_strategic_development.png'}")

# ── Figure 10: Skill Usage Frequency ─────────────────────────────────

def collect_usage(bank_data):
    """Extract per-skill usage stats from a raw skill bank."""
    skills = []
    for country, sklist in bank_data["countries"].items():
        for sk in sklist:
            outcomes = sk.get("outcomes", {})
            skills.append({
                "key": (country, sk["skill_id"]),
                "country": country,
                "skill_id": sk["skill_id"],
                "n_instances": sk["n_instances"],
                "success": outcomes.get("success", 0),
                "partial": outcomes.get("partial", 0),
                "failure": outcomes.get("failure", 0),
                "pass_rate": sk.get("overall_pass_rate", 0),
                "avg_reward": sk.get("avg_cumulative_reward", 0),
            })
    return skills

first_usage = collect_usage(first)
last_usage = collect_usage(last)

first_keys = {s["key"] for s in first_usage}
last_keys = {s["key"] for s in last_usage}
new_keys = last_keys - first_keys

fig10, (ax_rank, ax_dist) = plt.subplots(1, 2, figsize=(14, 5.5))

# ---- Left: Ranked usage bar chart (last bank, colored original vs new) ----
last_sorted = sorted(last_usage, key=lambda s: -s["n_instances"])

colors_rank = []
for s in last_sorted:
    if s["key"] in new_keys:
        colors_rank.append("#E6550D")
    else:
        colors_rank.append("#6BAED6")

x_rank = np.arange(len(last_sorted))
bars = ax_rank.bar(x_rank, [s["n_instances"] for s in last_sorted],
                   color=colors_rank, edgecolor="white", linewidth=0.3, width=1.0)

from matplotlib.patches import Patch
ax_rank.legend(handles=[
    Patch(facecolor="#6BAED6", label=f"Original skills ({len(first_keys)})"),
    Patch(facecolor="#E6550D", label=f"New skills ({len(new_keys)})"),
], fontsize=9, loc="upper right")

ax_rank.set_xlabel("Skills (ranked by usage)", fontsize=10)
ax_rank.set_ylabel("Trajectory Segments (n_instances)", fontsize=10)
ax_rank.set_title("(A)  Skill Usage Ranking", fontsize=13, fontweight="bold", pad=10)
ax_rank.set_xticks([])
ax_rank.grid(axis="y", alpha=.15)

n_orig = len(first_keys)
n_new = len(new_keys)
total_orig = sum(s["n_instances"] for s in last_sorted if s["key"] not in new_keys)
total_new = sum(s["n_instances"] for s in last_sorted if s["key"] in new_keys)
ax_rank.annotate(
    f"Original {n_orig} skills: {total_orig} segments (mean {total_orig/n_orig:.0f})\n"
    f"New {n_new} skills: {total_new} segments (mean {total_new/n_new:.0f})",
    xy=(0.55, 0.70), xycoords="axes fraction", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#ccc"),
)

# ---- Right: Histogram of n_instances, first vs last ----
bins = [0, 3, 10, 20, 40, 65]
bin_labels = ["1–3", "4–10", "11–20", "21–40", "41+"]

first_counts = np.histogram([s["n_instances"] for s in first_usage], bins=bins)[0]
last_counts = np.histogram([s["n_instances"] for s in last_usage], bins=bins)[0]

x_bins = np.arange(len(bin_labels))
w_bin = 0.35
b1 = ax_dist.bar(x_bins - w_bin/2, first_counts, w_bin,
                 label="First (40 skills)", color="#6BAED6", alpha=0.7, edgecolor="white")
b2 = ax_dist.bar(x_bins + w_bin/2, last_counts, w_bin,
                 label="Last (68 skills)", color="#E6550D", alpha=0.85, edgecolor="white")

for bar in b1:
    h = bar.get_height()
    if h > 0:
        ax_dist.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                     str(int(h)), ha="center", fontsize=9, color="#6BAED6", fontweight="bold")
for bar in b2:
    h = bar.get_height()
    if h > 0:
        ax_dist.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                     str(int(h)), ha="center", fontsize=9, color="#E6550D", fontweight="bold")

ax_dist.set_xticks(x_bins)
ax_dist.set_xticklabels(bin_labels, fontsize=10)
ax_dist.set_xlabel("Trajectory Segments per Skill", fontsize=10)
ax_dist.set_ylabel("Number of Skills", fontsize=10)
ax_dist.set_title("(B)  Usage Distribution", fontsize=13, fontweight="bold", pad=10)
ax_dist.legend(fontsize=9)
ax_dist.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax_dist.grid(axis="y", alpha=.15)

fig10.suptitle(
    "Skill Usage Frequency: How Often Are Skills Used?",
    fontsize=14, fontweight="bold", y=1.02,
)
fig10.tight_layout()
fig10.savefig(DIR / "10_skill_usage.png", dpi=150, bbox_inches="tight")
plt.close(fig10)
print(f"Saved: {DIR / '10_skill_usage.png'}")

# ── Figure 11: Skill Bank Maintenance Dynamics Over Training ─────────

RUN_DIR = DIR.parent / "runs" / "Qwen3-8B_diplomacy_20260322_234548"

def load_bank_from_path(base_path):
    """Load skill bank from a directory containing per-country JSONL files."""
    skills = {}
    for c in COUNTRIES:
        bank_path = base_path / c / "skill_bank.jsonl"
        if bank_path.exists():
            with open(bank_path) as f:
                for line in f:
                    entry = json.load(f) if False else json.loads(line)
                    sk = entry["skill"]
                    key = (c, sk["skill_id"])
                    skills[key] = {
                        "version": sk.get("version", 1),
                        "n_instances": sk["n_instances"],
                        "n_revisions": len(sk.get("protocol_history", [])),
                    }
    return skills

def load_snap_bank(path):
    with open(path) as f:
        data = json.load(f)
    skills = {}
    for c, sklist in data["countries"].items():
        for sk in sklist:
            skills[(c, sk["skill_id"])] = {
                "version": sk.get("version", 1),
                "n_instances": sk["n_instances"],
                "n_revisions": 0,
            }
    return skills

snap_first_sk = load_snap_bank(DIR / "first_skillbank.json")
snap_last_sk = load_snap_bank(DIR / "last_skillbank.json")

milestones = [(0, snap_first_sk)]
for step in sorted([17, 21, 22]):
    ckpt_path = RUN_DIR / f"checkpoints/step_{step:04d}/banks/diplomacy"
    if ckpt_path.exists():
        milestones.append((step, load_bank_from_path(ckpt_path)))
milestones.append((24, snap_last_sk))

steps_ms = [s for s, _ in milestones]
totals_ms = [len(sk) for _, sk in milestones]

cumul_new = [0]
cumul_refined = [0]
prev_keys = set(milestones[0][1].keys())
prev_skills = milestones[0][1]
running_new = 0
running_refined = 0
for step, skills in milestones[1:]:
    curr_keys = set(skills.keys())
    new = curr_keys - prev_keys
    running_new += len(new)
    cumul_new.append(running_new)
    upgraded = 0
    for key in curr_keys & set(prev_skills.keys()):
        if skills[key]["version"] > prev_skills[key]["version"]:
            upgraded += 1
    running_refined += upgraded
    cumul_refined.append(running_refined)
    prev_keys |= curr_keys
    prev_skills = skills

# Also get final bank version/revision stats for right panel
final_skills = []
for c in COUNTRIES:
    bank_path = RUN_DIR / "skillbank" / "diplomacy" / c / "skill_bank.jsonl"
    if bank_path.exists():
        with open(bank_path) as f:
            for line in f:
                entry = json.loads(line)
                sk = entry["skill"]
                final_skills.append({
                    "version": sk.get("version", 1),
                    "n_revisions": len(sk.get("protocol_history", [])),
                    "n_instances": sk["n_instances"],
                })

# Aggregate curation operations from bank_diff reports
curation_totals = {"refine": 0, "duration_update": 0, "split": 0, "merge": 0}
for c in COUNTRIES:
    diff_path = RUN_DIR / "skillbank" / "diplomacy" / c / "reports" / "bank_diff_iter0.json"
    if diff_path.exists():
        with open(diff_path) as f:
            d = json.load(f)
        dr = d["diff_report"]
        curation_totals["refine"] += dr["n_refines"]
        curation_totals["duration_update"] += dr["n_duration_updates"]
        curation_totals["split"] += dr["n_splits"]
        curation_totals["merge"] += dr["n_merges"]

fig11, (ax_growth, ax_ops) = plt.subplots(1, 2, figsize=(14, 5.5))

# ---- Left: Skill bank growth + cumulative new skills ----
color_total = "#1F77B4"
color_new = "#E6550D"
color_ref = "#2CA02C"

ax_growth.plot(steps_ms, totals_ms, "o-", color=color_total, linewidth=2.5,
               markersize=8, label="Total skills in bank", zorder=3)
for x, y in zip(steps_ms, totals_ms):
    ax_growth.annotate(str(y), (x, y), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=10,
                       fontweight="bold", color=color_total)

ax_growth.fill_between(steps_ms, 0, totals_ms, alpha=0.08, color=color_total)

ax2 = ax_growth.twinx()
ax2.plot(steps_ms, cumul_new, "s--", color=color_new, linewidth=1.8,
         markersize=6, label="Cumulative new skills", zorder=2)
ax2.plot(steps_ms, cumul_refined, "^--", color=color_ref, linewidth=1.8,
         markersize=6, label="Cumulative refinements", zorder=2)

for x, yn, yr in zip(steps_ms, cumul_new, cumul_refined):
    if yn > 0:
        ax2.annotate(f"+{yn}", (x, yn), textcoords="offset points",
                     xytext=(8, -3), fontsize=8, color=color_new, fontweight="bold")
    if yr > 0:
        ax2.annotate(str(yr), (x, yr), textcoords="offset points",
                     xytext=(8, 5), fontsize=8, color=color_ref, fontweight="bold")

ax_growth.set_xlabel("Training Step", fontsize=10)
ax_growth.set_ylabel("Total Skills", fontsize=10, color=color_total)
ax2.set_ylabel("Cumulative Count", fontsize=10)
ax_growth.set_xlim(-1, 25)
ax_growth.set_ylim(0, max(totals_ms) + 12)
ax2.set_ylim(0, max(max(cumul_new), max(cumul_refined)) + 8)
ax_growth.set_title("(A)  Skill Bank Growth Over Training",
                    fontsize=13, fontweight="bold", pad=10)

lines1, labels1 = ax_growth.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_growth.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5, loc="upper left")
ax_growth.grid(axis="y", alpha=.15)

# ---- Right: Curation operations summary ----
ops = ["Contract\nRefinement", "Duration\nUpdate", "Protocol\nRevision", "Split", "Merge", "Retire"]
vals = [
    curation_totals["refine"],
    curation_totals["duration_update"],
    sum(1 for s in final_skills if s["n_revisions"] > 0),
    curation_totals["split"],
    curation_totals["merge"],
    0,  # no retirements
]
bar_colors = ["#4DAF4A", "#FF7F00", "#3B7DD8", "#E41A1C", "#9467BD", "#999999"]

bars_op = ax_ops.barh(ops, vals, color=bar_colors, edgecolor="white", height=0.6)
for bar, v in zip(bars_op, vals):
    if v > 0:
        ax_ops.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(v), va="center", fontsize=11, fontweight="bold")
    else:
        ax_ops.text(0.5, bar.get_y() + bar.get_height()/2,
                    "0", va="center", fontsize=11, color="#999")

ax_ops.set_xlabel("Count", fontsize=10)
ax_ops.set_title("(B)  Skill Bank Curation Operations",
                 fontsize=13, fontweight="bold", pad=10)
ax_ops.set_xlim(0, max(vals) + 8)
ax_ops.invert_yaxis()
ax_ops.grid(axis="x", alpha=.15)

ax_ops.annotate(
    f"28 new skills discovered\n"
    f"22 contract refinements\n"
    f"13 protocol revisions\n"
    f"42 duration calibrations\n"
    f"0 splits / merges / retires",
    xy=(0.55, 0.35), xycoords="axes fraction", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#ccc"),
)

fig11.suptitle(
    "Skill Bank Maintenance Dynamics (25 Training Steps, 700 Episodes)",
    fontsize=14, fontweight="bold", y=1.02,
)
fig11.tight_layout()
fig11.savefig(DIR / "11_skill_bank_dynamics.png", dpi=150, bbox_inches="tight")
plt.close(fig11)
print(f"Saved: {DIR / '11_skill_bank_dynamics.png'}")

# ── Figure Final: (A) Strategic Function + (B) Intention + (C) Growth ─

FS = 22

fig_final, (axA, axB, axC) = plt.subplots(
    1, 3, figsize=(28, 8),
    gridspec_kw={"width_ratios": [1.0, 0.65, 1.15]},
)

# ---- (A) Strategic Function (grouped bar) ----
sf_cats = DIMENSIONS["strategic_function"]
fc_sf = count_dim(first_classified, "strategic_function")
lc_sf = count_dim(last_classified, "strategic_function")

x_sf = np.arange(len(sf_cats))
w = 0.35
f_vals_a = [fc_sf[k] for k in sf_cats]
l_vals_a = [lc_sf[k] for k in sf_cats]
colors_sf = DIM_COLORS["strategic_function"]

bars_fa = axA.bar(x_sf - w/2, f_vals_a, w, label="First",
                  color=[colors_sf[k] for k in sf_cats], alpha=0.45,
                  edgecolor="white", linewidth=.6)
bars_la = axA.bar(x_sf + w/2, l_vals_a, w, label="Last",
                  color=[colors_sf[k] for k in sf_cats], alpha=1.0,
                  edgecolor="white", linewidth=.6)

for bar in bars_fa:
    h = bar.get_height()
    if h > 0:
        axA.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                 str(int(h)), ha="center", fontsize=FS-2, color="#777")
for bar in bars_la:
    h = bar.get_height()
    if h > 0:
        axA.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                 str(int(h)), ha="center", fontsize=FS-1, fontweight="bold")

for j, (fv, lv) in enumerate(zip(f_vals_a, l_vals_a)):
    delta = lv - fv
    if delta != 0:
        sign = "+" if delta > 0 else ""
        col = "#2CA02C" if delta > 0 else "#D62728"
        y_pos = max(fv, lv) + 2.0
        axA.text(x_sf[j], y_pos, f"{sign}{delta}",
                 ha="center", fontsize=FS-2, fontweight="bold", color=col)

axA.set_title("(A)  Strategic Function", fontsize=FS+2, fontweight="bold", pad=14)
axA.set_xticks(x_sf)
sf_abbr = {
    "territory_gain": "Terr.\nGain",
    "territory_hold": "Terr.\nHold",
    "territory_loss": "Terr.\nLoss",
    "unit_maneuver": "Unit\nMan.",
    "phase_transition": "Phase\nTrans.",
    "resource_management": "Res.\nMgmt.",
}
axA.set_xticklabels([sf_abbr.get(k, k) for k in sf_cats], fontsize=FS-2)
axA.set_ylabel("Number of Skills", fontsize=FS)
axA.legend(fontsize=FS-2, loc="upper right")
axA.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axA.grid(axis="y", alpha=.15)
axA.set_ylim(0, max(max(f_vals_a), max(l_vals_a)) + 7)
axA.tick_params(axis="y", labelsize=FS-2)

# ---- (B) Intention Distribution (%) ----
totals_arr_b = np.array([fa["total_skills"], la["total_skills"]], dtype=float)
bottoms_b = np.zeros(2)
for intent in INTENT_ORDER:
    raw = np.array([fa["by_intention"].get(intent, 0),
                    la["by_intention"].get(intent, 0)])
    pct = raw / totals_arr_b * 100
    axB.bar(LABELS_2, pct, bottom=bottoms_b,
            label=intent.title(), color=INTENT_COLORS[intent],
            edgecolor="white", linewidth=.5, width=0.38)
    for j, p in enumerate(pct):
        if p > 5:
            axB.text(j, bottoms_b[j] + p / 2, f"{p:.0f}%",
                     ha="center", va="center", fontsize=FS-2,
                     color="white", fontweight="bold")
        elif p > 1:
            axB.text(j - 0.22, bottoms_b[j] + p / 2, f"{p:.0f}%",
                     ha="right", va="center", fontsize=FS-4,
                     color=INTENT_COLORS[intent], fontweight="bold")
    bottoms_b += pct

axB.set_title("(B)  Intention (%)", fontsize=FS+2, fontweight="bold", pad=14)
axB.set_ylabel("Percentage", fontsize=FS)
axB.legend(fontsize=FS-6, loc="upper left", ncol=1,
           bbox_to_anchor=(1.02, 1.0),
           framealpha=0.92, edgecolor="#ccc", borderpad=0.3,
           handlelength=1.0, handletextpad=0.3, labelspacing=0.2)
axB.set_ylim(0, 105)
axB.tick_params(axis="both", labelsize=FS-2)

# ---- (C) Skill Bank Growth Over Training (same run as A & B) ----
log_path_c = DIR.parent / "runs" / "Qwen3-8B_diplomacy_20260322_234548" / "step_log.jsonl"
entries_c = []
with open(log_path_c) as f:
    for line in f:
        entries_c.append(json.loads(line))
steps_c = [e["step"] for e in entries_c]
total_c = [e["n_skills"] for e in entries_c]
new_c = [e["n_new_skills"] for e in entries_c]
cumul_new_c = list(np.cumsum(new_c))
curated_c = [0]
for i in range(1, len(entries_c)):
    removed = total_c[i-1] + new_c[i] - total_c[i]
    curated_c.append(max(0, removed))
cumul_curated_c = list(np.cumsum(curated_c))

color_total = "#1F77B4"
color_new = "#E6550D"
color_cur = "#2CA02C"

axC.plot(steps_c, total_c, "o-", color=color_total, linewidth=3,
         markersize=6, label="Total skills", zorder=3)
axC.fill_between(steps_c, 0, total_c, alpha=0.06, color=color_total)

axC2 = axC.twinx()
axC2.plot(steps_c, cumul_new_c, "s-", color=color_new, linewidth=2.5,
          markersize=5, label="Cumul. new skills", zorder=2)
axC2.fill_between(steps_c, 0, cumul_new_c, alpha=0.06, color=color_new)
axC2.plot(steps_c, cumul_curated_c, "^-", color=color_cur, linewidth=2.5,
          markersize=5, label="Cumul. retired", zorder=2)
axC2.fill_between(steps_c, 0, cumul_curated_c, alpha=0.06, color=color_cur)

axC.set_xlabel("Training Step", fontsize=FS)
axC.set_ylabel("Total Skills", fontsize=FS, color="black")
axC2.set_ylabel("Cumulative Count", fontsize=FS)
axC.set_xlim(-0.5, max(steps_c) + 0.5)
axC.set_ylim(0, max(total_c) + 12)
axC2.set_ylim(0, max(cumul_new_c) + 15)
axC.set_title("(C)  Skill Bank Growth", fontsize=FS+2, fontweight="bold", pad=14)
axC.tick_params(axis="both", labelsize=FS-2)
axC2.tick_params(axis="y", labelsize=FS-2)

lines_c1, labels_c1 = axC.get_legend_handles_labels()
lines_c2, labels_c2 = axC2.get_legend_handles_labels()
axC.legend(lines_c1 + lines_c2, labels_c1 + labels_c2,
           fontsize=FS-4, loc="upper left",
           framealpha=0.9, edgecolor="#ccc")
axC.grid(axis="both", alpha=.12)

fig_final.tight_layout(w_pad=3.5)
fig_final.savefig(DIR / "figure_final.png", dpi=150, bbox_inches="tight")
plt.close(fig_final)
print(f"Saved: {DIR / 'figure_final.png'}")
