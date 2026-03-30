"""Visualize Diplomacy skill bank evolution across training runs.

Shows how the number and category distribution of skills change over
successive training runs for each of the 7 Diplomacy powers.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RUNS_DIR = Path(__file__).parent / "runs"
OUTPUT_DIR = Path(__file__).parent / "training_curves"
OUTPUT_DIR.mkdir(exist_ok=True)

RUNS = [
    ("Run 1\n(Mar 22)", "Qwen3-8B_diplomacy_20260322_234548"),
    ("Run 2\n(Mar 27a)", "Qwen3-8B_20260327_003449"),
    ("Run 3\n(Mar 27b)", "Qwen3-8B_diplomacy_20260327_042539"),
    ("Run 4\n(Mar 27c)", "Qwen3-8B_20260327_062035"),
]

COUNTRIES = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

PHASE_COLORS = {
    "opening": "#4C72B0",
    "orders": "#55A868",
    "late_orders": "#C44E52",
    "adjustment": "#8172B2",
    "retreat": "#CCB974",
}

INTENTION_COLORS = {
    "EXPLORE": "#4C72B0",
    "SETUP": "#55A868",
    "DEFEND": "#C44E52",
    "ATTACK": "#DD8452",
    "BUILD": "#8172B2",
    "NAVIGATE": "#64B5CD",
}


def load_skills(run_dir: str) -> dict[str, list[str]]:
    """Return {country: [skill_id, ...]} for a run."""
    result = {}
    for country in COUNTRIES:
        path = RUNS_DIR / run_dir / "skillbank" / "diplomacy" / country / "skill_bank.jsonl"
        skill_ids = []
        if path.exists():
            for line in path.read_text().strip().split("\n"):
                if line.strip():
                    obj = json.loads(line)
                    skill_ids.append(obj["skill"]["skill_id"])
        result[country] = skill_ids
    return result


def extract_phase(skill_id: str) -> str:
    return skill_id.rsplit(":", 1)[0]


def extract_intention(skill_id: str) -> str:
    return skill_id.rsplit(":", 1)[1]


all_data = []
for label, run_dir in RUNS:
    all_data.append((label, load_skills(run_dir)))

# ---------- Figure 1: Total skills per run (stacked by country) ----------
fig1, ax1 = plt.subplots(figsize=(10, 5))
country_colors = plt.cm.Set2(np.linspace(0, 1, len(COUNTRIES)))

bottoms = np.zeros(len(RUNS))
for i, country in enumerate(COUNTRIES):
    counts = [len(data[country]) for _, data in all_data]
    ax1.bar(
        [label for label, _ in all_data],
        counts,
        bottom=bottoms,
        label=country.title(),
        color=country_colors[i],
        edgecolor="white",
        linewidth=0.5,
    )
    for j, c in enumerate(counts):
        if c > 0:
            ax1.text(j, bottoms[j] + c / 2, str(c), ha="center", va="center", fontsize=7, fontweight="bold")
    bottoms += counts

for j in range(len(RUNS)):
    ax1.text(j, bottoms[j] + 1, str(int(bottoms[j])), ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_ylabel("Number of Skills")
ax1.set_title("Total Diplomacy Skills per Run (by Country)", fontsize=13, fontweight="bold")
ax1.legend(loc="upper right", ncol=4, fontsize=8)
ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.set_ylim(0, max(bottoms) + 8)
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "diplomacy_skills_total.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTPUT_DIR / 'diplomacy_skills_total.png'}")

# ---------- Figure 2: Phase distribution across runs ----------
all_phases = sorted(PHASE_COLORS.keys())

fig2, ax2 = plt.subplots(figsize=(10, 5))
bottoms = np.zeros(len(RUNS))
for phase in all_phases:
    counts = []
    for _, data in all_data:
        c = sum(1 for ids in data.values() for sid in ids if extract_phase(sid) == phase)
        counts.append(c)
    counts = np.array(counts)
    ax2.bar(
        [label for label, _ in all_data],
        counts,
        bottom=bottoms,
        label=phase.replace("_", " ").title(),
        color=PHASE_COLORS[phase],
        edgecolor="white",
        linewidth=0.5,
    )
    for j, c in enumerate(counts):
        if c > 0:
            ax2.text(j, bottoms[j] + c / 2, str(c), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    bottoms += counts

ax2.set_ylabel("Number of Skills")
ax2.set_title("Skill Distribution by Game Phase", fontsize=13, fontweight="bold")
ax2.legend(loc="upper right", fontsize=9)
ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax2.set_ylim(0, max(bottoms) + 5)
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "diplomacy_skills_by_phase.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTPUT_DIR / 'diplomacy_skills_by_phase.png'}")

# ---------- Figure 3: Intention distribution across runs ----------
all_intentions = sorted(INTENTION_COLORS.keys())

fig3, ax3 = plt.subplots(figsize=(10, 5))
bottoms = np.zeros(len(RUNS))
for intention in all_intentions:
    counts = []
    for _, data in all_data:
        c = sum(1 for ids in data.values() for sid in ids if extract_intention(sid) == intention)
        counts.append(c)
    counts = np.array(counts)
    ax3.bar(
        [label for label, _ in all_data],
        counts,
        bottom=bottoms,
        label=intention.title(),
        color=INTENTION_COLORS[intention],
        edgecolor="white",
        linewidth=0.5,
    )
    for j, c in enumerate(counts):
        if c > 0:
            ax3.text(j, bottoms[j] + c / 2, str(c), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    bottoms += counts

ax3.set_ylabel("Number of Skills")
ax3.set_title("Skill Distribution by Intention Type", fontsize=13, fontweight="bold")
ax3.legend(loc="upper right", fontsize=9)
ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax3.set_ylim(0, max(bottoms) + 5)
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "diplomacy_skills_by_intention.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTPUT_DIR / 'diplomacy_skills_by_intention.png'}")

# ---------- Figure 4: Heatmap – phase × intention per run ----------
fig4, axes4 = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
for idx, ((label, data), ax) in enumerate(zip(all_data, axes4)):
    matrix = np.zeros((len(all_phases), len(all_intentions)))
    for country_skills in data.values():
        for sid in country_skills:
            p = extract_phase(sid)
            intent = extract_intention(sid)
            if p in all_phases and intent in all_intentions:
                pi = all_phases.index(p)
                ii = all_intentions.index(intent)
                matrix[pi, ii] += 1

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(matrix.max(), 1))
    ax.set_xticks(range(len(all_intentions)))
    ax.set_xticklabels([i.title() for i in all_intentions], rotation=45, ha="right", fontsize=8)
    if idx == 0:
        ax.set_yticks(range(len(all_phases)))
        ax.set_yticklabels([p.replace("_", " ").title() for p in all_phases], fontsize=9)
    else:
        ax.set_yticks(range(len(all_phases)))
    ax.set_title(label.replace("\n", " "), fontsize=10, fontweight="bold")

    for pi in range(len(all_phases)):
        for ii in range(len(all_intentions)):
            v = int(matrix[pi, ii])
            if v > 0:
                ax.text(ii, pi, str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v > matrix.max() * 0.6 else "black")

fig4.suptitle("Phase × Intention Skill Counts Across Runs", fontsize=14, fontweight="bold", y=1.02)
fig4.colorbar(im, ax=axes4, shrink=0.8, label="Skill Count (across all 7 powers)")
fig4.tight_layout()
fig4.savefig(OUTPUT_DIR / "diplomacy_skills_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTPUT_DIR / 'diplomacy_skills_heatmap.png'}")

# ---------- Figure 5: Per-country skill count evolution ----------
fig5, ax5 = plt.subplots(figsize=(10, 5))
for i, country in enumerate(COUNTRIES):
    counts = [len(data[country]) for _, data in all_data]
    ax5.plot(
        range(len(RUNS)),
        counts,
        marker="o",
        linewidth=2,
        label=country.title(),
        color=country_colors[i],
    )
    for j, c in enumerate(counts):
        ax5.annotate(str(c), (j, c), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7, fontweight="bold", color=country_colors[i])

ax5.set_xticks(range(len(RUNS)))
ax5.set_xticklabels([label for label, _ in all_data])
ax5.set_ylabel("Number of Skills")
ax5.set_title("Per-Country Skill Count Across Runs", fontsize=13, fontweight="bold")
ax5.legend(loc="upper right", ncol=4, fontsize=8)
ax5.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax5.grid(axis="y", alpha=0.3)
fig5.tight_layout()
fig5.savefig(OUTPUT_DIR / "diplomacy_skills_per_country.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTPUT_DIR / 'diplomacy_skills_per_country.png'}")

# ---------- Figure 6: Combined overview (2x2) ----------
fig6, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Total skills
bottoms = np.zeros(len(RUNS))
for i, country in enumerate(COUNTRIES):
    counts = [len(data[country]) for _, data in all_data]
    a1.bar([label for label, _ in all_data], counts, bottom=bottoms,
           label=country.title(), color=country_colors[i], edgecolor="white", linewidth=0.5)
    bottoms += counts
for j in range(len(RUNS)):
    a1.text(j, bottoms[j] + 0.5, str(int(bottoms[j])), ha="center", va="bottom", fontsize=9, fontweight="bold")
a1.set_ylabel("Skills")
a1.set_title("(A) Total Skills by Country", fontweight="bold")
a1.legend(loc="upper right", ncol=4, fontsize=6)
a1.set_ylim(0, max(bottoms) + 8)

# Panel B: Phase distribution (normalized %)
bottoms = np.zeros(len(RUNS))
totals = np.array([sum(len(ids) for ids in data.values()) for _, data in all_data], dtype=float)
for phase in all_phases:
    counts = []
    for _, data in all_data:
        c = sum(1 for ids in data.values() for sid in ids if extract_phase(sid) == phase)
        counts.append(c)
    pcts = np.array(counts) / totals * 100
    a2.bar([label for label, _ in all_data], pcts, bottom=bottoms,
           label=phase.replace("_", " ").title(), color=PHASE_COLORS[phase],
           edgecolor="white", linewidth=0.5)
    for j, p in enumerate(pcts):
        if p > 3:
            a2.text(j, bottoms[j] + p / 2, f"{p:.0f}%", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
    bottoms += pcts
a2.set_ylabel("Percentage")
a2.set_title("(B) Phase Distribution (%)", fontweight="bold")
a2.legend(loc="upper right", fontsize=7)
a2.set_ylim(0, 105)

# Panel C: Intention distribution (normalized %)
bottoms = np.zeros(len(RUNS))
for intention in all_intentions:
    counts = []
    for _, data in all_data:
        c = sum(1 for ids in data.values() for sid in ids if extract_intention(sid) == intention)
        counts.append(c)
    pcts = np.array(counts) / totals * 100
    a3.bar([label for label, _ in all_data], pcts, bottom=bottoms,
           label=intention.title(), color=INTENTION_COLORS[intention],
           edgecolor="white", linewidth=0.5)
    for j, p in enumerate(pcts):
        if p > 3:
            a3.text(j, bottoms[j] + p / 2, f"{p:.0f}%", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
    bottoms += pcts
a3.set_ylabel("Percentage")
a3.set_title("(C) Intention Distribution (%)", fontweight="bold")
a3.legend(loc="upper right", fontsize=7)
a3.set_ylim(0, 105)

# Panel D: Line chart per-country
for i, country in enumerate(COUNTRIES):
    counts = [len(data[country]) for _, data in all_data]
    a4.plot(range(len(RUNS)), counts, marker="o", linewidth=2,
            label=country.title(), color=country_colors[i])
a4.set_xticks(range(len(RUNS)))
a4.set_xticklabels([label for label, _ in all_data])
a4.set_ylabel("Skills")
a4.set_title("(D) Per-Country Trajectories", fontweight="bold")
a4.legend(loc="upper right", ncol=4, fontsize=6)
a4.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
a4.grid(axis="y", alpha=0.3)

fig6.suptitle("Diplomacy Skill Bank Evolution Across Training Runs",
              fontsize=15, fontweight="bold", y=1.01)
fig6.tight_layout()
fig6.savefig(OUTPUT_DIR / "diplomacy_skills_overview.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTPUT_DIR / 'diplomacy_skills_overview.png'}")

# ---------- Print summary table ----------
print("\n" + "=" * 70)
print("DIPLOMACY SKILL BANK SUMMARY")
print("=" * 70)
for label, data in all_data:
    total = sum(len(ids) for ids in data.values())
    phases = Counter(extract_phase(sid) for ids in data.values() for sid in ids)
    intentions = Counter(extract_intention(sid) for ids in data.values() for sid in ids)
    print(f"\n{label.replace(chr(10), ' ')}  —  {total} total skills")
    print(f"  Phases:     {dict(phases)}")
    print(f"  Intentions: {dict(intentions)}")
    for country in COUNTRIES:
        print(f"    {country:8s}: {len(data[country]):2d} skills → {data[country]}")

plt.close("all")
print("\nDone. All plots saved to", OUTPUT_DIR)
