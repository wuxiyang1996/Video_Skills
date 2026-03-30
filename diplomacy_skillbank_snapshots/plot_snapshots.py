"""Compare first vs last Diplomacy skill bank snapshots."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DIR = Path(__file__).resolve().parent

with open(DIR / "first_skillbank.json") as f:
    first = json.load(f)
with open(DIR / "last_skillbank.json") as f:
    last = json.load(f)

COUNTRIES = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
LABELS = ["First\n(Step 0)", "Last\n(Step 24)"]

PHASE_ORDER = ["opening", "orders", "late_orders", "adjustment", "retreat"]
PHASE_COLORS = {
    "opening": "#3B7DD8", "orders": "#4DAF4A", "late_orders": "#E24A33",
    "adjustment": "#8B6DB0", "retreat": "#C9A832",
}
INTENT_ORDER = ["EXPLORE", "SETUP", "DEFEND", "ATTACK", "BUILD"]
INTENT_COLORS = {
    "EXPLORE": "#3B7DD8", "SETUP": "#4DAF4A", "DEFEND": "#E24A33",
    "ATTACK": "#F28C28", "BUILD": "#8B6DB0",
}
COUNTRY_CMAP = plt.cm.Set2(np.linspace(0, 1, len(COUNTRIES)))

fa, la = first["aggregate"], last["aggregate"]

# ── Figure 1: Total skills stacked by country ───────────────────────
fig1, ax = plt.subplots(figsize=(6, 5))
bottoms = np.zeros(2)
for i, c in enumerate(COUNTRIES):
    vals = np.array([fa["per_country"][c], la["per_country"][c]])
    ax.bar(LABELS, vals, bottom=bottoms, label=c.title(),
           color=COUNTRY_CMAP[i], edgecolor="white", linewidth=.5, width=0.55)
    for j, v in enumerate(vals):
        if v > 0:
            ax.text(j, bottoms[j] + v / 2, str(v),
                    ha="center", va="center", fontsize=8, fontweight="bold")
    bottoms += vals
for j in range(2):
    ax.text(j, bottoms[j] + 1, str(int(bottoms[j])),
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Skills")
ax.set_title("Total Skills: First vs Last  (by Country)", fontsize=13, fontweight="bold")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.set_ylim(0, max(bottoms) + 8)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
fig1.tight_layout()
fig1.savefig(DIR / "01_total_skills.png", dpi=150, bbox_inches="tight")
plt.close(fig1)

# ── Figure 2: Phase distribution (count + %) ────────────────────────
all_phases = PHASE_ORDER
fig2, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(11, 5))

for ax_t, use_pct in [(ax_c, False), (ax_p, True)]:
    totals = np.array([fa["total_skills"], la["total_skills"]], dtype=float)
    bottoms = np.zeros(2)
    for phase in all_phases:
        raw = np.array([fa["by_phase"].get(phase, 0), la["by_phase"].get(phase, 0)])
        vals = raw / totals * 100 if use_pct else raw
        ax_t.bar(LABELS, vals, bottom=bottoms,
                 label=phase.replace("_", " ").title(),
                 color=PHASE_COLORS[phase], edgecolor="white", linewidth=.5, width=0.55)
        for j, v in enumerate(vals):
            if v > (3 if use_pct else 0.5):
                txt = f"{v:.0f}%" if use_pct else str(int(v))
                ax_t.text(j, bottoms[j] + v / 2, txt, ha="center", va="center",
                          fontsize=8, color="white", fontweight="bold")
        bottoms += vals
    ax_t.set_ylabel("%" if use_pct else "Count")
    ax_t.set_title(f"Phase Distribution ({'%' if use_pct else 'Count'})", fontweight="bold")
    ax_t.legend(fontsize=8)
    if use_pct:
        ax_t.set_ylim(0, 110)

fig2.suptitle("Skill Distribution by Game Phase", fontsize=14, fontweight="bold", y=1.02)
fig2.tight_layout()
fig2.savefig(DIR / "02_phase_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# ── Figure 3: Intention distribution (count + %) ────────────────────
fig3, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(11, 5))

for ax_t, use_pct in [(ax_c, False), (ax_p, True)]:
    totals = np.array([fa["total_skills"], la["total_skills"]], dtype=float)
    bottoms = np.zeros(2)
    for intent in INTENT_ORDER:
        raw = np.array([fa["by_intention"].get(intent, 0), la["by_intention"].get(intent, 0)])
        vals = raw / totals * 100 if use_pct else raw
        ax_t.bar(LABELS, vals, bottom=bottoms,
                 label=intent.title(),
                 color=INTENT_COLORS[intent], edgecolor="white", linewidth=.5, width=0.55)
        for j, v in enumerate(vals):
            if v > (3 if use_pct else 0.5):
                txt = f"{v:.0f}%" if use_pct else str(int(v))
                ax_t.text(j, bottoms[j] + v / 2, txt, ha="center", va="center",
                          fontsize=8, color="white", fontweight="bold")
        bottoms += vals
    ax_t.set_ylabel("%" if use_pct else "Count")
    ax_t.set_title(f"Intention Distribution ({'%' if use_pct else 'Count'})", fontweight="bold")
    ax_t.legend(fontsize=8)
    if use_pct:
        ax_t.set_ylim(0, 110)

fig3.suptitle("Skill Distribution by Intention Type", fontsize=14, fontweight="bold", y=1.02)
fig3.tight_layout()
fig3.savefig(DIR / "03_intention_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig3)

# ── Figure 4: Heatmap side-by-side ──────────────────────────────────
fig4, (ax_f, ax_l) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for ax_t, data, title in [(ax_f, first, "First (Step 0)"), (ax_l, last, "Last (Step 24)")]:
    m = np.zeros((len(all_phases), len(INTENT_ORDER)))
    for country_skills in data["countries"].values():
        for sk in country_skills:
            p, i = sk["phase"], sk["intention"]
            if p in all_phases and i in INTENT_ORDER:
                m[all_phases.index(p), INTENT_ORDER.index(i)] += 1
    vmax = max(m.max(), 1)
    im = ax_t.imshow(m, cmap="YlOrRd", aspect="auto", vmin=0, vmax=7)
    ax_t.set_xticks(range(len(INTENT_ORDER)))
    ax_t.set_xticklabels([i.title() for i in INTENT_ORDER], rotation=35, ha="right", fontsize=9)
    ax_t.set_yticks(range(len(all_phases)))
    ax_t.set_yticklabels([p.replace("_", " ").title() for p in all_phases], fontsize=9)
    ax_t.set_title(title, fontsize=11, fontweight="bold")
    for pi in range(len(all_phases)):
        for ii in range(len(INTENT_ORDER)):
            v = int(m[pi, ii])
            if v > 0:
                ax_t.text(ii, pi, str(v), ha="center", va="center", fontsize=10,
                          fontweight="bold", color="white" if v > 4 else "black")

fig4.colorbar(im, ax=[ax_f, ax_l], shrink=0.8, pad=0.03, label="Count (all 7 powers)")
fig4.suptitle("Phase × Intention Skill Counts", fontsize=14, fontweight="bold")
fig4.tight_layout(rect=[0, 0, 0.92, 0.93])
fig4.savefig(DIR / "04_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig4)

# ── Figure 5: Per-country grouped bar ────────────────────────────────
fig5, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(COUNTRIES))
w = 0.35
first_vals = [fa["per_country"][c] for c in COUNTRIES]
last_vals = [la["per_country"][c] for c in COUNTRIES]

bars1 = ax.bar(x - w/2, first_vals, w, label="First (Step 0)", color="#6BAED6", edgecolor="white")
bars2 = ax.bar(x + w/2, last_vals, w, label="Last (Step 24)", color="#E6550D", edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold", color="#6BAED6")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold", color="#E6550D")

ax.set_xticks(x)
ax.set_xticklabels([c.title() for c in COUNTRIES])
ax.set_ylabel("Number of Skills")
ax.set_title("Per-Country Skill Count: First vs Last", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", alpha=.2)
fig5.tight_layout()
fig5.savefig(DIR / "05_per_country.png", dpi=150, bbox_inches="tight")
plt.close(fig5)

# ── Figure 6: Combined overview ──────────────────────────────────────
fig6, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14, 10))

# A – total by country
bottoms = np.zeros(2)
for i, c in enumerate(COUNTRIES):
    vals = np.array([fa["per_country"][c], la["per_country"][c]])
    a1.bar(LABELS, vals, bottom=bottoms, label=c.title(),
           color=COUNTRY_CMAP[i], edgecolor="white", linewidth=.5, width=0.5)
    bottoms += vals
for j in range(2):
    a1.text(j, bottoms[j] + .5, str(int(bottoms[j])),
            ha="center", va="bottom", fontsize=11, fontweight="bold")
a1.set_ylabel("Skills")
a1.set_title("(A) Total Skills by Country", fontweight="bold")
a1.legend(ncol=4, fontsize=7, loc="upper left")
a1.set_ylim(0, max(bottoms) + 8)

# B – phase %
totals_arr = np.array([fa["total_skills"], la["total_skills"]], dtype=float)
bottoms = np.zeros(2)
for phase in all_phases:
    raw = np.array([fa["by_phase"].get(phase, 0), la["by_phase"].get(phase, 0)])
    pct = raw / totals_arr * 100
    a2.bar(LABELS, pct, bottom=bottoms, label=phase.replace("_", " ").title(),
           color=PHASE_COLORS[phase], edgecolor="white", linewidth=.5, width=0.5)
    for j, p in enumerate(pct):
        if p > 4:
            a2.text(j, bottoms[j] + p / 2, f"{p:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    bottoms += pct
a2.set_ylabel("Percentage")
a2.set_title("(B) Phase Distribution (%)", fontweight="bold")
a2.legend(fontsize=8)
a2.set_ylim(0, 110)

# C – intention %
bottoms = np.zeros(2)
for intent in INTENT_ORDER:
    raw = np.array([fa["by_intention"].get(intent, 0), la["by_intention"].get(intent, 0)])
    pct = raw / totals_arr * 100
    a3.bar(LABELS, pct, bottom=bottoms, label=intent.title(),
           color=INTENT_COLORS[intent], edgecolor="white", linewidth=.5, width=0.5)
    for j, p in enumerate(pct):
        if p > 4:
            a3.text(j, bottoms[j] + p / 2, f"{p:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    bottoms += pct
a3.set_ylabel("Percentage")
a3.set_title("(C) Intention Distribution (%)", fontweight="bold")
a3.legend(fontsize=8)
a3.set_ylim(0, 110)

# D – per-country grouped bar
x = np.arange(len(COUNTRIES))
w = 0.35
a4.bar(x - w/2, first_vals, w, label="First (Step 0)", color="#6BAED6", edgecolor="white")
a4.bar(x + w/2, last_vals, w, label="Last (Step 24)", color="#E6550D", edgecolor="white")
for j in range(len(COUNTRIES)):
    delta = last_vals[j] - first_vals[j]
    if delta > 0:
        a4.annotate(f"+{delta}", (x[j] + w/2, last_vals[j]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, fontweight="bold", color="#E6550D")
a4.set_xticks(x)
a4.set_xticklabels([c.title() for c in COUNTRIES], fontsize=9)
a4.set_ylabel("Skills")
a4.set_title("(D) Per-Country Change", fontweight="bold")
a4.legend(fontsize=9)
a4.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
a4.grid(axis="y", alpha=.2)

fig6.suptitle(
    f"Diplomacy Skill Bank: First (Step 0, {fa['total_skills']} skills)"
    f"  →  Last (Step 24, {la['total_skills']} skills)",
    fontsize=14, fontweight="bold", y=1.01,
)
fig6.tight_layout()
fig6.savefig(DIR / "06_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig6)

print("Plots saved to", DIR)
for p in sorted(DIR.glob("*.png")):
    print(f"  {p.name}")
