"""Substack Figures 3 (flaw dumbbells) and 4 (grade ladder) prototypes."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/andrewhall/event_contracts/event-contracts-paper")
for p in (REPO/"src", REPO/"scripts", REPO/"paper"/"scripts"):
    sys.path.insert(0, str(p))
from dispute_regression import market_units
from grade_binning import fit_grades, _rates

OUT = Path("/Users/andrewhall/event_contracts/substack/figs")
BLUE, RED, GRAY = "#33618f", "#b3392f", "#8a8a8a"

# ── Fig 3: dispute rate with vs without each flaw ──────────────────────────
pm = market_units("polymarket")
pm = [u for u in pm if not u["disputed"] or u["classification"] == "specification_issue"]
y = np.array([u["disputed"] for u in pm])

ROWS = [  # (axis, plain-English label, predictive?)
    ("predicate_ambiguity",      "Key term left undefined", True),
    ("headline_rules_alignment", "Headline doesn't match the fine print", True),
    ("source_specification",     "No named settlement source", True),
    ("source_quality",           "Weak or unofficial source", True),
    ("edge_case_coverage",       "Edge cases unaddressed", False),
    ("manipulation_susceptibility", "One actor could sway the outcome", False),
]
with plt.rc_context({"font.family": "serif", "font.size": 11}):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    yy = np.arange(len(ROWS))[::-1]
    for yi, (a, lab, pred) in zip(yy, ROWS):
        v = np.array([u[a] for u in pm])
        lo, hi = 100*y[v <= 1].mean(), 100*y[v >= 2].mean()
        c = BLUE if pred else GRAY
        ax.plot([lo, hi], [yi, yi], color=c, lw=2, zorder=2)
        ax.scatter([lo], [yi], s=55, facecolor="white", edgecolor=c, lw=1.6, zorder=3)
        ax.scatter([hi], [yi], s=70, color=(RED if pred and hi > lo else c), zorder=3)
        ax.annotate(f"{lo:.0f}%", (lo, yi), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color="#666")
        ax.annotate(f"{hi:.0f}%", (hi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10,
                    color=(RED if pred and hi > lo else "#666"), fontweight="bold")
    ax.set_yticks(yy); ax.set_yticklabels([r[1] for r in ROWS], fontsize=11)
    ax.set_ylim(-0.85, len(ROWS) - 0.4)
    ax.set_xlabel("Share of markets that ended in a dispute (%)", fontsize=10)
    ax.set_xlim(0, 72)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="white", markerfacecolor="white",
               markeredgecolor=BLUE, markersize=8, label="without the flaw"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=RED, markersize=8,
               label="with the flaw (red = predicts disputes)")],
        frameon=False, fontsize=9.5, loc="lower center", ncol=2,
        bbox_to_anchor=(0.56, -0.04))
    ax.set_title("Which contract flaws actually forecast trouble", fontsize=13, pad=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "fig3_flaws.png", dpi=150, bbox_inches="tight")

# ── Fig 4: the letter grades, relative blow-up risk (held-out) ─────────────
f = fit_grades(target="confirmed")
g, G, tgt = f["g"], f["G"], f["tgt"]
te_r = _rates(g, tgt, f["test"], G)
letters = f["letters"]; nig = f["not_ig_from"]
rel = [r / te_r[0] for r in te_r]
with plt.rc_context({"font.family": "serif", "font.size": 11}):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(G)
    colors = ["#c9d6e3" if k < nig else "#b3392f" for k in range(G)]
    ax.bar(x, rel, color=colors, width=0.66, zorder=2)
    for xi, v in zip(x, rel):
        ax.text(xi, v + 0.7, f"{v:.0f}×", ha="center", fontsize=12, fontweight="bold",
                color="#333")
    ax.set_xticks(x); ax.set_xticklabels(letters, fontsize=13)
    ax.set_ylabel("Blow-up risk, relative to grade A\n(held-out markets)", fontsize=10)
    ax.axvline(nig - 0.5, color="#555", lw=0.8, ls=(0, (4, 3)))
    ax.text(nig - 0.42, max(rel)*0.97, "below\ninvestment grade", fontsize=9,
            color="#b3392f", va="top")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("A market's letter grade tells you its blow-up risk", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_grades.png", dpi=150)
print("test rates %:", [f"{100*r:.1f}" for r in te_r], "| letters:", letters, "| relative:", [f"{r:.1f}" for r in rel])
