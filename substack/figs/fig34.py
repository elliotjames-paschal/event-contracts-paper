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

# ── Fig 3: joint OLS — change in dispute probability per severity point ────
from collections import defaultdict
from dispute_regression import AXES

pm = market_units("polymarket")
pm = [u for u in pm if not u["disputed"] or u["classification"] == "specification_issue"]
y = np.array([u["disputed"] for u in pm], float)
M = np.array([[u[a] for a in AXES] for u in pm], float)          # raw 0-3 scores
groups = [u["group"] for u in pm]

# one OLS regression (linear probability model) with all ten dimensions at once
X = np.column_stack([np.ones(len(pm)), M])
XtXi = np.linalg.inv(X.T @ X)
b = XtXi @ (X.T @ y)
resid = y - X @ b
gi = defaultdict(list)
for i, g_ in enumerate(groups):
    gi[g_].append(i)
meat = np.zeros((X.shape[1],) * 2)                                # family-clustered
for idx in gi.values():
    ug = (X[idx] * resid[idx, None]).sum(0)
    meat += np.outer(ug, ug)
se = np.sqrt(np.diag(XtXi @ meat @ XtXi))

LABELS = {
    "predicate_ambiguity":        "Core question vaguely defined",
    "entity_ambiguity":           "Unclear which entity counts",
    "source_specification":       "Settlement source not identified",
    "outcome_concentration":      "Settlement controlled by one party",
    "source_quality":             "Weak or unofficial source",
    "edge_case_coverage":         "Edge cases unaddressed",
    "governing_body_engagement":  "No governing body engaged",
    "headline_rules_alignment":   "Headline differs from fine print",
    "temporal_precision":         "Time window imprecise",
    "manipulation_susceptibility":"One actor could sway the outcome",
}
rows = []
for j, a in enumerate(AXES):
    est = 100 * b[j + 1]                                          # percentage points
    lo, hi = 100 * (b[j + 1] - 1.96 * se[j + 1]), 100 * (b[j + 1] + 1.96 * se[j + 1])
    rows.append((a, est, lo, hi))
    print(f"{a:30s} {est:+6.1f} pp  [{lo:+6.1f}, {hi:+6.1f}]")
rows.sort(key=lambda r: -r[1])

with plt.rc_context({"font.family": "serif", "font.size": 11}):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    yy = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#888", lw=0.9, ls=(0, (4, 3)), zorder=1)
    for yi, (a, est, lo, hi) in zip(yy, rows):
        sig_up = lo > 0
        sig_dn = hi < 0
        c = RED if sig_up else (BLUE if sig_dn else GRAY)
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.8, zorder=2)
        ax.scatter([est], [yi], s=52, color=c, zorder=3)
        ax.text(hi + 0.7, yi, f"{est:+.1f}", va="center", fontsize=9.5,
                color=c, fontweight="bold" if (sig_up or sig_dn) else "normal")
    ax.set_yticks(yy)
    ax.set_yticklabels([LABELS[r[0]] for r in rows], fontsize=11)
    ax.set_xlabel("Change in probability of a dispute per one severity point\n(percentage points, 95% CI)", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Dispute probability and each contract dimension, estimated jointly (OLS, Polymarket)",
                 fontsize=11.5, pad=10)
    fig.text(0.5, 0.15,
             "Red: raises dispute probability (CI excludes 0) · blue: lowers · gray: indistinguishable from 0. "
             "Dispute-enriched sample (24% base rate). \u201cSway\u201d and \u201ccontrolled by one party\u201d are nearly collinear (r=0.98); read as a pair.",
             ha="center", fontsize=7.6, color="#666666")
    from fs_style import add_footer
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    add_footer(fig, credit="Data: Polymarket (Hall & Paschal)")
    fig.savefig(OUT / "fig3_flaws.png", dpi=150, bbox_inches="tight", facecolor="white")

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
    ax.set_title("Confirmed-dispute risk by grade, relative to grade A (held-out markets)", fontsize=12.5, pad=10)
    from fs_style import add_footer
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    add_footer(fig)
    fig.savefig(OUT / "fig4_grades.png", dpi=150, bbox_inches="tight", facecolor="white")
print("test rates %:", [f"{100*r:.1f}" for r in te_r], "| letters:", letters, "| relative:", [f"{r:.1f}" for r in rel])
