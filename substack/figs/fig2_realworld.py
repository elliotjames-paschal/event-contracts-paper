"""Fig 2 alternative: unweighted binscatter on the representative random sample.

Train per-platform on the case-control folds (clean + oversampled disputes),
score every market out-of-fold (grouped CV), then keep ONLY the random
representative sample and plot raw, unweighted dispute rates by predicted-risk
bin. Levels here are real-world rates.
"""
import sys, random, json
from pathlib import Path
import numpy as np
REPO = Path("/Users/andrewhall/event_contracts/event-contracts-paper")
for p in (REPO/"src", REPO/"scripts", REPO/"paper"/"scripts"):
    sys.path.insert(0, str(p))
from dispute_regression import HGBModel, LEARN_COLS, AXES, auc
from _dispute_data import graded_rows, group_of, grouped_fold_assign

def units_all(platform):
    out = []
    for r in graded_rows():
        if r["platform"] != platform: continue
        ax = {a: float(r["axes"][a]) for a in AXES}
        out.append(dict(
            **{c: ax[c] for c in AXES},
            structural=(ax["manipulation_susceptibility"]+ax["outcome_concentration"])/2,
            disputed=int(r["disputed"]), rand=int(r.get("in_historical") == 1),
            group=group_of(r)))
    return out

def oof(units, k=5, seed=0):
    M = np.array([[u[c] for c in LEARN_COLS] for u in units], float)
    y = np.array([u["disputed"] for u in units])
    fold = grouped_fold_assign([u["group"] for u in units], y, k, random.Random(seed))
    s = np.full(len(units), np.nan)
    for f in range(k):
        tr = fold != f
        mu, sd = M[tr].mean(0), M[tr].std(0); sd[sd==0] = 1
        s[~tr] = HGBModel().fit((M[tr]-mu)/sd, y[tr]).score((M[~tr]-mu)/sd)
    return s, y

rows_all, scores, ys, rands, plats = [], [], [], [], []
for plat in ("polymarket", "kalshi"):
    un = units_all(plat)
    s, y = oof(un)
    r = np.array([u["rand"] for u in un], bool)
    # percentile within the platform's random sample
    sr = s[r]
    pct = np.searchsorted(np.sort(sr), s, side="right") / len(sr)
    scores.append(pct[r]); ys.append(y[r]); plats += [plat]*int(r.sum())
    print(f"{plat}: random n={int(r.sum())}, disputed={int(y[r].sum())} "
          f"({100*y[r].mean():.2f}%), OOF AUC on random-only = {auc(y[r], s[r]):.3f}")

pct = np.concatenate(scores); y = np.concatenate(ys)
print(f"combined random sample: n={len(y)}, disputed={int(y.sum())} ({100*y.mean():.2f}%)")
for nb in (5, 8, 10):
    order = np.argsort(pct)
    bins = np.array_split(order, nb)
    rates = [100*y[b].mean() for b in bins]
    npos = [int(y[b].sum()) for b in bins]
    print(f"  {nb} bins: rates%", " ".join(f"{v:.2f}" for v in rates), "| positives/bin", npos)

# ── render: quintiles with 95% binomial CIs ─────────────────────────────────
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB = 5
order = np.argsort(pct)
bins = np.array_split(order, NB)
rates, los, his = [], [], []
for b in bins:
    p = y[b].mean(); n = len(b)
    se = math.sqrt(p*(1-p)/n)
    rates.append(100*p); los.append(100*max(0, p-1.96*se)); his.append(100*(p+1.96*se))

with plt.rc_context({"font.family":"serif","font.size":11}):
    fig, ax = plt.subplots(figsize=(7,4.4))
    x = np.arange(NB)
    ax.errorbar(x, rates, yerr=[np.array(rates)-np.array(los), np.array(his)-np.array(rates)],
                fmt="o", ms=9, color="#33618f", ecolor="#33618f", elinewidth=1.2,
                capsize=4, zorder=3)
    ax.axhline(100*y.mean(), color="#999", lw=0.8, ls=(0,(4,3)))
    ax.text(-0.35, 100*y.mean()+0.07, "average market (1.2%)", fontsize=9, color="#777")
    ax.set_xticks(x)
    ax.set_xticklabels(["safest\n20%", "", "middle\n20%", "", "riskiest\n20%"], fontsize=10)
    ax.set_xlabel("Markets ranked by the model's predicted risk (contract text only)", fontsize=10)
    ax.set_ylabel("Share that actually ended\nin a dispute (%)", fontsize=10)
    ax.set_ylim(0, 3.2)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(length=0)
    for xi, v in zip(x, rates):
        ax.text(xi+0.09, v+0.12, f"{v:.1f}%", fontsize=10, fontweight="bold", color="#333")
    ax.set_title("In a representative sample, the model's risk ranking holds up",
                 fontsize=12.5, pad=12)
    fig.tight_layout()
    fig.savefig("/Users/andrewhall/event_contracts/substack/figs/fig2b_realworld.png", dpi=150)
print("wrote fig2b_realworld.png")
