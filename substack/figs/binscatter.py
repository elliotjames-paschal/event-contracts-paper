"""Substack Figure 2 prototype: out-of-sample binscatter.

Every market is scored by a model that never saw its series/family (grouped
5-fold CV, out-of-fold predictions). x = predicted dispute risk (percentile);
y = share of markets in the bin that actually ended in a dispute.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/andrewhall/event_contracts/event-contracts-paper")
for p in (REPO/"src", REPO/"scripts", REPO/"paper"/"scripts"):
    sys.path.insert(0, str(p))
import random
from dispute_regression import market_units, HGBModel, LEARN_COLS
from _dispute_data import grouped_fold_assign

def oof_scores(units, k=5, seed=0):
    M = np.array([[u[c] for c in LEARN_COLS] for u in units], float)
    y = np.array([u["disputed"] for u in units])
    groups = [u["group"] for u in units]
    fold = grouped_fold_assign(groups, y, k, random.Random(seed))
    s = np.full(len(units), np.nan)
    for f in range(k):
        tr = fold != f
        mu, sd = M[tr].mean(0), M[tr].std(0); sd[sd == 0] = 1
        mdl = HGBModel().fit((M[tr]-mu)/sd, y[tr])
        s[~tr] = mdl.score((M[~tr]-mu)/sd)
    return s, y

def binscatter(ax, s, y, nbins=20, color="#33618f"):
    order = np.argsort(s)
    bins = np.array_split(order, nbins)
    xs = [100*(np.mean([np.searchsorted(np.sort(s), s[b]).mean() for _ in [0]])/len(s)) for b in bins]
    xs = [100*(i+0.5)/nbins for i in range(nbins)]
    ys = [100*y[b].mean() for b in bins]
    ax.scatter(xs, ys, s=42, color=color, zorder=3)
    ax.set_xlabel("Model's predicted dispute risk (percentile of markets)", fontsize=10)
    ax.set_ylabel("Share that actually ended\nin a dispute (%)", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    return xs, ys

pm = market_units("polymarket")
pm_spec = [u for u in pm if not u["disputed"] or u["classification"]=="specification_issue"]
s, y = oof_scores(pm_spec)

with plt.rc_context({"font.family":"serif","font.size":11}):
    fig, ax = plt.subplots(figsize=(7,4.4))
    xs, ys = binscatter(ax, s, y)
    ax.axhline(100*y.mean(), color="#999", lw=0.8, ls=(0,(4,3)))
    ax.text(1, 100*y.mean()+1.2, "average", fontsize=8.5, color="#777")
    ax.set_title("Dispute share by predicted-risk bin (Polymarket, out-of-sample)",
                 fontsize=12, pad=12)
    from fs_style import add_footer
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.text(0.5, 0.115, "Dispute-enriched evaluation sample (disputes oversampled ~75×): "
             "levels are not real-world rates; the gradient is the point.",
             ha="center", fontsize=8, color="#666666")
    add_footer(fig, credit="Data: Polymarket (Hall & Paschal)")
    fig.savefig("/Users/andrewhall/event_contracts/substack/figs/fig2_binscatter.png", dpi=150,
                bbox_inches="tight", facecolor="white")
print("bins (x%, dispute%):", [f"{x:.0f}:{v:.0f}" for x,v in zip(xs,ys)])
print("n =", len(pm_spec), "positives =", int(y.sum()))
