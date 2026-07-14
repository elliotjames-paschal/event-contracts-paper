"""Fig 2 variant C: prior-shift-corrected predicted probabilities vs observed rates.

Per platform: OOF scores from grouped CV on the case-control sample; predicted
odds rescaled by (real-world odds / training-sample odds); evaluation on the
representative random sample only. Both platforms then share honest probability
units, so they pool without the percentile trick.
"""
import sys, random, math
from pathlib import Path
import numpy as np
REPO = Path("/Users/andrewhall/event_contracts/event-contracts-paper")
for p in (REPO/"src", REPO/"scripts", REPO/"paper"/"scripts"):
    sys.path.insert(0, str(p))
from dispute_regression import HGBModel, LEARN_COLS, AXES
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

probs, ys = [], []
for plat in ("polymarket", "kalshi"):
    un = units_all(plat)
    s, y = oof(un)
    r = np.array([u["rand"] for u in un], bool)
    p_train = y.mean()                       # case-control prevalence the model saw
    p_real = y[r].mean()                     # real-world prevalence (random sample)
    o_ratio = (p_real/(1-p_real)) / (p_train/(1-p_train))
    s = np.clip(s, 1e-6, 1-1e-6)
    odds = s/(1-s) * o_ratio
    p_corr = odds/(1+odds)
    probs.append(p_corr[r]); ys.append(y[r])
    print(f"{plat}: train prev {100*p_train:.1f}%, real prev {100*p_real:.2f}%, "
          f"corrected prob range [{100*p_corr[r].min():.2f}%, {100*p_corr[r].max():.2f}%], "
          f"median {100*np.median(p_corr[r]):.2f}%")

p = np.concatenate(probs); y = np.concatenate(ys)
NB = 5
order = np.argsort(p)
bins = np.array_split(order, NB)
xs, rates, los, his = [], [], [], []
for b in bins:
    xs.append(100*p[b].mean())
    pr = y[b].mean(); n = len(b)
    se = math.sqrt(pr*(1-pr)/n)
    rates.append(100*pr); los.append(100*max(0, pr-1.96*se)); his.append(100*(pr+1.96*se))
print("bin means (pred%, obs%):", [f"{a:.2f}:{b:.2f}" for a, b in zip(xs, rates)])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
with plt.rc_context({"font.family":"serif","font.size":11}):
    fig, ax = plt.subplots(figsize=(6.4,5.2))
    lim = max(max(his), max(xs)) * 1.15
    ax.plot([0, lim], [0, lim], color="#999", lw=0.9, ls=(0,(4,3)), zorder=1)
    ax.text(lim*0.72, lim*0.78, "perfect calibration", fontsize=9, color="#888",
            rotation=38, rotation_mode="anchor")
    ax.errorbar(xs, rates, yerr=[np.array(rates)-np.array(los), np.array(his)-np.array(rates)],
                fmt="o", ms=9, color="#33618f", ecolor="#33618f", elinewidth=1.2,
                capsize=4, zorder=3)
    ax.set_xlabel("Model's predicted probability of a dispute (%)", fontsize=10)
    ax.set_ylabel("Share that actually ended in a dispute (%)", fontsize=10)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Predicted vs. observed dispute rates\n(representative sample, base-rate corrected)",
                 fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig("/Users/andrewhall/event_contracts/substack/figs/fig2c_probability.png", dpi=150)
print("wrote fig2c_probability.png")
