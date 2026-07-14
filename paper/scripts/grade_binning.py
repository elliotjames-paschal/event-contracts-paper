"""Monotonic supervised grade binning (shared by Table 6 and the appendix figure).

Design (Section 5.4):
  * Composite score = gradient-boosted trees (HGBModel, the deployed scoring
    model) on the axes, target = MATERIAL disputes, fit on the TRAIN split only
    (no leakage into the cutpoints or the test check). Pass model=HandLogit()
    etc. for the robustness panels.
  * The 70/30 split is GROUPED: whole Kalshi series / Polymarket slug families
    go to one side, so twin markets never straddle the train/test boundary.
  * Cutpoints from monotonic supervised binning of the score against the material
    label on TRAIN: fine pre-bins -> isotonic PAVA (enforce monotone risk) -> merge
    adjacent bins whose dispute rates are not statistically distinct (two-proportion
    z-test). The number of grades is therefore DATA-DRIVEN, not fixed at 8.
  * Baseline / reference population = the full random historical sample
    (in_historical == 1), INCLUDING the ~66 markets that were disputed -- this
    represents the statistical characteristics of all markets, not just clean ones.
  * The case-control oversample (disputed & not in the random sample) supplies extra
    material/confirmed positives for binning and populates the dispute columns; it is
    never the baseline.

Dispute RATES within a bin are case-control-inflated, so they order/merge bins only;
grades are labelled by relative risk tier, never as an absolute dispute probability.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from dispute_regression import HGBModel, market_units, LEARN_COLS  # noqa: E402
from _dispute_data import (graded_rows, grouped_train_mask, is_genuine,  # noqa: E402
                           is_confirmed)

SEED = 42
# Fixed top of the scale; the D-tier is anchored at the top of [DDD, DD, D] and
# extends downward only as far as the grade count needs, so more D's mean a better
# grade (DDD > DD > D) and the tier reads from the full form first. E.g. 8 grades
# -> ...,C,DDD; 9 -> ...,C,DDD,DD; 10 -> ...,C,DDD,DD,D.
GRADE_LETTERS = ["A", "BBB", "BB", "B", "CCC", "CC", "C"]
D_TIER = ["DDD", "DD", "D"]  # best -> worst


def grade_labels(G):
    if G <= len(GRADE_LETTERS):
        return GRADE_LETTERS[:G]
    nd = G - len(GRADE_LETTERS)
    # take the fullest D forms first; pad with single "D" in the unlikely event
    # the D-tier needs more than three grades.
    d = [D_TIER[i] if i < len(D_TIER) else "D" for i in range(nd)]
    return GRADE_LETTERS + d

# Investment-grade boundary on the monotonic scale: grade indices >= NOT_IG_FROM
# (0 = best) are "not investment grade". With the 5 material-trained grades
# (A, BBB, BB, B, CCC) this places B and CCC below the line.
NOT_IG_FROM = 3


def load():
    grades = graded_rows()
    rows = ([r for r in grades if r["platform"] == "polymarket"]
            + [r for r in grades if r["platform"] == "kalshi"])
    units = market_units("polymarket") + market_units("kalshi")
    pop = np.array([r.get("in_historical") == 1 for r in rows])          # random sample (incl. disputed)
    disp = np.array([r["disputed"] == 1 for r in rows])
    mat = np.array([is_genuine(r) for r in rows])
    conf = np.array([is_confirmed(r) for r in rows])
    return rows, units, pop, disp, mat, conf


def _ptest(n1, p1, n2, p2):
    P = (p1 + p2) / (n1 + n2)
    if P <= 0 or P >= 1:
        return 1.0
    se = math.sqrt(P * (1 - P) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = abs(p1 / n1 - p2 / n2) / se
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def _bins(s, t, K=50, alpha=0.05, maxshare=None, groups=None):
    """Monotonic supervised cutpoints on score s against binary label t.

    groups: optional per-observation cluster labels (series/family). Bin dispute
    RATES stay market-level (the estimand), but the merge z-tests use the number
    of distinct clusters as the effective sample size — twin markets are close
    to perfectly correlated (label-pure series, duplicated axis vectors), so
    counting them as independent over-splits the ladder into grades that do not
    replicate on a grouped test split."""
    qs = np.unique(np.quantile(s, np.linspace(0, 1, K + 1)))
    interior = qs[1:-1]
    bidx = np.searchsorted(interior, s, side="right")
    nb = int(bidx.max()) + 1
    n = np.zeros(nb); pos = np.zeros(nb); up = np.full(nb, -1e18)
    gsets = [set() for _ in range(nb)]
    if groups is None:
        groups = range(len(s))                      # each obs its own cluster
    for bi, ti, si, gi in zip(bidx, t, s, groups):
        n[bi] += 1; pos[bi] += ti; up[bi] = max(up[bi], si)
        gsets[bi].add(gi)
    keep = n > 0
    n, pos, up = n[keep], pos[keep], up[keep]
    gsets = [g for g, k in zip(gsets, keep) if k]

    def eff(b):
        """Effective (count, positives) of block b: clusters, not markets."""
        ne = len(b[4])
        return ne, b[1] / b[0] * ne

    blk = []  # [count, positives, start, end, cluster-set]
    for i in range(len(n)):
        blk.append([n[i], pos[i], i, i, set(gsets[i])])
        while len(blk) >= 2 and blk[-2][1] / blk[-2][0] > blk[-1][1] / blk[-1][0]:
            a = blk.pop(); c = blk.pop()
            blk.append([c[0] + a[0], c[1] + a[1], c[2], a[3], c[4] | a[4]])
    while len(blk) > 2:
        ps = [_ptest(*eff(blk[i]), *eff(blk[i + 1]))
              for i in range(len(blk) - 1)]
        j = int(np.argmax(ps))
        if ps[j] <= alpha:
            break
        a = blk.pop(j + 1); c = blk.pop(j)
        blk.insert(j, [c[0] + a[0], c[1] + a[1], c[2], a[3], c[4] | a[4]])
    if maxshare is not None:
        tot = n.sum(); out = []
        for N, P, st, en, _g in blk:
            if N <= maxshare * tot or en == st:
                out.append((st, en)); continue
            parts = math.ceil(N / (maxshare * tot)); tgt = N / parts; cs, acc = st, 0
            for pi in range(st, en + 1):
                acc += n[pi]
                if acc >= tgt and pi < en:
                    out.append((cs, pi)); cs = pi + 1; acc = 0
            out.append((cs, en))
        blk = [[0, 0, st, en, set()] for st, en in out]
    return np.array([up[b[3]] for b in blk[:-1]])


def fit_grades(target="material", maxshare=None, seed=SEED, cols=None,
               model=None, score=None):
    """Returns dict with cuts, per-contract grade index g, letters, and masks.
    cols: predictor columns for the composite score (default = the 9 LEARN_COLS;
    pass LEARN_COLS + ['logvol'] for the volume-augmented model).
    model: an optional registry model (.fit/.score) used to build the composite
    score; score: an optional precomputed full-length score (e.g. the
    equal-weighted sum). With both None the deployed gradient-boosted composite
    (HGBModel) is used. The 70/30 split is grouped by series/family."""
    if cols is None:
        cols = LEARN_COLS
    rows, units, pop, disp, mat, conf = load()
    X = np.array([[u[c] for c in cols] for u in units], float)
    tgt = {"material": mat, "confirmed": conf, "all": disp}[target]
    insamp = np.where(pop | tgt)[0]                # representative population + oversampled positives
    y = tgt[insamp].astype(int)
    # grouped split: whole series/families to one side (leakage-safe)
    groups_in = [units[i]["group"] for i in insamp]
    trmask = grouped_train_mask(groups_in, y, frac=0.7, seed=seed)
    train, test = insamp[trmask], insamp[~trmask]
    if score is not None:                          # precomputed score (e.g. equal sum)
        score = np.asarray(score, float)
    else:                                          # registry model, fit on the train split
        if model is None:
            model = HGBModel()                     # deployed boosted composite (Table 5)
        mu, sd = X[train].mean(0), X[train].std(0); sd[sd == 0] = 1
        gtr = [units[i]["group"] for i in train]
        model.fit((X[train] - mu) / sd, tgt[train].astype(float), groups=gtr)
        score = np.asarray(model.score((X - mu) / sd), float)
    cuts = _bins(score[train], tgt[train].astype(int), maxshare=maxshare,
                 groups=[units[i]["group"] for i in train])
    G = len(cuts) + 1
    g = np.digitize(score, cuts)
    letters = grade_labels(G)
    # Data-driven investment-grade cutoff via the dispute ODDS RATIO, which is
    # invariant to the (arbitrary) case-control sampling ratio --- a raw-rate cutoff is
    # not, because its base rate shifts with how many controls/disputes are sampled.
    # The IG line is set on the case-control sample (the representative random sample
    # has essentially no disputes --- e.g. 1 confirmed in 5,567 --- so per-grade rates
    # cannot be estimated from it). A grade is investment grade if its dispute odds
    # (disputes : clean) are at or below the overall odds, i.e. it holds no larger a
    # share of disputes than of clean markets; scaling the controls cancels in the
    # ratio. not_ig_from = first grade (best -> worst) whose odds ratio exceeds 1.
    ins_g = g[insamp]
    ins_t = tgt[insamp].astype(int)
    base_rate = float(ins_t.mean())
    D = int(ins_t.sum()); C = len(ins_t) - D
    base_odds = D / C if C else float("inf")
    not_ig_from = G
    for k in range(G):
        dk = int(ins_t[ins_g == k].sum()); ck = int((ins_g == k).sum()) - dk
        odds_ratio = (dk / ck) / base_odds if (ck and base_odds) else float("inf")
        if odds_ratio > 1:
            not_ig_from = k
            break
    return dict(rows=rows, score=score, cuts=cuts, g=g, G=G, letters=letters,
                pop=pop, disp=disp, mat=mat, conf=conf, train=train, test=test, tgt=tgt,
                not_ig_from=not_ig_from, base_rate=base_rate)


def _rates(g, tgt, idx, G):
    sub_g = g[idx]; sub_t = tgt[idx]
    return [sub_t[sub_g == k].mean() if (sub_g == k).sum() else float("nan") for k in range(G)]


def main():
    for target in ("material", "all", "confirmed"):
        f = fit_grades(target=target)
        g, G, tgt = f["g"], f["G"], f["tgt"]
        tr_r = _rates(g, tgt, f["train"], G); te_r = _rates(g, tgt, f["test"], G)
        mono = lambda r: all(r[i] <= r[i + 1] + 1e-9 for i in range(G - 1)
                             if not (math.isnan(r[i]) or math.isnan(r[i + 1])))
        shares = [ (g == k).mean() for k in range(G) ]
        print(f"\n=== target={target}: {G} grades  cuts={np.round(f['cuts'],2)} ===")
        print(f"  letters: {f['letters']}")
        print(f"  train rate%: " + " ".join(f"{100*x:.0f}" for x in tr_r) + f"  monotone={mono(tr_r)}")
        print(f"  test  rate%: " + " ".join(f"{100*x:.0f}" for x in te_r) + f"  monotone={mono(te_r)}")
        print(f"  herfindahl={sum(s*s for s in shares):.3f}  max-share={100*max(shares):.0f}%")
        print(f"  grade | share% | within-grade: pop% all% material% confirmed%")
        for k in range(G):
            m = g == k
            print(f"    {f['letters'][k]:3s}: {100*m.mean():4.1f}%  "
                  f"pop={100*f['pop'][m].mean():3.0f} all={100*f['disp'][m].mean():3.0f} "
                  f"mat={100*f['mat'][m].mean():3.0f} conf={100*f['conf'][m].mean():3.0f}  (n={m.sum()})")


if __name__ == "__main__":
    main()
