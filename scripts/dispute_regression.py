#!/usr/bin/env python3
"""Do the axes predict disputes? The defensible structure (Section 5.4 regression).

Two platforms answer two different questions, so they are NOT pooled:

  PRIMARY (predictive validity) -- Polymarket, market level.
    Disputes are organic UMA challenges, observed independently of our rating.
    Outcome = `specification_issue` (genuine spec failure; frivolous no_fault and
    resolution_error dropped). Robustness: ALL Polymarket disputes (any challenge).

  SECONDARY (concurrent validity) -- Kalshi, market level.
    Disputes were selected on the exchange's own spec flags, so this measures
    AGREEMENT between our rating and Kalshi's judgments, not prediction.

LEAKAGE CONTROL: both platforms list recurring near-identical contracts (Kalshi
series; Polymarket slug families like btc-updown-m), and >90% of markets share
an identical axis vector with a sibling. ALL cross-validation folds are
therefore grouped — a group (series/family) never straddles a fold boundary —
and inner hyperparameter tuning uses grouped folds as well. Coefficient SEs are
clustered on the same groups.

For each: equal-weighted-sum baseline vs. learned axis weights (logistic and
gradient-boosted trees) vs. +log(volume), all by grouped 5-fold out-of-sample
AUC. Volume is endogenous (disputes can drive volume), so it is a horse-race
robustness, not the headline. data/full_grades.jsonl is never modified.

Usage:
    PYTHONPATH=src:paper/scripts .venv/bin/python scripts/dispute_regression.py
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper" / "scripts"))
from _dispute_data import (graded_rows, group_of, grouped_fold_assign,
                           is_genuine, is_confirmed)

AXES = [
    "predicate_ambiguity", "entity_ambiguity", "temporal_precision",
    "manipulation_susceptibility", "outcome_concentration",
    "source_specification", "source_quality", "edge_case_coverage",
    "headline_rules_alignment", "governing_body_engagement",
]
# Predictors for the LEARNED models: the two collinear Core Principle 3 axes
# (manipulation + outcome concentration, VIF~22, r=0.98) are averaged into one
# "structural" term to remove the multicollinearity. Equal-sum baseline still
# uses all ten raw axes.
LEARN_COLS = [a for a in AXES if a not in
              ("manipulation_susceptibility", "outcome_concentration")] + ["structural"]
SEED = 42


def _vol(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def market_units(platform: str):
    """Market-level rows: {axes..., logvol, disputed, classification, group}.

    `group` is the leakage/clustering unit — Kalshi series or Polymarket slug
    family (see _dispute_data.group_of) — used for CV folds and clustered SEs.
    """
    out = []
    for r in graded_rows():
        if r["platform"] != platform:
            continue
        ax = {a: float(r["axes"][a]) for a in AXES}
        out.append({
            **ax,
            "structural": (ax["manipulation_susceptibility"] + ax["outcome_concentration"]) / 2,
            "logvol": math.log(_vol(r.get("volume")) + 1),
            "disputed": int(r["disputed"]),
            "genuine": int(is_genuine(r)),
            "confirmed": int(is_confirmed(r)),
            "classification": r.get("classification", ""),
            "group": group_of(r),
        })
    return out


def logit_l2(X, y, lam=0.0, iters=100):
    X = np.asarray(X, float); y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    P = np.eye(X.shape[1]) * lam; P[0, 0] = 0.0
    for _ in range(iters):
        p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-9, 1 - 1e-9)
        H = (X.T * (p * (1 - p))) @ X + P
        b = b + np.linalg.solve(H, X.T @ (y - p) - P @ b)
    return b


# ---------------------------------------------------------------------------
# Model registry. Each model exposes .fit(X, y, groups=None) / .score(X) where
# X is the standardized feature matrix WITHOUT an intercept column (cv_auc
# standardizes per train fold and passes that in); .score returns a value
# monotone in P(disputed=1) so it can be fed straight to auc(). `groups` is
# the leakage unit of the training rows: models that tune hyperparameters by
# inner CV MUST group those inner folds on it (twin markets would otherwise
# leak into the tuning and pick overfit settings); models with no tuning
# ignore it. This thin interface is what lets the same CV harness drive the
# unpenalized logit, the penalized logits, EBM, and the boosted trees without
# touching the fold / clustering logic.
# ---------------------------------------------------------------------------
class HandLogit:
    """The paper's hand-rolled Newton logit (default: unpenalized, lam=0).
    No hyperparameters, so `groups` is ignored."""

    def __init__(self, lam=0.0):
        self.lam = lam
        self.b = None

    def fit(self, X, y, groups=None):
        Xi = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        self.b = logit_l2(Xi, y, lam=self.lam)
        return self

    def score(self, X):
        Xi = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        return Xi @ self.b


class SkLogit:
    """Penalized logistic regression with the penalty strength C chosen by an
    INNER cross-validation on each outer training fold (nested CV — the test
    fold is never used to tune C). l1_ratio=1.0 is lasso (L1, sparse);
    l1_ratio=0.0 is ridge (L2, smooth shrinkage). Uses the saga solver, which
    is the one that supports the full L1/L2/elastic-net family.

    When `groups` is passed, the inner CV is grouped on it (GroupKFold), so
    C-selection cannot leak across twin markets; without groups it falls back
    to stratified inner folds."""

    def __init__(self, l1_ratio, n_cs=20, inner_k=5):
        self.l1_ratio = float(l1_ratio)
        self.n_cs = n_cs
        self.inner_k = inner_k
        self.clf = None

    def fit(self, X, y, groups=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, GroupKFold
        Cs = np.logspace(-3, 3, self.n_cs)
        base = LogisticRegression(
            l1_ratio=self.l1_ratio, solver="saga", max_iter=5000,
        )
        cv = GroupKFold(n_splits=self.inner_k) if groups is not None else self.inner_k
        self.clf = GridSearchCV(
            base, {"C": Cs}, scoring="roc_auc", cv=cv,
        ).fit(np.asarray(X, float), np.asarray(y), groups=groups)
        return self

    def score(self, X):
        # decision_function is the linear predictor — monotone in P(y=1).
        return self.clf.decision_function(np.asarray(X, float))


class HGBModel:
    """Histogram gradient-boosted trees (scikit-learn) — the deployed scoring
    model. Nonlinear per-axis effects plus axis interactions; fixed shallow
    hyperparameters (no inner tuning, so nothing to leak). Under grouped CV it
    dominates every linear fit on the primary Polymarket sample, and its AUC
    is flat across a depth/rate sweep, so the fixed setting is not cherry-
    picked. Scale-invariant: standardized inputs are harmless."""

    def __init__(self, **kw):
        self.kw = dict(max_depth=3, learning_rate=0.05, max_iter=300,
                       random_state=SEED)
        self.kw.update(kw)
        self.clf = None

    def fit(self, X, y, groups=None):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.clf = HistGradientBoostingClassifier(**self.kw).fit(
            np.asarray(X, float), np.asarray(y))
        return self

    def score(self, X):
        return self.clf.predict_proba(np.asarray(X, float))[:, 1]


class EBMClassifier:
    """Explainable Boosting Machine — a glass-box GAM (interpret pkg). Nonlinear
    per-axis shape functions, but still inspectable, so it sits between the
    linear logits and the black-box boosted trees on the flexibility ladder.
    Defaults only (no inner tuning), so `groups` is ignored.
    Scale-invariant: the standardized inputs from cv_auc are harmless."""

    def __init__(self, **kw):
        self.kw = kw
        self.clf = None

    def fit(self, X, y, groups=None):
        from interpret.glassbox import ExplainableBoostingClassifier
        self.clf = ExplainableBoostingClassifier(
            random_state=SEED, **self.kw,
        ).fit(np.asarray(X, float), np.asarray(y))
        return self

    def score(self, X):
        return self.clf.predict_proba(np.asarray(X, float))[:, 1]


class XGBClassifierModel:
    """Gradient-boosted trees (XGBoost) — the flexible black-box ceiling. Shallow
    trees + inner-CV tuning of depth / n_estimators / learning rate, so the
    out-of-sample comparison is fair rather than an overfit strawman on the
    small positive counts. The inner tuning CV is grouped when `groups` is
    passed. Requires the xgboost package (and libomp on macOS)."""

    def __init__(self, inner_k=5):
        self.inner_k = inner_k
        self.clf = None

    def fit(self, X, y, groups=None):
        from sklearn.model_selection import GridSearchCV, GroupKFold
        from xgboost import XGBClassifier
        grid = {
            "n_estimators": [100, 300],
            "max_depth": [2, 3],
            "learning_rate": [0.03, 0.1],
        }
        base = XGBClassifier(
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=SEED, n_jobs=1,
        )
        cv = GroupKFold(n_splits=self.inner_k) if groups is not None else self.inner_k
        self.clf = GridSearchCV(
            base, grid, scoring="roc_auc", cv=cv,
        ).fit(np.asarray(X, float), np.asarray(y), groups=groups)
        return self

    def score(self, X):
        return self.clf.predict_proba(np.asarray(X, float))[:, 1]


def make_models():
    """The model set for the dispute horse-race, in display order."""
    return {
        "logit": HandLogit(),           # unpenalized MLE (linear reference)
        "logit_l1": SkLogit(l1_ratio=1.0),  # lasso: CV-selected, sparse
        "logit_l2": SkLogit(l1_ratio=0.0),  # ridge: CV-selected, smooth shrinkage
        "hgb": HGBModel(),              # gradient-boosted trees (deployed model)
    }


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def make_folds(units, k, grouped, seed):
    rng = random.Random(seed)
    if grouped:
        # whole series/families per fold, positives balanced across folds
        # (many groups are label-pure, so naive round-robin can starve a fold)
        groups = [u["group"] for u in units]
        y = [u["disputed"] for u in units]
        return list(grouped_fold_assign(groups, y, k, rng))
    # stratified by outcome (UNSAFE here: twin markets straddle folds — kept
    # only for measuring the leakage gap)
    idx_pos = [i for i, u in enumerate(units) if u["disputed"]]
    idx_neg = [i for i, u in enumerate(units) if not u["disputed"]]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    fold = [0] * len(units)
    for j, i in enumerate(idx_pos): fold[i] = j % k
    for j, i in enumerate(idx_neg): fold[i] = j % k
    return fold


def cv_auc(units, cols, grouped=True, k=5, seed=SEED, model=None):
    """Pooled out-of-sample AUC over k folds; standardize on each train fold.

    grouped=True (the default, and the only defensible setting) keeps each
    series/family entirely inside one fold. model: any object with
    .fit(X, y, groups)/.score(X) (see the registry above); defaults to the
    unpenalized hand-rolled logit. The model is refit from scratch on every
    fold, and the train fold's group labels are passed through so tuned models
    group their inner CV too."""
    if model is None:
        model = HandLogit()
    fold = make_folds(units, k, grouped, seed)
    M = np.array([[u[c] for c in cols] for u in units], float)
    y = np.array([u["disputed"] for u in units])
    groups = np.array([u["group"] for u in units])
    oos_y, oos_s = [], []
    for f in range(k):
        tr = np.array(fold) != f; te = ~tr
        if y[te].sum() in (0, int(te.sum())):
            continue
        mean = M[tr].mean(0); sd = M[tr].std(0); sd[sd == 0] = 1
        Xtr = (M[tr] - mean) / sd
        Xte = (M[te] - mean) / sd
        model.fit(Xtr, y[tr], groups=groups[tr])
        oos_y.append(y[te]); oos_s.append(model.score(Xte))
    return auc(np.concatenate(oos_y), np.concatenate(oos_s))


def fit_full(units, cols, cluster=False):
    """Full-sample standardized logit; returns coefs and SEs (clustered if asked)."""
    M = np.array([[u[c] for c in cols] for u in units], float)
    y = np.array([u["disputed"] for u in units], float)
    mean = M.mean(0); sd = M.std(0); sd[sd == 0] = 1
    X = np.column_stack([np.ones(len(units)), (M - mean) / sd])
    b = logit_l2(X, y)
    p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-9, 1 - 1e-9)
    bread = np.linalg.inv((X.T * (p * (1 - p))) @ X + np.eye(X.shape[1]) * 1e-8)
    if cluster:
        scores = X * (y - p)[:, None]
        meat = np.zeros((X.shape[1], X.shape[1]))
        gi = defaultdict(list)
        for i, u in enumerate(units): gi[u["group"]].append(i)
        for idx in gi.values():
            ug = scores[idx].sum(0); meat += np.outer(ug, ug)
        cov = bread @ meat @ bread
    else:
        cov = bread
    se = np.sqrt(np.diag(cov))
    return list(zip(cols, b[1:], se[1:]))


def report(units, label, cluster=True, grouped=True):
    nd = sum(u["disputed"] for u in units)
    ng = len({u["group"] for u in units})
    print(f"\n{'='*70}\n{label}\n  n={len(units)}  disputed={nd} ({100*nd/len(units):.1f}%)  groups={ng}")
    # equal-sum baseline (no fitting): OOS == full-sample AUC of the raw sum
    y = [u["disputed"] for u in units]; s = [sum(u[a] for a in AXES) for u in units]
    print(f"  equal-weighted SUM            CV-AUC* = {auc(y, s):.3f}   (*no fitting)")
    aA = cv_auc(units, AXES, grouped=grouped)
    aB = cv_auc(units, AXES + ["logvol"], grouped=grouped)
    print(f"  learned 10 axes (logit)       CV-AUC  = {aA:.3f}")
    print(f"  learned 10 axes + log(volume) CV-AUC  = {aB:.3f}")
    # Horse-race on the 9 LEARN_COLS (structural pair pre-averaged); tuned
    # models select hyperparameters by grouped inner CV within each train fold.
    for name, mdl in make_models().items():
        if name == "logit":
            continue  # the unpenalized fit is already shown above
        a = cv_auc(units, LEARN_COLS, grouped=grouped, model=mdl)
        print(f"  {name:13s} (9 axes)         CV-AUC  = {a:.3f}")
    coefs = sorted(fit_full(units, AXES, cluster=cluster), key=lambda t: -abs(t[1]))
    tag = " (group-clustered SE)" if cluster else ""
    print(f"  standardized coefficients{tag}:")
    for name, c, se in coefs:
        z = c / se if se else float("nan")
        star = "*" if abs(z) > 1.96 else " "
        sign = "  <-- fewer disputes" if c < 0 else ""
        print(f"    {name:28s} {c:+.3f}  (z={z:+.1f}){star}{sign}")


def main():
    # All CV grouped by series/family; SEs clustered on the same groups.
    pm = market_units("polymarket")
    pm_clean = [u for u in pm if not u["disputed"]]
    pm_specissue = pm_clean + [u for u in pm if u["disputed"] and u["classification"] == "specification_issue"]
    report(pm_specissue, "PRIMARY  -- Polymarket, specification_issue vs clean (predictive validity)")
    report(pm, "ROBUSTNESS -- Polymarket, ALL disputes vs clean")
    kal = market_units("kalshi")
    report(kal, "SECONDARY -- Kalshi, all spec-flagged disputes (agreement with the "
                "exchange's own judgments, NOT prediction)")


if __name__ == "__main__":
    main()
