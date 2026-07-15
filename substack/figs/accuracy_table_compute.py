"""Apples-to-apples leaderboard: every method, every severity sample, same
repeated grouped CV (8 seeds x 5 folds, identical folds across methods)."""
import sys, json, random
from pathlib import Path
import numpy as np
REPO = Path("/Users/andrewhall/event_contracts/event-contracts-paper")
for p in (REPO/"src", REPO/"scripts", REPO/"paper"/"scripts"):
    sys.path.insert(0, str(p))
from dispute_regression import (market_units, AXES, LEARN_COLS, HandLogit,
                                HGBModel, EBMClassifier, auc)
from _dispute_data import grouped_fold_assign

class RidgeCV_:
    """CV-tuned L2 logistic (lbfgs, grouped inner CV)."""
    def fit(self, X, y, groups=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, GroupKFold
        cv = GroupKFold(n_splits=5) if groups is not None else 5
        self.clf = GridSearchCV(
            LogisticRegression(max_iter=2000),
            {"C": np.logspace(-3, 2, 8)}, scoring="roc_auc", cv=cv,
        ).fit(X, y, groups=groups)
        return self
    def score(self, X):
        return self.clf.decision_function(X)

class XGBFixed:
    """XGBoost with the same fixed shallow config as the deployed HGB."""
    def fit(self, X, y, groups=None):
        from xgboost import XGBClassifier
        self.clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=42, n_jobs=4
                                 ).fit(X, y)
        return self
    def score(self, X):
        return self.clf.predict_proba(X)[:, 1]

pooled = market_units("polymarket") + market_units("kalshi")
SAMPLES = {
    "all":       pooled,
    "material":  [u for u in pooled if not u["disputed"] or u["genuine"]],
    "confirmed": [u for u in pooled if not u["disputed"] or u["confirmed"]],
}
METHODS = [("logistic", HandLogit), ("ridge", RidgeCV_), ("xgb", XGBFixed),
           ("hgb", HGBModel), ("ebm", EBMClassifier)]
REPS, K = 8, 5

out = {}
for sname, units in SAMPLES.items():
    M = np.array([[u[c] for c in LEARN_COLS] for u in units], float)
    y = np.array([u["disputed"] for u in units])
    groups = np.array([u["group"] for u in units])
    rawsum = np.array([sum(u[a] for a in AXES) for u in units], float)
    res = {"equal": [float(auc(y, rawsum))] }   # fit-free: same every repeat
    folds = [grouped_fold_assign(list(groups), y, K, random.Random(r)) for r in range(REPS)]
    for mname, factory in METHODS:
        per = []
        for r in range(REPS):
            fold = folds[r]
            oy, os_ = [], []
            for f in range(K):
                tr = fold != f; te = ~tr
                if y[te].sum() in (0, int(te.sum())): continue
                mu, sd = M[tr].mean(0), M[tr].std(0); sd[sd == 0] = 1
                mdl = factory().fit((M[tr]-mu)/sd, y[tr], groups=groups[tr])
                oy.append(y[te]); os_.append(mdl.score((M[te]-mu)/sd))
            per.append(float(auc(np.concatenate(oy), np.concatenate(os_))))
        res[mname] = per
        print(f"{sname:9s} {mname:8s} {np.mean(per):.3f} ± {np.std(per):.3f}", flush=True)
    out[sname] = res
    print(f"{sname:9s} equal    {res['equal'][0]:.3f}", flush=True)
    print(f"{sname:9s} n={len(units)} positives={int(y.sum())}", flush=True)

json.dump(out, open("accuracy_table_results.json", "w"), indent=1)
print("DONE")
