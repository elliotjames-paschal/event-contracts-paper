#!/usr/bin/env python3
"""Generate the Section 5.4 dispute-regression tables from the live data.

Writes:
  paper/tables/dispute_models.tex -- model-comparison leaderboard (pooled, axes-only): Table 3
  paper/tables/dispute_auc.tex   -- out-of-sample AUC: equal sum vs learned axes vs +volume
  paper/tables/dispute_axes.tex  -- standardized axis coefficients (PM primary; Kalshi concurrent)

Reuses the regression functions in scripts/dispute_regression.py so the paper
numbers always match the analysis. data/full_grades.jsonl is never modified.

Usage:
    PYTHONPATH=src .venv/bin/python paper/scripts/generate_regression_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper" / "scripts"))
from dispute_regression import (  # noqa: E402
    AXES, LEARN_COLS, EBMClassifier, HandLogit, SkLogit, cv_auc, fit_full,
    logit_l2, market_units,
)
from _dispute_data import graded_rows, is_genuine, is_confirmed  # noqa: E402
from grade_binning import fit_grades, _rates, NOT_IG_FROM  # noqa: E402  (monotonic grades)

# Binning target for the main-text grade table (Table 6); the appendix shows all three.
GRADE_TARGET = "all"

TAB = ROOT / "paper" / "tables"
LABELS = {
    "predicate_ambiguity": "Predicate ambiguity",
    "entity_ambiguity": "Entity ambiguity",
    "temporal_precision": "Temporal precision",
    "manipulation_susceptibility": "Manipulation susceptibility",
    "outcome_concentration": "Outcome concentration",
    "source_specification": "Source specification",
    "source_quality": "Source quality",
    "edge_case_coverage": "Edge case coverage",
    "headline_rules_alignment": "Headline--rules alignment",
    "governing_body_engagement": "Governing body engagement",
    "structural": "Structural integrity (manip.\\ $+$ outcome)",
    "logvol": "Log trading volume",
}


def mat_sample(units):
    """Clean controls + MATERIAL disputed markets (spec-relevant)."""
    return ([u for u in units if not u["disputed"]]
            + [u for u in units if u["disputed"] and u["genuine"]])


def conf_sample(units):
    """Clean controls + CONFIRMED disputed markets (contract demonstrably failed)."""
    return ([u for u in units if not u["disputed"]]
            + [u for u in units if u["disputed"] and u["confirmed"]])


def fmt_auc(x):
    return f"{x:.2f}"


def model_auc_full_table(outfile="dispute_auc_full.tex", label="tab:dispute_auc_full",
                         targets=("all", "material", "confirmed")):
    """Appendix Table 8: out-of-sample AUC for every model, by platform and
    severity, one panel per training set. Each model is fit once on the pooled
    Random$+<target>$ train split (matching Table~\\ref{tab:dispute_models}) and
    evaluated within each platform; tie-correct AUC. Equal sum is fit-free."""
    import json as _json

    import numpy as _np
    from sklearn.metrics import roc_auc_score as _rauc

    rows = graded_rows()
    rows = ([r for r in rows if r["platform"] == "polymarket"]
            + [r for r in rows if r["platform"] == "kalshi"])

    def feat(r):
        ax = {a: float(r["axes"][a]) for a in AXES}
        ax["structural"] = (ax["manipulation_susceptibility"] + ax["outcome_concentration"]) / 2
        return [ax[c] for c in LEARN_COLS]

    X = _np.array([feat(r) for r in rows], float)
    rawsum = _np.array([sum(float(r["axes"][a]) for a in AXES) for r in rows])
    pop = _np.array([r.get("in_historical") == 1 for r in rows])
    disp = _np.array([r["disputed"] == 1 for r in rows])
    mat = _np.array([is_genuine(r) for r in rows])
    conf = _np.array([is_confirmed(r) for r in rows])
    clean = pop & ~disp
    rng = _np.random.default_rng(42)
    tr = _np.zeros(len(rows), bool)
    for cls in (False, True):
        ix = _np.where(disp == cls)[0]; rng.shuffle(ix)
        tr[ix[:int(0.7 * len(ix))]] = True
    te = ~tr
    platform = _np.array([r["platform"] for r in rows])
    sev_arr = {"all": disp, "material": mat, "confirmed": conf}

    factories = [("Logistic", lambda: HandLogit()),
                 ("Lasso", lambda: SkLogit(l1_ratio=1.0)),
                 ("Ridge", lambda: SkLogit(l1_ratio=0.0)),
                 ("EBM", lambda: EBMClassifier())]
    xgb_ok = True
    try:
        from dispute_regression import XGBClassifierModel
        factories.append(("XGBoost", lambda: XGBClassifierModel()))
    except Exception as exc:  # noqa: BLE001
        xgb_ok = False
        print(f"  [skip] XGBoost column omitted from Table 8 ({type(exc).__name__})")
    model_names = ["Equal sum"] + [n for n, _ in factories]

    def fit_scores(target_pos):
        m = (clean | target_pos) & tr
        mu, sd = X[m].mean(0), X[m].std(0); sd[sd == 0] = 1
        scores = {"Equal sum": rawsum}
        for name, factory in factories:
            mdl = factory().fit((X[m] - mu) / sd, target_pos[m].astype(float))
            scores[name] = mdl.score((X - mu) / sd)
        return scores

    def cell(score, postype, platmask):
        msk = (postype | clean) & platmask & te
        y = postype[msk].astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return "--"
        return f"{_rauc(y, score[msk]):.2f}"

    platforms = [("Polymarket --- predictive", platform == "polymarket"),
                 ("Kalshi --- concurrent", platform == "kalshi"),
                 ("Pooled (both platforms)", _np.ones(len(rows), bool))]
    sevs = [("all disputes", disp), ("material", mat), ("confirmed", conf)]
    ncol = len(model_names)

    body = []
    for ti, t in enumerate(targets):
        scores = fit_scores(sev_arr[t])
        if ti:
            body += ["\\addlinespace", "\\midrule"]
        body.append(f"\\multicolumn{{{ncol + 2}}}{{l}}{{\\textbf{{Trained on: Random $+$ {t}}}}} \\\\")
        for pname, platmask in platforms:
            body.append(f"\\quad \\emph{{{pname}}} \\\\")
            for sname, sarr in sevs:
                n = int((sarr & platmask & te).sum())
                cells = " & ".join(cell(scores[m], sarr, platmask) for m in model_names)
                body.append(f"\\quad\\quad predicts {sname} & {n:,} & {cells} \\\\")

    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[p]", "\\centering", "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.0}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\caption{Out-of-sample dispute prediction, by model}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * (ncol + 1) + "}", "\\toprule",
        " & Positives & " + " & ".join(model_names) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{Held-out AUC ($0.5=$ chance) on a common 30\\% split, by platform and "
        "severity, with one panel per training set (random clean markets $+$ that dispute "
        "set as positives). Each model is fit once on the pooled training split and "
        "evaluated within each platform; ``Equal sum'' is the unweighted ten-axis total. "
        "Severity nests as all $\\supset$ material $\\supset$ confirmed; ``--'' marks too "
        "few positives to estimate. The pooled, confirmed-trained column matches "
        "Table~\\ref{tab:dispute_models}.}",
        "\\end{table}", "",
    ]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")


def auc_table(pm, kal, targets=("all", "material", "confirmed"),
              outfile="dispute_auc.tex", label="tab:dispute_auc"):
    """Train x test AUC matrix. One panel per training set in `targets` (random clean
    markets + that dispute set as positives), each reporting out-of-sample AUC at
    predicting all / material / confirmed disputes, by platform and pooled. Common
    70/30 split (so the diagonal is not leaky); negatives are always the random clean
    markets; volume added separately as the last column."""
    import json as _json
    import numpy as _np
    from dispute_regression import auc

    rows = graded_rows()
    rows = ([r for r in rows if r["platform"] == "polymarket"]
            + [r for r in rows if r["platform"] == "kalshi"])

    def feat(r):
        ax = {a: float(r["axes"][a]) for a in AXES}
        ax["structural"] = (ax["manipulation_susceptibility"] + ax["outcome_concentration"]) / 2
        return [ax[c] for c in LEARN_COLS]

    def _vol(r):
        try:
            return _np.log(float(r.get("volume") or 0) + 1)
        except (TypeError, ValueError):
            return 0.0

    X = _np.array([feat(r) for r in rows], float)
    XV = _np.column_stack([X, [_vol(r) for r in rows]])           # + log volume
    rawsum = _np.array([sum(float(r["axes"][a]) for a in AXES) for r in rows])  # equal sum (10 axes)
    pop = _np.array([r.get("in_historical") == 1 for r in rows])
    disp = _np.array([r["disputed"] == 1 for r in rows])
    mat = _np.array([is_genuine(r) for r in rows])
    conf = _np.array([is_confirmed(r) for r in rows])
    clean = pop & ~disp                                            # random non-disputed (negatives)

    rng = _np.random.default_rng(42)                              # one common 70/30 split
    tr = _np.zeros(len(rows), bool)
    for cls in (False, True):
        ix = _np.where(disp == cls)[0]; rng.shuffle(ix)
        tr[ix[:int(0.7 * len(ix))]] = True
    te = ~tr

    platform = _np.array([r["platform"] for r in rows])

    def fmt(x):
        return "--" if x is None else f"{x:.2f}"

    def fit(M, label, fit_mask):
        m = fit_mask & tr
        mu, sd = M[m].mean(0), M[m].std(0); sd[sd == 0] = 1
        b = logit_l2(_np.column_stack([_np.ones(m.sum()), (M[m] - mu) / sd]), label[m].astype(float))
        return _np.column_stack([_np.ones(len(M)), (M - mu) / sd]) @ b

    def cell_auc(score, postype, platmask):
        mask = (postype | clean) & platmask & te
        y = postype[mask].astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return None
        return auc(y, score[mask])

    test_types = [("all disputes", disp), ("material", mat), ("confirmed", conf)]
    platforms = [("Polymarket --- predictive validity", platform == "polymarket"),
                 ("Kalshi --- concurrent validity", platform == "kalshi"),
                 ("Pooled (both platforms)", _np.ones(len(rows), bool))]

    def panel(name, train_markets, label):
        out = [f"\\multicolumn{{5}}{{l}}{{\\textbf{{Trained on: {name}}}}} \\\\"]
        for pname, platmask in platforms:
            fit_mask = train_markets & platmask
            s_learn = fit(X, label, fit_mask)
            s_vol = fit(XV, label, fit_mask)
            out.append(f"\\quad \\emph{{{pname}}} \\\\")
            for tname, tt in test_types:
                n = int((tt & platmask & te).sum())
                eq = cell_auc(rawsum, tt, platmask)
                le = cell_auc(s_learn, tt, platmask)
                vo = cell_auc(s_vol, tt, platmask)
                out.append(f"\\quad\\quad predicts {tname} & {n:,} & {fmt(eq)} & {fmt(le)} & {fmt(vo)} \\\\")
        return out

    spec = {"all": ("Random $+$ all disputes", clean | disp, disp),
            "material": ("Random $+$ material", clean | mat, mat),
            "confirmed": ("Random $+$ confirmed", clean | conf, conf)}
    body = []
    for i, t in enumerate(targets):
        if i:
            body.append("\\midrule")
        body += panel(*spec[t])

    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[p]" if len(targets) > 1 else "\\begin{table}[t]",
        "\\centering", "\\footnotesize",
        "\\renewcommand{\\arraystretch}{0.95}",
        "\\caption{Out-of-sample dispute prediction}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        " & Positives & Equal sum & Learned axes & $+$ volume \\\\",
        "\\midrule",
        *body,
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{AUC ($0.5=$ chance) on a held-out 30\\% split. ``Equal sum'' is the "
        "unweighted ten-axis sum; ``$+$ volume'' adds $\\log$ trading volume (partly "
        "post-treatment). Severity nests as all $\\supset$ material $\\supset$ confirmed; "
        "``--'' denotes too few positives to estimate.}",
        "\\end{table}", "",
    ]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")


def model_leaderboard(outfile="dispute_models.tex", label="tab:dispute_models"):
    """Model-comparison leaderboard (the new Table 3).

    Models are the ROWS; out-of-sample AUC at predicting all/material/confirmed
    disputes are the COLUMNS. All models share one feature set (the nine axes,
    structural pair averaged) and one training set --- random clean markets +
    confirmed disputes, pooled across platforms --- so each row isolates the
    effect of the estimator, not the inputs. Same data, 70/30 split (seed 42)
    and standardization as auc_table(), so the Logistic row reproduces that
    table's pooled ``Learned axes'' cells. New models (EBM, XGBoost, ...) drop
    in as additional rows. Volume and the all/material training sets are held
    for the appendix."""
    import json as _json

    import numpy as _np
    # Tie-correct AUC (mid-ranks): the equal-sum score is integer-valued (0--30)
    # and heavily tied, where the in-house rank AUC is order-dependent and biased
    # under bootstrap. roc_auc_score uses the trapezoidal / mid-rank convention.
    from sklearn.metrics import roc_auc_score as _rauc

    rows = graded_rows()
    rows = ([r for r in rows if r["platform"] == "polymarket"]
            + [r for r in rows if r["platform"] == "kalshi"])

    def feat(r):
        ax = {a: float(r["axes"][a]) for a in AXES}
        ax["structural"] = (ax["manipulation_susceptibility"] + ax["outcome_concentration"]) / 2
        return [ax[c] for c in LEARN_COLS]

    X = _np.array([feat(r) for r in rows], float)
    rawsum = _np.array([sum(float(r["axes"][a]) for a in AXES) for r in rows])  # equal sum (10 axes)
    pop = _np.array([r.get("in_historical") == 1 for r in rows])
    disp = _np.array([r["disputed"] == 1 for r in rows])
    mat = _np.array([is_genuine(r) for r in rows])
    conf = _np.array([is_confirmed(r) for r in rows])
    clean = pop & ~disp                                            # random non-disputed (negatives)

    rng = _np.random.default_rng(42)                              # same common 70/30 split as auc_table
    tr = _np.zeros(len(rows), bool)
    for cls in (False, True):
        ix = _np.where(disp == cls)[0]; rng.shuffle(ix)
        tr[ix[:int(0.7 * len(ix))]] = True
    te = ~tr

    train_mask = clean | conf                                     # Random + Confirmed underlying set

    def fit_model(model, M):
        """Standardize on the (Random+Confirmed) train split, fit, score all rows."""
        m = train_mask & tr
        mu, sd = M[m].mean(0), M[m].std(0); sd[sd == 0] = 1
        model.fit((M[m] - mu) / sd, conf[m].astype(float))
        return model.score((M - mu) / sd)

    def cell_auc(score, postype):
        mask = (postype | clean) & te                            # pooled across both platforms
        y = postype[mask].astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return None
        return _rauc(y, score[mask])

    # (display name, score vector). Equal sum is fit-free; the rest fit on the train split.
    models = [
        ("Equal sum", rawsum),
        ("Logistic", fit_model(HandLogit(), X)),
        ("Logistic (L1, lasso)", fit_model(SkLogit(l1_ratio=1.0), X)),
        ("Logistic (L2, ridge)", fit_model(SkLogit(l1_ratio=0.0), X)),
        ("EBM (glass-box GAM)", fit_model(EBMClassifier(), X)),
    ]
    # XGBoost needs the xgboost package (and libomp on macOS). Gate on import so
    # the build stays clean if it is unavailable, and auto-includes it once present.
    try:
        from dispute_regression import XGBClassifierModel
        models.append(("Gradient-boosted trees (XGBoost)", fit_model(XGBClassifierModel(), X)))
    except Exception as exc:  # noqa: BLE001  (import or native-lib load failure)
        print(f"  [skip] XGBoost row omitted ({type(exc).__name__}: {exc})")
    test_types = [("All disputes", disp), ("Material", mat), ("Confirmed", conf)]
    npos = {tname: int((tt & te).sum()) for tname, tt in test_types}

    def boot_stats(score_by_model, y, B=2000, seed=42):
        """Stratified test-set bootstrap (positives and negatives resampled
        separately, preserving prevalence). Returns per model a 95% AUC CI and
        whether its paired AUC gap vs the equal sum excludes zero (same resampled
        indices across models, so the comparison is paired)."""
        rng = _np.random.default_rng(seed)
        pos = _np.where(y == 1)[0]; neg = _np.where(y == 0)[0]
        names = list(score_by_model)
        acc = {n: _np.empty(B) for n in names}
        dlt = {n: _np.empty(B) for n in names}
        for b in range(B):
            idx = _np.concatenate([rng.choice(pos, pos.size, True),
                                   rng.choice(neg, neg.size, True)])
            yb = y[idx]
            a = {n: _rauc(yb, s[idx]) for n, s in score_by_model.items()}
            for n in names:
                acc[n][b] = a[n]; dlt[n][b] = a[n] - a["Equal sum"]
        out = {}
        for n in names:
            lo, hi = _np.percentile(acc[n], [2.5, 97.5])
            dl, dh = _np.percentile(dlt[n], [2.5, 97.5])
            sig = (n != "Equal sum") and (dl > 0 or dh < 0)   # ΔAUC CI excludes 0
            out[n] = (lo, hi, sig)
        return out

    # One paired bootstrap per test column (all models share the resampled rows).
    stats = {}
    for tname, tt in test_types:
        mask = (tt | clean) & te
        y = tt[mask].astype(int)
        stats[tname] = boot_stats({mn: sc[mask] for mn, sc in models}, y)

    def fmt_val(point, sig):
        if point is None:
            return "--"
        return f"{point:.2f}" + ("$^{*}$" if sig else "")

    def fmt_ci(point, lo, hi):
        if point is None:
            return ""
        return f"{{\\scriptsize\\color{{black!55}}$[{lo:.2f},\\,{hi:.2f}]$}}"

    # Two physical rows per model: AUC point estimates, then 95% CIs beneath
    # (the coefficient-over-standard-error convention).
    body = []
    for mname, score in models:
        vals, cis = [], []
        for tname, tt in test_types:
            point = cell_auc(score, tt)
            lo, hi, sig = stats[tname][mname]
            vals.append(fmt_val(point, sig))
            cis.append(fmt_ci(point, lo, hi))
        body.append(f"{mname} & " + " & ".join(vals) + " \\\\")
        body.append(" & " + " & ".join(cis) + " \\\\[3pt]")

    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[t]",
        "\\centering", "\\small",
        "\\caption{Out-of-sample dispute prediction: model comparison}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{12pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular}{lccc}", "\\toprule",
        " & \\multicolumn{3}{c}{Out-of-sample AUC} \\\\",
        "\\cmidrule(lr){2-4}",
        "Model & All & Material & Confirmed \\\\",
        "\\midrule",
        *body,
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{Out-of-sample AUC ($0.5=$ chance) on a held-out 30\\% split, pooled "
        "across platforms; positives are confirmed disputes, negatives random non-disputed "
        "markets. ``Equal sum'' is the unweighted ten-axis total, fit-free. L1/L2 and "
        "XGBoost hyperparameters are tuned by inner cross-validation on the training split; "
        "EBM uses its defaults with internal early stopping. Brackets: 95\\% CIs from "
        "2{,}000 stratified bootstrap resamples. $^{*}$: AUC differs from the equal sum "
        "(paired bootstrap, 95\\%). Positives: "
        f"{npos['All disputes']:,} all, {npos['Material']:,} material, "
        f"{npos['Confirmed']:,} confirmed.}}",
        "\\end{table}", "",
    ]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")


FIG_MODEL_KEYS = ["equal", "logistic", "l1", "l2", "ebm", "xgb", "volume"]


def auc_matrix(target="confirmed"):
    """Per-platform, per-severity out-of-sample AUC for every leaderboard model,
    for Figure~\\ref{fig:dispute_auc}. Each model is fit ONCE on the pooled
    Random$+<target>$ train split (so the pooled/confirmed cell matches
    Table~\\ref{tab:dispute_models}) and evaluated within each platform, using the
    same tie-correct AUC. The single '+volume' line is the logistic with
    log(volume) added (a diagnostic ceiling; not part of the grade). Returns
        {platform: {model_key: [all, material, confirmed], ..., "n": [...]}}.
    XGBoost ('xgb') is omitted if the package is unavailable."""
    import json as _json

    import numpy as _np
    from sklearn.metrics import roc_auc_score as _rauc

    rows = graded_rows()
    rows = ([r for r in rows if r["platform"] == "polymarket"]
            + [r for r in rows if r["platform"] == "kalshi"])

    def feat(r):
        ax = {a: float(r["axes"][a]) for a in AXES}
        ax["structural"] = (ax["manipulation_susceptibility"] + ax["outcome_concentration"]) / 2
        return [ax[c] for c in LEARN_COLS]

    def _vol(r):
        try:
            return _np.log(float(r.get("volume") or 0) + 1)
        except (TypeError, ValueError):
            return 0.0

    X = _np.array([feat(r) for r in rows], float)
    XV = _np.column_stack([X, [_vol(r) for r in rows]])
    rawsum = _np.array([sum(float(r["axes"][a]) for a in AXES) for r in rows])
    pop = _np.array([r.get("in_historical") == 1 for r in rows])
    disp = _np.array([r["disputed"] == 1 for r in rows])
    mat = _np.array([is_genuine(r) for r in rows])
    conf = _np.array([is_confirmed(r) for r in rows])
    clean = pop & ~disp

    rng = _np.random.default_rng(42)
    tr = _np.zeros(len(rows), bool)
    for cls in (False, True):
        ix = _np.where(disp == cls)[0]; rng.shuffle(ix)
        tr[ix[:int(0.7 * len(ix))]] = True
    te = ~tr
    platform = _np.array([r["platform"] for r in rows])

    target_pos = {"all": disp, "material": mat, "confirmed": conf}[target]
    train_mask = clean | target_pos

    def fit_model(model, M):
        m = train_mask & tr
        mu, sd = M[m].mean(0), M[m].std(0); sd[sd == 0] = 1
        model.fit((M[m] - mu) / sd, target_pos[m].astype(float))
        return model.score((M - mu) / sd)

    scores = {
        "equal": rawsum,                                   # fit-free baseline
        "logistic": fit_model(HandLogit(), X),
        "l1": fit_model(SkLogit(l1_ratio=1.0), X),
        "l2": fit_model(SkLogit(l1_ratio=0.0), X),
        "ebm": fit_model(EBMClassifier(), X),
        "volume": fit_model(HandLogit(), XV),              # logistic + log(volume)
    }
    try:
        from dispute_regression import XGBClassifierModel
        scores["xgb"] = fit_model(XGBClassifierModel(), X)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] XGBoost line omitted from figure ({type(exc).__name__})")

    def cell_auc(score, postype, platmask):
        mask = (postype | clean) & platmask & te
        y = postype[mask].astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return None
        return _rauc(y, score[mask])

    severities = [disp, mat, conf]
    platforms = {"Polymarket": platform == "polymarket",
                 "Kalshi": platform == "kalshi",
                 "Pooled": _np.ones(len(rows), bool)}

    result = {}
    for pname, platmask in platforms.items():
        result[pname] = {k: [cell_auc(s, tt, platmask) for tt in severities]
                         for k, s in scores.items()}
        result[pname]["n"] = [int((tt & platmask & te).sum()) for tt in severities]
    return result


def coef_table(outfile="dispute_axes.tex", label="tab:dispute_axes"):
    """Table 4: standard logistic regression of the confirmed-dispute indicator
    on the nine axis scores, with lasso (L1) and ridge (L2) penalized fits as
    robustness columns. Coefficients are the change in dispute log-odds per
    one-point increase on each axis's 0--3 scale; SEs in parentheses; stars from
    |z|. Column (1) uses series-clustered SEs; the penalized columns use a cluster
    bootstrap. EBM and XGBoost have no coefficients --- their predictive
    performance lives in Table~\\ref{tab:dispute_models} / Figure~\\ref{fig:dispute_auc}."""
    from collections import defaultdict

    import numpy as _np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    cols = LEARN_COLS
    sample = conf_sample(market_units("polymarket") + market_units("kalshi"))
    M = _np.array([[u[c] for c in cols] for u in sample], float)
    y = _np.array([u["disputed"] for u in sample], float)
    groups = [u["group"] for u in sample]
    sd = M.std(0); sd[sd == 0] = 1
    mu = M.mean(0)
    Xs = (M - mu) / sd                                   # standardized design
    n = len(sample)

    def stars(z):
        a = abs(z)
        return "^{***}" if a > 2.576 else "^{**}" if a > 1.96 else "^{*}" if a > 1.645 else ""

    # (1) Logistic: standardized coefs + series-clustered SEs from fit_full.
    logit = {nm: (c, se) for nm, c, se in fit_full(sample, cols, cluster=True)}

    # (2)-(3) Penalized: coef at the CV-selected C, SE from a cluster bootstrap.
    gidx = defaultdict(list)
    for i, g in enumerate(groups):
        gidx[g].append(i)
    glist = list(gidx)

    def penalized(l1_ratio, B=200):
        Cs = _np.logspace(-3, 3, 20)
        C = GridSearchCV(
            LogisticRegression(l1_ratio=l1_ratio, solver="saga", max_iter=5000),
            {"C": Cs}, scoring="roc_auc", cv=5,
        ).fit(Xs, y).best_params_["C"]

        def fit(Xi, yi):
            return LogisticRegression(l1_ratio=l1_ratio, solver="saga", max_iter=5000,
                                      C=C).fit(Xi, yi).coef_.ravel()
        coef = fit(Xs, y)
        rng = _np.random.default_rng(42); boots = []
        for _ in range(B):
            idx = _np.concatenate([gidx[glist[k]]
                                   for k in rng.choice(len(glist), len(glist), replace=True)])
            if 0 < int(y[idx].sum()) < len(idx):
                try:
                    boots.append(fit(Xs[idx], y[idx]))
                except Exception:  # noqa: BLE001
                    pass
        return coef, (_np.std(boots, 0) if boots else _np.full(len(cols), _np.nan))

    pen = {"Lasso": penalized(1.0), "Ridge": penalized(0.0)}

    # Standardized coefs -> per-point (0--3 scale) by dividing by the column SD;
    # z (hence stars) is scale-invariant.
    def cells(c_std, se_std, j):
        c, se = c_std / sd[j], se_std / sd[j]
        z = c_std / se_std if se_std and not _np.isnan(se_std) else 0.0
        cc = f"${c:+.2f}{stars(z)}$"
        ss = "" if _np.isnan(se) else f"$({se:.2f})$"
        return cc, ss

    order = sorted(range(len(cols)), key=lambda j: -abs(logit[cols[j]][0] / sd[j]))
    body = []
    for j in order:
        nm = cols[j]
        cc, ss = cells(*logit[nm], j)
        row_c, row_s = [cc], [ss]
        for cf, se in pen.values():
            c2, s2 = cells(cf[j], se[j], j)
            row_c.append(c2); row_s.append(s2)
        body.append(f"{LABELS[nm]} & " + " & ".join(row_c) + " \\\\")
        body.append(" & " + " & ".join(row_s) + " \\\\[2pt]")

    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[t]",
        "\\centering", "\\small",
        "\\caption{Which axes predict disputes}",
        f"\\label{{{label}}}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\setlength{\\tabcolsep}{12pt}",
        "\\begin{tabular}{lccc}", "\\toprule",
        " & (1) & (2) & (3) \\\\",
        " & Logistic & Lasso & Ridge \\\\",
        "\\midrule",
        *body,
        "\\midrule",
        f"Observations & {n:,} & {n:,} & {n:,} \\\\",
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{Logistic regression of the confirmed-dispute indicator (vs.\\ "
        "random non-disputed markets) on the nine axis scores. Coefficients give the "
        "change in dispute log-odds per one-point increase on each axis's 0--3 "
        "severity scale; standard errors in parentheses. Column~(1) is the unpenalized "
        "logit with series-clustered standard errors; columns~(2)--(3) are the $L_1$ "
        "(lasso) and $L_2$ (ridge) penalized fits, penalty chosen by cross-validation, "
        "with cluster-bootstrap standard errors (200 reps). $^{*}$, $^{**}$, $^{***}$ "
        "denote $p<0.10, 0.05, 0.01$. The two collinear Core Principle~3 axes are "
        "combined into one structural-integrity term. EBM and gradient-boosted trees "
        "have no coefficients; their predictive performance is in "
        "Table~\\ref{tab:dispute_models} and Appendix Table~\\ref{tab:dispute_auc_full}.}",
        "\\end{table}", "",
    ]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")


def nonlinear_importance_table(outfile="axis_importance_ml.tex",
                               label="tab:axis_importance_ml"):
    """Appendix companion to Table~\\ref{tab:dispute_axes}: relative feature
    importance of the two nonlinear models, which have no coefficients. Cells =
    each axis's share (\\%) of a model's total importance --- EBM term importance
    (mean$|$contribution$|$, exact for an additive model) and XGBoost mean
    $|$TreeSHAP$|$. Fit on the same pooled confirmed sample as Table~4."""
    import numpy as _np

    cols = LEARN_COLS
    sample = conf_sample(market_units("polymarket") + market_units("kalshi"))
    M = _np.array([[u[c] for c in cols] for u in sample], float)
    y = _np.array([u["disputed"] for u in sample], float)
    sd = M.std(0); sd[sd == 0] = 1
    Xs = (M - M.mean(0)) / sd
    p = len(cols)

    def signed_importance(phi):
        """Signed relative importance: share (%) of total mean|contribution|,
        with the sign of each axis's contribution--value relationship attached."""
        phi = _np.asarray(phi)[:, :p]
        ma = _np.abs(phi).mean(0)
        direction = _np.array([
            _np.sign(_np.corrcoef(Xs[:, j], phi[:, j])[0, 1])
            if _np.ptp(phi[:, j]) > 0 else 1.0 for j in range(p)
        ])
        return direction * 100 * ma / ma.sum()

    imp = {}
    ebm = EBMClassifier().fit(Xs, y).clf
    imp["EBM"] = signed_importance(ebm.eval_terms(Xs))
    try:
        import shap
        from dispute_regression import XGBClassifierModel
        xg = XGBClassifierModel().fit(Xs, y).clf.best_estimator_
        imp["XGBoost"] = signed_importance(shap.TreeExplainer(xg).shap_values(Xs))
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] XGBoost importance omitted ({type(exc).__name__}: {exc})")

    model_labels = [m for m in ["EBM", "XGBoost"] if m in imp]
    order = sorted(range(p), key=lambda j: -_np.mean([abs(imp[m][j]) for m in model_labels]))
    body = [f"{LABELS[cols[j]]} & " + " & ".join(f"${imp[m][j]:+.0f}$" for m in model_labels)
            + " \\\\" for j in order]
    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[t]",
        "\\centering", "\\small",
        "\\caption{Axis importance in the nonlinear models}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{12pt}",
        "\\begin{tabular}{l" + "c" * len(model_labels) + "}", "\\toprule",
        "Axis & " + " & ".join(model_labels) + " \\\\",
        "\\midrule",
        *body,
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{Signed relative feature importance for the two nonlinear models, "
        "which have no coefficients (companion to Table~\\ref{tab:dispute_axes}). The "
        "magnitude is each axis's share (\\%) of a model's total importance --- EBM "
        "mean$|$term contribution$|$ (exact for an additive model), XGBoost mean "
        "$|$TreeSHAP$|$; the sign is the direction of the effect (positive $=$ a worse "
        "axis score goes with more disputes, like a positive coefficient). Magnitudes "
        "sum to 100 per model; fit on the pooled confirmed sample, rows ordered by "
        "importance.}",
        "\\end{table}", "",
    ]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")


def axes_table(pm, kal, cols=LEARN_COLS, outfile="dispute_axes.tex",
               label="tab:dispute_axes", volume=False,
               targets=("all", "material", "confirmed")):
    import numpy as _np
    pooled = pm + kal

    def raw_coefs(sample):
        # raw per-point log-odds = standardized coef / column SD; z (=> stars) unchanged.
        M = _np.array([[u[c] for c in cols] for u in sample], float)
        sd = M.std(0); sd[sd == 0] = 1
        out = {}
        for name, c, se in fit_full(sample, cols, cluster=True):
            j = cols.index(name)
            out[name] = (c / sd[j], (c / se if se else 0.0))
        return out

    # same 70/30 stratified split as Table 4 (seed 42, units order == rows order)
    disp = _np.array([u["disputed"] == 1 for u in pooled])
    rng = _np.random.default_rng(42)
    tr = _np.zeros(len(pooled), bool)
    for cls in (False, True):
        ix = _np.where(disp == cls)[0]; rng.shuffle(ix)
        tr[ix[:int(0.7 * len(ix))]] = True
    train = [u for u, t in zip(pooled, tr) if t]
    all_s = train                                                # random clean + all disputes
    mat_s = [u for u in train if (not u["disputed"]) or u["genuine"]]
    conf_s = [u for u in train if (not u["disputed"]) or u["confirmed"]]

    co = {"all": raw_coefs(all_s), "material": raw_coefs(mat_s), "confirmed": raw_coefs(conf_s)}
    colmap = {"all": "All", "material": "Material", "confirmed": "Confirmed"}
    cols_sel = [(colmap[t], co[t]) for t in targets]

    def cell(d, name):
        c, z = d[name]
        star = "^{*}" if abs(z) > 1.96 else ""
        return f"${c:+.2f}{star}$"

    # keep logvol last; order the rest by |first column's coefficient|
    ref = cols_sel[0][1]
    axis_cols = [c for c in cols if c != "logvol"]
    order = sorted(axis_cols, key=lambda a: -abs(ref[a][0]))
    if "logvol" in cols:
        order = order + ["logvol"]
    ncol = len(cols_sel)
    vol_note = (" Bottom row adds $\\log$ trading volume (partly post-treatment)."
                if volume else "")
    title = ("Which axes predict disputes, with trading volume" if volume
             else "Which axes predict disputes")
    setname = ("the training set of Table~\\ref{tab:dispute_auc} (random clean markets"
               " plus material disputes)" if ncol == 1 else
               "the three training sets of Table~\\ref{tab:dispute_auc} (random clean"
               " markets plus all / material / confirmed disputes)")
    header2 = "Predictor & " + " & ".join(nm for nm, _ in cols_sel) + " \\\\"
    grp = ("" if ncol == 1 else
           f" & \\multicolumn{{{ncol}}}{{c}}{{Training set (random $+$ \\ldots)}} \\\\\n"
           f"\\cmidrule(lr){{2-{ncol + 1}}}")
    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{{title}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{l" + "c" * ncol + "}", "\\toprule",
    ]
    if grp:
        lines.append(grp)
    lines += [header2, "\\midrule"]
    for a in order:
        if a == "logvol":
            lines.append("\\addlinespace")
        lines.append(f"{LABELS[a]} & " + " & ".join(cell(d, a) for _, d in cols_sel) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\tabnote{Pooled logistic coefficients: change in dispute log-odds per "
              "one-point increase on each axis's 0--3 severity scale; positive $=$ a worse "
              "score goes with more disputes. $^{*}$ denotes $|z|>1.96$ "
              f"(series-clustered SEs).{vol_note}}}",
              "\\end{table}", ""]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")


def vif_table():
    """Appendix: variance inflation factors for the ten axes (all contracts)."""
    import json as _json
    import numpy as _np
    rows = graded_rows()
    M = _np.array([[r["axes"][a] for a in AXES] for r in rows], float)

    def vif(k):
        y = M[:, k]
        X = _np.column_stack([_np.ones(len(M)), _np.delete(M, k, axis=1)])
        b, *_ = _np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return 1 / (1 - r2) if r2 < 1 else float("inf")

    vifs = sorted(((LABELS[a], vif(k)) for k, a in enumerate(AXES)),
                  key=lambda t: -t[1])
    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\caption{Multicollinearity among the ten axes (VIF)}",
        "\\label{app:vif}",
        "\\setlength{\\tabcolsep}{12pt}",
        "\\begin{tabular}{lc}", "\\toprule", "Axis & VIF \\\\", "\\midrule",
    ]
    for name, v in vifs:
        lines.append(f"{name} & {v:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\tabnote{Variance inflation factors over all graded contracts. Only the "
              "two Core Principle 3 axes (manipulation susceptibility, outcome "
              "concentration) exceed the usual 5--10 threshold; we average them into one "
              "structural-integrity term in the learned models.}",
              "\\end{table}", ""]
    (TAB / "vif.tex").write_text("\n".join(lines))
    print("wrote tables/vif.tex")


def grade_tier_table(cols=None, outfile="grade_tiers.tex", label="tab:grade_tiers",
                     volume=False, targets=("all", "material", "confirmed")):
    """Monotonic supervised grades (Section 5.4), three stacked panels --- one per
    training set (random clean markets + all / material / confirmed disputes). Each
    panel learns its own composite weights and grade cutpoints on the 70\\% training
    split: the predicted-risk distribution is recursively split into the maximum
    number of adjacent buckets whose dispute rates differ significantly (isotonic
    regression + two-proportion $z$-test, $p<0.05$), so the grade count is data-driven.
    Cells are market counts per population; investment grade = top tiers."""
    import numpy as _np

    def panel(target, lbl):
        f = fit_grades(target=target, cols=cols)
        g, G, letters, cuts = f["g"], f["G"], f["letters"], f["cuts"]
        groups = [("Population", f["pop"]), ("All disputes", f["disp"]),
                  ("Material", f["mat"]), ("Confirmed", f["conf"])]

        def cells(m, bold=False):
            out = [f"{int((m & sub).sum()):,}" for _, sub in groups]
            return " & ".join(f"\\textbf{{{v}}}" if bold else v for v in out)

        def srange(klo, khi):
            lo = cuts[klo - 1] if klo > 0 else None
            hi = cuts[khi] if khi < G - 1 else None
            if lo is None:
                return f"$\\leq {hi:+.2f}$"
            if hi is None:
                return f"$> {lo:+.2f}$"
            return f"${lo:+.2f}$ to ${hi:+.2f}$"

        nf = f["not_ig_from"]                       # data-driven IG cutoff for this panel
        ig, nig = list(range(nf)), list(range(nf, G))
        out = [f"\\multicolumn{{6}}{{l}}{{\\textbf{{Trained on: Random $+$ {lbl}}}}} \\\\"]
        out += [f"\\quad {letters[k]} & {srange(k, k)} & {cells(g == k)} \\\\" for k in ig]
        if ig:
            out += [f"\\quad\\textbf{{Investment grade}} & {srange(min(ig), max(ig))} & "
                    f"{cells(_np.isin(g, ig), bold=True)} \\\\"]
        out += [f"\\quad {letters[k]} & {srange(k, k)} & {cells(g == k)} \\\\" for k in nig]
        if nig:
            out += [f"\\quad\\textbf{{Not investment grade}} & {srange(min(nig), max(nig))} & "
                    f"{cells(_np.isin(g, nig), bold=True)} \\\\"]
        return out, groups

    lbls = {"all": "all disputes", "material": "material", "confirmed": "confirmed"}
    body = []
    groups = None
    for i, t in enumerate(targets):
        p, groups = panel(t, lbls[t])
        if i:
            body += ["\\addlinespace", "\\midrule"]
        body += p
    head_names = " & ".join(nm for nm, _ in groups)
    head_ns = " & ".join(f"($n{{=}}{int(sub.sum()):,}$)".replace(",", "{,}") for _, sub in groups)
    title = ("Monotonic supervised grades with trading volume, by training set"
             if volume else "Monotonic supervised grades, by training set")
    vol_note = (" Composite score also includes $\\log$ trading volume (partly"
                " post-treatment): a diagnostic, not the deployed grade." if volume else "")
    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[p]" if len(targets) > 1 else "\\begin{table}[t]",
        "\\centering", "\\footnotesize",
        "\\renewcommand{\\arraystretch}{0.95}",
        f"\\caption{{{title}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        f"Grade & Score range & {head_names} \\\\",
        f" & & {head_ns} \\\\",
        "\\midrule",
        *body,
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{Grades from monotonic supervised binning of the composite risk score; "
        "the grade count is data-driven. Cells count the markets in each population ($n$ "
        "in header) falling in each grade. Score range is the grade's interval on the "
        "composite log-odds scale. Rates are case-control inflated, so grades denote "
        "relative risk. The investment-grade line is the dispute odds-ratio cutoff (a grade "
        "is investment grade if its disputes-to-clean odds are at or below the overall), "
        f"which is invariant to the case-control sampling ratio.{vol_note}}}",
        "\\end{table}", "",
    ]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")


def grade_robustness_table(outfile="grade_robustness.tex",
                           label="tab:grade_robustness", target="confirmed"):
    """Appendix: the actual monotonic supervised grades (same format as
    Table~\\ref{tab:grade_tiers}) produced by each candidate scoring model --- one
    stacked panel per model, all trained on the confirmed target --- so you can read
    off whether the grade ladder itself moves when the binned score changes."""
    import numpy as _np

    units = market_units("polymarket") + market_units("kalshi")
    rawsum = _np.array([sum(u[a] for a in AXES) for u in units], float)
    specs = [
        ("Equal sum", dict(score=rawsum)),
        ("Logistic", dict(model=HandLogit())),
        ("Lasso (L1)", dict(model=SkLogit(l1_ratio=1.0))),
        ("Ridge (L2)", dict(model=SkLogit(l1_ratio=0.0))),
        ("EBM", dict(model=EBMClassifier())),
    ]
    try:
        from dispute_regression import XGBClassifierModel
        specs.append(("XGBoost", dict(model=XGBClassifierModel())))
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] XGBoost panel omitted from grade robustness ({type(exc).__name__})")

    def panel(name, kw):
        f = fit_grades(target=target, **kw)
        g, G, letters, cuts = f["g"], f["G"], f["letters"], f["cuts"]
        groups = [("Population", f["pop"]), ("All disputes", f["disp"]),
                  ("Material", f["mat"]), ("Confirmed", f["conf"])]

        def cells(m, bold=False):
            out = [f"{int((m & sub).sum()):,}" for _, sub in groups]
            return " & ".join(f"\\textbf{{{v}}}" if bold else v for v in out)

        def srange(klo, khi):
            lo = cuts[klo - 1] if klo > 0 else None
            hi = cuts[khi] if khi < G - 1 else None
            if lo is None:
                return f"$\\leq {hi:+.2f}$"
            if hi is None:
                return f"$> {lo:+.2f}$"
            return f"${lo:+.2f}$ to ${hi:+.2f}$"

        nf = f["not_ig_from"]
        ig, nig = list(range(nf)), list(range(nf, G))
        out = [f"\\multicolumn{{6}}{{l}}{{\\textbf{{Scoring model: {name}}}}} \\\\"]
        out += [f"\\quad {letters[k]} & {srange(k, k)} & {cells(g == k)} \\\\" for k in ig]
        if ig:
            out += [f"\\quad\\textbf{{Investment grade}} & {srange(min(ig), max(ig))} & "
                    f"{cells(_np.isin(g, ig), bold=True)} \\\\"]
        out += [f"\\quad {letters[k]} & {srange(k, k)} & {cells(g == k)} \\\\" for k in nig]
        if nig:
            out += [f"\\quad\\textbf{{Not investment grade}} & {srange(min(nig), max(nig))} & "
                    f"{cells(_np.isin(g, nig), bold=True)} \\\\"]
        return out, groups

    body, groups = [], None
    for i, (name, kw) in enumerate(specs):
        p, groups = panel(name, kw)
        if i:
            body += ["\\addlinespace", "\\midrule"]
        body += p
    head_names = " & ".join(nm for nm, _ in groups)
    head_ns = " & ".join(f"($n{{=}}{int(sub.sum()):,}$)".replace(",", "{,}") for _, sub in groups)
    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[p]", "\\centering", "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.0}",
        "\\caption{Monotonic supervised grades, by scoring model}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lccccc}", "\\toprule",
        f"Grade & Score range & {head_names} \\\\",
        f" & & {head_ns} \\\\",
        "\\midrule",
        *body,
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{Each panel applies the monotonic supervised binning of "
        "Table~\\ref{tab:grade_tiers} to a different scoring model; cells count markets "
        "per population. Score ranges are on each model's own scale --- comparable "
        "within a panel, not across. The interpretable scores give near-identical "
        "ladders; the nonlinear models resolve a few more tiers.}",
        "\\end{table}", "",
    ]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")


def grade_summary_table(outfile="grade_tiers.tex", label="tab:grade_tiers", target="confirmed"):
    """Table 5 (merges the old grade-tier and volume tables): per grade, the count of
    representative-sample markets and of confirmed disputes, plus total dollar volume ---
    the monotonic-binning grade ladder together with the ``where the money sits''
    result. Each grade's share of markets and of dollar volume sits in small type
    beneath the counts, as standard errors do elsewhere in the paper."""
    import numpy as _np

    f = fit_grades(target=target)
    g, G, pop, conf = f["g"], f["G"], f["pop"], f["conf"]
    letters, nig, cuts = f["letters"], f["not_ig_from"], f["cuts"]
    vol = _np.array([(r.get("volume") or 0) for r in f["rows"]], float)
    Npop = int(pop.sum()); Vpop = float(vol[pop].sum()); Nconf = int(conf.sum())

    def grade_rows(name, gmask, bold=False):
        popm = pop & gmask
        nmk = int(popm.sum()); ncf = int((conf & gmask).sum()); tv = float(vol[popm].sum())
        pctm = 100 * nmk / Npop
        pctc = 100 * ncf / Nconf if Nconf else 0.0
        pctv = 100 * tv / Vpop if Vpop else 0.0
        w = (lambda x: f"\\textbf{{{x}}}") if bold else (lambda x: x)
        line1 = (f"{w(name)} & {w(f'{pctm:.1f}\\%')} & {w(f'{pctc:.1f}\\%')} & "
                 f"{w(f'{pctv:.1f}\\%')} \\\\")
        sub = lambda s: f"{{\\scriptsize\\color{{black!55}}{s}}}"
        line2 = (f" & {sub(f'{nmk:,}')} & {sub(str(ncf))} & "
                 f"{sub(f'\\${tv / 1e6:,.0f}M')} \\\\[3pt]")
        return [line1, line2]

    npop_s = f"{Npop:,}".replace(",", "{,}")
    nconf_s = f"{Nconf:,}".replace(",", "{,}")
    ig_idx = list(range(nig)); nig_idx = list(range(nig, G))
    body = []
    for k in range(G):
        body += grade_rows(letters[k], g == k)
        if k == nig - 1 and ig_idx:
            body.append("\\addlinespace[1pt]")
            body += grade_rows("Investment grade", _np.isin(g, ig_idx), bold=True)
            body.append("\\addlinespace[1pt]")
    if nig_idx:
        body += grade_rows("Not investment grade", _np.isin(g, nig_idx), bold=True)

    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Monotonic supervised grades: markets, disputes, and dollar volume}",
        f"\\label{{{label}}}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\setlength{\\tabcolsep}{12pt}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Grade & Population & Confirmed & Total volume \\\\",
        "\\midrule",
        *body,
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{Each grade from monotonic supervised binning of the composite risk "
        "score (confirmed-trained); the grade count is data-driven. ``Population'' is the "
        f"count of representative-sample markets ($n{{=}}{npop_s}$), ``Confirmed'' the "
        f"count of confirmed disputes ($n{{=}}{nconf_s}$, case-control oversampled), and "
        "``Total volume'' the grade's dollar volume in the representative sample. The large "
        "figure is each grade's share of markets, of confirmed disputes, and of dollar "
        "volume; the small figure beneath is the underlying count (markets, disputes) or "
        "dollar amount. The investment-grade line is the dispute "
        "odds-ratio cutoff (a grade is investment grade if its disputes-to-clean odds are "
        "at or below the overall), invariant to the case-control sampling ratio; rates are "
        "case-control inflated, so grades denote relative risk.}",
        "\\end{table}", "",
    ]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")


# Axis groups for the ablation table. The seven domain-judgment axes are the
# specification axes from the design session (Section 3.1); the regulatory pair
# is the Core Principle 3 structural term + governing-body engagement (26-08).
_JUDGMENT_7 = ["predicate_ambiguity", "entity_ambiguity", "temporal_precision",
               "source_specification", "source_quality", "edge_case_coverage",
               "headline_rules_alignment"]
_REGULATORY = ["structural", "governing_body_engagement"]
_NEGATIVE_3 = ["temporal_precision", "headline_rules_alignment", "edge_case_coverage"]


def axis_ablation_table(outfile="axis_ablation.tex", label="tab:axis_ablation",
                        full_outfile="axis_ablation_full.tex",
                        full_label="tab:axis_ablation_full"):
    """Axis ablations (main text: logistic; appendix: all models, confirmed).

    Same design as the leaderboard: common 70/30 split (seed 42), train on
    random clean + confirmed, pooled tie-correct AUC on the held-out 30%.
    Panel A drops one predictor at a time (necessity); Panel B keeps/drops
    groups (sufficiency of the judgment axes, the regulatory pair, and the
    negative-coefficient axes). Stars: paired stratified bootstrap ΔAUC vs.
    the full model excludes zero (confirmed target)."""
    import numpy as _np
    from sklearn.metrics import roc_auc_score as _rauc

    rows = graded_rows()
    rows = ([r for r in rows if r["platform"] == "polymarket"]
            + [r for r in rows if r["platform"] == "kalshi"])
    ax = {a: _np.array([float(r["axes"][a]) for r in rows]) for a in AXES}
    feats = dict(ax)
    feats["structural"] = (ax["manipulation_susceptibility"]
                           + ax["outcome_concentration"]) / 2
    pop = _np.array([r.get("in_historical") == 1 for r in rows])
    disp = _np.array([r["disputed"] == 1 for r in rows])
    mat = _np.array([is_genuine(r) for r in rows])
    conf = _np.array([is_confirmed(r) for r in rows])
    clean = pop & ~disp
    rng = _np.random.default_rng(42)
    tr = _np.zeros(len(rows), bool)
    for cls in (False, True):
        ix = _np.where(disp == cls)[0]; rng.shuffle(ix)
        tr[ix[:int(0.7 * len(ix))]] = True
    te = ~tr
    train_mask = (clean | conf) & tr

    def fit_scores(model, cols):
        M = _np.column_stack([feats[c] for c in cols])
        mu, sd = M[train_mask].mean(0), M[train_mask].std(0); sd[sd == 0] = 1
        model.fit((M[train_mask] - mu) / sd, conf[train_mask].astype(float))
        return model.score((M - mu) / sd)

    # Equal-sum variants operate on the RAW ten axes: the structural term maps
    # back to its two constituent axes.
    def raw_cols(cols):
        out = []
        for c in cols:
            out += (["manipulation_susceptibility", "outcome_concentration"]
                    if c == "structural" else [c])
        return out

    def sum_scores(cols):
        return _np.sum([ax[c] for c in raw_cols(cols)], axis=0)

    def cell(score, postype):
        msk = (postype | clean) & te
        y = postype[msk].astype(int)
        if y.sum() in (0, len(y)):
            return None
        return _rauc(y, score[msk])

    variants = [("Full model (9 predictors)", LEARN_COLS)]
    loo = [(f"$-$ {LABELS[c]}", [x for x in LEARN_COLS if x != c])
           for c in LEARN_COLS]
    groups = [
        ("Domain-judgment axes only (7)", _JUDGMENT_7),
        ("Regulatory axes only (2)", _REGULATORY),
        ("$-$ Negative-coefficient axes", [c for c in LEARN_COLS
                                           if c not in _NEGATIVE_3]),
    ]

    log_scores = {n: fit_scores(HandLogit(), cols)
                  for n, cols in variants + loo + groups}

    # Paired stratified bootstrap on the confirmed test target: ΔAUC vs. full.
    msk = (conf | clean) & te
    y = conf[msk].astype(int)
    sub = {n: s[msk] for n, s in log_scores.items()}
    brng = _np.random.default_rng(42)
    pos = _np.where(y == 1)[0]; neg = _np.where(y == 0)[0]
    B = 2000
    dlt = {n: _np.empty(B) for n in sub if n != "Full model (9 predictors)"}
    for b in range(B):
        idx = _np.concatenate([brng.choice(pos, pos.size, True),
                               brng.choice(neg, neg.size, True)])
        yb = y[idx]
        ref = _rauc(yb, sub["Full model (9 predictors)"][idx])
        for n in dlt:
            dlt[n][b] = _rauc(yb, sub[n][idx]) - ref
    sig = {n: (lambda lo, hi: lo > 0 or hi < 0)(*_np.percentile(d, [2.5, 97.5]))
           for n, d in dlt.items()}

    full_conf = cell(log_scores["Full model (9 predictors)"], conf)

    def line(name):
        a = cell(log_scores[name], disp)
        m = cell(log_scores[name], mat)
        c = cell(log_scores[name], conf)
        if name == "Full model (9 predictors)":
            d = "---"
        else:
            d = f"${c - full_conf:+.2f}$" + ("$^{*}$" if sig[name] else "")
        return f"{name} & {a:.2f} & {m:.2f} & {c:.2f} & {d} \\\\"

    loo_sorted = sorted((n for n, _ in loo),
                        key=lambda n: cell(log_scores[n], conf))
    body = [line("Full model (9 predictors)"), "\\midrule",
            f"\\multicolumn{{5}}{{l}}{{\\textit{{Panel A: leave one out}}}} \\\\",
            *[line(n) for n in loo_sorted], "\\midrule",
            f"\\multicolumn{{5}}{{l}}{{\\textit{{Panel B: axis groups}}}} \\\\",
            *[line(n) for n, _ in groups]]

    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Axis ablations: out-of-sample dispute prediction (logistic)}",
        f"\\label{{{label}}}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        " & All & Material & Confirmed & $\\Delta$ Confirmed \\\\",
        "\\midrule",
        *body,
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{Out-of-sample AUC of the unpenalized logistic under axis "
        "ablations; same design as Table~\\ref{tab:dispute_models} (held-out "
        "30\\% split, pooled across platforms, trained on random clean $+$ "
        "confirmed disputes; negatives are random non-disputed markets). Panel~A "
        "refits without one predictor at a time, ordered by impact; Panel~B "
        "keeps or drops axis groups: the seven domain-judgment specification "
        "axes, the two regulatory dimensions (structural integrity, "
        "governing-body engagement), and the three axes with negative fitted "
        "coefficients (temporal precision, headline--rules alignment, edge-case "
        "coverage). $\\Delta$: change vs.\\ the full model on the confirmed "
        "target; $^{*}$ marks $\\Delta$AUC whose 95\\% CI excludes zero "
        "(paired stratified bootstrap, 2{,}000 resamples). The same ablations "
        "under the other scoring models are in Appendix "
        f"Table~\\ref{{{full_label}}}.}}",
        "\\end{table}", "",
    ]
    (TAB / outfile).write_text("\n".join(lines))
    print(f"wrote tables/{outfile}")

    # ── Appendix: confirmed-target AUC for the same ablations, all models ──
    models = [("Equal sum", None), ("Logistic", lambda: HandLogit()),
              ("EBM", lambda: EBMClassifier())]
    try:
        from dispute_regression import XGBClassifierModel
        models.append(("XGBoost", lambda: XGBClassifierModel()))
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] XGBoost column omitted from ablation appendix "
              f"({type(exc).__name__})")

    def conf_auc(mname, factory, cols):
        if mname == "Equal sum":
            return cell(sum_scores(cols), conf)
        if mname == "Logistic":
            return cell(log_scores_by_cols[tuple(cols)], conf)
        return cell(fit_scores(factory(), cols), conf)

    log_scores_by_cols = {tuple(cols): log_scores[n]
                          for n, cols in variants + loo + groups}
    all_variants = variants + [(n, c) for n, c in loo] + groups
    cols_hdr = " & ".join(m for m, _ in models)
    fbody = []
    for i, (n, cols) in enumerate(all_variants):
        if i == 1:
            fbody.append(f"\\multicolumn{{{len(models) + 1}}}{{l}}"
                         "{\\textit{Panel A: leave one out}} \\\\")
        if i == 1 + len(loo):
            fbody.append(f"\\multicolumn{{{len(models) + 1}}}{{l}}"
                         "{\\textit{Panel B: axis groups}} \\\\")
        cells = " & ".join(f"{conf_auc(m, f, cols):.2f}" for m, f in models)
        fbody.append(f"{n} & {cells} \\\\")
        if i == 0:
            fbody.append("\\midrule")

    lines = [
        "% auto-generated by paper/scripts/generate_regression_table.py",
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Axis ablations under each scoring model (confirmed target)}",
        f"\\label{{{full_label}}}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{l" + "c" * len(models) + "}", "\\toprule",
        " & " + cols_hdr + " \\\\",
        "\\midrule",
        *fbody,
        "\\bottomrule", "\\end{tabular}",
        "\\tabnote{Out-of-sample AUC at predicting confirmed disputes for the "
        "ablations of Table~\\ref{tab:axis_ablation}, under each scoring model. "
        "``Equal sum'' drops the axis from the unweighted raw-axis total (the "
        "structural term maps back to its two constituent Core Principle~3 "
        "axes); fitted models are refit on the reduced feature set. Same split "
        "and training design as Table~\\ref{tab:dispute_models}.}",
        "\\end{table}", "",
    ]
    (TAB / full_outfile).write_text("\n".join(lines))
    print(f"wrote tables/{full_outfile}")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    pm = market_units("polymarket")
    kal = market_units("kalshi")
    ALL = ("all", "material", "confirmed")

    # --- Main body: confirmed training set only (strictest, highest-precision) ---
    # Table 3 is the model-comparison leaderboard; the old platform x severity AUC
    # table (auc_table) is retired from the main text --- its platform split now
    # lives in Figure~\ref{fig:dispute_auc}. The full version stays in the appendix.
    model_leaderboard()                       # Table 3: model comparison (pooled, axes-only)
    coef_table()                              # Table 4: logistic / lasso / ridge coefficients
    axis_ablation_table()                     # Table 5: axis ablations (+ appendix twin)
    grade_summary_table()                     # Table 5: grades + dispute counts + dollar volume

    # --- Appendix ---
    # Table 8: full AUC by model (all training sets, by platform/severity). The
    # per-training-set coefficient/grade and volume tables were retired; nonlinear
    # interpretability is shown as native figures (EBM shapes + XGBoost SHAP).
    model_auc_full_table(targets=ALL)
    grade_robustness_table()                  # per-model grade ladders (companion to Table 5)
    vif_table()


if __name__ == "__main__":
    main()
