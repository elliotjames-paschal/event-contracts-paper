"""Shared dispute-analysis data layer for the Section 5.4 figures.

The underlying dataset (data/full_grades.jsonl) is never modified. All filtering
and aggregation happens here, in memory, so every figure draws from one
definition:

  * Kalshi is aggregated to the SERIES level (one observation per contract
    series), because Kalshi disputes are recorded at the event/series level and
    the ~5,700 markets cluster into ~1,300 series of near-identical contracts;
    treating each market as independent overstates the sample by ~4x. A series
    is "disputed" if any of its markets is; its score is the mean across markets.
  * Polymarket stays at the MARKET level (slugs are largely standalone) and, for
    predictive tests, is restricted to genuine `specification_issue` disputes
    (frivolous `no_fault` and `resolution_error` challenges dropped).
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "data" / "full_grades.jsonl"
UMA_PATH = ROOT / "data" / "fetched" / "uma_disputes_enriched.json"
KALSHI_EVENTS = ROOT / "data" / "disputed_kalshi_events.json"

# UMA settlement-price sentinels (int256-encoded)
_UMA_UNKNOWN = 5 * 10 ** 17       # 50/50 -- oracle judged unresolvable as written
_UMA_EARLY = -(2 ** 255)          # type(int256).min: "too early" / premature proposal


def graded_rows() -> list[dict]:
    return [json.loads(l) for l in open(RESULTS) if not json.loads(l).get("error")]


# ── Genuine-dispute determination ──────────────────────────────────────────
# A dispute is "genuine" if it plausibly reflects a specification problem:
#   * Polymarket: the UMA oracle settled the market to a definite Yes/No or to
#     "Unknown." Early-return settlements (the event had not yet occurred) are
#     premature-listing artifacts, not drafting flaws, and are dropped.
#   * Kalshi: the exchange applied a substantive post-resolution flag. Note-only
#     OTHER_INFO annotations are dropped.

_pm_settlement: dict | None = None
_kalshi_flags: tuple[set, set] | None = None


def pm_settlement_by_slug() -> dict:
    """slug -> 'definite' | 'unknown' | 'early' from the UMA settlement price."""
    global _pm_settlement
    if _pm_settlement is None:
        out = {}
        for r in json.load(open(UMA_PATH))["disputes"]:
            pm = r.get("polymarket") or {}
            slug = pm.get("slug")
            if not slug:
                continue
            try:
                p = int(r.get("settlement_price"))
            except (TypeError, ValueError):
                out[slug] = "definite"   # rare odd encoding = a concrete price
                continue
            if p == _UMA_UNKNOWN:
                out[slug] = "unknown"
            elif p == _UMA_EARLY or p < 0:
                out[slug] = "early"
            else:
                out[slug] = "definite"   # 0 (No), 1e18 (Yes), or other concrete
        _pm_settlement = out
    return _pm_settlement


def kalshi_flag_sets() -> tuple[set, set]:
    """(note_only_tickers, genuine_tickers) from the disputed Kalshi events."""
    global _kalshi_flags
    if _kalshi_flags is None:
        note_only, genuine = set(), set()
        for e in json.load(open(KALSHI_EVENTS))["events"]:
            tks = [m["ticker"] for m in e.get("markets", [])]
            (genuine if set(e["dispute_flags"]) != {"OTHER_INFO"} else note_only).update(tks)
        _kalshi_flags = (note_only, genuine)
    return _kalshi_flags


def is_genuine(row: dict) -> bool:
    """MATERIAL: does this disputed row reflect a real specification problem?
    (Polymarket: definite/Unknown UMA settlement; Kalshi: any substantive flag.)"""
    if not row.get("disputed"):
        return False
    if row["platform"] == "polymarket":
        return pm_settlement_by_slug().get(row["id"], "early") in ("definite", "unknown")
    note_only, genuine = kalshi_flag_sets()
    return not (row["id"] in note_only and row["id"] not in genuine)


# CONFIRMED: the strictest tier --- the contract demonstrably failed, not a one-off.
#   * Polymarket: a material market disputed two or more times (repeatedly challenged,
#     not a single proposer slip or a lone serial disputer).
#   * Kalshi: a material market the exchange had to void, reverse, or pause
#     (Rule 13.1) --- escalation beyond a mild clarification.
_KALSHI_CONFIRMED_FLAGS = {"VOIDED", "REVERSED", "RULE_13", "PAUSED"}
_pm_dispute_count: dict | None = None
_kalshi_ticker_flags: dict | None = None


def pm_dispute_count_by_slug() -> dict:
    """slug -> number of UMA dispute records (rounds) for that market."""
    global _pm_dispute_count
    if _pm_dispute_count is None:
        c: dict = {}
        for r in json.load(open(UMA_PATH))["disputes"]:
            pm = r.get("polymarket") or {}
            if pm.get("slug"):
                c[pm["slug"]] = c.get(pm["slug"], 0) + 1
        _pm_dispute_count = c
    return _pm_dispute_count


def kalshi_flags_by_ticker() -> dict:
    global _kalshi_ticker_flags
    if _kalshi_ticker_flags is None:
        d = {}
        for e in json.load(open(KALSHI_EVENTS))["events"]:
            fl = set(e.get("dispute_flags", []))
            for m in e.get("markets", []):
                d[m["ticker"]] = fl
        _kalshi_ticker_flags = d
    return _kalshi_ticker_flags


def is_confirmed(row: dict) -> bool:
    """CONFIRMED: a material dispute where the contract demonstrably failed."""
    if not is_genuine(row):
        return False
    if row["platform"] == "polymarket":
        return pm_dispute_count_by_slug().get(row["id"], 0) >= 2
    return bool(kalshi_flags_by_ticker().get(row["id"], set()) & _KALSHI_CONFIRMED_FLAGS)


def series_of(ticker: str) -> str:
    """Kalshi series id = ticker prefix before the first -<digit> (e.g.
    KXHOUSEMOV-24-R-T16 -> KXHOUSEMOV)."""
    m = re.match(r"^([A-Z]+[A-Z0-9]*?)(?=-\d|$)", ticker or "")
    return m.group(1) if m else (ticker or "").split("-")[0]


# ── Grouping and leakage-safe splits ────────────────────────────────────────
# Both platforms list recurring, near-identical contracts (Kalshi: ~5,700
# markets in ~1,300 series, dispute flags applied at the series level;
# Polymarket: recurring slug families like btc-updown-m). >90% of markets share
# an identical 10-axis vector with a sibling, so any train/test split that
# ignores this structure leaks: the model memorizes the twin instead of
# predicting. Every split and CV fold must therefore be GROUP-aware.

_PM_MONTHS = (r"(20\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|"
              r"february|march|april|june|july|august|september|october|"
              r"november|december)")


def pm_family(slug: str) -> str:
    """Polymarket family id: the slug with digits and month names stripped, so
    recurring listings (btc-updown-m-..., weekly temperature markets, ...)
    share one id. Heuristic, but errs toward merging — the safe direction."""
    s = re.sub(rf"\b{_PM_MONTHS}\b", "", slug or "")
    s = re.sub(r"\d+", "", s)
    return re.sub(r"-+", "-", s).strip("-")


def group_of(row: dict) -> str:
    """Leakage grouping unit for a graded row: Kalshi series / Polymarket
    family, prefixed by platform so ids never collide across platforms."""
    if row["platform"] == "kalshi":
        return "k:" + series_of(row["id"])
    return "p:" + pm_family(row["id"])


def grouped_fold_assign(groups: list, y, k: int, rng) -> np.ndarray:
    """Assign each group to one of k folds, greedily balancing positives (then
    sizes) so every fold has both classes even though groups are label-pure.
    rng: a random.Random (shuffles tie order between repeats)."""
    from collections import Counter
    gpos, gn = Counter(), Counter()
    for g, yi in zip(groups, y):
        gpos[g] += int(yi); gn[g] += 1
    gs = list(gpos)
    rng.shuffle(gs)
    gs.sort(key=lambda g: -gpos[g])          # place dispute-heavy groups first
    fpos = [0] * k; fn = [0] * k; assign = {}
    for g in gs:
        # balance positive groups on positive counts, label-pure clean groups
        # on fold size (a lexicographic (fpos, fn) key would funnel every clean
        # group into whichever fold is short one positive)
        if gpos[g] > 0:
            f = min(range(k), key=lambda i: (fpos[i], fn[i]))
        else:
            f = min(range(k), key=lambda i: (fn[i], fpos[i]))
        assign[g] = f; fpos[f] += gpos[g]; fn[f] += gn[g]
    return np.array([assign[g] for g in groups])


def grouped_train_mask(groups: list, y, frac: float = 0.7, seed: int = 42) -> np.ndarray:
    """Group-aware ~frac/1-frac split: whole groups go to train or test,
    greedily balancing positives so both sides keep both classes. Replaces the
    stratified market-level 70/30 split, which leaked twin markets across the
    boundary."""
    import random
    # 10 balanced slots, frac*10 of them to train: keeps the split fraction
    # while reusing the greedy positive balance.
    slots = 10
    fold = grouped_fold_assign(groups, y, slots, random.Random(seed))
    n_train_slots = int(round(frac * slots))
    return np.isin(fold, np.arange(n_train_slots))


def units(platform: str, spec_only: bool = True) -> list[dict]:
    """Return market-level analysis units {spec_score, disputed} for a platform.

    Both platforms are analyzed at the market (listing) level, since the rating
    grades each market's own listing. For Polymarket, if spec_only, disputes are
    restricted to specification_issue (no_fault / resolution_error dropped, clean
    kept); Kalshi disputes are all spec-flagged so spec_only is a no-op there.
    """
    out = []
    for r in graded_rows():
        if r["platform"] != platform:
            continue
        if (platform == "polymarket" and r["disputed"] and spec_only
                and r.get("classification") != "specification_issue"):
            continue
        out.append({"spec_score": r["spec_score"], "disputed": r["disputed"], "n": 1})
    return out


def binary_or(items: list[dict], c: int):
    """Odds ratio of dispute for below-grade (score>c) vs investment-grade
    (score<=c), with 95% CI. Returns (OR, lo, hi) or (nan,nan,nan)."""
    a = sum(1 for r in items if r["disputed"] and r["spec_score"] > c)
    b = sum(1 for r in items if r["disputed"] and r["spec_score"] <= c)
    d = sum(1 for r in items if not r["disputed"] and r["spec_score"] > c)
    e = sum(1 for r in items if not r["disputed"] and r["spec_score"] <= c)
    if min(a, b, d, e) == 0:
        return float("nan"), float("nan"), float("nan")
    OR = (a * e) / (b * d)
    se = math.sqrt(1 / a + 1 / b + 1 / d + 1 / e)
    return OR, math.exp(math.log(OR) - 1.96 * se), math.exp(math.log(OR) + 1.96 * se)


def perpoint_or(items: list[dict]):
    """OR per +1 specification-score point from a univariate logistic fit, with
    95% CI. Returns (OR, lo, hi)."""
    x = np.array([r["spec_score"] for r in items], float)
    y = np.array([r["disputed"] for r in items], float)
    X = np.column_stack([np.ones(len(x)), x])
    b = np.zeros(2)
    for _ in range(60):
        p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-9, 1 - 1e-9)
        W = p * (1 - p)
        H = (X.T * W) @ X + np.eye(2) * 1e-8
        b = b + np.linalg.solve(H, X.T @ (y - p))
    p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-9, 1 - 1e-9)
    cov = np.linalg.inv((X.T * (p * (1 - p))) @ X + np.eye(2) * 1e-8)
    se = math.sqrt(cov[1, 1])
    return math.exp(b[1]), math.exp(b[1] - 1.96 * se), math.exp(b[1] + 1.96 * se)
