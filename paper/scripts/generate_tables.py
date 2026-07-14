#!/usr/bin/env python3
"""
Generate the Table 1 disputed-market data-and-interpretation breakdown.

Reads:  data/full_grades.jsonl, data/fetched/uma_disputes_enriched.json,
        data/disputed_kalshi_events.json
Writes: paper/tables/overall_breakdown.tex

(The former rating-cross-tab / platform-comparison / volume-breakdown tables, built
on the deprecated GPT-4o-mini classification, were removed from the paper; their
code is archived in historical/generate_supplementary_tables.py.)

Usage:
    .venv/bin/python paper/scripts/generate_tables.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_tables")

ROOT = Path(__file__).resolve().parent.parent.parent
TABLES_DIR = ROOT / "paper" / "tables"

# Raw dispute sources for the data-and-interpretation breakdown (Table 1)
FULL_GRADES = ROOT / "data" / "full_grades.jsonl"
UMA_PATH = ROOT / "data" / "fetched" / "uma_disputes_enriched.json"
KALSHI_PATH = ROOT / "data" / "disputed_kalshi_events.json"

# UMA settlement-price sentinels (int256-encoded)
_UMA_YES = 10 ** 18
_UMA_UNKNOWN = 5 * 10 ** 17
_UMA_EARLY = -(2 ** 255)  # type(int256).min: "too early" / unresolvable


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "---"
    return f"{100 * n / total:.1f}\\%"


# ── Table 1: Overall breakdown ────────────────────────────────────────────

def _decode_uma_settlement(price) -> str:
    """Map a UMA settlement price to a human-readable outcome category."""
    try:
        p = int(price)
    except (ValueError, TypeError):
        return "definite"  # rare odd encodings: a non-sentinel definite price
    if p == 0:
        return "definite"          # No
    if p == _UMA_YES:
        return "definite"          # Yes
    if p == _UMA_UNKNOWN:
        return "unknown"           # 50/50 — unresolvable as written
    if p == _UMA_EARLY or p < 0:
        return "early"             # too-early / premature proposal
    return "definite"              # any other concrete price


def table_overall_breakdown(_unused=None) -> str:
    """Disputed-market data and how we interpret it.

    Polymarket disputes are broken out by the UMA oracle's *final settlement*
    (definite Yes/No, Unknown, or early-return); Kalshi disputes by the
    exchange's *post-resolution flag* (per-category vs.\\ note-only OTHER_INFO).
    The ``Material'' column marks the disputes we treat as reflecting a real
    specification problem. Computed on the graded analysis sample that feeds the
    later regression and grade tables.
    """
    # ── Polymarket: settlement outcome per disputed graded market ──────────
    uma = json.load(open(UMA_PATH))["disputes"]
    sett = {}
    for r in uma:
        pm = r.get("polymarket") or {}
        if pm.get("slug"):
            sett[pm["slug"]] = _decode_uma_settlement(r.get("settlement_price"))

    grades = [json.loads(l) for l in open(FULL_GRADES) if not json.loads(l).get("error")]
    pm_disp = [r for r in grades if r["platform"] == "polymarket" and r["disputed"] == 1]
    k_disp = [r for r in grades if r["platform"] == "kalshi" and r["disputed"] == 1]

    pm = Counter(sett.get(r["id"], "early") for r in pm_disp)  # unmatched ~ premature
    pm_definite = pm["definite"]
    pm_unknown = pm["unknown"]
    pm_early = pm["early"]
    pm_total = len(pm_disp)
    pm_material = pm_definite + pm_unknown

    # ── Kalshi: per-category flag (mutually exclusive by severity) vs. note-only
    kev = json.load(open(KALSHI_PATH))["events"]
    flags_by_ticker = {}
    for e in kev:
        fl = set(e.get("dispute_flags", []))
        for m in e.get("markets", []):
            flags_by_ticker[m["ticker"]] = fl
    # flags can co-occur; assign each market to its single most-severe category so
    # the rows sum to the substantive total. RULE_13 and PAUSED are merged: Rule
    # 13.1 is the pause-pending-review provision, so they are the same event type.
    PRIORITY = [
        ({"VOIDED"}, "Voided"),
        ({"REVERSED"}, "Reversed"),
        ({"RULE_13", "PAUSED"}, "Paused (Rule 13.1)"),
        ({"OUTCOME_REVIEW_COMMITTEE"}, "Review committee"),
        ({"REIMBURSEMENT"}, "Reimbursement"),
        ({"CLARIFICATION"}, "Clarification"),
    ]
    from _dispute_data import is_confirmed
    sub = Counter(); sub_conf = Counter()
    k_note = k_subst = 0
    for r in k_disp:
        fl = flags_by_ticker.get(r["id"], set())
        if fl == {"OTHER_INFO"}:
            k_note += 1
            continue
        k_subst += 1
        for keys, lab in PRIORITY:
            if fl & keys:
                sub[lab] += 1
                if is_confirmed(r):
                    sub_conf[lab] += 1
                break
    k_total = len(k_disp)

    # CONFIRMED subset (strictest): PM disputed >=2 rounds; Kalshi void/reverse/pause.
    # Counts are nested within each row: n >= material >= confirmed.
    pm_conf_def = sum(1 for r in pm_disp if is_confirmed(r) and sett.get(r["id"], "early") == "definite")
    pm_conf_unk = sum(1 for r in pm_disp if is_confirmed(r) and sett.get(r["id"], "early") == "unknown")
    pm_conf = pm_conf_def + pm_conf_unk
    k_conf = sum(sub_conf.values())

    total_disp = pm_total + k_total
    total_material = pm_material + k_subst
    total_confirmed = pm_conf + k_conf

    def cnt(x):
        return f"{x:,}" if x else "--"

    def row(label, n, base, material, confirmed):
        return (f"\\quad {label} & {n:,} & {_pct(n, base)} & "
                f"{cnt(material)} & {cnt(confirmed)} \\\\")

    kalshi_rows = [row(lab, sub[lab], k_total, sub[lab], sub_conf[lab])
                   for _, lab in PRIORITY if sub[lab]]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\caption{Disputed markets, by platform and severity}",
        r"\label{tab:overall_breakdown}",
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r" & $n$ & \% & Material & Confirmed \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Polymarket} --- UMA settlement outcome} \\",
        row("Resolved to a definite outcome (Yes/No)", pm_definite, pm_total, pm_definite, pm_conf_def),
        row("Resolved to Unknown (50/50)", pm_unknown, pm_total, pm_unknown, pm_conf_unk),
        row("Early return (event had not occurred)", pm_early, pm_total, 0, 0),
        f"\\quad\\textbf{{Polymarket total}} & \\textbf{{{pm_total:,}}} & & "
        f"\\textbf{{{cnt(pm_material)}}} & \\textbf{{{cnt(pm_conf)}}} \\\\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Kalshi} --- exchange post-resolution flag} \\",
        *kalshi_rows,
        row("Note-only (\\texttt{OTHER\\_INFO})", k_note, k_total, 0, 0),
        f"\\quad\\textbf{{Kalshi total}} & \\textbf{{{k_total:,}}} & & "
        f"\\textbf{{{cnt(k_subst)}}} & \\textbf{{{cnt(k_conf)}}} \\\\",
        r"\midrule",
        f"\\textbf{{Total}} & \\textbf{{{total_disp:,}}} & & "
        f"\\textbf{{{cnt(total_material)}}} & \\textbf{{{cnt(total_confirmed)}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\tabnote{\emph{Material} disputes reflect a specification problem; "
        r"\emph{Confirmed} disputes are those where the contract demonstrably failed. "
        r"The two are nested ($n\geq$ Material $\geq$ Confirmed). Percentages are "
        r"within-platform.}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / "overall_breakdown.tex"
    path.write_text(table_overall_breakdown())
    log.info("Wrote %s", path)


if __name__ == "__main__":
    main()
