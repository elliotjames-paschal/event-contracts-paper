#!/usr/bin/env python3
"""Generate paper figures from pilot results.

Reads:  data/pilot_v07_results.json
Writes: paper/figures/reliability.pdf        (per-axis agreement bars)
        paper/figures/confusion_by_axis.pdf  (per-axis 4x4 score confusion)

Figures regenerate on compile via paper/.latexmkrc.

Usage:
    PYTHONPATH=src .venv/bin/python paper/scripts/generate_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "data" / "pilot_v07_results.json"
FIG_DIR = ROOT / "paper" / "figures"

AXES = [
    "predicate_ambiguity", "entity_ambiguity", "temporal_precision",
    "manipulation_susceptibility", "outcome_concentration",
    "source_specification", "source_quality", "edge_case_coverage",
    "headline_rules_alignment", "governing_body_engagement",
]
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
}


def quadratic_weighted_kappa(human: list[int], llm: list[int], k: int = 4) -> float:
    """QWK for ordinal scores in 0..k-1."""
    O = np.zeros((k, k))
    for h, l in zip(human, llm):
        O[h, l] += 1
    if O.sum() == 0:
        return float("nan")
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            w[i, j] = (i - j) ** 2 / (k - 1) ** 2
    hist_h = O.sum(axis=1)
    hist_l = O.sum(axis=0)
    E = np.outer(hist_h, hist_l) / O.sum()
    denom = (w * E).sum()
    return 1 - (w * O).sum() / denom if denom else float("nan")


def main() -> None:
    res = [r for r in json.loads(RESULTS.read_text()) if not r["error"]]
    n_contracts = len(res)
    exact, within1 = {}, {}
    pooled = []
    for a in AXES:
        pairs = [(r["human"][a], r["llm"][a]) for r in res if r["human"][a] is not None]
        n = len(pairs)
        exact[a] = sum(x == y for x, y in pairs) / n
        within1[a] = sum(abs(x - y) <= 1 for x, y in pairs) / n
        pooled += pairs
    exact["__overall__"] = sum(x == y for x, y in pooled) / len(pooled)
    within1["__overall__"] = sum(abs(x - y) <= 1 for x, y in pooled) / len(pooled)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_adjudication_fig(exact, within1, n_contracts)


def _adj_llm_pct(path: Path, severe_only: bool = False) -> dict[str, tuple[float, int]]:
    import csv
    rows = list(csv.DictReader(open(path)))
    if severe_only:  # |diff| >= 2 only (drop the diff=1 "three-pivotal" cells)
        rows = [r for r in rows if r.get("stratum") == "severe"]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["axis"], []).append(r["preferred"])
    out = {a: (100 * sum(p == "llm" for p in v) / len(v), len(v)) for a, v in by.items()}
    out["__overall__"] = (100 * sum(r["preferred"] == "llm" for r in rows) / len(rows), len(rows))
    return out


def make_adjudication_fig(exact: dict, within1: dict, n_contracts: int) -> None:
    """Two-panel Figure 1: (a) per-axis human--LLM agreement (exact + within-1)
    over all graded contracts; (b) when they materially disagree (|diff|>=2),
    the share where a blind adjudication prefers the LLM grade. Shared axis
    order so each row is the same axis across both panels."""
    from matplotlib.patches import Patch
    cons_path = ROOT / "data" / "adjudication_final_verdicts.csv"
    adj = _adj_llm_pct(cons_path, severe_only=True) if cons_path.exists() else {}

    order = sorted(AXES, key=lambda a: -exact[a]) + ["__overall__"]
    labels = [LABELS.get(a, a) for a in order[:-1]] + ["Overall"]
    y = np.arange(len(order))[::-1]
    # panel (b): LLM preferred = dark blue (the headline), human = light blue
    # (the empty n=0 row reads as blank since no bar is drawn)
    LLM, HUM, EX, W1 = "#33618f", "#c9d6e3", "#33618f", "#c9d6e3"

    with plt.rc_context({"font.family": "serif", "font.size": 10,
                         "axes.linewidth": 0.6}):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)

        # ── Panel (a): agreement ──
        axA.barh(y, [within1[a] * 100 for a in order], color=W1, height=0.72)
        axA.barh(y, [exact[a] * 100 for a in order], color=EX, height=0.72)
        for yi, a in zip(y, order):
            axA.text(exact[a] * 100 - 1.5, yi, f"{exact[a]*100:.0f}", va="center",
                     ha="right", color="white", fontsize=7.5)
            axA.text(within1[a] * 100 + 1.5, yi, f"{within1[a]*100:.0f}",
                     va="center", fontsize=7, color="#777")
        axA.set_xlim(0, 100)
        axA.set_title(f"(a) Human--LLM agreement (n={n_contracts})", fontsize=10)
        axA.set_xlabel("Agreement across all contracts (%)", fontsize=9)
        axA.set_yticks(y); axA.set_yticklabels(labels)
        axA.spines[["top", "right", "left"]].set_visible(False)

        # ── Panel (b): adjudication preference on |diff|>=2 ──
        for yi, a in zip(y, order):
            v, n = adj.get(a, (float("nan"), 0))
            if v == v:
                axB.barh(yi, v, color=LLM, height=0.72)
                axB.barh(yi, 100 - v, left=v, color=HUM, height=0.72)
                axB.text(v - 1.5, yi, f"{v:.0f}", va="center", ha="right",
                         color="white", fontsize=7.5)  # white text on dark blue
                axB.text(101, yi, f"{n}", va="center", fontsize=7, color="#777")
            else:
                axB.text(2, yi, "n=0", va="center", fontsize=7, color="#999")
        axB.axvline(50, color="#444", lw=0.7, ls=(0, (4, 3)))
        axB.set_xlim(0, 100)
        axB.set_title(r"(b) Preference when $|\Delta|\geq2$", fontsize=10)
        axB.set_xlabel("Share preferring the LLM grade (%)", fontsize=9)
        axB.spines[["top", "right", "left"]].set_visible(False)

        # color key as one figure-level legend, above the panels (no bar overlap)
        fig.legend(handles=[Patch(color=EX, label="(a) exact"),
                            Patch(color=W1, label="(a) within 1 point"),
                            Patch(color=LLM, label="(b) LLM preferred"),
                            Patch(color=HUM, label="(b) human preferred")],
                   loc="upper center", ncol=4, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 1.06))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(FIG_DIR / "adjudication.pdf", bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'adjudication.pdf'} (two-panel: agreement + adjudication)")


if __name__ == "__main__":
    main()
