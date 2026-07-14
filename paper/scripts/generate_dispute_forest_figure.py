#!/usr/bin/env python3
"""Figure (familiar format): forest / coefficient plot of the odds ratio that a
contract is disputed, by platform, for two specifications:
  (i)  per +1 specification-score point (continuous logit), and
  (ii) below investment grade (score > 7) vs. investment grade.
Kalshi at the series level, Polymarket at the market level (spec-issue disputes).
A point at OR=1 means no association. See _dispute_data.py.

Reads:  data/full_grades.jsonl   (never modified)
Writes: paper/figures/dispute_forest.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dispute_data import units, binary_or, perpoint_or

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
COLORS = {"kalshi": "#33618f", "polymarket": "#2a9d8f"}


def main() -> None:
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    data = {k: units(k, spec_only=True) for k in ("kalshi", "polymarket")}
    specs = [("Per +1 specification-score point", "perpoint"),
             ("Below investment grade (vs. investment grade)", "binary")]
    DODGE = 0.17
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "serif", "font.size": 10,
                         "axes.linewidth": 0.6}):
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        ax.axvline(1.0, color="#888", lw=0.9, ls=(0, (4, 3)), zorder=1)
        for gi, (spec_label, kind) in enumerate(specs):
            for plat, off in (("kalshi", -DODGE), ("polymarket", +DODGE)):
                OR, lo, hi = (perpoint_or(data[plat]) if kind == "perpoint"
                              else binary_or(data[plat], 7))
                yy = gi + off
                ax.plot([lo, hi], [yy, yy], "-", color=COLORS[plat], lw=1.7, zorder=2)
                ax.plot(OR, yy, "o", color=COLORS[plat], ms=6, zorder=3)
                ax.text(hi * 1.03, yy, f"{OR:.2f} [{lo:.2f}, {hi:.2f}]",
                        va="center", fontsize=7.5, color="#444")
        ax.set_yticks(range(len(specs)))
        ax.set_yticklabels([s[0] for s in specs], fontsize=9)
        ax.set_ylim(-0.55, len(specs) - 0.45)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.FixedLocator([0.7, 1, 1.5, 2, 3]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(["0.7", "1", "1.5", "2", "3"]))
        ax.set_xlim(0.6, 3.6)
        ax.set_xlabel("Odds ratio that the contract is disputed (95% CI)",
                      fontsize=9.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(handles=[Line2D([0], [0], color=COLORS["kalshi"], marker="o",
                                  lw=1.7, label="Kalshi (series)"),
                           Line2D([0], [0], color=COLORS["polymarket"], marker="o",
                                  lw=1.7, label="Polymarket (markets)")],
                  frameon=False, fontsize=8.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "dispute_forest.pdf", bbox_inches="tight")
    print(f"wrote {FIG_DIR / 'dispute_forest.pdf'}")
    for spec_label, kind in specs:
        for plat in ("kalshi", "polymarket"):
            OR, lo, hi = (perpoint_or(data[plat]) if kind == "perpoint"
                          else binary_or(data[plat], 7))
            print(f"  {spec_label:24s} {plat:11s} OR={OR:.2f} [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
