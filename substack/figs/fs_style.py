"""Free Systems standard footer for the event-contracts figures.

Same layout as fundraising_emails/scripts/fig_style.py: light-circle logo
bottom-left, freesystems.substack.com center, right-aligned data credit.
"""
from pathlib import Path

import matplotlib.pyplot as plt

LOGO = Path("/Users/andrewhall/Logos/logo-light-circle.png")
CREDIT = "Data: Polymarket & Kalshi (Hall & Paschal)"


def add_footer(fig, credit: str = CREDIT, height: float = 0.075,
               facecolor: str = "white") -> None:
    footer_ax = fig.add_axes([0, 0, 1, height])
    footer_ax.set_facecolor(facecolor)
    footer_ax.set_xticks([])
    footer_ax.set_yticks([])
    footer_ax.grid(False)
    for spine in footer_ax.spines.values():
        spine.set_visible(False)
    if LOGO.exists():
        logo_img = plt.imread(LOGO)
        lh = height * 0.9
        logo_ax = fig.add_axes([0.02, height * 0.08, 0.20, lh])
        logo_ax.imshow(logo_img)
        logo_ax.axis("off")
        logo_ax.grid(False)
    footer_ax.text(0.24, 0.5, "freesystems.substack.com", ha="left", va="center",
                   fontsize=10, fontweight="semibold", family="sans-serif",
                   transform=footer_ax.transAxes)
    footer_ax.text(0.975, 0.5, credit, ha="right", va="center", fontsize=7.8,
                   fontweight="semibold", family="sans-serif", color="#444444",
                   transform=footer_ax.transAxes)
