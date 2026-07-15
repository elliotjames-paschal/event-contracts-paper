"""Substack Figure 1 prototype: the annotated suit-market contract card."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREEN, RED, INK, MUT = "#2e7d32", "#b3392f", "#222", "#666"

with plt.rc_context({"font.family": "serif"}):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    card = FancyBboxPatch((0.3, 0.4), 9.4, 9.2, boxstyle="round,pad=0.15",
                          facecolor="#fbfaf7", edgecolor="#b9b2a4", lw=1.2)
    ax.add_patch(card)

    ax.text(0.8, 9.15, "“Will Zelenskyy wear a suit before July?”",
            fontsize=14.5, fontweight="bold", color=INK, va="center")
    ax.text(0.8, 8.55, "Polymarket, 2025 — \$240 million traded",
            fontsize=10.5, color=MUT, va="center")
    ax.plot([0.8, 9.4], [8.2, 8.2], color="#d8d2c4", lw=1)

    ax.text(0.8, 7.65, "What the rules pinned down:", fontsize=10.5, color=MUT, style="italic")
    for i, t in enumerate([
            "Exact date window (May 22 – June 30, 2025 ET)",
            "Must be photographed or videotaped",
            "AI-generated or edited images excluded"]):
        ax.text(1.1, 7.15 - 0.62*i, r"$\checkmark$", fontsize=12, color=GREEN)
        ax.text(1.6, 7.15 - 0.62*i, t, fontsize=11.5, color=INK)

    ax.text(0.8, 4.9, "What the rules never said:", fontsize=10.5, color=MUT, style="italic")
    for i, t in enumerate([
            "What counts as a “suit” — the entire question",
            "Who decides: only “a consensus of credible reporting”"]):
        ax.text(1.1, 4.25 - 0.62*i, r"$\times$", fontsize=13, color=RED)
        ax.text(1.6, 4.25 - 0.62*i, t, fontsize=11.5, color=RED)

    ax.plot([0.8, 9.4], [2.9, 2.9], color="#d8d2c4", lw=1)
    ax.text(0.8, 2.15, "Risk grade (from the text alone):", fontsize=11, color=INK)
    ax.text(4.75, 2.15, "BB — below investment grade", fontsize=12.5, fontweight="bold", color=RED)
    ax.text(0.8, 1.25, "What happened:", fontsize=11, color=INK)
    ax.text(4.75, 1.25, "disputed 5 times — most in our data", fontsize=11,
            fontweight="bold", color=RED)

    from fs_style import add_footer
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    add_footer(fig)
    fig.savefig("/Users/andrewhall/event_contracts/substack/figs/fig1_contract.png",
                dpi=150, bbox_inches="tight", facecolor="white")
print("wrote fig1_contract.png")
