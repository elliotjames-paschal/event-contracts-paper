"""The AI-analyst pipeline figure: contract text -> structured spec -> 10 scores.

Provenance (rigor): every quoted string is verbatim from a repo artifact —
  * rules text: the market's UMA dispute record (data/fetched/uma_disputes_enriched.json)
  * standing instructions: paper/prompts/grading_system_prompt.txt (line 3)
  * the ten axis scores: data/full_grades.jsonl, id=will-zelenskyy-wear-a-suit-before-july
  * axis names: the ResolutionSpec schema (src/parser/schema.py)
  * grade band: spec_grade() bands (11 < 12 <= 15 -> CC)
Step 2's field VALUES are the contract's own language arranged into the real
schema fields (the stored per-market parse is not in this repo); the caption
discloses this. Score glosses paraphrase the grading rubric's 0-3 anchors.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUT, RED, GREEN, AMBER, BLUE = "#1c1c1c", "#6b6b6b", "#a33327", "#2e6b34", "#a06b00", "#2f5680"
CARD, EDGE, RULE = "#fbfaf6", "#c4bcab", "#e2dccd"

# (schema axis name, plain-English question, stored score, rubric-anchored gloss)
SCORES = [
    ("predicate_ambiguity",        "Is the question precisely defined?",   0, None),
    ("entity_ambiguity",           "Are the entities identifiable?",       0, None),
    ("temporal_precision",         "Is the time window pinned down?",      0, None),
    ("manipulation_susceptibility","Cost for one actor to move it",        2, "a feasible single-actor path exists"),
    ("outcome_concentration",      "How concentrated is settlement?",      2, "turns on one person's discrete act"),
    ("source_specification",       "Is a settlement source identified?",   2, "generic; no organization named"),
    ("source_quality",             "Is the source authoritative?",         2, "no primary source; secondary only"),
    ("edge_case_coverage",         "Are edge cases covered?",              1, "authenticity yes; key scenario no"),
    ("headline_rules_alignment",   "Does headline match the rules?",       0, None),
    ("governing_body_engagement",  "Is a governing body engaged?",         3, "none exists for the domain"),
]

with plt.rc_context({"font.family": "serif"}):
    fig, ax = plt.subplots(figsize=(13.6, 7.4))
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 7.4); ax.axis("off")

    def card(x, y, w, h):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.09",
                                    facecolor=CARD, edgecolor=EDGE, lw=1.1))

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=22, color="#8a8577", lw=1.6))

    ax.text(0.25, 7.12, "An AI analyst reads every contract — and shows its work",
            fontsize=15, fontweight="bold", color=INK)

    # ── STEP 1: the contract ────────────────────────────────────────────────
    card(0.25, 2.15, 3.6, 4.45)
    ax.text(0.5, 6.28, "STEP 1 — THE CONTRACT", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(0.5, 5.88, "“Will Zelenskyy wear a suit\nbefore July?”", fontsize=11.5,
            fontweight="bold", color=INK, va="top", linespacing=1.25)
    ax.text(0.5, 4.98, "Polymarket · $242M traded", fontsize=8.5, color=MUT, va="top")
    rules = ("Resolves Yes if Zelenskyy is\n"
             "“photographed or videotaped wearing\n"
             "a suit between May 22 and June 30,\n"
             "2025 ET.” “The images or video must\n"
             "be authentic, not the result of\n"
             "artificial intelligence or video\n"
             "editing.” “The resolution source will\n"
             "be a consensus of credible reporting.”")
    ax.text(0.5, 4.6, rules, fontsize=8.8, color="#3d3d3d", va="top",
            linespacing=1.45, style="italic")
    # standing instructions chip
    card(0.25, 0.6, 3.6, 1.25)
    ax.text(0.5, 1.62, "THE AI'S STANDING INSTRUCTIONS (verbatim excerpt)",
            fontsize=7, color=MUT)
    ax.text(0.5, 1.32, "“Be conservative. When the contract is\nambiguous, flag the ambiguity — do not\npaper over it.”",
            fontsize=8.7, color=INK, va="top", linespacing=1.3)

    arrow(4.0, 4.6, 4.35)

    # ── STEP 2: structured spec ─────────────────────────────────────────────
    card(4.75, 0.6, 3.85, 6.0)
    ax.text(5.0, 6.28, "STEP 2 — PARSED INTO A", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(5.0, 6.0, "STRUCTURED SPEC", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(7.1, 6.0, "(schema v0.5)", fontsize=8, color=MUT)
    fields = [
        ("predicate",       "Zelenskyy wearing “a suit”\n(operative term undefined)", RED),
        ("entities",        "Volodymyr Zelenskyy — person,\nuniquely identified", GREEN),
        ("temporal_bounds", "2025-05-22 → 2025-06-30,\ntimezone ET, stated in rules", GREEN),
        ("source_hierarchy","named organizations: none;\n“a consensus of credible\nreporting” (methodology only)", RED),
        ("edge_cases",      "media authenticity: addressed\nmeaning of “suit”: unaddressed", RED),
    ]
    yy = 5.55
    for name, val, c in fields:
        ax.text(5.0, yy, name, fontsize=8.3, color=MUT, va="top", family="monospace")
        nlines = val.count("\n") + 1
        ax.text(5.0, yy - 0.28, val, fontsize=9.2, color=c, va="top", linespacing=1.25)
        yy -= 0.52 + 0.245 * nlines

    arrow(8.75, 9.35, 4.35)

    # ── STEP 3: the ten scores ──────────────────────────────────────────────
    card(9.5, 0.6, 3.85, 6.0)
    ax.text(9.75, 6.28, "STEP 3 — TEN RISK SCORES", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(9.75, 6.0, "0 = no concern · 3 = severe", fontsize=8, color=MUT)
    yy = 5.62
    for name, label, sc, gloss in SCORES:
        c = RED if sc >= 2 else (AMBER if sc == 1 else GREEN)
        for d in range(3):
            ax.scatter(9.85 + 0.185 * d, yy, s=44, facecolor=(c if d < sc else "white"),
                       edgecolor=c, linewidth=0.9, zorder=3)
        ax.text(10.62, yy, str(sc), fontsize=9.5, color=c, fontweight="bold", va="center")
        ax.text(10.88, yy + 0.075, label, fontsize=8.4, color=INK, va="center")
        sub = name + (" — " + gloss if gloss else "")
        ax.text(10.88, yy - 0.145, sub, fontsize=6.4,
                color=(c if gloss else "#9a9a9a"), va="center", family="monospace")
        yy -= 0.462
    ax.plot([9.75, 13.1], [yy + 0.21, yy + 0.21], color=RULE, lw=1)
    ax.text(9.75, yy - 0.06, "Sum 12/30  →  grade CC (disclosed bands)",
            fontsize=10, fontweight="bold", color=RED, va="center")
    ax.text(9.75, yy - 0.32, "In fact: challenged 5× at the UMA oracle — most in the sample",
            fontsize=7.6, color=MUT, va="center")

    # footer: provenance + link to the prediction model
    ax.text(0.25, 0.22, "The ten axis scores are also the inputs to the dispute-prediction model (Figs. 2–3).   "
            "All quoted text and all scores are verbatim from the replication data "
            "(market rules; grading prompt; stored grades).",
            fontsize=7.6, color=MUT)

    fig.savefig("/Users/andrewhall/event_contracts/substack/figs/fig_pipeline.png",
                dpi=150, bbox_inches="tight")
print("wrote fig_pipeline.png")
