# You Can Tell Which Prediction Markets Will Blow Up — Before Anyone Trades

*The fine print of an event contract predicts whether the market ends in a fight. That's measurable, it's cheap to check, and everyone — traders, platforms, regulators — can use it.*

<!-- DRAFT v1 — numbers pulled from the analysis pipeline in event-contracts-paper (grouped holdouts). Figures referenced below live in paper/figures/. -->

---

In 2025, roughly $240 million changed hands on a single Polymarket question: **"Will Zelenskyy wear a suit before July?"**

He wore *something* — at a NATO summit, in late June. Jacket, collared shirt, no tie, military styling. Was it a suit? The market's rules didn't say what a suit was, or who got to decide. What followed was weeks of chaos: the market's oracle was challenged five separate times — the most of any market in our data — the resolution flipped, and traders on both sides came away convinced they'd been robbed.

Here's the thing: you didn't need to know anything about menswear to see this coming. You just needed to read the contract. The rules were scrupulous about the wrong details — they pinned the date window to the hour and even preemptively excluded AI-generated images — but never defined the one word the entire market turned on, and named as the settlement authority only "a consensus of credible reporting." No definition, no designated decider, for a judgment call everyone should have known was coming. Our grading system — which reads *only the text of a market's rules* — puts it in the bottom tier of contract quality: **CC**, on a scale from A down.

That's the finding, in one sentence: **the text of an event contract predicts whether the market will end in a dispute — before a single dollar is traded.**

## The claim, with numbers

We graded roughly 9,800 markets from Polymarket and Kalshi on ten dimensions of contract quality: how precisely the core question is defined, whether the entities involved are identifiable, whether the time window is pinned down, whether a settlement source is *named* (not just gestured at), how authoritative that source is, whether edge cases are covered, whether the headline matches the fine print, and so on. Each dimension is scored 0–3 by an AI reading the contract text. (We checked the AI against human graders on 300 hand-graded contracts; when the two materially disagreed, a blind referee sided with the AI more often than with the human.)

Then we asked whether those scores predict *actual* dispute outcomes the grader never saw — oracle challenges on Polymarket, voided or reversed resolutions on Kalshi — for markets held out from the model entirely.

They do. Given one market that later blew up in a confirmed dispute and one that resolved cleanly, the model picks the future blow-up **about 85% of the time** (out-of-sample AUC 0.83–0.85, pooled; ~0.90 on Polymarket's organic oracle disputes). And when we bin markets into letter grades, dispute risk climbs monotonically down the ladder in held-out data — the bottom grade's confirmed-dispute rate is roughly **thirty times** the top grade's.

**[FIGURE: grade ladder / dispute-rate-by-grade]**

Two honest limits, stated once. First, these are *relative* risks — most markets resolve fine everywhere; the grades tell you which ones are many times more likely to end badly, not that any given market will. Second, prediction works on disputes that arise from the contract itself. Frivolous oracle challenges — someone disputing a clearly-correct outcome to grief the system — are noise, and no reading of the fine print will forecast them.

## What actually makes a contract blow up

The predictive signal isn't spread evenly across the ten dimensions, and where it concentrates is the practical payoff. The flaws that forecast disputes:

**An undefined word carrying the whole question.** "Suit." "Ceasefire." "Invade." "Recession." If the payoff hinges on a term two reasonable people can read differently, the market is a dispute waiting for a news cycle. This is the single strongest textual predictor.

**No named settlement source.** There's a world of difference between "according to the Bureau of Labor Statistics release of June 6" and "according to credible reporting." A contract that doesn't say *who* decides has quietly reserved the fight for later. Vague sourcing is the other heavyweight predictor — and it's also the cheapest thing to fix.

**Ambiguous entities and fuzzy time windows.** *Which* index, *which* election, *whose* announcement, in *what* timezone? Each unresolved reference is another door a dispute can walk through.

Just as interesting is what *doesn't* predict disputes: whether a market is theoretically manipulable — resolvable by one person's action or announcement. Markets like the Trump–Zelenskyy handshake-length series score terribly on manipulation risk, but manipulation-prone markets usually resolve *unambiguously* (the announcement happens or it doesn't). Disputes feed on interpretive wiggle room, not on who controls the outcome. Regulation aimed at manipulation and regulation aimed at dispute-proneness are aiming at different targets — the fine print is what tells you which is which.

## The money mostly knows — but not entirely

Grade the representative sample and you get a market-quality census: **about 84% of markets are investment grade** (our line for "the dispute odds are no worse than the market average"), and they carry **91% of all dollar volume**. Traders, in aggregate, already steer capital toward well-specified markets — a real vote of confidence in the ecosystem.

But that leaves the tail: 16% of markets sit below the line, they hold about **$71 million** in volume in our sample alone, and they're where **85% of the confirmed blow-ups live**. The suit market — $240 million, grade CC — is a reminder that when a badly-specified market does catch fire, the dollars at stake can be enormous.

**[FIGURE: share of markets vs. share of volume by grade]**

## Why this is worth acting on

**If you trade:** the fine print is alpha. Before you take a position, the question isn't just "will X happen?" — it's "will this market agree on whether X happened?" A bottom-grade market carries a second, uncompensated risk: that you're right about the world and still lose the resolution fight. Reading the rules takes two minutes; a grade takes seconds.

**If you run a platform:** this is a pre-listing screen that costs pennies per market. Every flaw the model keys on has a mechanical fix — define the operative term, name the settlement source and a backup, state the timezone, write down the tie-breaker. A market can be re-drafted from CC to A *before launch*, at the cost of one editing pass. The dispute that never happens saves the oracle fees, the support tickets, and the Reddit thread calling your platform rigged.

**If you regulate:** the CFTC is right now deciding how to police event contracts, and the emerging approach leans on category lines — sports injuries yes or no, assassination markets banned, and so on. Categories are blunt. Contract quality is measurable, auditable, and continuous: a disclosed grading standard ("markets must meet specification grade B, here are the criteria") targets the actual failure mode — contracts that can't deliver what they promise — without adjudicating which topics are virtuous. Our grades line up with the concerns in the CFTC's own advisory framework, but they turn those concerns into a number anyone can reproduce from the public rules text.

## The cheapest fix in finance

Event contracts are the rare financial product where the main risk factor is a *writing problem*. Nobody can redraft the economy before an inflation print, but anyone can add a sentence to a market's rules naming who decides what counts as a suit. The disputes that burn traders and embarrass platforms are, to a striking degree, forecastable from the text — which means they were preventable in the text.

That's the practical bottom line. The suit market's authors spent a sentence ruling out deepfakes and zero sentences defining "suit." One was a hypothetical risk; the other was a $240 million fire drill anyone could have flagged in advance — and now, cheaply and at scale, anyone can. The contracts tell you where the fires will be.

---

*Based on joint work analyzing ~9,800 graded event contracts across Polymarket and Kalshi. Full methodology, data, and replication code: [link to repo/paper].*
