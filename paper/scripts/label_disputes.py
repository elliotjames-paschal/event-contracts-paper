#!/usr/bin/env python3
"""
Classify disputed prediction-market contracts into:
  - specification_issue : contract text was flawed
  - resolution_error    : contract was clear, resolver submitted wrong answer
  - no_fault            : contract was clear, resolution was correct, dispute was unsuccessful
  - ambiguous           : mixed signals, can't confidently classify

Uses GPT-4o-mini for consistent classification across all disputes.

Data sources (relative to repo root):
  - data/fetched/uma_disputes_enriched.json   (Polymarket UMA disputes)
  - data/disputed_audit_results.json          (10-axis ratings)
  - data/disputed_kalshi_events.json          (Kalshi disputed events)

Output:
  - paper/data/labeled_disputes.json

Usage:
    .venv/bin/python paper/scripts/label_disputes.py [--limit N] [--no-cache]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
PAPER_DIR = ROOT / "paper"
CACHE_DIR = PAPER_DIR / "data" / ".cache"
OUTPUT_PATH = PAPER_DIR / "data" / "labeled_disputes.json"

UMA_PATH = DATA_DIR / "fetched" / "uma_disputes_enriched.json"
AUDIT_PATH = DATA_DIR / "disputed_audit_results.json"
KALSHI_PATH = DATA_DIR / "disputed_kalshi_events.json"

# ── Logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("label_disputes")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_WORKERS = 30

VALID_CLASSIFICATIONS = (
    "specification_issue", "resolution_error", "no_fault", "ambiguous"
)

SYSTEM_PROMPT = """\
You are a prediction-market contract analyst. Your job is to classify why a \
prediction-market dispute occurred.

CATEGORIES (pick exactly one):

1. **specification_issue** — The contract text itself was flawed. This includes:
   - Ambiguous or vague resolution criteria
   - Missing edge cases or undefined terms
   - Contradictory rules
   - Unresolvable conditions (e.g. no reliable data source exists)
   - Voided/cancelled markets (contract was too broken to resolve)
   - UMA "early return" settlements (settled to Unknown because contract couldn't be resolved)
   - Markets where the question didn't match the resolution rules
   - Overly broad or unclear predicates

2. **resolution_error** — The contract was clear and well-specified, but the \
proposer/resolver submitted the wrong answer. The dispute CORRECTED a mistake. Signs:
   - The dispute flipped the outcome (proposer said Yes, final answer was No, or vice versa)
   - Clear evidence the original proposal was simply wrong
   - The contract rules unambiguously point to a different answer than what was proposed
   - The dispute was about a factual mistake, not an interpretation disagreement

3. **no_fault** — The contract was clear, AND the resolution was correct. Someone \
filed a dispute but the original resolution stood. The dispute was unsuccessful \
or frivolous — there was no error in either the contract or the resolution. Signs:
   - The dispute did NOT flip the outcome (proposed and settled to the same answer)
   - The contract rules clearly support the resolution that was given
   - No evidence of ambiguity in the contract language
   - The disputer appears to have been wrong, not the resolver

4. **ambiguous** — Mixed signals. The dispute might involve both a specification \
weakness AND a resolution disagreement, or there isn't enough information to \
confidently classify.

IMPORTANT HEURISTICS:
- If the settlement was to "Unknown" or an early-return price, classify as \
specification_issue (the contract was too broken to resolve).
- If the dispute flipped the outcome AND the contract rules are clear, classify \
as resolution_error.
- If the dispute flipped the outcome BUT the flip happened because of ambiguous \
contract language, classify as specification_issue.
- If the dispute did NOT flip the outcome AND the contract rules clearly support \
the resolution, classify as no_fault.
- If the dispute did NOT flip the outcome BUT the contract has ambiguous language, \
classify as specification_issue (the ambiguity likely motivated the dispute even \
though the resolution stood).
- Voided markets are always specification_issue.
- Markets with PAUSED, RULE_13, OUTCOME_REVIEW_COMMITTEE, or REIMBURSEMENT \
flags strongly suggest specification_issue.
- Markets with CLARIFICATION flags suggest the original contract needed \
clarification — lean toward specification_issue.
- REVERSED markets are specification_issue.

Respond with a JSON object:
{
  "classification": "specification_issue" | "resolution_error" | "no_fault" | "ambiguous",
  "confidence": <float 0.0-1.0>,
  "rationale": "<1-3 sentence explanation>"
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────

def _decode_uma_price(raw_price: str | int) -> str:
    """Decode UMA settlement/proposed price to human-readable string."""
    try:
        p = int(raw_price)
    except (ValueError, TypeError):
        return f"unknown ({raw_price})"
    if p == 0:
        return "No"
    if p == 1_000_000_000_000_000_000:
        return "Yes"
    if p == 500_000_000_000_000_000:
        return "50/50 (Unknown)"
    if p < 0:
        return "Early Return / Unknown"
    return f"other ({p})"


def _build_polymarket_message(dispute: dict, audit: dict | None) -> str:
    """Build user message for a Polymarket dispute."""
    pm = dispute["polymarket"]
    parts = [
        "PLATFORM: Polymarket (UMA Oracle)",
        f"QUESTION: {pm.get('question', 'N/A')}",
        f"SLUG: {pm.get('slug', 'N/A')}",
    ]

    desc = pm.get("description", "")
    if desc:
        # Truncate very long descriptions
        if len(desc) > 1500:
            desc = desc[:1500] + "... [truncated]"
        parts.append(f"RESOLUTION RULES:\n{desc}")

    outcomes = pm.get("outcomes", "")
    if outcomes:
        parts.append(f"OUTCOMES: {outcomes}")

    outcome_prices = pm.get("outcome_prices", "")
    if outcome_prices:
        parts.append(f"FINAL OUTCOME PRICES: {outcome_prices}")

    parts.append(f"PROPOSED RESOLUTION: {_decode_uma_price(dispute.get('proposed_price'))}")
    parts.append(f"FINAL SETTLEMENT: {_decode_uma_price(dispute.get('settlement_price'))}")
    parts.append(f"DISPUTE FLIPPED OUTCOME: {dispute.get('dispute_flipped', 'unknown')}")

    return "\n".join(parts)


def _build_kalshi_message(market: dict, event: dict, audit: dict | None) -> str:
    """Build user message for a Kalshi dispute."""
    parts = [
        "PLATFORM: Kalshi",
        "NOTE: Kalshi resolves markets internally — there is no public dispute/flip "
        "mechanism. We cannot observe whether a resolution was corrected. Therefore, "
        "do NOT classify as resolution_error. Only use specification_issue, no_fault, "
        "or ambiguous for Kalshi markets.",
        f"QUESTION: {market.get('title', 'N/A')}",
        f"TICKER: {market.get('ticker', 'N/A')}",
        f"EVENT: {event.get('title', 'N/A')}",
    ]

    rules_primary = market.get("rules_primary", "")
    if rules_primary:
        if len(rules_primary) > 1500:
            rules_primary = rules_primary[:1500] + "... [truncated]"
        parts.append(f"RULES (PRIMARY):\n{rules_primary}")

    rules_secondary = market.get("rules_secondary", "")
    if rules_secondary:
        if len(rules_secondary) > 500:
            rules_secondary = rules_secondary[:500] + "... [truncated]"
        parts.append(f"RULES (SECONDARY):\n{rules_secondary}")

    flags = event.get("dispute_flags", [])
    if flags:
        parts.append(f"DISPUTE FLAGS: {', '.join(flags)}")

    important_info = event.get("important_info_markdown", "")
    if important_info:
        if len(important_info) > 800:
            important_info = important_info[:800] + "... [truncated]"
        parts.append(f"IMPORTANT INFO / CLARIFICATIONS:\n{important_info}")

    result = market.get("result", "")
    settlement = market.get("settlement_value_dollars", "")
    parts.append(f"RESULT: {result or 'not settled'}")
    if settlement is not None and settlement != "":
        parts.append(f"SETTLEMENT VALUE: ${settlement}")

    return "\n".join(parts)


# ── Cache ─────────────────────────────────────────────────────────────────

def _cache_key(platform: str, market_id: str) -> str:
    raw = f"{platform}|{market_id}|{SYSTEM_PROMPT}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_cache(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _write_cache(key: str, result: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(result))


# ── LLM classification ───────────────────────────────────────────────────

def classify(client: OpenAI, user_message: str, platform: str,
             market_id: str, *, use_cache: bool = True) -> dict:
    """Classify a single dispute via GPT-4o-mini."""
    key = _cache_key(platform, market_id)

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            text = resp.choices[0].message.content
            result = json.loads(text)

            # Validate
            if result.get("classification") not in VALID_CLASSIFICATIONS:
                raise ValueError(f"Invalid classification: {result.get('classification')}")

            if use_cache:
                _write_cache(key, result)
            return result

        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                log.warning("Attempt %d failed for %s/%s: %s — retrying in %ds",
                            attempt + 1, platform, market_id, e, wait)
                time.sleep(wait)
            else:
                log.error("All attempts failed for %s/%s: %s", platform, market_id, e)
                return {
                    "classification": "ambiguous",
                    "confidence": 0.0,
                    "rationale": f"LLM classification failed: {e}",
                }


# ── Load data ─────────────────────────────────────────────────────────────

def _pick_best_dispute(disputes: list[dict]) -> dict:
    """Pick the most informative dispute for a slug.

    Priority:
      1. A dispute that flipped the outcome (most informative)
      2. If none flipped, take the last dispute (most recent)
    """
    flipped = [d for d in disputes if d.get("dispute_flipped")]
    if flipped:
        return flipped[0]
    return disputes[-1]


def load_polymarket_disputes() -> list[dict]:
    """Load and deduplicate Polymarket disputes by slug."""
    log.info("Loading Polymarket UMA disputes from %s", UMA_PATH)
    with open(UMA_PATH) as f:
        data = json.load(f)

    # Group all disputes by slug
    by_slug: dict[str, list[dict]] = {}

    for d in data["disputes"]:
        pm = d.get("polymarket")
        if pm is None:
            continue
        slug = pm.get("slug", "")
        if not slug:
            continue
        by_slug.setdefault(slug, []).append(d)

    # Pick the best dispute per slug
    disputes = []
    multi_count = 0
    for slug, slug_disputes in by_slug.items():
        best = _pick_best_dispute(slug_disputes)
        best["_dispute_count"] = len(slug_disputes)
        disputes.append(best)
        if len(slug_disputes) > 1:
            multi_count += 1

    log.info("  %d matched disputes → %d unique slugs (%d with multiple disputes)",
             sum(len(v) for v in by_slug.values()), len(disputes), multi_count)
    return disputes


def load_kalshi_disputes() -> list[dict]:
    """Load Kalshi disputed markets with their parent events."""
    log.info("Loading Kalshi disputed events from %s", KALSHI_PATH)
    with open(KALSHI_PATH) as f:
        data = json.load(f)

    results = []
    for event in data["events"]:
        for market in event["markets"]:
            results.append({"market": market, "event": event})

    log.info("  %d events → %d markets", len(data["events"]), len(results))
    return results


def load_audit_results() -> tuple[dict, dict]:
    """Load audit ratings. Returns (kalshi_by_ticker, pm_by_slug)."""
    log.info("Loading audit results from %s", AUDIT_PATH)
    with open(AUDIT_PATH) as f:
        data = json.load(f)

    kalshi = {r["ticker"]: r for r in data["kalshi"]["results"]}
    pm = {r["slug"]: r for r in data["polymarket"]["results"]}

    log.info("  Kalshi: %d rated markets, Polymarket: %d rated slugs",
             len(kalshi), len(pm))
    return kalshi, pm


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classify prediction-market disputes")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max disputes to classify per platform (0=all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable cache, re-classify everything")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Concurrent workers (default: {MAX_WORKERS})")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load data
    pm_disputes = load_polymarket_disputes()
    kalshi_disputes = load_kalshi_disputes()
    kalshi_audit, pm_audit = load_audit_results()

    if args.limit:
        pm_disputes = pm_disputes[:args.limit]
        kalshi_disputes = kalshi_disputes[:args.limit]
        log.info("Limiting to %d disputes per platform", args.limit)

    use_cache = not args.no_cache
    all_results: list[dict] = []

    # ── Polymarket ────────────────────────────────────────────────────
    log.info("Classifying %d Polymarket disputes...", len(pm_disputes))

    def _classify_pm(dispute):
        slug = dispute["polymarket"]["slug"]
        audit = pm_audit.get(slug)
        msg = _build_polymarket_message(dispute, audit)
        result = classify(client, msg, "polymarket", slug, use_cache=use_cache)
        return {
            "id": slug,
            "platform": "polymarket",
            "question": dispute["polymarket"].get("question", ""),
            "volume": _safe_float(dispute["polymarket"].get("volume")),
            "rating": audit["rating"] if audit else None,
            "dimension_scores": audit.get("dimension_scores") if audit else None,
            "classification": result["classification"],
            "confidence": result.get("confidence", 0.0),
            "rationale": result.get("rationale", ""),
            "dispute_flipped": dispute.get("dispute_flipped"),
            "dispute_count": dispute.get("_dispute_count", 1),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_classify_pm, d): d for d in pm_disputes}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 100 == 0 or done == len(futs):
                log.info("  Polymarket progress: %d/%d", done, len(futs))
            try:
                all_results.append(fut.result())
            except Exception as e:
                slug = futs[fut]["polymarket"]["slug"]
                log.error("  Failed %s: %s", slug, e)

    # ── Kalshi (deterministic — all specification_issue) ────────────
    # Every Kalshi market in the disputed dataset has a post-publication
    # notice (OTHER_INFO, CLARIFICATION, PAUSED, VOIDED, etc.).
    # Kalshi has no public dispute/flip mechanism, so resolution_error
    # and no_fault cannot be reliably determined. The existence of the
    # notice itself is evidence the original specification was insufficient.
    log.info("Assigning %d Kalshi markets as specification_issue (deterministic)",
             len(kalshi_disputes))

    for item in kalshi_disputes:
        market = item["market"]
        event = item["event"]
        ticker = market["ticker"]
        audit = kalshi_audit.get(ticker)
        flags = event.get("dispute_flags", [])
        all_results.append({
            "id": ticker,
            "platform": "kalshi",
            "question": market.get("title", ""),
            "volume": _safe_float(market.get("volume_fp")),
            "rating": audit["rating"] if audit else None,
            "dimension_scores": audit.get("dimension_scores") if audit else None,
            "classification": "specification_issue",
            "confidence": 1.0,
            "rationale": f"Kalshi dispute flags: {', '.join(flags)}",
            "dispute_flipped": None,
            "dispute_count": 1,
        })

    # ── Write output ──────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "total_disputes": len(all_results),
        "polymarket_count": sum(1 for r in all_results if r["platform"] == "polymarket"),
        "kalshi_count": sum(1 for r in all_results if r["platform"] == "kalshi"),
        "classification_summary": dict(Counter(r["classification"] for r in all_results)),
        "disputes": all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    log.info("Wrote %d labeled disputes to %s", len(all_results), OUTPUT_PATH)

    # Summary
    for platform in ("polymarket", "kalshi"):
        subset = [r for r in all_results if r["platform"] == platform]
        counts = Counter(r["classification"] for r in subset)
        log.info("  %s: %s", platform,
                 ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    main()
