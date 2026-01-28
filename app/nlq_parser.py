from __future__ import annotations
import re
from dataclasses import dataclass
from difflib import get_close_matches

@dataclass(frozen=True)
class NLQ:
    intent: str
    metric: str | None          # "spend" | "income" | "net" | None
    category: str | None
    limit: int | None

def detect_intent_and_metric(text: str) -> tuple[str, str | None]:
    t = text.lower()

    # budget risk
    if any(k in t for k in ["over budget", "exceed", "budget risk", "risk"]):
        return "budget_risk", None

    # forecast
    if any(k in t for k in ["forecast", "predict", "prediction", "next month spend"]):
        return "forecast", "spend"

    # tops
    if "top" in t and any(k in t for k in ["merchant", "description", "transactions"]):
        return "top_descriptions", "spend"
    if "top" in t and any(k in t for k in ["category", "categories"]):
        return "by_category", "spend"

    # breakdown
    if any(k in t for k in ["by category", "breakdown", "categories"]):
        return "by_category", "spend"

    # summary / metric selection
    if "income" in t or "salary" in t:
        return "summary", "income"
    if any(k in t for k in ["expense", "spent", "spend", "spending"]):
        return "summary", "spend"
    if "net" in t or "balance" in t:
        return "summary", "net"

    return "summary", None

def extract_limit(text: str, default: int = 10) -> int:
    t = text.lower()
    m = re.search(r"\btop\s+(\d{1,3})\b", t)
    if m:
        return max(1, min(int(m.group(1)), 100))
    return default

def fuzzy_match_category(text: str, known_categories: list[str]) -> str | None:
    t = text.lower()

    if "uncategorized" in t or "un-categorized" in t:
        return "Uncategorized"

    # Exact substring match (best)
    for cat in known_categories:
        if cat.lower() in t:
            return cat

    # Try patterns like "on groceries", "for groceries"
    m = re.search(r"\b(on|for|in)\s+([a-zA-Z &/-]{3,40})\b", text, flags=re.IGNORECASE)
    candidate = m.group(2).strip() if m else None

    # Fuzzy match against known categories using difflib
    if candidate:
        choices = {c.lower(): c for c in known_categories}
        matches = get_close_matches(candidate.lower(), list(choices.keys()), n=1, cutoff=0.72)
        if matches:
            return choices[matches[0]]

    # Also try fuzzy matching against any single token (handles "groceris")
    tokens = re.findall(r"[a-zA-Z]{3,}", t)
    choices = {c.lower(): c for c in known_categories}
    for tok in tokens:
        matches = get_close_matches(tok.lower(), list(choices.keys()), n=1, cutoff=0.80)
        if matches:
            return choices[matches[0]]

    return None

def parse_nlq(text: str, known_categories: list[str]) -> NLQ:
    intent, metric = detect_intent_and_metric(text)
    limit = extract_limit(text) if intent in ("top_descriptions",) else None
    category = fuzzy_match_category(text, known_categories)
    return NLQ(intent=intent, metric=metric, category=category, limit=limit)
