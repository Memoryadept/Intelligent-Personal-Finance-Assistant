import re
from typing import Iterable

from .models import CategoryRule

def apply_rules_to_description(description: str, rules: Iterable[CategoryRule]) -> str | None:
    """
    Returns the matched category, or None if no rule matches.
    First match wins (rules should be sorted by priority ASC).
    """
    desc = (description or "").strip()
    if not desc:
        return None

    for rule in rules:
        if rule.is_regex:
            try:
                if re.search(rule.pattern, desc, flags=re.IGNORECASE):
                    return rule.category
            except re.error:
                # Invalid regex pattern in DB - ignore this rule
                continue
        else:
            if rule.pattern.lower() in desc.lower():
                return rule.category

    return None
