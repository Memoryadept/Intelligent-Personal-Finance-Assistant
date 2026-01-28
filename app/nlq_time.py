from __future__ import annotations
from dataclasses import dataclass
from datetime import date
import re

@dataclass(frozen=True)
class MonthRange:
    start: str  # YYYY-MM inclusive
    end: str    # YYYY-MM inclusive

def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"

def shift_month(ym: str, delta: int) -> str:
    y, m = map(int, ym.split("-"))
    total = (y * 12 + (m - 1)) + delta
    ny = total // 12
    nm = (total % 12) + 1
    return f"{ny:04d}-{nm:02d}"

def normalize_month_text(text: str) -> str:
    # allow "2025/01" or "2025.01" -> "2025-01"
    t = text.strip()
    t = t.replace("/", "-").replace(".", "-")
    return t

def find_explicit_month(text: str) -> str | None:
    t = normalize_month_text(text.lower())
    m = re.search(r"\b(20\d{2})-(0[1-9]|1[0-2])\b", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None

def find_explicit_year(text: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None

def resolve_range(text: str, today: date | None = None) -> MonthRange:
    """
    Returns a MonthRange always.
    Supports:
      - explicit month: '2025-01'
      - 'last month', 'this month', 'next month'
      - 'last N months' / 'past N months'
      - 'this year'
      - 'in 2025'
    """
    if today is None:
        today = date.today()

    t = text.lower()

    base = month_key(today)

    # explicit month => range is that single month
    explicit_month = find_explicit_month(t)
    if explicit_month:
        return MonthRange(start=explicit_month, end=explicit_month)

    # last/this/next month
    if "last month" in t:
        m = shift_month(base, -1)
        return MonthRange(m, m)
    if "next month" in t:
        m = shift_month(base, +1)
        return MonthRange(m, m)
    if "this month" in t or "current month" in t:
        return MonthRange(base, base)

    # last/past N months
    m_n = re.search(r"\b(last|past)\s+(\d{1,2})\s+months?\b", t)
    if m_n:
        n = int(m_n.group(2))
        n = max(1, min(n, 24))
        start = shift_month(base, -(n - 1))
        end = base
        return MonthRange(start=start, end=end)

    # this year
    if "this year" in t:
        y = today.year
        return MonthRange(start=f"{y:04d}-01", end=f"{y:04d}-{today.month:02d}")

    # in 2025
    y = find_explicit_year(t)
    if y and ("in " in t or "year" in t):
        return MonthRange(start=f"{y:04d}-01", end=f"{y:04d}-12")

    # default: this month
    return MonthRange(base, base)
