from __future__ import annotations
from .utils_budget import parse_month_key

def prev_months(month_key: str, n: int) -> list[str]:
    y, m = parse_month_key(month_key)
    keys = []
    for _ in range(n):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        keys.append(f"{y:04d}-{m:02d}")
    keys.reverse()
    return keys
