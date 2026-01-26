from __future__ import annotations
from datetime import date

def parse_month_key(month_key: str) -> tuple[int, int]:
    # 'YYYY-MM' -> (YYYY, MM)
    y_s, m_s = month_key.split("-")
    return int(y_s), int(m_s)

def month_start_end(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    return start, end
