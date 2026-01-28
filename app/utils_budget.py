from __future__ import annotations
from datetime import date

def parse_month_key(month_key: str) -> tuple[int, int]:
    """
    Accepts 'YYYY-MM' only. Raises ValueError with a clear message if invalid.
    """
    if not isinstance(month_key, str):
        raise ValueError("month must be a string in format YYYY-MM")

    parts = month_key.split("-")
    if len(parts) != 2:
        raise ValueError("month must be in format YYYY-MM (example: 2025-01)")

    y_s, m_s = parts
    if len(y_s) != 4 or len(m_s) != 2:
        raise ValueError("month must be in format YYYY-MM (example: 2025-01)")

    y = int(y_s)
    m = int(m_s)
    if m < 1 or m > 12:
        raise ValueError("month must be in format YYYY-MM with MM from 01 to 12")

    return y, m


def month_start_end(year: int, month: int) -> tuple[date, date]:
    """
    Returns (start_date_inclusive, end_date_exclusive)
    """
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    return start, end
