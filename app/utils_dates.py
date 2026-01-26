from __future__ import annotations
from datetime import date

def month_range(month: str) -> tuple[date, date]:
    """
    month: 'YYYY-MM'
    Returns (start_date_inclusive, end_date_exclusive)
    """
    year_s, mon_s = month.split("-")
    y = int(year_s)
    m = int(mon_s)
    start = date(y, m, 1)

    if m == 12:
        end = date(y + 1, 1, 1)
    else:
        end = date(y, m + 1, 1)

    return start, end
