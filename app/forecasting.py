from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import warnings
import pandas as pd

from sqlalchemy.orm import Session
from sqlalchemy import select

from .models import Transaction


@dataclass(frozen=True)
class MonthPoint:
    month: str          # 'YYYY-MM'
    spend: Decimal      # positive spend amount (expenses only)


def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    # month 1..12
    total = (year * 12 + (month - 1)) + delta
    y = total // 12
    m = (total % 12) + 1
    return y, m


def get_monthly_spend_series(db: Session, category: str | None = None) -> list[MonthPoint]:
    """
    Returns monthly spend series (expenses only), sorted oldest->newest.
    category=None => all categories combined
    category='Uncategorized' => tx.category is None
    else => exact match
    """
    txs = list(db.execute(select(Transaction)).scalars().all())

    per_month = defaultdict(lambda: Decimal("0"))

    for tx in txs:
        amt = Decimal(tx.amount)
        if amt >= 0:
            continue  # spend only

        tx_cat = tx.category or "Uncategorized"
        if category:
            if category != tx_cat:
                continue

        mk = month_key(tx.date)
        per_month[mk] += (-amt)

    months = sorted(per_month.keys())
    return [MonthPoint(m, per_month[m]) for m in months]


def moving_average_forecast(series: list[MonthPoint], months_ahead: int, window: int = 6) -> list[MonthPoint]:
    if not series:
        return []

    last_month = series[-1].month
    y, m = map(int, last_month.split("-"))

    values = [p.spend for p in series]
    preds: list[MonthPoint] = []

    for i in range(1, months_ahead + 1):
        w = values[-window:] if len(values) >= window else values[:]
        pred = sum(w) / Decimal(len(w))
        ny, nm = add_months(y, m, i)
        mk = f"{ny:04d}-{nm:02d}"
        preds.append(MonthPoint(mk, pred))
        values.append(pred)  # roll forward

    return preds


def seasonal_forecast(series: list[MonthPoint], months_ahead: int) -> list[MonthPoint]:
    """
    For each target month, predict using average spend of same MM across previous years.
    Falls back to overall average if no seasonal history.
    """
    if not series:
        return []

    month_to_vals = defaultdict(list)
    for p in series:
        mm = p.month.split("-")[1]
        month_to_vals[mm].append(p.spend)

    overall_avg = sum((p.spend for p in series), Decimal("0")) / Decimal(len(series))

    last_month = series[-1].month
    y, m = map(int, last_month.split("-"))

    preds = []
    for i in range(1, months_ahead + 1):
        ny, nm = add_months(y, m, i)
        mm = f"{nm:02d}"
        vals = month_to_vals.get(mm)
        pred = (sum(vals) / Decimal(len(vals))) if vals else overall_avg
        preds.append(MonthPoint(f"{ny:04d}-{nm:02d}", pred))

    return preds


def blend_forecast(series: list[MonthPoint], months_ahead: int, window: int = 6, alpha: float = 0.7) -> list[MonthPoint]:
    """
    alpha=0.7 => 70% moving average + 30% seasonal
    """
    ma = moving_average_forecast(series, months_ahead, window=window)
    se = seasonal_forecast(series, months_ahead)

    preds = []
    for p_ma, p_se in zip(ma, se):
        pred = (Decimal(str(alpha)) * p_ma.spend) + (Decimal("1") - Decimal(str(alpha))) * p_se.spend
        preds.append(MonthPoint(p_ma.month, pred))
    return preds

def sarimax_forecast(
    series: list[MonthPoint],
    months_ahead: int,
    seasonal_period: int = 12,
) -> tuple[list[MonthPoint], list[MonthPoint], list[MonthPoint]]:
    """
    Returns (pred, lower, upper) MonthPoint lists.
    Uses SARIMAX with a small default structure:
      order=(1,1,1), seasonal_order=(1,1,1,seasonal_period)
    Falls back if not enough data.
    """
    if not series:
        return [], [], []

    # Need enough history to estimate seasonality; rule of thumb: >= 2 seasons
    if len(series) < max(18, seasonal_period * 2):
        # fallback: moving average
        preds = moving_average_forecast(series, months_ahead, window=min(6, len(series)))
        # No intervals here
        return preds, [], []

    # Build a pandas monthly time series
    # Convert 'YYYY-MM' to period start dates
    idx = pd.to_datetime([p.month + "-01" for p in series])
    y = pd.Series([float(p.spend) for p in series], index=idx).asfreq("MS")

    # statsmodels import inside function so the rest of app works without statsmodels
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        preds = moving_average_forecast(series, months_ahead, window=min(6, len(series)))
        return preds, [], []

    # Fit SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_period),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        res = model.fit(disp=False)

    fc = res.get_forecast(steps=months_ahead)
    mean = fc.predicted_mean

    # Confidence intervals (if available)
    try:
        conf = fc.conf_int(alpha=0.05)  # 95% CI
        lower_s = conf.iloc[:, 0]
        upper_s = conf.iloc[:, 1]
    except Exception:
        lower_s = None
        upper_s = None

    # Build month keys for forecast months
    last_month = series[-1].month
    y0, m0 = map(int, last_month.split("-"))

    pred_points: list[MonthPoint] = []
    lower_points: list[MonthPoint] = []
    upper_points: list[MonthPoint] = []

    for i in range(1, months_ahead + 1):
        ny, nm = add_months(y0, m0, i)
        mk = f"{ny:04d}-{nm:02d}"

        pred_val = max(0.0, float(mean.iloc[i - 1]))
        pred_points.append(MonthPoint(mk, Decimal(str(pred_val))))

        if lower_s is not None and upper_s is not None:
            lo = max(0.0, float(lower_s.iloc[i - 1]))
            hi = max(0.0, float(upper_s.iloc[i - 1]))
            lower_points.append(MonthPoint(mk, Decimal(str(lo))))
            upper_points.append(MonthPoint(mk, Decimal(str(hi))))

    return pred_points, lower_points, upper_points

def forecast_next_month(
    db,
    category: str | None,
    method: str = "sarimax",
    window: int = 6,
    alpha: float = 0.7,
    seasonal_period: int = 12,
):
    series = get_monthly_spend_series(db, category=category)
    if not series:
        return None

    method = method.lower().strip()
    lower = None
    upper = None

    if method == "ma":
        preds = moving_average_forecast(series, months_ahead=1, window=window)
        pred = preds[0].spend if preds else None
    elif method == "seasonal":
        preds = seasonal_forecast(series, months_ahead=1)
        pred = preds[0].spend if preds else None
    elif method == "blend":
        preds = blend_forecast(series, months_ahead=1, window=window, alpha=alpha)
        pred = preds[0].spend if preds else None
    elif method == "sarimax":
        preds, lo, hi = sarimax_forecast(series, months_ahead=1, seasonal_period=seasonal_period)
        pred = preds[0].spend if preds else None
        if lo and hi:
            lower = lo[0].spend
            upper = hi[0].spend
    else:
        raise ValueError("method must be one of: ma, seasonal, blend, sarimax")

    if pred is None:
        return None

    return {
        "next_month": preds[0].month,
        "pred": pred,
        "lower": lower,
        "upper": upper,
        "history_points": len(series),
    }