from datetime import date
from unicodedata import category
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func
from decimal import Decimal
from collections import defaultdict

from .db import Base, engine, get_db
from .models import Transaction, CategoryRule, Budget
from .categorizer import apply_rules_to_description
from .schemas import TransactionOut, CategoryRuleCreate, CategoryRuleOut, BudgetUpsert, BudgetOut, BudgetSuggestionsOut
from .importers import read_transactions_csv
from .utils_dates import month_range
from .utils_budget import parse_month_key, month_start_end
from .utils_history import prev_months
from .nlq_time import resolve_range, shift_month, find_explicit_month
from .nlq_parser import parse_nlq
from .forecasting import (
    get_monthly_spend_series,
    moving_average_forecast,
    seasonal_forecast,
    blend_forecast,
    sarimax_forecast,
    forecast_next_month,
)


Base.metadata.create_all(bind=engine)

app = FastAPI(title="Personal Finance Assistant API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transactions/import", response_model=dict)
async def import_transactions(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    content = await file.read()
    try:
        records = read_transactions_csv(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Load rules ordered by priority (lowest first)
    rules = list(
        db.execute(
            select(CategoryRule).order_by(CategoryRule.priority.asc(), CategoryRule.id.asc())
        ).scalars().all()
    )

    inserted = 0
    for r in records:
        # Normalize empty strings to None
        if isinstance(r.get("category"), str) and not r["category"].strip():
            r["category"] = None

        # Auto-categorize only if missing
        if r.get("category") is None:
            matched = apply_rules_to_description(r.get("description", ""), rules)
            if matched:
                r["category"] = matched

        db.add(Transaction(**r))
        inserted += 1

    db.commit()
    return {"inserted": inserted}


@app.get("/transactions", response_model=list[TransactionOut])
def list_transactions(
    db: Session = Depends(get_db),
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    category: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
):
    stmt = select(Transaction)

    if start:
        stmt = stmt.where(Transaction.date >= start)
    if end:
        stmt = stmt.where(Transaction.date <= end)
    if category:
        if category.lower() == "uncategorized":
            stmt = stmt.where(Transaction.category.is_(None))
        else:
            stmt = stmt.where(Transaction.category == category)

    stmt = stmt.order_by(Transaction.date.desc()).limit(limit)
    return list(db.execute(stmt).scalars().all())

@app.post("/rules", response_model=CategoryRuleOut)
def create_rule(payload: CategoryRuleCreate, db: Session = Depends(get_db)):
    rule = CategoryRule(
        pattern=payload.pattern.strip(),
        category=payload.category.strip(),
        priority=payload.priority,
        is_regex=payload.is_regex,
    )
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule


@app.get("/rules", response_model=list[CategoryRuleOut])
def list_rules(db: Session = Depends(get_db)):
    rules = db.execute(
        select(CategoryRule).order_by(CategoryRule.priority.asc(), CategoryRule.id.asc())
    ).scalars().all()
    return list(rules)


@app.delete("/rules/{rule_id}", response_model=dict)
def delete_rule(rule_id: int, db: Session = Depends(get_db)):
    rule = db.get(CategoryRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    db.delete(rule)
    db.commit()
    return {"deleted": rule_id}

@app.post("/transactions/re-categorize", response_model=dict)
def recategorize_transactions(
    only_uncategorized: bool = True,
    db: Session = Depends(get_db),
):
    rules = list(
        db.execute(
            select(CategoryRule).order_by(CategoryRule.priority.asc(), CategoryRule.id.asc())
        ).scalars().all()
    )

    stmt = select(Transaction)
    if only_uncategorized:
        stmt = stmt.where(Transaction.category.is_(None))

    txs = list(db.execute(stmt).scalars().all())

    updated = 0
    for tx in txs:
        matched = apply_rules_to_description(tx.description, rules)
        if matched and tx.category != matched:
            tx.category = matched
            updated += 1

    db.commit()
    return {"checked": len(txs), "updated": updated, "only_uncategorized": only_uncategorized}

@app.get("/analytics/summary")
def analytics_summary(month: str, db: Session = Depends(get_db)):
    start, end = month_range(month)

    txs = list(
        db.execute(
            select(Transaction).where(Transaction.date >= start, Transaction.date < end)
        ).scalars().all()
    )

    income = Decimal("0")
    expense = Decimal("0")

    for tx in txs:
        amt = Decimal(tx.amount)
        if amt >= 0:
            income += amt
        else:
            expense += (-amt)

    net = income - expense
    return {
        "month": month,
        "income": float(income),
        "expense": float(expense),
        "net": float(net),
        "transactions": len(txs),
    }


@app.get("/analytics/by-category")
def analytics_by_category(month: str, db: Session = Depends(get_db)):
    start, end = month_range(month)

    txs = list(
        db.execute(
            select(Transaction).where(Transaction.date >= start, Transaction.date < end)
        ).scalars().all()
    )

    totals = defaultdict(lambda: Decimal("0"))
    for tx in txs:
        amt = Decimal(tx.amount)
        if amt < 0:  # spending only
            cat = tx.category or "Uncategorized"
            totals[cat] += (-amt)

    result = [{"category": k, "total": float(v)} for k, v in totals.items()]
    result.sort(key=lambda x: x["total"], reverse=True)

    return {"month": month, "items": result}


@app.get("/analytics/top-descriptions")
def analytics_top_descriptions(month: str, limit: int = 10, db: Session = Depends(get_db)):
    start, end = month_range(month)

    txs = list(
        db.execute(
            select(Transaction).where(Transaction.date >= start, Transaction.date < end)
        ).scalars().all()
    )

    totals = defaultdict(lambda: Decimal("0"))
    for tx in txs:
        amt = Decimal(tx.amount)
        if amt < 0:
            key = (tx.description or "").strip() or "Unknown"
            totals[key] += (-amt)

    items = [{"description": k, "total": float(v)} for k, v in totals.items()]
    items.sort(key=lambda x: x["total"], reverse=True)

    return {"month": month, "limit": limit, "items": items[:limit]}


@app.get("/analytics/trend")
def analytics_trend(
    months: int = 12,
    category: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Returns monthly spending totals for the last N months (including current month).
    If category is provided, only that category (plus Uncategorized if category='Uncategorized').
    """
    # Find max date in DB (so "last N months" is based on your data, not today)
    max_date = db.execute(select(Transaction.date).order_by(Transaction.date.desc()).limit(1)).scalar_one_or_none()
    if not max_date:
        return {"months": months, "category": category, "items": []}

    # Build month keys backwards
    y, m = max_date.year, max_date.month
    month_keys = []
    for _ in range(months):
        month_keys.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    month_keys.reverse()

    series = []
    for mk in month_keys:
        start, end = month_range(mk)
        stmt = select(Transaction).where(Transaction.date >= start, Transaction.date < end)

        txs = list(db.execute(stmt).scalars().all())

        total = Decimal("0")
        for tx in txs:
            amt = Decimal(tx.amount)
            if amt >= 0:
                continue  # spending trend only

            tx_cat = tx.category or "Uncategorized"
            if category:
                if category == "Uncategorized":
                    if tx_cat != "Uncategorized":
                        continue
                else:
                    if tx_cat != category:
                        continue

            total += (-amt)

        series.append({"month": mk, "spend": float(total)})

    return {"months": months, "category": category, "items": series}

@app.post("/budgets", response_model=BudgetOut)
def upsert_budget(payload: BudgetUpsert, db: Session = Depends(get_db)):
    if payload.month < 1 or payload.month > 12:
        raise HTTPException(status_code=400, detail="month must be 1-12")

    category = payload.category.strip()
    if not category:
        raise HTTPException(status_code=400, detail="category cannot be empty")

    existing = db.execute(
        select(Budget).where(
            Budget.year == payload.year,
            Budget.month == payload.month,
            Budget.category == category,
        )
    ).scalar_one_or_none()

    if existing:
        existing.limit_amount = payload.limit_amount
        db.commit()
        db.refresh(existing)
        return existing

    b = Budget(
        year=payload.year,
        month=payload.month,
        category=category,
        limit_amount=payload.limit_amount,
    )
    db.add(b)
    db.commit()
    db.refresh(b)
    return b


@app.get("/budgets", response_model=list[BudgetOut])
def list_budgets(month: str | None = None, db: Session = Depends(get_db)):
    stmt = select(Budget).order_by(Budget.year.desc(), Budget.month.desc(), Budget.category.asc())

    if month:
        y, m = parse_month_key(month)
        stmt = stmt.where(Budget.year == y, Budget.month == m)

    return list(db.execute(stmt).scalars().all())


@app.delete("/budgets/{budget_id}", response_model=dict)
def delete_budget(budget_id: int, db: Session = Depends(get_db)):
    b = db.get(Budget, budget_id)
    if not b:
        raise HTTPException(status_code=404, detail="Budget not found")
    db.delete(b)
    db.commit()
    return {"deleted": budget_id}

@app.get("/budgets/status")
def budget_status(month: str, db: Session = Depends(get_db)):
    y, m = parse_month_key(month)
    start, end = month_start_end(y, m)

    budgets = list(
        db.execute(
            select(Budget).where(Budget.year == y, Budget.month == m)
        ).scalars().all()
    )

    # Pull transactions for month
    txs = list(
        db.execute(
            select(Transaction).where(Transaction.date >= start, Transaction.date < end)
        ).scalars().all()
    )

    # spending per category (expenses only)
    spent = defaultdict(lambda: Decimal("0"))
    for tx in txs:
        amt = Decimal(tx.amount)
        if amt < 0:
            cat = tx.category or "Uncategorized"
            spent[cat] += (-amt)

    items = []
    total_limit = Decimal("0")
    total_spent = Decimal("0")

    # report all budgets
    for b in budgets:
        cat = b.category
        limit_amt = Decimal(b.limit_amount)
        spent_amt = spent.get(cat, Decimal("0"))
        remaining = limit_amt - spent_amt
        pct = float(spent_amt / limit_amt * 100) if limit_amt != 0 else None

        items.append({
            "category": cat,
            "limit": float(limit_amt),
            "spent": float(spent_amt),
            "remaining": float(remaining),
            "percent_used": pct,
            "status": "over" if remaining < 0 else ("warning" if pct is not None and pct >= 90 else "ok"),
        })

        total_limit += limit_amt
        total_spent += spent_amt

    # Sort: most used first
    items.sort(key=lambda x: (x["percent_used"] is None, -(x["percent_used"] or 0)))

    totals = {
        "limit": float(total_limit),
        "spent": float(total_spent),
        "remaining": float(total_limit - total_spent),
        "percent_used": float(total_spent / total_limit * 100) if total_limit != 0 else None,
    }

    # Also show categories you spent in that have no budget (optional but helpful)
    unbudgeted = []
    budgeted_categories = {b.category for b in budgets}
    for cat, amt in spent.items():
        if cat not in budgeted_categories:
            unbudgeted.append({"category": cat, "spent": float(amt)})

    unbudgeted.sort(key=lambda x: x["spent"], reverse=True)

    return {"month": month, "items": items, "totals": totals, "unbudgeted_spend": unbudgeted}

@app.post("/budgets/suggest", response_model=BudgetSuggestionsOut)
def suggest_budgets(
    month: str,
    lookback_months: int = 3,
    buffer_pct: float = 10.0,
    db: Session = Depends(get_db),
):
    try:
        _ = parse_month_key(month)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if lookback_months < 1 or lookback_months > 24:
        raise HTTPException(status_code=400, detail="lookback_months must be 1..24")
    if buffer_pct < 0 or buffer_pct > 200:
        raise HTTPException(status_code=400, detail="buffer_pct must be 0..200")

    months = prev_months(month, lookback_months)

    # sum spend per category across lookback months
    totals = defaultdict(lambda: Decimal("0"))
    month_counts = defaultdict(int)

    for mk in months:
        y, m = parse_month_key(mk)
        start, end = month_start_end(y, m)

        txs = list(
            db.execute(
                select(Transaction).where(Transaction.date >= start, Transaction.date < end)
            ).scalars().all()
        )

        per_month = defaultdict(lambda: Decimal("0"))
        for tx in txs:
            amt = Decimal(tx.amount)
            if amt < 0:
                cat = tx.category or "Uncategorized"
                per_month[cat] += (-amt)

        for cat, amt in per_month.items():
            totals[cat] += amt
            month_counts[cat] += 1

    items = []
    for cat, total in totals.items():
        count = month_counts[cat] or lookback_months
        avg = total / Decimal(count)
        suggested = avg * (Decimal("1") + Decimal(str(buffer_pct)) / Decimal("100"))
        items.append({
            "category": cat,
            "avg_spend": float(avg),
            "suggested_limit": float(suggested),
        })

    items.sort(key=lambda x: x["suggested_limit"], reverse=True)

    return {
        "month": month,
        "lookback_months": lookback_months,
        "buffer_pct": buffer_pct,
        "items": items,
    }

@app.post("/budgets/apply-suggestions", response_model=dict)
def apply_budget_suggestions(
    month: str,
    lookback_months: int = 3,
    buffer_pct: float = 10.0,
    overwrite: bool = False,
    db: Session = Depends(get_db),
):
    # Validate month early
    try:
        year, mon = parse_month_key(month)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Reuse the same suggestion logic (inline, to keep it simple)
    months = prev_months(month, lookback_months)

    totals = defaultdict(lambda: Decimal("0"))
    month_counts = defaultdict(int)

    for mk in months:
        y, m = parse_month_key(mk)
        start, end = month_start_end(y, m)

        txs = list(
            db.execute(
                select(Transaction).where(Transaction.date >= start, Transaction.date < end)
            ).scalars().all()
        )

        per_month = defaultdict(lambda: Decimal("0"))
        for tx in txs:
            amt = Decimal(tx.amount)
            if amt < 0:
                cat = tx.category or "Uncategorized"
                per_month[cat] += (-amt)

        for cat, amt in per_month.items():
            totals[cat] += amt
            month_counts[cat] += 1

    created = 0
    updated = 0
    skipped = 0

    for cat, total in totals.items():
        count = month_counts[cat] or lookback_months
        avg = total / Decimal(count)
        suggested = avg * (Decimal("1") + Decimal(str(buffer_pct)) / Decimal("100"))

        # Find existing budget for (year, month, category)
        existing = db.execute(
            select(Budget).where(Budget.year == year, Budget.month == mon, Budget.category == cat)
        ).scalar_one_or_none()

        if existing:
            if overwrite:
                existing.limit_amount = float(suggested)
                updated += 1
            else:
                skipped += 1
        else:
            db.add(Budget(year=year, month=mon, category=cat, limit_amount=float(suggested)))
            created += 1

    db.commit()

    return {
        "month": month,
        "lookback_months": lookback_months,
        "buffer_pct": buffer_pct,
        "overwrite": overwrite,
        "created": created,
        "updated": updated,
        "skipped": skipped,
        "categories_suggested": len(totals),
    }

@app.get("/forecast/spend")
def forecast_spend(
    months_ahead: int = 3,
    category: str | None = None,
    method: str = "blend",  # "ma" | "seasonal" | "blend"
    window: int = 6,
    alpha: float = 0.7,
    db: Session = Depends(get_db),
):
    if months_ahead < 1 or months_ahead > 24:
        raise HTTPException(status_code=400, detail="months_ahead must be 1..24")
    if window < 1 or window > 24:
        raise HTTPException(status_code=400, detail="window must be 1..24")
    if alpha < 0 or alpha > 1:
        raise HTTPException(status_code=400, detail="alpha must be 0..1")
    method = method.lower().strip()

    series = get_monthly_spend_series(db, category=category)

    if not series:
        return {"category": category, "method": method, "history": [], "forecast": []}

    lower = []
    upper = []

    if method == "ma":
        preds = moving_average_forecast(series, months_ahead, window=window)
    elif method == "seasonal":
        preds = seasonal_forecast(series, months_ahead)
    elif method == "blend":
        preds = blend_forecast(series, months_ahead, window=window, alpha=alpha)
    elif method == "sarimax":
        preds, lower, upper = sarimax_forecast(series, months_ahead, seasonal_period=12)
    else:
        raise HTTPException(status_code=400, detail="method must be one of: ma, seasonal, blend, sarimax")


    return {
        "category": category,
        "method": method,
        "params": {"months_ahead": months_ahead, "window": window, "alpha": alpha},
        "history": [{"month": p.month, "spend": float(p.spend)} for p in series[-24:]],
        "forecast": [{"month": p.month, "spend": float(p.spend)} for p in preds],
        "interval_95": None if not lower or not upper else {
            "lower": [{"month": p.month, "spend": float(p.spend)} for p in lower],
            "upper": [{"month": p.month, "spend": float(p.spend)} for p in upper],
        },
}

@app.get("/risk/budget-overrun")
def budget_overrun_risk(
    month: str,
    method: str = "sarimax",
    window: int = 6,
    alpha: float = 0.7,
    db: Session = Depends(get_db),
):
    # month is the budget month you want to evaluate (typically next month)
    try:
        year, mon = parse_month_key(month)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    budgets = list(
        db.execute(
            select(Budget).where(Budget.year == year, Budget.month == mon)
        ).scalars().all()
    )

    if not budgets:
        return {"month": month, "method": method, "items": [], "detail": "No budgets set for this month"}

    items = []
    for b in budgets:
        fc = forecast_next_month(
            db,
            category=b.category,
            method=method,
            window=window,
            alpha=alpha,
        )

        if not fc:
            items.append({
                "category": b.category,
                "budget": float(b.limit_amount),
                "forecast_month": None,
                "forecast": None,
                "risk": "unknown",
                "reason": "Not enough history",
            })
            continue

        budget = Decimal(b.limit_amount)
        pred = Decimal(fc["pred"])
        upper = Decimal(fc["upper"]) if fc["upper"] is not None else None

        overrun = pred - budget
        # Risk rule of thumb:
        # - high: upper CI exceeds budget OR predicted exceeds by >10%
        # - medium: predicted exceeds budget or within 10% of budget
        # - low: comfortably below
        risk = "low"
        if upper is not None and upper > budget:
            risk = "high"
        elif pred > budget * Decimal("1.10"):
            risk = "high"
        elif pred > budget or pred > budget * Decimal("0.90"):
            risk = "medium"

        items.append({
            "category": b.category,
            "budget": float(budget),
            "forecast_month": fc["next_month"],
            "forecast": float(pred),
            "interval_95": None if fc["lower"] is None or fc["upper"] is None else {
                "lower": float(fc["lower"]),
                "upper": float(fc["upper"]),
            },
            "overrun": float(overrun),
            "risk": risk,
            "history_points": fc["history_points"],
        })

    # Sort: highest risk first, then biggest overrun
    risk_rank = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    items.sort(key=lambda x: (risk_rank.get(x["risk"], 9), -(x.get("overrun") or 0)))

    return {"month": month, "method": method, "items": items}

@app.post("/nlq")
def nlq(query: str, db: Session = Depends(get_db)):
    # known categories from transactions + budgets
    tx_cats = db.execute(select(Transaction.category).distinct()).scalars().all()
    known = sorted({c for c in tx_cats if c})

    bud_cats = []
    try:
        bud_cats = db.execute(select(Budget.category).distinct()).scalars().all()
    except Exception:
        pass
    known = sorted(set(known).union({c for c in bud_cats if c}))

    # time range
    rng = resolve_range(query)

    # parse intent/metric/category
    parsed = parse_nlq(query, known)

    # Route
    if parsed.intent == "summary":
        # single month -> use /analytics/summary
        if rng.start == rng.end:
            res = analytics_summary(rng.start, db)
        else:
            res = analytics_summary_range(rng.start, rng.end, db)

        # metric filter
        if parsed.metric == "income":
            res = {"value": res.get("income"), "range": {"start": rng.start, "end": rng.end}}
        elif parsed.metric == "spend":
            res = {"value": res.get("expense"), "range": {"start": rng.start, "end": rng.end}}
        elif parsed.metric == "net":
            res = {"value": res.get("net"), "range": {"start": rng.start, "end": rng.end}}

        return {"query": query, "parsed": parsed.__dict__, "range": rng.__dict__, "result": res}

    if parsed.intent == "by_category":
        # for ranges, you can either:
        # - return the latest month breakdown, or
        # - implement a by-category range endpoint later
        # For now: return breakdown for end month
        return {"query": query, "parsed": parsed.__dict__, "range": rng.__dict__, "result": analytics_by_category(rng.end, db)}

    if parsed.intent == "top_descriptions":
        limit = parsed.limit or 10
        return {"query": query, "parsed": parsed.__dict__, "range": rng.__dict__, "result": analytics_top_descriptions(rng.end, limit, db)}

    if parsed.intent == "forecast":
        return {
            "query": query,
            "parsed": parsed.__dict__,
            "range": rng.__dict__,
            "result": forecast_spend(months_ahead=6, category=parsed.category, method="sarimax", window=6, alpha=0.7, db=db),
        }

    if parsed.intent == "budget_risk":
        # If query doesn’t specify a month, rng defaults to this month.
        # Most “risk” questions mean next month, so auto-shift if user didn’t specify.
        target = rng.start
        if "next month" in query.lower():
            target = rng.start  # already next month by resolve_range
        elif "last month" in query.lower():
            target = rng.start
        elif find_explicit_month(query) is None:
            # assume next month for risk if not explicit
            target = shift_month(rng.start, 1)

        return {
            "query": query,
            "parsed": parsed.__dict__,
            "target_month": target,
            "result": budget_overrun_risk(month=target, method="sarimax", window=6, alpha=0.7, db=db),
        }

    return {"query": query, "parsed": parsed.__dict__, "range": rng.__dict__, "result": {"detail": "No handler"}}


def month_to_dates(month: str):
    y, m = parse_month_key(month)
    return month_start_end(y, m)

@app.get("/analytics/summary-range")
def analytics_summary_range(start: str, end: str, db: Session = Depends(get_db)):
    # start/end are inclusive months (YYYY-MM)
    try:
        y1, m1 = parse_month_key(start)
        y2, m2 = parse_month_key(end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # build list of months from start..end
    months = []
    cur = start
    while True:
        months.append(cur)
        if cur == end:
            break
        cur = shift_month(cur, 1)  # reuse from nlq_time, so import it (see below)

    from decimal import Decimal
    income = Decimal("0")
    expense = Decimal("0")
    tx_count = 0

    for mk in months:
        ys, ms = parse_month_key(mk)
        s, e = month_start_end(ys, ms)
        txs = list(db.execute(select(Transaction).where(Transaction.date >= s, Transaction.date < e)).scalars().all())
        tx_count += len(txs)
        for tx in txs:
            amt = Decimal(tx.amount)
            if amt >= 0:
                income += amt
            else:
                expense += (-amt)

    return {
        "start": start,
        "end": end,
        "income": float(income),
        "expense": float(expense),
        "net": float(income - expense),
        "transactions": tx_count,
        "months": months,
    }
