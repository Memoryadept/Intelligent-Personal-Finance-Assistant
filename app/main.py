from datetime import date
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from .db import Base, engine, get_db
from .models import Transaction, CategoryRule
from .categorizer import apply_rules_to_description
from .schemas import TransactionOut, CategoryRuleCreate, CategoryRuleOut
from .importers import read_transactions_csv

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
