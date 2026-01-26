from datetime import date
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from .db import Base, engine, get_db
from .models import Transaction
from .schemas import TransactionOut
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

    for r in records:
        db.add(Transaction(**r))

    db.commit()
    return {"inserted": len(records)}

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
