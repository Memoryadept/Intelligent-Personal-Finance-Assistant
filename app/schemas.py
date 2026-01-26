from datetime import date
from pydantic import BaseModel

class TransactionOut(BaseModel):
    id: int
    date: date
    description: str
    amount: float
    currency: str
    category: str | None = None
    account: str | None = None

    class Config:
        from_attributes = True
