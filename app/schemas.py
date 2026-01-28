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

class CategoryRuleCreate(BaseModel):
    pattern: str
    category: str
    priority: int = 100
    is_regex: bool = False


class CategoryRuleOut(BaseModel):
    id: int
    pattern: str
    category: str
    priority: int
    is_regex: bool

    class Config:
        from_attributes = True

class BudgetUpsert(BaseModel):
    year: int
    month: int  # 1-12
    category: str
    limit_amount: float


class BudgetOut(BaseModel):
    id: int
    year: int
    month: int
    category: str
    limit_amount: float

    class Config:
        from_attributes = True

class BudgetSuggestion(BaseModel):
    category: str
    avg_spend: float
    suggested_limit: float

class BudgetSuggestionsOut(BaseModel):
    month: str
    lookback_months: int
    buffer_pct: float
    items: list[BudgetSuggestion]
