from sqlalchemy import Column, Integer, String, Date, Numeric, Boolean
from .db import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    description = Column(String, nullable=False, index=True)
    amount = Column(Numeric(12, 2), nullable=False)  # negative = expense
    currency = Column(String(3), nullable=False, default="EUR")
    category = Column(String, nullable=True, index=True)
    account = Column(String, nullable=True)

class CategoryRule(Base):
    __tablename__ = "category_rules"

    id = Column(Integer, primary_key=True, index=True)
    pattern = Column(String, nullable=False, index=True)     # keyword or regex
    category = Column(String, nullable=False, index=True)
    priority = Column(Integer, nullable=False, default=100)  # lower number = applied first
    is_regex = Column(Boolean, nullable=False, default=False)