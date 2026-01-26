import pandas as pd

REQUIRED_COLUMNS = {"date", "description", "amount"}

def read_transactions_csv(file_bytes: bytes) -> list[dict]:
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.date
    df["description"] = df["description"].astype(str).str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="raise")

    if "currency" not in df.columns:
        df["currency"] = "EUR"
    if "category" not in df.columns:
        df["category"] = None
    if "account" not in df.columns:
        df["account"] = None

    return df[["date", "description", "amount", "currency", "category", "account"]].to_dict("records")
