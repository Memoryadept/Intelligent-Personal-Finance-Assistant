# tools/generate_sample_data.py
from __future__ import annotations

import csv
import random
from datetime import date, timedelta

random.seed(7)

MERCHANTS = [
    ("K-Market", "Groceries", (5, 40)),
    ("S-Market", "Groceries", (10, 80)),
    ("Lidl", "Groceries", (8, 60)),
    ("HSL", "Transport", (2, 15)),
    ("VR", "Transport", (8, 45)),
    ("Netflix", "Subscriptions", (10, 18)),
    ("Spotify", "Subscriptions", (7, 14)),
    ("Gym", "Health", (20, 60)),
    ("Pharmacy", "Health", (5, 35)),
    ("Wolt", "Eating Out", (12, 45)),
    ("Restaurant", "Eating Out", (20, 90)),
    ("Electricity", "Utilities", (20, 120)),
    ("Internet", "Utilities", (15, 45)),
]

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"

def main():
    # Adjust these
    start = date(2024, 1, 1)
    end = date(2026, 1, 31)
    out = "data/generated.csv"

    rows = []

    # Monthly income + rent on predictable days
    income_amount = 2600
    rent_amount = 950

    for d in daterange(start, end):
        # Salary on 1st
        if d.day == 1:
            rows.append({
                "date": d.isoformat(),
                "description": "Salary",
                "amount": f"{income_amount:.2f}",
                "currency": "EUR",
                "category": "Income",
                "account": "Main",
            })

        # Rent on 2nd
        if d.day == 2:
            rows.append({
                "date": d.isoformat(),
                "description": "Rent",
                "amount": f"{-rent_amount:.2f}",
                "currency": "EUR",
                "category": "Housing",
                "account": "Main",
            })

        # Random daily spend (some days none, some days multiple)
        # Weekends a bit higher
        base_tx_count = 0 if random.random() < 0.35 else 1
        if d.weekday() >= 5 and random.random() < 0.35:
            base_tx_count += 1

        for _ in range(base_tx_count):
            merchant, cat, (lo, hi) = random.choice(MERCHANTS)
            amount = round(random.uniform(lo, hi), 2)
            rows.append({
                "date": d.isoformat(),
                "description": merchant,
                "amount": f"{-amount:.2f}",
                "currency": "EUR",
                "category": cat,
                "account": "Main",
            })

    # Optional: add some “one-off” bigger expenses
    one_offs = [
        ("Laptop", "Shopping", 899.00),
        ("Trip", "Travel", 450.00),
        ("Insurance", "Insurance", 220.00),
    ]
    for desc, cat, amt in one_offs:
        for mk in ["2024-06", "2025-02", "2025-11"]:
            y, m = map(int, mk.split("-"))
            d = date(y, m, random.randint(10, 25))
            rows.append({
                "date": d.isoformat(),
                "description": desc,
                "amount": f"{-amt:.2f}",
                "currency": "EUR",
                "category": cat,
                "account": "Main",
            })

    # Sort rows by date
    rows.sort(key=lambda r: r["date"])

    # Write CSV
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "description", "amount", "currency", "category", "account"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Quick stats
    per_month = {}
    for r in rows:
        mk = month_key(date.fromisoformat(r["date"]))
        per_month[mk] = per_month.get(mk, 0) + 1

    print(f"Wrote {len(rows)} rows to {out}")
    print(f"Months covered: {min(per_month)} .. {max(per_month)}")
    print(f"Avg rows/month: {sum(per_month.values())/len(per_month):.1f}")

if __name__ == "__main__":
    main()
