import pandas as pd

FILE_PATH = "data/raw/historical_mandi.csv"

df = pd.read_csv(FILE_PATH, low_memory=False)

print(f"Total Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}\n")

print("--- FIRST 5 ROWS ---")
print(df.head(5).to_string(index=False))

print("\n--- TOP 10 MARKETS BY RECORD COUNT ---")
# Adjust column name below if your CSV uses 'Market' instead of 'market'
market_col = [c for c in df.columns if "market" in c.lower()][0]
comm_col = [c for c in df.columns if "commodity" in c.lower()][0]
date_col = [c for c in df.columns if "date" in c.lower()][0]

print(df[market_col].value_counts().head(10))

print("\n--- TOP 10 COMMODITIES BY RECORD COUNT ---")
print(df[comm_col].value_counts().head(10))

print(f"\n--- DATE RANGE ({date_col}) ---")
dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
print(f"Min Date: {dates.min()} | Max Date: {dates.max()} | Invalid Dates: {dates.isna().sum()}")