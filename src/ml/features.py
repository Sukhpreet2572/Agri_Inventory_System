import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/historical_mandi.csv")
PROCESSED_PATH = Path("data/processed/features.parquet")
META_PATH = Path("models/artifacts/entity_mappings.json")

TARGET_COMMODITIES = ["Potato", "Onion", "Wheat", "Tomato", "Rice"]
TOP_N_MARKETS = 25


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize both historical CSV and live data.gov.in API schemas into one standard format."""
    rename_map = {
        "STATE": "state",
        "State": "state",
        "District Name": "district",
        "District": "district",
        "Market Name": "market",
        "Market": "market",
        "Commodity": "commodity",
        "Variety": "variety",
        "Grade": "grade",
        "Min_Price": "min_price",
        "Min Price": "min_price",
        "Max_Price": "max_price",
        "Max Price": "max_price",
        "Modal_Price": "modal_price",
        "Modal Price": "modal_price",
        "Price Date": "date",
        "Arrival_Date": "date",
        "arrival_date": "date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ["state", "district", "market", "commodity"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df.loc[df[col].isin(["Nan", "None", ""]), col] = np.nan

    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True, errors="coerce")
    return df


def clean_and_aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Filter valid rows, focus on dense top markets & commodities, and build daily series."""
    df = standardize_columns(df)
    df = df.dropna(subset=["market", "commodity", "date", "modal_price"]).copy()

    # Remove extreme outliers or zero/negative prices
    df = df[(df["modal_price"] > 50) & (df["modal_price"] < 50000)]

    # Filter to our core 5 commodities
    df = df[df["commodity"].isin(TARGET_COMMODITIES)].copy()

    # Select top N markets with the richest historical coverage
    top_markets = df["market"].value_counts().head(TOP_N_MARKETS).index.tolist()
    df = df[df["market"].isin(top_markets)].copy()

    # Aggregate multiple varieties/grades on the same day into one daily market-commodity price
    daily = (
        df.groupby(["state", "district", "market", "commodity", "date"], as_index=False)
        .agg(
            min_price=("min_price", "median"),
            max_price=("max_price", "median"),
            modal_price=("modal_price", "median"),
        )
        .sort_values(["market", "commodity", "date"])
    )

    # Reindex each (market, commodity) series onto a continuous daily calendar (ffill up to 3 days for mandi holidays)
    filled_groups = []
    for (state, district, market, commodity), grp in daily.groupby(
        ["state", "district", "market", "commodity"]
    ):
        if len(grp) < 60:
            continue
        full_idx = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        grp = grp.set_index("date").reindex(full_idx)
        grp.index.name = "date"
        grp["state"] = state
        grp["district"] = district
        grp["market"] = market
        grp["commodity"] = commodity
        for pcol in ["min_price", "max_price", "modal_price"]:
            grp[pcol] = grp[pcol].ffill(limit=3)
        grp = grp.dropna(subset=["modal_price"]).reset_index()
        filled_groups.append(grp)

    clean_df = pd.concat(filled_groups, ignore_index=True)

    # Create deterministic integer IDs for modeling & PostgreSQL mapping
    markets_sorted = sorted(clean_df["market"].unique())
    commodities_sorted = sorted(clean_df["commodity"].unique())
    market_to_id = {m: idx + 1 for idx, m in enumerate(markets_sorted)}
    commodity_to_id = {c: idx + 1 for idx, c in enumerate(commodities_sorted)}

    clean_df["market_id"] = clean_df["market"].map(market_to_id)
    clean_df["commodity_id"] = clean_df["commodity"].map(commodity_to_id)

    mappings = {
        "market_to_id": market_to_id,
        "id_to_market": {str(v): k for k, v in market_to_id.items()},
        "commodity_to_id": commodity_to_id,
        "id_to_commodity": {str(v): k for k, v in commodity_to_id.items()},
    }
    return clean_df, mappings


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leak-free lag, rolling window, price spread, and cyclical calendar features."""
    df = df.sort_values(["market_id", "commodity_id", "date"]).copy()
    grp = df.groupby(["market_id", "commodity_id"])

    # 1. Price spread & intraday volatility (lagged by 1 day so we never leak today's min/max into today's prediction)
    df["prev_min_price"] = grp["min_price"].shift(1)
    df["prev_max_price"] = grp["max_price"].shift(1)
    df["prev_price_spread"] = (df["prev_max_price"] - df["prev_min_price"]).fillna(0)

    # 2. Lag features (7, 14, 28 days)
    for lag in [1, 7, 14, 28]:
        df[f"price_lag_{lag}"] = grp["modal_price"].shift(lag)

    # 3. Rolling statistics (shifted by 1 day BEFORE rolling to prevent data leakage)
    shifted_price = grp["modal_price"].shift(1)
    df["rolling_mean_7"] = shifted_price.groupby([df["market_id"], df["commodity_id"]]).transform(
        lambda s: s.rolling(7, min_periods=3).mean()
    )
    df["rolling_std_7"] = shifted_price.groupby([df["market_id"], df["commodity_id"]]).transform(
        lambda s: s.rolling(7, min_periods=3).std()
    ).fillna(0)
    df["rolling_mean_28"] = shifted_price.groupby([df["market_id"], df["commodity_id"]]).transform(
        lambda s: s.rolling(28, min_periods=7).mean()
    )
    df["rolling_std_28"] = shifted_price.groupby([df["market_id"], df["commodity_id"]]).transform(
        lambda s: s.rolling(28, min_periods=7).std()
    ).fillna(0)

    # 4. Momentum / Trend ratio
    df["momentum_7_28"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-5)

    # 5. Calendar & cyclical seasonality features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # Drop initial rows where lag_28 is NaN
    df = df.dropna(subset=["price_lag_28", "rolling_mean_28"]).reset_index(drop=True)
    return df


def run_pipeline():
    print(f"[1/4] Loading raw dataset from {RAW_PATH}...")
    raw_df = pd.read_csv(RAW_PATH, low_memory=False)

    print("[2/4] Cleaning, normalizing schemas, and building daily series...")
    clean_df, mappings = clean_and_aggregate(raw_df)

    print("[3/4] Engineering leak-free time-series features...")
    feat_df = engineer_features(clean_df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)

    feat_df.to_parquet(PROCESSED_PATH, index=False)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)

    print(f"[4/4] Done! Saved {len(feat_df):,} clean feature rows to {PROCESSED_PATH}")
    print(f"      Markets included ({len(mappings['market_to_id'])}): {list(mappings['market_to_id'].keys())[:5]} ...")
    print(f"      Commodities included ({len(mappings['commodity_to_id'])}): {list(mappings['commodity_to_id'].keys())}")
    print(f"      Clean Date Range: {feat_df['date'].min().date()} -> {feat_df['date'].max().date()}")


if __name__ == "__main__":
    run_pipeline()