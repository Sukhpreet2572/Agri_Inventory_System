import os
import requests
import pandas as pd
from dotenv import load_dotenv
from src.ml.features import standardize_columns

load_dotenv()

API_KEY = os.getenv("DATA_GOV_API_KEY")
RESOURCE_ID = os.getenv("DATA_GOV_RESOURCE_ID", "9ef84268-d588-465a-a308-a864a43d0070")
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"


def fetch_live_mandi_prices(
    target_commodities: list[str] | None = None,
    max_records: int = 5000,
    page_size: int = 1000,
) -> pd.DataFrame:
    """
    Fetch today's live market prices from data.gov.in, paginate cleanly,
    and normalize schemas via `standardize_columns`.
    """
    if not API_KEY:
        raise ValueError("DATA_GOV_API_KEY is missing in .env")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    all_records = []
    offset = 0

    while offset < max_records:
        params = {
            "api-key": API_KEY,
            "format": "json",
            "offset": offset,
            "limit": min(page_size, max_records - offset),
        }
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        batch = payload.get("records", [])
        if not batch:
            break

        all_records.extend(batch)
        offset += len(batch)
        if len(batch) < params["limit"]:
            break

    if not all_records:
        return pd.DataFrame()

    raw_df = pd.DataFrame(all_records)
    clean_df = standardize_columns(raw_df)
    clean_df = clean_df.dropna(subset=["market", "commodity", "date", "modal_price"]).copy()

    if target_commodities:
        clean_df = clean_df[clean_df["commodity"].isin(target_commodities)].copy()

    # Aggregate multiple varieties/grades per market-commodity-date using median modal_price
    daily_df = (
        clean_df.groupby(["state", "district", "market", "commodity", "date"], as_index=False)
        .agg(
            min_price=("min_price", "median"),
            max_price=("max_price", "median"),
            modal_price=("modal_price", "median"),
        )
    )
    return daily_df


if __name__ == "__main__":
    df = fetch_live_mandi_prices(target_commodities=["Onion", "Potato", "Wheat"], max_records=3000)
    print(f"Fetched & standardized {len(df)} unique (Market, Commodity, Date) rows from live API!")
    if not df.empty:
        print(df.head(10).to_string(index=False))