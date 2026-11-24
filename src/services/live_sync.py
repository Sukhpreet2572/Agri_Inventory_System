import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.database.models import DailyPriceHistory
from src.database.seed import match_market_id
from src.ml.live_api_client import fetch_live_mandi_prices

META_PATH = Path("models/artifacts/entity_mappings.json")


def sync_live_gov_prices(db: Session) -> dict:
    """
    Fetch latest live daily records from data.gov.in, normalize them via `standardize_columns`,
    and upsert matching markets/commodities into PostgreSQL.
    """
    with open(META_PATH, "r", encoding="utf-8") as f:
        mappings = json.load(f)

    target_commodities = list(mappings["commodity_to_id"].keys())
    live_df = fetch_live_mandi_prices(
        target_commodities=target_commodities,
        max_records=20000,
        page_size=5000,
    )

    upserted_count = 0
    matched_records = []

    if not live_df.empty:
        for _, r in live_df.iterrows():
            m_id = match_market_id(str(r["market"]), mappings["market_to_id"])
            c_id = mappings["commodity_to_id"].get(str(r["commodity"]))
            if m_id and c_id:
                price_date = pd.to_datetime(r["date"]).date()
                modal_val = float(r["modal_price"])
                stmt = pg_insert(DailyPriceHistory).values(
                    market_id=int(m_id),
                    commodity_id=int(c_id),
                    price_date=price_date,
                    min_price=float(r["min_price"]),
                    max_price=float(r["max_price"]),
                    modal_price=modal_val,
                    source="LIVE_GOV_API",
                ).on_conflict_do_update(
                    index_elements=["market_id", "commodity_id", "price_date"],
                    set_={
                        "min_price": float(r["min_price"]),
                        "max_price": float(r["max_price"]),
                        "modal_price": modal_val,
                        "source": "LIVE_GOV_API",
                    },
                )
                db.execute(stmt)
                upserted_count += 1
                matched_records.append(
                    {
                        "market": str(r["market"]),
                        "commodity": str(r["commodity"]),
                        "date": str(price_date),
                        "modal_price": modal_val,
                    }
                )
        db.commit()

    return {
        "synced_at": datetime.utcnow().isoformat() + "Z",
        "total_api_rows_scanned": int(len(live_df)),
        "matched_records_upserted": upserted_count,
        "updated_items": matched_records,
    }