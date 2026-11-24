import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.database.connection import Base, engine, SessionLocal
from src.database.models import (
    Market,
    Commodity,
    Supplier,
    DailyPriceHistory,
    Inventory,
    PurchaseOrder,
    OrderStatus,
)
from src.ml.live_api_client import fetch_live_mandi_prices

PROCESSED_PATH = Path("data/processed/features.parquet")
META_PATH = Path("models/artifacts/entity_mappings.json")


def match_market_id(api_market_name: str, market_to_id: dict[str, int]) -> int | None:
    """Exact match first, then prefix match (e.g., 'Ahmedabad' -> 'Ahmedabad(Chimanbhai...)')."""
    if api_market_name in market_to_id:
        return market_to_id[api_market_name]
    base_name = api_market_name.split("(")[0].strip()
    for hist_market, m_id in market_to_id.items():
        if hist_market.split("(")[0].strip() == base_name:
            return m_id
    return None


def seed_database():
    print("[1/5] Creating tables in PostgreSQL...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            mappings = json.load(f)

        hist_df = pd.read_parquet(PROCESSED_PATH)

        # 2. Seed Markets (matching exact ML entity IDs)
        print("[2/5] Seeding Markets & Commodities...")
        market_meta = (
            hist_df[["market_id", "market", "state", "district"]]
            .drop_duplicates(subset=["market_id"])
            .sort_values("market_id")
        )
        for _, row in market_meta.iterrows():
            if not db.get(Market, int(row["market_id"])):
                db.add(
                    Market(
                        id=int(row["market_id"]),
                        name=str(row["market"]),
                        state=str(row["state"]),
                        district=str(row["district"]),
                    )
                )

        # Seed Commodities (matching exact ML entity IDs)
        shelf_lives = {"Onion": 45, "Potato": 60, "Wheat": 180}
        for comm_name, comm_id in mappings["commodity_to_id"].items():
            if not db.get(Commodity, int(comm_id)):
                db.add(
                    Commodity(
                        id=int(comm_id),
                        name=str(comm_name),
                        unit="Quintal (100 kg)",
                        shelf_life_days=shelf_lives.get(comm_name, 30),
                    )
                )
        db.commit()

        # 3. Seed Suppliers
        print("[3/5] Seeding Suppliers...")
        if db.query(Supplier).count() == 0:
            suppliers = [
                Supplier(name="Kisan Agro Logistics", contact_email="orders@kisanagro.in", lead_time_days=2, reliability_score=0.98),
                Supplier(name="Bharat Mandi Direct", contact_email="supply@bharatmandi.in", lead_time_days=4, reliability_score=0.94),
                Supplier(name="GreenField ColdChain", contact_email="dispatch@greenfieldcc.in", lead_time_days=3, reliability_score=0.96),
                Supplier(name="Deccan Wholesale Corp", contact_email="trade@deccanwholesale.in", lead_time_days=6, reliability_score=0.91),
            ]
            db.add_all(suppliers)
            db.commit()

        supplier_ids = [int(s.id) for s in db.query(Supplier).all()]

        # 4. Load Historical Daily Prices (last 60 days per pair for fast API inference)
        print("[4/5] Loading recent historical & live API prices into daily_price_history...")
        recent_hist = hist_df.sort_values("date").groupby(["market_id", "commodity_id"]).tail(60)
        hist_rows = [
            {
                "market_id": int(r["market_id"]),
                "commodity_id": int(r["commodity_id"]),
                "price_date": pd.to_datetime(r["date"]).date(),
                "min_price": float(r["min_price"]),
                "max_price": float(r["max_price"]),
                "modal_price": float(r["modal_price"]),
                "source": "HISTORICAL",
            }
            for _, r in recent_hist.iterrows()
        ]
        if hist_rows:
            stmt = pg_insert(DailyPriceHistory).values(hist_rows).on_conflict_do_nothing(
                index_elements=["market_id", "commodity_id", "price_date"]
            )
            db.execute(stmt)
            db.commit()

        # Pull ALL Live API pages (20,000 cap, page_size=5000) & Upsert matching Markets/Commodities
        live_df = fetch_live_mandi_prices(
            target_commodities=list(mappings["commodity_to_id"].keys()),
            max_records=20000,
            page_size=5000,
        )
        live_inserted = 0
        if not live_df.empty:
            for _, r in live_df.iterrows():
                m_id = match_market_id(str(r["market"]), mappings["market_to_id"])
                c_id = mappings["commodity_to_id"].get(str(r["commodity"]))
                if m_id and c_id:
                    stmt = pg_insert(DailyPriceHistory).values(
                        market_id=int(m_id),
                        commodity_id=int(c_id),
                        price_date=pd.to_datetime(r["date"]).date(),
                        min_price=float(r["min_price"]),
                        max_price=float(r["max_price"]),
                        modal_price=float(r["modal_price"]),
                        source="LIVE_GOV_API",
                    ).on_conflict_do_update(
                        index_elements=["market_id", "commodity_id", "price_date"],
                        set_={"modal_price": float(r["modal_price"]), "source": "LIVE_GOV_API"},
                    )
                    db.execute(stmt)
                    live_inserted += 1
            db.commit()
        print(f"      Upserted {live_inserted} live government API records into PostgreSQL!")

        # 5. Seed Warehouse Inventory & Sample Pending Purchase Orders (using pure Python floats)
        print("[5/5] Seeding Warehouse Inventory & Purchase Orders...")
        if db.query(Inventory).count() == 0:
            rng = np.random.default_rng(42)
            active_pairs = recent_hist[["market_id", "commodity_id", "modal_price"]].groupby(
                ["market_id", "commodity_id"], as_index=False
            ).last()

            for idx, r in active_pairs.iterrows():
                sup_id = int(rng.choice(supplier_ids))
                lead_time = int(db.get(Supplier, sup_id).lead_time_days)
                avg_demand = float(round(float(rng.uniform(12.0, 35.0)), 1))
                demand_std = float(round(avg_demand * 0.25, 1))
                safety_stock = float(round(1.65 * demand_std * float(np.sqrt(lead_time)), 1))
                reorder_point = float(round((avg_demand * lead_time) + safety_stock, 1))

                # Make ~25% of items critically low or below reorder point for realistic alerts
                if idx % 4 == 0:
                    curr_stock = float(round(reorder_point * float(rng.uniform(0.35, 0.85)), 1))
                else:
                    curr_stock = float(round(reorder_point * float(rng.uniform(1.15, 2.2)), 1))

                db.add(
                    Inventory(
                        market_id=int(r["market_id"]),
                        commodity_id=int(r["commodity_id"]),
                        supplier_id=sup_id,
                        current_stock_qtl=curr_stock,
                        avg_daily_demand_qtl=avg_demand,
                        demand_std_qtl=demand_std,
                        safety_stock_qtl=safety_stock,
                        reorder_point_qtl=reorder_point,
                    )
                )

                # Create pending POs for some low-stock items
                if idx % 8 == 0:
                    db.add(
                        PurchaseOrder(
                            market_id=int(r["market_id"]),
                            commodity_id=int(r["commodity_id"]),
                            supplier_id=sup_id,
                            quantity_qtl=float(round(avg_demand * 7, 1)),
                            locked_unit_price=float(r["modal_price"]),
                            procurement_strategy="STANDARD_REORDER",
                            status=OrderStatus.PENDING,
                            order_date=date.today() - timedelta(days=1),
                            expected_delivery_date=date.today() + timedelta(days=max(1, lead_time - 1)),
                        )
                    )
            db.commit()

        print("[+] PostgreSQL seeding complete! Refresh pgAdmin 4 -> Tables to inspect your live data.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()