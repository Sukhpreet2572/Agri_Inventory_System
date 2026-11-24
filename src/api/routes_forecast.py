import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.routes_inventory import _get_pending_po_qty
from src.api.schemas import ForecastAndProcurementResponse
from src.database.connection import get_db
from src.database.models import Commodity, DailyPriceHistory, Inventory, Market
from src.ml.inference import PriceForecaster
from src.services.live_sync import sync_live_gov_prices
from src.services.reorder_engine import evaluate_inventory_and_procurement

router = APIRouter(prefix="/api/v1", tags=["Forecasting & Live Sync"])

# Load trained Random Forest model once when API starts
forecaster = PriceForecaster()


@router.get("/forecast/{market_id}/{commodity_id}", response_model=ForecastAndProcurementResponse)
def get_price_forecast_and_procurement(
    market_id: int,
    commodity_id: int,
    horizon: int = Query(14, ge=3, le=30, description="Forecast horizon in days (3 to 30)"),
    db: Session = Depends(get_db),
):
    market = db.get(Market, market_id)
    comm = db.get(Commodity, commodity_id)
    if not market or not comm:
        raise HTTPException(status_code=404, detail="Market or Commodity not found.")

    rows = (
        db.query(DailyPriceHistory)
        .filter(
            DailyPriceHistory.market_id == market_id,
            DailyPriceHistory.commodity_id == commodity_id,
        )
        .order_by(DailyPriceHistory.price_date.asc())
        .all()
    )
    if len(rows) < 7:
        raise HTTPException(status_code=400, detail="Not enough historical price rows to forecast.")

    history_df = pd.DataFrame(
        [
            {
                "market": market.name,
                "commodity": comm.name,
                "date": r.price_date,
                "min_price": r.min_price or r.modal_price,
                "max_price": r.max_price or r.modal_price,
                "modal_price": r.modal_price,
            }
            for r in rows
        ]
    )

    forecast_res = forecaster.forecast(
        history_df=history_df,
        market_name=market.name,
        commodity_name=comm.name,
        horizon_days=horizon,
    )

    # Combine ML price trend with warehouse inventory to produce price-aware procurement advice
    inv = (
        db.query(Inventory)
        .filter(Inventory.market_id == market_id, Inventory.commodity_id == commodity_id)
        .first()
    )
    if inv:
        pending_qty = _get_pending_po_qty(db, market_id, commodity_id)
        procurement = evaluate_inventory_and_procurement(
            current_stock_qtl=inv.current_stock_qtl,
            pending_po_qtl=pending_qty,
            avg_daily_demand_qtl=inv.avg_daily_demand_qtl,
            demand_std_qtl=inv.demand_std_qtl,
            lead_time_days=inv.supplier.lead_time_days,
            current_price=forecast_res["current_price"],
            expected_price_change_pct=forecast_res["expected_price_change_pct"],
        )
    else:
        procurement = {"message": "No warehouse inventory record tracked for this pair."}

    return ForecastAndProcurementResponse(
        market_id=market_id,
        market_name=market.name,
        commodity_id=commodity_id,
        commodity_name=comm.name,
        model_used=forecast_res["model_used"],
        current_price=forecast_res["current_price"],
        avg_forecast_price=forecast_res["avg_forecast_price"],
        expected_price_change_pct=forecast_res["expected_price_change_pct"],
        horizon_days=horizon,
        historical_tail=forecast_res["historical_tail"],
        forecast=forecast_res["forecast"],
        procurement_recommendation=procurement,
    )


@router.post("/sync/live-gov-data")
def trigger_live_gov_sync(db: Session = Depends(get_db)):
    """Pull today's live market prices from data.gov.in and upsert into PostgreSQL."""
    return sync_live_gov_prices(db)


@router.get("/buying-opportunities")
def get_price_timing_opportunities(db: Session = Depends(get_db)):
    """Scan tracked warehouse SKUs with Random Forest to identify best 'Buy Now (Price Low)' vs 'Wait (Price High)' windows."""
    items = db.query(Inventory).join(Market).join(Commodity).limit(20).all()
    opportunities = []

    for inv in items:
        rows = (
            db.query(DailyPriceHistory)
            .filter(
                DailyPriceHistory.market_id == inv.market_id,
                DailyPriceHistory.commodity_id == inv.commodity_id,
            )
            .order_by(DailyPriceHistory.price_date.asc())
            .all()
        )
        if len(rows) < 7:
            continue

        hist_df = pd.DataFrame(
            [
                {
                    "market": inv.market.name,
                    "commodity": inv.commodity.name,
                    "date": r.price_date,
                    "min_price": r.min_price or r.modal_price,
                    "max_price": r.max_price or r.modal_price,
                    "modal_price": r.modal_price,
                }
                for r in rows
            ]
        )
        fc = forecaster.forecast(hist_df, inv.market.name, inv.commodity.name, horizon_days=14)
        chg = fc["expected_price_change_pct"]

        if chg >= 5.0:
            timing_signal = "🟢 BUY NOW (Price Low — Surge Ahead)"
        elif chg <= -5.0:
            timing_signal = "🟡 WAIT / MIN BUY (Price High — Drop Ahead)"
        else:
            timing_signal = "⚪ NEUTRAL (Stable Price)"

        opportunities.append(
            {
                "market_name": inv.market.name,
                "commodity_name": inv.commodity.name,
                "current_price_inr": fc["current_price"],
                "forecast_14d_avg_inr": fc["avg_forecast_price"],
                "expected_change_pct": chg,
                "timing_signal": timing_signal,
                "current_stock_qtl": inv.current_stock_qtl,
            }
        )

    opportunities.sort(key=lambda x: x["expected_change_pct"], reverse=True)
    return opportunities