from datetime import date, datetime
from pydantic import BaseModel, Field


class InventoryItemResponse(BaseModel):
    inventory_id: int
    market_id: int
    market_name: str
    state: str
    commodity_id: int
    commodity_name: str
    supplier_name: str
    lead_time_days: int
    latest_modal_price: float
    status: str
    urgency: str = "NONE"
    current_stock_qtl: float
    pending_po_qtl: float
    net_inventory_qtl: float
    safety_stock_qtl: float
    reorder_point_qtl: float
    days_of_cover: float
    recommended_order_qtl: float
    procurement_strategy: str
    estimated_order_cost_inr: float
    rationale: str


class PurchaseOrderCreate(BaseModel):
    market_id: int
    commodity_id: int
    quantity_qtl: float = Field(..., gt=0, description="Quantity in quintals to order")
    procurement_strategy: str = "STANDARD_REORDER"


class PurchaseOrderResponse(BaseModel):
    id: int
    market_id: int
    market_name: str
    commodity_id: int
    commodity_name: str
    supplier_name: str
    quantity_qtl: float
    locked_unit_price: float
    total_cost_inr: float
    procurement_strategy: str
    status: str
    order_date: date
    expected_delivery_date: date


class ForecastPoint(BaseModel):
    date: str
    predicted_price: float
    lower_bound: float
    upper_bound: float


class HistoricalPoint(BaseModel):
    date: str
    actual_price: float


class ForecastAndProcurementResponse(BaseModel):
    market_id: int
    market_name: str
    commodity_id: int
    commodity_name: str
    model_used: str
    current_price: float
    avg_forecast_price: float
    expected_price_change_pct: float
    horizon_days: int
    historical_tail: list[HistoricalPoint]
    forecast: list[ForecastPoint]
    procurement_recommendation: dict