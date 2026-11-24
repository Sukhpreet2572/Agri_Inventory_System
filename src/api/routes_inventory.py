from datetime import date, datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.schemas import (
    InventoryItemResponse,
    PurchaseOrderCreate,
    PurchaseOrderResponse,
)
from src.database.connection import get_db
from src.database.models import (
    Commodity,
    DailyPriceHistory,
    Inventory,
    Market,
    OrderStatus,
    PurchaseOrder,
    Supplier,
    VendorSale,
)
from src.services.reorder_engine import (
    calculate_reorder_point,
    calculate_safety_stock,
    evaluate_inventory_and_procurement,
)

router = APIRouter(prefix="/api/v1", tags=["Inventory & Orders"])


class InventoryCreateRequest(BaseModel):
    market_id: int
    commodity_id: int
    supplier_id: int = 1
    current_stock_qtl: float = Field(..., ge=0)
    avg_daily_demand_qtl: float = Field(..., gt=0)
    demand_std_qtl: float = Field(4.0, ge=0)


class VendorPurchaseRequest(BaseModel):
    vendor_name: str = Field(..., min_length=2)
    market_id: int
    commodity_id: int
    quantity_qtl: float = Field(..., gt=0)


def _get_latest_price(db: Session, market_id: int, commodity_id: int) -> float:
    latest = (
        db.query(DailyPriceHistory)
        .filter(
            DailyPriceHistory.market_id == market_id,
            DailyPriceHistory.commodity_id == commodity_id,
        )
        .order_by(DailyPriceHistory.price_date.desc())
        .first()
    )
    return float(latest.modal_price) if latest else 1500.0


def _get_pending_po_qty(db: Session, market_id: int, commodity_id: int) -> float:
    total = (
        db.query(func.coalesce(func.sum(PurchaseOrder.quantity_qtl), 0.0))
        .filter(
            PurchaseOrder.market_id == market_id,
            PurchaseOrder.commodity_id == commodity_id,
            PurchaseOrder.status.in_([OrderStatus.PENDING, OrderStatus.SHIPPED]),
        )
        .scalar()
    )
    return float(total or 0.0)


@router.get("/catalog")
def get_markets_commodities_suppliers(db: Session = Depends(get_db)):
    """Return available markets, commodities, and suppliers for Admin/Vendor dropdowns."""
    return {
        "markets": [{"id": m.id, "name": m.name, "state": m.state} for m in db.query(Market).order_by(Market.name).all()],
        "commodities": [{"id": c.id, "name": c.name} for c in db.query(Commodity).order_by(Commodity.name).all()],
        "suppliers": [{"id": s.id, "name": s.name, "lead_time_days": s.lead_time_days} for s in db.query(Supplier).all()],
    }


@router.get("/inventory", response_model=list[InventoryItemResponse])
def list_inventory(
    market_id: int | None = Query(None, description="Filter by Market ID"),
    status: str | None = Query(None, description="Filter by status: CRITICAL, LOW, HEALTHY, OVERSTOCKED"),
    db: Session = Depends(get_db),
):
    query = db.query(Inventory).join(Market).join(Commodity).join(Supplier)
    if market_id is not None:
        query = query.filter(Inventory.market_id == market_id)

    items = query.all()
    results = []

    for inv in items:
        latest_price = _get_latest_price(db, inv.market_id, inv.commodity_id)
        pending_qty = _get_pending_po_qty(db, inv.market_id, inv.commodity_id)

        eval_res = evaluate_inventory_and_procurement(
            current_stock_qtl=inv.current_stock_qtl,
            pending_po_qtl=pending_qty,
            avg_daily_demand_qtl=inv.avg_daily_demand_qtl,
            demand_std_qtl=inv.demand_std_qtl,
            lead_time_days=inv.supplier.lead_time_days,
            current_price=latest_price,
            expected_price_change_pct=0.0,
        )

        if status and eval_res["status"] != status.upper():
            continue

        results.append(
            InventoryItemResponse(
                inventory_id=inv.id,
                market_id=inv.market_id,
                market_name=inv.market.name,
                state=inv.market.state,
                commodity_id=inv.commodity_id,
                commodity_name=inv.commodity.name,
                supplier_name=inv.supplier.name,
                lead_time_days=inv.supplier.lead_time_days,
                latest_modal_price=latest_price,
                **eval_res,
            )
        )
    return results


@router.get("/inventory/alerts", response_model=list[InventoryItemResponse])
def get_reorder_alerts(market_id: int | None = None, db: Session = Depends(get_db)):
    """Return all inventory items currently in CRITICAL or LOW stock status."""
    all_items = list_inventory(market_id=market_id, status=None, db=db)
    alerts = [item for item in all_items if item.status in ("CRITICAL", "LOW")]
    alerts.sort(key=lambda x: (0 if x.status == "CRITICAL" else 1, x.days_of_cover))
    return alerts


@router.post("/inventory", status_code=201)
def add_inventory_item(payload: InventoryCreateRequest, db: Session = Depends(get_db)):
    """Admin: Add a new Market-Commodity SKU into warehouse reorder monitoring."""
    existing = (
        db.query(Inventory)
        .filter(Inventory.market_id == payload.market_id, Inventory.commodity_id == payload.commodity_id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="This Market & Commodity is already tracked in inventory.")

    sup = db.get(Supplier, payload.supplier_id)
    if not sup:
        raise HTTPException(status_code=404, detail="Supplier not found.")

    ss = calculate_safety_stock(payload.demand_std_qtl, sup.lead_time_days)
    rop = calculate_reorder_point(payload.avg_daily_demand_qtl, sup.lead_time_days, ss)

    new_inv = Inventory(
        market_id=payload.market_id,
        commodity_id=payload.commodity_id,
        supplier_id=payload.supplier_id,
        current_stock_qtl=round(payload.current_stock_qtl, 1),
        avg_daily_demand_qtl=round(payload.avg_daily_demand_qtl, 1),
        demand_std_qtl=round(payload.demand_std_qtl, 1),
        safety_stock_qtl=ss,
        reorder_point_qtl=rop,
    )
    db.add(new_inv)
    db.commit()
    db.refresh(new_inv)
    return {"message": "Product added to warehouse monitoring.", "inventory_id": new_inv.id, "reorder_point_qtl": rop}


@router.delete("/inventory/{inventory_id}")
def delete_inventory_item(inventory_id: int, db: Session = Depends(get_db)):
    """Admin: Remove an unnecessary Market-Commodity SKU from warehouse monitoring."""
    inv = db.get(Inventory, inventory_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Inventory item not found.")
    db.delete(inv)
    db.commit()
    return {"message": f"Removed inventory item #{inventory_id} from tracking."}


@router.get("/purchase-orders", response_model=list[PurchaseOrderResponse])
def list_purchase_orders(db: Session = Depends(get_db)):
    orders = db.query(PurchaseOrder).order_by(PurchaseOrder.id.desc()).all()
    output = []
    for po in orders:
        market = db.get(Market, po.market_id)
        comm = db.get(Commodity, po.commodity_id)
        sup = db.get(Supplier, po.supplier_id)
        output.append(
            PurchaseOrderResponse(
                id=po.id,
                market_id=po.market_id,
                market_name=market.name if market else "Unknown",
                commodity_id=po.commodity_id,
                commodity_name=comm.name if comm else "Unknown",
                supplier_name=sup.name if sup else "Unknown",
                quantity_qtl=po.quantity_qtl,
                locked_unit_price=po.locked_unit_price,
                total_cost_inr=round(po.quantity_qtl * po.locked_unit_price, 2),
                procurement_strategy=po.procurement_strategy,
                status=po.status.value,
                order_date=po.order_date,
                expected_delivery_date=po.expected_delivery_date,
            )
        )
    return output


@router.post("/purchase-orders", response_model=PurchaseOrderResponse, status_code=201)
def create_purchase_order(payload: PurchaseOrderCreate, db: Session = Depends(get_db)):
    inv = (
        db.query(Inventory)
        .filter(
            Inventory.market_id == payload.market_id,
            Inventory.commodity_id == payload.commodity_id,
        )
        .first()
    )
    if not inv:
        raise HTTPException(status_code=404, detail="Inventory record not found for this Market & Commodity.")

    latest_price = _get_latest_price(db, payload.market_id, payload.commodity_id)
    lead_time = inv.supplier.lead_time_days

    new_po = PurchaseOrder(
        market_id=payload.market_id,
        commodity_id=payload.commodity_id,
        supplier_id=inv.supplier_id,
        quantity_qtl=round(payload.quantity_qtl, 1),
        locked_unit_price=latest_price,
        procurement_strategy=payload.procurement_strategy,
        status=OrderStatus.PENDING,
        order_date=date.today(),
        expected_delivery_date=date.today() + timedelta(days=lead_time),
    )
    db.add(new_po)
    db.commit()
    db.refresh(new_po)

    return PurchaseOrderResponse(
        id=new_po.id,
        market_id=new_po.market_id,
        market_name=inv.market.name,
        commodity_id=new_po.commodity_id,
        commodity_name=inv.commodity.name,
        supplier_name=inv.supplier.name,
        quantity_qtl=new_po.quantity_qtl,
        locked_unit_price=new_po.locked_unit_price,
        total_cost_inr=round(new_po.quantity_qtl * new_po.locked_unit_price, 2),
        procurement_strategy=new_po.procurement_strategy,
        status=new_po.status.value,
        order_date=new_po.order_date,
        expected_delivery_date=new_po.expected_delivery_date,
    )


@router.patch("/purchase-orders/{po_id}/deliver")
def confirm_po_delivery(po_id: int, db: Session = Depends(get_db)):
    """Admin: Confirm ordered stock reached the warehouse; marks PO DELIVERED and adds stock."""
    po = db.get(PurchaseOrder, po_id)
    if not po:
        raise HTTPException(status_code=404, detail="Purchase Order not found.")
    if po.status == OrderStatus.DELIVERED:
        raise HTTPException(status_code=400, detail="Purchase Order is already marked DELIVERED.")

    inv = (
        db.query(Inventory)
        .filter(Inventory.market_id == po.market_id, Inventory.commodity_id == po.commodity_id)
        .first()
    )
    if inv:
        inv.current_stock_qtl = round(inv.current_stock_qtl + po.quantity_qtl, 1)

    po.status = OrderStatus.DELIVERED
    db.commit()
    return {
        "message": f"PO #{po_id} marked DELIVERED. Added {po.quantity_qtl} qtl to physical warehouse stock.",
        "new_physical_stock_qtl": inv.current_stock_qtl if inv else None,
    }


@router.post("/vendor-orders", status_code=201)
def create_vendor_purchase(payload: VendorPurchaseRequest, db: Session = Depends(get_db)):
    """Vendor Portal: Downstream vendor buys stock from our warehouse, decrementing stock & triggering alerts if low."""
    inv = (
        db.query(Inventory)
        .filter(Inventory.market_id == payload.market_id, Inventory.commodity_id == payload.commodity_id)
        .first()
    )
    if not inv:
        raise HTTPException(status_code=404, detail="Requested commodity is not stocked at this warehouse.")
    if payload.quantity_qtl > inv.current_stock_qtl:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient warehouse stock. Available: {inv.current_stock_qtl} qtl, Requested: {payload.quantity_qtl} qtl.",
        )

    unit_price = _get_latest_price(db, payload.market_id, payload.commodity_id)
    total_amt = round(payload.quantity_qtl * unit_price, 2)

    inv.current_stock_qtl = round(inv.current_stock_qtl - payload.quantity_qtl, 1)
    sale = VendorSale(
        vendor_name=payload.vendor_name.strip(),
        market_id=payload.market_id,
        commodity_id=payload.commodity_id,
        quantity_qtl=round(payload.quantity_qtl, 1),
        unit_sale_price=unit_price,
        total_amount_inr=total_amt,
        sale_timestamp=datetime.utcnow(),
    )
    db.add(sale)
    db.commit()

    below_rop = inv.current_stock_qtl <= inv.reorder_point_qtl
    return {
        "message": f"Vendor order confirmed for {payload.vendor_name}. Deducted {payload.quantity_qtl} qtl from warehouse.",
        "remaining_stock_qtl": inv.current_stock_qtl,
        "reorder_alert_triggered": below_rop,
    }


@router.get("/vendor-orders")
def list_vendor_purchases(db: Session = Depends(get_db)):
    """List recent downstream vendor purchases."""
    sales = db.query(VendorSale).order_by(VendorSale.id.desc()).limit(50).all()
    out = []
    for s in sales:
        m = db.get(Market, s.market_id)
        c = db.get(Commodity, s.commodity_id)
        out.append(
            {
                "id": s.id,
                "vendor_name": s.vendor_name,
                "market_name": m.name if m else "Unknown",
                "commodity_name": c.name if c else "Unknown",
                "quantity_qtl": s.quantity_qtl,
                "unit_sale_price": s.unit_sale_price,
                "total_amount_inr": s.total_amount_inr,
                "sale_timestamp": s.sale_timestamp.strftime("%Y-%m-%d %H:%M"),
            }
        )
    return out