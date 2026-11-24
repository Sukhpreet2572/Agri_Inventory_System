import math


def calculate_safety_stock(
    demand_std_qtl: float, lead_time_days: int, service_level_z: float = 1.65
) -> float:
    """Calculate buffer stock required to protect against demand variability during lead time."""
    if demand_std_qtl <= 0 or lead_time_days <= 0:
        return 0.0
    return round(service_level_z * demand_std_qtl * math.sqrt(lead_time_days), 1)


def calculate_reorder_point(
    avg_daily_demand_qtl: float, lead_time_days: int, safety_stock_qtl: float
) -> float:
    """Calculate inventory threshold at which a new purchase order must be triggered."""
    if avg_daily_demand_qtl <= 0:
        return round(safety_stock_qtl, 1)
    return round((avg_daily_demand_qtl * max(1, lead_time_days)) + safety_stock_qtl, 1)


def evaluate_inventory_and_procurement(
    current_stock_qtl: float,
    pending_po_qtl: float,
    avg_daily_demand_qtl: float,
    demand_std_qtl: float,
    lead_time_days: int,
    current_price: float,
    expected_price_change_pct: float = 0.0,
) -> dict:
    """
    Combine warehouse stock health with ML price trend forecasts to determine
    exact procurement strategy, urgency level, and order quantity.
    """
    safety_stock = calculate_safety_stock(demand_std_qtl, lead_time_days)
    reorder_point = calculate_reorder_point(avg_daily_demand_qtl, lead_time_days, safety_stock)
    net_inventory = round(current_stock_qtl + pending_po_qtl, 1)
    days_of_cover = round(current_stock_qtl / avg_daily_demand_qtl, 1) if avg_daily_demand_qtl > 0 else 999.0

    # 1. Stock Health Classification
    if current_stock_qtl <= safety_stock:
        status = "CRITICAL"
    elif net_inventory <= reorder_point:
        status = "LOW"
    elif current_stock_qtl > reorder_point * 2.2:
        status = "OVERSTOCKED"
    else:
        status = "HEALTHY"

    # 2. Price-Aware Procurement Decision Matrix
    if net_inventory <= reorder_point:
        if expected_price_change_pct >= 15.0:
            target_cover_days = 14
            strategy = "URGENT_BULK_BUY"
            urgency = "CRITICAL_URGENCY"
            rationale = (
                f"Low stock ({net_inventory} qtl <= ROP {reorder_point} qtl) combined with a sharp "
                f"+{expected_price_change_pct:.1f}% price surge. Place an immediate 14-day bulk order "
                f"at INR {current_price}/qtl before prices spike."
            )
        elif expected_price_change_pct >= 5.0:
            target_cover_days = 10
            strategy = "EARLY_REORDER"
            urgency = "HIGH_URGENCY"
            rationale = (
                f"Stock below ROP ({reorder_point} qtl) and prices are rising moderately "
                f"(+{expected_price_change_pct:.1f}%). Ordering 10-day cover now to beat gradual price increases."
            )
        elif expected_price_change_pct <= -5.0:
            target_cover_days = max(3, lead_time_days + 1)
            strategy = "MIN_JIT_BUY"
            urgency = "LOW_URGENCY"
            rationale = (
                f"Stock below ROP ({reorder_point} qtl), but prices are forecasted to drop "
                f"({expected_price_change_pct:.1f}%). Ordering minimal {target_cover_days}-day cover and waiting for cheaper rates."
            )
        else:
            target_cover_days = 7
            strategy = "STANDARD_REORDER"
            urgency = "NORMAL_URGENCY"
            rationale = (
                f"Stock below ROP ({reorder_point} qtl) with stable prices ({expected_price_change_pct:+.1f}%). "
                f"Ordering standard 7-day replenishment."
            )

        target_stock = (avg_daily_demand_qtl * (lead_time_days + target_cover_days)) + safety_stock
        recommended_qty = round(max(0.0, target_stock - net_inventory), 1)

    elif status == "HEALTHY" and expected_price_change_pct >= 15.0:
        target_cover_days = 14
        target_stock = (avg_daily_demand_qtl * (lead_time_days + target_cover_days)) + safety_stock
        recommended_qty = round(max(0.0, target_stock - net_inventory), 1)
        if recommended_qty > 0:
            strategy = "PROACTIVE_SURGE_BUY"
            urgency = "MEDIUM_URGENCY"
            rationale = (
                f"Stock is HEALTHY ({net_inventory} qtl > ROP {reorder_point} qtl), but ML predicts a sharp "
                f"+{expected_price_change_pct:.1f}% surge. Forward-buying {recommended_qty} qtl today "
                f"locks in INR {current_price}/qtl before the spike."
            )
        else:
            strategy = "NO_ACTION_NEEDED"
            urgency = "NONE"
            rationale = f"Net stock ({net_inventory} qtl) already covers a full 14-day surge buffer."

    elif status == "HEALTHY" and 5.0 <= expected_price_change_pct < 15.0:
        recommended_qty = 0.0
        strategy = "WATCHLIST_MODERATE_RISE"
        urgency = "LOW_URGENCY"
        rationale = (
            f"Prices are rising moderately (+{expected_price_change_pct:.1f}%), but warehouse stock is HEALTHY "
            f"({net_inventory} qtl vs ROP {reorder_point} qtl). No urgent order needed yet—keep on watchlist."
        )

    else:
        recommended_qty = 0.0
        strategy = "NO_ACTION_NEEDED"
        urgency = "NONE"
        if expected_price_change_pct <= -5.0:
            rationale = (
                f"Stock is sufficient ({net_inventory} qtl) and prices are falling ({expected_price_change_pct:.1f}%). "
                f"No action needed—hold off purchasing."
            )
        else:
            rationale = f"Net stock ({net_inventory} qtl) is safely above reorder point ({reorder_point} qtl)."

    estimated_order_cost = round(recommended_qty * current_price, 2)

    return {
        "status": status,
        "urgency": urgency,
        "current_stock_qtl": round(current_stock_qtl, 1),
        "pending_po_qtl": round(pending_po_qtl, 1),
        "net_inventory_qtl": net_inventory,
        "safety_stock_qtl": safety_stock,
        "reorder_point_qtl": reorder_point,
        "days_of_cover": days_of_cover,
        "recommended_order_qtl": recommended_qty,
        "procurement_strategy": strategy,
        "estimated_order_cost_inr": estimated_order_cost,
        "rationale": rationale,
    }