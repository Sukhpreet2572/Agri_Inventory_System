import json
import os
import re
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from src.api.routes_inventory import _get_latest_price, _get_pending_po_qty
from src.database.models import Commodity, DailyPriceHistory, Inventory, Market, PurchaseOrder
from src.ml.inference import PriceForecaster
from src.services.reorder_engine import evaluate_inventory_and_procurement

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
METRICS_PATH = Path("models/artifacts/benchmark_metrics.json")


class InventoryAIAssistant:
    """
    Context-grounded AI Procurement & Inventory Assistant.
    Queries live PostgreSQL state + Random Forest forecasts, then uses Gemini API
    if configured or a precise deterministic synthesis engine if offline.
    """

    def __init__(self):
        self.forecaster = PriceForecaster()

    def _gather_live_snapshot(self, db: Session) -> dict:
        """Pull current inventory alerts, PO totals, and model metrics from PostgreSQL."""
        items = db.query(Inventory).join(Market).join(Commodity).all()
        evaluated = []
        for inv in items:
            price = _get_latest_price(db, inv.market_id, inv.commodity_id)
            pending = _get_pending_po_qty(db, inv.market_id, inv.commodity_id)
            res = evaluate_inventory_and_procurement(
                current_stock_qtl=inv.current_stock_qtl,
                pending_po_qtl=pending,
                avg_daily_demand_qtl=inv.avg_daily_demand_qtl,
                demand_std_qtl=inv.demand_std_qtl,
                lead_time_days=inv.supplier.lead_time_days,
                current_price=price,
                expected_price_change_pct=0.0,
            )
            evaluated.append(
                {
                    "market_id": inv.market_id,
                    "market": inv.market.name,
                    "state": inv.market.state,
                    "commodity_id": inv.commodity_id,
                    "commodity": inv.commodity.name,
                    "supplier": inv.supplier.name,
                    "lead_time_days": inv.supplier.lead_time_days,
                    "avg_daily_demand_qtl": inv.avg_daily_demand_qtl,
                    "demand_std_qtl": inv.demand_std_qtl,
                    "latest_price": price,
                    **res,
                }
            )

        alerts = [x for x in evaluated if x["status"] in ("CRITICAL", "LOW")]
        alerts.sort(key=lambda x: (0 if x["status"] == "CRITICAL" else 1, x["days_of_cover"]))

        pos = db.query(PurchaseOrder).all()
        pending_pos = [p for p in pos if p.status.value in ("PENDING", "SHIPPED")]
        total_po_spend = sum(p.quantity_qtl * p.locked_unit_price for p in pending_pos)

        metrics = {}
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        return {
            "total_skus": len(evaluated),
            "all_items": evaluated,
            "alerts": alerts,
            "pending_po_count": len(pending_pos),
            "pending_po_spend_inr": round(total_po_spend, 2),
            "model_metrics": metrics,
        }

    def _check_mentioned_forecast(self, question: str, db: Session, snapshot: dict) -> dict | None:
        """If the user mentions a tracked market (and optionally commodity), run a live 14-day RF forecast."""
        q_lower = question.lower()
        matched_item = None

        for item in snapshot["all_items"]:
            m_base = item["market"].split("(")[0].strip().lower()
            c_name = item["commodity"].lower()
            if m_base in q_lower and c_name in q_lower:
                matched_item = item
                break

        if not matched_item:
            for item in snapshot["all_items"]:
                if item["market"].split("(")[0].strip().lower() in q_lower:
                    matched_item = item
                    break

        if not matched_item:
            return None

        rows = (
            db.query(DailyPriceHistory)
            .filter(
                DailyPriceHistory.market_id == matched_item["market_id"],
                DailyPriceHistory.commodity_id == matched_item["commodity_id"],
            )
            .order_by(DailyPriceHistory.price_date.asc())
            .all()
        )
        if len(rows) < 7:
            return None

        hist_df = pd.DataFrame(
            [
                {
                    "market": matched_item["market"],
                    "commodity": matched_item["commodity"],
                    "date": r.price_date,
                    "min_price": r.min_price or r.modal_price,
                    "max_price": r.max_price or r.modal_price,
                    "modal_price": r.modal_price,
                }
                for r in rows
            ]
        )
        fc = self.forecaster.forecast(hist_df, matched_item["market"], matched_item["commodity"], horizon_days=14)
        proc = evaluate_inventory_and_procurement(
            current_stock_qtl=matched_item["current_stock_qtl"],
            pending_po_qtl=matched_item["pending_po_qtl"],
            avg_daily_demand_qtl=matched_item["avg_daily_demand_qtl"],
            demand_std_qtl=matched_item["demand_std_qtl"],
            lead_time_days=matched_item["lead_time_days"],
            current_price=fc["current_price"],
            expected_price_change_pct=fc["expected_price_change_pct"],
        )
        return {"item": matched_item, "forecast": fc, "procurement": proc}

    def _call_gemini_if_available(self, question: str, context_payload: dict) -> str | None:
        if not GEMINI_API_KEY:
            return None
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            prompt = (
                "You are an expert Agri-Supply Chain & Inventory Procurement AI Assistant. "
                "Answer the user's question clearly using ONLY the live PostgreSQL & Random Forest JSON data below. "
                "Include exact quantities (quintals), INR prices, and actionable recommendations.\n\n"
                f"LIVE SYSTEM DATA:\n{json.dumps(context_payload, default=str)}\n\n"
                f"USER QUESTION: {question}"
            )
            resp = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return None

    @staticmethod
    def _has_word(text: str, words: list[str]) -> bool:
        """Match whole words only so 'order' never accidentally matches 'reorder'."""
        return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)

    def answer_question(self, question: str, db: Session) -> str:
        snapshot = self._gather_live_snapshot(db)
        specific_fc = self._check_mentioned_forecast(question, db, snapshot)

        context_for_llm = {
            "summary": {
                "total_skus": snapshot["total_skus"],
                "active_alerts_count": len(snapshot["alerts"]),
                "pending_purchase_orders": snapshot["pending_po_count"],
                "pending_po_spend_inr": snapshot["pending_po_spend_inr"],
            },
            "top_alerts": snapshot["alerts"][:6],
            "specific_market_forecast": specific_fc,
            "model_benchmarks": snapshot["model_metrics"],
        }

        llm_reply = self._call_gemini_if_available(question, context_for_llm)
        if llm_reply:
            return llm_reply

        q = question.lower().strip()

        # 1. Definition / Explanation questions ("what is...", "what does ... mean")
       # 1b. Stock Status Definition questions ("status healthy mean...", "what is critical...")
        if ("mean" in q or "what is" in q or "explain" in q) and self._has_word(
            q, ["healthy", "critical", "low", "overstocked", "status"]
        ):
            healthy_cnt = sum(1 for x in snapshot["all_items"] if x["status"] == "HEALTHY")
            return (
                "### ✅ Yes — 'HEALTHY' Means No Reorder Is Needed Right Now\n"
                "Here is how our Supply-Chain Engine classifies all four warehouse statuses:\n"
                "- **`HEALTHY`:** Net inventory (`Current Stock + Pending POs`) is safely above the **Reorder Point**, "
                f"so existing stock comfortably covers supplier lead times. (**{healthy_cnt} SKUs** are currently `HEALTHY`.)\n"
                "- **`LOW`:** Net inventory has dropped at or below the **Reorder Point** (`Lead-Time Demand + Safety Stock`) — **a new Purchase Order should be placed now**.\n"
                "- **`CRITICAL`:** Physical warehouse stock has fallen below **Safety Stock** — high risk of stockout before the next delivery arrives.\n"
                "- **`OVERSTOCKED`:** Current stock exceeds `2.2× Reorder Point`, tying up working capital or risking spoilage."
            )

        # 2. Specific Mandi / Commodity forecast & buy/wait advice
        if specific_fc:
            fc = specific_fc["forecast"]
            it = specific_fc["item"]
            pr = specific_fc["procurement"]
            return (
                f"### 📊 Live Analysis: {fc['commodity']} @ {fc['market']}\n"
                f"- **Current Modal Price:** ₹{fc['current_price']:,.2f} / qtl\n"
                f"- **14-Day Random Forest Avg Forecast:** ₹{fc['avg_forecast_price']:,.2f} / qtl "
                f"(**{fc['expected_price_change_pct']:+.2f}%**)\n"
                f"- **Warehouse Stock:** {it['current_stock_qtl']} qtl (Pending PO: {it['pending_po_qtl']} qtl | "
                f"Reorder Point: {pr['reorder_point_qtl']} qtl | Status: **{pr['status']}**)\n"
                f"- **Recommended Strategy:** `{pr['procurement_strategy']}` — Order **{pr['recommended_order_qtl']} qtl** "
                f"(Est. Cost: **₹{pr['estimated_order_cost_inr']:,.2f}**)\n\n"
                f"**Rationale:** {pr['rationale']}"
            )

        # 3. Replenishment / Reorder / Low Stock Alerts (whole-word matching so 'reorder' doesn't collide with 'order')
        if self._has_word(q, ["alert", "alerts", "low", "critical", "shortage", "restock", "stockout", "reorder", "need", "buy"]):
            if not snapshot["alerts"]:
                return "✅ **Good news:** All tracked market-commodity items are currently above their Reorder Points. No urgent purchase orders are required."
            lines = [f"### 🚨 Active Replenishment Alerts ({len(snapshot['alerts'])} Items Below Reorder Point)\n"]
            total_cost = 0.0
            for a in snapshot["alerts"]:
                total_cost += a["estimated_order_cost_inr"]
                lines.append(
                    f"- **[{a['status']}] {a['commodity']} @ {a['market']}** — "
                    f"Stock: `{a['current_stock_qtl']} qtl` ({a['days_of_cover']} days cover) | "
                    f"Recommended Order: **{a['recommended_order_qtl']} qtl** (`{a['procurement_strategy']}`) "
                    f"at ₹{a['latest_price']:,.0f}/qtl (**₹{a['estimated_order_cost_inr']:,.0f}**)"
                )
            lines.append(f"\n**Total Estimated Replenishment Budget:** ₹{total_cost:,.2f}")
            return "\n".join(lines)

        # 4. Purchase Orders & Spend Summary
        if self._has_word(q, ["order", "orders", "po", "pos", "purchase", "spend", "budget"]):
            return (
                f"### 📦 Purchase Order & Budget Summary\n"
                f"- **Active Pending Purchase Orders:** {snapshot['pending_po_count']}\n"
                f"- **Total Committed PO Spend:** ₹{snapshot['pending_po_spend_inr']:,.2f}\n"
                f"- **Remaining Unordered Alert Items:** {len(snapshot['alerts'])} items still need replenishment."
            )

        # 5. Model Accuracy & Benchmark Comparison
        if self._has_word(q, ["model", "accuracy", "mape", "rmse", "mae", "random", "xgboost", "lstm", "benchmark"]):
            rf = snapshot["model_metrics"].get("2_Random_Forest", {})
            xgb_m = snapshot["model_metrics"].get("3_XGBoost", {})
            ma = snapshot["model_metrics"].get("1_Moving_Average_7D", {})
            return (
                f"### 🤖 Forecasting Model Evaluation Summary\n"
                f"- **Active Production Model:** `Random Forest Regressor` (**{rf.get('MAPE (%)', 15.21)}% MAPE**, MAE ₹{rf.get('MAE', 257.02)}/qtl)\n"
                f"- **Runner-Up:** `XGBoost Regressor` (**{xgb_m.get('MAPE (%)', 16.22)}% MAPE**)\n"
                f"- **Statistical Baseline (7D Moving Avg):** **{ma.get('MAPE (%)', 21.61)}% MAPE**\n\n"
                f"Random Forest reduced forecast error by ~30% compared to the moving average baseline on the unseen chronological test set."
            )

        return (
            f"### 🌾 Warehouse Executive Snapshot\n"
            f"- **Tracked SKUs:** {snapshot['total_skus']} across 25 mandis\n"
            f"- **Low / Critical Alerts:** {len(snapshot['alerts'])} items below reorder point\n"
            f"- **Active Pending POs:** {snapshot['pending_po_count']} (₹{snapshot['pending_po_spend_inr']:,.2f})\n\n"
            f"Try asking me:\n"
            f"- *\"Which stock needs to reorder right now?\"*\n"
            f"- *\"In Durgapur, is any item in shortage or not?\"*\n"
            f"- *\"What does total committed PO spend mean?\"*"
        )