import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Smart Agri-Inventory & Price Forecasting",
    page_icon="🌾",
    layout="wide",
)


def fetch_json(endpoint: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend at http://127.0.0.1:8000. Ensure `uvicorn src.api.main:app --reload` is running.")
        st.stop()
    except Exception as exc:
        st.error(f"API Error ({endpoint}): {exc}")
        return None


# ---------------- SIDEBAR & ROLE SWITCHER ----------------
st.sidebar.title("🌾 Smart Agri-Supply")
role = st.sidebar.radio(
    "👤 Active Portal View:",
    ["🛡️ Warehouse Admin", "🏪 Vendor Purchase Portal"],
)
st.sidebar.divider()

if st.sidebar.button("🔄 Sync Live data.gov.in Prices", use_container_width=True):
    with st.spinner("Fetching live government mandi prices..."):
        try:
            sync_resp = requests.post(f"{API_BASE}/api/v1/sync/live-gov-data", timeout=45)
            sync_resp.raise_for_status()
            sync_data = sync_resp.json()
            st.sidebar.success(
                f"Scanned {sync_data['total_api_rows_scanned']} live rows | "
                f"Upserted {sync_data['matched_records_upserted']} matched market prices!"
            )
        except Exception as e:
            st.sidebar.error(f"Sync failed: {e}")

st.sidebar.divider()
st.sidebar.markdown("**System Stack**")
st.sidebar.markdown(
    "- **ML Forecaster:** Random Forest (`15.21% MAPE`)\n"
    "- **Database:** PostgreSQL 18\n"
    "- **API:** FastAPI REST\n"
    "- **Data:** AGMARKNET + Live `data.gov.in`"
)

# Shared live data
inventory_data = fetch_json("/api/v1/inventory") or []
alerts_data = fetch_json("/api/v1/inventory/alerts") or []
po_data = fetch_json("/api/v1/purchase-orders") or []
catalog_data = fetch_json("/api/v1/catalog") or {"markets": [], "commodities": [], "suppliers": []}

inv_df = pd.DataFrame(inventory_data)
alerts_df = pd.DataFrame(alerts_data)
po_df = pd.DataFrame(po_data)

# ==============================================================================
# PORTAL 1: VENDOR PURCHASE PORTAL (Downstream Buyers Ordering From Warehouse)
# ==============================================================================
if role == "🏪 Vendor Purchase Portal":
    st.title("🏪 Vendor Wholesale Purchase Portal")
    st.caption("Downstream retailers & distributors purchase commodities from our warehouse—automatically decrementing stock and triggering Admin reorder alerts when thresholds are crossed.")

    if inv_df.empty:
        st.warning("No warehouse stock available right now.")
    else:
        col_form, col_info = st.columns([1.4, 1])
        with col_form:
            st.subheader("Place Wholesale Stock Order")
            vendor_name = st.selectbox(
                "Vendor / Buyer Organization:",
                ["FreshMart Retail Pvt Ltd", "Metro Agro Distributors", "City Mandi Wholesalers", "GreenBasket Supermarkets"],
            )
            market_options = sorted(inv_df["market_name"].unique().tolist())
            v_market = st.selectbox("Select Warehouse Market:", market_options)

            market_skus = inv_df[inv_df["market_name"] == v_market]
            comm_options = market_skus["commodity_name"].tolist()
            v_comm = st.selectbox("Select Commodity:", comm_options)

            selected_sku = market_skus[market_skus["commodity_name"] == v_comm].iloc[0]
            avail_stock = float(selected_sku["current_stock_qtl"])
            unit_price = float(selected_sku["latest_modal_price"])

            max_buy = max(1.0, avail_stock)
            buy_qty = st.number_input(
                f"Order Quantity in Quintals (Available: {avail_stock} qtl):",
                min_value=1.0,
                max_value=max_buy,
                value=min(25.0, max_buy),
                step=5.0,
                disabled=(avail_stock <= 0),
            )
            st.write(f"**Unit Price:** ₹{unit_price:,.2f} / qtl  |  **Total Invoice:** ₹{buy_qty * unit_price:,.2f}")

            if st.button("🛒 Confirm Vendor Purchase & Dispatch Stock", type="primary", disabled=(avail_stock <= 0)):
                res = requests.post(
                    f"{API_BASE}/api/v1/vendor-orders",
                    json={
                        "vendor_name": vendor_name,
                        "market_id": int(selected_sku["market_id"]),
                        "commodity_id": int(selected_sku["commodity_id"]),
                        "quantity_qtl": float(buy_qty),
                    },
                    timeout=15,
                )
                if res.status_code == 201:
                    data = res.json()
                    st.success(data["message"])
                    if data["reorder_alert_triggered"]:
                        st.warning(
                            f"⚠️ Reorder Alert Triggered! Remaining stock ({data['remaining_stock_qtl']} qtl) "
                            f"dropped below the Reorder Point. Switch to **Warehouse Admin** to restock!"
                        )
                    st.rerun()
                else:
                    st.error(f"Order failed: {res.text}")

        with col_info:
            st.subheader("Live Warehouse Shelf Snapshot")
            st.metric("Available Stock", f"{avail_stock} qtl")
            st.metric("Reorder Point Threshold", f"{selected_sku['reorder_point_qtl']} qtl")
            st.metric("Current Stock Status", selected_sku["status"])

    st.divider()
    st.subheader("Recent Vendor Dispatches Ledger")
    vendor_sales = fetch_json("/api/v1/vendor-orders") or []
    if vendor_sales:
        st.dataframe(pd.DataFrame(vendor_sales), use_container_width=True, hide_index=True)
    else:
        st.info("No vendor purchases recorded yet. Place an order above to test live stock deduction!")

# ==============================================================================
# PORTAL 2: WAREHOUSE ADMIN PORTAL
# ==============================================================================
else:
    st.title("🛡️ Warehouse Admin — Smart Inventory & Price Forecasting")

    col1, col2, col3, col4 = st.columns(4)
    total_skus = len(inv_df)
    critical_low_count = len(alerts_df)
    pending_pos = len(po_df[po_df["status"] == "PENDING"]) if not po_df.empty else 0
    total_rec_spend = float(alerts_df["estimated_order_cost_inr"].sum()) if not alerts_df.empty else 0.0

    col1.metric("Tracked Market-Commodity SKUs", f"{total_skus}")
    col2.metric("Low / Critical Stock Alerts", f"{critical_low_count}", delta=f"{critical_low_count} need reorder", delta_color="inverse")
    col3.metric("Active Pending Purchase Orders", f"{pending_pos}")
    col4.metric("Recommended Replenishment Budget", f"₹{total_rec_spend:,.0f}")

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🚨 Reorder Alerts & SKU Admin",
            "📈 Price Forecast & Best Time to Buy",
            "📦 Purchase Orders & Receiving",
            "📊 Model Benchmarks",
            "🤖 AI Inventory Assistant",
        ]
    )

    # === TAB 1: REORDER ALERTS & ADMIN SKU MANAGEMENT ===
    with tab1:
        st.subheader("Urgent Replenishment Queue (Stock Below Reorder Point)")
        if alerts_df.empty:
            st.success("All warehouse items are currently above their Reorder Points!")
        else:
            display_cols = [
                "status",
                "urgency",
                "market_name",
                "commodity_name",
                "current_stock_qtl",
                "pending_po_qtl",
                "reorder_point_qtl",
                "days_of_cover",
                "latest_modal_price",
                "recommended_order_qtl",
                "procurement_strategy",
                "estimated_order_cost_inr",
            ]
            st.dataframe(alerts_df[display_cols], use_container_width=True, hide_index=True)

            st.markdown("#### Quick 1-Click Purchase Order Execution")
            c_sel, c_btn = st.columns([3, 1])
            alert_options = {
                f"[{r['status']}] {r['commodity_name']} @ {r['market_name']} — Order {r['recommended_order_qtl']} qtl (₹{r['estimated_order_cost_inr']:,.0f})": r
                for _, r in alerts_df.iterrows()
                if r["recommended_order_qtl"] > 0
            }
            if alert_options:
                chosen_label = c_sel.selectbox("Select Alert Item to Order:", list(alert_options.keys()))
                chosen_row = alert_options[chosen_label]
                if c_btn.button("✅ Place Recommended PO", type="primary", use_container_width=True):
                    payload = {
                        "market_id": int(chosen_row["market_id"]),
                        "commodity_id": int(chosen_row["commodity_id"]),
                        "quantity_qtl": float(chosen_row["recommended_order_qtl"]),
                        "procurement_strategy": str(chosen_row["procurement_strategy"]),
                    }
                    res = requests.post(f"{API_BASE}/api/v1/purchase-orders", json=payload, timeout=15)
                    if res.status_code == 201:
                        st.success(f"Purchase Order created for {chosen_row['commodity_name']} @ {chosen_row['market_name']}!")
                        st.rerun()
                    else:
                        st.error(f"Failed to create PO: {res.text}")

        st.divider()
        st.subheader("🛠️ Admin Inventory Controls (Add or Remove Monitored SKUs)")
        admin_c1, admin_c2 = st.columns(2)

        with admin_c1:
            with st.expander("➕ Add New Market-Commodity SKU to Monitoring"):
                m_dict = {f"{m['name']} ({m['state']})": m["id"] for m in catalog_data["markets"]}
                c_dict = {c["name"]: c["id"] for c in catalog_data["commodities"]}
                s_dict = {f"{s['name']} (Lead: {s['lead_time_days']}d)": s["id"] for s in catalog_data["suppliers"]}

                new_m = st.selectbox("Market:", list(m_dict.keys()), key="add_m")
                new_c = st.selectbox("Commodity:", list(c_dict.keys()), key="add_c")
                new_s = st.selectbox("Assigned Supplier:", list(s_dict.keys()), key="add_s")
                init_stock = st.number_input("Initial Physical Stock (qtl):", min_value=0.0, value=40.0, step=10.0)
                daily_dem = st.number_input("Avg Daily Demand (qtl/day):", min_value=1.0, value=20.0, step=2.0)

                if st.button("Add SKU to Reorder Monitoring", type="primary"):
                    res = requests.post(
                        f"{API_BASE}/api/v1/inventory",
                        json={
                            "market_id": m_dict[new_m],
                            "commodity_id": c_dict[new_c],
                            "supplier_id": s_dict[new_s],
                            "current_stock_qtl": init_stock,
                            "avg_daily_demand_qtl": daily_dem,
                            "demand_std_qtl": round(daily_dem * 0.25, 1),
                        },
                        timeout=15,
                    )
                    if res.status_code == 201:
                        st.success(res.json()["message"])
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", res.text))

        with admin_c2:
            with st.expander("🗑️ Remove Unnecessary SKU from Monitoring"):
                if not inv_df.empty:
                    del_map = {
                        f"#{r['inventory_id']} — {r['commodity_name']} @ {r['market_name']} (Stock: {r['current_stock_qtl']} qtl)": int(r["inventory_id"])
                        for _, r in inv_df.iterrows()
                    }
                    del_choice = st.selectbox("Select SKU to Delete:", list(del_map.keys()), key="del_sku")
                    if st.button("Delete Selected SKU"):
                        res = requests.delete(f"{API_BASE}/api/v1/inventory/{del_map[del_choice]}", timeout=15)
                        if res.status_code == 200:
                            st.success(res.json()["message"])
                            st.rerun()
                        else:
                            st.error(res.text)

        st.subheader("Full Warehouse Inventory Status")
        if not inv_df.empty:
            status_filter = st.multiselect(
                "Filter by Stock Status:",
                options=["CRITICAL", "LOW", "HEALTHY", "OVERSTOCKED"],
                default=["CRITICAL", "LOW", "HEALTHY", "OVERSTOCKED"],
            )
            filtered_inv = inv_df[inv_df["status"].isin(status_filter)]
            st.dataframe(
                filtered_inv[
                    [
                        "status",
                        "market_name",
                        "state",
                        "commodity_name",
                        "supplier_name",
                        "lead_time_days",
                        "current_stock_qtl",
                        "safety_stock_qtl",
                        "reorder_point_qtl",
                        "days_of_cover",
                        "latest_modal_price",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # === TAB 2: PRICE FORECAST & BEST TIME TO BUY ===
    with tab2:
        st.subheader("Recursive Multi-Day Price Forecast & Price-Aware Buying Strategy")
        if not inv_df.empty:
            markets_map = (
                inv_df[["market_id", "market_name"]]
                .drop_duplicates()
                .sort_values("market_name")
                .set_index("market_name")["market_id"]
                .to_dict()
            )
            f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
            selected_market_name = f_col1.selectbox("Select Mandi (Market):", list(markets_map.keys()))
            selected_market_id = int(markets_map[selected_market_name])

            avail_comms = (
                inv_df[inv_df["market_id"] == selected_market_id][["commodity_id", "commodity_name"]]
                .drop_duplicates()
                .set_index("commodity_name")["commodity_id"]
                .to_dict()
            )
            selected_comm_name = f_col2.selectbox("Select Commodity:", list(avail_comms.keys()))
            selected_comm_id = int(avail_comms[selected_comm_name])
            horizon_days = f_col3.slider("Forecast Horizon (Days):", min_value=7, max_value=30, value=14)

            fc_data = fetch_json(
                f"/api/v1/forecast/{selected_market_id}/{selected_comm_id}",
                params={"horizon": horizon_days},
            )

            if fc_data:
                rec = fc_data["procurement_recommendation"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Modal Price", f"₹{fc_data['current_price']:,.2f} / qtl")
                m2.metric(
                    f"{horizon_days}-Day Avg Forecast",
                    f"₹{fc_data['avg_forecast_price']:,.2f} / qtl",
                    delta=f"{fc_data['expected_price_change_pct']:+.2f}%",
                )
                m3.metric("Recommended Strategy", rec.get("procurement_strategy", "N/A"))
                m4.metric("Recommended Order Qty", f"{rec.get('recommended_order_qtl', 0)} qtl")

                hist_tail = pd.DataFrame(fc_data["historical_tail"])
                fc_df = pd.DataFrame(fc_data["forecast"])

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=hist_tail["date"],
                        y=hist_tail["actual_price"],
                        mode="lines+markers",
                        name="Historical Actual Price",
                        line=dict(color="#1f77b4", width=2.5),
                    )
                )
                bridge_dates = [hist_tail["date"].iloc[-1]] + fc_df["date"].tolist()
                bridge_prices = [hist_tail["actual_price"].iloc[-1]] + fc_df["predicted_price"].tolist()

                fig.add_trace(
                    go.Scatter(
                        x=bridge_dates,
                        y=bridge_prices,
                        mode="lines+markers",
                        name="Random Forest Forecast",
                        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=fc_df["date"].tolist() + fc_df["date"].tolist()[::-1],
                        y=fc_df["upper_bound"].tolist() + fc_df["lower_bound"].tolist()[::-1],
                        fill="toself",
                        fillcolor="rgba(255, 127, 14, 0.15)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        name="Confidence Band (±MAE)",
                    )
                )
                fig.update_layout(
                    title=f"{selected_comm_name} Modal Price Trajectory — {selected_market_name}",
                    xaxis_title="Date",
                    yaxis_title="Price (INR / Quintal)",
                    hovermode="x unified",
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"**Procurement Engine Decision ({rec.get('urgency', '')}):** {rec.get('rationale', '')}")

                # Allow immediate PO placement if proactive/urgent buy is recommended
                if rec.get("recommended_order_qtl", 0) > 0:
                    if st.button(
                        f"⚡ Execute {rec['procurement_strategy']} ({rec['recommended_order_qtl']} qtl @ ₹{fc_data['current_price']:,.0f}/qtl)",
                        type="primary",
                    ):
                        res = requests.post(
                            f"{API_BASE}/api/v1/purchase-orders",
                            json={
                                "market_id": selected_market_id,
                                "commodity_id": selected_comm_id,
                                "quantity_qtl": float(rec["recommended_order_qtl"]),
                                "procurement_strategy": str(rec["procurement_strategy"]),
                            },
                            timeout=15,
                        )
                        if res.status_code == 201:
                            st.success("Forward Purchase Order placed successfully!")
                            st.rerun()

        st.divider()
        st.subheader("💰 Market-Wide Price Timing Scanner (When to Buy Low vs. Wait)")
        opps = fetch_json("/api/v1/buying-opportunities") or []
        if opps:
            st.dataframe(pd.DataFrame(opps), use_container_width=True, hide_index=True)

    # === TAB 3: PURCHASE ORDERS & RECEIVING ===
    with tab3:
        st.subheader("Purchase Order Ledger & Warehouse Receiving")
        if po_df.empty:
            st.info("No purchase orders recorded yet.")
        else:
            pending_df = po_df[po_df["status"] == "PENDING"]
            if not pending_df.empty:
                st.markdown("#### 🚚 Confirm Supplier Delivery (Convert `PENDING` $\\rightarrow$ `DELIVERED` & Add Stock)")
                d_col1, d_col2 = st.columns([3, 1])
                po_map = {
                    f"PO #{r['id']} — {r['commodity_name']} @ {r['market_name']} | {r['quantity_qtl']} qtl from {r['supplier_name']} (Expected: {r['expected_delivery_date']})": int(r["id"])
                    for _, r in pending_df.iterrows()
                }
                chosen_po = d_col1.selectbox("Select Pending Shipment Arrived at Warehouse:", list(po_map.keys()))
                if d_col2.button("✅ Confirm Stock Reached Warehouse", type="primary", use_container_width=True):
                    res = requests.patch(f"{API_BASE}/api/v1/purchase-orders/{po_map[chosen_po]}/deliver", timeout=15)
                    if res.status_code == 200:
                        st.success(res.json()["message"])
                        st.rerun()
                    else:
                        st.error(res.text)

            st.dataframe(po_df, use_container_width=True, hide_index=True)

    # === TAB 4: MODEL BENCHMARKS ===
    with tab4:
        st.subheader("Empirical Model Evaluation on Chronological Holdout Test Set")
        metrics_data = fetch_json("/api/v1/model-benchmarks")
        if metrics_data:
            bench_df = pd.DataFrame(metrics_data).T.reset_index().rename(columns={"index": "Model"})
            c_tbl, c_cht = st.columns([1, 1.3])
            c_tbl.dataframe(bench_df, use_container_width=True, hide_index=True)

            fig_bench = px.bar(
                bench_df,
                x="Model",
                y="MAPE (%)",
                text="MAPE (%)",
                color="Model",
                title="Test Set MAPE (%) — Lower Is Better",
            )
            fig_bench.update_traces(textposition="outside")
            c_cht.plotly_chart(fig_bench, use_container_width=True)

    # === TAB 5: AI INVENTORY ASSISTANT ===
    with tab5:
        st.subheader("🤖 AI Procurement & Inventory Co-Pilot")
        st.caption("Ask questions grounded in live PostgreSQL warehouse data and Random Forest price forecasts.")

        q_cols = st.columns(3)
        starter_q = None
        if q_cols[0].button("🚨 Which items need restocking now?", use_container_width=True):
            starter_q = "Which items are low in stock right now and how much will it cost to restock?"
        if q_cols[1].button("🥔 Should I buy Potato in Alipurduar?", use_container_width=True):
            starter_q = "Should I buy Potato in Alipurduar right now?"
        if q_cols[2].button("📦 Summarize PO spend & model accuracy", use_container_width=True):
            starter_q = "Summarize our pending purchase orders and model accuracy."

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "content": "Hello! I am connected to your live PostgreSQL warehouse inventory and Random Forest price forecaster. Ask me about stock alerts, price forecasts for any mandi, or pending purchase orders.",
                }
            ]

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_prompt = st.chat_input("Ask about stock levels, mandi forecasts, or reorder strategies...") or starter_q
        if user_prompt:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Querying live inventory & forecasting engine..."):
                    resp = requests.post(
                        f"{API_BASE}/api/v1/assistant/chat",
                        json={"question": user_prompt},
                        timeout=25,
                    )
                    ans = resp.json().get("answer", "No response.") if resp.status_code == 200 else f"Error: {resp.text}"
                    st.markdown(ans)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})