import json
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("models/artifacts")
PROCESSED_PATH = Path("data/processed/features.parquet")


class PriceForecaster:
    """Loads the winning Random Forest artifact and generates recursive multi-day price forecasts."""

    def __init__(self):
        self.model = joblib.load(ARTIFACTS_DIR / "rf_price_model.joblib")
        with open(ARTIFACTS_DIR / "feature_columns.json", "r", encoding="utf-8") as f:
            self.feature_cols = json.load(f)
        with open(ARTIFACTS_DIR / "entity_mappings.json", "r", encoding="utf-8") as f:
            self.mappings = json.load(f)
        with open(ARTIFACTS_DIR / "best_model_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.base_mae = float(self.metadata["metrics"]["MAE"])

    def _build_single_day_features(
        self,
        market_id: int,
        commodity_id: int,
        target_date: pd.Timestamp,
        price_history: list[float],
        last_spread: float,
        last_min: float,
        last_max: float,
    ) -> pd.DataFrame:
        """Compute exact feature vector for `target_date` using the latest rolling price buffer."""
        p_lag_1 = price_history[-1]
        p_lag_7 = price_history[-7] if len(price_history) >= 7 else price_history[0]
        p_lag_14 = price_history[-14] if len(price_history) >= 14 else price_history[0]
        p_lag_28 = price_history[-28] if len(price_history) >= 28 else price_history[0]

        win_7 = price_history[-7:]
        win_28 = price_history[-28:]

        r_mean_7 = float(np.mean(win_7))
        r_std_7 = float(np.std(win_7, ddof=1)) if len(win_7) > 1 else 0.0
        r_mean_28 = float(np.mean(win_28))
        r_std_28 = float(np.std(win_28, ddof=1)) if len(win_28) > 1 else 0.0
        momentum = r_mean_7 / (r_mean_28 + 1e-5)

        dow = target_date.dayofweek
        dom = target_date.day
        month = target_date.month
        quarter = target_date.quarter
        is_weekend = int(dow >= 5)

        row = {
            "market_id": market_id,
            "commodity_id": commodity_id,
            "prev_min_price": last_min,
            "prev_max_price": last_max,
            "prev_price_spread": last_spread,
            "price_lag_1": p_lag_1,
            "price_lag_7": p_lag_7,
            "price_lag_14": p_lag_14,
            "price_lag_28": p_lag_28,
            "rolling_mean_7": r_mean_7,
            "rolling_std_7": r_std_7,
            "rolling_mean_28": r_mean_28,
            "rolling_std_28": r_std_28,
            "momentum_7_28": momentum,
            "day_of_week": dow,
            "day_of_month": dom,
            "month": month,
            "quarter": quarter,
            "is_weekend": is_weekend,
            "sin_month": np.sin(2 * np.pi * month / 12.0),
            "cos_month": np.cos(2 * np.pi * month / 12.0),
            "sin_dow": np.sin(2 * np.pi * dow / 7.0),
            "cos_dow": np.cos(2 * np.pi * dow / 7.0),
        }
        return pd.DataFrame([row])[self.feature_cols]

    def forecast(
        self,
        history_df: pd.DataFrame,
        market_name: str,
        commodity_name: str,
        horizon_days: int = 14,
    ) -> dict:
        """
        Generate a recursive multi-day price forecast for a specific market and commodity.
        `history_df` must contain columns: ['market', 'commodity', 'date', 'modal_price', 'min_price', 'max_price'].
        """
        market_key = market_name.strip().title()
        comm_key = commodity_name.strip().title()

        if market_key not in self.mappings["market_to_id"]:
            raise ValueError(f"Unknown market '{market_name}'. Available: {list(self.mappings['market_to_id'].keys())[:5]}...")
        if comm_key not in self.mappings["commodity_to_id"]:
            raise ValueError(f"Unknown commodity '{commodity_name}'. Available: {list(self.mappings['commodity_to_id'].keys())}")

        market_id = self.mappings["market_to_id"][market_key]
        commodity_id = self.mappings["commodity_to_id"][comm_key]

        sub = (
            history_df[(history_df["market"] == market_key) & (history_df["commodity"] == comm_key)]
            .sort_values("date")
            .tail(35)
            .copy()
        )
        if len(sub) < 7:
            raise ValueError(f"Insufficient recent history (<7 days) for {comm_key} in {market_key}.")

        price_buffer = sub["modal_price"].astype(float).tolist()
        last_date = pd.to_datetime(sub["date"].iloc[-1])
        last_min = float(sub["min_price"].iloc[-1])
        last_max = float(sub["max_price"].iloc[-1])
        last_spread = max(0.0, last_max - last_min)
        current_price = price_buffer[-1]

        predictions = []
        for step in range(1, horizon_days + 1):
            next_date = last_date + timedelta(days=step)
            X_step = self._build_single_day_features(
                market_id=market_id,
                commodity_id=commodity_id,
                target_date=next_date,
                price_history=price_buffer,
                last_spread=last_spread,
                last_min=last_min,
                last_max=last_max,
            )
            pred_price = float(self.model.predict(X_step)[0])
            pred_price = round(max(100.0, pred_price), 2)

            # Confidence band widens gradually over the forecast horizon
            margin = round(self.base_mae * (1.0 + 0.04 * (step - 1)), 2)

            predictions.append(
                {
                    "date": next_date.strftime("%Y-%m-%d"),
                    "predicted_price": pred_price,
                    "lower_bound": round(max(50.0, pred_price - margin), 2),
                    "upper_bound": round(pred_price + margin, 2),
                }
            )

            # Feed predicted price back into buffer for the next recursive step
            price_buffer.append(pred_price)
            last_min = round(pred_price - last_spread / 2.0, 2)
            last_max = round(pred_price + last_spread / 2.0, 2)

        avg_predicted = float(np.mean([p["predicted_price"] for p in predictions]))
        pct_change = ((avg_predicted - current_price) / current_price) * 100.0

        return {
            "market": market_key,
            "commodity": comm_key,
            "model_used": self.metadata["best_model"],
            "current_price": round(current_price, 2),
            "avg_forecast_price": round(avg_predicted, 2),
            "expected_price_change_pct": round(pct_change, 2),
            "horizon_days": horizon_days,
            "historical_tail": [
                {"date": pd.to_datetime(r["date"]).strftime("%Y-%m-%d"), "actual_price": round(float(r["modal_price"]), 2)}
                for _, r in sub.tail(14).iterrows()
            ],
            "forecast": predictions,
        }


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_PATH)
    # Pick the first available (market, commodity) pair in the dataset for a live smoke test
    sample_market = df["market"].iloc[0]
    sample_comm = df["commodity"].iloc[0]

    forecaster = PriceForecaster()
    result = forecaster.forecast(df, market_name=sample_market, commodity_name=sample_comm, horizon_days=14)

    print(f"=== 14-DAY PRICE FORECAST: {result['commodity']} @ {result['market']} ===")
    print(f"Model: {result['model_used']} | Current Price: INR {result['current_price']}/qtl")
    print(f"14-Day Avg Forecast: INR {result['avg_forecast_price']}/qtl ({result['expected_price_change_pct']:+.2f}%)\n")
    for row in result["forecast"][:7]:
        print(f"  {row['date']}: INR {row['predicted_price']}  [Range: {row['lower_bound']} - {row['upper_bound']}]")