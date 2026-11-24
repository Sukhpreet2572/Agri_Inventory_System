import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.ml.evaluate import evaluate_predictions

PROCESSED_PATH = Path("data/processed/features.parquet")
ARTIFACTS_DIR = Path("models/artifacts")
SPLIT_DATE = "2025-06-01"

FEATURE_COLS = [
    "market_id",
    "commodity_id",
    "prev_min_price",
    "prev_max_price",
    "prev_price_spread",
    "price_lag_1",
    "price_lag_7",
    "price_lag_14",
    "price_lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
    "momentum_7_28",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "sin_month",
    "cos_month",
    "sin_dow",
    "cos_dow",
]
TARGET_COL = "modal_price"


def build_lstm_sequences(
    df: pd.DataFrame, feature_cols: list[str], x_scaler: StandardScaler, seq_len: int = 14
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Create sliding sequence tensors (samples, seq_len, n_features) grouped by market & commodity."""
    X_list, y_list, date_list = [], [], []
    scaled_features = x_scaler.transform(df[feature_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)
    df_scaled["market_id"] = df["market_id"].values
    df_scaled["commodity_id"] = df["commodity_id"].values
    df_scaled[TARGET_COL] = df[TARGET_COL].values
    df_scaled["date"] = df["date"].values

    for _, grp in df_scaled.groupby(["market_id", "commodity_id"]):
        if len(grp) <= seq_len:
            continue
        vals = grp[feature_cols].to_numpy(dtype=np.float32)
        targets = grp[TARGET_COL].to_numpy(dtype=np.float32)
        dates = grp["date"].to_numpy()
        for i in range(seq_len, len(grp)):
            X_list.append(vals[i - seq_len : i])
            y_list.append(targets[i])
            date_list.append(dates[i])

    return np.array(X_list), np.array(y_list), pd.Series(date_list)


def train_and_benchmark():
    print(f"[1/5] Loading engineered dataset from {PROCESSED_PATH}...")
    df = pd.read_parquet(PROCESSED_PATH).sort_values(["market_id", "commodity_id", "date"])

    train_df = df[df["date"] < SPLIT_DATE].copy()
    test_df = df[df["date"] >= SPLIT_DATE].copy()
    print(f"      Chronological Split @ {SPLIT_DATE}: Train={len(train_df):,} rows | Test={len(test_df):,} rows")

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL].to_numpy()
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL].to_numpy()

    results = {}

    # 1. Statistical Baseline: 7-Day Moving Average
    print("[2/5] Evaluating Baseline (7-Day Moving Average)...")
    baseline_preds = test_df["rolling_mean_7"].to_numpy()
    results["1_Moving_Average_7D"] = evaluate_predictions(y_test, baseline_preds)

    # 2. Random Forest Regressor
    print("[3/5] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=150, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    results["2_Random_Forest"] = evaluate_predictions(y_test, rf_preds)

    # 3. XGBoost Regressor
    print("[4/5] Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_preds = xgb_model.predict(X_test)
    results["3_XGBoost"] = evaluate_predictions(y_test, xgb_preds)

    # 4. Global Deep Learning LSTM (with both X and y scaled!)
    print("[5/5] Training Global LSTM Neural Network...")
    tf.random.set_seed(42)
    x_scaler = StandardScaler().fit(train_df[FEATURE_COLS])
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_seq_all, y_seq_all, seq_dates = build_lstm_sequences(df, FEATURE_COLS, x_scaler, seq_len=14)
    train_mask = seq_dates < pd.Timestamp(SPLIT_DATE)
    test_mask = seq_dates >= pd.Timestamp(SPLIT_DATE)

    X_seq_train = X_seq_all[train_mask]
    y_seq_train_scaled = y_scaler.transform(y_seq_all[train_mask].reshape(-1, 1)).flatten()

    X_seq_test = X_seq_all[test_mask]
    y_seq_test_raw = y_seq_all[test_mask]
    y_seq_test_scaled = y_scaler.transform(y_seq_test_raw.reshape(-1, 1)).flatten()

    lstm_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(14, len(FEATURE_COLS))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    lstm_model.fit(
        X_seq_train,
        y_seq_train_scaled,
        validation_data=(X_seq_test, y_seq_test_scaled),
        epochs=30,
        batch_size=128,
        callbacks=[early_stop],
        verbose=0,
    )
    lstm_preds_scaled = lstm_model.predict(X_seq_test, verbose=0).reshape(-1, 1)
    lstm_preds = y_scaler.inverse_transform(lstm_preds_scaled).flatten()
    results["4_Global_LSTM"] = evaluate_predictions(y_seq_test_raw, lstm_preds)

    # Print Benchmark Summary Table
    print("\n================ MODEL BENCHMARK RESULTS (TEST SET) ================")
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    print("====================================================================\n")

    # Identify best model by lowest RMSE
    best_model_name = summary_df["RMSE"].idxmin()
    print(f"[*] Best Performing Model: {best_model_name} (RMSE: {summary_df.loc[best_model_name, 'RMSE']})")

    # Save Artifacts cleanly
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(ARTIFACTS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    with open(ARTIFACTS_DIR / "best_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "metrics": results[best_model_name],
                "split_date": SPLIT_DATE,
            },
            f,
            indent=2,
        )

    # Use get_booster().save_model() to bypass scikit-learn 1.6+ wrapper tag bug
    xgb_model.get_booster().save_model(ARTIFACTS_DIR / "xgboost_price_model.json")
    joblib.dump(rf_model, ARTIFACTS_DIR / "rf_price_model.joblib")
    lstm_model.save(ARTIFACTS_DIR / "lstm_price_model.keras")
    joblib.dump({"x_scaler": x_scaler, "y_scaler": y_scaler}, ARTIFACTS_DIR / "lstm_scalers.joblib")

    print(f"[+] All model artifacts and benchmark metrics saved cleanly to '{ARTIFACTS_DIR}/'")


if __name__ == "__main__":
    train_and_benchmark()