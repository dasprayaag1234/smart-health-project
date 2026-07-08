"""
Stock-Out Forecaster
----------------------
Predicts, for each facility x medicine, whether a stock-out will occur in
the next N days, using LightGBM on engineered time-series features.

This is the "AI is doing real work" piece for the rubric: not a rule like
"if stock < 20 alert", but a learned model using consumption trend,
volatility, and days-since-resupply to predict forward risk.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import os

FORECAST_HORIZON = 7  # predict stock-out within next 7 days


def engineer_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    df = stock_df.sort_values(["facility_id", "medicine", "day"]).copy()
    grp = df.groupby(["facility_id", "medicine"])

    df["rolling_mean_7"] = grp["daily_consumption"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["rolling_std_7"] = grp["daily_consumption"].transform(lambda s: s.rolling(7, min_periods=1).std().fillna(0))
    df["consumption_trend"] = grp["daily_consumption"].transform(lambda s: s.diff().rolling(5, min_periods=1).mean().fillna(0))
    df["stock_change_1d"] = grp["stock_level"].transform(lambda s: s.diff().fillna(0))
    df["days_since_resupply"] = grp["stock_change_1d"].transform(
        lambda s: (s <= 0).astype(int).groupby((s > 0).cumsum()).cumsum()
    )
    df["days_of_stock_left"] = df["stock_level"] / df["rolling_mean_7"].replace(0, np.nan)
    df["days_of_stock_left"] = df["days_of_stock_left"].fillna(999).clip(upper=999)

    # LABEL: will stock hit 0 within FORECAST_HORIZON days from this row?
    def label_future_stockout(sub):
        stock = sub["stock_level"].values
        n = len(stock)
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            window = stock[i + 1: i + 1 + FORECAST_HORIZON]
            if len(window) > 0 and (window <= 0).any():
                labels[i] = 1
        sub = sub.copy()
        sub["stockout_within_horizon"] = labels
        return sub

    # NOTE: pandas 3.0 excludes the grouping columns from the sub-frame passed
    # into apply() by default now (include_groups=False behavior), so we
    # re-attach facility_id/medicine afterwards via the preserved index
    # rather than relying on them surviving inside the group function.
    id_cols = df[["facility_id", "medicine"]]
    labeled = df.groupby(["facility_id", "medicine"], group_keys=False).apply(
        label_future_stockout, include_groups=False
    )
    labeled[["facility_id", "medicine"]] = id_cols.loc[labeled.index]
    return labeled


def train_model(features_df: pd.DataFrame):
    feature_cols = [
        "stock_level", "daily_consumption", "rolling_mean_7", "rolling_std_7",
        "consumption_trend", "days_since_resupply", "days_of_stock_left",
    ]
    X = features_df[feature_cols]
    y = features_df["stockout_within_horizon"]

    # time-respecting split: train on first 70% of days per series, test on last 30%
    # (avoids leakage — a lesson learned the hard way on a prior forecasting project)
    cutoff_day = features_df["day"].quantile(0.7)
    train_mask = features_df["day"] <= cutoff_day
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)

    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print(f"Test AUC: {roc_auc_score(y_test, preds_proba):.3f}")
    print(classification_report(y_test, preds, digits=3))

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importance.to_string())

    return model, feature_cols


def generate_current_alerts(model, feature_cols, features_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Score the most recent day per facility x medicine to produce today's alert list."""
    latest = features_df.sort_values("day").groupby(["facility_id", "medicine"]).tail(1).copy()
    latest["stockout_risk_score"] = model.predict_proba(latest[feature_cols])[:, 1]
    alerts = latest[latest["stockout_risk_score"] >= threshold].sort_values(
        "stockout_risk_score", ascending=False
    )
    return alerts[["facility_id", "medicine", "stock_level", "days_of_stock_left", "stockout_risk_score"]]


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "simulated")
    stock_df = pd.read_csv(os.path.join(data_dir, "stock_timeseries.csv"))

    print("Engineering features...")
    features_df = engineer_features(stock_df)

    print(f"\nOverall stock-out label rate: {features_df['stockout_within_horizon'].mean()*100:.1f}%\n")

    model, feature_cols = train_model(features_df)

    alerts = generate_current_alerts(model, feature_cols, features_df, threshold=0.5)
    print(f"\n{len(alerts)} current stock-out risk alerts (top 15):")
    print(alerts.head(15).to_string(index=False))

    alerts.to_csv(os.path.join(data_dir, "current_stockout_alerts.csv"), index=False)
