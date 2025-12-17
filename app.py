import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Chiang Rai PM2.5 Early Warning Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# =========================
# Helpers
# =========================
def load_artifacts():
    model = joblib.load("xgb_model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    metrics = joblib.load("metrics.pkl")
    df = pd.read_csv("daily_pm25_dataset.csv")

    # If your CSV saved index=True, you might have an unnamed index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Try to parse datetime if present
    for c in ["date", "datetime", "time", "timestamp"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    return model, feature_cols, metrics, df


def detect_pm25_column(df: pd.DataFrame) -> str:
    """
    Best-effort detection of the true PM2.5 column (not lag/roll).
    """
    candidates = []
    for col in df.columns:
        low = col.lower()
        if "pm25" in low and not any(x in low for x in ["lag", "roll", "rolling"]):
            candidates.append(col)

    # Prefer exact common names
    for preferred in ["pm25", "pm2_5", "pm25_value", "pm25_mean", "pm25_avg", "value"]:
        for c in candidates:
            if c.lower() == preferred:
                return c

    if candidates:
        return candidates[0]

    raise ValueError("Could not auto-detect the PM2.5 column in daily_pm25_dataset.csv")


def pm25_to_aqi_category(pm: float) -> str:
    """
    Simple category bands (rough guidance). Adjust to the standard you follow.
    (Using widely used PM2.5 cut points similar to US EPA ranges.)
    """
    if pm <= 12.0:
        return "Good"
    elif pm <= 35.4:
        return "Moderate"
    elif pm <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm <= 150.4:
        return "Unhealthy"
    elif pm <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def action_guidance(category: str) -> str:
    tips = {
        "Good": "✅ Normal outdoor activities are fine.",
        "Moderate": "🙂 Most people are OK. Sensitive individuals should monitor symptoms.",
        "Unhealthy for Sensitive Groups": "⚠️ Sensitive groups (asthma, elderly, children) should reduce prolonged outdoor exertion; consider masks.",
        "Unhealthy": "🚫 Everyone should limit outdoor exertion; sensitive groups should stay indoors; use air purifier if possible.",
        "Very Unhealthy": "🛑 Avoid outdoor activity; keep windows closed; run air purifier; wear a well-fitted mask if going outside.",
        "Hazardous": "🚨 Health alert: stay indoors; avoid outdoor exposure; consider seeking medical advice if symptoms appear."
    }
    return tips.get(category, "")


def data_quality_score(df: pd.DataFrame, required_cols: list, lookback_days: int = 14) -> tuple[str, float]:
    """
    Simple data-quality indicator based on missingness in recent rows.
    """
    recent = df.tail(lookback_days).copy()
    if recent.empty:
        return ("Unknown", np.nan)

    # Missing % over required columns
    miss = recent[required_cols].isna().mean().mean() * 100.0
    # Convert to a simple label
    if miss < 1.0:
        label = "High"
    elif miss < 5.0:
        label = "Medium"
    else:
        label = "Low"
    return (label, miss)


def forecast_next_days(model, feature_cols, df, pm25_col: str, horizon: int = 7):
    """
    Iterative forecast:
    - Uses last known weather values as constants (simple assumption)
    - Updates pm25_lag1/2/3 and rolling means (roll3, roll7) using predicted values.
    """
    last = df.iloc[-1].copy()

    # history for rolling computations (use last 7 actual values if available)
    hist = df[pm25_col].dropna().astype(float).tolist()
    if len(hist) < 7:
        # if not enough history, pad with last known value
        if len(hist) == 0:
            hist = [float(last.get(pm25_col, 0.0))] * 7
        else:
            hist = ([hist[-1]] * (7 - len(hist))) + hist

    preds = []
    dates = []

    # Determine "date" axis: if dataset has a datetime column, use it; otherwise use today.
    date_col = None
    for c in ["date", "datetime", "timestamp", "time"]:
        if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]):
            date_col = c
            break

    if date_col and pd.notna(last[date_col]):
        base_date = pd.to_datetime(last[date_col], utc=True).normalize()
    else:
        base_date = pd.Timestamp(datetime.now(timezone.utc)).normalize()

    # Identify feature names we know how to update
    has_lag1 = "pm25_lag1" in feature_cols
    has_lag2 = "pm25_lag2" in feature_cols
    has_lag3 = "pm25_lag3" in feature_cols
    has_roll3 = "pm25_roll3" in feature_cols
    has_roll7 = "pm25_roll7" in feature_cols

    # Prepare constant features from last row
    const = {}
    for c in feature_cols:
        if c in ["pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_roll3", "pm25_roll7"]:
            continue
        # take last known value if exists, else 0
        const[c] = float(last[c]) if c in df.columns and pd.notna(last[c]) else 0.0

    for i in range(1, horizon + 1):
        # compute dynamic features from history
        feats = dict(const)

        if has_lag1:
            feats["pm25_lag1"] = float(hist[-1])
        if has_lag2:
            feats["pm25_lag2"] = float(hist[-2])
        if has_lag3:
            feats["pm25_lag3"] = float(hist[-3])
        if has_roll3:
            feats["pm25_roll3"] = float(np.mean(hist[-3:]))
        if has_roll7:
            feats["pm25_roll7"] = float(np.mean(hist[-7:]))

        X = np.array([feats[c] for c in feature_cols], dtype=float).reshape(1, -1)
        yhat = float(model.predict(X)[0])

        preds.append(yhat)
        hist.append(yhat)
        dates.append(base_date + pd.Timedelta(days=i))

    return pd.DataFrame({"date": dates, "predicted_pm25": preds})


# =========================
# Load everything
# =========================
st.title("🌫️ Chiang Rai PM2.5 Early Warning Dashboard")

with st.spinner("Loading model + data artifacts..."):
    model, feature_cols, metrics, df = load_artifacts()
    pm25_col = detect_pm25_column(df)

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Controls")
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=3, max_value=14, value=7, step=1)

# =========================
# Compute core outputs
# =========================
latest_pm = float(df[pm25_col].dropna().iloc[-1]) if df[pm25_col].dropna().shape[0] > 0 else float("nan")
latest_cat = pm25_to_aqi_category(latest_pm) if np.isfinite(latest_pm) else "Unknown"

forecast_df = forecast_next_days(model, feature_cols, df, pm25_col, horizon=horizon)

# Uncertainty band (simple): use RMSE as ± band
rmse = float(metrics.get("rmse", np.nan))
forecast_df["lower"] = forecast_df["predicted_pm25"] - rmse
forecast_df["upper"] = forecast_df["predicted_pm25"] + rmse

# Data quality indicator (based on required input features)
dq_label, dq_missing_pct = data_quality_score(df, required_cols=[c for c in feature_cols if c in df.columns], lookback_days=14)

# =========================
# Layout
# =========================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Today’s PM2.5", f"{latest_pm:.2f} µg/m³" if np.isfinite(latest_pm) else "N/A")

with c2:
    st.metric("Category", latest_cat)

with c3:
    st.metric("Model (R²)", f"{float(metrics.get('r2', np.nan)):.3f}" if metrics.get("r2") is not None else "N/A")

with c4:
    if np.isfinite(dq_missing_pct):
        st.metric("Data quality (14d)", f"{dq_label} ({dq_missing_pct:.1f}% missing)")
    else:
        st.metric("Data quality (14d)", "Unknown")

st.info(action_guidance(latest_cat))

# =========================
# Trend + forecast
# =========================
left, right = st.columns([2, 1])

with left:
    st.subheader("Short-term trend (recent + forecast)")
    # recent window for context
    recent_window = df.tail(30).copy()
    # Build a timeline index if possible
    date_col = None
    for c in ["date", "datetime", "timestamp", "time"]:
        if c in recent_window.columns and pd.api.types.is_datetime64_any_dtype(recent_window[c]):
            date_col = c
            break

    if date_col:
        recent_plot = recent_window[[date_col, pm25_col]].dropna().rename(columns={date_col: "date", pm25_col: "actual_pm25"})
        recent_plot = recent_plot.sort_values("date")
    else:
        recent_plot = pd.DataFrame({"date": range(len(recent_window)), "actual_pm25": recent_window[pm25_col].values})

    st.line_chart(
        data=pd.concat([
            recent_plot[["date", "actual_pm25"]].set_index("date"),
            forecast_df[["date", "predicted_pm25"]].set_index("date")
        ], axis=1)
    )

    st.caption("Forecast uncertainty shown as ±RMSE band (simple proxy).")

    # Show uncertainty as a table (clean + quick)
    show_tbl = forecast_df.copy()
    show_tbl["category"] = show_tbl["predicted_pm25"].apply(pm25_to_aqi_category)
    st.dataframe(
        show_tbl.rename(columns={"predicted_pm25": "pm25_forecast"}).round(2),
        use_container_width=True
    )

with right:
    st.subheader("Decision-support panel")

    # Flag “hazardous” style events (operational metric preview)
    hazard_threshold = st.number_input("Hazard threshold (PM2.5)", min_value=0.0, value=55.4, step=1.0)
    hazard_days = (forecast_df["predicted_pm25"] >= hazard_threshold).sum()
    st.write(f"**Days ≥ {hazard_threshold:.1f}:** {int(hazard_days)} out of {len(forecast_df)}")

    # Lead-time proxy: first day crossing threshold
    cross_idx = np.where(forecast_df["predicted_pm25"].values >= hazard_threshold)[0]
    if len(cross_idx) > 0:
        first_cross = forecast_df.iloc[int(cross_idx[0])]
        st.warning(f"⚠️ First threshold crossing: **{first_cross['date'].date()}** (~{first_cross['predicted_pm25']:.1f})")
    else:
        st.success("✅ No threshold crossing in forecast window.")

    # Show saved evaluation metrics
    st.markdown("### Offline performance")
    st.write({
        "RMSE": float(metrics.get("rmse", np.nan)),
        "MAE": float(metrics.get("mae", np.nan)),
        "R²": float(metrics.get("r2", np.nan)),
    })

# =========================
# Images (from your exported .png files)
# =========================
st.subheader("Model explainability & evaluation plots")

img_cols = st.columns(3)
images = [
    ("Feature Importance (XGBoost)", "feature_importance_xgb.png"),
    ("Actual vs Predicted (XGBoost)", "xgb_pred_vs_actual.png"),
    ("7-Day Forecast Plot", "pm25_forecast_7days.png"),
]

for i, (title, fname) in enumerate(images):
    with img_cols[i % 3]:
        st.caption(title)
        if os.path.exists(fname):
            st.image(fname, use_container_width=True)
        else:
            st.warning(f"Missing file: {fname}")

# Footer
st.caption("Note: Forecast uses last-known weather features as constants (simple deployment assumption).")
