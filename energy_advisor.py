# energy_advisor.py
# AI-Based School Energy Consumption Advisor
# Streamlit + Regression ML + Optional OpenAI Assistant

import os
import json
import math
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

# Optional OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="⚡ School Energy AI Advisor",
    layout="wide"
)

st.title("⚡ AI-Based School Energy Consumption Advisor")
st.caption("Predictive analytics + Peak detection + AI Recommendations")


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("⚙ Settings")

data_file = st.sidebar.file_uploader("Upload energy CSV", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo dataset", value=(data_file is None))

interval_minutes = st.sidebar.selectbox(
    "Sensor Interval (minutes)", [5, 10, 15, 30, 60], index=2
)

tariff = st.sidebar.number_input("Tariff (KES per kWh)", value=30.0)
demand_charge = st.sidebar.number_input("Demand charge (KES per kW)", value=0.0)

enable_openai = st.sidebar.checkbox("Enable OpenAI Assistant", value=False)
model_name = st.sidebar.text_input("OpenAI Model", value="gpt-5.2")

run_btn = st.sidebar.button("▶ Run Analysis")


# -------------------------------------------------
# DEMO DATA
# -------------------------------------------------
def generate_demo_data(days=28, interval=15):
    periods = int((24 * 60) / interval) * days
    timestamps = pd.date_range(
        end=pd.Timestamp.today(),
        periods=periods,
        freq=f"{interval}min"
    )

    hour = timestamps.hour + timestamps.minute / 60

    usage = (
        6
        + 8 * np.exp(-0.5 * ((hour - 9) / 2) ** 2)
        + 6 * np.exp(-0.5 * ((hour - 14) / 2) ** 2)
        + np.random.normal(0, 0.4, len(timestamps))
    )

    return pd.DataFrame({
        "timestamp": timestamps,
        "kwh": np.clip(usage, 0.5, None)
    })


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
if use_demo:
    df = generate_demo_data(interval=interval_minutes)
else:
    df = pd.read_csv(data_file)
    df.columns = df.columns.str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["kwh"] = pd.to_numeric(df["kwh"])

df = df.sort_values("timestamp")
df = df.dropna()


# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

df["lag1"] = df["kwh"].shift(1)
df["rolling4"] = df["kwh"].rolling(4).mean()

df = df.dropna()

X = df[["hour", "dayofweek", "is_weekend", "lag1", "rolling4"]]
y = df["kwh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, pred)
accuracy = (1 - mape) * 100
r2 = r2_score(y_test, pred)


# -------------------------------------------------
# KPI CALCULATIONS
# -------------------------------------------------
total_kwh = df["kwh"].sum()
avg_daily = df.groupby(df["timestamp"].dt.date)["kwh"].sum().mean()

peak_interval = df["kwh"].max()
peak_kw = peak_interval / (interval_minutes / 60)

total_cost = total_kwh * tariff
monthly_est = avg_daily * 30 * tariff + peak_kw * demand_charge


# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Energy (kWh)", f"{total_kwh:.0f}")
col2.metric("Avg Daily (kWh)", f"{avg_daily:.1f}")
col3.metric("Peak Demand (kW)", f"{peak_kw:.1f}")
col4.metric("Model Accuracy", f"{accuracy:.1f}%")

st.divider()

st.subheader("Energy Usage Over Time")
st.line_chart(df.set_index("timestamp")["kwh"])

st.subheader("Model Performance")
st.write(f"R² Score: {r2:.3f}")
st.write(f"MAPE: {mape*100:.2f}%")

st.divider()

st.subheader("Cost Analysis")
st.metric("Observed Cost (KES)", f"{total_cost:,.0f}")
st.metric("Estimated Monthly Cost (KES)", f"{monthly_est:,.0f}")


# -------------------------------------------------
# BASIC RECOMMENDATIONS (Rule-Based)
# -------------------------------------------------
st.subheader("🔎 Automated Recommendations")

peak_hours = df.groupby("hour")["kwh"].mean().sort_values(ascending=False).head(3).index

st.write(f"Peak hours detected: {', '.join([str(h)+':00' for h in peak_hours])}")

st.markdown("- Stagger heavy equipment during peak hours.")
st.markdown("- Optimize lighting in high-usage periods.")
st.markdown("- Reduce after-hours standby loads.")


# -------------------------------------------------
# OPENAI ASSISTANT
# -------------------------------------------------
if enable_openai:

    st.divider()
    st.header("🤖 OpenAI Energy Advisor")

    api_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("OPENAI_API_KEY not found in Secrets.")
    elif OpenAI is None:
        st.error("openai package not installed.")
    else:
        client = OpenAI(api_key=api_key)

        summary = {
            "total_kwh": total_kwh,
            "avg_daily_kwh": avg_daily,
            "peak_kw": peak_kw,
            "accuracy_pct": accuracy,
            "r2": r2,
            "peak_hours": list(map(int, peak_hours))
        }

        if st.button("Generate AI Report"):

            with st.spinner("Generating report..."):
                response = client.responses.create(
                    model=model_name,
                    instructions="You are an energy efficiency advisor for a secondary school.",
                    input=json.dumps(summary)
                )

            report_text = response.output_text
            st.text_area("AI Generated Report", report_text, height=300)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()

with st.expander("Deployment Instructions"):
    st.markdown("""
### To Deploy on Streamlit Cloud:

1. Add to `requirements.txt`:
