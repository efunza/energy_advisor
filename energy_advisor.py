# energy_advisor_app_openai.py
# Streamlit App: AI-Based School Energy Consumption Advisor
# + Optional OpenAI-powered "Energy Advisor" (chat + report writer) using the Responses API.
#
# Run: streamlit run energy_advisor_app_openai.py
#
# Requirements (requirements.txt):
# streamlit
# pandas
# numpy
# scikit-learn
# openai

import os
import json
import hashlib
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# OpenAI (optional)
# Uses official Python SDK and Responses API per OpenAI docs. :contentReference[oaicite:0]{index=0}
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Page config + Styling
# -----------------------------
st.set_page_config(page_title="School Energy Advisor (AI)", page_icon="⚡", layout="wide")

CUSTOM_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 12% 18%, rgba(59,130,246,0.10), transparent 40%),
      radial-gradient(circle at 82% 22%, rgba(16,185,129,0.11), transparent 45%),
      radial-gradient(circle at 50% 90%, rgba(234,179,8,0.08), transparent 42%);
}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
div[data-testid="stMetric"]{
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 16px;
    padding: 14px 14px 10px 14px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.05);
}
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(2,6,23,0.96));
}
section[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
section[data-testid="stSidebar"] a{ color:#93c5fd !important; }
section[data-testid="stSidebar"] .stButton button{ width:100%; border-radius: 14px; }
.stButton button{
    border-radius: 16px;
    padding: 0.65rem 1rem;
    font-weight: 800;
}
details{
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 16px;
    padding: 0.25rem 0.8rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
}
.badge{
    display:inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    border: 1px solid rgba(0,0,0,0.08);
    background: rgba(255,255,255,0.75);
}
.smallnote{
    opacity: 0.8;
    font-size: 0.92rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLS_MIN = {"timestamp", "kwh"}

def fingerprint_df(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out

def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    cols = set(df.columns)
    missing = REQUIRED_COLS_MIN - cols
    if missing:
        return False, f"Missing required columns: {sorted(missing)}. Required: timestamp, kwh"
    return True, ""

def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=False)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out

def make_demo_data(days: int = 28, freq_minutes: int = 15, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods_per_day = int((24 * 60) / freq_minutes)
    n = days * periods_per_day

    start = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    ts = pd.date_range(start=start, periods=n, freq=f"{freq_minutes}min")

    hour = ts.hour + ts.minute / 60
    dow = ts.dayofweek
    is_weekend = (dow >= 5).astype(int)

    base = 6.5 + rng.normal(0, 0.4, n)
    morning_peak = np.exp(-0.5 * ((hour - 9.5) / 1.6) ** 2) * 7.5
    afternoon_peak = np.exp(-0.5 * ((hour - 14.0) / 1.9) ** 2) * 6.0
    evening_security = np.exp(-0.5 * ((hour - 20.5) / 2.4) ** 2) * 2.0

    weekend_factor = np.where(is_weekend == 1, 0.55, 1.0)

    temp = 28 + 2.0 * np.sin((ts.dayofyear.to_numpy() / 365.0) * 2 * np.pi) + rng.normal(0, 0.6, n)
    occ = np.where(
        (hour >= 7.0) & (hour <= 17.0) & (is_weekend == 0),
        0.65 + 0.25 * rng.random(n),
        0.10 + 0.10 * rng.random(n),
    )

    kwh = (base + morning_peak + afternoon_peak + evening_security) * weekend_factor
    kwh += 0.12 * (temp - 28)
    kwh += 1.8 * occ
    kwh += rng.normal(0, 0.35, n)
    kwh = np.clip(kwh, 0.5, None)

    return pd.DataFrame({"timestamp": ts, "kwh": kwh, "temperature_c": temp, "occupancy": occ})

def feature_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["minute"] = d["timestamp"].dt.minute
    d["dow"] = d["timestamp"].dt.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["day"] = d["timestamp"].dt.date.astype(str)

    hour_float = d["hour"] + d["minute"] / 60.0
    d["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)

    d["kwh_lag1"] = d["kwh"].shift(1)
    d["kwh_lag2"] = d["kwh"].shift(2)
    d["kwh_roll4"] = d["kwh"].rolling(4).mean()
    d["kwh_roll16"] = d["kwh"].rolling(16).mean()

    if "temperature_c" in d.columns:
        d["temperature_c"] = pd.to_numeric(d["temperature_c"], errors="coerce")
    if "occupancy" in d.columns:
        d["occupancy"] = pd.to_numeric(d["occupancy"], errors="coerce")

    d = d.dropna().reset_index(drop=True)

    y = d["kwh"].astype(float)
    feature_cols = [
        "hour", "minute", "dow", "is_weekend",
        "hour_sin", "hour_cos",
        "kwh_lag1", "kwh_lag2", "kwh_roll4", "kwh_roll16",
    ]
    if "temperature_c" in d.columns:
        feature_cols.append("temperature_c")
    if "occupancy" in d.columns:
        feature_cols.append("occupancy")

    X = d[feature_cols].astype(float)
    return X, y, d

def time_split(df_feat: pd.DataFrame, test_days: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    ts = df_feat["timestamp"]
    cutoff = ts.max().normalize() - pd.Timedelta(days=test_days)
    train_idx = ts < cutoff
    test_idx = ~train_idx
    if train_idx.sum() < 200 or test_idx.sum() < 50:
        idx = np.arange(len(df_feat))
        tr, te = train_test_split(idx, test_size=0.25, random_state=42, shuffle=True)
        mask_tr = np.zeros(len(df_feat), dtype=bool)
        mask_te = np.zeros(len(df_feat), dtype=bool)
        mask_tr[tr] = True
        mask_te[te] = True
        return mask_tr, mask_te
    return train_idx.to_numpy(), test_idx.to_numpy()

def compute_kpis(df_raw: pd.DataFrame, interval_minutes: int) -> Dict[str, float]:
    d = df_raw.copy()
    d["date"] = d["timestamp"].dt.date
    daily = d.groupby("date")["kwh"].sum()
    total = float(d["kwh"].sum())
    avg_daily = float(daily.mean()) if len(daily) else 0.0
    peak_interval = float(d["kwh"].max()) if len(d) else 0.0
    peak_kw = peak_interval / (interval_minutes / 60.0) if interval_minutes > 0 else 0.0
    return {
        "total_kwh": total,
        "avg_daily_kwh": avg_daily,
        "peak_interval_kwh": peak_interval,
        "peak_kw_est": float(peak_kw),
        "days": float(d["date"].nunique()),
    }

def make_recommendations(df_raw: pd.DataFrame, interval_minutes: int) -> Dict[str, object]:
    d = df_raw.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["dow"] = d["timestamp"].dt.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype(int)

    by_hour = d.groupby("hour")["kwh"].mean().sort_values(ascending=False)
    top_hours = list(by_hour.head(3).index.astype(int))

    after_mask = (d["hour"] >= 19) | (d["hour"] <= 5)
    after_kwh = float(d.loc[after_mask, "kwh"].sum())
    total_kwh = float(d["kwh"].sum()) if len(d) else 0.0
    after_pct = (after_kwh / total_kwh * 100.0) if total_kwh > 0 else 0.0

    weekend_kwh = float(d.loc[d["is_weekend"] == 1, "kwh"].sum())
    weekend_pct = (weekend_kwh / total_kwh * 100.0) if total_kwh > 0 else 0.0

    peak_interval = float(d["kwh"].max()) if len(d) else 0.0
    peak_kw = peak_interval / (interval_minutes / 60.0)

    recs = [
        {
            "title": "Stagger high-load equipment during peak hours",
            "why": f"Highest average consumption occurs around: {', '.join([f'{h:02d}:00' for h in top_hours])}.",
            "action": "Shift or stagger energy-heavy equipment so they don’t start at the same time.",
            "impact_hint": "Reduces peak demand and improves grid stability."
        },
        {
            "title": "Optimize lighting controls",
            "why": "Lighting is a major controllable load in classrooms, dorms, corridors, and offices.",
            "action": "Use zoning, maximize daylight, and add timers/sensors where possible.",
            "impact_hint": "Quick savings with low disruption."
        },
        {
            "title": "Reduce after-hours energy leakage",
            "why": f"After-hours (19:00–05:00) usage is {after_pct:.1f}% of total energy.",
            "action": "Shutdown checklist (lights, labs, ICT, standby power strips) + timers for outdoor lights.",
            "impact_hint": "Cuts waste without affecting learning time."
        },
        {
            "title": "Review weekend consumption",
            "why": f"Weekend usage contributes {weekend_pct:.1f}% of total energy.",
            "action": "Ensure nonessential rooms stay off; schedule maintenance loads midday.",
            "impact_hint": "Reduces idle-time consumption."
        },
        {
            "title": "Peak demand management",
            "why": f"Estimated peak demand is ~{peak_kw:.1f} kW (from max interval).",
            "action": "Avoid simultaneous switching of large loads; set policies for high-load rooms.",
            "impact_hint": "Can reduce peak demand significantly."
        },
    ]

    return {
        "top_hours": top_hours,
        "after_hours_pct": after_pct,
        "weekend_pct": weekend_pct,
        "peak_kw_est": float(peak_kw),
        "recommendations": recs,
        "by_hour": by_hour.reset_index().rename(columns={"index": "hour", "kwh": "avg_kwh"}),
    }


# -----------------------------
# OpenAI helpers (Responses API)
# -----------------------------
def get_openai_client() -> Tuple[object, str]:
    """
    Priority:
    1) st.secrets["OPENAI_API_KEY"]
    2) env var OPENAI_API_KEY
    """
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return None, "No API key found. Add OPENAI_API_KEY to Streamlit secrets or environment."
    if OpenAI is None:
        return None, "OpenAI Python package not installed. Add `openai` to requirements.txt."

    client = OpenAI(api_key=api_key)
    return client, ""

def openai_generate_text(client, model: str, instructions: str, user_input: str) -> str:
    """
    Uses the OpenAI Responses API. :contentReference[oaicite:1]{index=1}
    """
    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
    )
    # The SDK exposes `output_text` in docs/examples. :contentReference[oaicite:2]{index=2}
    return getattr(resp, "output_text", "") or ""


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## ⚡ Energy Advisor")
st.sidebar.markdown('<span class="badge">Regression ML • Peak Detection • Recommendations • OpenAI Add-on</span>', unsafe_allow_html=True)
st.sidebar.divider()

with st.sidebar.expander("📦 Data input", expanded=True):
    data_file = st.file_uploader("Upload energy CSV (required: timestamp, kwh)", type=["csv"])
    use_demo = st.checkbox("Use demo dataset (4 weeks)", value=(data_file is None))

with st.sidebar.expander("⏱ Data settings", expanded=True):
    interval_minutes = st.selectbox("Sensor interval (minutes)", [5, 10, 15, 30, 60], index=2)

with st.sidebar.expander("🧠 Model settings", expanded=False):
    test_days = st.slider("Test window (days)", 3, 14, 7, 1)
    n_estimators = st.slider("Random Forest trees", 100, 800, 350, 50)
    max_depth = st.slider("Max depth", 4, 30, 14, 1)
    min_samples_leaf = st.slider("Min samples/leaf", 1, 10, 2, 1)
    random_state = st.number_input("Random seed", value=42, step=1)

with st.sidebar.expander("💸 Cost & savings", expanded=True):
    tariff_kes_per_kwh = st.number_input("Tariff (KES per kWh)", value=30.0, min_value=0.0, step=1.0)
    demand_charge_kes_per_kw = st.number_input("Demand charge (KES per kW, optional)", value=0.0, min_value=0.0, step=10.0)
    st.caption("If you don’t have demand charges, set it to 0.")

with st.sidebar.expander("🤖 OpenAI settings (optional)", expanded=False):
    st.caption("Add OPENAI_API_KEY in Streamlit secrets for deployment.")
    openai_model = st.text_input("Model", value="gpt-5.2")  # latest family guide references GPT-5.2. :contentReference[oaicite:3]{index=3}
    enable_openai = st.checkbox("Enable OpenAI features", value=False)

run_btn = st.sidebar.button("▶ Analyze & Train Model", type="primary")


# -----------------------------
# Load data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data(use_demo: bool, uploaded_file, interval_minutes: int) -> pd.DataFrame:
    if use_demo:
        return make_demo_data(days=28, freq_minutes=interval_minutes, seed=7)

    df = pd.read_csv(uploaded_file)
    df = normalize_columns(df)
    ok, msg = validate_schema(df)
    if not ok:
        raise ValueError(msg)
    df = parse_timestamp(df)
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["kwh"]).reset_index(drop=True)
    return df

try:
    df_raw = load_data(use_demo, data_file, int(interval_minutes))
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

fp = fingerprint_df(df_raw)
if st.session_state.get("data_fp") != fp:
    for k in ["metrics", "pred_df", "daily_actual", "forecast_df", "kpis", "recs", "chat_messages", "ai_report"]:
        st.session_state.pop(k, None)
    st.session_state["data_fp"] = fp


# -----------------------------
# Train / analyze
# -----------------------------
def train_and_predict(df_raw: pd.DataFrame) -> Dict[str, object]:
    X, y, df_feat = feature_engineer(df_raw)
    train_mask, test_mask = time_split(df_feat, test_days=int(test_days))

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    yhat_test = model.predict(X_test)

    mape = float(mean_absolute_percentage_error(y_test, yhat_test)) if len(y_test) else float("nan")
    accuracy_pct = float((1.0 - mape) * 100.0) if not math.isnan(mape) else float("nan")
    r2 = float(r2_score(y_test, yhat_test)) if len(y_test) else float("nan")
    mae = float(mean_absolute_error(y_test, yhat_test)) if len(y_test) else float("nan")

    pred_df = df_feat.loc[test_mask, ["timestamp"]].copy()
    pred_df["actual_kwh"] = y_test.to_numpy()
    pred_df["pred_kwh"] = yhat_test
    pred_df["error_kwh"] = pred_df["pred_kwh"] - pred_df["actual_kwh"]

    daily = df_raw.copy()
    daily["date"] = daily["timestamp"].dt.date
    daily_actual = daily.groupby("date")["kwh"].sum().reset_index().rename(columns={"kwh": "daily_kwh"})

    return {
        "metrics": {"mape": mape, "accuracy_pct": accuracy_pct, "r2": r2, "mae": mae},
        "pred_df": pred_df,
        "daily_actual": daily_actual,
    }

if run_btn or (st.session_state.get("metrics") is None):
    with st.spinner("Analyzing data and training regression model..."):
        result = train_and_predict(df_raw)
        st.session_state["metrics"] = result["metrics"]
        st.session_state["pred_df"] = result["pred_df"]
        st.session_state["daily_actual"] = result["daily_actual"]
        st.session_state["kpis"] = compute_kpis(df_raw, int(interval_minutes))
        st.session_state["recs"] = make_recommendations(df_raw, int(interval_minutes))

metrics = st.session_state.get("metrics")
pred_df = st.session_state.get("pred_df")
daily_actual = st.session_state.get("daily_actual")
kpis = st.session_state.get("kpis")
recs = st.session_state.get("recs")


# -----------------------------
# Header + Tabs
# -----------------------------
title_col, badge_col = st.columns([0.75, 0.25])
with title_col:
    st.markdown("# ⚡ AI-Based School Energy Consumption Advisor")
    st.caption("Regression-based prediction + peak analysis + recommendations. Optional OpenAI assistant for reports & Q&A.")
with badge_col:
    st.markdown(
        """
        <div style="text-align:right;">
            <span class="badge">Monitoring</span>
            <span class="badge">Predictive</span>
            <span class="badge">Sustainability</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

tabs = st.tabs(["📊 Dashboard", "🔮 Forecast & Accuracy", "🧭 Advisor Recommendations", "🤖 OpenAI Assistant", "🧾 Data & Export"])


# -----------------------------
# Tab 1: Dashboard
# -----------------------------
with tabs[0]:
    left, right = st.columns([0.65, 0.35], gap="large")

    with left:
        st.subheader("Key energy indicators")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total energy (kWh)", f"{kpis['total_kwh']:.0f}")
        m2.metric("Avg daily energy (kWh)", f"{kpis['avg_daily_kwh']:.1f}")
        m3.metric("Peak demand (kW est.)", f"{kpis['peak_kw_est']:.1f}")
        m4.metric("Days monitored", f"{int(kpis['days'])}")

        st.divider()
        st.subheader("Usage over time")
        plot_df = df_raw[["timestamp", "kwh"]].copy()
        if len(plot_df) > 6000:
            plot_df = plot_df.iloc[::4, :].copy()
        st.line_chart(plot_df.set_index("timestamp"))

        st.divider()
        st.subheader("Daily totals")
        st.bar_chart(daily_actual.set_index("date"))

    with right:
        st.subheader("Cost snapshot")
        total_cost = kpis["total_kwh"] * float(tariff_kes_per_kwh)
        monthly_kwh_est = kpis["avg_daily_kwh"] * 30.0
        monthly_cost_est = monthly_kwh_est * float(tariff_kes_per_kwh)
        monthly_demand_cost = float(demand_charge_kes_per_kw) * float(kpis["peak_kw_est"])
        monthly_total_est = monthly_cost_est + monthly_demand_cost

        c1, c2 = st.columns(2)
        c1.metric("Observed cost (KES)", f"{total_cost:,.0f}")
        c2.metric("Est. monthly cost (KES)", f"{monthly_total_est:,.0f}")
        st.markdown("<div class='smallnote'>Monthly = avg daily × 30 days + optional demand charge.</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Quick savings simulator")
        energy_reduction_pct = st.slider("Planned energy reduction (%)", 0, 40, 14, 1)
        peak_reduction_pct = st.slider("Planned peak reduction (%)", 0, 40, 11, 1)

        new_monthly_kwh = monthly_kwh_est * (1 - energy_reduction_pct / 100.0)
        new_peak_kw = kpis["peak_kw_est"] * (1 - peak_reduction_pct / 100.0)
        new_monthly_cost = new_monthly_kwh * float(tariff_kes_per_kwh)
        new_demand_cost = float(demand_charge_kes_per_kw) * float(new_peak_kw)
        new_total = new_monthly_cost + new_demand_cost
        savings = monthly_total_est - new_total

        s1, s2 = st.columns(2)
        s1.metric("New est. monthly cost (KES)", f"{new_total:,.0f}")
        s2.metric("Est. monthly savings (KES)", f"{savings:,.0f}")

        st.divider()
        st.subheader("Peak hours (average)")
        by_hour = recs["by_hour"].copy()
        by_hour["hour_label"] = by_hour["hour"].apply(lambda h: f"{int(h):02d}:00")
        st.dataframe(by_hour[["hour_label", "avg_kwh"]].head(12), use_container_width=True, hide_index=True)


# -----------------------------
# Tab 2: Forecast & Accuracy
# -----------------------------
with tabs[1]:
    st.subheader("Model performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction accuracy (1 - MAPE)", f"{metrics['accuracy_pct']:.1f}%")
    m2.metric("R² score", f"{metrics['r2']:.3f}")
    m3.metric("MAE (kWh/interval)", f"{metrics['mae']:.3f}")
    m4.metric("MAPE", f"{metrics['mape']*100:.2f}%")

    st.divider()
    st.subheader("Actual vs Predicted (test period)")
    if pred_df is not None and len(pred_df) > 0:
        st.line_chart(pred_df.set_index("timestamp")[["actual_kwh", "pred_kwh"]])
        st.caption("Test window uses a time-based split (last N days).")
    else:
        st.info("Not enough data for a stable test window.")


# -----------------------------
# Tab 3: Recommendations
# -----------------------------
with tabs[2]:
    st.subheader("Advisory Recommendations")
    st.markdown(
        f"<span class='badge'>Peak hours: {', '.join([f'{h:02d}:00' for h in recs['top_hours']])}</span> "
        f"<span class='badge'>After-hours share: {recs['after_hours_pct']:.1f}%</span> "
        f"<span class='badge'>Weekend share: {recs['weekend_pct']:.1f}%</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    for i, r in enumerate(recs["recommendations"], start=1):
        with st.expander(f"✅ Recommendation {i}: {r['title']}", expanded=(i <= 2)):
            st.markdown(f"**Why:** {r['why']}")
            st.markdown(f"**Action:** {r['action']}")
            st.markdown(f"**Expected impact:** {r['impact_hint']}")


# -----------------------------
# Tab 4: OpenAI Assistant (chat + report)
# -----------------------------
with tabs[3]:
    st.subheader("🤖 OpenAI Energy Advisor (optional)")
    st.caption("If you enable OpenAI in the sidebar and provide an API key, you can generate a polished report and ask questions about the data.")

    if not enable_openai:
        st.info("Enable **OpenAI features** in the sidebar to use this tab.")
    else:
        client, err = get_openai_client()
        if err:
            st.error(err)
        else:
            # Build a compact, privacy-friendly summary (aggregated info only)
            by_hour = recs["by_hour"].copy()
            by_hour = by_hour.sort_values("avg_kwh", ascending=False).head(10)
            daily_top = daily_actual.sort_values("daily_kwh", ascending=False).head(7)

            summary = {
                "interval_minutes": int(interval_minutes),
                "kpis": kpis,
                "model_metrics": metrics,
                "peak_hours_avg": [int(h) for h in recs["top_hours"]],
                "after_hours_pct": recs["after_hours_pct"],
                "weekend_pct": recs["weekend_pct"],
                "top_hours_table": by_hour.to_dict(orient="records"),
                "top_daily_totals": daily_top.to_dict(orient="records"),
                "current_recommendations": recs["recommendations"],
                "tariff_kes_per_kwh": float(tariff_kes_per_kwh),
                "demand_charge_kes_per_kw": float(demand_charge_kes_per_kw),
            }

            colA, colB = st.columns([0.55, 0.45], gap="large")

            with colA:
                st.markdown("### 📄 Generate a report")
                report_style = st.selectbox(
                    "Report style",
                    ["Admin summary (1 page)", "Technical report (method + results)", "Presentation bullets"],
                    index=0,
                )
                include_targets = st.checkbox("Include target results (e.g., 14% energy, 11% peak)", value=True)
                target_energy = st.slider("Target energy reduction (%)", 0, 40, 14, 1)
                target_peak = st.slider("Target peak reduction (%)", 0, 40, 11, 1)

                if st.button("✨ Generate OpenAI Report", type="primary"):
                    instructions = (
                        "You are an energy efficiency advisor for a secondary school. "
                        "Write clear, practical recommendations grounded in the provided summary. "
                        "Do not invent measurements not present in the summary. "
                        "Use bullet points and short sections."
                    )
                    user_input = {
                        "task": "Write an energy consumption advisory report.",
                        "style": report_style,
                        "include_targets": bool(include_targets),
                        "targets": {"energy_reduction_pct": int(target_energy), "peak_reduction_pct": int(target_peak)},
                        "summary": summary,
                    }

                    with st.spinner("Calling OpenAI..."):
                        text = openai_generate_text(
                            client=client,
                            model=openai_model,
                            instructions=instructions,
                            user_input=json.dumps(user_input),
                        )
                    st.session_state["ai_report"] = text

                if st.session_state.get("ai_report"):
                    st.text_area("Generated report", st.session_state["ai_report"], height=420)
                    st.download_button(
                        "⬇️ Download report (TXT)",
                        data=st.session_state["ai_report"].encode("utf-8"),
                        file_name="energy_advisor_report.txt",
                        mime="text/plain",
                    )

            with colB:
                st.markdown("### 💬 Ask the advisor")
                if "chat_messages" not in st.session_state:
                    st.session_state["chat_messages"] = [
                        {"role": "system", "content": "You help interpret school energy data summaries and suggest practical actions."}
                    ]

                # Show chat history (skip system)
                for msg in st.session_state["chat_messages"]:
                    if msg["role"] == "user":
                        st.chat_message("user").write(msg["content"])
                    elif msg["role"] == "assistant":
                        st.chat_message("assistant").write(msg["content"])

                q = st.chat_input("Ask about peaks, costs, or what actions to prioritize…")
                if q:
                    st.session_state["chat_messages"].append({"role": "user", "content": q})

                    instructions = (
                        "You are an energy advisor. Answer only using the provided summary and general energy best practices. "
                        "Be specific and actionable. If the summary lacks info, say what’s missing."
                    )

                    # Keep context short: last 8 turns + summary
                    last_msgs = [m for m in st.session_state["chat_messages"] if m["role"] != "system"][-8:]
                    chat_payload = {
                        "summary": summary,
                        "conversation": last_msgs,
                        "user_question": q,
                    }

                    with st.spinner("Calling OpenAI..."):
                        answer = openai_generate_text(
                            client=client,
                            model=openai_model,
                            instructions=instructions,
                            user_input=json.dumps(chat_payload),
                        )

                    st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
                    st.rerun()


# -----------------------------
# Tab 5: Data & Export
# -----------------------------
with tabs[4]:
    st.subheader("Data preview")
    st.dataframe(df_raw.head(50), use_container_width=True)

    st.divider()
    st.subheader("Downloads")
    st.download_button(
        "⬇️ Download cleaned dataset (CSV)",
        data=df_raw.to_csv(index=False).encode("utf-8"),
        file_name="energy_cleaned.csv",
        mime="text/csv",
    )

    if pred_df is not None and len(pred_df) > 0:
        st.download_button(
            "⬇️ Download predictions (CSV)",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name="energy_predictions.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("CSV template")
    tmpl = pd.DataFrame(
        {
            "timestamp": ["2026-02-01 08:00:00", "2026-02-01 08:15:00"],
            "kwh": [12.3, 11.8],
            "temperature_c": [29.0, 29.1],
            "occupancy": [0.75, 0.78],
        }
    )
    st.code(tmpl.to_csv(index=False), language="text")


# -----------------------------
# Footer notes
# -----------------------------
st.divider()
with st.expander("Deployment notes (OpenAI)", expanded=False):
    st.markdown(
        """
**To enable OpenAI on Streamlit Cloud**
1. Add `openai` to `requirements.txt`. :contentReference[oaicite:4]{index=4}  
2. In Streamlit Cloud → **App settings → Secrets**, add:
