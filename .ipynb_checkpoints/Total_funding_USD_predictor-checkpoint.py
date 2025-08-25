# app.py
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from datetime import datetime, timezone

# -------------------------------
# CONFIG – update paths if needed
# -------------------------------
FUND_MODEL_PATH   = Path("total_funding_model.pkl")
FUND_SCALER_X     = Path("scaler_X.pkl")
FUND_SCALER_Y     = Path("scaler_Y.pkl")

REV_MODEL_PATH    = Path("rev_range_model.pkl")
REV_SCALER_X      = Path("scaler_X_rev.pkl")
REV_SCALER_Y      = Path("scaler_Y_rev.pkl")

FEATURES_JSON     = Path("feature_columns.json")
OPTIONS_SOURCE_CSV= Path("final_df_totrain.csv")   # used only to prepare UI options

# One-hot prefixes (match your training!)
IND_PREFIX   = "Industry_"
CITY_PREFIX  = "City_"
STATE_PREFIX = "State_"
CTRY_PREFIX  = "Country_"

# Employee ranges → (min_employees, max_employees)
EMPLOYEE_BUCKETS = {
    "1-10":        (1, 10),
    "11-50":       (11, 50),
    "51-100":      (51, 100),
    "101-250":     (101, 250),
    "251-500":     (251, 500),
    "501-1000":    (501, 1000),
    "1001-5000":   (1001, 5000),
    "5001-10000":  (5001, 10000),
    "10001+":      (10001, 10001),
}

# -------------------------------
# Helpers
# -------------------------------
def split_and_clean(items_str, sep=","):
    if pd.isna(items_str):
        return []
    parts = [p.strip() for p in str(items_str).split(sep)]
    return [p for p in parts if p]

def parse_hq(row_val):
    """Parse 'City, State, Country' -> (city, state, country)."""
    if pd.isna(row_val):
        return ("", "", "")
    parts = [p.strip() for p in str(row_val).split(",") if p.strip() != ""]
    if len(parts) == 3: return parts[0], parts[1], parts[2]
    if len(parts) == 2: return parts[0], parts[1], ""
    if len(parts) == 1: return parts[0], "", ""
    return ("", "", "")

def epoch_seconds_from_date(d):
    # Streamlit date_input returns a date; convert to UTC epoch seconds
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp())

def fmt_usd(x):
    try:
        x = float(x)
    except:
        return str(x)
    if x >= 1e9: return f"${x/1e9:,.2f}B"
    if x >= 1e6: return f"${x/1e6:,.2f}M"
    return f"${x:,.0f}"

@st.cache_resource(show_spinner=False)
def load_feature_columns():
    return json.loads(FEATURES_JSON.read_text(encoding="utf-8"))

@st.cache_resource(show_spinner=False)
def load_models_and_scalers():
    # Funding artifacts
    fund_model  = joblib.load(FUND_MODEL_PATH)
    fund_sX     = joblib.load(FUND_SCALER_X)
    fund_sY     = joblib.load(FUND_SCALER_Y)
    # Revenue artifacts
    rev_model   = joblib.load(REV_MODEL_PATH)
    rev_sX      = joblib.load(REV_SCALER_X)
    rev_sY      = joblib.load(REV_SCALER_Y)
    return fund_model, fund_sX, fund_sY, rev_model, rev_sX, rev_sY

@st.cache_data(show_spinner=False)
def load_options():
    df = pd.read_csv(OPTIONS_SOURCE_CSV, low_memory=False)
    ind_col = next((c for c in ["Industries","Industry","industries","industry"] if c in df.columns), None)
    hq_col  = next((c for c in ["Headquarters Location","Headquarters","HQ","Headquarters_Location"] if c in df.columns), None)

    industries = set()
    cities, states, countries = set(), set(), set()

    if ind_col:
        for s in df[ind_col].dropna():
            industries.update(split_and_clean(s))
    if hq_col:
        for s in df[hq_col].dropna():
            cty, stt, ctyr = parse_hq(s)
            if cty:  cities.add(cty)
            if stt:  states.add(stt)
            if ctyr: countries.add(ctyr)

    # Optionally surface common categoricals if you used them in training (one-hot prefixes will handle them if present)
    cat_choices = {}
    for col in ["Company Type", "Funding Status", "Last Funding Type"]:
        if col in df.columns:
            vals = [x for x in df[col].dropna().astype(str).unique().tolist() if x != ""]
            cat_choices[col] = sorted(vals)

    return {
        "industries": sorted(industries),
        "cities": sorted(cities),
        "states": sorted(states),
        "countries": sorted(countries),
        "cat_choices": cat_choices
    }

def build_zero_row(feature_cols):
    return pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols, dtype=float)

def set_onehot(row_df, prefix, value, feature_cols):
    if not value: return
    col = f"{prefix}{value}"
    if col in feature_cols:
        row_df.at[0, col] = 1.0

def set_multihot(row_df, prefix, values, feature_cols):
    for v in values:
        set_onehot(row_df, prefix, v, feature_cols)

def set_numeric(row_df, name, value):
    if name in row_df.columns:
        row_df.at[0, name] = float(value)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Funding & Revenue Predictor", page_icon="💰", layout="wide")
st.title("💰 Startup Financials — Prediction Suite")

with st.sidebar:
    st.subheader("Files")
    st.caption(f"Funding model: `{FUND_MODEL_PATH.name}` | scalers: `{FUND_SCALER_X.name}`, `{FUND_SCALER_Y.name}`")
    st.caption(f"Revenue model: `{REV_MODEL_PATH.name}` | scalers: `{REV_SCALER_X.name}`, `{REV_SCALER_Y.name}`")
    st.caption(f"Features: `{FEATURES_JSON.name}`")
    st.caption(f"Options CSV: `{OPTIONS_SOURCE_CSV.name}`")
    st.markdown("---")
    st.caption("This app constructs inputs to exactly match your training feature columns.")

# Load artifacts
feature_cols = load_feature_columns()
fund_model, fund_sX, fund_sY, rev_model, rev_sX, rev_sY = load_models_and_scalers()
opts = load_options()

# Shared Inputs (one time, used by both tabs)
left, right = st.columns(2, gap="large")

with left:
    st.subheader("🏷️ Company Profile")
    industries_sel = st.multiselect("Industries", options=opts["industries"])
    city_sel       = st.selectbox("City",    options=[""] + opts["cities"])
    state_sel      = st.selectbox("State",   options=[""] + opts["states"])
    country_sel    = st.selectbox("Country", options=[""] + opts["countries"])
    employee_bucket= st.selectbox("Employees (range)", options=list(EMPLOYEE_BUCKETS.keys()), index=0)

    # Optionally include these if you one-hot encoded them in training (prefix must match)
    company_type   = st.selectbox("Company Type", options=[""] + opts["cat_choices"].get("Company Type", []))
    funding_status = st.selectbox("Funding Status", options=[""] + opts["cat_choices"].get("Funding Status", []))
    last_funding_type = st.selectbox("Last Funding Type", options=[""] + opts["cat_choices"].get("Last Funding Type", []))

with right:
    st.subheader("📈 Operating Metrics")
    num_founders     = st.number_input("Number of Founders", min_value=0, value=1, step=1)
    num_investors    = st.number_input("Number of Investors", min_value=0, value=0, step=1)
    num_rounds       = st.number_input("Number of Funding Rounds", min_value=0, value=0, step=1)
    last_funding_usd = st.number_input("Last Funding Amount Currency (in USD)", min_value=0.0, value=0.0, step=10000.0)
    patents_granted  = st.number_input("IPqwery - Patents Granted", min_value=0, value=0, step=1)
    min_rev_hint     = st.number_input("Min Revenue (hint only, optional)", min_value=0.0, value=0.0, step=10000.0, help="Optional hint; only used if present in training as a numeric feature.")
    max_rev_hint     = st.number_input("Max Revenue (hint only, optional)", min_value=0.0, value=0.0, step=10000.0, help="Optional hint; only used if present in training as a numeric feature.")
    actively_hiring  = st.selectbox("Actively Hiring Binary", options=[0, 1], index=0)

st.subheader("📅 Founded Date")
founded_date = st.date_input("Founding date", format="YYYY-MM-DD")
founded_ts   = epoch_seconds_from_date(founded_date)  # multiply by 1000 if your training used ms

st.markdown("---")

# Tabs
tab_fund, tab_rev = st.tabs(["💵 Funding Prediction", "📊 Revenue Range Prediction"])

def make_model_row():
    """Build a single row vector aligned to training feature columns."""
    row = build_zero_row(feature_cols)

    # Multi-hot
    set_multihot(row, IND_PREFIX, industries_sel, feature_cols)

    # HQ one-hots
    set_onehot(row, CITY_PREFIX,  city_sel,    feature_cols)
    set_onehot(row, STATE_PREFIX, state_sel,   feature_cols)
    set_onehot(row, CTRY_PREFIX,  country_sel, feature_cols)

    # Employees
    emp_min, emp_max = EMPLOYEE_BUCKETS[employee_bucket]
    set_numeric(row, "Minimum Employees", emp_min)
    set_numeric(row, "Maximum Employees", emp_max)

    # Other categoricals (if one-hot columns exist with these prefixes)
    set_onehot(row, "Company Type_",      company_type,   feature_cols)
    set_onehot(row, "Funding Status_",    funding_status, feature_cols)
    set_onehot(row, "Last Funding Type_", last_funding_type, feature_cols)

    # Numerics (only set if they exist in training)
    set_numeric(row, "Number of Founders", num_founders)
    set_numeric(row, "Number of Investors", num_investors)
    set_numeric(row, "Number of Funding Rounds", num_rounds)
    set_numeric(row, "Last Funding Amount Currency (in USD)", last_funding_usd)
    set_numeric(row, "IPqwery - Patents Granted", patents_granted)
    set_numeric(row, "Minimum Estimated Revenue (USD)", min_rev_hint)  # safe: will only apply if present
    set_numeric(row, "Maximum Estimated Revenue (USD)", max_rev_hint)  # safe: will only apply if present
    set_numeric(row, "Founded Date_Timestamp", founded_ts)
    set_numeric(row, "Actively Hiring Binary", actively_hiring)

    # Re-align to training order
    row = row[feature_cols]
    return row

with tab_fund:
    st.subheader("💵 Predict Total Funding (USD)")
    if st.button("Predict Funding", type="primary"):
        row = make_model_row()
        X_scaled = fund_sX.transform(row.values)
        y_scaled = fund_model.predict(X_scaled).reshape(-1, 1)
        y_pred   = fund_sY.inverse_transform(y_scaled).ravel()[0]
        st.success(f"Predicted Total Funding: **{fmt_usd(y_pred)}**")

        with st.expander("Show non-zero encoded inputs"):
            st.dataframe(row.loc[:, row.iloc[0] != 0].T, use_container_width=True)

        # Optional importances
        if hasattr(fund_model, "feature_importances_"):
            st.markdown("**Top 20 Feature Importances**")
            imp = pd.Series(fund_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
            st.dataframe(imp.rename("importance").to_frame(), use_container_width=True)

with tab_rev:
    st.subheader("📊 Predict Revenue Range (USD)")
    st.caption("Predicts both **Minimum** and **Maximum** revenue together.")
    if st.button("Predict Revenue Range", type="primary"):
        row = make_model_row()
        X_scaled = rev_sX.transform(row.values)
        y_scaled_2 = rev_model.predict(X_scaled).reshape(1, -1)  # shape (1,2)
        y_pred_2   = rev_sY.inverse_transform(y_scaled_2).ravel()
        min_pred, max_pred = float(y_pred_2[0]), float(y_pred_2[1])
        if min_pred > max_pred:
            min_pred, max_pred = max_pred, min_pred

        st.success(f"Predicted Revenue Range: **{fmt_usd(min_pred)} – {fmt_usd(max_pred)}**")

        with st.expander("Show non-zero encoded inputs"):
            st.dataframe(row.loc[:, row.iloc[0] != 0].T, use_container_width=True)
