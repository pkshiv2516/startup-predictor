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
MODEL_PATH        = Path("total_funding_model.pkl")
SCALER_X_PATH     = Path("scaler_X.pkl")
SCALER_Y_PATH     = Path("scaler_Y.pkl")
FEATURES_JSON     = Path("feature_columns.json")
OPTIONS_SOURCE_CSV= Path("final_df_totrain.csv")   # used for Industries & HQ options only

# Expected feature name prefixes (match your training columns)
IND_PREFIX   = "Industry_"
CITY_PREFIX  = "City_"
STATE_PREFIX = "State_"
CTRY_PREFIX  = "Country_"

# Numeric feature names used during training (existing in your model’s feature list)
NUM_FEATURES = [
    "Number of Founders","Number of Investors","Number of Funding Rounds",
    "Last Funding Amount Currency (in USD)","IPqwery - Patents Granted",
    "Minimum Estimated Revenue (USD)","Maximum Estimated Revenue (USD)",
    "Minimum Employees","Maximum Employees",
    "Founded Date_Timestamp","Actively Hiring Binary"
]

TARGET_READABLE = "Total Funding Amount Currency (in USD)"

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
    "10001+":      (10001, 10001),  # keep same to avoid arbitrary ceilings
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
    """
    Parse 'City, State, Country' → (city, state, country).
    Accepts shorter forms (fills right parts empty).
    """
    if pd.isna(row_val):
        return ("", "", "")
    parts = [p.strip() for p in str(row_val).split(",") if p.strip() != ""]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    if len(parts) == 1:
        return parts[0], "", ""
    return ("", "", "")

def epoch_seconds_from_date(d):
    """UTC epoch seconds from a Python date/datetime."""
    if isinstance(d, datetime):
        dt = d
    else:
        # Streamlit date_input returns a date
        dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp())

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model     = joblib.load(MODEL_PATH)
    scaler_X  = joblib.load(SCALER_X_PATH)
    scaler_Y  = joblib.load(SCALER_Y_PATH)
    feature_cols = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    return model, scaler_X, scaler_Y, feature_cols

@st.cache_data(show_spinner=False)
def load_options():
    # read minimal columns to build options quickly
    usecols = None  # load all; if huge, set to relevant columns
    df = pd.read_csv(OPTIONS_SOURCE_CSV, low_memory=False, usecols=usecols)
    # Guess columns
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

    return {
        "industries": sorted(industries),
        "cities": sorted(cities),
        "states": sorted(states),
        "countries": sorted(countries),
    }

def build_zero_row(feature_cols):
    return pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols, dtype=float)

def set_onehot(row_df, prefix, value, feature_cols):
    if not value:
        return
    col = f"{prefix}{value}"
    if col in feature_cols:
        row_df.at[0, col] = 1.0

def set_multihot(row_df, prefix, values, feature_cols):
    for v in values:
        set_onehot(row_df, prefix, v, feature_cols)

def set_numeric(row_df, name, value):
    if name in row_df.columns:
        row_df.at[0, name] = float(value)

def fmt_usd(x):
    if x >= 1e9:
        return f"${x/1e9:,.2f}B"
    if x >= 1e6:
        return f"${x/1e6:,.2f}M"
    return f"${x:,.0f}"

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Funding Predictor", page_icon="💰", layout="wide")
st.title("💰 Total Funding (USD) — Predictor")

with st.sidebar:
    st.subheader("Files")
    st.caption(f"Model: `{MODEL_PATH.name}`")
    st.caption(f"Scalers: `{SCALER_X_PATH.name}`, `{SCALER_Y_PATH.name}`")
    st.caption(f"Features: `{FEATURES_JSON.name}`")
    st.caption(f"Options CSV: `{OPTIONS_SOURCE_CSV.name}`")
    st.markdown("---")
    st.caption("This app builds inputs to exactly match your training feature columns.")

# Load artifacts + choices
model, scaler_X, scaler_Y, feature_cols = load_artifacts()
opts = load_options()

# ----- Inputs
left, right = st.columns(2, gap="large")

with left:
    st.subheader("🏷️ Company Profile")
    industries_sel = st.multiselect(
        "Industries (comma-separated source)",
        options=opts["industries"],
        help="Values parsed from the 'Industries' column in your CSV."
    )
    city_sel    = st.selectbox("City",    options=[""] + opts["cities"])
    state_sel   = st.selectbox("State",   options=[""] + opts["states"])
    country_sel = st.selectbox("Country", options=[""] + opts["countries"])

    employee_bucket = st.selectbox(
        "Employees (range)",
        options=list(EMPLOYEE_BUCKETS.keys()),
        index=0
    )

with right:
    st.subheader("📈 Operating Metrics")
    num_founders     = st.number_input("Number of Founders", min_value=0, value=1, step=1)
    num_investors    = st.number_input("Number of Investors", min_value=0, value=0, step=1)
    num_rounds       = st.number_input("Number of Funding Rounds", min_value=0, value=0, step=1)
    last_funding_usd = st.number_input("Last Funding Amount Currency (in USD)", min_value=0.0, value=0.0, step=10000.0)
    patents_granted  = st.number_input("IPqwery - Patents Granted", min_value=0, value=0, step=1)
    min_rev          = st.number_input("Minimum Estimated Revenue (USD)", min_value=0.0, value=0.0, step=10000.0)
    max_rev          = st.number_input("Maximum Estimated Revenue (USD)", min_value=0.0, value=0.0, step=10000.0)
    actively_hiring  = st.selectbox("Actively Hiring Binary", options=[0, 1], index=0)

st.subheader("📅 Founded Date")
founded_date = st.date_input("Select founding date", format="YYYY-MM-DD")
founded_ts   = epoch_seconds_from_date(founded_date)  # epoch seconds (adjust if you trained with ms)

st.markdown("---")
if st.button("Predict Total Funding (USD)", type="primary"):
    # Start from all-zero feature vector in training order
    row = build_zero_row(feature_cols)

    # Multi-hot: industries
    set_multihot(row, IND_PREFIX, industries_sel, feature_cols)

    # HQ one-hots (set only if columns exist)
    set_onehot(row, CITY_PREFIX,  city_sel,    feature_cols)
    set_onehot(row, STATE_PREFIX, state_sel,   feature_cols)
    set_onehot(row, CTRY_PREFIX,  country_sel, feature_cols)

    # Employees
    emp_min, emp_max = EMPLOYEE_BUCKETS[employee_bucket]
    set_numeric(row, "Minimum Employees", emp_min)
    set_numeric(row, "Maximum Employees", emp_max)

    # Numerics
    set_numeric(row, "Number of Founders", num_founders)
    set_numeric(row, "Number of Investors", num_investors)
    set_numeric(row, "Number of Funding Rounds", num_rounds)
    set_numeric(row, "Last Funding Amount Currency (in USD)", last_funding_usd)
    set_numeric(row, "IPqwery - Patents Granted", patents_granted)
    set_numeric(row, "Minimum Estimated Revenue (USD)", min_rev)
    set_numeric(row, "Maximum Estimated Revenue (USD)", max_rev)
    set_numeric(row, "Founded Date_Timestamp", founded_ts)
    set_numeric(row, "Actively Hiring Binary", actively_hiring)

    # Re-align to the exact feature order
    row = row[feature_cols]

    # Scale, predict, inverse-scale
    X_scaled = scaler_X.transform(row.values)
    y_scaled = model.predict(X_scaled).reshape(-1, 1)
    y_pred   = scaler_Y.inverse_transform(y_scaled).ravel()[0]

    st.success(f"💵 Predicted Total Funding: **{fmt_usd(y_pred)}**")

    with st.expander("Show non-zero encoded inputs"):
        nonzero = row.loc[:, row.iloc[0] != 0].T
        st.dataframe(nonzero, use_container_width=True)

st.markdown("---")
st.caption("Note: If your training used milliseconds for timestamp, multiply by 1000 before setting `Founded Date_Timestamp`.")
