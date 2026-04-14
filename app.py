"""
Loan Tape Optimizer — Streamlit MVP
Capital Markets / Mortgage Secondary Marketing Tool

Run with:
    pip install streamlit pandas openpyxl xlsxwriter
    streamlit run loan_tape_optimizer.py
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Tape Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Base theme */
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a3045; }
    .main .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a2035;
        border: 1px solid #2a3045;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label { color: #7b8ab8 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8ff !important; font-size: 1.6rem; font-weight: 700; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.82rem; }

    /* Section headers */
    .section-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4a90e2;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2a3045;
    }

    /* Flag badges */
    .flag-strong   { background:#1a4a2e; color:#4ade80; border:1px solid #22c55e; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .flag-standard { background:#1a2e4a; color:#60a5fa; border:1px solid #3b82f6; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .flag-review   { background:#3a2e1a; color:#fbbf24; border:1px solid #f59e0b; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
    .flag-outlier  { background:#3a1a1a; color:#f87171; border:1px solid #ef4444; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; }

    /* Strategy cards */
    .strategy-card {
        background: #1a2035;
        border: 1px solid #2a3045;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .strategy-title { font-size: 1rem; font-weight: 700; color: #e2e8ff; margin-bottom: 0.3rem; }
    .strategy-desc  { font-size: 0.82rem; color: #7b8ab8; margin-bottom: 0.8rem; }

    /* Recommendation box */
    .rec-box {
        background: linear-gradient(135deg, #1a2e4a 0%, #1a2035 100%);
        border: 1px solid #3b82f6;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1rem;
    }
    .rec-title { font-size: 1rem; font-weight: 700; color: #60a5ff; margin-bottom: 0.5rem; }
    .rec-body  { font-size: 0.88rem; color: #c0cce8; line-height: 1.6; }

    /* Dataframe tweaks */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Sidebar labels */
    .sidebar-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4a90e2;
        margin-top: 1rem;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS — DATA CLEANING
# ─────────────────────────────────────────────

def load_excel(file, sheet_name: str) -> pd.DataFrame:
    """Load a single sheet from an Excel file into a DataFrame."""
    return pd.read_excel(file, sheet_name=sheet_name)


def get_sheet_names(file) -> list[str]:
    """Return list of sheet names from an Excel file."""
    xl = pd.ExcelFile(file)
    return xl.sheet_names


def clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, stripping stray characters."""
    return pd.to_numeric(series.astype(str).str.replace(r"[%,$,\s]", "", regex=True), errors="coerce")


def normalize_ltv(series: pd.Series) -> pd.Series:
    """If LTV is stored as a decimal (e.g. 0.75), convert to percentage."""
    if series.dropna().median() < 2:
        return series * 100
    return series


def normalize_price(series: pd.Series) -> pd.Series:
    """Prices sometimes stored as 100.00 handle-based, leave as-is."""
    return series


def clean_dataframe(
    df: pd.DataFrame,
    col_map: dict,
) -> pd.DataFrame:
    """
    Rename columns per user mapping, coerce types, normalize LTV, drop
    rows missing any core field.

    col_map keys: loan_id, buy_price, fico, ltv
    Optional keys: loan_amount, product, occupancy
    """
    rename = {v: k for k, v in col_map.items() if v}
    df = df.rename(columns=rename)

    required = ["loan_id", "buy_price", "fico", "ltv"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column not mapped: {col}")

    df["buy_price"] = clean_numeric(df["buy_price"])
    df["fico"]      = clean_numeric(df["fico"])
    df["ltv"]       = clean_numeric(df["ltv"])
    df["ltv"]       = normalize_ltv(df["ltv"])

    if "loan_amount" in df.columns:
        df["loan_amount"] = clean_numeric(df["loan_amount"])

    # Drop rows missing core fields
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# HELPER FUNCTIONS — SCORING
# ─────────────────────────────────────────────

def compute_scores(
    df: pd.DataFrame,
    w_price: float,
    w_fico: float,
    w_ltv: float,
) -> pd.Series:
    """
    Min-max normalise each component, then compute a weighted score (0–100).
    LTV is inverted so lower LTV = higher score.
    """
    def minmax(s):
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / rng

    norm_price = minmax(df["buy_price"])
    norm_fico  = minmax(df["fico"])
    norm_ltv   = 1 - minmax(df["ltv"])          # inverted

    total_w = w_price + w_fico + w_ltv
    if total_w == 0:
        total_w = 1

    score = (
        (w_price * norm_price + w_fico * norm_fico + w_ltv * norm_ltv)
        / total_w
    ) * 100

    return score.round(2)


# ─────────────────────────────────────────────
# HELPER FUNCTIONS — BUCKETING
# ─────────────────────────────────────────────

FICO_BINS   = [0,    680,  720,  760,  9999]
FICO_LABELS = ["<680", "680–719", "720–759", "760+"]

LTV_BINS    = [0,    70,   75,   80,   200]
LTV_LABELS  = ["<70", "70–75", "75–80", ">80"]


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fico_bucket"] = pd.cut(
        df["fico"], bins=FICO_BINS, labels=FICO_LABELS, right=False
    ).astype(str)
    df["ltv_bucket"] = pd.cut(
        df["ltv"], bins=LTV_BINS, labels=LTV_LABELS, right=False
    ).astype(str)
    df["combined_bucket"] = df["fico_bucket"] + " / " + df["ltv_bucket"]
    return df


# ─────────────────────────────────────────────
# HELPER FUNCTIONS — FLAGGING
# ─────────────────────────────────────────────

def assign_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score_10th = df["score"].quantile(0.10)

    conditions = [
        (df["ltv"] > 75) | (df["fico"] < 700) | (df["score"] <= score_10th),
        (df["ltv"] > 75) | (df["fico"] < 700),
        (df["score"] <= score_10th),
    ]

    def flag_row(row):
        is_outlier_score = row["score"] <= score_10th
        is_high_ltv      = row["ltv"] > 75
        is_low_fico      = row["fico"] < 700

        risk_count = sum([is_high_ltv, is_low_fico])

        if risk_count >= 2 or (risk_count >= 1 and is_outlier_score):
            return "Outlier"
        elif risk_count == 1:
            return "Review"
        elif is_outlier_score:
            return "Review"
        elif row["score"] >= df["score"].quantile(0.75):
            return "Strong"
        else:
            return "Standard"

    df["flag"] = df.apply(flag_row, axis=1)
    return df


# ─────────────────────────────────────────────
# HELPER FUNCTIONS — GROUPING STRATEGIES
# ─────────────────────────────────────────────

def strategy_a(df: pd.DataFrame) -> pd.DataFrame:
    """All loans in one pool."""
    result = df.copy()
    result["group"] = "Pool A — All Loans"
    return (
        result.groupby("group")
        .apply(lambda g: pd.Series({
            "loan_count":     len(g),
            "avg_buy_price":  g["buy_price"].mean(),
            "avg_fico":       g["fico"].mean(),
            "avg_ltv":        g["ltv"].mean(),
            "pct_high_ltv":   (g["ltv"] > 75).mean() * 100,
        }))
        .reset_index()
    )


def strategy_b(df: pd.DataFrame) -> pd.DataFrame:
    """Split by LTV ≤75 vs >75."""
    result = df.copy()
    result["group"] = result["ltv"].apply(
        lambda x: "Pool B1 — LTV ≤75%" if x <= 75 else "Pool B2 — LTV >75%"
    )
    return (
        result.groupby("group")
        .apply(lambda g: pd.Series({
            "loan_count":     len(g),
            "avg_buy_price":  g["buy_price"].mean(),
            "avg_fico":       g["fico"].mean(),
            "avg_ltv":        g["ltv"].mean(),
            "pct_high_ltv":   (g["ltv"] > 75).mean() * 100,
        }))
        .reset_index()
    )


def strategy_c(df: pd.DataFrame) -> pd.DataFrame:
    """Split by combined FICO + LTV bucket."""
    result = df.copy()
    result["group"] = "Pool C — " + result["combined_bucket"]
    return (
        result.groupby("group")
        .apply(lambda g: pd.Series({
            "loan_count":     len(g),
            "avg_buy_price":  g["buy_price"].mean(),
            "avg_fico":       g["fico"].mean(),
            "avg_ltv":        g["ltv"].mean(),
            "pct_high_ltv":   (g["ltv"] > 75).mean() * 100,
        }))
        .reset_index()
    )


def format_strategy_table(strat_df: pd.DataFrame) -> pd.DataFrame:
    out = strat_df.copy()
    out["avg_buy_price"] = out["avg_buy_price"].round(4)
    out["avg_fico"]      = out["avg_fico"].round(0).astype(int)
    out["avg_ltv"]       = out["avg_ltv"].round(2)
    out["pct_high_ltv"]  = out["pct_high_ltv"].round(1).astype(str) + "%"
    out.columns          = ["Group", "# Loans", "Avg Buy Price", "Avg FICO", "Avg LTV", "% High LTV"]
    return out


# ─────────────────────────────────────────────
# HELPER FUNCTIONS — RECOMMENDATIONS
# ─────────────────────────────────────────────

def generate_recommendations(df: pd.DataFrame, sa: pd.DataFrame, sb: pd.DataFrame) -> list[dict]:
    """Return a list of recommendation dicts {title, body}."""
    recs = []

    # 1. LTV separation lift
    price_all = sa["avg_buy_price"].iloc[0]
    sb_clean = sb[sb["group"].str.contains("≤75")]
    if not sb_clean.empty:
        price_clean = sb_clean["avg_buy_price"].iloc[0]
        lift_bps = round((price_clean - price_all) * 100, 1)
        if lift_bps > 0:
            recs.append({
                "title": "🎯 Isolate High-LTV Loans for Pricing Lift",
                "body": (
                    f"Separating loans with LTV >75% improves your clean pool's average buy price "
                    f"by approximately <strong>{lift_bps} bps</strong> "
                    f"({price_all:.4f} → {price_clean:.4f}). "
                    "Consider pricing the clean pool separately to capture this execution advantage."
                )
            })

    # 2. Outlier concentration
    n_outlier = (df["flag"] == "Outlier").sum()
    pct_outlier = n_outlier / len(df) * 100
    if pct_outlier > 10:
        recs.append({
            "title": "⚠️ High Outlier Concentration",
            "body": (
                f"{n_outlier} loans ({pct_outlier:.1f}%) are flagged as Outliers. "
                "This is above the 10% threshold where pool contamination risk becomes material. "
                "Consider bid-list exclusions or separate pricing for these loans."
            )
        })

    # 3. FICO degradation warning
    pct_low_fico = (df["fico"] < 700).mean() * 100
    if pct_low_fico > 15:
        recs.append({
            "title": "📉 Sub-700 FICO Exposure",
            "body": (
                f"{pct_low_fico:.1f}% of loans have FICO below 700. "
                "Mixing these with higher-quality collateral will compress pricing across the pool. "
                "Evaluate whether a separate flow or whole-loan sale for sub-700 paper improves net execution."
            )
        })

    # 4. Strong pool highlight
    n_strong = (df["flag"] == "Strong").sum()
    strong_df = df[df["flag"] == "Strong"]
    if n_strong > 0:
        avg_strong_price = strong_df["buy_price"].mean()
        recs.append({
            "title": "✅ Clean Pool Opportunity",
            "body": (
                f"{n_strong} loans are flagged as Strong with an average buy price of "
                f"<strong>{avg_strong_price:.4f}</strong>. "
                "Pooling these together creates a premium execution vehicle. "
                "Protect this collateral by keeping it away from high-LTV or low-FICO paper."
            )
        })

    if not recs:
        recs.append({
            "title": "✅ Pool Quality Looks Healthy",
            "body": (
                "No major risk concentrations detected. "
                "Your loan tape appears relatively clean — proceed with standard pooling strategy."
            )
        })

    return recs


# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

def build_export(df: pd.DataFrame, sa, sb, sc) -> bytes:
    """
    Build a multi-sheet Excel export:
      Sheet 1 — Full enriched tape
      Sheet 2 — Strategy A
      Sheet 3 — Strategy B
      Sheet 4 — Strategy C
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        # Sheet 1 — enriched tape
        export_cols = [c for c in [
            "loan_id", "buy_price", "fico", "ltv",
            "loan_amount", "product", "occupancy",
            "score", "fico_bucket", "ltv_bucket", "combined_bucket", "flag"
        ] if c in df.columns]
        df[export_cols].to_excel(writer, sheet_name="Enriched Tape", index=False)

        # Strategy sheets
        format_strategy_table(sa).to_excel(writer, sheet_name="Strategy A — All", index=False)
        format_strategy_table(sb).to_excel(writer, sheet_name="Strategy B — LTV Split", index=False)
        format_strategy_table(sc).to_excel(writer, sheet_name="Strategy C — FICO+LTV", index=False)

        # Auto-fit columns (best effort)
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.set_zoom(90)

    return buf.getvalue()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    """Render sidebar and return all user-configured settings."""
    with st.sidebar:
        st.markdown("## 📊 Loan Tape Optimizer")
        st.markdown("---")

        # ── File Upload ──────────────────────────────
        st.markdown('<div class="sidebar-label">1 · Upload Tape</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose Excel file (.xlsx)", type=["xlsx"])

        sheet_name = None
        if uploaded:
            sheets = get_sheet_names(uploaded)
            sheet_name = st.selectbox("Sheet", sheets)

        st.markdown("---")

        # ── Column Mapping ───────────────────────────
        st.markdown('<div class="sidebar-label">2 · Column Mapping</div>', unsafe_allow_html=True)

        col_map = {}
        # Placeholders filled after file load
        col_map["uploaded"] = uploaded
        col_map["sheet"]    = sheet_name

        st.markdown("---")

        # ── Score Weights ────────────────────────────
        st.markdown('<div class="sidebar-label">3 · Score Weights</div>', unsafe_allow_html=True)
        w_price = st.slider("Buy Price Weight",  0.0, 1.0, 0.5, 0.05)
        w_fico  = st.slider("FICO Weight",        0.0, 1.0, 0.3, 0.05)
        w_ltv   = st.slider("LTV Weight (Inv.)", 0.0, 1.0, 0.2, 0.05)

        st.markdown("---")

        # ── Filters ──────────────────────────────────
        st.markdown('<div class="sidebar-label">4 · Filters</div>', unsafe_allow_html=True)
        min_fico = st.number_input("Min FICO",      value=0,   min_value=0,    max_value=900)
        max_ltv  = st.number_input("Max LTV (%)",   value=100, min_value=0,    max_value=200)
        min_price= st.number_input("Min Buy Price", value=0.0, min_value=0.0,  format="%.4f")

        return {
            "uploaded":  uploaded,
            "sheet":     sheet_name,
            "w_price":   w_price,
            "w_fico":    w_fico,
            "w_ltv":     w_ltv,
            "min_fico":  min_fico,
            "max_ltv":   max_ltv,
            "min_price": min_price,
        }


# ─────────────────────────────────────────────
# COLUMN MAPPING FORM
# ─────────────────────────────────────────────

def render_column_mapping(raw_df: pd.DataFrame) -> dict:
    """Render column mapping selectors below the preview table."""
    cols = ["(none)"] + list(raw_df.columns)

    st.markdown('<div class="section-header">Column Mapping</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    loan_id   = c1.selectbox("Loan ID *",    cols, index=_best_match(cols, ["loan_id","loanid","loan","id","loan_number","loannumber"]))
    buy_price = c2.selectbox("Buy Price *",  cols, index=_best_match(cols, ["bs_buypluslpc","buypluslpc","buy_price","buyprice","price"]))
    fico      = c3.selectbox("FICO *",       cols, index=_best_match(cols, ["fico","credit_score","creditscore","score"]))
    ltv       = c4.selectbox("LTV *",        cols, index=_best_match(cols, ["ltv","loan_to_value","loantoval"]))

    c5, c6, c7 = st.columns(3)
    loan_amt  = c5.selectbox("Loan Amount (opt.)",cols, index=_best_match(cols, ["loan_amount","loanamount","amount","balance","unpaid_balance"]))
    product   = c6.selectbox("Product (opt.)",    cols, index=_best_match(cols, ["product","loan_type","loantype","program"]))
    occupancy = c7.selectbox("Occupancy (opt.)",  cols, index=_best_match(cols, ["occupancy","occ","occupancytype"]))

    return {
        "loan_id":     loan_id   if loan_id   != "(none)" else None,
        "buy_price":   buy_price if buy_price  != "(none)" else None,
        "fico":        fico      if fico       != "(none)" else None,
        "ltv":         ltv       if ltv        != "(none)" else None,
        "loan_amount": loan_amt  if loan_amt   != "(none)" else None,
        "product":     product   if product    != "(none)" else None,
        "occupancy":   occupancy if occupancy  != "(none)" else None,
    }


def _best_match(options: list, candidates: list) -> int:
    """Return index of first option that fuzzy-matches any candidate."""
    opts_lower = [o.lower().replace(" ", "_").replace("-", "_") for o in options]
    for c in candidates:
        for i, o in enumerate(opts_lower):
            if c in o or o in c:
                return i
    return 0


# ─────────────────────────────────────────────
# TAB: OVERVIEW
# ─────────────────────────────────────────────

def render_overview(df: pd.DataFrame):
    st.markdown('<div class="section-header">Portfolio Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Loans",    f"{len(df):,}")
    c2.metric("Avg Buy Price",  f"{df['buy_price'].mean():.4f}")
    c3.metric("Avg FICO",       f"{df['fico'].mean():.0f}")
    c4.metric("Avg LTV",        f"{df['ltv'].mean():.2f}%")
    c5.metric("Avg Score",      f"{df['score'].mean():.1f}")

    st.markdown("<br>", unsafe_allow_html=True)
    c6, c7, c8, c9 = st.columns(4)
    c6.metric("LTV > 75%",       f"{(df['ltv']>75).sum():,}",  delta=f"{(df['ltv']>75).mean()*100:.1f}% of pool", delta_color="inverse")
    c7.metric("FICO < 700",      f"{(df['fico']<700).sum():,}", delta=f"{(df['fico']<700).mean()*100:.1f}% of pool", delta_color="inverse")
    c8.metric("Strong Loans",    f"{(df['flag']=='Strong').sum():,}")
    c9.metric("Outlier Loans",   f"{(df['flag']=='Outlier').sum():,}", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Flag Distribution</div>', unsafe_allow_html=True)

    flag_counts = df["flag"].value_counts().reset_index()
    flag_counts.columns = ["Flag", "Count"]
    flag_counts["Pct"] = (flag_counts["Count"] / len(df) * 100).round(1).astype(str) + "%"

    # Simple color-coded display
    flag_color = {"Strong":"🟢","Standard":"🔵","Review":"🟡","Outlier":"🔴"}
    for _, row in flag_counts.iterrows():
        icon = flag_color.get(row["Flag"], "⚪")
        st.markdown(
            f"{icon} **{row['Flag']}** — {row['Count']:,} loans ({row['Pct']})"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">FICO Bucket Distribution</div>', unsafe_allow_html=True)
    fico_dist = df.groupby("fico_bucket", observed=True).agg(
        Count=("loan_id","count"),
        Avg_Buy_Price=("buy_price","mean"),
        Avg_LTV=("ltv","mean"),
    ).rename(columns={"Avg_Buy_Price":"Avg Buy Price","Avg_LTV":"Avg LTV"})
    fico_dist["Avg Buy Price"] = fico_dist["Avg Buy Price"].round(4)
    fico_dist["Avg LTV"]       = fico_dist["Avg LTV"].round(2)
    st.dataframe(fico_dist, use_container_width=True)


# ─────────────────────────────────────────────
# TAB: LOAN TABLE
# ─────────────────────────────────────────────

def render_loan_table(df: pd.DataFrame):
    st.markdown('<div class="section-header">Interactive Loan Table</div>', unsafe_allow_html=True)

    display_cols = [c for c in [
        "loan_id","buy_price","fico","ltv","loan_amount",
        "product","occupancy","score","combined_bucket","flag"
    ] if c in df.columns]

    col_rename = {
        "loan_id":        "Loan ID",
        "buy_price":      "Buy Price",
        "fico":           "FICO",
        "ltv":            "LTV (%)",
        "loan_amount":    "Loan Amt",
        "product":        "Product",
        "occupancy":      "Occupancy",
        "score":          "Score",
        "combined_bucket":"Bucket",
        "flag":           "Flag",
    }

    display = df[display_cols].rename(columns=col_rename).copy()

    # Sort control
    sort_col = st.selectbox("Sort by", list(display.columns), index=list(display.columns).index("Score") if "Score" in display.columns else 0)
    sort_asc = st.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Ascending"
    display = display.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    st.dataframe(display, use_container_width=True, height=500)
    st.caption(f"Showing {len(display):,} loans")


# ─────────────────────────────────────────────
# TAB: GROUPING ANALYSIS
# ─────────────────────────────────────────────

def render_grouping(df: pd.DataFrame):
    sa = strategy_a(df)
    sb = strategy_b(df)
    sc = strategy_c(df)

    st.markdown('<div class="section-header">Strategy A — Single Pool (Baseline)</div>', unsafe_allow_html=True)
    st.markdown("_All loans combined into one pool. No segmentation._")
    st.dataframe(format_strategy_table(sa), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Strategy B — LTV Split</div>', unsafe_allow_html=True)
    st.markdown("_Two pools: clean (LTV ≤75%) and high-LTV (>75%). Most common institutional approach._")
    st.dataframe(format_strategy_table(sb), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Strategy C — FICO + LTV Granular Buckets</div>', unsafe_allow_html=True)
    st.markdown("_Finest segmentation — each FICO × LTV bucket forms its own pool. Best for large tapes._")
    st.dataframe(format_strategy_table(sc), use_container_width=True)

    return sa, sb, sc


# ─────────────────────────────────────────────
# TAB: RECOMMENDATIONS
# ─────────────────────────────────────────────

def render_recommendations(df: pd.DataFrame, sa, sb):
    recs = generate_recommendations(df, sa, sb)
    st.markdown('<div class="section-header">Analyst Recommendations</div>', unsafe_allow_html=True)

    for rec in recs:
        st.markdown(
            f'<div class="rec-box"><div class="rec-title">{rec["title"]}</div>'
            f'<div class="rec-body">{rec["body"]}</div></div>',
            unsafe_allow_html=True,
        )

    # Bucket-level price table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Bucket-Level Pricing Summary</div>', unsafe_allow_html=True)
    bucket_summary = (
        df.groupby("combined_bucket")
        .agg(
            Loans=("loan_id","count"),
            Avg_Buy_Price=("buy_price","mean"),
            Avg_FICO=("fico","mean"),
            Avg_LTV=("ltv","mean"),
            Avg_Score=("score","mean"),
        )
        .rename(columns={
            "Avg_Buy_Price":"Avg Buy Price",
            "Avg_FICO":"Avg FICO",
            "Avg_LTV":"Avg LTV",
            "Avg_Score":"Avg Score",
        })
        .sort_values("Avg Buy Price", ascending=False)
    )
    bucket_summary["Avg Buy Price"] = bucket_summary["Avg Buy Price"].round(4)
    bucket_summary["Avg FICO"]      = bucket_summary["Avg FICO"].round(0).astype(int)
    bucket_summary["Avg LTV"]       = bucket_summary["Avg LTV"].round(2)
    bucket_summary["Avg Score"]     = bucket_summary["Avg Score"].round(1)
    st.dataframe(bucket_summary, use_container_width=True)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    settings = render_sidebar()

    uploaded  = settings["uploaded"]
    sheet     = settings["sheet"]

    if uploaded is None:
        st.markdown("## 📊 Loan Tape Optimizer")
        st.markdown(
            "Upload an Excel bid tape using the sidebar to get started. "
            "The tool will help you:\n"
            "- Score and bucket loans by risk quality\n"
            "- Identify outliers dragging down pool pricing\n"
            "- Compare pooling strategies to maximize execution\n"
            "- Generate analyst-grade recommendations"
        )
        st.info("← Use the sidebar to upload your file.")
        return

    # ── Load raw data ───────────────────────────────────────
    try:
        raw_df = load_excel(uploaded, sheet)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return

    st.markdown("### Preview — Raw Data")
    st.dataframe(raw_df.head(10), use_container_width=True)

    # ── Column mapping ──────────────────────────────────────
    col_map = render_column_mapping(raw_df)

    # Validate required fields selected
    if not all([col_map.get("loan_id"), col_map.get("buy_price"), col_map.get("fico"), col_map.get("ltv")]):
        st.warning("Please map all required columns (Loan ID, Buy Price, FICO, LTV) above.")
        return

    # ── Clean ───────────────────────────────────────────────
    try:
        df = clean_dataframe(raw_df, col_map)
    except Exception as e:
        st.error(f"Data cleaning error: {e}")
        return

    # ── Score ───────────────────────────────────────────────
    df["score"] = compute_scores(df, settings["w_price"], settings["w_fico"], settings["w_ltv"])

    # ── Bucket ──────────────────────────────────────────────
    df = assign_buckets(df)

    # ── Flag ────────────────────────────────────────────────
    df = assign_flags(df)

    # ── Apply sidebar filters ───────────────────────────────
    df_filtered = df[
        (df["fico"]      >= settings["min_fico"])  &
        (df["ltv"]       <= settings["max_ltv"])   &
        (df["buy_price"] >= settings["min_price"])
    ].copy()

    n_filtered = len(df) - len(df_filtered)
    if n_filtered > 0:
        st.info(f"Filters active: {n_filtered:,} loans excluded · Showing {len(df_filtered):,} loans")

    if df_filtered.empty:
        st.warning("No loans match the current filters.")
        return

    st.markdown("---")

    # ── Tabs ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📋 Loan Table", "🗂 Grouping Analysis", "💡 Recommendations"])

    with tab1:
        render_overview(df_filtered)

    with tab2:
        render_loan_table(df_filtered)

    with tab3:
        sa, sb, sc = render_grouping(df_filtered)

    with tab4:
        # Need sa, sb from grouping tab — recompute if tab4 opened first
        sa_r = strategy_a(df_filtered)
        sb_r = strategy_b(df_filtered)
        render_recommendations(df_filtered, sa_r, sb_r)

    # ── Export ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    sa_x = strategy_a(df_filtered)
    sb_x = strategy_b(df_filtered)
    sc_x = strategy_c(df_filtered)
    excel_bytes = build_export(df_filtered, sa_x, sb_x, sc_x)
    st.download_button(
        label="⬇️ Download Enriched Tape + Strategy Summary (.xlsx)",
        data=excel_bytes,
        file_name="loan_tape_optimized.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
