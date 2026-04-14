bash -lc cat > /mnt/data/loan_tape_optimizer_improved.py <<'PY'
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Loan Tape Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2a3045; }
    .main .block-container { padding-top: 1.5rem; }

    [data-testid="metric-container"] {
        background: #1a2035;
        border: 1px solid #2a3045;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    [data-testid="metric-container"] label {
        color: #7b8ab8 !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8ff !important;
        font-size: 1.6rem;
        font-weight: 700;
    }

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

    .rec-box {
        background: linear-gradient(135deg, #1a2e4a 0%, #1a2035 100%);
        border: 1px solid #3b82f6;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 1.0rem 1.2rem;
        margin-bottom: 1rem;
    }
    .rec-title { font-size: 1rem; font-weight: 700; color: #60a5ff; margin-bottom: 0.35rem; }
    .rec-body  { font-size: 0.9rem; color: #c0cce8; line-height: 1.55; }

    .alert-box {
        background: #1a2035;
        border: 1px solid #2a3045;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .good-box {
        background: #16281f;
        border: 1px solid #25553b;
        border-left: 4px solid #22c55e;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

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

FICO_BINS = [0, 680, 720, 760, 9999]
FICO_LABELS = ["<680", "680-719", "720-759", "760+"]
LTV_BINS = [0, 70, 75, 80, 200]
LTV_LABELS = ["<70", "70-75", "75-80", ">80"]


def load_excel(file, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(file, sheet_name=sheet_name)


def get_sheet_names(file) -> list[str]:
    xl = pd.ExcelFile(file)
    return xl.sheet_names


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[%,$,\s]", "", regex=True), errors="coerce")


def normalize_ltv(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return series
    if series.dropna().median() < 2:
        return series * 100
    return series


def clean_dataframe(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    rename = {v: k for k, v in col_map.items() if v}
    df = df.rename(columns=rename)

    required = ["loan_id", "buy_price", "fico", "ltv"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column not mapped: {col}")

    df["buy_price"] = clean_numeric(df["buy_price"])
    df["fico"] = clean_numeric(df["fico"])
    df["ltv"] = normalize_ltv(clean_numeric(df["ltv"]))

    if "loan_amount" in df.columns:
        df["loan_amount"] = clean_numeric(df["loan_amount"])

    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def compute_scores(df: pd.DataFrame, w_price: float, w_fico: float, w_ltv: float) -> pd.Series:
    def minmax(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        if pd.isna(rng) or rng == 0:
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / rng

    norm_price = minmax(df["buy_price"])
    norm_fico = minmax(df["fico"])
    norm_ltv = 1 - minmax(df["ltv"])

    total_w = max(w_price + w_fico + w_ltv, 1e-9)
    score = ((w_price * norm_price + w_fico * norm_fico + w_ltv * norm_ltv) / total_w) * 100
    return score.round(2)


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fico_bucket"] = pd.cut(out["fico"], bins=FICO_BINS, labels=FICO_LABELS, right=False).astype(str)
    out["ltv_bucket"] = pd.cut(out["ltv"], bins=LTV_BINS, labels=LTV_LABELS, right=False).astype(str)
    out["combined_bucket"] = out["fico_bucket"] + " / " + out["ltv_bucket"]
    return out


def assign_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score_10th = out["score"].quantile(0.10)
    score_75th = out["score"].quantile(0.75)

    def flag_row(row):
        high_ltv = row["ltv"] > 75
        low_fico = row["fico"] < 700
        low_score = row["score"] <= score_10th
        risk_count = sum([high_ltv, low_fico])

        if risk_count >= 2 or (risk_count >= 1 and low_score):
            return "Outlier"
        if risk_count == 1 or low_score:
            return "Review"
        if row["score"] >= score_75th:
            return "Strong"
        return "Standard"

    out["flag"] = out.apply(flag_row, axis=1)
    return out


def weighted_avg(series: pd.Series, weights: pd.Series | None) -> float:
    if weights is None or weights.isna().all() or (weights.fillna(0) <= 0).all():
        return float(series.mean())
    valid = ~(series.isna() | weights.isna())
    if valid.sum() == 0:
        return float(series.mean())
    return float(np.average(series[valid], weights=weights[valid]))


def summarize_group(g: pd.DataFrame) -> pd.Series:
    weights = g["loan_amount"] if "loan_amount" in g.columns else None
    return pd.Series({
        "loan_count": len(g),
        "avg_buy_price": weighted_avg(g["buy_price"], weights),
        "avg_fico": weighted_avg(g["fico"], weights),
        "avg_ltv": weighted_avg(g["ltv"], weights),
        "pct_high_ltv": (g["ltv"] > 75).mean() * 100,
    })


def strategy_a(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["group"] = "Pool A - All Loans"
    return result.groupby("group").apply(summarize_group).reset_index()


def strategy_b(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["group"] = result["ltv"].apply(lambda x: "Pool B1 - LTV <=75%" if x <= 75 else "Pool B2 - LTV >75%")
    return result.groupby("group").apply(summarize_group).reset_index()


def strategy_c(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["group"] = "Pool C - " + result["combined_bucket"]
    return result.groupby("group").apply(summarize_group).reset_index()


def strategy_d(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    def label(row):
        if row["ltv"] > 75 or row["fico"] < 700:
            return "Pool D2 - Exceptions"
        return "Pool D1 - Clean Core"
    result["group"] = result.apply(label, axis=1)
    return result.groupby("group").apply(summarize_group).reset_index()


def format_strategy_table(strat_df: pd.DataFrame) -> pd.DataFrame:
    out = strat_df.copy()
    if out.empty:
        return out
    out["avg_buy_price"] = out["avg_buy_price"].round(4)
    out["avg_fico"] = out["avg_fico"].round(0)
    out["avg_ltv"] = out["avg_ltv"].round(2)
    out["pct_high_ltv"] = out["pct_high_ltv"].round(1).astype(str) + "%"
    return out.rename(columns={
        "group": "Group",
        "loan_count": "# Loans",
        "avg_buy_price": "Avg Buy Price",
        "avg_fico": "Avg FICO",
        "avg_ltv": "Avg LTV",
        "pct_high_ltv": "% High LTV",
    })


def compare_filter_impact(df_all: pd.DataFrame, df_filtered: pd.DataFrame) -> pd.DataFrame:
    weights_all = df_all["loan_amount"] if "loan_amount" in df_all.columns else None
    weights_filtered = df_filtered["loan_amount"] if "loan_amount" in df_filtered.columns else None

    rows = []
    metrics = [
        ("Loan Count", len(df_all), len(df_filtered), 0),
        ("Avg Buy Price", weighted_avg(df_all["buy_price"], weights_all), weighted_avg(df_filtered["buy_price"], weights_filtered), 4),
        ("Avg FICO", weighted_avg(df_all["fico"], weights_all), weighted_avg(df_filtered["fico"], weights_filtered), 1),
        ("Avg LTV", weighted_avg(df_all["ltv"], weights_all), weighted_avg(df_filtered["ltv"], weights_filtered), 2),
        ("High LTV %", (df_all["ltv"] > 75).mean() * 100, (df_filtered["ltv"] > 75).mean() * 100, 1),
        ("Sub-700 FICO %", (df_all["fico"] < 700).mean() * 100, (df_filtered["fico"] < 700).mean() * 100, 1),
    ]

    for name, before, after, decimals in metrics:
        delta = after - before
        if name == "Avg Buy Price":
            delta_str = f"{delta * 100:+.1f} bps"
        elif name == "Loan Count":
            delta_str = f"{int(delta):+d}"
        elif "%" in name:
            delta_str = f"{delta:+.{decimals}f} pts"
        else:
            delta_str = f"{delta:+.{decimals}f}"
        rows.append({
            "Metric": name,
            "Before": round(before, decimals) if name != "Loan Count" else int(before),
            "After": round(after, decimals) if name != "Loan Count" else int(after),
            "Change": delta_str,
        })
    return pd.DataFrame(rows)


def generate_auto_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = []
    base_price = weighted_avg(df["buy_price"], df["loan_amount"] if "loan_amount" in df.columns else None)

    candidates = [
        ("Exclude LTV > 75%", df[df["ltv"] <= 75]),
        ("Exclude FICO < 700", df[df["fico"] >= 700]),
        ("Exclude LTV > 75% OR FICO < 700", df[(df["ltv"] <= 75) & (df["fico"] >= 700)]),
        ("Keep Strong loans only", df[df["flag"] == "Strong"]),
    ]

    for name, subset in candidates:
        if subset.empty or len(subset) == len(df):
            continue
        price = weighted_avg(subset["buy_price"], subset["loan_amount"] if "loan_amount" in subset.columns else None)
        removed = len(df) - len(subset)
        scenarios.append({
            "Scenario": name,
            "Loans Remaining": len(subset),
            "Loans Removed": removed,
            "Avg Buy Price": round(price, 4),
            "Improvement vs Base": f"{(price - base_price) * 100:+.1f} bps",
            "Avg FICO": round(weighted_avg(subset["fico"], subset["loan_amount"] if "loan_amount" in subset.columns else None), 1),
            "Avg LTV": round(weighted_avg(subset["ltv"], subset["loan_amount"] if "loan_amount" in subset.columns else None), 2),
        })

    if not scenarios:
        return pd.DataFrame(columns=["Scenario", "Loans Remaining", "Loans Removed", "Avg Buy Price", "Improvement vs Base", "Avg FICO", "Avg LTV"])

    out = pd.DataFrame(scenarios)
    out["_improve_num"] = out["Improvement vs Base"].str.replace(" bps", "", regex=False).astype(float)
    out = out.sort_values(["_improve_num", "Loans Remaining"], ascending=[False, False]).drop(columns=["_improve_num"])
    return out.reset_index(drop=True)


def build_recommendation_text(df_all: pd.DataFrame, df_filtered: pd.DataFrame, scenarios: pd.DataFrame) -> list[dict]:
    recs = []
    impact = compare_filter_impact(df_all, df_filtered)
    price_before = float(impact.loc[impact["Metric"] == "Avg Buy Price", "Before"].iloc[0])
    price_after = float(impact.loc[impact["Metric"] == "Avg Buy Price", "After"].iloc[0])
    loan_before = int(impact.loc[impact["Metric"] == "Loan Count", "Before"].iloc[0])
    loan_after = int(impact.loc[impact["Metric"] == "Loan Count", "After"].iloc[0])

    if loan_after != loan_before:
        recs.append({
            "title": "Filter impact",
            "body": f"Your current filters reduced the tape from {loan_before} loans to {loan_after}. The filtered pool moved from {price_before:.4f} to {price_after:.4f} in average buy price, a change of {(price_after - price_before) * 100:+.1f} bps.",
        })

    if not scenarios.empty:
        best = scenarios.iloc[0]
        recs.append({
            "title": "Best quick test",
            "body": f"The strongest automated scenario is <strong>{best['Scenario']}</strong>. It leaves {int(best['Loans Remaining'])} loans and changes average buy price to <strong>{best['Avg Buy Price']:.4f}</strong> ({best['Improvement vs Base']}).",
        })

    outlier_count = int((df_filtered["flag"] == "Outlier").sum())
    if outlier_count > 0:
        recs.append({
            "title": "Loans likely hurting execution",
            "body": f"There are {outlier_count} loans flagged as Outlier in the current view. These are the first loans to review for separation, exceptions, or a different bid path.",
        })

    if (df_filtered["ltv"] > 75).any():
        recs.append({
            "title": "High-LTV warning",
            "body": "Loans above 75% LTV are still present in this filtered view. If you want a cleaner execution bucket, test isolating them instead of leaving them blended with the core tape.",
        })

    if not recs:
        recs.append({
            "title": "Tape looks clean",
            "body": "The current filtered tape is already relatively strong. Use the grouping tab to compare whether keeping one pool or splitting by LTV produces the cleaner execution story.",
        })
    return recs


def build_export(df: pd.DataFrame, sa: pd.DataFrame, sb: pd.DataFrame, sc: pd.DataFrame, sd: pd.DataFrame, impact: pd.DataFrame, scenarios: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        export_cols = [c for c in [
            "loan_id", "buy_price", "fico", "ltv", "loan_amount", "product", "occupancy",
            "score", "fico_bucket", "ltv_bucket", "combined_bucket", "flag"
        ] if c in df.columns]
        df[export_cols].to_excel(writer, sheet_name="Enriched Tape", index=False)
        format_strategy_table(sa).to_excel(writer, sheet_name="Strategy A", index=False)
        format_strategy_table(sb).to_excel(writer, sheet_name="Strategy B", index=False)
        format_strategy_table(sc).to_excel(writer, sheet_name="Strategy C", index=False)
        format_strategy_table(sd).to_excel(writer, sheet_name="Strategy D", index=False)
        impact.to_excel(writer, sheet_name="Filter Impact", index=False)
        scenarios.to_excel(writer, sheet_name="Scenario Tests", index=False)
    return buf.getvalue()


def _best_match(options: list[str], candidates: list[str]) -> int:
    opts_lower = [o.lower().replace(" ", "_").replace("-", "_") for o in options]
    for cand in candidates:
        for i, opt in enumerate(opts_lower):
            if cand in opt or opt in cand:
                return i
    return 0


def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 Loan Tape Optimizer")
        st.markdown("---")
        st.markdown('<div class="sidebar-label">1 · Upload Tape</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose Excel file (.xlsx)", type=["xlsx"])

        sheet_name = None
        if uploaded:
            sheets = get_sheet_names(uploaded)
            sheet_name = st.selectbox("Sheet", sheets)

        st.markdown("---")
        st.markdown('<div class="sidebar-label">2 · Score Weights</div>', unsafe_allow_html=True)
        w_price = st.slider("Buy Price Weight", 0.0, 1.0, 0.5, 0.05)
        w_fico = st.slider("FICO Weight", 0.0, 1.0, 0.3, 0.05)
        w_ltv = st.slider("LTV Weight (lower is better)", 0.0, 1.0, 0.2, 0.05)

        st.markdown("---")
        st.markdown('<div class="sidebar-label">3 · Filters</div>', unsafe_allow_html=True)
        min_fico = st.number_input("Min FICO", value=0, min_value=0, max_value=900)
        max_ltv = st.number_input("Max LTV (%)", value=100, min_value=0, max_value=200)
        min_price = st.number_input("Min Buy Price", value=0.0, min_value=0.0, format="%.4f")
        show_only_flagged = st.checkbox("Show only Review / Outlier loans", value=False)

        return {
            "uploaded": uploaded,
            "sheet": sheet_name,
            "w_price": w_price,
            "w_fico": w_fico,
            "w_ltv": w_ltv,
            "min_fico": min_fico,
            "max_ltv": max_ltv,
            "min_price": min_price,
            "show_only_flagged": show_only_flagged,
        }


def render_column_mapping(raw_df: pd.DataFrame) -> dict:
    cols = ["(none)"] + list(raw_df.columns)
    st.markdown('<div class="section-header">Column Mapping</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    loan_id = c1.selectbox("Loan ID *", cols, index=_best_match(cols, ["loan_id", "loanid", "loan_number", "id"]))
    buy_price = c2.selectbox("Buy Price *", cols, index=_best_match(cols, ["bs_buypluslpc", "buypluslpc", "buy_price", "price"]))
    fico = c3.selectbox("FICO *", cols, index=_best_match(cols, ["fico", "credit_score", "score"]))
    ltv = c4.selectbox("LTV *", cols, index=_best_match(cols, ["ltv", "loan_to_value"]))
    c5, c6, c7 = st.columns(3)
    loan_amt = c5.selectbox("Loan Amount (opt.)", cols, index=_best_match(cols, ["loan_amount", "amount", "balance", "upb"]))
    product = c6.selectbox("Product (opt.)", cols, index=_best_match(cols, ["product", "program", "loan_type"]))
    occupancy = c7.selectbox("Occupancy (opt.)", cols, index=_best_match(cols, ["occupancy", "occ"]))
    return {
        "loan_id": None if loan_id == "(none)" else loan_id,
        "buy_price": None if buy_price == "(none)" else buy_price,
        "fico": None if fico == "(none)" else fico,
        "ltv": None if ltv == "(none)" else ltv,
        "loan_amount": None if loan_amt == "(none)" else loan_amt,
        "product": None if product == "(none)" else product,
        "occupancy": None if occupancy == "(none)" else occupancy,
    }


def render_overview(df_all: pd.DataFrame, df_filtered: pd.DataFrame, impact: pd.DataFrame, scenarios: pd.DataFrame):
    st.markdown('<div class="section-header">Current Pool Snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Loans", f"{len(df_filtered):,}")
    c2.metric("Avg Buy Price", f"{weighted_avg(df_filtered['buy_price'], df_filtered['loan_amount'] if 'loan_amount' in df_filtered.columns else None):.4f}")
    c3.metric("Avg FICO", f"{weighted_avg(df_filtered['fico'], df_filtered['loan_amount'] if 'loan_amount' in df_filtered.columns else None):.0f}")
    c4.metric("Avg LTV", f"{weighted_avg(df_filtered['ltv'], df_filtered['loan_amount'] if 'loan_amount' in df_filtered.columns else None):.2f}%")
    c5.metric("Outliers", f"{(df_filtered['flag'] == 'Outlier').sum():,}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Before vs After Filters</div>', unsafe_allow_html=True)
    st.dataframe(impact, use_container_width=True, hide_index=True)

    if not scenarios.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Best Quick Improvement Ideas</div>', unsafe_allow_html=True)
        st.dataframe(scenarios, use_container_width=True, hide_index=True)

    outliers = df_filtered[df_filtered["flag"] == "Outlier"].sort_values(["score", "ltv", "fico"], ascending=[True, False, True])
    if not outliers.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Loans Most Likely Hurting Execution</div>', unsafe_allow_html=True)
        cols = [c for c in ["loan_id", "buy_price", "fico", "ltv", "loan_amount", "score", "combined_bucket", "flag"] if c in outliers.columns]
        st.dataframe(outliers[cols].head(15), use_container_width=True, hide_index=True)


def render_loan_table(df_filtered: pd.DataFrame):
    st.markdown('<div class="section-header">Interactive Loan Table</div>', unsafe_allow_html=True)
    display_cols = [c for c in ["loan_id", "buy_price", "fico", "ltv", "loan_amount", "product", "occupancy", "score", "combined_bucket", "flag"] if c in df_filtered.columns]
    display = df_filtered[display_cols].copy()
    sort_col = st.selectbox("Sort by", list(display.columns), index=list(display.columns).index("score") if "score" in display.columns else 0)
    asc = st.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Ascending"
    st.dataframe(display.sort_values(sort_col, ascending=asc), use_container_width=True, height=520, hide_index=True)


def render_grouping(df_filtered: pd.DataFrame):
    sa = strategy_a(df_filtered)
    sb = strategy_b(df_filtered)
    sc = strategy_c(df_filtered)
    sd = strategy_d(df_filtered)

    st.markdown('<div class="section-header">Strategy A - All Loans Together</div>', unsafe_allow_html=True)
    st.dataframe(format_strategy_table(sa), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Strategy B - Split by LTV</div>', unsafe_allow_html=True)
    st.dataframe(format_strategy_table(sb), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Strategy C - Split by FICO and LTV Buckets</div>', unsafe_allow_html=True)
    st.dataframe(format_strategy_table(sc), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Strategy D - Clean Core vs Exceptions</div>', unsafe_allow_html=True)
    st.dataframe(format_strategy_table(sd), use_container_width=True, hide_index=True)
    return sa, sb, sc, sd


def render_recommendations(df_all: pd.DataFrame, df_filtered: pd.DataFrame, scenarios: pd.DataFrame):
    st.markdown('<div class="section-header">What To Do Next</div>', unsafe_allow_html=True)
    for rec in build_recommendation_text(df_all, df_filtered, scenarios):
        st.markdown(
            f'<div class="rec-box"><div class="rec-title">{rec["title"]}</div><div class="rec-body">{rec["body"]}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Bucket-Level Summary</div>', unsafe_allow_html=True)
    bucket_summary = (
        df_filtered.groupby("combined_bucket")
        .apply(summarize_group)
        .reset_index()
        .sort_values("avg_buy_price", ascending=False)
    )
    st.dataframe(format_strategy_table(bucket_summary.rename(columns={"combined_bucket": "group"})), use_container_width=True, hide_index=True)


def main():
    settings = render_sidebar()
    uploaded = settings["uploaded"]
    sheet = settings["sheet"]

    if uploaded is None:
        st.markdown("## 📊 Loan Tape Optimizer")
        st.write("Upload your bid tape on the left. This version adds filter impact analysis, automatic scenario testing, stronger exception surfacing, and a cleaner recommendation workflow.")
        return

    try:
        raw_df = load_excel(uploaded, sheet)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return

    st.markdown("### Preview - Raw Data")
    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

    col_map = render_column_mapping(raw_df)
    if not all([col_map.get("loan_id"), col_map.get("buy_price"), col_map.get("fico"), col_map.get("ltv")]):
        st.warning("Please map Loan ID, Buy Price, FICO, and LTV.")
        return

    try:
        df = clean_dataframe(raw_df, col_map)
    except Exception as e:
        st.error(f"Data cleaning error: {e}")
        return

    df["score"] = compute_scores(df, settings["w_price"], settings["w_fico"], settings["w_ltv"])
    df = assign_buckets(df)
    df = assign_flags(df)

    df_filtered = df[
        (df["fico"] >= settings["min_fico"]) &
        (df["ltv"] <= settings["max_ltv"]) &
        (df["buy_price"] >= settings["min_price"])
    ].copy()

    if settings["show_only_flagged"]:
        df_filtered = df_filtered[df_filtered["flag"].isin(["Review", "Outlier"])]

    if df_filtered.empty:
        st.warning("No loans match the current filters.")
        return

    removed = len(df) - len(df_filtered)
    if removed > 0:
        st.info(f"Filters changed the tape: {removed:,} loans removed, {len(df_filtered):,} remain.")

    impact = compare_filter_impact(df, df_filtered)
    scenarios = generate_auto_scenarios(df_filtered)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Loan Table", "Grouping Analysis", "Recommendations"])
    with tab1:
        render_overview(df, df_filtered, impact, scenarios)
    with tab2:
        render_loan_table(df_filtered)
    with tab3:
        sa, sb, sc, sd = render_grouping(df_filtered)
    with tab4:
        render_recommendations(df, df_filtered, scenarios)

    st.markdown('---')
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    sa, sb, sc, sd = strategy_a(df_filtered), strategy_b(df_filtered), strategy_c(df_filtered), strategy_d(df_filtered)
    excel_bytes = build_export(df_filtered, sa, sb, sc, sd, impact, scenarios)
    st.download_button(
        label="Download enriched tape and analysis (.xlsx)",
        data=excel_bytes,
        file_name="loan_tape_optimizer_improved.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
PY
python -m py_compile /mnt/data/loan_tape_optimizer_improved.py
