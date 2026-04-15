import io
import warnings
import uuid
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
    .main .block-container { padding-top: 1.1rem; }

    [data-testid="metric-container"] {
        background: #1a2035;
        border: 1px solid #2a3045;
        border-radius: 10px;
        padding: 0.9rem 1rem;
    }

    .section-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #4a90e2;
        margin-top: 0.3rem;
        margin-bottom: 0.6rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #2a3045;
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

    .note-box {
        background: #1a2035;
        border: 1px solid #2a3045;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #d7def5;
        margin-bottom: 0.8rem;
    }

    .rec-box {
        background: linear-gradient(135deg, #1a2e4a 0%, #1a2035 100%);
        border: 1px solid #3b82f6;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.8rem;
    }
    .rec-title { font-size: 1rem; font-weight: 700; color: #60a5ff; margin-bottom: 0.35rem; }
    .rec-body  { font-size: 0.88rem; color: #c0cce8; line-height: 1.55; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Default investor / pool templates
# -------------------------------------------------------------------
DEFAULT_POOL_TEMPLATES = [
    {
        "id": "prime_auto",
        "name": "Prime",
        "priority": 1,
        "enabled": True,
        "product_mode": "Any",
        "allowed_products": [],
        "min_fico": 740,
        "max_fico": 850,
        "max_ltv": 75.0,
        "min_pool_fico": 740.0,
        "max_pool_ltv": 75.0,
        "min_loans": 1,
        "min_avg_price": 0.0,
    },
    {
        "id": "near_prime_auto",
        "name": "Near-prime",
        "priority": 2,
        "enabled": True,
        "product_mode": "Any",
        "allowed_products": [],
        "min_fico": 640,
        "max_fico": 700,
        "max_ltv": 100.0,
        "min_pool_fico": 640.0,
        "max_pool_ltv": 100.0,
        "min_loans": 1,
        "min_avg_price": 0.0,
    },
    {
        "id": "dscr_auto",
        "name": "DSCR",
        "priority": 3,
        "enabled": True,
        "product_mode": "Contains",
        "allowed_products": ["DSCR"],
        "min_fico": 0,
        "max_fico": 850,
        "max_ltv": 100.0,
        "min_pool_fico": 0.0,
        "max_pool_ltv": 100.0,
        "min_loans": 1,
        "min_avg_price": 0.0,
    },
]


# -------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------
def get_sheet_names(file) -> list[str]:
    xl = pd.ExcelFile(file)
    return xl.sheet_names


def load_excel(file, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(file, sheet_name=sheet_name)


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[%,$,\s,]", "", regex=True),
        errors="coerce",
    )


def normalize_ltv(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if not valid.empty and valid.median() < 2:
        return series * 100
    return series


def detect_summary_rows(df: pd.DataFrame, loan_id_col: str | None) -> pd.Series:
    """
    Detect likely non-loan summary rows such as:
    Average, Avg, Total, Summary, Pool, Grand Total

    Only checks the mapped loan ID column to avoid false positives
    from numeric or mixed-type data elsewhere in the sheet.
    """
    terms = ["average", "avg", "total", "summary", "pool", "grand total"]
    mask = pd.Series(False, index=df.index, dtype=bool)

    if loan_id_col is None or loan_id_col not in df.columns:
        return mask

    loan_id_text = (
        df[loan_id_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for term in terms:
        mask = mask | loan_id_text.str.contains(term, na=False, regex=False)

    return mask


def init_row_exclusions():
    if "excluded_rows" not in st.session_state:
        st.session_state.excluded_rows = set()


def suggest_excluded_rows(raw_df: pd.DataFrame, loan_id_col: str | None = None) -> list[int]:
    terms = ["average", "avg", "total", "summary", "pool", "grand total", "count"]

    def looks_like_summary(value) -> bool:
        text = "" if pd.isna(value) else str(value).strip().lower()
        return any(term in text for term in terms)

    mask = pd.Series(False, index=raw_df.index, dtype=bool)

    if loan_id_col and loan_id_col in raw_df.columns:
        mask = mask | raw_df[loan_id_col].apply(looks_like_summary)

    return raw_df.loc[mask, "source_row"].tolist() if "source_row" in raw_df.columns else []


def render_row_exclusion_manager(raw_df: pd.DataFrame, suggested_loan_id_col: str | None = None):
    st.markdown('<div class="section-header">Row Exclusions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note-box">Exclude title rows, summary rows, average rows, totals, or any non-loan rows from the working tape. '
        'This does not delete anything from the uploaded file.</div>',
        unsafe_allow_html=True,
    )

    preview_cols = [c for c in ["source_row", suggested_loan_id_col, "buy_price", "fico", "ltv"] if c and c in raw_df.columns]
    if len(preview_cols) <= 1:
        preview_cols = ["source_row"] + [c for c in raw_df.columns if c != "source_row"][:4]

    st.dataframe(raw_df[preview_cols], use_container_width=True, height=250)

    c1, c2 = st.columns(2)
    if c1.button("Auto-suggest summary rows"):
        st.session_state.excluded_rows = set(suggest_excluded_rows(raw_df, suggested_loan_id_col))
        st.rerun()

    if c2.button("Clear row exclusions"):
        st.session_state.excluded_rows = set()
        st.rerun()

    selected_rows = st.multiselect(
        "Select source row numbers to exclude",
        options=raw_df["source_row"].tolist(),
        default=sorted(st.session_state.excluded_rows),
    )

    if st.button("Apply row exclusions"):
        st.session_state.excluded_rows = set(selected_rows)
        st.rerun()

    if st.session_state.excluded_rows:
        excluded_preview = raw_df[raw_df["source_row"].isin(st.session_state.excluded_rows)][preview_cols]
        st.markdown("**Currently excluded rows**")
        st.dataframe(excluded_preview, use_container_width=True, height=180)


def clean_dataframe(df: pd.DataFrame, col_map: dict, remove_summary_rows: bool = True) -> tuple[pd.DataFrame, int]:
    rename = {v: k for k, v in col_map.items() if v}
    working = df.rename(columns=rename).copy()

    required = ["loan_id", "buy_price", "fico", "ltv"]
    for col in required:
        if col not in working.columns:
            raise ValueError(f"Required column not mapped: {col}")

    removed_summary_count = 0
    if remove_summary_rows:
        summary_mask = detect_summary_rows(working, "loan_id")
        removed_summary_count = int(summary_mask.sum())
        working = working.loc[~summary_mask].copy()

    working["buy_price"] = clean_numeric(working["buy_price"])
    working["fico"] = clean_numeric(working["fico"])
    working["ltv"] = normalize_ltv(clean_numeric(working["ltv"]))

    if "loan_amount" in working.columns:
        working["loan_amount"] = clean_numeric(working["loan_amount"])

    before_drop = len(working)
    working = working.dropna(subset=required).reset_index(drop=True)
    removed_null_count = before_drop - len(working)

    return working, removed_summary_count + removed_null_count


def minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if pd.isna(rng) or rng == 0:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / rng


def compute_scores(df: pd.DataFrame, w_price: float, w_fico: float, w_ltv: float) -> pd.Series:
    norm_price = minmax(df["buy_price"])
    norm_fico = minmax(df["fico"])
    norm_ltv = 1 - minmax(df["ltv"])
    total_w = max(w_price + w_fico + w_ltv, 1e-9)
    score = ((w_price * norm_price) + (w_fico * norm_fico) + (w_ltv * norm_ltv)) / total_w * 100
    return score.round(2)


FICO_BINS = [0, 680, 720, 760, 9999]
FICO_LABELS = ["<680", "680-719", "720-759", "760+"]
LTV_BINS = [0, 70, 75, 80, 200]
LTV_LABELS = ["<70", "70-75", "75-80", ">80"]


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
        risk_count = int(high_ltv) + int(low_fico)
        if risk_count >= 2 or (risk_count >= 1 and low_score):
            return "Outlier"
        elif risk_count == 1 or low_score:
            return "Review"
        elif row["score"] >= score_75th:
            return "Strong"
        return "Standard"

    out["flag"] = out.apply(flag_row, axis=1)
    return out


def weighted_avg(series: pd.Series, weights: pd.Series | None) -> float:
    valid = series.notna()
    if weights is None:
        return float(series[valid].mean()) if valid.any() else np.nan
    w = weights[valid]
    x = series[valid]
    if len(x) == 0:
        return np.nan
    if w.isna().all() or w.sum() == 0:
        return float(x.mean())
    return float(np.average(x, weights=w))


def pool_metrics(df: pd.DataFrame) -> dict:
    weights = df["loan_amount"] if "loan_amount" in df.columns else None
    return {
        "loan_count": len(df),
        "avg_buy_price": float(df["buy_price"].mean()) if len(df) else np.nan,
        "pool_fico": weighted_avg(df["fico"], weights),
        "pool_ltv": weighted_avg(df["ltv"], weights),
        "avg_score": float(df["score"].mean()) if len(df) else np.nan,
        "pct_high_ltv": float((df["ltv"] > 75).mean() * 100) if len(df) else 0.0,
        "pct_low_fico": float((df["fico"] < 700).mean() * 100) if len(df) else 0.0,
    }


def format_bps(delta: float) -> str:
    if pd.isna(delta):
        return "-"
    bps = round(delta * 100, 1)
    sign = "+" if bps > 0 else ""
    return f"{sign}{bps} bps"


# -------------------------------------------------------------------
# Pool template / investor helpers
# -------------------------------------------------------------------
def init_templates():
    if "pool_templates" not in st.session_state:
        st.session_state.pool_templates = [dict(t) for t in DEFAULT_POOL_TEMPLATES]


def get_product_options(df: pd.DataFrame, product_col_exists: bool) -> list[str]:
    if not product_col_exists:
        return []
    vals = (
        df["product"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(vals)


def best_match(options: list[str], candidates: list[str]) -> int:
    opts_lower = [str(o).lower().replace(" ", "_").replace("-", "_") for o in options]
    for cand in candidates:
        for i, opt in enumerate(opts_lower):
            if cand in opt or opt in cand:
                return i
    return 0


def product_match(series: pd.Series, template: dict) -> pd.Series:
    mode = template.get("product_mode", "Any")
    values = template.get("allowed_products", []) or []

    if mode == "Any" or "product" not in series.index.names:
        pass

    if mode == "Any":
        return pd.Series(True, index=series.index)
    if len(values) == 0:
        return pd.Series(True, index=series.index)

    text = series.astype(str).str.upper()

    if mode == "Exact":
        targets = {str(v).upper() for v in values}
        return text.isin(targets)

    # Contains
    mask = pd.Series(False, index=series.index)
    for v in values:
        mask = mask | text.str.contains(str(v).upper(), na=False, regex=False)
    return mask


def apply_template_loan_rules(df: pd.DataFrame, template: dict) -> pd.DataFrame:
    eligible = df.copy()
    eligible = eligible[(eligible["fico"] >= template["min_fico"]) & (eligible["fico"] <= template["max_fico"])]
    eligible = eligible[eligible["ltv"] <= template["max_ltv"]]

    if "product" in eligible.columns:
        pmask = product_match(eligible["product"], template)
        eligible = eligible[pmask]
    elif template.get("product_mode") != "Any":
        eligible = eligible.iloc[0:0]

    if template.get("min_avg_price", 0.0) > 0:
        eligible = eligible[eligible["buy_price"] >= template["min_avg_price"]]

    return eligible.copy()


def template_pool_valid(df: pd.DataFrame, template: dict) -> tuple[bool, dict]:
    metrics = pool_metrics(df)
    valid = True

    if metrics["loan_count"] < template["min_loans"]:
        valid = False
    if metrics["pool_fico"] < template["min_pool_fico"]:
        valid = False
    if metrics["pool_ltv"] > template["max_pool_ltv"]:
        valid = False
    if metrics["avg_buy_price"] < template.get("min_avg_price", 0.0):
        valid = False

    return valid, metrics


def assign_loans_to_templates(df: pd.DataFrame, templates: list[dict]) -> pd.DataFrame:
    remaining = df.copy()
    assigned_frames = []

    ordered = sorted([t for t in templates if t.get("enabled", True)], key=lambda x: x.get("priority", 999))

    for template in ordered:
        eligible = apply_template_loan_rules(remaining, template)
        if eligible.empty:
            continue

        valid, _ = template_pool_valid(eligible, template)
        if not valid:
            continue

        chunk = eligible.copy()
        chunk["assigned_pool"] = template["name"]
        chunk["assigned_pool_id"] = template["id"]
        assigned_frames.append(chunk)
        remaining = remaining.loc[~remaining.index.isin(chunk.index)].copy()

    if assigned_frames:
        assigned = pd.concat(assigned_frames, axis=0)
    else:
        assigned = df.iloc[0:0].copy()

    unassigned = remaining.copy()
    unassigned["assigned_pool"] = "Unassigned"
    unassigned["assigned_pool_id"] = "unassigned"

    final = pd.concat([assigned, unassigned], axis=0).sort_index().copy()
    return final


def summarize_assigned_pools(df: pd.DataFrame, templates: list[dict]) -> pd.DataFrame:
    rows = []
    template_map = {t["id"]: t for t in templates}
    for pool_id, g in df.groupby("assigned_pool_id", dropna=False):
        pool_name = g["assigned_pool"].iloc[0]
        template = template_map.get(pool_id)
        metrics = pool_metrics(g)
        valid = True if pool_id == "unassigned" else template_pool_valid(g, template)[0]
        rows.append({
            "Pool": pool_name,
            "Loans": metrics["loan_count"],
            "Avg Buy Price": round(metrics["avg_buy_price"], 4),
            "Pool FICO": round(metrics["pool_fico"], 0),
            "Pool LTV": round(metrics["pool_ltv"], 2),
            "% High LTV": round(metrics["pct_high_ltv"], 1),
            "% Low FICO": round(metrics["pct_low_fico"], 1),
            "Avg Score": round(metrics["avg_score"], 1),
            "Valid": "Yes" if valid else "No",
        })
    if not rows:
        return pd.DataFrame(columns=["Pool", "Loans", "Avg Buy Price", "Pool FICO", "Pool LTV", "% High LTV", "% Low FICO", "Avg Score", "Valid"])
    out = pd.DataFrame(rows)
    return out.sort_values(["Pool"], ascending=True).reset_index(drop=True)


def run_analysis_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = []
    base_metrics = pool_metrics(df)

    scenario_defs = [
        ("Current Selection", df.copy()),
        ("Prime-style slice", df[(df["fico"] >= 740) & (df["ltv"] <= 75)].copy()),
        ("Near-prime slice", df[(df["fico"] >= 640) & (df["fico"] <= 700)].copy()),
    ]

    if "product" in df.columns:
        dscr_mask = df["product"].astype(str).str.upper().str.contains("DSCR", na=False, regex=False)
        scenario_defs.append(("DSCR slice", df[dscr_mask].copy()))

    scenario_defs.extend([
        ("Exclude LTV > 75%", df[df["ltv"] <= 75].copy()),
        ("Exclude FICO < 700", df[df["fico"] >= 700].copy()),
        ("Exclude LTV > 75% and FICO < 700", df[(df["ltv"] <= 75) & (df["fico"] >= 700)].copy()),
    ])

    for name, sdf in scenario_defs:
        if sdf.empty:
            continue
        m = pool_metrics(sdf)
        scenarios.append({
            "Scenario": name,
            "Loans": m["loan_count"],
            "Avg Buy Price": round(m["avg_buy_price"], 4),
            "Pool FICO": round(m["pool_fico"], 0),
            "Pool LTV": round(m["pool_ltv"], 2),
            "Avg Score": round(m["avg_score"], 1),
            "Price Change": format_bps(m["avg_buy_price"] - base_metrics["avg_buy_price"]),
            "FICO Change": round(m["pool_fico"] - base_metrics["pool_fico"], 0),
            "LTV Change": round(m["pool_ltv"] - base_metrics["pool_ltv"], 2),
            "Loans Removed": base_metrics["loan_count"] - m["loan_count"],
            "_price_delta_raw": m["avg_buy_price"] - base_metrics["avg_buy_price"],
        })

    out = pd.DataFrame(scenarios)
    if not out.empty:
        out = out.sort_values(["_price_delta_raw", "Pool FICO"], ascending=[False, False]).reset_index(drop=True)
    return out


# -------------------------------------------------------------------
# UI rendering
# -------------------------------------------------------------------
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
        st.markdown('<div class="sidebar-label">3 · Tape Cleaning + Base Filters</div>', unsafe_allow_html=True)
        remove_summary_rows = st.checkbox("Ignore summary / average rows", value=True)
        min_fico = st.number_input("Base Min FICO", min_value=0, max_value=900, value=0)
        max_ltv = st.number_input("Base Max LTV (%)", min_value=0, max_value=200, value=100)
        min_price = st.number_input("Base Min Buy Price", min_value=0.0, value=0.0, format="%.4f")

        st.markdown("---")
        st.markdown('<div class="sidebar-label">4 · Analysis Mode</div>', unsafe_allow_html=True)
        analysis_mode = st.toggle("Enable analysis mode", value=False)
        st.caption("Analysis mode compares alternate slices without changing final one-pool assignment logic.")

    return {
        "uploaded": uploaded,
        "sheet": sheet_name,
        "w_price": w_price,
        "w_fico": w_fico,
        "w_ltv": w_ltv,
        "remove_summary_rows": remove_summary_rows,
        "min_fico": min_fico,
        "max_ltv": max_ltv,
        "min_price": min_price,
        "analysis_mode": analysis_mode,
    }


def render_column_mapping(raw_df: pd.DataFrame) -> dict:
    cols = ["(none)"] + list(raw_df.columns)

    st.markdown('<div class="section-header">Column Mapping</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    loan_id = c1.selectbox("Loan ID *", cols, index=best_match(cols, ["loan_id", "loanid", "loan", "id", "loan_number"]))
    buy_price = c2.selectbox("Buy Price *", cols, index=best_match(cols, ["bs_buypluslpc", "buypluslpc", "buy_price", "price"]))
    fico = c3.selectbox("FICO *", cols, index=best_match(cols, ["fico", "credit_score", "score"]))
    ltv = c4.selectbox("LTV *", cols, index=best_match(cols, ["ltv", "loan_to_value"]))

    c5, c6, c7 = st.columns(3)
    loan_amount = c5.selectbox("Loan Amount (opt.)", cols, index=best_match(cols, ["loan_amount", "amount", "balance", "upb"]))
    product = c6.selectbox("Product (opt.)", cols, index=best_match(cols, ["product", "program", "loan_type"]))
    occupancy = c7.selectbox("Occupancy (opt.)", cols, index=best_match(cols, ["occupancy", "occ"]))

    def clean_choice(v):
        return None if v == "(none)" else v

    return {
        "loan_id": clean_choice(loan_id),
        "buy_price": clean_choice(buy_price),
        "fico": clean_choice(fico),
        "ltv": clean_choice(ltv),
        "loan_amount": clean_choice(loan_amount),
        "product": clean_choice(product),
        "occupancy": clean_choice(occupancy),
    }


def render_pool_summary(current_df: pd.DataFrame, original_df: pd.DataFrame):
    current = pool_metrics(current_df)
    original = pool_metrics(original_df)

    st.markdown('<div class="section-header">Current Tape / Pool Row</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Loans", f"{current['loan_count']:,}", delta=f"{current['loan_count'] - original['loan_count']:+d} vs cleaned tape")
    c2.metric("Avg Buy Price", f"{current['avg_buy_price']:.4f}", delta=format_bps(current["avg_buy_price"] - original["avg_buy_price"]))
    c3.metric("Pool FICO", f"{current['pool_fico']:.0f}", delta=f"{current['pool_fico'] - original['pool_fico']:+.0f}")
    c4.metric("Pool LTV", f"{current['pool_ltv']:.2f}%", delta=f"{current['pool_ltv'] - original['pool_ltv']:+.2f}%")
    c5.metric("Avg Score", f"{current['avg_score']:.1f}")

    st.markdown(
        '<div class="note-box">This is the live pool-style row for the current working tape. '
        'Use it to judge whether the current selection hits target metrics like pool FICO above 740 or pool LTV below 75%.</div>',
        unsafe_allow_html=True,
    )


def render_pool_template_editor(product_options: list[str], has_product_col: bool):
    st.markdown('<div class="section-header">Investor / Pool Templates</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note-box">Loans are assigned to only one pool, based on template priority. '
        'Default templates are Prime, Near-prime, and DSCR. You can edit them, delete them, or add new ones.</div>',
        unsafe_allow_html=True,
    )

    if st.button("Reset templates to defaults"):
        st.session_state.pool_templates = [dict(t) for t in DEFAULT_POOL_TEMPLATES]
        st.rerun()

    if st.button("Add new pool template"):
        st.session_state.pool_templates.append({
            "id": f"custom_{uuid.uuid4().hex[:8]}",
            "name": f"Custom Pool {len(st.session_state.pool_templates) + 1}",
            "priority": len(st.session_state.pool_templates) + 1,
            "enabled": True,
            "product_mode": "Any",
            "allowed_products": [],
            "min_fico": 0,
            "max_fico": 850,
            "max_ltv": 100.0,
            "min_pool_fico": 0.0,
            "max_pool_ltv": 100.0,
            "min_loans": 1,
            "min_avg_price": 0.0,
        })
        st.rerun()

    updated_templates = []
    delete_ids = set()

    for idx, template in enumerate(sorted(st.session_state.pool_templates, key=lambda x: x.get("priority", 999))):
        with st.expander(f"{template['name']}  |  Priority {template['priority']}", expanded=False):
            c1, c2, c3 = st.columns([2, 1, 1])
            name = c1.text_input("Pool name", value=template["name"], key=f"name_{template['id']}")
            priority = c2.number_input("Priority", min_value=1, value=int(template["priority"]), step=1, key=f"priority_{template['id']}")
            enabled = c3.checkbox("Enabled", value=bool(template["enabled"]), key=f"enabled_{template['id']}")

            c4, c5, c6 = st.columns(3)
            min_fico = c4.number_input("Loan-level Min FICO", min_value=0, max_value=900, value=int(template["min_fico"]), key=f"min_fico_{template['id']}")
            max_fico = c5.number_input("Loan-level Max FICO", min_value=0, max_value=900, value=int(template["max_fico"]), key=f"max_fico_{template['id']}")
            max_ltv = c6.number_input("Loan-level Max LTV", min_value=0.0, max_value=200.0, value=float(template["max_ltv"]), step=0.5, key=f"max_ltv_{template['id']}")

            c7, c8, c9 = st.columns(3)
            min_pool_fico = c7.number_input("Pool-level Min Avg FICO", min_value=0.0, max_value=900.0, value=float(template["min_pool_fico"]), step=1.0, key=f"min_pool_fico_{template['id']}")
            max_pool_ltv = c8.number_input("Pool-level Max Avg LTV", min_value=0.0, max_value=200.0, value=float(template["max_pool_ltv"]), step=0.5, key=f"max_pool_ltv_{template['id']}")
            min_loans = c9.number_input("Min loans in pool", min_value=1, value=int(template["min_loans"]), step=1, key=f"min_loans_{template['id']}")

            c10, c11 = st.columns(2)
            min_avg_price = c10.number_input("Min Avg Buy Price", min_value=0.0, value=float(template.get("min_avg_price", 0.0)), step=0.01, key=f"min_avg_price_{template['id']}")
            product_mode = c11.selectbox(
                "Product rule",
                ["Any", "Contains", "Exact"],
                index=["Any", "Contains", "Exact"].index(template.get("product_mode", "Any")),
                key=f"product_mode_{template['id']}",
                disabled=not has_product_col,
            )

            allowed_products = template.get("allowed_products", [])
            if has_product_col:
                allowed_products = st.multiselect(
                    "Allowed products",
                    options=product_options,
                    default=[p for p in allowed_products if p in product_options],
                    key=f"allowed_products_{template['id']}",
                )
            else:
                st.caption("No product column mapped, so product rules are disabled.")
                allowed_products = []

            delete_clicked = st.button("Delete this pool", key=f"delete_{template['id']}")

            if delete_clicked:
                delete_ids.add(template["id"])
            else:
                updated_templates.append({
                    "id": template["id"],
                    "name": name.strip() or template["name"],
                    "priority": int(priority),
                    "enabled": bool(enabled),
                    "product_mode": product_mode if has_product_col else "Any",
                    "allowed_products": allowed_products if has_product_col else [],
                    "min_fico": int(min_fico),
                    "max_fico": int(max_fico),
                    "max_ltv": float(max_ltv),
                    "min_pool_fico": float(min_pool_fico),
                    "max_pool_ltv": float(max_pool_ltv),
                    "min_loans": int(min_loans),
                    "min_avg_price": float(min_avg_price),
                })

    if delete_ids:
        updated_templates = [t for t in updated_templates if t["id"] not in delete_ids]

    st.session_state.pool_templates = sorted(updated_templates, key=lambda x: x["priority"])


def render_assignment_summary(assigned_df: pd.DataFrame, templates: list[dict]):
    st.markdown('<div class="section-header">Auto-Generated Pool Assignments</div>', unsafe_allow_html=True)
    summary = summarize_assigned_pools(assigned_df, templates)
    st.dataframe(summary, use_container_width=True)

    st.markdown('<div class="section-header">Assigned Loan Detail</div>', unsafe_allow_html=True)
    display_cols = [c for c in [
        "loan_id", "assigned_pool", "buy_price", "fico", "ltv", "loan_amount", "product", "score", "flag", "combined_bucket"
    ] if c in assigned_df.columns]
    show = assigned_df[display_cols].copy().rename(columns={
        "loan_id": "Loan ID",
        "assigned_pool": "Assigned Pool",
        "buy_price": "Buy Price",
        "fico": "FICO",
        "ltv": "LTV",
        "loan_amount": "Loan Amount",
        "product": "Product",
        "score": "Score",
        "flag": "Flag",
        "combined_bucket": "Bucket",
    })
    st.dataframe(show.sort_values(["Assigned Pool", "Buy Price"], ascending=[True, False]), use_container_width=True, height=450)


def render_analysis_mode(df: pd.DataFrame, analysis_mode: bool):
    if not analysis_mode:
        st.info("Analysis mode is off. Turn it on in the sidebar to compare alternate scenarios without changing final loan assignments.")
        return

    st.markdown('<div class="section-header">Analysis Mode: Alternate Scenario Pricing</div>', unsafe_allow_html=True)
    scenarios = run_analysis_scenarios(df)
    if scenarios.empty:
        st.warning("No scenario results available.")
        return

    st.dataframe(scenarios.drop(columns=["_price_delta_raw"]), use_container_width=True)

    non_base = scenarios[scenarios["Scenario"] != "Current Selection"].copy()
    if not non_base.empty:
        best = non_base.sort_values(["_price_delta_raw", "Pool FICO"], ascending=[False, False]).iloc[0]
        st.markdown(
            f'<div class="rec-box"><div class="rec-title">Best analysis slice</div>'
            f'<div class="rec-body">{best["Scenario"]} currently gives the best price change at '
            f'<strong>{best["Price Change"]}</strong>, with pool FICO <strong>{int(best["Pool FICO"])}</strong> '
            f'and pool LTV <strong>{best["Pool LTV"]:.2f}%</strong>.</div></div>',
            unsafe_allow_html=True,
        )


def render_recommendations(assigned_df: pd.DataFrame):
    st.markdown('<div class="section-header">Recommendations</div>', unsafe_allow_html=True)

    unassigned = assigned_df[assigned_df["assigned_pool"] == "Unassigned"]
    assigned = assigned_df[assigned_df["assigned_pool"] != "Unassigned"]

    assigned_metrics = pool_metrics(assigned) if not assigned.empty else None

    cards = []
    if assigned_metrics is not None:
        cards.append({
            "title": "Current assigned production view",
            "body": (
                f"Assigned pools currently contain <strong>{assigned_metrics['loan_count']}</strong> loans with "
                f"average buy price <strong>{assigned_metrics['avg_buy_price']:.4f}</strong>, pool FICO "
                f"<strong>{assigned_metrics['pool_fico']:.0f}</strong>, and pool LTV "
                f"<strong>{assigned_metrics['pool_ltv']:.2f}%</strong>."
            )
        })

    if not unassigned.empty:
        um = pool_metrics(unassigned)
        cards.append({
            "title": "Unassigned opportunity set",
            "body": (
                f"There are <strong>{um['loan_count']}</strong> unassigned loans still available to target with new investor rules. "
                f"Their pool FICO is <strong>{um['pool_fico']:.0f}</strong> and pool LTV is <strong>{um['pool_ltv']:.2f}%</strong>. "
                "This is where additional custom investor templates may unlock more execution."
            )
        })

    if "product" in assigned_df.columns:
        dscr_unassigned = assigned_df[
            (assigned_df["assigned_pool"] == "Unassigned")
            & (assigned_df["product"].astype(str).str.upper().str.contains("DSCR", na=False, regex=False))
        ]
        if not dscr_unassigned.empty:
            cards.append({
                "title": "DSCR template check",
                "body": (
                    f"You still have <strong>{len(dscr_unassigned)}</strong> DSCR-style loans unassigned. "
                    "Consider widening the DSCR template or adding a second DSCR investor profile."
                )
            })

    for card in cards:
        st.markdown(
            f'<div class="rec-box"><div class="rec-title">{card["title"]}</div><div class="rec-body">{card["body"]}</div></div>',
            unsafe_allow_html=True,
        )


def build_export(assigned_df: pd.DataFrame, templates: list[dict], analysis_mode: bool) -> bytes:
    buf = io.BytesIO()

    pool_summary = summarize_assigned_pools(assigned_df, templates)
    templates_df = pd.DataFrame(templates)
    scenario_df = run_analysis_scenarios(assigned_df).drop(columns=["_price_delta_raw"], errors="ignore") if analysis_mode else pd.DataFrame()

    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        export_cols = [c for c in [
            "source_row", "loan_id", "assigned_pool", "buy_price", "fico", "ltv", "loan_amount",
            "product", "occupancy", "score", "fico_bucket", "ltv_bucket",
            "combined_bucket", "flag"
        ] if c in assigned_df.columns]
        assigned_df[export_cols].to_excel(writer, sheet_name="Assigned Loans", index=False)
        pool_summary.to_excel(writer, sheet_name="Pool Summary", index=False)
        templates_df.to_excel(writer, sheet_name="Pool Templates", index=False)
        if not scenario_df.empty:
            scenario_df.to_excel(writer, sheet_name="Analysis Scenarios", index=False)

    return buf.getvalue()


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    init_templates()
    init_row_exclusions()
    settings = render_sidebar()

    if settings["uploaded"] is None:
        st.markdown("## 📊 Loan Tape Optimizer")
        st.markdown(
            "This version is built around pool construction, not hiding loans. "
            "Use it to turn a tape into investor-targeted pools with single-loan assignment and optional analysis mode."
        )
        st.markdown(
            "- Upload your bid tape\n"
            "- Map loan ID, buy price, FICO, and LTV\n"
            "- Ignore any average / summary rows\n"
            "- Start with Prime, Near-prime, and DSCR templates\n"
            "- Customize pool templates and priorities\n"
            "- Assign each loan to one pool\n"
            "- Toggle analysis mode to compare alternate slices"
        )
        st.info("Use the sidebar to upload your file.")
        return

    try:
        raw_df = load_excel(settings["uploaded"], settings["sheet"]).copy()
        raw_df.insert(0, "source_row", range(1, len(raw_df) + 1))
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return

    st.markdown("### Raw Data Preview")
    st.dataframe(raw_df.head(10), use_container_width=True)

    render_row_exclusion_manager(raw_df)

    col_map = render_column_mapping(raw_df)
    required = ["loan_id", "buy_price", "fico", "ltv"]
    if not all(col_map.get(k) for k in required):
        st.warning("Please map Loan ID, Buy Price, FICO, and LTV.")
        return

    working_raw_df = raw_df[~raw_df["source_row"].isin(st.session_state.excluded_rows)].copy()

    try:
        cleaned_df, removed_count = clean_dataframe(
            working_raw_df,
            col_map,
            remove_summary_rows=settings["remove_summary_rows"],
        )
    except Exception as e:
        st.error(f"Data cleaning error: {e}")
        return

    cleaned_df["score"] = compute_scores(cleaned_df, settings["w_price"], settings["w_fico"], settings["w_ltv"])
    cleaned_df = assign_buckets(cleaned_df)
    cleaned_df = assign_flags(cleaned_df)

    filtered_df = cleaned_df[
        (cleaned_df["fico"] >= settings["min_fico"])
        & (cleaned_df["ltv"] <= settings["max_ltv"])
        & (cleaned_df["buy_price"] >= settings["min_price"])
    ].copy()

    manual_excluded_count = len(st.session_state.excluded_rows)
    if manual_excluded_count > 0:
        st.info(f"Manually excluded {manual_excluded_count:,} source rows before cleaning.")

    if removed_count > 0:
        st.info(f"Removed {removed_count:,} non-loan or incomplete rows during cleaning.")

    excluded = len(cleaned_df) - len(filtered_df)
    if excluded > 0:
        st.info(f"Base filters excluded {excluded:,} loans. Current working tape has {len(filtered_df):,} loans.")

    if filtered_df.empty:
        st.warning("No loans remain after the current base filters.")
        return

    render_pool_summary(filtered_df, cleaned_df)

    product_options = get_product_options(filtered_df, "product" in filtered_df.columns)
    render_pool_template_editor(product_options, "product" in filtered_df.columns)

    assigned_df = assign_loans_to_templates(filtered_df, st.session_state.pool_templates)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Auto Pooling",
        "Pool Detail",
        "Analysis Mode",
        "Recommendations",
    ])

    with tab1:
        render_assignment_summary(assigned_df, st.session_state.pool_templates)

    with tab2:
        st.markdown('<div class="section-header">Pool-Level Views</div>', unsafe_allow_html=True)
        for pool_name, g in assigned_df.groupby("assigned_pool", dropna=False):
            with st.expander(f"{pool_name} | {len(g)} loans", expanded=False):
                metrics = pool_metrics(g)
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Buy Price", f"{metrics['avg_buy_price']:.4f}")
                c2.metric("Pool FICO", f"{metrics['pool_fico']:.0f}")
                c3.metric("Pool LTV", f"{metrics['pool_ltv']:.2f}%")

                cols = [c for c in ["loan_id", "buy_price", "fico", "ltv", "loan_amount", "product", "score", "flag"] if c in g.columns]
                st.dataframe(g[cols].sort_values("buy_price", ascending=False), use_container_width=True, height=250)

    with tab3:
        render_analysis_mode(filtered_df, settings["analysis_mode"])

    with tab4:
        render_recommendations(assigned_df)

    st.markdown("---")
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    st.download_button(
        label="Download Assigned Pools and Scenario Summary",
        data=build_export(assigned_df, st.session_state.pool_templates, settings["analysis_mode"]),
        file_name="loan_tape_optimizer_investor_pooling.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
