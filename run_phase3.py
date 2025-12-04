# run_phase3.py
# Phase 3 – Household Stress Simulation
# Uses Annex-I (groups), Annex-III (state CPI), Annex-VII (All-India YoY)
# -----------------------------------------------------------------------------------------------

import os, re, warnings, zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------ Paths ------------------------------
ROOT  = os.path.abspath(os.path.dirname(__file__))
DATAI = os.path.join(ROOT, "data_interim")
PLOTS = os.path.join(DATAI, "plots")

IN_HH       = os.path.join(DATAI, "households_features.csv")
IN_ANNEX    = os.path.join(DATAI, "annex_priors.csv")
IN_MACRO    = os.path.join(DATAI, "macro_snapshot_t0.csv")
IN_RATES_X  = os.path.join(DATAI, "Structure of Interest Rates.xlsx")
IN_ANNEXZIP = os.path.join(DATAI, "annex.zip")

OUT_DETAILED = os.path.join(DATAI, "phase3_household_results.csv")
OUT_SUMMARY  = os.path.join(DATAI, "phase3_scenario_summary.csv")
OUT_SHOCKS   = os.path.join(DATAI, "phase3_scenario_shocks.csv")

os.makedirs(DATAI, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

# ------------------------------ Config / Scenarios ------------------------------
SCENARIOS = {
    "Base": {
        "cpi": 0.00,  # YoY overlay to add on top of Annex-VII YoY (0 => just Annex)
        "cat_elasticity": {"housing":1.0,"food":1.0,"health":1.0,"education":1.0,"transport":1.0,"other":1.0},
        "equity_ret": 0.00,
        "gold_ret": 0.00,
        "rate_bps": 0,
        "rebalance_to_target": False,
    },
    "Moderate": {
        "cpi": 0.03,  # +3% YoY overlay
        "cat_elasticity": {"housing":0.7,"food":1.2,"health":1.1,"education":0.8,"transport":1.0,"other":0.9},
        "equity_ret": -0.08,
        "gold_ret": 0.05,
        "rate_bps": 50,
        "rebalance_to_target": False,
    },
    "Severe": {
        "cpi": 0.06,  # +6% YoY overlay
        "cat_elasticity": {"housing":0.6,"food":1.3,"health":1.2,"education":0.7,"transport":1.1,"other":0.8},
        "equity_ret": -0.15,
        "gold_ret": 0.10,
        "rate_bps": 125,
        "rebalance_to_target": True,
    },
}

# functional categories used in the HH dataset
CATS = ["housing","food","health","education","transport","other"]

# simple income shock rules you were already using
INC_SCEN_BASE = {"Base": 0.00, "Moderate": -0.02, "Severe": -0.05}
INC_COHORT    = {
    "salaried":      {"low": -0.01, "moderate": -0.02, "high": -0.04},
    "self_employed": {"low": -0.03, "moderate": -0.06, "high": -0.10},
}

# CPI group → functional category
GROUP_MAP = {
    "food and beverages": "food",
    "housing": "housing",
    # health / education / transport sit inside Misc; we’ll map them to "other"
    "miscellaneous": "other",
}

MONTH_MAP = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
}

# Debt duration (years) for MTM
DURATION_DEBT = 3.0

# ------------------------------ Helpers ------------------------------
def nz(x, fallback=0.0):
    try:
        return fallback if pd.isna(x) else float(x)
    except Exception:
        return fallback

def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=(b!=0))

def norm_state(x):
    return str(x).strip().replace("&", "AND").replace(".", "").replace("-", " ").upper()

def row_income_shock(row, scenario):
    et = str(row.get("employment_type","")).strip().lower()
    rt = str(row.get("risk_tolerance","")).strip().lower()
    return INC_SCEN_BASE.get(scenario, 0.0) + INC_COHORT.get(et, {}).get(rt, 0.0)

def classify_risk(total_burden, buffer_months):
    tb = nz(total_burden)
    br = buffer_months if pd.notna(buffer_months) else np.inf
    if (tb > 1.0) or ((br < 1.0) and (tb > 0.9)):
        return "Distressed"
    if (0.8 < tb <= 1.0) or (1.0 <= br < 2.0):
        return "At-Risk"
    return "Stable"

def emi_amount(P, r_m, N):
    if r_m == 0:
        return P / max(N,1)
    x = (1 + r_m)**N
    return P * r_m * x / (x - 1)

def infer_emi_uplift_from_rate_change(df, delta_bps, assumed_tenor_months=180, r0_annual=0.085):
    r1_annual = r0_annual + (delta_bps / 10000.0)
    r0_m = r0_annual / 12.0
    r1_m = r1_annual / 12.0

    emi_now = df["emi_total"].astype(float)
    x0 = (1 + r0_m)**assumed_tenor_months
    denom = (r0_m * x0)
    P_eff = np.where(denom != 0, emi_now * (x0 - 1) / denom, 0.0)
    emi_new = np.where(P_eff > 0, emi_amount(P_eff, r1_m, assumed_tenor_months), emi_now)
    uplift = safe_div(emi_new - emi_now, emi_now)
    return np.clip(uplift, -0.5, 1.0)

def load_macro_snapshot(path):
    if not os.path.exists(path):
        return {}
    mac = pd.read_csv(path)
    return mac.iloc[0].to_dict() if not mac.empty else {}

def condition_equity_shock(base_equity_ret, macro):
    if not macro:
        return base_equity_ret
    vol = float(macro.get("nifty_vol_36m_t0", 0.04))
    dd  = float(macro.get("nifty_drawdown_36m_t0", -0.25))  # negative
    scale = (1.0 + 0.50*vol) * (1.0 - 0.50*dd)
    return base_equity_ret * scale

def equity_scale_only(macro):
    if not macro:
        return 1.0
    vol = float(macro.get("nifty_vol_36m_t0", 0.04))
    dd  = float(macro.get("nifty_drawdown_36m_t0", -0.25))
    return (1.0 + 0.50*vol) * (1.0 - 0.50*dd)

def read_rate_delta_bps_from_xlsx(path, scenario_name, fallback_bps):
    if not os.path.exists(path):
        return fallback_bps, "default_map"
    try:
        xl = pd.ExcelFile(path)
        df0 = xl.parse(xl.sheet_names[0])
        found = any(any(k in str(c).lower() for k in ["mclr","home","lending","rate","repo"]) for c in df0.columns)
        if not found:
            return fallback_bps, "default_map"
        base_map = {"Base":0, "Moderate":50, "Severe":125}
        return int(base_map.get(scenario_name, fallback_bps)), "xlsx_heuristic"
    except Exception:
        return fallback_bps, "default_map"

def zscore(s):
    s = pd.Series(s, dtype=float)
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def robust_quintiles(series):
    s = pd.Series(series, dtype=float)
    if s.dropna().nunique() < 2:
        return pd.Series([3]*len(s), dtype="Int64")
    try:
        q = pd.qcut(s.rank(method="first"), 5, labels=[1,2,3,4,5], duplicates="drop").astype("Int64")
        if q.isna().any():
            rk = (s.rank(method="average")/len(s))*5
            qfill = rk.clip(1,5).round().astype("Int64")
            q = q.fillna(qfill)
        return q
    except Exception:
        rk = (s.rank(method="average")/len(s))*5
        return rk.clip(1,5).round().astype("Int64")

def extract_month_year_from_filename(name: str):
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(\d{4})",
                  name, flags=re.I)
    if not m:
        return None
    mon_txt = m.group(1).lower()
    year = int(m.group(2))
    mon = MONTH_MAP.get(mon_txt[:3], 1)
    return datetime(year, mon, 1).date()

# ------------------------------ Annex parsers ------------------------------
def parse_annexI_category_mom_from_bytes(xlsx_bytes: bytes):
    """
    Annex-I: All-India group-level CPI (Rural/Urban/Combined).
    Returns dict {group_desc_lower: mom_decimal} for Combined.
    """
    xl = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    if "Annex-I" not in xl.sheet_names:
        return {}
    raw = xl.parse("Annex-I", header=None)

    hdr_row = None
    for i in range(20):
        if any(str(x).strip().upper() == "GROUP CODE" for x in raw.iloc[i].tolist()):
            hdr_row = i
            break
    if hdr_row is None:
        return {}

    h1 = raw.iloc[hdr_row].fillna("")
    h2 = raw.iloc[hdr_row+1].fillna("")
    # forward-fill group headers (Rural/Urban/Combined)
    h1_ff = h1.replace("", np.nan).fillna(method="ffill")

    cols = []
    for a,b in zip(h1_ff, h2):
        a = str(a).strip()
        b = str(b).strip()
        if a and b:
            cols.append(f"{a}__{b}")
        elif a:
            cols.append(a)
        elif b:
            cols.append(b)
        else:
            cols.append("")
    df = raw.iloc[hdr_row+2:].copy()
    df.columns = cols

    subcol = "Sub-group Code"
    gcol   = "Group Code"
    if gcol not in df.columns:
        return {}

    # keep only group-level rows (no subgroup code)
    if subcol in df.columns:
        df = df[df[subcol].isna()].copy()

    comb_idx = [c for c in df.columns if c.lower().startswith("combined__") and "index" in c.lower()]
    comb_idx = sorted(comb_idx)
    if len(comb_idx) < 2:
        return {}

    prev_c, curr_c = comb_idx[-2], comb_idx[-1]
    prev = pd.to_numeric(df[prev_c], errors="coerce")
    curr = pd.to_numeric(df[curr_c], errors="coerce")
    mom = safe_div(curr, prev) - 1.0

    desc_col = "Description"
    if desc_col not in df.columns:
        for c in df.columns:
            if "description" in str(c).lower():
                desc_col = c
                break
        else:
            return {}

    desc = df[desc_col].astype(str).str.strip().str.lower()
    return dict(zip(desc, mom))

def parse_annexIII_state_mom_from_bytes(xlsx_bytes: bytes):
    """
    Annex-III: CPI for States – we use Combined July vs current month index
    to compute MoM for each state.
    Returns DataFrame [state, mom].
    """
    xl = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    if "Annex-III" not in xl.sheet_names:
        return pd.DataFrame(columns=["state","mom"])
    raw = xl.parse("Annex-III", header=None)

    hdr_row = None
    for i in range(30):
        if any("name of the state/ut" in str(x).strip().lower()
               for x in raw.iloc[i].tolist()):
            hdr_row = i
            break
    if hdr_row is None:
        return pd.DataFrame(columns=["state","mom"])

    h1 = raw.iloc[hdr_row].fillna("")
    h2 = raw.iloc[hdr_row+1].fillna("")
    h1_ff = h1.replace("", np.nan).fillna(method="ffill")

    cols = []
    for a,b in zip(h1_ff, h2):
        a = str(a).strip()
        b = str(b).strip()
        if a and b:
            cols.append(f"{a}__{b}")
        elif a:
            cols.append(a)
        elif b:
            cols.append(b)
        else:
            cols.append("")
    df = raw.iloc[hdr_row+2:].copy()
    df.columns = cols

    state_col = None
    for c in df.columns:
        if "name of the state/ut" in str(c).lower():
            state_col = c
            break
    if state_col is None:
        return pd.DataFrame(columns=["state","mom"])

    comb_idx = [c for c in df.columns
                if str(c).lower().startswith("combined") and "index" in str(c).lower()]
    comb_idx = sorted(comb_idx)
    if len(comb_idx) < 2:
        return pd.DataFrame(columns=["state","mom"])

    prev_c, curr_c = comb_idx[-2], comb_idx[-1]
    prev = pd.to_numeric(df[prev_c], errors="coerce")
    curr = pd.to_numeric(df[curr_c], errors="coerce")
    mom = safe_div(curr, prev) - 1.0

    out = pd.DataFrame({
        "state": df[state_col].astype(str).str.strip(),
        "mom": mom
    })
    out = out[out["state"].ne("") & out["mom"].notna()]
    return out

def parse_annexVII_yoy_for_date_from_bytes(xlsx_bytes: bytes, as_of_date):
    """
    Annex-VII: All-India year-on-year CPI inflation time series.
    Given as_of_date (month/year), returns YoY (decimal) for that month.
    """
    xl = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    sheet = "Annex-VII" if "Annex-VII" in xl.sheet_names else xl.sheet_names[0]
    raw = xl.parse(sheet, header=None)

    hdr_row = None
    for i in range(10):
        row = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        if "year" in row:
            hdr_row = i
            break
    if hdr_row is None:
        return None

    header = raw.iloc[hdr_row].tolist()
    # map month names to columns
    month_cols = {}
    for j,val in enumerate(header):
        v = str(val).strip().lower()
        if v in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]:
            month_cols[v] = j
    if not month_cols:
        return None

    rev_month = {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',
                 6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'}
    mname = rev_month[as_of_date.month]
    col_idx = month_cols.get(mname)
    if col_idx is None:
        return None

    df = raw.iloc[hdr_row+1:, :max(month_cols.values())+1]
    df.columns = header[:max(month_cols.values())+1]
    row_year = df[df['Year'] == as_of_date.year]
    if row_year.empty:
        return None

    val = row_year.iloc[0, col_idx]
    try:
        val_str = str(val).replace("*","").replace("\u00a0","").strip()
        return float(val_str)/100.0
    except Exception:
        return None

def load_annex_cpi_from_zip(zip_path: str):
    """
    Central loader:
      - picks latest Annex workbook in zip
      - parses Annex-III → state MoM (Combined)
      - parses Annex-I → group-level MoM, mapped to functional categories
      - parses Annex-VII → All-India YoY at as_of_date
    Returns:
      state_df (state, mom, as_of_date, workbook),
      cat_mom_full: dict {cat: mom_decimal} for CATS,
      as_of_date_str, workbook_name, yoy_allindia_decimal (may be None)
    """
    if not os.path.exists(zip_path):
        return pd.DataFrame(columns=["state","mom","as_of_date","workbook"]), \
               {c: np.nan for c in CATS}, None, None, None

    with zipfile.ZipFile(zip_path, "r") as z:
        names = sorted([n for n in z.namelist() if n.lower().endswith(".xlsx")])
        latest_name = None
        latest_date = None
        for name in names:
            dt = extract_month_year_from_filename(name)
            if dt and (latest_date is None or dt > latest_date):
                latest_date = dt
                latest_name = name

        if latest_name is None:
            return pd.DataFrame(columns=["state","mom","as_of_date","workbook"]), \
                   {c: np.nan for c in CATS}, None, None, None

        with z.open(latest_name) as f:
            bytes_latest = f.read()

    # Annex-III → state MoM
    state_df = parse_annexIII_state_mom_from_bytes(bytes_latest)
    if not state_df.empty:
        state_df["as_of_date"] = latest_date.isoformat()
        state_df["workbook"] = latest_name

    # Annex-I → group MoM → functional cats
    group_mom = parse_annexI_category_mom_from_bytes(bytes_latest)
    cat_mom_raw = {}
    for desc, m in group_mom.items():
        key = GROUP_MAP.get(desc)
        if key and isinstance(m,(int,float)) and not np.isnan(m):
            cat_mom_raw[key] = float(m)

    # expand to full CATS; health/education/transport share Misc's inflation
    cat_mom_full = {}
    for c in CATS:
        if c in cat_mom_raw:
            cat_mom_full[c] = cat_mom_raw[c]
        elif c in ["health","education","transport"] and "other" in cat_mom_raw:
            cat_mom_full[c] = cat_mom_raw["other"]
        else:
            cat_mom_full[c] = np.nan

    # Annex-VII → All-India YoY
    yoy_allindia = parse_annexVII_yoy_for_date_from_bytes(bytes_latest, latest_date)

    return state_df, cat_mom_full, latest_date.isoformat(), latest_name, yoy_allindia

# ------------------------------ Load data ------------------------------
if not os.path.exists(IN_HH):
    raise FileNotFoundError(f"Missing Phase 2.5 file: {IN_HH}")

hh = pd.read_csv(IN_HH)
macro = load_macro_snapshot(IN_MACRO)

# Annex priors (fallback functional shares)
if os.path.exists(IN_ANNEX):
    a = pd.read_csv(IN_ANNEX).head(1)
    annex_shares = {
        "housing":   float(a.get("share_housing",   pd.Series([0.0])).iloc[0]),
        "food":      float(a.get("share_food",      pd.Series([0.0])).iloc[0]),
        "health":    float(a.get("share_health",    pd.Series([0.0])).iloc[0]),
        "education": float(a.get("share_education", pd.Series([0.0])).iloc[0]),
        "transport": float(a.get("share_transport", pd.Series([0.0])).iloc[0]),
        "other":     float(a.get("share_other",     pd.Series([0.0])).iloc[0]),
    }
    s = sum(annex_shares.values()) or 1.0
    annex_shares = {k:v/s for k,v in annex_shares.items()}
else:
    annex_shares = {c: 1.0/len(CATS) for c in CATS}

# ensure numeric baseline columns
for c in [
    "income_total","expense_total","emi_total","bank_savings",
    "equity_value","debt_funds_value","gold_gms",
    "gold_price_per_g_t0","gold_mark_to_market_t0"
]:
    if c not in hh.columns:
        print(f"[warn] column not found: {c}")
        hh[c] = 0.0
    hh[c] = hh[c].fillna(0.0).astype(float)

# if gold MTM missing, compute from gms × price
if "gold_mark_to_market_t0" not in hh.columns or hh["gold_mark_to_market_t0"].isna().all():
    hh["gold_mark_to_market_t0"] = hh["gold_gms"] * hh["gold_price_per_g_t0"]

# ------------------------------ Build category amounts ------------------------------
have_all_prior_cols = all(f"prior_share_{c}" in hh.columns for c in CATS)

def build_category_amounts(row):
    exp = nz(row["expense_total"])
    use_prior = have_all_prior_cols and np.isclose(
        sum(nz(row[f"prior_share_{c}"]) for c in CATS), 1.0, atol=1e-3
    )
    if use_prior:
        shares = {c: max(0.0, nz(row[f"prior_share_{c}"])) for c in CATS}
    else:
        shares = annex_shares.copy()
    s = sum(shares.values()) or 1.0
    shares = {k: v/s for k,v in shares.items()}
    return {c: exp * shares[c] for c in CATS}

missing_cats = [c for c in CATS if c not in hh.columns]
if missing_cats:
    cat_df = hh.apply(lambda r: pd.Series(build_category_amounts(r)), axis=1)
    hh = pd.concat([hh, cat_df], axis=1)
else:
    for c in CATS:
        hh[c] = hh[c].fillna(0.0).clip(lower=0.0).astype(float)

# ------------------------------ Annex CPI (III + I + VII) ------------------------------
import io  # needed for the Annex parsers

state_df, cat_mom_full, annex_asof, annex_wb, annex_yoy = load_annex_cpi_from_zip(IN_ANNEXZIP)

state_mom_map = {}
if not state_df.empty:
    state_mom_map = {norm_state(s): float(m) for s,m in zip(state_df["state"], state_df["mom"])}

nat_cat_mom_values = [v for v in cat_mom_full.values()
                      if isinstance(v,(int,float)) and not np.isnan(v)]
nat_combined_mom = float(np.mean(nat_cat_mom_values)) if nat_cat_mom_values else np.nan

# ------------------------------ Simulation ------------------------------
detailed_rows = []
scenario_shocks = []

equity_scale = equity_scale_only(macro)
equity_cond_vol = float(macro.get("nifty_vol_36m_t0", np.nan)) if macro else np.nan
equity_cond_dd  = float(macro.get("nifty_drawdown_36m_t0", np.nan)) if macro else np.nan

for scen_name, cfg in SCENARIOS.items():
    scen_cpi_yoy_overlay = float(cfg.get("cpi", 0.0))    # YoY overlay on top of Annex-VII
    cat_elast   = dict(cfg.get("cat_elasticity", {}))
    base_eq_ret = float(cfg.get("equity_ret", 0.0))
    gold_ret    = float(cfg.get("gold_ret", 0.0))
    base_rate_bps = int(cfg.get("rate_bps", 0))
    do_rebal      = bool(cfg.get("rebalance_to_target", False))

    # macro-conditional equity shock
    eq_ret = condition_equity_shock(base_eq_ret, macro)
    # interest rate delta
    delta_bps, rates_source = read_rate_delta_bps_from_xlsx(IN_RATES_X, scen_name, base_rate_bps)

    df = hh.copy()

    # ---- Robust state handling ----
    if "state" in df.columns:
        st_series = df["state"]
    elif "state_x" in df.columns:
        st_series = df["state_x"]
    else:
        # No state info → fall back to NATIONAL CPI only
        print("[WARN] 'state' column missing in Phase 2.5 features; using NATIONAL CPI.")
        st_series = pd.Series([""] * len(df))

    df["_STATE_KEY_"] = st_series.astype(str).map(norm_state)
    # --------------------------------
    
    # income shocks
    df["income_total_shk"] = df.apply(
        lambda r: r["income_total"] * (1.0 + row_income_shock(r, scen_name)), axis=1
    )

    # scenario-level CPI monthly overlay, derived from YoY overlay
    scen_mom_add = scen_cpi_yoy_overlay / 12.0

    # apply CPI to each functional category
    for c in CATS:
        base_amt = df[c].astype(float)

        # state combined MoM
        state_mom = df["_STATE_KEY_"].map(lambda s: state_mom_map.get(s, np.nan))
        # national category MoM (from Annex-I)
        cat_nat_mom = cat_mom_full.get(c, np.nan)

        # combine: cat_nat_mom + (state_combined - nat_combined)
        if not np.isnan(cat_nat_mom) and not np.isnan(nat_combined_mom):
            eff_rate = cat_nat_mom + (state_mom - nat_combined_mom)
        else:
            eff_rate = state_mom.copy()

        # fallbacks: national category, then scenario overlay only
        eff_rate = eff_rate.fillna(cat_nat_mom if not np.isnan(cat_nat_mom) else np.nan)
        eff_rate = eff_rate.fillna(0.0)

        # add scenario overlay
        eff_rate = eff_rate + scen_mom_add

        elast = float(cat_elast.get(c, 1.0))
        df[f"cpi_mom_{c}"] = eff_rate.astype(float)
        df[f"{c}_shk"] = base_amt * (1.0 + eff_rate * elast)

    # shocked expense total (primary path)
    df["expense_total_shk"] = df[[f"{c}_shk" for c in CATS]].sum(axis=1)
    fallback_mask = (df[CATS].sum(axis=1) == 0) & (df["expense_total"] > 0)
    df.loc[fallback_mask, "expense_total_shk"] = (
        df.loc[fallback_mask, "expense_total"] * (1.0 + scen_mom_add)
    )

    # EMI repricing
    emi_uplift = infer_emi_uplift_from_rate_change(df, delta_bps)
    df["emi_total_shk"] = df["emi_total"] * (1.0 + emi_uplift)

    # Asset MTM
    df["equity_value_shk"]         = df["equity_value"] * (1.0 + eq_ret)
    df["gold_price_per_g_shk"]     = df["gold_price_per_g_t0"] * (1.0 + gold_ret)
    df["gold_mark_to_market_shk"]  = df["gold_gms"] * df["gold_price_per_g_shk"]

    dy = delta_bps / 10000.0
    df["debt_funds_value_shk"] = df["debt_funds_value"] * (1.0 - DURATION_DEBT * dy)
    df["debt_funds_value_shk"] = df["debt_funds_value_shk"].clip(lower=0.0)

    # budgets & buffers
    df["expense_to_income_shk"] = safe_div(df["expense_total_shk"], df["income_total_shk"])
    df["emi_to_income_shk"]     = safe_div(df["emi_total_shk"],     df["income_total_shk"])
    df["total_burden_shk"]      = df["expense_to_income_shk"].fillna(0) + \
                                  df["emi_to_income_shk"].fillna(0)
    df["disposable_income_shk"] = df["income_total_shk"] - (df["expense_total_shk"] + df["emi_total_shk"])

    shortfall = (df["expense_total_shk"] + df["emi_total_shk"] - df["income_total_shk"]).clip(lower=0)
    df["buffer_months_shk"] = safe_div(df["bank_savings"], shortfall.replace(0, np.nan))

    # stress components
    df["asset_value_t0"]  = df["equity_value"] + df["gold_mark_to_market_t0"] + df["debt_funds_value"]
    df["asset_value_shk"] = df["equity_value_shk"] + df["gold_mark_to_market_shk"] + df["debt_funds_value_shk"]
    df["loss_assets"]     = df["asset_value_t0"] - df["asset_value_shk"]
    df["inflation_hit"]   = df["expense_total_shk"] - df["expense_total"]
    df["income_gap"]      = (df["expense_total_shk"] + df["emi_total_shk"]) - df["income_total_shk"]

    w_loss, w_gap, w_inf = 0.45, 0.35, 0.20
    df["stress_index"] = (
        w_loss*zscore(df["loss_assets"]) +
        w_gap*zscore(df["income_gap"]) +
        w_inf*zscore(df["inflation_hit"])
    )
    df["stress_quintile"] = robust_quintiles(df["stress_index"])

    df["risk_class_shk"] = [
        classify_risk(tb, br)
        for tb, br in zip(df["total_burden_shk"], df["buffer_months_shk"])
    ]

    # optional rebalancing
    if do_rebal:
        for a in ["alloc_cash","alloc_equity","alloc_debt","alloc_gold"]:
            if a not in df.columns:
                df[a] = 0.0
        post_assets = (
            df["bank_savings"] +
            df["equity_value_shk"] +
            df["debt_funds_value_shk"] +
            df["gold_mark_to_market_shk"]
        )
        tgt_equity = post_assets * df["alloc_equity"]
        tgt_debt   = post_assets * df["alloc_debt"]
        tgt_gold   = post_assets * df["alloc_gold"]
        tgt_cash   = post_assets * df["alloc_cash"]

        df["trade_equity"] = tgt_equity - df["equity_value_shk"]
        df["trade_debt"]   = tgt_debt   - df["debt_funds_value_shk"]
        df["trade_gold"]   = tgt_gold   - df["gold_mark_to_market_shk"]
        df["trade_cash"]   = tgt_cash   - df["bank_savings"]
        df["gross_rebalance_notional"] = df[
            ["trade_equity","trade_debt","trade_gold","trade_cash"]
        ].abs().sum(axis=1)
    else:
        for col in ["trade_equity","trade_debt","trade_gold","trade_cash","gross_rebalance_notional"]:
            df[col] = 0.0

    # ------------- 3.1 Macro Shock Mapping (scenario_shocks table) -------------
    used_rates = []
    for c in CATS:
        col = f"cpi_mom_{c}"
        if col in df.columns:
            used_rates.append(df[col].astype(float))
    used_mom_mean = float(pd.concat(used_rates, axis=0).mean()) if used_rates else scen_mom_add

    cpi_source = "annex_iii_state+annex_i_groups"
    cpi_yoy_source = "annex_vii_allindia"

    scenario_shocks.append({
        "scenario": scen_name,
        "as_of_date": annex_asof,
        "cpi_source": cpi_source,
        "cpi_workbook": annex_wb,
        "cpi_yoy_source": cpi_yoy_source,
        "cpi_yoy_allindia_t0": float(annex_yoy) if annex_yoy is not None else np.nan,
        "scenario_cpi_yoy_overlay": scen_cpi_yoy_overlay,
        "scenario_cpi_yoy_total": (float(annex_yoy) if annex_yoy is not None else 0.0) + scen_cpi_yoy_overlay,
        "shock_equity": float(eq_ret),
        "shock_gold": float(gold_ret),
        "shock_cpi": float(used_mom_mean),  # mean MoM actually applied
        "cpi_food": float(df["cpi_mom_food"].mean()) if "cpi_mom_food" in df else np.nan,
        "cpi_housing": float(df["cpi_mom_housing"].mean()) if "cpi_mom_housing" in df else np.nan,
        "cpi_health": float(df["cpi_mom_health"].mean()) if "cpi_mom_health" in df else np.nan,
        "cpi_education": float(df["cpi_mom_education"].mean()) if "cpi_mom_education" in df else np.nan,
        "cpi_transport": float(df["cpi_mom_transport"].mean()) if "cpi_mom_transport" in df else np.nan,
        "cpi_other": float(df["cpi_mom_other"].mean()) if "cpi_mom_other" in df else np.nan,
        "shock_rate_bps": int(delta_bps),
        "rates_source": rates_source,
        "equity_cond_vol": equity_cond_vol,
        "equity_cond_dd": equity_cond_dd,
        "equity_scale": equity_scale,
    })

    # ------------- spec-named post_* fields -------------
    df["post_portfolio_total"] = (
        df["bank_savings"] +
        df["equity_value_shk"] +
        df["debt_funds_value_shk"] +
        df["gold_mark_to_market_shk"]
    )
    df["post_expense_total"] = df["expense_total_shk"]
    df["post_exp"] = df["post_expense_total"]
    df["inflation_pass_through"] = safe_div(
        df["post_expense_total"] - df["expense_total"], df["expense_total"]
    )
    df["post_savings"] = df["disposable_income_shk"]
    df["liquidity_ratio"] = safe_div(
        df["bank_savings"], (df["post_expense_total"] + df["emi_total_shk"])
    )
    df["scenario"] = scen_name
    df["as_of_date"] = annex_asof

    # keep columns for output
    keep_cols = [
        "lead_id","state","employment_type","risk_tolerance","as_of_date",
        "income_total","expense_total","emi_total","bank_savings",
        "equity_value","debt_funds_value","gold_price_per_g_t0","gold_gms","gold_mark_to_market_t0",
        *CATS,
        *[f"{c}_shk" for c in CATS],
        *[f"cpi_mom_{c}" for c in CATS],
        "income_total_shk","expense_total_shk","emi_total_shk",
        "expense_to_income_shk","emi_to_income_shk","total_burden_shk",
        "disposable_income_shk","buffer_months_shk",
        "equity_value_shk","debt_funds_value_shk","gold_price_per_g_shk","gold_mark_to_market_shk",
        "loss_assets","inflation_hit","income_gap","stress_index","stress_quintile","risk_class_shk",
        "trade_equity","trade_debt","trade_gold","trade_cash","gross_rebalance_notional",
        "post_portfolio_total","post_expense_total","post_exp",
        "inflation_pass_through","post_savings","liquidity_ratio",
        "scenario",
    ]
    present = [c for c in keep_cols if c in df.columns]
    detailed_rows.append(df[present])

# ------------------------------ Save CSVs ------------------------------
detailed = pd.concat(detailed_rows, ignore_index=True)

summary = (
    detailed.groupby(["as_of_date","scenario"], as_index=False)
    .agg(
        households=("lead_id","count"),
        mean_total_burden=("total_burden_shk","mean"),
        p90_total_burden=("total_burden_shk", lambda s: float(np.nanpercentile(s, 90))),
        share_distressed=("risk_class_shk", lambda s: float((s=="Distressed").mean())),
        share_atrisk=("risk_class_shk", lambda s: float((s=="At-Risk").mean())),
        share_stable=("risk_class_shk", lambda s: float((s=="Stable").mean())),
        mean_buffer_months=("buffer_months_shk","mean"),
        median_buffer_months=("buffer_months_shk","median"),
        mean_disposable=("disposable_income_shk","mean"),
        median_disposable=("disposable_income_shk","median"),
        mean_stress_index=("stress_index","mean"),
        p90_stress_index=("stress_index", lambda s: float(np.nanpercentile(s, 90))),
        mean_gross_rebalance_notional=("gross_rebalance_notional","mean"),
    )
)

pd.DataFrame(scenario_shocks).drop_duplicates(["as_of_date","scenario"]).to_csv(OUT_SHOCKS, index=False)
detailed.to_csv(OUT_DETAILED, index=False)
summary.to_csv(OUT_SUMMARY, index=False)

print(f"[Phase 3] Wrote per-household results → {OUT_DETAILED}")
print(f"[Phase 3] Wrote scenario summary     → {OUT_SUMMARY}")
print(f"[Phase 3] Wrote scenario shock map   → {OUT_SHOCKS}")

# ------------------------------ Plots ------------------------------
latest_date = (
    summary["as_of_date"].dropna().sort_values().iloc[-1]
    if not summary.empty else None
)
plot_df = summary[summary["as_of_date"] == latest_date] if latest_date else summary

# 1) Risk class shares by scenario
plt.figure()
x = np.arange(len(plot_df["scenario"]))
w = 0.25
plt.bar(x - w, plot_df["share_distressed"], width=w, label="Distressed")
plt.bar(x,       plot_df["share_atrisk"],  width=w, label="At-Risk")
plt.bar(x + w,   plot_df["share_stable"],  width=w, label="Stable")
plt.xticks(x, plot_df["scenario"])
plt.ylabel("Share")
plt.title("Risk Class Shares by Scenario")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "risk_class_shares_by_scenario.png"), dpi=150)

# 2) Total burden (Mean & P90)
plt.figure()
plt.plot(plot_df["scenario"], plot_df["mean_total_burden"], marker="o", label="Mean Total Burden")
plt.plot(plot_df["scenario"], plot_df["p90_total_burden"],  marker="o", label="P90 Total Burden")
plt.ylabel("Total Burden")
plt.title("Total Burden (Mean & P90) by Scenario")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "total_burden_mean_p90_by_scenario.png"), dpi=150)

# 3) Disposable income distributions (Base vs Severe)
plt.figure()
for scen in ["Base","Severe"]:
    vals = detailed.loc[
        (detailed["scenario"]==scen) &
        ((detailed["as_of_date"]==latest_date) if latest_date else True),
        "disposable_income_shk"
    ].values
    plt.hist(vals, bins=30, alpha=0.5, label=scen)
plt.xlabel("Disposable Income (shocked)")
plt.ylabel("Count")
plt.title("Disposable Income Distribution (Base vs Severe)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "disposable_income_hist_base_vs_severe.png"), dpi=150)

# 4) Top 10 states by distressed share under Severe
if "state" in detailed.columns:
    severe = detailed[
        (detailed["scenario"]=="Severe") &
        ((detailed["as_of_date"]==latest_date) if latest_date else True)
    ]
    state_tbl = (
        severe.groupby("state")["risk_class_shk"]
        .apply(lambda s: (s=="Distressed").mean())
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="distressed_share")
    )
    plt.figure()
    plt.bar(state_tbl["state"], state_tbl["distressed_share"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Share Distressed")
    plt.title("Top 10 States — Distressed Share (Severe)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "top10_states_distressed_severe.png"), dpi=150)

print(f"[Phase 3] Plots saved → {PLOTS}")
