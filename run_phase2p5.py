import os, sys, pandas as pd, numpy as np

# --------------------------- Paths & Imports ---------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
PKG_PATH = os.path.join(ROOT, "stress_sim", "src")
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from stress_sim.cpi_mapper import map_households_to_cpi
from stress_sim.hces_annex_priors import load_annex_priors
from stress_sim.household_features import compute_static_features

DATAI = os.path.join(ROOT, "data_interim")
PARTB = os.path.join(ROOT, "synth_partB")

paths = {
    "cpi":   os.path.join(DATAI, "cpi_long.csv"),
    "annex": os.path.join(DATAI, "annex_priors.csv"),
    "hh":    os.path.join(PARTB, "households.csv"),
    "exp":   os.path.join(PARTB, "household_expenses.csv"),
    "loans": os.path.join(PARTB, "household_loans.csv"),
    "assets":os.path.join(PARTB, "household_assets.csv"),
}

missing = [k for k, p in paths.items() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing inputs: { {k: paths[k] for k in missing} }")

# --------------------------- Load Data ---------------------------
cpi_long = pd.read_csv(paths["cpi"], parse_dates=["date"])
annex    = load_annex_priors(paths["annex"])
df_hh    = pd.read_csv(paths["hh"])
df_exp   = pd.read_csv(paths["exp"])
df_loans = pd.read_csv(paths["loans"])
df_assets= pd.read_csv(paths["assets"])

# Normalize all incoming Part-B headers to lowercase
for d in (df_hh, df_exp, df_loans, df_assets):
    d.columns = [c.strip().lower() for c in d.columns]

# --------------------------- Phase 2: Static HH features + CPI map ---------------------------
mapper, cpi_combined = map_households_to_cpi(df_hh, cpi_long)  # (mapper per lead_id, derived CPI table)
features = compute_static_features(df_hh, df_exp, df_loans, df_assets)
features = features.merge(mapper, on="lead_id", how="left")    # adds cpi_state_mapped, etc.

# Safety: if emi_total is NaN, treat as 0 (prevents NaN burden)
if "emi_total" in features.columns:
    features["emi_total"] = features["emi_total"].fillna(0)

# If ratios missing, recompute conservative versions
if {"expense_total","income_total"}.issubset(features.columns):
    if "expense_to_income" not in features.columns or features["expense_to_income"].isna().any():
        features["expense_to_income"] = features["expense_total"] / features["income_total"].replace(0, np.nan)

if {"emi_total","income_total"}.issubset(features.columns):
    if "emi_to_income" not in features.columns or features["emi_to_income"].isna().any():
        features["emi_to_income"] = features["emi_total"] / features["income_total"].replace(0, np.nan)

if {"expense_to_income","emi_to_income"}.issubset(features.columns):
    if "total_burden" not in features.columns or features["total_burden"].isna().any():
        features["total_burden"] = features["expense_to_income"].fillna(0) + features["emi_to_income"].fillna(0)

# (Optional) write pure Phase-2 snapshot (no macro snapshots yet)
phase2_out = os.path.join(DATAI, "households_features.csv")
features.to_csv(phase2_out, index=False)
print(f"Phase 2 features written: {phase2_out}  | rows={len(features)}")

# --------------------------- Phase 2.5: Attach Part-A Snapshot Features ---------------------------

# Ensure we have 'state' once (prefer household state)
if "state" not in features.columns and "state" in df_hh.columns:
    features = features.merge(df_hh[["lead_id","state"]], on="lead_id", how="left")

# Load macro series
nifty = pd.read_csv(os.path.join(DATAI, "nifty_monthly.csv"), parse_dates=["date"])
gold  = pd.read_csv(os.path.join(DATAI, "gold_monthly.csv"),  parse_dates=["date"])

# CPI combined slice (segment='combined' only); add NATIONAL if needed
cpi_combined = cpi_long.copy()
if "segment" in cpi_combined.columns:
    cpi_combined = cpi_combined[cpi_combined["segment"].str.lower() == "combined"].copy()
else:
    cpi_combined["segment"] = "combined"

if "state" in cpi_combined.columns:
    by_date = cpi_combined.groupby("date", as_index=False)["cpi_value"].median()
    by_date["state"] = "NATIONAL"; by_date["segment"] = "combined"
    cpi_combined = pd.concat([cpi_combined, by_date], ignore_index=True)

# t0 is the most recent CPI month
t0 = cpi_combined["date"].max()

def cpi_for(state: str, dt: pd.Timestamp) -> float:
    """Exact-month CPI for state if present, else NATIONAL median for that month."""
    sub = cpi_combined[cpi_combined["date"] == dt]
    if sub.empty:
        return np.nan
    if "state" in sub.columns and state in set(sub["state"].astype(str)):
        return float(sub.loc[sub["state"].astype(str) == state, "cpi_value"].median())
    return float(sub["cpi_value"].median())  # NATIONAL fallback

# Month lags (exact month-end anchors)
lags = {
    "t0":  t0,
    "t1":  (t0 - pd.offsets.MonthEnd(1)),
    "t3":  (t0 - pd.offsets.MonthEnd(3)),
    "t6":  (t0 - pd.offsets.MonthEnd(6)),
    "t12": (t0 - pd.offsets.MonthEnd(12)),
}

# Build per-household CPI snapshots
feat_cpi = features[["lead_id"]].copy()
feat_cpi["state_for_cpi"] = features.get("cpi_state_mapped", features.get("state", "NATIONAL")).astype(str)
for tag, d in lags.items():
    feat_cpi[f"cpi_{tag}"] = feat_cpi["state_for_cpi"].apply(lambda s: cpi_for(s, d))

# Compute MoM / 3m ann / 6m ann
feat_cpi["cpi_level_t0"]         = feat_cpi["cpi_t0"]
feat_cpi["cpi_mom_t0"]           = (feat_cpi["cpi_t0"] / feat_cpi["cpi_t1"] - 1.0)
feat_cpi["cpi_3m_annualized_t0"] = ((feat_cpi["cpi_t0"] / feat_cpi["cpi_t3"]) ** (12/3) - 1.0)
feat_cpi["cpi_6m_annualized_t0"] = ((feat_cpi["cpi_t0"] / feat_cpi["cpi_t6"]) ** (12/6) - 1.0)

# ---- YoY with annualized fallback (Option-2) ----
# True YoY if t12 exists, else fallback using most recent k in [11..6]
def yoy_annualized_fallback(state: str, t0_date: pd.Timestamp, min_months=6, max_months=11) -> float:
    sub = cpi_combined.copy()
    if "state" in sub.columns:
        sub_state = sub[sub["state"].astype(str) == state]
        if sub_state.empty:
            sub_state = sub[sub["state"].astype(str) == "NATIONAL"]
        sub = sub_state
    if sub.empty:
        return np.nan

    for k in range(max_months, 0, -1):
        anchor = t0_date - pd.offsets.MonthEnd(k)
        row_t0 = sub[sub["date"] == t0_date]
        row_k  = sub[sub["date"] == anchor]
        if not row_t0.empty and not row_k.empty and k >= min_months:
            cpi_t0 = float(row_t0["cpi_value"].median())
            cpi_k  = float(row_k["cpi_value"].median())
            if cpi_k > 0 and np.isfinite(cpi_t0) and np.isfinite(cpi_k):
                return (cpi_t0 / cpi_k) ** (12.0 / k) - 1.0
    return np.nan

cpi_yoy_true = (feat_cpi["cpi_t0"] / feat_cpi["cpi_t12"] - 1.0)
cpi_yoy_fallback = feat_cpi["state_for_cpi"].apply(lambda s: yoy_annualized_fallback(s, lags["t0"], min_months=6, max_months=11))
feat_cpi["cpi_yoy_t0"] = cpi_yoy_true.where(cpi_yoy_true.notna(), cpi_yoy_fallback)

# Drop helper columns
feat_cpi = feat_cpi.drop(columns=[c for c in feat_cpi.columns if c.startswith("cpi_t")] + ["state_for_cpi"])

# Merge CPI snapshot columns ONCE
features = features.merge(feat_cpi, on="lead_id", how="left")

# --------------------------- Priors + Gaps ---------------------------
annex_df = annex if isinstance(annex, pd.DataFrame) else load_annex_priors(paths["annex"])
pri = annex_df.iloc[0].to_dict() if not annex_df.empty else {}

for k in ["housing","food","health","education","transport","other"]:
    features[f"prior_share_{k}"] = float(pri.get(f"share_{k}", 0.0))

needed = {"housing","food","health","education","transport","other","expense_total"}
if needed.issubset(features.columns):
    for k in ["housing","food","health","education","transport","other"]:
        obs_share = features[k] / features["expense_total"].replace(0, np.nan)
        features[f"exp_share_gap_{k}"] = obs_share - features[f"prior_share_{k}"]

# --------------------------- Gold t0 snapshot (auto unit detection) ---------------------------
# Heuristic: if median price in [10_000, 100_000], treat as per 10g (divide by 10)
#            if in [1_000, 10_000], likely per gram (no change)
#            else leave as-is (advanced series -> extend mapping as needed)
gold_sorted = gold.sort_values("date")
price_median = float(gold_sorted["price"].median()) if "price" in gold_sorted.columns and not gold_sorted.empty else np.nan

divisor = 1.0
if np.isfinite(price_median):
    if 10_000 <= price_median <= 100_000:
        divisor = 10.0  # per 10g → per g
    elif price_median > 100_000:  # weirdly large → assume already scaled; try 100? (per 100g)
        divisor = 100.0
    else:
        divisor = 1.0  # assume per g

gold_t0_row = gold_sorted[gold_sorted["date"] == gold_sorted["date"].max()]
gold_t0_price = (float(gold_t0_row["price"].iloc[0]) / divisor) if not gold_t0_row.empty else np.nan

features["gold_price_per_g_t0"]    = gold_t0_price
if "gold_gms" in features.columns:
    features["gold_mark_to_market_t0"] = features["gold_gms"].fillna(0) * gold_t0_price

# --------------------------- Tidy 'state' ---------------------------
if "state_x" in features.columns or "state_y" in features.columns:
    features["state"] = features["state_x"] if "state_x" in features.columns else features.get("state_y")
    for col in ["state_x", "state_y"]:
        if col in features.columns:
            features.drop(columns=[col], inplace=True)

# --------------------------- Outputs (Phase 2.5) ---------------------------
features_out = os.path.join(DATAI, "households_features_phase2p5.csv")
features.to_csv(features_out, index=False)
print(f"Phase 2.5 features written: {features_out}  | rows={len(features)}")

# --------------------------- Macro snapshot (global) ---------------------------
def ret_window(df, col, months):
    sub = df.sort_values("date").tail(months)
    if sub.empty or col not in sub.columns:
        return np.nan
    return float((sub[col] + 1.0).prod() - 1.0)

macro = {}
if {"ret_nifty_m","lr_nifty_m"}.issubset(set(nifty.columns)):
    macro["nifty_ret_12m_t0"]  = ret_window(nifty, "ret_nifty_m", 12)
    macro["nifty_vol_36m_t0"]  = float(nifty["lr_nifty_m"].tail(36).std(ddof=1))
    tri = nifty.sort_values("date")["tri"].dropna()
    peak = tri.cummax()
    macro["nifty_drawdown_36m_t0"] = float((tri / peak - 1.0).min()) if not tri.empty else np.nan

if {"ret_gold_m","lr_gold_m"}.issubset(set(gold.columns)):
    macro["gold_ret_12m_t0"] = ret_window(gold, "ret_gold_m", 12)
    macro["gold_vol_36m_t0"] = float(gold["lr_gold_m"].tail(36).std(ddof=1))

pd.DataFrame([macro]).to_csv(os.path.join(DATAI, "macro_snapshot_t0.csv"), index=False)
print("Macro snapshot written: data_interim/macro_snapshot_t0.csv")
# --------------------------- End Phase 2.5 ---------------------------
