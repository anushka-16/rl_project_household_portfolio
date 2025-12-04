"""
Phase 6 — Validation & Backtests

Sanity, calibration and historical backtests.

Reads
-----
data_interim/households_features.csv
(Phase 5 KPI outputs) outputs/kpis_S*.csv
data_interim/nifty_monthly.csv
data_interim/gold_monthly.csv
data_interim/cpi_long.csv
data_interim/annex_priors.csv   (Annex-II style priors OR wide share_* priors)

Writes
------
data_interim/phase6_hces_alignment.csv
data_interim/phase5_household_panel.csv
data_interim/phase6_accept_reject_summary.csv
data_interim/phase6_backtest_timeseries.csv
data_interim/phase6_backtest_summary.csv
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------- Paths & Config ---------------------------

# phase6_validation.py is in Project/partA/
# data_interim and outputs are in Project/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATAI = os.path.join(ROOT, "data_interim")
OUTPUTS = os.path.join(ROOT, "outputs")


@dataclass
class Phase6Config:
    # Core inputs
    households_features_path: str = os.path.join(DATAI, "households_features.csv")
    # We'll create this combined panel from kpis_S*.csv if it doesn't exist
    panel_path: str = os.path.join(DATAI, "phase5_household_panel.csv")

    # Annex-II-like priors
    annex2_priors_path: Optional[str] = os.path.join(DATAI, "annex_priors.csv")

    # Macro series
    nifty_monthly_path: str = os.path.join(DATAI, "nifty_monthly.csv")
    gold_monthly_path: str = os.path.join(DATAI, "gold_monthly.csv")
    cpi_long_path: str = os.path.join(DATAI, "cpi_long.csv")

    # Outputs
    hces_alignment_out: str = os.path.join(DATAI, "phase6_hces_alignment.csv")
    kpi_accept_reject_out: str = os.path.join(DATAI, "phase6_accept_reject_summary.csv")
    backtest_ts_out: str = os.path.join(DATAI, "phase6_backtest_timeseries.csv")
    backtest_summary_out: str = os.path.join(DATAI, "phase6_backtest_summary.csv")

    # Thresholds / rules
    burden_min: float = 0.0
    burden_max: float = 3.0
    kpi_ok_share_min: float = 0.95

    # Default-risk heuristic thresholds for the backtest engine
    high_burden_threshold: float = 1.10
    short_runway_months_threshold: float = 3.0

    # Historical windows (inclusive start, inclusive end).
    # If None/empty → auto-detect from macro data (Option C).
    historical_windows: Optional[Dict[str, Tuple[str, str]]] = None

    def __post_init__(self):
        # No hard-coded windows; we infer them from macro data if not supplied.
        if self.historical_windows is None:
            self.historical_windows = {}


# --------------------------- Helpers ---------------------------

def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise safe division → NaN on zero / invalid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def quantile_iqr(series: pd.Series) -> Tuple[float, float, float]:
    """Return (median, p25, p75) ignoring NaNs."""
    q25 = series.quantile(0.25)
    q50 = series.quantile(0.50)
    q75 = series.quantile(0.75)
    return q50, q25, q75


# --------------------------- 1) HCES Alignment ---------------------------

def load_household_shares(cfg: Phase6Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.households_features_path)
    share_cols = [
        "prior_share_housing",
        "prior_share_food",
        "prior_share_health",
        "prior_share_education",
        "prior_share_transport",
        "prior_share_other",
    ]
    present = [c for c in share_cols if c in df.columns]
    if not present:
        raise ValueError(
            "No prior_share_* columns found in households_features. "
            "Phase 2/3 HCES mapping must have run before Phase 6."
        )
    return df[["lead_id"] + present].copy()


def load_annex2_priors(cfg: Phase6Config) -> Optional[pd.DataFrame]:
    """
    Load Annex-II-like priors.

    Supports two formats:

    1) Proper long Annex-II style:
         functional_category, median_share, p25_share, p75_share
    2) Wide 'share_*' style (your current annex_priors.csv):
         share_housing, share_food, ..., plus optional extra columns.

       In this case we:
         - create functional_category from the 'share_*' suffix,
         - set median_share, p25_share, p75_share from the distribution of each
           share_* column (with a single row, all three are the same).
    """
    path = cfg.annex2_priors_path
    if not path or not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    expected_cols = {"functional_category", "median_share", "p25_share", "p75_share"}
    if expected_cols.issubset(df.columns):
        # Already in Annex-II long format
        return df[list(expected_cols)]

    # Try wide share_* format
    share_cols = [c for c in df.columns if c.startswith("share_")]
    if share_cols:
        rows = []
        for col in share_cols:
            cat = col.replace("share_", "")
            s = df[col].dropna().astype(float)
            if s.empty:
                continue
            median = float(s.median())
            p25 = float(s.quantile(0.25))
            p75 = float(s.quantile(0.75))
            rows.append(
                {
                    "functional_category": cat,
                    "median_share": median,
                    "p25_share": p25,
                    "p75_share": p75,
                }
            )
        if not rows:
            return None
        out = pd.DataFrame(rows)
        print("[Phase6] Converted wide annex_priors.csv (share_*) to long Annex-II-style priors.")
        return out

    # If neither format matches, we can't use this file
    raise ValueError(
        f"Annex priors file at {path} is not in an expected format. "
        "Either provide columns "
        "['functional_category','median_share','p25_share','p75_share'] "
        "or wide 'share_*' columns."
    )


def compute_hces_alignment(cfg: Phase6Config) -> pd.DataFrame:
    df_shares = load_household_shares(cfg)
    df_annex = load_annex2_priors(cfg)

    share_cols = [c for c in df_shares.columns if c.startswith("prior_share_")]
    df_long = df_shares.melt(
        id_vars="lead_id",
        value_vars=share_cols,
        var_name="share_col",
        value_name="share",
    )
    df_long["functional_category"] = df_long["share_col"].str.replace(
        "prior_share_", "", regex=False
    )

    rows = []
    for cat, sub in df_long.groupby("functional_category"):
        med, p25, p75 = quantile_iqr(sub["share"].dropna())
        row = {
            "functional_category": cat,
            "hh_median": med,
            "hh_p25": p25,
            "hh_p75": p75,
        }
        if df_annex is not None:
            ref = df_annex[df_annex["functional_category"] == cat]
            if not ref.empty:
                row["annex_median"] = float(ref["median_share"].iloc[0])
                row["annex_p25"] = float(ref["p25_share"].iloc[0])
                row["annex_p75"] = float(ref["p75_share"].iloc[0])
                row["delta_median_pp"] = (
                    row["hh_median"] - row["annex_median"]
                ) * 100.0
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(cfg.hces_alignment_out, index=False)
    print(f"[Phase6] HCES alignment written to {cfg.hces_alignment_out}")
    return df_out


# --------------------------- Phase 5 Panel Loader ---------------------------

def load_phase5_panel(cfg: Phase6Config) -> pd.DataFrame:
    """
    Build (or read) the combined Phase 5 household panel.

    If cfg.panel_path exists: read and return it.
    Else: concatenate outputs/kpis_S0..S3, recompute key ratios
    (fixing the liquidity_runway bug), and write cfg.panel_path.
    """
    if os.path.exists(cfg.panel_path):
        print(f"[Phase6] Using existing Phase 5 panel at {cfg.panel_path}")
        return pd.read_csv(cfg.panel_path)

    scenario_files = ["kpis_S0.csv", "kpis_S1.csv", "kpis_S2.csv", "kpis_S3.csv"]
    dfs = []
    for fname in scenario_files:
        path = os.path.join(OUTPUTS, fname)
        if os.path.exists(path):
            print(f"[Phase6] Loading Phase 5 KPIs from {path}")
            dfs.append(pd.read_csv(path))
        else:
            print(f"[Phase6][WARN] Expected Phase 5 file not found: {path}")

    if not dfs:
        raise FileNotFoundError(
            "No Phase 5 KPI files found in outputs/ (kpis_S*.csv)."
        )

    panel = pd.concat(dfs, ignore_index=True)

    # Recompute core ratios (in case earlier code had mistakes)
    panel["emi_to_income"] = safe_div(panel["emi_total"], panel["income_total"])
    panel["expense_to_income"] = safe_div(panel["expense_total"], panel["income_total"])
    panel["total_burden"] = panel["emi_to_income"] + panel["expense_to_income"]
    panel["liquidity_runway_months"] = safe_div(
        panel["bank_savings"], panel["expense_total"] + panel["emi_total"]
    )

    panel.to_csv(cfg.panel_path, index=False)
    print(f"[Phase6] Combined Phase 5 panel written to {cfg.panel_path}")
    return panel


# --------------------------- 2) KPI Sanity / Accept-Reject ---------------------------

def run_kpi_accept_reject(cfg: Phase6Config) -> pd.DataFrame:
    panel = load_phase5_panel(cfg)

    required_cols = [
        "scenario",
        "lead_id",
        "month_index",
        "total_burden",
        "expense_to_income",
        "emi_to_income",
        "liquidity_runway_months",
        "net_worth_change_pct",
        "net_worth_drawdown_pct",
    ]
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Phase 5 panel is missing required columns: {missing}. "
            "Please make sure Phase 5 output includes the KPI columns defined in the spec."
        )

    numeric_cols = [
        "total_burden",
        "expense_to_income",
        "emi_to_income",
        "liquidity_runway_months",
        "net_worth_change_pct",
        "net_worth_drawdown_pct",
    ]

    panel[numeric_cols] = panel[numeric_cols].replace([np.inf, -np.inf], np.nan)

    panel["kpi_non_missing"] = ~panel[numeric_cols].isna().any(axis=1)
    panel["burden_in_range"] = (
        (panel["total_burden"] >= cfg.burden_min)
        & (panel["total_burden"] <= cfg.burden_max)
        | panel["total_burden"].isna()
    )
    panel["row_ok"] = panel["kpi_non_missing"] & panel["burden_in_range"]

    hh_ok = (
        panel.groupby(["scenario", "lead_id"])["row_ok"]
        .all()
        .rename("hh_ok")
        .reset_index()
    )

    summary = (
        hh_ok.groupby("scenario")["hh_ok"]
        .agg(
            households_total="size",
            households_ok="sum",
        )
        .reset_index()
    )
    summary["ok_share"] = summary["households_ok"] / summary["households_total"]
    summary["ok_share_ge_threshold"] = summary["ok_share"] >= cfg.kpi_ok_share_min

    summary.to_csv(cfg.kpi_accept_reject_out, index=False)
    print(f"[Phase6] KPI accept/reject summary written to {cfg.kpi_accept_reject_out}")

    for _, r in summary.iterrows():
        status = "PASS" if r["ok_share_ge_threshold"] else "FAIL"
        print(
            f"  Scenario {r['scenario']}: {r['ok_share']:.2%} households ok → {status}"
        )

    return summary


# --------------------------- 3) Historical Backtests ---------------------------

def load_macro_series(cfg: Phase6Config):
    """
    Uses your actual file schemas:

      nifty_monthly.csv: date, tri, ret_nifty_m, lr_nifty_m
      gold_monthly.csv:  date, price, ret_gold_m, lr_gold_m
      cpi_long.csv:      date, segment, centre, state, cpi_value, conversion_factor

    Returns:
      nifty: date, eq_ret
      gold:  date, gold_ret
      cpi:   date, cpi_mom (All-India, combined)
    """
    nifty = pd.read_csv(cfg.nifty_monthly_path, parse_dates=["date"], dayfirst=True)
    gold = pd.read_csv(cfg.gold_monthly_path, parse_dates=["date"], dayfirst=True)
    cpi = pd.read_csv(cfg.cpi_long_path, parse_dates=["date"], dayfirst=True)

    # Equity returns
    nifty = nifty.sort_values("date")
    if "ret_nifty_m" in nifty.columns:
        nifty["eq_ret"] = nifty["ret_nifty_m"].astype(float)
    elif "tri" in nifty.columns:
        nifty["eq_ret"] = nifty["tri"].pct_change()
    else:
        raise ValueError("nifty_monthly.csv must have 'ret_nifty_m' or 'tri'.")

    # Gold returns
    gold = gold.sort_values("date")
    if "ret_gold_m" in gold.columns:
        gold["gold_ret"] = gold["ret_gold_m"].astype(float)
    elif "price" in gold.columns:
        gold["gold_ret"] = gold["price"].pct_change()
    else:
        raise ValueError("gold_monthly.csv must have 'ret_gold_m' or 'price'.")

    # CPI: All-India, combined segment
    if "centre" not in cpi.columns or "segment" not in cpi.columns:
        raise ValueError("cpi_long.csv must have 'centre' and 'segment' columns.")

    cpi["segment"] = cpi["segment"].str.lower()
    mask_nat = (cpi["centre"] == "All-India") & (cpi["segment"] == "combined")
    cpi_nat = cpi.loc[mask_nat].sort_values("date").copy()
    if cpi_nat.empty:
        raise ValueError("No All-India combined CPI rows found in cpi_long.csv.")

    if "cpi_mom" not in cpi_nat.columns:
        cpi_nat["cpi_mom"] = cpi_nat["cpi_value"].pct_change()

    nifty = nifty[["date", "eq_ret"]]
    gold = gold[["date", "gold_ret"]]
    cpi_nat = cpi_nat[["date", "cpi_mom"]]

    return nifty, gold, cpi_nat


def build_macro_window(df_merged: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Slice an already-merged macro DataFrame for [start, end] and drop NaNs.
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    mask = (df_merged["date"] >= start_dt) & (df_merged["date"] <= end_dt)
    df_win = df_merged.loc[mask].sort_values("date").reset_index(drop=True)

    df_win = df_win.dropna(subset=["eq_ret", "gold_ret", "cpi_mom"]).reset_index(
        drop=True
    )
    if df_win.empty:
        raise ValueError(f"No overlapping monthly macro data for window {start}–{end}.")
    return df_win


def infer_historical_windows(df_macro_all: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Option C: Auto-detect historical windows from macro data.

    - Baseline window = first available calendar year in the merged macro data.
    - If year 2020 exists → add 'equity_crash_2020'.
    - If year 2022 exists → add 'inflation_spike_2022'.
    """
    years = df_macro_all["date"].dt.year.unique()
    years = np.sort(years)

    if len(years) == 0:
        raise ValueError("No dates found in merged macro data.")

    baseline_year = int(years[0])
    windows: Dict[str, Tuple[str, str]] = {}

    baseline_name = f"baseline_{baseline_year}"
    baseline_start = f"{baseline_year}-01-01"
    baseline_end = f"{baseline_year}-12-31"
    windows[baseline_name] = (baseline_start, baseline_end)

    print(
        f"[Phase6] Auto-inferred baseline window '{baseline_name}' "
        f"= {baseline_start} to {baseline_end}"
    )

    if 2020 in years:
        windows["equity_crash_2020"] = ("2020-01-01", "2020-12-31")
        print("[Phase6] Added equity_crash_2020 window (2020-01-01 to 2020-12-31).")

    if 2022 in years:
        windows["inflation_spike_2022"] = ("2022-01-01", "2022-12-31")
        print("[Phase6] Added inflation_spike_2022 window (2022-01-01 to 2022-12-31).")

    return windows


def run_backtest_for_window(
    cfg: Phase6Config,
    window_name: str,
    macro_path: pd.DataFrame,
    df_hh: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = [
        "lead_id",
        "income_total",
        "expense_total",
        "emi_total",
        "bank_savings",
        "equity_value",
        "debt_funds_value",
    ]
    gold_candidates = ["gold_mark_to_market_t0", "gold_value", "gold_mark_to_market"]
    gold_col = next((c for c in gold_candidates if c in df_hh.columns), None)

    missing = [c for c in base_cols if c not in df_hh.columns]
    if gold_col is None:
        missing.append("gold_value (any of: " + ", ".join(gold_candidates) + ")")
    if missing:
        raise ValueError(
            "Households features missing required exposure columns for backtest: "
            f"{missing}"
        )

    hh = df_hh.copy()

    current_income = hh["income_total"].astype(float)
    current_expense = hh["expense_total"].astype(float)
    current_emi = hh["emi_total"].astype(float)
    current_cash = hh["bank_savings"].astype(float)
    current_equity = hh["equity_value"].astype(float)
    current_debt = hh["debt_funds_value"].astype(float)
    current_gold = hh[gold_col].astype(float)

    base_net_assets = current_cash + current_equity + current_debt + current_gold
    peak_net_assets = base_net_assets.copy()

    all_rows = []

    for t, row in enumerate(macro_path.itertuples(index=False), start=1):
        eq_ret = float(row.eq_ret)
        gold_ret = float(row.gold_ret)
        cpi_mom = float(row.cpi_mom)

        current_income = current_income * (1.0 + cpi_mom)
        current_expense = current_expense * (1.0 + cpi_mom)
        current_equity = current_equity * (1.0 + eq_ret)
        current_gold = current_gold * (1.0 + gold_ret)
        current_debt = current_debt  # flat for now

        cashflow = current_income - current_expense - current_emi
        current_cash = current_cash + cashflow

        net_assets = current_cash + current_equity + current_debt + current_gold
        peak_net_assets = np.maximum(peak_net_assets, net_assets)

        emi_to_income = safe_div(current_emi, current_income)
        expense_to_income = safe_div(current_expense, current_income)
        total_burden = emi_to_income + expense_to_income
        liquidity_runway = safe_div(
            current_cash, current_expense + current_emi
        )

        net_worth_change_pct = safe_div(
            net_assets - base_net_assets, base_net_assets
        )
        net_worth_drawdown_pct = safe_div(
            peak_net_assets - net_assets, peak_net_assets
        )

        high_burden_flag = total_burden > cfg.high_burden_threshold
        short_runway_flag = liquidity_runway < cfg.short_runway_months_threshold
        default_high_risk = high_burden_flag | short_runway_flag

        df_step = pd.DataFrame(
            {
                "window": window_name,
                "month_index": t,
                "date": row.date,
                "lead_id": hh["lead_id"],
                "income_total": current_income,
                "expense_total": current_expense,
                "emi_total": current_emi,
                "bank_savings": current_cash,
                "equity_value": current_equity,
                "debt_funds_value": current_debt,
                "gold_value": current_gold,
                "net_assets": net_assets,
                "emi_to_income": emi_to_income,
                "expense_to_income": expense_to_income,
                "total_burden": total_burden,
                "liquidity_runway_months": liquidity_runway,
                "net_worth_change_pct": net_worth_change_pct,
                "net_worth_drawdown_pct": net_worth_drawdown_pct,
                "high_burden_flag": high_burden_flag.astype(int),
                "short_runway_flag": short_runway_flag.astype(int),
                "default_high_risk": default_high_risk.astype(int),
            }
        )
        all_rows.append(df_step)

    df_out = pd.concat(all_rows, ignore_index=True)
    return df_out


def run_historical_backtests(cfg: Phase6Config):
    nifty, gold, cpi_nat = load_macro_series(cfg)

    # Merge once and clean
    df_macro_all = (
        nifty.merge(gold, on="date", how="inner")
        .merge(cpi_nat, on="date", how="inner")
    )
    df_macro_all = (
        df_macro_all
        .dropna(subset=["eq_ret", "gold_ret", "cpi_mom"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if df_macro_all.empty:
        raise ValueError("Merged macro series is empty after dropping NaNs.")

    # Infer windows if not provided (Option C)
    if not cfg.historical_windows:
        cfg.historical_windows = infer_historical_windows(df_macro_all)
    windows = cfg.historical_windows

    df_hh = pd.read_csv(cfg.households_features_path)

    all_ts = []
    for win_name, (start, end) in windows.items():
        try:
            macro_path = build_macro_window(df_macro_all, start, end)
        except ValueError as e:
            print(f"[Phase6][WARN] Skipping window '{win_name}': {e}")
            continue

        df_win = run_backtest_for_window(cfg, win_name, macro_path, df_hh)
        all_ts.append(df_win)
        print(
            f"[Phase6] Backtest window '{win_name}' "
            f"→ {macro_path.shape[0]} months, {df_hh.shape[0]} households"
        )

    if not all_ts:
        raise ValueError("No historical windows could be run (no overlapping macro data).")

    backtest_ts = pd.concat(all_ts, ignore_index=True)
    backtest_ts.to_csv(cfg.backtest_ts_out, index=False)
    print(f"[Phase6] Backtest time series written to {cfg.backtest_ts_out}")

    summary = (
        backtest_ts.groupby("window")
        .agg(
            mean_default_risk_share=("default_high_risk", "mean"),
            mean_total_burden=("total_burden", "mean"),
            mean_drawdown=("net_worth_drawdown_pct", "mean"),
        )
        .reset_index()
    )

    # Auto-detect baseline window: first one whose name starts with 'baseline'
    baseline_window = next(
        (w for w in summary["window"].unique() if str(w).startswith("baseline")),
        None,
    )
    if baseline_window is not None:
        baseline_share = float(
            summary.loc[
                summary["window"] == baseline_window, "mean_default_risk_share"
            ].iloc[0]
        )
        summary["higher_default_risk_vs_baseline"] = (
            summary["mean_default_risk_share"] > baseline_share + 1e-6
        )
    else:
        summary["higher_default_risk_vs_baseline"] = np.nan

    summary.to_csv(cfg.backtest_summary_out, index=False)
    print(f"[Phase6] Backtest summary written to {cfg.backtest_summary_out}")

    print("\n[Phase6] Backtest default-risk shares:")
    for _, r in summary.iterrows():
        print(
            f"  {r['window']}: "
            f"default-risk share = {r['mean_default_risk_share']:.2%}, "
            f"higher vs baseline? {r['higher_default_risk_vs_baseline']}"
        )

    return backtest_ts, summary


# --------------------------- Main ---------------------------

def main():
    cfg = Phase6Config()

    print("========== Phase 6 — Validation & Backtests ==========\n")

    # 1) HCES alignment
    print("[1/3] HCES alignment vs Annex-II priors")
    try:
        compute_hces_alignment(cfg)
    except Exception as e:
        print(f"[Phase6][WARN] HCES alignment step failed or incomplete: {e}")

    # 2) KPI sanity + accept/reject gates
    print("\n[2/3] KPI sanity & accept/reject gates")
    try:
        run_kpi_accept_reject(cfg)
    except Exception as e:
        print(f"[Phase6][WARN] KPI sanity step failed: {e}")

    # 3) Historical backtests
    print("\n[3/3] Historical backtests (auto baseline + 2020/2022 if available)")
    try:
        run_historical_backtests(cfg)
    except Exception as e:
        print(f"[Phase6][WARN] Historical backtest step failed: {e}")

    print("\n========== Phase 6 complete (see CSV outputs in data_interim/) ==========")


if __name__ == "__main__":
    main()
