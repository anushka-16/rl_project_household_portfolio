# phase5_kpis.py
"""
Phase 5 – KPIs, thresholds, narratives

Reads:
  data_interim/phase4_mc_household_paths.csv
  data_interim/households_features.csv

Outputs:
  outputs/kpis_S0.csv ... kpis_Sn.csv   (per scenario, tidy)
  outputs/narratives.jsonl              (one JSON record per scenario)

Core KPIs (per household, per month):
  EMI-to-Income          = emi_total / income_total
  Expense-to-Income      = expense_total / income_total
  Total Burden           = (emi_total + expense_total) / income_total
  Liquidity Runway (m)   = bank_savings / max(1, (expense + emi - income)+)
                           where (x)+ = max(x, 0); we only use positive deficit.
  Net-Worth Drawdown (%) = % drop vs t0 net_assets (month_index=1)

Default Risk Heuristic (per household-path):
  High risk if:
    (Total Burden > 1.1 for >=3 consecutive months) OR
    (Liquidity Runway < 2 months in any month)

Insurance adequacy (per household, time-invariant):
  Health: sum_insured_health vs HEALTH_MULTIPLIER * annual_income
  Life:   sum_insured_life   vs LIFE_MULTIPLIER   * annual_income

Note:
  Column names for sum insured are configurable; if not found, health/life
  adequacy are left as NaN.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------- Paths ---------------------------

ROOT = os.path.abspath(os.path.dirname(__file__))
DATAI = os.path.join(ROOT, "data_interim")
OUT_DIR = os.path.join(ROOT, "outputs")

IN_HH_PATHS = os.path.join(DATAI, "phase4_mc_household_paths.csv")
IN_HH_BASE = os.path.join(DATAI, "households_features.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------- Config ---------------------------

# Maximum number of scenario KPI rows to keep in memory at once when writing per-scenario CSVs
# (we keep everything in memory here; tune if needed)
HEALTH_MULTIPLIER = 2.5   # 2–3× annual income benchmark for health cover
LIFE_MULTIPLIER = 10.0    # 10× annual income benchmark for life cover

# Try these column names for health / life sum insured; adjust if your features use different ones
HEALTH_SI_COLS = ["health_sum_insured", "sum_insured_health", "health_cover_si"]
LIFE_SI_COLS = ["life_sum_insured", "term_sum_insured", "term_cover_si"]


# --------------------------- Helpers ---------------------------

def _find_first_existing_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_insurance_adequacy(hh_base: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-household insurance adequacy flags (no time dimension).

    Uses:
      - monthly income_total from households_features if present,
        else will be filled later from MC t0 income if needed.
    """
    df = hh_base.copy()

    # Ensure lead_id as string
    if "lead_id" not in df.columns:
        raise ValueError("households_features.csv must contain 'lead_id' column")
    df["lead_id"] = df["lead_id"].astype(str)

    # Monthly income: prefer 'income_total' if present
    if "income_total" in df.columns:
        monthly_income = df["income_total"].astype(float).clip(lower=0)
    else:
        # Placeholder; will be replaced if we merge income from MC later.
        monthly_income = pd.Series(0.0, index=df.index)

    annual_income = monthly_income * 12.0

    # Detect health / life SI columns
    health_col = _find_first_existing_col(df, HEALTH_SI_COLS)
    life_col = _find_first_existing_col(df, LIFE_SI_COLS)

    if health_col is None:
        warnings.warn(
            f"No health sum insured column found among {HEALTH_SI_COLS}; "
            "health_cover_adequate will be NaN.",
            RuntimeWarning,
        )
        df["health_cover_adequate"] = np.nan
    else:
        df[health_col] = pd.to_numeric(df[health_col], errors="coerce").clip(lower=0)
        required_health = HEALTH_MULTIPLIER * annual_income
        df["health_cover_adequate"] = np.where(
            (required_health > 0) & (df[health_col] >= required_health),
            1,
            0,
        )

    if life_col is None:
        warnings.warn(
            f"No life sum insured column found among {LIFE_SI_COLS}; "
            "life_cover_adequate will be NaN.",
            RuntimeWarning,
        )
        df["life_cover_adequate"] = np.nan
    else:
        df[life_col] = pd.to_numeric(df[life_col], errors="coerce").clip(lower=0)
        required_life = LIFE_MULTIPLIER * annual_income
        df["life_cover_adequate"] = np.where(
            (required_life > 0) & (df[life_col] >= required_life),
            1,
            0,
        )

    return df[["lead_id", "health_cover_adequate", "life_cover_adequate"]]


def add_kpis(hh_paths: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-row KPIs (per household, per path, per month)
    to the MC household paths table.
    """
    df = hh_paths.copy()

    # Cast numeric
    for col in [
        "income_total",
        "expense_total",
        "emi_total",
        "bank_savings",
        "net_assets",
    ]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from phase4_mc_household_paths.csv")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Basic ratios – handle division by zero safely
    income = df["income_total"].replace(0, np.nan)

    df["emi_to_income"] = df["emi_total"] / income
    df["expense_to_income"] = df["expense_total"] / income
    df["total_burden"] = (df["emi_total"] + df["expense_total"]) / income

    # Liquidity runway (months)
    deficit = (df["expense_total"] + df["emi_total"] - df["income_total"]).clip(lower=0.0)
    denom = deficit.where(deficit > 0.0, 1.0)  # if no deficit, use 1 to avoid div-by-0
    df["liquidity_runway_months"] = df["bank_savings"] / denom

    # Net-worth drawdown vs t0 (month_index = 1) for each (scenario, path_id, lead_id)
    df = df.sort_values(["scenario", "path_id", "lead_id", "month_index"])
    group_keys = ["scenario", "path_id", "lead_id"]

    net0 = df.groupby(group_keys)["net_assets"].transform("first")
    df["net_worth_change_pct"] = np.where(
        net0 > 0,
        (df["net_assets"] - net0) / net0 * 100.0,
        np.nan,
    )
    df["net_worth_drawdown_pct"] = np.where(
        net0 > 0,
        np.maximum(0.0, (net0 - df["net_assets"]) / net0 * 100.0),
        np.nan,
    )

    # Row-level flags used for the default risk heuristic
    df["high_burden_flag"] = df["total_burden"] > 1.1
    df["short_runway_flag"] = df["liquidity_runway_months"] < 2.0

    return df


def _max_consecutive_true(arr: np.ndarray) -> int:
    """
    Compute maximum run length of consecutive True in a boolean array.
    """
    if arr.size == 0:
        return 0
    # Convert True/False to 1/0
    x = arr.astype(int)
    # Where zero, reset run; cumulative sum trick
    # e.g., [1,1,0,1] -> [1,2,0,1]; max is 2
    # But we need to break at zeros, so we use masked cumsum
    runs = np.zeros_like(x)
    run = 0
    for i, v in enumerate(x):
        if v == 1:
            run += 1
        else:
            run = 0
        runs[i] = run
    return int(runs.max())


def compute_default_risk_flags(df_kpis: pd.DataFrame) -> pd.DataFrame:
    """
    Compute default risk heuristic at path level, then attach back.

    High risk if:
      (Total Burden > 1.1 for >=3 consecutive months) OR
      (Runway < 2 months in any month)
    """
    df = df_kpis.copy()
    group_keys = ["scenario", "path_id", "lead_id"]

    def _path_risk(g: pd.DataFrame) -> pd.Series:
        # g is already sorted by month_index from previous step
        hb = g["high_burden_flag"].to_numpy()
        sr = g["short_runway_flag"].to_numpy()

        max_run = _max_consecutive_true(hb)
        rule1 = max_run >= 3
        rule2 = bool(np.any(sr))

        default_high_risk = rule1 or rule2
        return pd.Series(
            {
                "max_consec_high_burden": max_run,
                "has_short_runway": int(rule2),
                "default_high_risk": int(default_high_risk),
            }
        )

    path_risk = df.groupby(group_keys, sort=False).apply(_path_risk).reset_index()

    # Attach back to each row
    df = df.merge(path_risk, on=group_keys, how="left")

    return df


def attach_insurance_flags(df_kpis: pd.DataFrame, ins_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge per-household insurance adequacy flags onto the KPI table.
    """
    out = df_kpis.merge(ins_df, on="lead_id", how="left")
    return out


def write_per_scenario_kpis(df_all: pd.DataFrame) -> list[str]:
    """
    Split KPI table by scenario and write one CSV per scenario.
    Returns list of paths written.
    """
    written_paths = []
    scenarios = sorted(df_all["scenario"].unique())

    for scen in scenarios:
        df_s = df_all[df_all["scenario"] == scen].copy()
        out_path = os.path.join(OUT_DIR, f"kpis_{scen}.csv")
        df_s.to_csv(out_path, index=False)
        written_paths.append(out_path)
        print(f"[Phase 5] Wrote KPIs for {scen} → {out_path}")
    return written_paths


def build_narratives(df_all: pd.DataFrame, out_path: str) -> None:
    """
    Build simple scenario-level narratives based on aggregate KPIs
    and default risk flags. Writes a JSONL file, one record per scenario.
    """
    records = []
    scenarios = (
        df_all[["scenario", "scenario_name"]]
        .drop_duplicates()
        .sort_values("scenario")
        .to_dict(orient="records")
    )

    for s in scenarios:
        scen = s["scenario"]
        scen_name = s["scenario_name"]

        df_s = df_all[df_all["scenario"] == scen]

        # Path-level risk probability per household
        path_risk = (
            df_s[["scenario", "path_id", "lead_id", "default_high_risk"]]
            .drop_duplicates()
        )
        # Fraction of (lead_id, path_id) that are high risk
        frac_high_risk = float(path_risk["default_high_risk"].mean())

        # Median burden & runway at month 6, if exists
        df_m6 = df_s[df_s["month_index"] == 6]
        med_burden_m6 = float(df_m6["total_burden"].median()) if not df_m6.empty else float("nan")
        med_runway_m6 = float(df_m6["liquidity_runway_months"].median()) if not df_m6.empty else float("nan")

        # Simple template-based narrative
        if frac_high_risk < 0.1:
            narrative = (
                f"In the {scen_name} scenario, most households remain comfortable. "
                f"Typical total burden around month 6 is about {med_burden_m6:.2f}× income, "
                f"and the median liquidity runway is roughly {med_runway_m6:.1f} months. "
                f"Households can focus on incremental savings and small portfolio rebalancing "
                f"to preserve a runway above 6 months."
            )
        elif frac_high_risk < 0.3:
            narrative = (
                f"In the {scen_name} scenario, some households start to show financial strain. "
                f"About {frac_high_risk:.0%} of simulated household-paths enter a high-risk zone "
                f"either due to rising burden or falling liquidity. "
                f"By month 6, the median total burden is ~{med_burden_m6:.2f}× income and "
                f"liquidity runway is around {med_runway_m6:.1f} months. "
                f"Reducing discretionary expenses and tilting 3–5% from equity towards safer assets "
                f"can help preserve a buffer above 3–4 months of expenses."
            )
        else:
            narrative = (
                f"In the {scen_name} scenario, a large share of households face elevated stress. "
                f"Roughly {frac_high_risk:.0%} of household-paths breach the high-risk thresholds "
                f"(burden > 1.1× income for multiple months or runway < 2 months). "
                f"Median liquidity runway by month 6 is only about {med_runway_m6:.1f} months. "
                f"Households may need to aggressively cut non-essential spending, avoid new debt, and "
                f"consider shifting 5–10% of their portfolio from volatile assets into cash or short-duration debt "
                f"to stabilise their runway."
            )

        rec = {
            "scenario": scen,
            "scenario_name": scen_name,
            "n_households": int(df_s["lead_id"].nunique()),
            "n_paths": int(df_s["path_id"].nunique()),
            "frac_high_risk": frac_high_risk,
            "median_total_burden_m6": med_burden_m6,
            "median_liquidity_runway_m6": med_runway_m6,
            "narrative": narrative,
        }
        records.append(rec)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Phase 5] Wrote narratives → {out_path}")


# --------------------------- Main ---------------------------

def main():
    if not os.path.exists(IN_HH_PATHS):
        raise FileNotFoundError(f"Missing Phase 4 household paths file: {IN_HH_PATHS}")
    if not os.path.exists(IN_HH_BASE):
        raise FileNotFoundError(f"Missing households_features file: {IN_HH_BASE}")

    print("[Phase 5] Loading households_features...")
    hh_base = pd.read_csv(IN_HH_BASE)

    print("[Phase 5] Computing insurance adequacy...")
    ins_df = compute_insurance_adequacy(hh_base)

    print("[Phase 5] Loading Phase 4 household paths (subsetting columns)...")
    usecols = [
        "scenario",
        "scenario_name",
        "path_id",
        "month_index",
        "lead_id",
        "income_total",
        "expense_total",
        "emi_total",
        "bank_savings",
        "net_assets",
    ]
    hh_paths = pd.read_csv(IN_HH_PATHS, usecols=usecols)
    hh_paths["lead_id"] = hh_paths["lead_id"].astype(str)

    print("[Phase 5] Adding per-row KPIs...")
    df_kpis = add_kpis(hh_paths)

    print("[Phase 5] Computing default risk heuristic (per path)...")
    df_kpis = compute_default_risk_flags(df_kpis)

    print("[Phase 5] Attaching insurance adequacy flags...")
    df_kpis = attach_insurance_flags(df_kpis, ins_df)

    print("[Phase 5] Writing per-scenario KPI CSVs...")
    kpi_files = write_per_scenario_kpis(df_kpis)

    # Narratives
    narratives_path = os.path.join(OUT_DIR, "narratives.jsonl")
    print("[Phase 5] Building narratives...")
    build_narratives(df_kpis, narratives_path)

    print("[Phase 5] Done.")
    print("KPI files:")
    for p in kpi_files:
        print("  ", p)
    print("Narratives:", narratives_path)


if __name__ == "__main__":
    main()
