# monte_carlo.py
"""
Phase 4 – Scenario design & Monte-Carlo

Reads:
  data_interim/households_features.csv
  data_interim/nifty_monthly.csv
  data_interim/gold_monthly.csv
  data_interim/cpi_long.csv
  data_interim/interest_benchmarks.csv
  configs/scenarios.yaml

Outputs:
  data_interim/phase4_mc_household_paths.csv
  data_interim/phase4_mc_scenario_returns.csv

For each scenario, path and month we generate household-level time series:
  expense_total(t), emi_total(t), income_total(t),
  bank_savings(t), equity_value(t), gold_mark_to_market(t),
  debt_funds_value(t), net_assets(t), liquidity_ratio(t)
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------- Paths ---------------------------

ROOT = os.path.abspath(os.path.dirname(__file__))
DATAI = os.path.join(ROOT, "data_interim")
CONFIG_DIR = os.path.join(ROOT, "configs")

IN_HH = os.path.join(DATAI, "households_features.csv")
IN_NIFTY = os.path.join(DATAI, "nifty_monthly.csv")
IN_GOLD = os.path.join(DATAI, "gold_monthly.csv")
IN_CPI = os.path.join(DATAI, "cpi_long.csv")
IN_RATES = os.path.join(DATAI, "interest_benchmarks.csv")

OUT_PATHS = os.path.join(DATAI, "phase4_mc_household_paths.csv")
OUT_RETURNS = os.path.join(DATAI, "phase4_mc_scenario_returns.csv")
SCEN_YAML = os.path.join(CONFIG_DIR, "scenarios.yaml")

os.makedirs(DATAI, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


# --------------------------- Helper dataclasses ---------------------------

@dataclass
class MCConfig:
    random_seed: int
    horizon_months: int
    block_size_months: int
    n_paths: int
    correlate_equity_down_with_cpi: bool
    correlate_rate_hikes_with_equity_vol: bool


@dataclass
class ScenarioSpec:
    key: str
    name: str
    cpi_annual_overlay: float
    rate_total_bps: float
    rate_horizon_months: int
    equity_drawdown_total: float
    equity_drawdown_months: int
    gold_total_return_overlay: float
    description: str = ""


# --------------------------- Load config ---------------------------

def load_config() -> Tuple[MCConfig, Dict[str, ScenarioSpec]]:
    if not os.path.exists(SCEN_YAML):
        raise FileNotFoundError(f"Missing scenarios config: {SCEN_YAML}")

    with open(SCEN_YAML, "r") as f:
        cfg = yaml.safe_load(f)

    mc_cfg = MCConfig(
        random_seed=int(cfg.get("random_seed", 123)),
        horizon_months=int(cfg.get("horizon_months", 120)),
        block_size_months=int(cfg.get("block_size_months", 6)),
        n_paths=int(cfg.get("n_paths", 1000)),
        correlate_equity_down_with_cpi=bool(cfg.get("correlate_equity_down_with_cpi", True)),
        correlate_rate_hikes_with_equity_vol=bool(cfg.get("correlate_rate_hikes_with_equity_vol", True)),
    )

    scen_specs: Dict[str, ScenarioSpec] = {}
    for key, s in cfg.get("scenarios", {}).items():
        scen_specs[key] = ScenarioSpec(
            key=key,
            name=str(s.get("name", key)),
            description=str(s.get("description", "")),
            cpi_annual_overlay=float(s.get("cpi_annual_overlay", 0.0)),
            rate_total_bps=float(s.get("rate_total_bps", 0.0)),
            rate_horizon_months=int(s.get("rate_horizon_months", 12)),
            equity_drawdown_total=float(s.get("equity_drawdown_total", 0.0)),
            equity_drawdown_months=int(s.get("equity_drawdown_months", 6)),
            gold_total_return_overlay=float(s.get("gold_total_return_overlay", 0.0)),
        )
    return mc_cfg, scen_specs


# --------------------------- Macro series loaders ---------------------------

def load_macro_series():
    """
    Load monthly macro series for Monte Carlo:
      - NIFTY total return (nifty_monthly.csv)
      - Gold price (gold_monthly.csv)
      - CPI (cpi_long.csv)
      - Policy / lending rate (interest_benchmarks.csv → policy_rate)
    Returns a dict of DataFrames indexed by date (month-end or month-start).
    """
    date_col = "date"

    # ----- Equity (NIFTY) -----
    nifty_path = IN_NIFTY
    if not os.path.exists(nifty_path):
        raise FileNotFoundError(f"Missing equity file: {nifty_path}")
    eq = pd.read_csv(nifty_path)

    if date_col not in eq.columns:
        raise ValueError(f"{nifty_path} must contain a 'date' column")
    eq[date_col] = pd.to_datetime(eq[date_col], dayfirst=True, errors="coerce")
    eq = eq.sort_values(date_col)

    # Detect price/level column – supports your 'tri' column and others
    price_col = None
    for cand in ["tr_index", "tri", "close", "price"]:
        for col in eq.columns:
            if col.lower() == cand:
                price_col = col
                break
        if price_col is not None:
            break

    if price_col is None:
        raise ValueError(
            f"{nifty_path} must contain one of tr_index/tri/close/price columns "
            f"(found columns: {list(eq.columns)})"
        )

    # Use existing monthly return column if present, else derive from price
    if "ret_nifty_m" in eq.columns:
        eq["ret"] = pd.to_numeric(eq["ret_nifty_m"], errors="coerce")
    else:
        eq[price_col] = pd.to_numeric(eq[price_col], errors="coerce")
        eq["ret"] = eq[price_col].pct_change()

    eq = eq.dropna(subset=["ret"]).set_index(date_col)
    if eq.index.has_duplicates:
        eq = eq[~eq.index.duplicated(keep="last")]

    # ----- Gold -----
    gold_path = IN_GOLD
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"Missing gold file: {gold_path}")
    gold = pd.read_csv(gold_path)
    if date_col not in gold.columns:
        raise ValueError(f"{gold_path} must contain a 'date' column")
    gold[date_col] = pd.to_datetime(gold[date_col], dayfirst=True, errors="coerce")
    gold = gold.sort_values(date_col)

    g_price_col = None
    for cand in ["price", "close", "value"]:
        for col in gold.columns:
            if col.lower() == cand:
                g_price_col = col
                break
        if g_price_col is not None:
            break
    if g_price_col is None:
        raise ValueError(f"{gold_path} must contain a price/close/value column")

    gold[g_price_col] = pd.to_numeric(gold[g_price_col], errors="coerce")
    gold["ret"] = gold[g_price_col].pct_change()
    gold = gold.dropna(subset=["ret"]).set_index(date_col)
    if gold.index.has_duplicates:
        gold = gold[~gold.index.duplicated(keep="last")]

    # ----- CPI -----
    cpi_path = IN_CPI
    if not os.path.exists(cpi_path):
        raise FileNotFoundError(f"Missing CPI file: {cpi_path}")
    cpi = pd.read_csv(cpi_path)
    if "date" not in cpi.columns:
        raise ValueError(f"{cpi_path} must contain a 'date' column")
    cpi["date"] = pd.to_datetime(cpi["date"], dayfirst=True, errors="coerce")
    cpi = cpi.sort_values("date").set_index("date")

    cpi_level_col = None
    for c in ["cpi_level", "cpi", "index"]:
        if c in cpi.columns:
            cpi_level_col = c
            break
    if cpi_level_col is None:
        num_cols = cpi.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            raise ValueError(f"{cpi_path} must have at least one numeric CPI column")
        cpi_level_col = num_cols[0]

    cpi[cpi_level_col] = pd.to_numeric(cpi[cpi_level_col], errors="coerce")
    cpi["cpi_mom"] = cpi[cpi_level_col].pct_change()
    if cpi.index.has_duplicates:
        cpi = cpi[~cpi.index.duplicated(keep="last")]

    # ----- Rates (interest_benchmarks.csv → policy_rate) -----
    rates_path = IN_RATES
    if not os.path.exists(rates_path):
        raise FileNotFoundError(f"Missing rates file: {rates_path}")
    rates = pd.read_csv(rates_path)

    if "date" not in rates.columns:
        raise ValueError(f"{rates_path} must contain a 'date' column")

    # assume %Y-%m-%d
    rates["date"] = pd.to_datetime(rates["date"], dayfirst=False, errors="coerce")

    # Choose benchmark: MCLR_1_3 or generic MCLR
    if "benchmark_clean" in rates.columns:
        mask = rates["benchmark_clean"].eq("MCLR_1_3")
    else:
        mask = pd.Series(False, index=rates.index)

    if not mask.any() and "benchmark_group" in rates.columns:
        mask = rates["benchmark_group"].eq("MCLR")

    rates_work = rates.loc[mask, ["date", "value"]].copy()
    if rates_work.empty:
        raise ValueError(
            f"{rates_path} does not contain usable MCLR_1_3 or MCLR rows; "
            f"found benchmarks: "
            f"{sorted(set(rates.get('benchmark_clean', [])) | set(rates.get('benchmark_group', [])))}"
        )

    rates_work = rates_work.sort_values("date").dropna(subset=["value"])
    rates_work["policy_rate"] = pd.to_numeric(rates_work["value"], errors="coerce")
    rates_work = rates_work.dropna(subset=["policy_rate"])

    rates_work = rates_work.set_index("date")
    if rates_work.index.has_duplicates:
        rates_work = rates_work[~rates_work.index.duplicated(keep="last")]

    # level in bps
    rates_work["rate_level_bps"] = rates_work["policy_rate"] * 100.0
    # month-to-month change in bps
    rates_work["rate_d_bps"] = rates_work["policy_rate"].diff() * 100.0
    rates_work = rates_work.dropna(subset=["rate_d_bps"])

    return {
        "equity": eq,          # index → ret
        "gold": gold,          # index → ret
        "cpi": cpi,            # index → cpi_level, cpi_mom, ...
        "rates": rates_work,   # index → policy_rate, rate_level_bps, rate_d_bps
    }


# --------------------------- Block bootstrap ---------------------------

def block_bootstrap_paths(
    macro: Dict[str, pd.DataFrame],
    horizon: int,
    block_size: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    macro: dict with keys equity, gold, cpi, rates
    Returns dict of arrays with shape (n_paths, horizon)
    """
    # Align by date
    df = pd.DataFrame({
        "equity_ret": macro["equity"]["ret"],
        "gold_ret": macro["gold"]["ret"],
        "cpi_mom": macro["cpi"]["cpi_mom"],
        "rate_d_bps": macro["rates"]["rate_d_bps"],
    }).dropna()

    if df.empty:
        raise ValueError("No overlapping macro history after alignment; check input series.")

    cols = ["equity_ret", "gold_ret", "cpi_mom", "rate_d_bps"]
    X = df[cols].values
    T_hist = X.shape[0]

    # adapt block_size if history is shorter
    if T_hist < block_size:
        warnings.warn(
            f"Configured block_size={block_size} is larger than available history={T_hist}. "
            f"Using block_size={T_hist} instead.",
            RuntimeWarning,
        )
        block_size = T_hist

    if T_hist <= 0 or block_size <= 0:
        raise ValueError("Insufficient history length for block bootstrap.")

    n_blocks = int(np.ceil(horizon / block_size))
    max_start = T_hist - block_size

    out = np.zeros((n_paths, horizon, len(cols)), dtype=np.float32)

    for p in range(n_paths):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, max_start + 1)
            blocks.append(X[start: start + block_size])
        path = np.vstack(blocks)[:horizon, :]
        out[p, :, :] = path

    return {
        "equity_ret": out[:, :, 0],
        "gold_ret": out[:, :, 1],
        "cpi_mom": out[:, :, 2],
        "rate_d_bps": out[:, :, 3],
    }


# --------------------------- Correlation hooks ---------------------------

def apply_correlation_hooks(
    paths: Dict[str, np.ndarray],
    cfg: MCConfig,
    rng: np.random.Generator,
) -> None:
    """
    In-place tweaks:
      - equity down months → bump CPI a bit
      - rate hikes → slightly raise equity volatility
    """
    eq = paths["equity_ret"]
    cpi = paths["cpi_mom"]
    rate = paths["rate_d_bps"]

    if cfg.correlate_equity_down_with_cpi:
        # For months where equity return is <-2%, add 0.5 * std(CPI) to cpi_mom
        cpi_std = float(np.nanstd(cpi))
        bump = 0.5 * cpi_std
        mask = eq < -0.02
        cpi[mask] += bump

    if cfg.correlate_rate_hikes_with_equity_vol:
        # If rate_d_bps > 0, multiply equity return by a factor >1
        factor = 1.0 + 0.3 * (rate / 100.0)
        paths["equity_ret"] = eq * factor


# --------------------------- Overlay scenario shocks ---------------------------

def overlay_scenario_on_paths(
    base_paths: Dict[str, np.ndarray],
    scen: ScenarioSpec,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """
    Returns a NEW dict with scenario-specific overlays applied.
    """
    paths = {k: v.copy() for k, v in base_paths.items()}

    # CPI overlay: annual → monthly additive drift on cpi_mom
    if scen.cpi_annual_overlay != 0.0:
        monthly = scen.cpi_annual_overlay / 12.0
        paths["cpi_mom"] += monthly

    # Rate overlay: spread rate_total_bps over rate_horizon_months
    if scen.rate_total_bps != 0.0 and scen.rate_horizon_months > 0:
        per_month = scen.rate_total_bps / scen.rate_horizon_months
        overlay = np.zeros((paths["rate_d_bps"].shape[0], horizon), dtype=np.float32)
        m = min(horizon, scen.rate_horizon_months)
        overlay[:, :m] = per_month
        paths["rate_d_bps"] += overlay

    # Equity drawdown: extra total return over first equity_drawdown_months
    if scen.equity_drawdown_total != 0.0 and scen.equity_drawdown_months > 0:
        per_month = scen.equity_drawdown_total / scen.equity_drawdown_months
        overlay = np.zeros_like(paths["equity_ret"], dtype=np.float32)
        m = min(horizon, scen.equity_drawdown_months)
        overlay[:, :m] = per_month
        paths["equity_ret"] += overlay

    # Gold overlay: distribute total overlay over horizon equally
    if scen.gold_total_return_overlay != 0.0:
        per_month = scen.gold_total_return_overlay / horizon
        paths["gold_ret"] += per_month

    return paths


# --------------------------- Household simulation ---------------------------

def simulate_households_for_scenario(
    scen_key: str,
    scen: ScenarioSpec,
    hh: pd.DataFrame,
    paths: Dict[str, np.ndarray],
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      hh_paths: long table of household metrics
      scen_returns: per-path macro paths for this scenario

    Vectorised version (no nested DataFrame loops), so it’s fast even for
    480-month horizons and hundreds of paths.
    """
    n_paths, T = paths["equity_ret"].shape
    assert T == horizon

    df0 = hh.copy()
    n_hh = df0.shape[0]

    # --- ensure columns exist & are numeric ---
    for col in [
        "income_total",
        "expense_total",
        "emi_total",
        "bank_savings",
        "equity_value",
        "debt_funds_value",
        "gold_gms",
        "gold_price_per_g_t0",
        "gold_mark_to_market_t0",
    ]:
        if col not in df0.columns:
            df0[col] = 0.0
        df0[col] = df0[col].fillna(0.0).astype(float)

    if df0["gold_mark_to_market_t0"].isna().all():
        df0["gold_mark_to_market_t0"] = (
            df0["gold_gms"] * df0["gold_price_per_g_t0"]
        )

    base_income = df0["income_total"].values.astype(np.float32)
    base_exp = df0["expense_total"].values.astype(np.float32)
    base_emi = df0["emi_total"].values.astype(np.float32)
    base_cash = df0["bank_savings"].values.astype(np.float32)
    base_equity = df0["equity_value"].values.astype(np.float32)
    base_debt = df0["debt_funds_value"].values.astype(np.float32)
    base_gold = df0["gold_mark_to_market_t0"].values.astype(np.float32)

    # shape: (n_paths, horizon, n_hh)
    income_paths = np.repeat(base_income[None, None, :], n_paths, axis=0)
    income_paths = np.repeat(income_paths, horizon, axis=1).astype(np.float32)

    expense_paths = np.zeros((n_paths, horizon, n_hh), dtype=np.float32)
    emi_paths = np.zeros_like(expense_paths)
    cash_paths = np.zeros_like(expense_paths)
    equity_paths = np.zeros_like(expense_paths)
    debt_paths = np.zeros_like(expense_paths)
    gold_paths = np.zeros_like(expense_paths)
    net_assets_paths = np.zeros_like(expense_paths)
    liquidity_ratio_paths = np.zeros_like(expense_paths)

    expense_prev = np.tile(base_exp, (n_paths, 1)).astype(np.float32)
    emi_prev = np.tile(base_emi, (n_paths, 1)).astype(np.float32)
    cash_prev = np.tile(base_cash, (n_paths, 1)).astype(np.float32)
    equity_prev = np.tile(base_equity, (n_paths, 1)).astype(np.float32)
    debt_prev = np.tile(base_debt, (n_paths, 1)).astype(np.float32)
    gold_prev = np.tile(base_gold, (n_paths, 1)).astype(np.float32)

    cpi_paths = paths["cpi_mom"].astype(np.float32)
    eq_ret_paths = paths["equity_ret"].astype(np.float32)
    gold_ret_paths = paths["gold_ret"].astype(np.float32)
    rate_d_paths = paths["rate_d_bps"].astype(np.float32)

    # ---------- main monthly loop (vectorised over paths & households) ----------
    for t in range(horizon):
        cpi_m = cpi_paths[:, t][:, None]
        eq_r = eq_ret_paths[:, t][:, None]
        g_r = gold_ret_paths[:, t][:, None]

        expense_t = expense_prev * (1.0 + cpi_m)
        emi_t = emi_prev  # hook to rate_d_paths if needed

        equity_t = equity_prev * (1.0 + eq_r)
        gold_t = gold_prev * (1.0 + g_r)

        dy = rate_d_paths[:, t][:, None] / 10000.0
        DURATION_DEBT = 3.0
        debt_t = debt_prev * (1.0 - DURATION_DEBT * dy)
        debt_t = np.clip(debt_t, 0.0, None)

        income_t = income_paths[:, t, :]
        disposable_t = income_t - (expense_t + emi_t)
        cash_t = cash_prev + disposable_t
        cash_t = np.clip(cash_t, 0.0, None)

        net_assets_t = cash_t + equity_t + gold_t + debt_t
        denom = expense_t + emi_t
        with np.errstate(divide="ignore", invalid="ignore"):
            liq_t = np.where(denom > 0, cash_t / denom, np.nan).astype(np.float32)

        expense_paths[:, t, :] = expense_t
        emi_paths[:, t, :] = emi_t
        cash_paths[:, t, :] = cash_t
        equity_paths[:, t, :] = equity_t
        debt_paths[:, t, :] = debt_t
        gold_paths[:, t, :] = gold_t
        net_assets_paths[:, t, :] = net_assets_t
        liquidity_ratio_paths[:, t, :] = liq_t

        expense_prev = expense_t
        emi_prev = emi_t
        cash_prev = cash_t
        equity_prev = equity_t
        debt_prev = debt_t
        gold_prev = gold_t

    # ---------- flatten to long table (no Python loops) ----------
    lead_ids = df0["lead_id"].astype(str).values  # shape (n_hh,)
    n_paths_, T_, n_hh_ = expense_paths.shape
    assert n_paths_ == n_paths and T_ == horizon and n_hh_ == n_hh

    path_idx = np.arange(n_paths, dtype=np.int32)
    month_idx = np.arange(horizon, dtype=np.int32)
    hh_idx = np.arange(n_hh, dtype=np.int32)

    p_grid, t_grid, h_grid = np.meshgrid(
        path_idx, month_idx, hh_idx, indexing="ij"
    )

    flat_size = n_paths * horizon * n_hh

    hh_paths = pd.DataFrame({
        "scenario": scen_key,
        "scenario_name": scen.name,
        "path_id": p_grid.reshape(flat_size),
        "month_index": t_grid.reshape(flat_size) + 1,
        "lead_id": lead_ids[h_grid.reshape(flat_size)],
        "income_total": income_paths.reshape(flat_size),
        "expense_total": expense_paths.reshape(flat_size),
        "emi_total": emi_paths.reshape(flat_size),
        "bank_savings": cash_paths.reshape(flat_size),
        "equity_value": equity_paths.reshape(flat_size),
        "debt_funds_value": debt_paths.reshape(flat_size),
        "gold_mark_to_market": gold_paths.reshape(flat_size),
        "net_assets": net_assets_paths.reshape(flat_size),
        "liquidity_ratio": liquidity_ratio_paths.reshape(flat_size),
    })

    # ---------- macro scenario returns table ----------
    months = np.arange(1, horizon + 1, dtype=np.int32)
    path_id_col = np.repeat(np.arange(n_paths, dtype=np.int32), horizon)
    month_col = np.tile(months, n_paths)

    scen_returns = pd.DataFrame({
        "scenario": scen_key,
        "scenario_name": scen.name,
        "path_id": path_id_col,
        "month_index": month_col,
        "equity_ret": eq_ret_paths.reshape(-1),
        "gold_ret": gold_ret_paths.reshape(-1),
        "cpi_mom": cpi_paths.reshape(-1),
        "rate_d_bps": rate_d_paths.reshape(-1),
    })

    return hh_paths, scen_returns

# --------------------------- Main ---------------------------

def main():
    if not os.path.exists(IN_HH):
        raise FileNotFoundError(f"Missing households file: {IN_HH}")

    print("[Phase 4] Loading config & macro series...")
    mc_cfg, scen_specs = load_config()
    macro = load_macro_series()
    hh = pd.read_csv(IN_HH)

    # --------- MEMORY SAFETY: auto-shrink n_paths if too big ----------
    n_hh = hh.shape[0]
    horizon = mc_cfg.horizon_months
    n_paths_orig = mc_cfg.n_paths

    # total "cells" per metric = n_paths * horizon * n_hh
    MAX_CELLS = 5_000_000  # tune if needed
    cells = n_paths_orig * horizon * n_hh
    if cells > MAX_CELLS:
        n_paths_eff = int(MAX_CELLS // (horizon * n_hh))
        if n_paths_eff < 1:
            raise MemoryError(
                f"Even 1 path with horizon={horizon} and households={n_hh} "
                f"would exceed MAX_CELLS={MAX_CELLS}. "
                f"Reduce horizon or number of households."
            )
        warnings.warn(
            f"Requested n_paths={n_paths_orig} with horizon={horizon} and n_households={n_hh} "
            f"would create ~{cells:,} cells per metric, which is too large for memory. "
            f"Using n_paths={n_paths_eff} instead.",
            RuntimeWarning,
        )
        mc_cfg.n_paths = n_paths_eff

    print(
        f"[Phase 4] Using n_paths={mc_cfg.n_paths}, "
        f"horizon_months={mc_cfg.horizon_months}, n_households={n_hh}"
    )

    print("[Phase 4] Building base block-bootstrap paths...")
    rng = np.random.default_rng(mc_cfg.random_seed)
    base_paths = block_bootstrap_paths(
        macro=macro,
        horizon=mc_cfg.horizon_months,
        block_size=mc_cfg.block_size_months,
        n_paths=mc_cfg.n_paths,
        rng=rng,
    )

    apply_correlation_hooks(base_paths, mc_cfg, rng)

    all_hh_paths = []
    all_scen_returns = []

    for key, scen in scen_specs.items():
        print(f"[Phase 4] Simulating scenario {key} – {scen.name}")
        scen_paths = overlay_scenario_on_paths(
            base_paths,
            scen,
            horizon=mc_cfg.horizon_months,
        )
        hh_paths, scen_returns = simulate_households_for_scenario(
            key,
            scen,
            hh,
            scen_paths,
            horizon=mc_cfg.horizon_months,
        )
        all_hh_paths.append(hh_paths)
        all_scen_returns.append(scen_returns)

    hh_out = pd.concat(all_hh_paths, ignore_index=True)
    ret_out = pd.concat(all_scen_returns, ignore_index=True)

    print(f"[Phase 4] Writing household paths → {OUT_PATHS}")
    hh_out.to_csv(OUT_PATHS, index=False)

    print(f"[Phase 4] Writing scenario return paths → {OUT_RETURNS}")
    ret_out.to_csv(OUT_RETURNS, index=False)

    print("[Phase 4] Done.")


if __name__ == "__main__":
    main()
