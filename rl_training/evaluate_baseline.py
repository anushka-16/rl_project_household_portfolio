"""
evaluate_baseline.py

Evaluate a simple baseline portfolio strategy (fixed equal weights)
on HouseholdPortfolioEnv over multiple episodes.

Run from Project root as:
    python -m rl_training.evaluate_baseline
"""

import os
from typing import List, Dict

import numpy as np
import pandas as pd

from rl_env.household_portfolio_env import HouseholdPortfolioEnv

# ---------------------- Config ----------------------

SCENARIO_KEY = "S2"        # baseline scenario
MAX_MONTHS = 120
TRANSACTION_COST_BPS = 10.0
RISK_AVERSION = 3.0

N_EPISODES = 50

OUT_DIR = os.path.join("outputs", "rl_eval")
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_PREFIX = "baseline_equal_weight"   # used in filenames


# ---------------------- Helpers ----------------------

def run_one_episode(env: HouseholdPortfolioEnv, seed: int) -> Dict:
    """
    Run one eval episode with a fixed equal-weight strategy.
    Returns dict with episode-level metrics plus full path.
    """
    reset_out = env.reset(seed=seed)
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, _ = reset_out
    else:
        obs = reset_out

    done = False
    total_reward = 0.0
    step = 0
    values = []
    net_rets = []
    equity_rets = []
    gold_rets = []
    weights_list = []

    # Fixed equal weights [equity, cash, gold]
    fixed_weights = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    while not done:
        # Environment expects an action, but we ignore obs and always use fixed weights
        action = fixed_weights
        step_out = env.step(action)

        # Handle gymnasium vs gym signatures
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        total_reward += float(reward)
        step += 1

        values.append(float(info.get("portfolio_value", np.nan)))
        net_rets.append(float(info.get("net_return", np.nan)))
        equity_rets.append(float(info.get("equity_ret", np.nan)))
        gold_rets.append(float(info.get("gold_ret", np.nan)))
        # env.weights reflects the actual weights after action
        weights_list.append(env.weights.copy())

    final_value = values[-1] if values else np.nan
    net_rets_arr = np.array(net_rets, dtype=float)
    if np.std(net_rets_arr) > 1e-8:
        sharpe_monthly = np.mean(net_rets_arr) / np.std(net_rets_arr)
    else:
        sharpe_monthly = np.nan

    return {
        "total_reward": total_reward,
        "final_value": final_value,
        "steps": step,
        "sharpe_monthly": sharpe_monthly,
        "values": values,
        "net_rets": net_rets,
        "equity_rets": equity_rets,
        "gold_rets": gold_rets,
        "weights": weights_list,
    }


def main():
    episode_summaries: List[Dict] = []
    all_paths_rows = []

    for ep in range(N_EPISODES):
        env = HouseholdPortfolioEnv(
            scenario_key=SCENARIO_KEY,
            max_months=MAX_MONTHS,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            risk_aversion=RISK_AVERSION,
            drawdown_aversion=1.5,
            concentration_aversion=0.05,
            seed=3000 + ep,
        )

        result = run_one_episode(env, seed=4000 + ep)
        env.close()

        episode_summaries.append(
            {
                "episode": ep,
                "total_reward": result["total_reward"],
                "final_value": result["final_value"],
                "steps": result["steps"],
                "sharpe_monthly": result["sharpe_monthly"],
            }
        )

        for t, (v, r, er, gr, w) in enumerate(
            zip(
                result["values"],
                result["net_rets"],
                result["equity_rets"],
                result["gold_rets"],
                result["weights"],
            )
        ):
            all_paths_rows.append(
                {
                    "episode": ep,
                    "t": t,
                    "portfolio_value": v,
                    "net_return": r,
                    "equity_ret": er,
                    "gold_ret": gr,
                    "w_equity": float(w[0]),
                    "w_cash": float(w[1]),
                    "w_gold": float(w[2]),
                }
            )

    # Episode-level summary
    df_summary = pd.DataFrame(episode_summaries)
    summary_path = os.path.join(OUT_DIR, f"{BASELINE_PREFIX}_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[BASELINE] Episode summary saved to: {summary_path}")

    # Path-level details
    df_paths = pd.DataFrame(all_paths_rows)
    paths_path = os.path.join(OUT_DIR, f"{BASELINE_PREFIX}_paths.csv")
    df_paths.to_csv(paths_path, index=False)
    print(f"[BASELINE] Per-step paths saved to: {paths_path}")

    print("\n[BASELINE] Aggregate stats over episodes:")
    print(df_summary[["final_value", "total_reward", "sharpe_monthly"]].describe())


if __name__ == "__main__":
    main()
