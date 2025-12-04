"""
household_portfolio_env.py

Gym-style environment for Dynamic Portfolio Optimization
using your Phase 4 Monte Carlo macro simulator.

Assets:
    0: Equity (NIFTY TR)
    1: Cash / Risk-free (proxied from policy rate)
    2: Gold

Each step = 1 month.
Action  = target portfolio weights over 3 assets (continuous, will be normalized).
State   = [equity_ret_t, gold_ret_t, cpi_mom_t, rate_d_bps_t,
           w_equity, w_cash, w_gold,
           portfolio_value, time_frac,
           drawdown]

Reward  = net_portfolio_return
          - risk_aversion * (net_portfolio_return^2)
          - drawdown_aversion * drawdown_t
          - concentration_aversion * concentration_t
          (mean-variance with penalties for drawdown and concentration)

This environment REUSES:
    - monte_carlo.load_config
    - monte_carlo.load_macro_series
    - monte_carlo.block_bootstrap_paths
    - monte_carlo.apply_correlation_hooks
    - monte_carlo.overlay_scenario_on_paths

NOTE:
    Run your training scripts from the Project root so that
    "import monte_carlo" finds monte_carlo.py correctly.
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import gym
    from gym import spaces
    GYM_API = "gym"
except ImportError:
    # fallback to gymnasium if you prefer that
    import gymnasium as gym
    from gymnasium import spaces
    GYM_API = "gymnasium"

# Import Phase 4 simulator pieces
from monte_carlo import (  # type: ignore
    MCConfig,
    ScenarioSpec,
    load_config,
    load_macro_series,
    block_bootstrap_paths,
    apply_correlation_hooks,
    overlay_scenario_on_paths,
)


class HouseholdPortfolioEnv(gym.Env):
    """
    Dynamic portfolio environment on top of Phase 4 Monte Carlo.

    Episodes:
        - Sample one macro Monte Carlo path (equity, gold, CPI, rates)
        - Apply scenario overlay (Base / Moderate / Severe etc.)
        - Agent chooses monthly weights in [Equity, Cash, Gold]

    Observation:
        np.array of shape (10,):
            [0] equity_ret_t          (approx monthly return, e.g. 0.01 = +1%)
            [1] gold_ret_t
            [2] cpi_mom_t             (monthly inflation rate)
            [3] rate_d_bps_t          (change in policy rate in bps)
            [4] w_equity_t
            [5] w_cash_t
            [6] w_gold_t
            [7] portfolio_value_t     (starting at 1.0)
            [8] time_frac_t           (t / horizon)
            [9] drawdown_t            (current % drawdown from running peak)

    Action:
        Box(3,) with values in [0,1]. We renormalise to sum to 1 so that they
        represent portfolio weights.

    Reward:
        r_t = (net_return_t
               - risk_aversion * net_return_t**2
               - drawdown_aversion * drawdown_t
               - concentration_aversion * concentration_t)
        where net_return_t = gross_return_t - transaction_cost(turnover)

        gross_return_t = dot(weights_t, asset_returns_t)

    Termination:
        After horizon_months (or max_months, if provided) steps.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        scenario_key: str = "Base",
        max_months: Optional[int] = None,
        transaction_cost_bps: float = 10.0,
        risk_aversion: float = 3.0,
        drawdown_aversion: float = 1.5,
        concentration_aversion: float = 0.05,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # ---------------- Phase 4 config & macro setup ----------------
        self.mc_cfg, self.scenarios = load_config()  # reads configs/scenarios.yaml

        if scenario_key not in self.scenarios:
            raise ValueError(
                f"scenario_key='{scenario_key}' not found in scenarios.yaml. "
                f"Available keys: {list(self.scenarios.keys())}"
            )
        self.scenario_key = scenario_key

        # Use MC horizon unless user forces a shorter horizon
        self.horizon = max_months or self.mc_cfg.horizon_months
        self.max_steps = self.horizon

        # Random generator
        base_seed = seed if seed is not None else self.mc_cfg.random_seed
        self.rng = np.random.default_rng(int(base_seed))

        # Load macro history and pre-generate Monte Carlo base paths
        macro = load_macro_series()  # dict of DataFrames
        base_paths = block_bootstrap_paths(
            macro=macro,
            horizon=self.horizon,
            block_size=self.mc_cfg.block_size_months,
            n_paths=self.mc_cfg.n_paths,
            rng=self.rng,
        )
        apply_correlation_hooks(base_paths, self.mc_cfg, self.rng)

        # Store base paths → shape (n_paths, horizon)
        self.base_paths = base_paths
        self.n_paths = self.base_paths["equity_ret"].shape[0]

        # ---------------- RL-specific hyperparameters ----------------
        # Convert bps to fraction (e.g. 10 bps → 0.001)
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.risk_aversion = float(risk_aversion)
        self.drawdown_aversion = float(drawdown_aversion)
        self.concentration_aversion = float(concentration_aversion)

        # ---------------- Gym spaces ----------------
        # Action: raw weights in [0,1], we renormalise to sum to 1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # Observation: 10-dim continuous vector
        obs_dim = 10
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Internal state placeholders
        self._reset_internal_state()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _reset_internal_state(self):
        self.t = 0
        self.portfolio_value = 1.0
        self.weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

        # Risk tracking
        self.peak_value = 1.0          # running maximum of portfolio value
        self.max_drawdown = 0.0        # worst drawdown so far

        # These will be filled in reset()
        self._equity_path = None
        self._gold_path = None
        self._cpi_path = None
        self._rate_d_path = None

    def _sample_scenario_paths(self) -> Dict[str, np.ndarray]:
            """
            Take the pre-generated base_paths and apply the chosen scenario overlay.
            Returns a dict with the same keys as base_paths but scenario-adjusted.
            """
            scen: ScenarioSpec = self.scenarios[self.scenario_key]

            # IMPORTANT: match monte_carlo.overlay_scenario_on_paths signature:
            # def overlay_scenario_on_paths(base_paths, scen, horizon)
            scen_paths = overlay_scenario_on_paths(
                self.base_paths,   # base_paths: Dict[str, np.ndarray]
                scen,              # scen: ScenarioSpec
                self.horizon,      # horizon: int (NOT MCConfig)
            )
            return scen_paths


    def _choose_episode_path(self):
        """
        Sample one Monte Carlo path index and cache the series for this episode.
        """
        scen_paths = self._sample_scenario_paths()
        idx = self.rng.integers(0, self.n_paths)

        self._equity_path = scen_paths["equity_ret"][idx]
        self._gold_path = scen_paths["gold_ret"][idx]
        self._cpi_path = scen_paths["cpi_mom"][idx]
        self._rate_d_path = scen_paths["rate_d_bps"][idx]

    def _current_macro(self) -> Tuple[float, float, float, float]:
        """
        Get macro values at time t.
        """
        if self.t >= self.horizon:
            return 0.0, 0.0, 0.0, 0.0

        eq = float(self._equity_path[self.t])
        gold = float(self._gold_path[self.t])
        cpi = float(self._cpi_path[self.t])
        rate_d = float(self._rate_d_path[self.t])
        return eq, gold, cpi, rate_d

    def _risk_free_return(self, rate_d_bps: float) -> float:
        """
        Very simple mapping from rate changes to a monthly 'cash' return.
        You can refine this later (e.g. level of rate, yield curve, etc.).
        """
        # baseline monthly rf ~ 3% annual → ~0.25% monthly
        base_annual = 0.03
        base_monthly = (1 + base_annual) ** (1 / 12) - 1

        # extra from rate change this month (bps → fraction)
        # e.g. +25 bps → +0.25% annual ≈ +0.25/12% monthly
        extra_annual = rate_d_bps / 10000.0
        extra_monthly = extra_annual / 12.0

        return base_monthly + extra_monthly

    def _get_obs(self) -> np.ndarray:
        eq, gold, cpi, rate_d = self._current_macro()
        time_frac = float(self.t) / float(self.horizon)

        # current drawdown from running peak
        if self.peak_value > 0:
            drawdown = max(0.0, 1.0 - self.portfolio_value / self.peak_value)
        else:
            drawdown = 0.0

        obs = np.array(
            [
                eq,
                gold,
                cpi,
                rate_d,
                self.weights[0],
                self.weights[1],
                self.weights[2],
                self.portfolio_value,
                time_frac,
                drawdown,
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            # Combine env-level base seed with episode seed to avoid collisions
            self.rng = np.random.default_rng(int(seed))

        self._reset_internal_state()
        self._choose_episode_path()
        obs = self._get_obs()

        if GYM_API == "gymnasium":
            return obs, {}
        else:
            # Classic gym reset returns just obs
            return obs

    def step(self, action):
        # Clip and renormalise action → weights
        raw = np.asarray(action, dtype=np.float32).flatten()
        raw = np.clip(raw, 0.0, 1.0)

        if raw.sum() <= 0.0:
            weights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        else:
            weights = raw / raw.sum()

        prev_weights = self.weights.copy()
        self.weights = weights

        # Macro at current step
        eq_ret, gold_ret, cpi_mom, rate_d_bps = self._current_macro()
        rf_ret = self._risk_free_return(rate_d_bps)

        # Asset returns vector [equity, cash, gold]
        asset_rets = np.array(
            [
                eq_ret,
                rf_ret,
                gold_ret,
            ],
            dtype=np.float32,
        )

        # Gross portfolio return
        gross_ret = float(np.dot(self.weights, asset_rets))

        # Transaction cost on turnover (sum of abs weight changes)
        turnover = float(np.abs(self.weights - prev_weights).sum())
        tc = self.transaction_cost * turnover

        # Net return after transaction costs
        net_ret = gross_ret - tc

        # Update portfolio value
        self.portfolio_value *= (1.0 + net_ret)

        # Update drawdown statistics
        self.peak_value = max(self.peak_value, self.portfolio_value)
        if self.peak_value > 0:
            drawdown = max(0.0, 1.0 - self.portfolio_value / self.peak_value)
        else:
            drawdown = 0.0
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Concentration measure via Herfindahl index (sum w_i^2, baseline 1/3)
        hhi = float(np.sum(self.weights ** 2))
        concentration = max(0.0, hhi - 1.0 / 3.0)

        # Reward: mean-variance + explicit risk penalties
        reward = (
            net_ret
            - self.risk_aversion * (net_ret ** 2)
            - self.drawdown_aversion * drawdown
            - self.concentration_aversion * concentration
        )

        # Advance time
        self.t += 1
        terminated = self.t >= self.horizon
        truncated = False

        obs = self._get_obs()
        info = {
            "portfolio_value": self.portfolio_value,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "turnover": turnover,
            "equity_ret": eq_ret,
            "gold_ret": gold_ret,
            "cpi_mom": cpi_mom,
            "rate_d_bps": rate_d_bps,
            "drawdown": drawdown,
            "max_drawdown": self.max_drawdown,
            "concentration_hhi": hhi,
        }

        if GYM_API == "gymnasium":
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated, info

    # Optional: simple human render
    def render(self, mode="human"):
        eq_ret, gold_ret, cpi_mom, rate_d_bps = self._current_macro()
        print(
            f"t={self.t:3d} | "
            f"V={self.portfolio_value: .4f} | "
            f"w=[{self.weights[0]:.2f},{self.weights[1]:.2f},{self.weights[2]:.2f}] | "
            f"eq={eq_ret:+.3f} gold={gold_ret:+.3f} "
            f"cpi={cpi_mom:+.3f} rate_d={rate_d_bps:+.1f}bps"
        )

    def close(self):
        pass
