"""
train_ppo.py

Train a PPO agent on HouseholdPortfolioEnv.

Run from the Project root as:
    python -m rl_training.train_ppo
"""

import os
from datetime import datetime

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_env.household_portfolio_env import HouseholdPortfolioEnv


# ------------------------- Configs -------------------------

N_ENVS = 4                 # how many parallel envs
TOTAL_TIMESTEPS = 200_000  # increase later (e.g. 500k or 1M)
SCENARIO_KEY = "S2"        # baseline scenario from scenarios.yaml
MAX_MONTHS = 120           # episode length (months)
TRANSACTION_COST_BPS = 10.0
RISK_AVERSION = 3.0

LOG_DIR = os.path.join("outputs", "rl_logs")
MODEL_DIR = os.path.join("outputs", "rl_models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ------------------------- Env factory -------------------------

def make_env(seed: int = 0):
    """
    Factory to create one HouseholdPortfolioEnv wrapped in Monitor.
    Used by DummyVecEnv for parallel training.
    """
    def _init():
        env = HouseholdPortfolioEnv(
            scenario_key=SCENARIO_KEY,
            max_months=MAX_MONTHS,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            risk_aversion=RISK_AVERSION,
            drawdown_aversion=1.5,
            concentration_aversion=0.05,
            seed=999,
        )
        # Monitor logs per-episode reward, length, etc.
        return Monitor(env)
    return _init


def build_vec_env(n_envs: int) -> DummyVecEnv:
    env_fns = [make_env(seed=i) for i in range(n_envs)]
    return DummyVecEnv(env_fns)


# ------------------------- Main training -------------------------

def main():
    # 1) Build vectorized env
    vec_env = build_vec_env(N_ENVS)

    # 2) Define PPO model
    # Policy network: small MLP is enough for now
    policy_kwargs = dict(
        net_arch=[64, 64],   # two hidden layers with 64 units each
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,             # rollout length per env
        batch_size=2048,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs,
    )

    # 3) Train
    run_name = datetime.now().strftime("ppo_household_%Y%m%d_%H%M%S")
    print(f"[TRAIN] Starting training run: {run_name}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=run_name,
    )

    # 4) Save model
    model_path = os.path.join(MODEL_DIR, f"{run_name}.zip")
    model.save(model_path)
    print(f"[TRAIN] Model saved to: {model_path}")

    # 5) Quick sanity evaluation on 1 env
    eval_env = HouseholdPortfolioEnv(
        scenario_key=SCENARIO_KEY,
        max_months=MAX_MONTHS,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        risk_aversion=RISK_AVERSION,
        drawdown_aversion=1.5,
        concentration_aversion=0.05,
        seed=999,
    )

    # Gymnasium-style reset returns (obs, info)
    reset_out = eval_env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, _ = reset_out
    else:
        obs = reset_out

    done = False
    total_reward = 0.0
    values = []

    while not done:
        # PPO expects batch dimension, so reshape obs
        action, _ = model.predict(obs, deterministic=True)
        step_out = eval_env.step(action)

        # Handle gymnasium vs gym return signatures
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        total_reward += float(reward)
        values.append(info.get("portfolio_value", np.nan))

    print(f"[EVAL] Total reward over one episode: {total_reward:.4f}")
    if values:
        print(f"[EVAL] Final portfolio value: {values[-1]:.4f}")

    eval_env.close()


if __name__ == "__main__":
    main()
