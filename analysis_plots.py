import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Paths ---
BASE_DIR = "."
RL_PATHS_CSV = os.path.join(BASE_DIR, "outputs", "rl_eval", "ppo_eval_paths.csv")
BASELINE_PATHS_CSV = os.path.join(BASE_DIR, "outputs", "rl_eval", "baseline_equal_weight_paths.csv")

# --- Load data ---
df_rl = pd.read_csv(RL_PATHS_CSV)
df_baseline = pd.read_csv(BASELINE_PATHS_CSV)

# Just take episode 0 for a simple curve comparison
rl_ep0 = df_rl[df_rl["episode"] == 0].copy()
bl_ep0 = df_baseline[df_baseline["episode"] == 0].copy()

# --- 1. Plot portfolio value over time ---

plt.figure(figsize=(10, 6))
plt.plot(rl_ep0["t"], rl_ep0["portfolio_value"], label="RL (PPO) – episode 0")
plt.plot(bl_ep0["t"], bl_ep0["portfolio_value"], label="Baseline (equal weight) – episode 0", linestyle="--")
plt.xlabel("Time step (months)")
plt.ylabel("Portfolio value (starting at 1.0)")
plt.title("Portfolio value over time – RL vs S2 ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 2. Distribution of final portfolio values across episodes ---

# For RL, final row per episode
final_rl = df_rl.groupby("episode")["portfolio_value"].last().reset_index(name="final_value")
final_bl = df_baseline.groupby("episode")["portfolio_value"].last().reset_index(name="final_value")

plt.figure(figsize=(10, 6))
plt.hist(final_rl["final_value"], bins=10, alpha=0.6, label="RL (PPO)")
plt.hist(final_bl["final_value"], bins=10, alpha=0.6, label="Baseline (equal weight)")
plt.xlabel("Final portfolio value")
plt.ylabel("Count of episodes")
plt.title("Distribution of final portfolio values – RL vs Baseline")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 3. Average allocation over time (RL vs Baseline, episode 0) ---

plt.figure(figsize=(10, 6))
plt.stackplot(
    rl_ep0["t"],
    rl_ep0["w_equity"],
    rl_ep0["w_cash"],
    rl_ep0["w_gold"],
    labels=["Equity", "Cash", "Gold"],
)
plt.xlabel("Time step (months)")
plt.ylabel("Weight")
plt.title("RL (PPO) – asset allocation over time (episode 0)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.stackplot(
    bl_ep0["t"],
    bl_ep0["w_equity"],
    bl_ep0["w_cash"],
    bl_ep0["w_gold"],
    labels=["Equity", "Cash", "Gold"],
)
plt.xlabel("Time step (months)")
plt.ylabel("Weight")
plt.title("Baseline (equal weight) – asset allocation over time (episode 0)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
