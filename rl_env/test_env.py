from rl_env.household_portfolio_env import HouseholdPortfolioEnv

env = HouseholdPortfolioEnv(
    scenario_key="S2",
    max_months=120,
    transaction_cost_bps=10.0,
    risk_aversion=3.0,
    drawdown_aversion=2.0,
    concentration_aversion=0.3,
)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    step_out = env.step(action)

    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        obs, reward, done, info = step_out

print("Environment test completed successfully.")
