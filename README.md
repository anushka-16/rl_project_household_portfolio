# RL Household Portfolio Optimization (PPO – S2 Scenario)

This project implements a **Reinforcement Learning (RL)** agent using **Proximal Policy Optimization (PPO)** to solve a long-horizon **household asset allocation** problem.  
The agent learns optimal allocations among **Equity, Cash, and Gold** over **120-month (10-year)** episodes in a custom Gym-like environment.

A full **Streamlit dashboard** is included for evaluation, comparison against multiple baselines, and live interactive experiments.

---

## Project Overview

### Problem Statement  
Design an RL agent that maximizes long-term risk-adjusted portfolio growth for a household by choosing dynamic monthly allocations to three assets.  
The agent must balance:

- Return maximization  
- Drawdown control  
- Concentration limits  
- Risk sensitivity  

This problem mirrors real-world long-term wealth planning.

### Why PPO?  
- Continuous action space (asset allocation weights)  
- Stable training  
- Supports constraints via reward shaping  
- Well-suited for financial RL where gradients must be controlled  