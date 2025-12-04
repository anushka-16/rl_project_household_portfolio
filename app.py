import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt

# --------------------------
# 1. Data loading helpers
# --------------------------

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        st.error(f"File not found: {p.resolve()}")
        return pd.DataFrame()
    return pd.read_csv(p)


def load_all_data(
    ppo_paths_path: str,
    ppo_summary_path: str,
    base_paths_path: str,
    base_summary_path: str,
):
    ppo_paths = load_csv(ppo_paths_path)
    ppo_summary = load_csv(ppo_summary_path)
    base_paths = load_csv(base_paths_path)
    base_summary = load_csv(base_summary_path)
    return ppo_paths, ppo_summary, base_paths, base_summary


# --------------------------
# 2. High-level numbers
# --------------------------

def make_summary_table(ppo_summary: pd.DataFrame,
                       base_summary: pd.DataFrame) -> pd.DataFrame:
    if ppo_summary.empty or base_summary.empty:
        return pd.DataFrame()

    def stats(col: str, df: pd.DataFrame):
        return {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "min": df[col].min(),
            "max": df[col].max(),
            "std": df[col].std(),
        }

    ppo_fv = stats("final_value", ppo_summary)
    base_fv = stats("final_value", base_summary)
    ppo_sh = stats("sharpe_monthly", ppo_summary)
    base_sh = stats("sharpe_monthly", base_summary)

    rows = []

    def add_row(metric, rl_val, base_val, fmt="{:.3f}"):
        diff = rl_val - base_val
        if base_val != 0:
            rel = (rl_val / base_val - 1.0) * 100
        else:
            rel = np.nan
        rows.append({
            "Metric": metric,
            "RL (PPO)": fmt.format(rl_val),
            "Baseline (equal weight)": fmt.format(base_val),
            "Difference": fmt.format(diff),
            "RL vs Baseline %": (f"{rel:+.1f}%" if np.isfinite(rel) else "N/A"),
        })

    add_row("Final value – mean", ppo_fv["mean"], base_fv["mean"], "{:.4f}")
    add_row("Final value – median", ppo_fv["median"], base_fv["median"], "{:.4f}")
    add_row("Final value – min", ppo_fv["min"], base_fv["min"], "{:.4f}")
    add_row("Final value – max", ppo_fv["max"], base_fv["max"], "{:.4f}")

    add_row("Sharpe (monthly) – mean", ppo_sh["mean"], base_sh["mean"], "{:.3f}")
    add_row("Sharpe (monthly) – median", ppo_sh["median"], base_sh["median"], "{:.3f}")

    return pd.DataFrame(rows)


def generate_insights(ppo_summary: pd.DataFrame,
                      base_summary: pd.DataFrame) -> str:
    if ppo_summary.empty or base_summary.empty:
        return "Not enough data loaded to generate insights."

    rl_mean_fv = ppo_summary["final_value"].mean()
    base_mean_fv = base_summary["final_value"].mean()
    rl_mean_sh = ppo_summary["sharpe_monthly"].mean()
    base_mean_sh = base_summary["sharpe_monthly"].mean()

    fv_mult = rl_mean_fv / base_mean_fv if base_mean_fv != 0 else np.nan
    sh_diff = rl_mean_sh - base_mean_sh

    lines = []

    # 1. Performance gap
    lines.append(
        f"- **Performance:** Across episodes, the RL policy grows the "
        f"portfolio to ~{rl_mean_fv:.2f}× the starting value on average, "
        f"vs ~{base_mean_fv:.2f}× for the equal-weight baseline "
        f"(≈ {fv_mult:.1f}× higher ending wealth)."
    )

    # 2. Risk-adjusted
    lines.append(
        f"- **Risk-adjusted return:** Mean monthly Sharpe is "
        f"{rl_mean_sh:.2f} for RL vs {base_mean_sh:.2f} for baseline "
        f" (Δ ≈ {sh_diff:+.2f}). The baseline is effectively destroying "
        f"risk-adjusted performance in this scenario."
    )

    # 3. Behaviour / allocation
    lines.append(
        "- **Behaviour:** The trained PPO policy tends to move quickly into a "
        "high-conviction allocation (e.g., shifting out of gold and cash) and "
        "then holds that stance, whereas the baseline maintains a static "
        "33-33-33 mix regardless of market conditions."
    )

    # 4. Practical recommendation
    lines.append(
        "- **Practical plan:** For this synthetic household and return "
        "environment, a rule-based or RL-driven dynamic allocation appears "
        "significantly superior to naive equal-weighting. For deployment, you’d "
        "still cap maximum allocation per asset, inject transaction costs, and "
        "stress-test under alternative scenarios before using it with real "
        "money."
    )

    # 5. Next-step recommendations
    lines.append(
        "- **Next steps:**\n"
        "  * Run stress tests with different macro scenarios (bear market, "
        "   high inflation, flat equity).\n"
        "  * Add constraints: max equity weight, minimum cash buffer.\n"
        "  * Compare against simpler heuristics (e.g., 60-40, target-vol "
        "   rebalancing) so RL vs. ‘good human rules’ is clear.\n"
        "  * Use these results to write a short interpretation section in your "
        "   report: motivation → setup → RL vs baseline numbers → takeaway."
    )

    return "\n".join(lines)


# --------------------------
# 3. Streamlit UI
# --------------------------

st.set_page_config(
    page_title="RL Household Portfolio – Evaluation Dashboard",
    layout="wide",
)

st.title("🏠📈 RL Household Portfolio – PPO vs Equal-Weight Baseline")

st.markdown(
    """
This dashboard visualises the **evaluation outputs** from your RL pipeline:

- `ppo_eval_paths.csv`, `ppo_eval_summary.csv`
- `baseline_equal_weight_paths.csv`, `baseline_equal_weight_summary.csv`

Use it to compare **RL vs baseline** behaviour, distributions and summary stats,
and to get ready-to-use **interpretation & recommendations** for your report.
"""
)

# ---- Sidebar: file paths + controls ----
st.sidebar.header("📂 Data sources")

# ---- Sidebar: file paths + controls ----
st.sidebar.header("📂 Data sources")

ppo_paths_path = st.sidebar.text_input(
    "PPO paths CSV",
    value="outputs/rl_eval/ppo_eval_paths.csv",
)

ppo_summary_path = st.sidebar.text_input(
    "PPO summary CSV",
    value="outputs/rl_eval/ppo_eval_summary.csv",
)

base_paths_path = st.sidebar.text_input(
    "Baseline paths CSV",
    value="outputs/rl_eval/baseline_equal_weight_paths.csv",
)

base_summary_path = st.sidebar.text_input(
    "Baseline summary CSV",
    value="outputs/rl_eval/baseline_equal_weight_summary.csv",
)

ppo_paths, ppo_summary, base_paths, base_summary = load_all_data(
    ppo_paths_path, ppo_summary_path, base_paths_path, base_summary_path
)

if any(df.empty for df in [ppo_paths, ppo_summary, base_paths, base_summary]):
    st.error("One or more input CSVs are empty or not found.")
    st.stop()

# 🔧 FIX: If no episode column, default everything to episode = 0
for df in [ppo_paths, ppo_summary, base_paths, base_summary]:
    if "episode" not in df.columns:
        df["episode"] = 0

# OPTIONAL: ensure episode is int
for df in [ppo_paths, ppo_summary, base_paths, base_summary]:
    df["episode"] = df["episode"].astype(int)

# Number of episodes for selectors
episode_max_values = [
    ppo_paths["episode"].max(),
    base_paths["episode"].max(),
    ppo_summary["episode"].max(),
    base_summary["episode"].max(),
]

# Drop NaNs, just in case
episode_max_values = [v for v in episode_max_values if pd.notna(v)]

if not episode_max_values:
    # fallback: at least one episode (0)
    max_episode = 0
else:
    max_episode = int(max(episode_max_values))

n_episodes = max_episode + 1


st.sidebar.header("🧪 Episode selection")
episode = st.sidebar.slider(
    "Episode for path / allocation plots",
    min_value=0,
    max_value=max(0, n_episodes - 1),
    value=0,
    step=1,
)

bins = st.sidebar.slider(
    "Bins for distribution plots",
    min_value=5,
    max_value=40,
    value=20,
    step=1,
)

# Filter for chosen episode
ppo_ep = ppo_paths[ppo_paths["episode"] == episode].copy()
base_ep = base_paths[base_paths["episode"] == episode].copy()

# Add cumulative growth for plotting
ppo_ep["cum_value"] = ppo_ep["portfolio_value"]
base_ep["cum_value"] = base_ep["portfolio_value"]


# --------------------------
# 4. Layout – tabs
# --------------------------

tab_overview, tab_paths, tab_alloc, tab_dists, tab_insights = st.tabs(
    ["Overview", "Paths (episode)", "Asset allocation", "Distributions", "Interpretation & Plan"]
)

# ---- Overview tab ----
with tab_overview:
    st.subheader("Summary comparison – RL vs Baseline")

    summary_df = make_summary_table(ppo_summary, base_summary)
    st.dataframe(summary_df, use_container_width=True)

    col1, col2 = st.columns(2)

   

    with col1:
        st.markdown("**Boxplot: final portfolio value**")
        box_df = pd.DataFrame({
            "Final value": pd.concat([
                ppo_summary["final_value"],
                base_summary["final_value"]
            ], ignore_index=True),
            "Strategy": (["RL (PPO)"] * len(ppo_summary)
                        + ["Baseline"] * len(base_summary)),
        })

        chart = (
            alt.Chart(box_df)
            .mark_boxplot()
            .encode(
                x=alt.X("Strategy:N", title="Strategy"),
                y=alt.Y("Final value:Q", title="Final portfolio value"),
                color="Strategy:N"
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)


    with col2:
        st.markdown("**Boxplot: Sharpe (monthly)**")
        sh_df = pd.DataFrame({
            "Sharpe": pd.concat([
                ppo_summary["sharpe_monthly"],
                base_summary["sharpe_monthly"]
            ], ignore_index=True),
            "Strategy": (["RL (PPO)"] * len(ppo_summary)
                        + ["Baseline"] * len(base_summary)),
        })

        chart = (
            alt.Chart(sh_df)
            .mark_boxplot()
            .encode(
                x=alt.X("Strategy:N", title="Strategy"),
                y=alt.Y("Sharpe:Q", title="Sharpe (monthly)"),
                color="Strategy:N"
            )
            .properties(height=300)
        )
    st.altair_chart(chart, use_container_width=True)


# ---- Paths tab ----
with tab_paths:
    st.subheader(f"Portfolio value over time – episode {episode}")

    path_df = pd.DataFrame({
        "t": ppo_ep["t"],
        "RL (PPO)": ppo_ep["cum_value"],
        "Baseline (equal weight)": base_ep["cum_value"],
    }).set_index("t")

    st.line_chart(path_df)

    st.caption(
        "Both series start at 1.0 (normalised). This replicates the Matplotlib "
        "plot you already generated, but now it’s interactive."
    )


# ---- Allocation tab ----
with tab_alloc:
    st.subheader(f"Asset allocation over time – episode {episode}")

    alloc_cols = ["w_equity", "w_cash", "w_gold"]

    rl_alloc = ppo_ep[["t"] + alloc_cols].set_index("t")
    base_alloc = base_ep[["t"] + alloc_cols].set_index("t")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**RL (PPO)** – weights")
        rl_alloc.columns = ["Equity", "Cash", "Gold"]
        st.area_chart(rl_alloc, use_container_width=True)

    with col2:
        st.markdown("**Baseline (equal weight)** – weights")
        base_alloc.columns = ["Equity", "Cash", "Gold"]
        st.area_chart(base_alloc, use_container_width=True)


# ---- Distributions tab ----
with tab_dists:
    st.subheader("Distribution of final portfolio values")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Final portfolio values**")
        hist_df = pd.DataFrame({
            "RL (PPO)": ppo_summary["final_value"],
            "Baseline (equal weight)": base_summary["final_value"],
        })
        st.bar_chart(hist_df, use_container_width=True)

    with col2:
        st.markdown("**Sharpe ratio (monthly)**")
        sharpe_df = pd.DataFrame({
            "RL (PPO)": ppo_summary["sharpe_monthly"],
            "Baseline (equal weight)": base_summary["sharpe_monthly"],
        })
        st.bar_chart(sharpe_df, use_container_width=True)

    with st.expander("Raw summary data"):
        st.write("PPO summary")
        st.dataframe(ppo_summary, use_container_width=True)
        st.write("Baseline summary")
        st.dataframe(base_summary, use_container_width=True)

# ---- Interpretation tab ----
with tab_insights:
    st.subheader("Interpretation, plans & recommendations")

    # High-level textual interpretation (same function as before)
    insights_text = generate_insights(ppo_summary, base_summary)
    st.markdown(insights_text)

    st.markdown("---")
    st.subheader("Household outcome calculator")

    # Inputs
    colA, colB = st.columns(2)
    with colA:
        initial_capital = st.number_input(
            "Initial investment (₹)",
            min_value=10_000.0,
            value=500_000.0,
            step=50_000.0,
            help="Notional starting portfolio value for the simulation horizon (120 months ≈ 10 years).",
        )
    with colB:
        st.write("Time horizon: **10 years (120 months)** (fixed by this simulation).")

    # Mean multipliers from the evaluation summaries
    rl_mult = float(ppo_summary["final_value"].mean())
    base_mult = float(base_summary["final_value"].mean())

    rl_terminal = initial_capital * rl_mult
    base_terminal = initial_capital * base_mult
    diff_abs = rl_terminal - base_terminal
    diff_pct = (rl_mult / base_mult - 1.0) * 100.0 if base_mult != 0 else np.nan

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RL (PPO) – expected final wealth", f"₹{rl_terminal:,.0f}")
    with col2:
        st.metric("Baseline – expected final wealth", f"₹{base_terminal:,.0f}")
    with col3:
        st.metric(
            "RL advantage",
            f"₹{diff_abs:,.0f}",
            f"{diff_pct:+.1f}% vs baseline",
        )

    st.caption(
        "These numbers simply scale the **mean final portfolio value multipliers** "
        f"from the simulations (≈{rl_mult:.2f}× for RL vs ≈{base_mult:.2f}× for the equal-weight baseline)."
    )

