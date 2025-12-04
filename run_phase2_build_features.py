import os, sys, pandas as pd, numpy as np
ROOT = os.path.abspath(os.path.dirname(__file__))
PKG_PATH = os.path.join(ROOT, "stress_sim", "src")
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from stress_sim.cpi_mapper import map_households_to_cpi
from stress_sim.hces_annex_priors import load_annex_priors
from stress_sim.household_features import compute_static_features

DATAI = os.path.join(ROOT, "data_interim")
PARTB = os.path.join(ROOT, "synth_partB")

paths = {
    "cpi": os.path.join(DATAI, "cpi_long.csv"),
    "annex": os.path.join(DATAI, "annex_priors.csv"),
    "hh": os.path.join(PARTB, "households.csv"),
    "exp": os.path.join(PARTB, "household_expenses.csv"),
    "loans": os.path.join(PARTB, "household_loans.csv"),
    "assets": os.path.join(PARTB, "household_assets.csv"),
}

missing = [k for k,p in paths.items() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing inputs: { {k: paths[k] for k in missing} }")

cpi_long = pd.read_csv(paths["cpi"], parse_dates=["date"])
annex    = load_annex_priors(paths["annex"])
df_hh    = pd.read_csv(paths["hh"])
df_exp   = pd.read_csv(paths["exp"])
df_loans = pd.read_csv(paths["loans"])
df_assets= pd.read_csv(paths["assets"])

# normalize all incoming Part-B headers to lowercase
for d in (df_hh, df_exp, df_loans, df_assets):
    d.columns = [c.strip().lower() for c in d.columns]

mapper, cpi_combined = map_households_to_cpi(df_hh, cpi_long)
features = compute_static_features(df_hh, df_exp, df_loans, df_assets)
features = features.merge(mapper, on="lead_id", how="left")

out_path = os.path.join(DATAI, "households_features.csv")
features.to_csv(out_path, index=False)
print(f"Phase 2 features written: {out_path}  | rows={len(features)}")