"""
02_analysis.py
==============
The experiment. Three questions:

  Q1 (ablation)  Do terrain/infrastructure features improve prediction over
                 rainfall alone?  Do physics-derived features add more?
  Q2 (leakage)   How much does the evaluation protocol matter? We compare a
                 naive random split against a leave-hotspots-out split.
  Q3 (warning)   Translated into a binary flood warning, what are the hit
                 rate, false alarm ratio and critical success index?

Outputs: results/*.csv, figures/*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os, json

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

THRESHOLD_CM = 15.0
df = pd.read_csv("data/dataset.csv")

MET = ["rain_intensity_mmhr", "rain_duration_min", "rain_total_mm",
       "peak15_mmhr", "antecedent_rain_mm"]
TERRAIN = ["catchment_ha", "imperviousness", "mean_slope_pct",
           "depression_depth_cm", "dist_to_inlet_m", "inlet_density_per_ha",
           "flow_accum_log"]
PHYSICS = ["phys_net_inflow_mmhr", "phys_capacity_ratio",
           "phys_fill_rate_cmhr", "phys_analytic_depth_cm",
           "phys_time_to_thresh_min"]

FEATURE_SETS = {
    "F1: rainfall only": MET,
    "F2: + terrain & drainage": MET + TERRAIN,
    "F3: + physics-derived": MET + TERRAIN + PHYSICS,
}

MODELS = {
    "Ridge regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0),
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, random_state=0),
}

y = df["peak_depth_cm"].values
groups = df["hotspot_id"].values


def contingency(y_true, y_pred, thr=THRESHOLD_CM):
    """Standard forecast-verification counts."""
    obs, fc = y_true >= thr, y_pred >= thr
    hits = int(np.sum(obs & fc))
    misses = int(np.sum(obs & ~fc))
    false_alarms = int(np.sum(~obs & fc))
    pod = hits / (hits + misses) if hits + misses else np.nan   # hit rate
    far = false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan
    csi = hits / (hits + misses + false_alarms) if hits + misses + false_alarms else np.nan
    return dict(hits=hits, misses=misses, false_alarms=false_alarms,
                POD=pod, FAR=far, CSI=csi)


def run_cv(X, y, model_fn, splitter, groups=None):
    """Out-of-fold predictions: every row predicted by a model that never saw it."""
    oof = np.zeros(len(y))
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for tr, te in it:
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    return oof


# ----------------------------------------------------------------------
# Main experiment: 3 feature sets x 3 models x 2 validation protocols
# ----------------------------------------------------------------------
records = []
oof_store = {}

for proto_name, splitter, grp in [
        ("Random 5-fold (LEAKY)", KFold(5, shuffle=True, random_state=0), None),
        ("Leave-hotspots-out 5-fold", GroupKFold(n_splits=5), groups)]:

    for fs_name, cols in FEATURE_SETS.items():
        X = df[cols].values
        for m_name, m_fn in MODELS.items():
            oof = run_cv(X, y, m_fn, splitter, grp)
            oof = np.clip(oof, 0, None)
            cm = contingency(y, oof)
            records.append({
                "protocol": proto_name, "features": fs_name, "model": m_name,
                "RMSE_cm": np.sqrt(mean_squared_error(y, oof)),
                "MAE_cm": mean_absolute_error(y, oof),
                "R2": r2_score(y, oof),
                **cm,
            })
            oof_store[(proto_name, fs_name, m_name)] = oof
            print(f"{proto_name:28s} | {fs_name:26s} | {m_name:18s} "
                  f"RMSE {records[-1]['RMSE_cm']:5.2f}  R2 {records[-1]['R2']:.3f}  "
                  f"CSI {cm['CSI']:.3f}")

res = pd.DataFrame(records)
res.to_csv("results/main_results.csv", index=False)

STRICT = "Leave-hotspots-out 5-fold"
strict = res[res.protocol == STRICT]

# ----------------------------------------------------------------------
# Permutation importance for the best strict-protocol model
# ----------------------------------------------------------------------
best = strict.sort_values("RMSE_cm").iloc[0]
cols = FEATURE_SETS[best.features]
X = df[cols].values
gkf = GroupKFold(n_splits=5)
tr, te = next(gkf.split(X, y, groups))
m = MODELS[best.model]()
m.fit(X[tr], y[tr])
pi = permutation_importance(m, X[te], y[te], n_repeats=10,
                            random_state=0, scoring="neg_root_mean_squared_error")
imp = pd.DataFrame({"feature": cols,
                    "importance": pi.importances_mean,
                    "std": pi.importances_std}).sort_values("importance", ascending=False)
imp.to_csv("results/permutation_importance.csv", index=False)

# ----------------------------------------------------------------------
# Per-hotspot error (where does the model fail?)
# ----------------------------------------------------------------------
oof_best = oof_store[(STRICT, best.features, best.model)]
per_hs = (pd.DataFrame({"hotspot_id": df.hotspot_id, "err": oof_best - y,
                        "abs": np.abs(oof_best - y)})
          .groupby("hotspot_id")
          .agg(MAE=("abs", "mean"), bias=("err", "mean")).reset_index())
per_hs = per_hs.merge(df.groupby("hotspot_id")
                      .agg(flood_rate=("flooded", "mean"),
                           depression=("depression_depth_cm", "first"),
                           drain_cap=("drain_capacity_mmhr", "first")).reset_index())
per_hs.to_csv("results/per_hotspot_error.csv", index=False)

# ======================================================================
# FIGURES
# ======================================================================
C = {"F1: rainfall only": "#9aa5b1", "F2: + terrain & drainage": "#4c78a8",
     "F3: + physics-derived": "#d1782f"}

# Fig 1 -- ablation
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
w = 0.26
mods = list(MODELS)
for i, (fs, c) in enumerate(C.items()):
    sub = strict[strict.features == fs].set_index("model").loc[mods]
    xpos = np.arange(len(mods)) + (i - 1) * w
    ax[0].bar(xpos, sub.RMSE_cm, w, label=fs, color=c)
    ax[1].bar(xpos, sub.CSI, w, label=fs, color=c)
for a, t in zip(ax, ["RMSE (cm) — lower is better", "Critical Success Index — higher is better"]):
    a.set_xticks(np.arange(len(mods)))
    a.set_xticklabels(mods, rotation=12, ha="right")
    a.set_title(t, fontsize=9)
ax[0].legend(fontsize=7, frameon=False)
fig.suptitle("Figure 1. Feature-set ablation (leave-hotspots-out CV)", fontsize=10)
fig.tight_layout()
fig.savefig("figures/fig1_ablation.png", bbox_inches="tight")
plt.close(fig)

# Fig 2 -- leakage
fig, ax = plt.subplots(figsize=(5.6, 3.2))
for i, proto in enumerate(res.protocol.unique()):
    sub = res[(res.protocol == proto) & (res.features == "F3: + physics-derived")]
    sub = sub.set_index("model").loc[mods]
    ax.bar(np.arange(len(mods)) + (i - 0.5) * 0.34, sub.RMSE_cm, 0.34,
           label=proto, color=["#c44e52", "#4c78a8"][i])
ax.set_xticks(np.arange(len(mods)))
ax.set_xticklabels(mods, rotation=12, ha="right")
ax.set_ylabel("RMSE (cm)")
ax.set_title("Figure 2. Random splits overstate skill\n(same model, same features)", fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("figures/fig2_leakage.png", bbox_inches="tight")
plt.close(fig)

# Fig 3 -- predicted vs observed
fig, ax = plt.subplots(figsize=(4.2, 4.0))
ax.scatter(y, oof_best, s=4, alpha=0.18, color="#4c78a8", edgecolors="none")
lim = [0, max(y.max(), oof_best.max()) * 1.02]
ax.plot(lim, lim, "k--", lw=1)
ax.axhline(THRESHOLD_CM, color="#c44e52", lw=0.8, ls=":")
ax.axvline(THRESHOLD_CM, color="#c44e52", lw=0.8, ls=":")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed peak depth (cm)"); ax.set_ylabel("Predicted peak depth (cm)")
ax.set_title(f"Figure 3. {best.model}, {best.features}\nR² = {best.R2:.3f}, "
             f"RMSE = {best.RMSE_cm:.2f} cm", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig3_scatter.png", bbox_inches="tight")
plt.close(fig)

# Fig 4 -- permutation importance
fig, ax = plt.subplots(figsize=(5.4, 4.2))
top = imp.head(12).iloc[::-1]
cols_c = ["#d1782f" if f.startswith("phys_") else "#4c78a8" for f in top.feature]
ax.barh(top.feature, top.importance, xerr=top["std"], color=cols_c)
ax.set_xlabel("Increase in RMSE when feature is shuffled (cm)")
ax.set_title("Figure 4. Permutation importance\n(orange = physics-derived)", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig4_importance.png", bbox_inches="tight")
plt.close(fig)

# Fig 5 -- per-hotspot error
fig, ax = plt.subplots(figsize=(5.6, 3.2))
p = per_hs.sort_values("MAE")
ax.bar(p.hotspot_id, p.MAE, color="#4c78a8")
ax.set_ylabel("MAE (cm)"); ax.set_xlabel("Hotspot (held out)")
ax.tick_params(axis="x", rotation=90, labelsize=6)
ax.set_title("Figure 5. Error varies strongly by hotspot", fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig5_per_hotspot.png", bbox_inches="tight")
plt.close(fig)

summary = {
    "best_model": best.model, "best_features": best.features,
    "best_RMSE": round(float(best.RMSE_cm), 3), "best_R2": round(float(best.R2), 4),
    "best_CSI": round(float(best.CSI), 4), "best_POD": round(float(best.POD), 4),
    "best_FAR": round(float(best.FAR), 4),
    "n_rows": int(len(df)), "n_hotspots": int(df.hotspot_id.nunique()),
    "n_events": int(df.event_id.nunique()),
    "flood_rate": round(float(df.flooded.mean()), 4),
}
json.dump(summary, open("results/summary.json", "w"), indent=2)
print("\n", json.dumps(summary, indent=2))
