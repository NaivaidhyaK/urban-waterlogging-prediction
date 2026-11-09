"""
01_rainfall_data.py
===================

THE TOY PHYSICS (a deliberately simplified "rational method" + reservoir):

  1. Runoff coefficient        C = 0.10 + 0.80 * imperviousness
  2. Inflow rate               q_in  = C * rainfall_intensity           [mm/hr]
  3. Effective drain capacity  q_out = q_cap * (1 - 0.35*antecedent_wetness)
                                            * (1 - blockage)           [mm/hr]
  4. Net filling rate          q_net = q_in - q_out                     [mm/hr]
  5. If q_net <= 0  -> no ponding.
     Else the hotspot behaves as a bowl of storage depth D_max (cm) fed by
     q_net over a contributing area ratio R (catchment area / ponded area):

        depth(t) = min( q_net * R * t / 10 , D_max )      [cm, t in hours]

  6. Onset time  = time to exceed the hazard threshold (15 cm)
     Peak depth  = depth at end of storm
     Duration    = time above threshold during storm + drain-down afterwards

  7. Observation noise is added to all three targets to mimic imperfect
     ground truth (crowd-sourced reports, complaint logs, CCTV estimates).

Outputs: data/hotspots.csv, data/events.csv, data/dataset.csv
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(20260812)   # fixed seed => reproducible
N_HOTSPOTS = 24
N_EVENTS = 420
THRESHOLD_CM = 15.0

os.makedirs("data", exist_ok=True)

# ----------------------------------------------------------------------
# 1. Static hotspot characteristics (what a DEM + GIS analysis would give)
# ----------------------------------------------------------------------
hs = pd.DataFrame({
    "hotspot_id":      [f"H{i:02d}" for i in range(1, N_HOTSPOTS + 1)],
    "catchment_ha":    RNG.uniform(1.5, 22.0, N_HOTSPOTS),
    "imperviousness":  RNG.uniform(0.45, 0.95, N_HOTSPOTS),
    "mean_slope_pct":  RNG.uniform(0.2, 4.5, N_HOTSPOTS),
    "depression_depth_cm": RNG.uniform(18, 75, N_HOTSPOTS),
    "dist_to_inlet_m": RNG.uniform(5, 160, N_HOTSPOTS),
    "inlet_density_per_ha": RNG.uniform(0.3, 3.2, N_HOTSPOTS),
})

# Flow accumulation is strongly (not perfectly) tied to catchment size --
# this is what a D8 flow-accumulation raster in QGIS would report.
hs["flow_accum_log"] = np.log1p(hs["catchment_ha"] * 1e4 / 30**2) \
                       + RNG.normal(0, 0.15, N_HOTSPOTS)

# Drain capacity: better where inlets are dense, close, and slope helps.
hs["drain_capacity_mmhr"] = (
    9.0
    + 5.5 * hs["inlet_density_per_ha"]
    - 0.030 * hs["dist_to_inlet_m"]
    + 2.2 * hs["mean_slope_pct"]
    + RNG.normal(0, 1.6, N_HOTSPOTS)
).clip(2.5, 42.0)

# Contributing-area ratio: how much catchment drains into the small pond.
hs["area_ratio"] = (3.0 + 0.9 * hs["catchment_ha"] + RNG.normal(0, 1.5, N_HOTSPOTS)).clip(3, 26)

# ----------------------------------------------------------------------
# 2. Rainfall events (what IMERG / IMD gauge records would give)
# ----------------------------------------------------------------------
# Intensity is heavy-tailed: many modest storms, few extreme ones.
intensity = RNG.gamma(shape=2.0, scale=11.0, size=N_EVENTS).clip(3, 165)
duration_min = RNG.gamma(shape=2.4, scale=26.0, size=N_EVENTS).clip(15, 320)
antecedent_mm = RNG.gamma(shape=1.5, scale=13.0, size=N_EVENTS).clip(0, 130)

ev = pd.DataFrame({
    "event_id": [f"E{i:03d}" for i in range(1, N_EVENTS + 1)],
    "rain_intensity_mmhr": intensity,
    "rain_duration_min": duration_min,
    "antecedent_rain_mm": antecedent_mm,
})
ev["rain_total_mm"] = ev.rain_intensity_mmhr * ev.rain_duration_min / 60.0
# Peak 15-min intensity: bursts inside the storm
ev["peak15_mmhr"] = ev.rain_intensity_mmhr * RNG.uniform(1.05, 2.10, N_EVENTS)

# ----------------------------------------------------------------------
# 3. Apply the toy physics to every (hotspot, event) pair
# ----------------------------------------------------------------------
rows = []
for _, h in hs.iterrows():
    # Blockage varies by hotspot and event (litter, silt) -- unobserved noise
    for _, e in ev.iterrows():
        C = 0.10 + 0.80 * h.imperviousness
        q_in = C * e.peak15_mmhr

        wetness = min(e.antecedent_rain_mm / 100.0, 1.0)
        blockage = RNG.beta(1.6, 6.0)                      # usually small
        q_out = h.drain_capacity_mmhr * (1 - 0.35 * wetness) * (1 - blockage)

        q_net = q_in - q_out                                # mm/hr
        T_hr = e.rain_duration_min / 60.0

        if q_net <= 0:
            peak, onset, dur = 0.0, np.nan, 0.0
        else:
            fill_rate_cm_hr = q_net * h.area_ratio / 10.0
            peak_raw = fill_rate_cm_hr * T_hr
            peak = min(peak_raw, h.depression_depth_cm)

            if peak < THRESHOLD_CM:
                onset, dur = np.nan, 0.0
            else:
                onset = THRESHOLD_CM / fill_rate_cm_hr * 60.0        # minutes
                # drain-down after rain stops, at the drain's own rate
                recess_cm_hr = max(h.drain_capacity_mmhr * h.area_ratio / 10.0 * 0.55, 3.0)
                above = max(e.rain_duration_min - onset, 0)
                dur = above + (peak - THRESHOLD_CM) / recess_cm_hr * 60.0

        # --- observation noise (imperfect ground truth) -----------------
        if peak > 0:
            peak = max(peak + RNG.normal(0, 1.8 + 0.06 * peak), 0)
        if not np.isnan(onset):
            onset = max(onset + RNG.normal(0, 3.5), 1.0)
        if dur > 0:
            dur = max(dur + RNG.normal(0, 6.0), 1.0)

        rows.append({
            "hotspot_id": h.hotspot_id,
            "event_id": e.event_id,
            "peak_depth_cm": peak,
            "onset_min": onset,
            "duration_min": dur,
            "flooded": int(peak >= THRESHOLD_CM),
        })

df = pd.DataFrame(rows)
df = df.merge(hs, on="hotspot_id").merge(ev, on="event_id")

# ----------------------------------------------------------------------
# 4. Physics-derived features -- the "physics guidance"
#    These are computed from KNOWN hydrology, not learned. They are the
#    treatment whose value the paper tests.
# ----------------------------------------------------------------------
Cest = 0.10 + 0.80 * df.imperviousness
q_in_est = Cest * df.peak15_mmhr
q_out_est = df.drain_capacity_mmhr * (1 - 0.35 * (df.antecedent_rain_mm / 100).clip(0, 1))
df["phys_net_inflow_mmhr"] = q_in_est - q_out_est
df["phys_capacity_ratio"] = q_in_est / df.drain_capacity_mmhr
df["phys_fill_rate_cmhr"] = df.phys_net_inflow_mmhr.clip(lower=0) * df.area_ratio / 10
df["phys_analytic_depth_cm"] = np.minimum(
    df.phys_fill_rate_cmhr * df.rain_duration_min / 60, df.depression_depth_cm)
df["phys_time_to_thresh_min"] = np.where(
    df.phys_fill_rate_cmhr > 0,
    THRESHOLD_CM / df.phys_fill_rate_cmhr.replace(0, np.nan) * 60, 999)
df["phys_time_to_thresh_min"] = df.phys_time_to_thresh_min.clip(upper=999)

hs.to_csv("data/hotspots.csv", index=False)
ev.to_csv("data/events.csv", index=False)
df.to_csv("data/dataset.csv", index=False)

print(f"rows: {len(df)}  hotspots: {df.hotspot_id.nunique()}  events: {df.event_id.nunique()}")
print(f"flood rate (>= {THRESHOLD_CM} cm): {df.flooded.mean():.3f}")
print(f"peak depth  mean {df.peak_depth_cm.mean():.1f} cm  max {df.peak_depth_cm.max():.1f} cm")
print(f"flooded events per hotspot: min {df.groupby('hotspot_id').flooded.sum().min()}, "
      f"max {df.groupby('hotspot_id').flooded.sum().max()}")
