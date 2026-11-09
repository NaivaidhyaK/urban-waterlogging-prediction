"""
cleaning_check.py
=================
Stage 10 of the Data Collection Protocol, applied to the template dataset.

Run this BEFORE any analysis, every time, on any dataset. Each check below has
caught a real error in somebody's real project. Six problems are planted in
raw_dataset_TEMPLATE.csv; this script finds all six and writes a cleaned file.

Usage:  python3 cleaning_check.py
"""

import pandas as pd
import numpy as np

RAW = "/mnt/user-data/outputs/sample-data/raw_dataset_TEMPLATE.csv"
OUT = "/mnt/user-data/outputs/sample-data/cleaned_dataset.csv"

df = pd.read_csv(RAW)
print(f"Loaded {len(df)} rows x {df.shape[1]} columns\n")
print("=" * 62)

issues = []

# --- CHECK 1: duplicate rows ------------------------------------------
key = ["site_id", "event_date"]
dupes = df.duplicated(subset=key, keep=False)
print(f"1. DUPLICATES on {key}: {dupes.sum()} rows involved")
if dupes.any():
    print(df.loc[dupes, key + ["flooded"]].to_string(index=False))
    issues.append("duplicates")
df = df.drop_duplicates(subset=key, keep="first")

# --- CHECK 2: date format ---------------------------------------------
parsed = pd.to_datetime(df.event_date, format="%Y-%m-%d", errors="coerce")
bad_dates = parsed.isna()
print(f"\n2. BAD DATE FORMAT: {bad_dates.sum()} rows")
if bad_dates.any():
    print("   offending values:", df.loc[bad_dates, "event_date"].unique())
    issues.append("date format")
# repair by trying a second format
df["event_date"] = parsed.fillna(
    pd.to_datetime(df.event_date, format="%d/%m/%Y", errors="coerce"))
assert df.event_date.isna().sum() == 0, "still unparseable dates"

# --- CHECK 3: impossible values ---------------------------------------
print("\n3. RANGE CHECKS")
rules = {
    "rain_24h_mm":      (0, 600),
    "rain_max30_mmhr":  (0, 400),
    "rain_prev7d_mm":   (0, 900),
    "slope_pct":        (0, 100),
    "impervious_frac":  (0, 1),
    "depth_cm":         (0, 300),
    "dist_to_drain_m":  (0, 5000),
    "lat":              (26.5, 27.3),
    "lon":              (75.4, 76.2),
}
for col, (lo, hi) in rules.items():
    bad = df[(df[col] < lo) | (df[col] > hi)]
    flag = "  <-- PROBLEM" if len(bad) else ""
    print(f"   {col:22s} [{lo}, {hi}]  violations: {len(bad)}{flag}")
    if len(bad):
        issues.append(f"range: {col}")
        df.loc[bad.index, col] = np.nan          # set impossible -> missing

# --- CHECK 4: missing values ------------------------------------------
print("\n4. MISSING VALUES (columns with any)")
miss = df.isna().sum()
for c, k in miss[miss > 0].sort_values(ascending=False).items():
    print(f"   {c:22s} {k:5d}  ({k/len(df):.1%})")
if miss.sum():
    issues.append("missing values")

# --- CHECK 5: sites with too few labels -------------------------------
print("\n5. LABEL COVERAGE BY SITE")
cov = df.groupby("site_id").flooded.apply(lambda s: s.notna().sum())
thin = cov[cov < 10]
print(f"   sites with <10 labelled events: {len(thin)}")
if len(thin):
    print("  ", dict(thin))
    issues.append("sparse sites")
    df = df[~df.site_id.isin(thin.index)]
    print(f"   -> dropped, {df.site_id.nunique()} sites remain")

# --- CHECK 6: inconsistent categories ---------------------------------
print("\n6. CATEGORY CONSISTENCY")
for c in ["evidence_type", "source_type", "confidence", "depth_class"]:
    vals = df[c].dropna().unique()
    lowered = {str(v).lower() for v in vals}
    if len(lowered) < len(vals):
        print(f"   {c}: {len(vals)} raw values -> {len(lowered)} after lowercasing"
              "  <-- PROBLEM")
        issues.append(f"case: {c}")
    df[c] = df[c].astype(str).str.lower().replace("nan", np.nan)
print("   normalised to lowercase")

# --- CHECK 7: class balance -------------------------------------------
print("\n7. CLASS BALANCE")
print(f"   overall flood rate : {df.flooded.mean():.3f}")
print(f"   hotspots           : {df[df.site_type=='hotspot'].flooded.mean():.3f}")
print(f"   controls           : {df[df.site_type=='control'].flooded.mean():.3f}")
if df.flooded.mean() > 0.9 or df.flooded.mean() < 0.02:
    print("   <-- PROBLEM: almost no variation to learn from")
    issues.append("class balance")

# --- CHECK 8: does the signal exist at all? ---------------------------
print("\n8. SANITY: is there any signal?")
for c in ["rain_max30_mmhr", "depression_depth_m", "slope_pct", "impervious_frac"]:
    r = df[[c, "flooded"]].dropna().corr().iloc[0, 1]
    print(f"   corr(flooded, {c:20s}) = {r:+.3f}")
print("   (slope should be NEGATIVE; the others positive. If signs are")
print("    backwards, suspect a units or merge error before believing it.)")

# ----------------------------------------------------------------------
df.to_csv(OUT, index=False)
print("\n" + "=" * 62)
print(f"issues found: {len(set(issues))} -> {sorted(set(issues))}")
print(f"cleaned file: {OUT}  ({len(df)} rows, {df.site_id.nunique()} sites)")
