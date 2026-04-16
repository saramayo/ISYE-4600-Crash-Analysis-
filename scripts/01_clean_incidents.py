"""
ISYE 4600 AV Crash Severity Project — Step 1: Data cleaning (run first)

Reads raw SGO CSVs from Data/, writes Cleaned/sgo_cleaned_incidents.csv and
Cleaned/data_dictionary.csv.

Pipeline:
1. Load 4 SGO CSVs (ADS/L2 x current/archived) with correct encodings
2. Harmonize schemas across current/archived
3. Add dataset indicators (automation_level, era)
4. Stack into one unified dataframe
5. Keep latest Report Version per Report ID
6. Aggregate to one row per Same Incident ID
7. Build severity label (injury >= Minor OR airbag deployed OR towed)
8. Add missingness indicators + impute
9. Save cleaned dataset + data dictionary
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = PROJECT_ROOT / "Cleaned"
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    "ADS_current": (DATA_DIR / "SGO-ADS_June_25_Jan_15Incident_Reports_ADS.csv", "utf-8",   "ADS", "current"),
    "L2_current":  (DATA_DIR / "SGO-June_25_Jan_15_ADS_lev_2_Incident_Reports_ADAS.csv", "utf-8",   "L2",  "current"),
    "ADS_prev":    (DATA_DIR / "SGO-Prev_Incident_Reports_ADS.csv", "utf-8",   "ADS", "archived"),
    "L2_prev":     (DATA_DIR / "SGO-_level_2_ADS_PrevIncident_Reports_ADAS.csv", "latin-1", "L2",  "archived"),
}

# -----------------------------------------------------------------------------
# Step 1: Load
# -----------------------------------------------------------------------------
print("=" * 70)
print("Step 1: Loading 4 raw files")
print("=" * 70)
raw = {}
for name, (path, enc, lvl, era) in FILES.items():
    df = pd.read_csv(path, encoding=enc, low_memory=False)
    df["automation_level"] = lvl
    df["era"] = era
    raw[name] = df
    print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols  (encoding={enc})")

# -----------------------------------------------------------------------------
# Step 2: Harmonize schemas (rename archived-only names to current names)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 2: Harmonizing column names across archived/current")
print("=" * 70)

# In archived files, subject-vehicle (SV) fields are prefixed "SV ".
# In current files, these same SV fields drop the "SV " prefix.
# We'll rename archived SV-* to match current naming.
archived_rename = {
    "SV Any Air Bags Deployed?":      "Any Air Bags Deployed?",
    "SV Was Vehicle Towed?":          "Was Any Vehicle Towed?",
    "SV Were All Passengers Belted?": "Were All Passengers Belted?",
    "Weather - Fog/Smoke":            "Weather - Fog/Smoke/Haze",
    "Weather - Unknown":              "Weather - Unk - See Narrative",
    "VIN - Unknown":                  "VIN Decoded",   # both are essentially VIN-availability flags
}
for key in ["ADS_prev", "L2_prev"]:
    raw[key] = raw[key].rename(columns=archived_rename)
    print(f"  Renamed {len(archived_rename)} columns in {key}")

# -----------------------------------------------------------------------------
# Step 3: Stack
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 3: Stacking into one dataframe (union of columns)")
print("=" * 70)
df = pd.concat(list(raw.values()), ignore_index=True, sort=False)
print(f"  Unified shape: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"  Class counts by era x level:")
print(df.groupby(["era", "automation_level"]).size().to_string())

# -----------------------------------------------------------------------------
# Step 4: Deduplicate - keep latest Report Version per Report ID
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 4: Keep latest Report Version per Report ID")
print("=" * 70)
before = len(df)
df["Report Version"] = pd.to_numeric(df["Report Version"], errors="coerce")
df = (df.sort_values(["Report ID", "Report Version"])
        .drop_duplicates(subset=["Report ID"], keep="last")
        .reset_index(drop=True))
after = len(df)
print(f"  {before} -> {after} rows ({before - after} older versions removed)")

# -----------------------------------------------------------------------------
# Step 5: Build severity indicators BEFORE aggregating incidents
#   - Injury flag:   Highest Injury Severity Alleged >= "Minor"
#   - Airbag flag:   Any Air Bags Deployed? == Yes
#   - Towed flag:    Was Any Vehicle Towed? == Yes
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 5: Building severity component flags")
print("=" * 70)

def yes_flag(s: pd.Series) -> pd.Series:
    """Map yes-style cells to 1, otherwise 0. Preserves NaN where truly missing.

    Handles both archived format ("Yes"/"No") and current-era format
    ("Yes Subject Vehicle, No Crash Partner", etc.). We flag 1 if the
    SUBJECT VEHICLE (SV) had a Yes; 0 if SV had a No; NaN if Unknown.
    """
    s_str = s.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float")

    # Archived format
    out[s_str.isin(["y", "yes", "true", "1"])] = 1.0
    out[s_str.isin(["n", "no", "false", "0", "not applicable"])] = 0.0

    # Current-era format: fire 1 if "yes subject vehicle" anywhere in string
    sv_yes = s_str.str.contains("yes subject vehicle", na=False)
    sv_no  = s_str.str.contains("no subject vehicle", na=False)
    out[sv_yes] = 1.0
    out[sv_no & ~sv_yes] = 0.0   # explicitly no SV, and not a yes-SV row

    # "Unknown" or "nan" stay as NaN
    return out

# Injury rank — handles both archived short labels and current verbose labels
injury_rank = {
    # Archived format
    "no injuries reported": 0,
    "no apparent injury": 0,
    "minor": 1,
    "moderate": 2,
    "serious": 3,
    "fatality": 4,
    # Current-era verbose format
    "no injured reported": 0,
    "property damage. no injured reported": 0,
    "minor w/o hospitalization": 1,
    "minor w/ hospitalization": 1,
    "moderate w/o hospitalization": 2,
    "moderate w/ hospitalization": 2,
    "serious w/o hospitalization": 3,
    "serious w/ hospitalization": 3,
}
def rank_injury(v):
    if pd.isna(v): return np.nan
    key = str(v).strip().lower()
    if key in injury_rank:
        return injury_rank[key]
    if "fatality" in key:
        return 4
    return np.nan

df["injury_rank"]   = df["Highest Injury Severity Alleged"].apply(rank_injury)
df["airbag_flag"]   = yes_flag(df["Any Air Bags Deployed?"])
df["towed_flag"]    = yes_flag(df["Was Any Vehicle Towed?"])
df["injury_flag"]   = (df["injury_rank"] >= 2).astype("float")  # Moderate or worse only
df.loc[df["injury_rank"].isna(), "injury_flag"] = np.nan

print(f"  injury_flag  : {df['injury_flag'].sum():.0f} positive / {df['injury_flag'].notna().sum()} known")
print(f"  airbag_flag  : {df['airbag_flag'].sum():.0f} positive / {df['airbag_flag'].notna().sum()} known")
print(f"  towed_flag   : {df['towed_flag'].sum():.0f} positive / {df['towed_flag'].notna().sum()} known")

# -----------------------------------------------------------------------------
# Step 6: Aggregate by Same Incident ID (one row per unique crash)
#   - When Same Incident ID is blank, treat each Report ID as its own incident.
#   - For numeric columns, take MAX (preserves severity signal).
#   - For categorical columns, take the first non-null value.
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 6: Aggregate to one row per Same Incident ID")
print("=" * 70)

# Give each orphan (blank Same Incident ID) a unique group key
df["incident_key"] = df["Same Incident ID"].where(
    df["Same Incident ID"].notna() & (df["Same Incident ID"].astype(str).str.strip() != ""),
    "SINGLE_" + df["Report ID"].astype(str),
)

before = len(df)

# Define aggregation: numeric -> max, object -> first non-null
def first_nonnull(s):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan

agg_map = {}
for c in df.columns:
    if c == "incident_key":
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        agg_map[c] = "max"
    else:
        agg_map[c] = first_nonnull

incident_df = df.groupby("incident_key", as_index=False).agg(agg_map)
after = len(incident_df)
print(f"  {before} report-rows -> {after} unique incidents")

# -----------------------------------------------------------------------------
# Step 7: Build binary severity label (OR rule)
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 7: Building severity label (severe = injury >= Moderate OR airbag OR towed)")
print("=" * 70)

sev_components = incident_df[["injury_flag", "airbag_flag", "towed_flag"]]
# severe if any component is 1
incident_df["severe"] = (sev_components.fillna(0).sum(axis=1) > 0).astype(int)
# mark unknown severity if ALL three components are missing
all_missing = sev_components.isna().all(axis=1)
incident_df["severity_known"] = (~all_missing).astype(int)

print("  Severity distribution (all incidents):")
print(incident_df["severe"].value_counts(dropna=False).to_string())
print(f"\n  Incidents with at least one known severity signal: {incident_df['severity_known'].sum()}/{len(incident_df)}")
print(f"  Severity among 'known' rows:")
print(incident_df.loc[incident_df["severity_known"]==1, "severe"].value_counts().to_string())

# -----------------------------------------------------------------------------
# Step 8: Missing-value handling - add is_missing indicators + impute
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 8: Missingness indicators + imputation")
print("=" * 70)

# Don't create indicators for ID-like or narrative columns
skip_indicator_cols = {
    "incident_key", "Report ID", "Report Version", "Same Incident ID",
    "Same Vehicle ID", "Narrative", "VIN", "Serial Number", "Address",
    "Investigating Officer Name", "Investigating Officer Email", "Investigating Officer Phone",
    "severe", "severity_known", "injury_flag", "airbag_flag", "towed_flag",
    "injury_rank",
}

# Create is_missing flags for features with any missingness (and not skipped)
cols_with_missing = [
    c for c in incident_df.columns
    if c not in skip_indicator_cols and incident_df[c].isna().any()
]
print(f"  Creating is_missing indicators for {len(cols_with_missing)} columns with missingness")
for c in cols_with_missing:
    incident_df[f"{c}__is_missing"] = incident_df[c].isna().astype(int)

# Impute: numerics -> median; categoricals -> 'Unknown'
imputed_numeric = 0
imputed_cat = 0
for c in incident_df.columns:
    if c in skip_indicator_cols or c.endswith("__is_missing"):
        continue
    if pd.api.types.is_numeric_dtype(incident_df[c]):
        if incident_df[c].isna().any():
            med = incident_df[c].median()
            incident_df[c] = incident_df[c].fillna(med)
            imputed_numeric += 1
    else:
        if incident_df[c].isna().any():
            incident_df[c] = incident_df[c].fillna("Unknown")
            imputed_cat += 1

print(f"  Median-imputed numeric columns: {imputed_numeric}")
print(f"  'Unknown'-imputed categorical columns: {imputed_cat}")

# -----------------------------------------------------------------------------
# Step 9: Save
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 9: Saving cleaned dataset + data dictionary")
print("=" * 70)

out_csv = OUT_DIR / "sgo_cleaned_incidents.csv"
incident_df.to_csv(out_csv, index=False)
print(f"  Wrote {out_csv}  ({incident_df.shape[0]} rows x {incident_df.shape[1]} cols)")

# Data dictionary
dd_rows = []
for c in incident_df.columns:
    col = incident_df[c]
    dd_rows.append({
        "column": c,
        "dtype": str(col.dtype),
        "n_nonnull": int(col.notna().sum()),
        "n_unique": int(col.nunique(dropna=True)),
        "example": str(col.dropna().iloc[0]) if col.notna().any() else "",
    })
dd = pd.DataFrame(dd_rows)
dd.to_csv(OUT_DIR / "data_dictionary.csv", index=False)
print(f"  Wrote data_dictionary.csv ({len(dd)} entries)")

print("\nDone.")
