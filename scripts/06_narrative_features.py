"""
ISYE 4600 AV Crash Severity Project — Step 6: Narrative scenario features

Extracts interpretable, non-leaky scenario flags from free-text narratives,
combines them with the existing tabular features, and evaluates whether they
improve the ADS-only logistic regression model (which sits at AUC ~0.50 on
tabular features alone).

Key design constraint: NO outcome words (towed, injured, airbag, etc.) are
used as features — those are the label components. Only pre-crash context
and crash dynamics from the narrative body are extracted.

Outputs (written to Modeling/):
  narrative_feature_descriptions.csv  — what each flag means + prevalence
  narrative_ads_model_comparison.csv  — ADS model: tabular vs tabular+text
  narrative_pooled_model_comparison.csv — pooled model: tabular vs tabular+text
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = PROJECT_ROOT / "Cleaned" / "sgo_cleaned_incidents.csv"
OUT_DIR      = PROJECT_ROOT / "Modeling"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and split (same temporal split as all other scripts)
# ---------------------------------------------------------------------------
print("=" * 70)
print("1. Loading cleaned data")
print("=" * 70)

df = pd.read_csv(DATA_PATH, low_memory=False)
df_known = df[df["severity_known"] == 1].copy()
print(f"   severity_known=1 rows: {len(df_known)}")

train_df = df_known[df_known["era"] == "archived"].copy()
test_df  = df_known[df_known["era"] == "current"].copy()
print(f"   Train (archived): {len(train_df)} | severe rate {train_df['severe'].mean()*100:.1f}%")
print(f"   Test  (current):  {len(test_df)}  | severe rate {test_df['severe'].mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# 2. Narrative cleaning helpers
# ---------------------------------------------------------------------------

# Boilerplate Waymo header pattern — strip it so we work on the actual story
_HEADER_PAT = re.compile(
    r"(?:Pursuant to|Under|Filed under|Submitted pursuant to|In accordance with)"
    r".*?(?:Standing General Order|SGO).*?(?:\.\s+|\n)",
    re.IGNORECASE | re.DOTALL,
)
_WAYMO_SUPPLEMENT = re.compile(
    r"Waymo may supplement.*?(?:\.\s+|\n)",
    re.IGNORECASE | re.DOTALL,
)

def clean_narrative(text: str) -> str:
    """Strip boilerplate, redaction markers, and address placeholders."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = _HEADER_PAT.sub("", text)
    text = _WAYMO_SUPPLEMENT.sub("", text)
    text = re.sub(r"\[XXX\]", " ", text)
    text = re.sub(r"\[REDACTED[^\]]*\]", " ", text, flags=re.IGNORECASE)
    return text.strip()


def is_redacted(text: str) -> bool:
    """True if the narrative is mostly a CBI/redaction placeholder."""
    if pd.isna(text):
        return True
    return bool(re.search(r"REDACTED|CBI|CONFIDENTIAL", str(text), re.IGNORECASE))

# ---------------------------------------------------------------------------
# 3. Scenario flag extraction
#    Each flag is extracted from the cleaned narrative body only.
#    None of these describe post-crash outcomes (tow, injury, airbag).
# ---------------------------------------------------------------------------

def extract_narrative_features(raw_text: str) -> dict:
    """
    Return a dict of binary scenario flags extracted from a single narrative.
    Returns all-zero dict if narrative is redacted or empty.
    """
    flags = {
        # AV state at time of impact
        "nav_av_stopped":         0,  # AV was stopped / parked / stationary
        "nav_av_moving":          0,  # AV was traveling / proceeding / in motion
        "nav_av_turning":         0,  # AV was making a turn (left/right/U-turn)
        "nav_av_changing_lanes":  0,  # AV was changing or merging lanes
        "nav_av_reversing":       0,  # AV was reversing / backing up

        # Fault / contact direction
        "nav_av_struck_other":    0,  # AV initiated contact (AV struck / AV made contact)
        "nav_other_struck_av":    0,  # Other party initiated contact (struck the AV)
        "nav_rear_approach":      0,  # Other party approached from behind / rear-end scenario

        # Scenario / location type
        "nav_at_intersection":    0,  # Crash at intersection, red light, stop sign, yield
        "nav_on_highway":         0,  # Freeway, highway, interstate
        "nav_in_parking_lot":     0,  # Parking lot / garage
        "nav_left_turn":          0,  # Left turn involved (AV or other party)
        "nav_lane_change":        0,  # Lane change / merge involved

        # Other party type
        "nav_vulnerable_user":    0,  # Pedestrian / cyclist / e-bike / scooter
        "nav_emergency_vehicle":  0,  # Police, fire, ambulance, emergency vehicle
        "nav_large_vehicle":      0,  # Bus, truck, semi, delivery vehicle

        # Special conditions
        "nav_av_disengaged":      0,  # Operator disengaged / took over just before crash
        "nav_hazard_lights":      0,  # Hazard / emergency lights on (unusual situation)
        "nav_double_parked":      0,  # Double-parked obstruction in road
        "nav_speed_mentioned":    0,  # Any mph / speed figure in narrative
        "nav_minor_damage_lang":  0,  # "minor damage" / "no visible damage" language
        "nav_has_narrative":      0,  # 1 if usable text existed, 0 if redacted
    }

    if is_redacted(raw_text):
        return flags

    text = clean_narrative(raw_text)
    if len(text) < 20:
        return flags

    flags["nav_has_narrative"] = 1
    t = text.lower()

    # --- AV state ---
    flags["nav_av_stopped"] = int(bool(re.search(
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b.*?"
        r"\b(stopped|stationary|parked|came to a stop|slowed to a stop|remained stopped|at rest)\b"
        r"|\b(stopped|stationary|parked|came to a stop|slowed to a stop|remained stopped)\b.*?"
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b",
        t, re.DOTALL
    )))

    flags["nav_av_moving"] = int(bool(re.search(
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b.*?"
        r"\b(traveling|proceeding|moving|driving|was in motion|drove)\b"
        r"|\b(traveling|proceeding|moving|driving)\b.*?"
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b",
        t, re.DOTALL
    )))

    flags["nav_av_turning"] = int(bool(re.search(
        r"\b(left turn|right turn|u-turn|uturn|turning left|turning right|initiating a.{0,10}turn|executing.{0,15}turn)\b",
        t
    )))

    flags["nav_av_changing_lanes"] = int(bool(re.search(
        r"\b(chang(ing|ed) lanes?|lane change|merg(ing|ed)|chang(ing|ed) into)\b",
        t
    )))

    flags["nav_av_reversing"] = int(bool(re.search(
        r"\b(revers(ing|ed)|backing up|backed up|in reverse)\b", t
    )))

    # --- Fault / contact ---
    # AV initiated contact
    flags["nav_av_struck_other"] = int(bool(re.search(
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle)\b.{0,80}"
        r"\b(struck|hit|made contact with|collided with|ran into)\b",
        t, re.DOTALL
    )))

    # Other party initiated contact (other party struck the AV)
    flags["nav_other_struck_av"] = int(bool(re.search(
        r"\b(struck|hit|made contact with|collided with|ran into|rear.ended|sideswiped)\b.{0,80}"
        r"\b(waymo av|cruise av?|zoox vehicle|av|subject vehicle|stationary waymo|stopped waymo)\b"
        r"|\b(other.{0,15}(vehicle|party|car|truck|bus|cyclist|pedestrian))\b.{0,80}"
        r"\b(struck|hit|made contact with|collided with)\b",
        t, re.DOTALL
    )))

    flags["nav_rear_approach"] = int(bool(re.search(
        r"\b(approached.{0,30}from behind|rear.end(ed)?|from the rear|"
        r"approach(ed|ing) from behind|rear of the|behind the (waymo|av|subject)|"
        r"collided with the rear)\b",
        t
    )))

    # --- Scenario / location ---
    flags["nav_at_intersection"] = int(bool(re.search(
        r"\b(intersection|red light|stop (sign|light)|traffic signal|yield sign|"
        r"crosswalk|at the light|at a light)\b",
        t
    )))

    flags["nav_on_highway"] = int(bool(re.search(
        r"\b(freeway|highway|interstate|i-\d+|us-\d+|expressway|on-ramp|off-ramp)\b",
        t
    )))

    flags["nav_in_parking_lot"] = int(bool(re.search(
        r"\b(parking lot|parking garage|parking structure|parking area)\b", t
    )))

    flags["nav_left_turn"] = int(bool(re.search(
        r"\b(left turn|turning left|turn(ing)? left|left-turn lane|dedicated.*left)\b", t
    )))

    flags["nav_lane_change"] = int(bool(re.search(
        r"\b(lane change|chang(ing|ed) lanes?|merg(ing|ed) (into|from)|"
        r"mov(ing|ed) (into|from).{0,20}lane)\b",
        t
    )))

    # --- Other party type ---
    flags["nav_vulnerable_user"] = int(bool(re.search(
        r"\b(pedestrian|cyclist|bicyclist|bicycle|e.?bike|scooter|"
        r"person on foot|moped|motorcyclist)\b",
        t
    )))

    flags["nav_emergency_vehicle"] = int(bool(re.search(
        r"\b(police|fire truck|firetruck|ambulance|emergency vehicle|"
        r"first responder|law enforcement|patrol car)\b",
        t
    )))

    flags["nav_large_vehicle"] = int(bool(re.search(
        r"\b(bus|semi.?truck|semi|tractor.?trailer|delivery truck|"
        r"garbage truck|box truck|heavy truck|18.?wheeler)\b",
        t
    )))

    # --- Special conditions ---
    flags["nav_av_disengaged"] = int(bool(re.search(
        r"\b(disengaged|took (manual )?control|operator took over|"
        r"manual override|disengagement|switched to manual)\b",
        t
    )))

    flags["nav_hazard_lights"] = int(bool(re.search(
        r"\b(hazard lights?|emergency lights?|flashers|hazard flashers?)\b", t
    )))

    flags["nav_double_parked"] = int(bool(re.search(
        r"\b(double.?parked|blocking (the|a) lane|stopped in (the|a) (travel )?lane|"
        r"obstructing (the|a) lane)\b",
        t
    )))

    flags["nav_speed_mentioned"] = int(bool(re.search(
        r"\b\d+\s*mph\b|\bspeed of \d+\b|\btraveling at \d+\b", t
    )))

    flags["nav_minor_damage_lang"] = int(bool(re.search(
        r"\b(minor damage|minimal damage|no visible damage|cosmetic damage|"
        r"no damage|undamaged|slight damage)\b",
        t
    )))

    return flags


# ---------------------------------------------------------------------------
# 4. Apply to full dataset
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. Extracting narrative scenario flags")
print("=" * 70)

flag_records = df_known["Narrative"].apply(extract_narrative_features)
flag_df = pd.DataFrame(list(flag_records), index=df_known.index)
df_known = pd.concat([df_known, flag_df], axis=1)

NAV_FEATURES = [c for c in flag_df.columns if c != "nav_has_narrative"]

print(f"   Extracted {len(NAV_FEATURES)} scenario flags (plus nav_has_narrative)")
print(f"\n   Flag prevalence across all known rows (sorted):")
prev = flag_df[NAV_FEATURES].mean().sort_values(ascending=False)
for feat, rate in prev.items():
    print(f"     {feat:<30}  {rate*100:5.1f}%")

# Save feature descriptions
desc_rows = []
descriptions = {
    "nav_av_stopped":        "AV was stopped/parked/stationary at time of impact",
    "nav_av_moving":         "AV was traveling/proceeding at time of impact",
    "nav_av_turning":        "AV was making a left/right/U-turn",
    "nav_av_changing_lanes": "AV was changing or merging lanes",
    "nav_av_reversing":      "AV was reversing/backing up",
    "nav_av_struck_other":   "AV initiated contact (AV struck the other party)",
    "nav_other_struck_av":   "Other party initiated contact (struck the AV)",
    "nav_rear_approach":     "Other party approached from behind / rear-end scenario",
    "nav_at_intersection":   "Crash occurred at intersection, red light, or stop sign",
    "nav_on_highway":        "Crash on freeway, highway, or interstate",
    "nav_in_parking_lot":    "Crash in parking lot or garage",
    "nav_left_turn":         "Left turn was involved in the crash sequence",
    "nav_lane_change":       "Lane change or merge was involved",
    "nav_vulnerable_user":   "Pedestrian, cyclist, or other vulnerable road user involved",
    "nav_emergency_vehicle": "Emergency or police vehicle involved",
    "nav_large_vehicle":     "Bus, semi-truck, or heavy vehicle involved",
    "nav_av_disengaged":     "Operator disengaged / took manual control near crash",
    "nav_hazard_lights":     "AV had hazard/emergency lights on (unusual situation flag)",
    "nav_double_parked":     "Double-parked or lane-blocking obstruction in the scene",
    "nav_speed_mentioned":   "A specific speed in mph was mentioned in the narrative",
    "nav_minor_damage_lang": '"Minor damage" or "no visible damage" language present',
}
for feat in NAV_FEATURES:
    desc_rows.append({
        "feature":     feat,
        "description": descriptions.get(feat, ""),
        "overall_pct": round(flag_df[feat].mean() * 100, 1),
        "severe_pct":  round(df_known.loc[df_known["severe"]==1, feat].mean() * 100, 1),
        "nonsevere_pct": round(df_known.loc[df_known["severe"]==0, feat].mean() * 100, 1),
    })
desc_df = pd.DataFrame(desc_rows)
desc_df.to_csv(OUT_DIR / "narrative_feature_descriptions.csv", index=False)
print(f"\n   Saved narrative_feature_descriptions.csv")

# ---------------------------------------------------------------------------
# 5. Rebuild train/test with narrative flags
# ---------------------------------------------------------------------------
train_df = df_known[df_known["era"] == "archived"].copy()
test_df  = df_known[df_known["era"] == "current"].copy()

# ---------------------------------------------------------------------------
# 6. Shared helpers
# ---------------------------------------------------------------------------
TABULAR_FEATURES = [
    "Automation System Engaged?",
    "Roadway Type",
    "Roadway-Wet Surface Condition",
    "Roadway-Work Zone",
    "Roadway-Traffic Incident",
    "Weather - Clear",
    "Weather - Rain",
    "Weather - Snow",
    "Weather - Fog/Smoke/Haze",
    "Weather - Severe Wind",
    "Weather - Cloudy",
    "Crash With",
    "SV Pre-Crash Movement",
    "CP Pre-Crash Movement",
    "SV Precrash Speed (MPH)",
    "Report Month",
]

def build_X(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    """One-hot encode categoricals + scale numerics. Returns (X_train, X_test)."""
    num_cols = [f for f in features if pd.api.types.is_numeric_dtype(train[f])]
    cat_cols = [f for f in features if f not in num_cols]

    Xtr = train[features].copy().astype({c: float for c in num_cols if c in train.columns})
    Xte = test[features].copy().astype({c: float for c in num_cols if c in test.columns})

    Xtr[num_cols] = Xtr[num_cols].fillna(Xtr[num_cols].median())
    Xte[num_cols] = Xte[num_cols].fillna(Xtr[num_cols].median())
    Xtr[cat_cols] = Xtr[cat_cols].fillna("Unknown")
    Xte[cat_cols] = Xte[cat_cols].fillna("Unknown")

    Xtr = pd.get_dummies(Xtr, columns=cat_cols, drop_first=False)
    Xte = pd.get_dummies(Xte, columns=cat_cols, drop_first=False)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    scaler = StandardScaler()
    if num_cols:
        train_num_idx = [Xtr.columns.get_loc(c) for c in num_cols if c in Xtr.columns]
        Xtr.iloc[:, train_num_idx] = scaler.fit_transform(Xtr.iloc[:, train_num_idx])
        Xte.iloc[:, train_num_idx] = scaler.transform(Xte.iloc[:, train_num_idx])

    return Xtr, Xte


def evaluate(name: str, y_true, y_pred, y_prob=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true)) > 1) else np.nan
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    print(f"\n  [{name}]")
    print(f"    Precision : {prec:.3f}")
    print(f"    Recall    : {rec:.3f}")
    print(f"    F1        : {f1:.3f}")
    print(f"    ROC-AUC   : {auc:.3f}" if not np.isnan(auc) else "    ROC-AUC   : N/A")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    False-negative rate: {fn_rate:.1%}")
    return dict(model=name, precision=prec, recall=rec, f1=f1, roc_auc=auc,
                TP=tp, FP=fp, FN=fn, TN=tn, fn_rate=fn_rate)


def run_lr(X_train, X_test, y_train, y_test, name: str) -> dict:
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    return evaluate(name, y_test, y_pred, y_prob)


# ---------------------------------------------------------------------------
# 7. ADS-only model comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. ADS-only model: tabular vs tabular + narrative features")
print("=" * 70)

ads_train = train_df[train_df["automation_level"] == "ADS"].copy()
ads_test  = test_df[test_df["automation_level"]  == "ADS"].copy()
print(f"   ADS train: {len(ads_train)} (severe={ads_train['severe'].sum()})")
print(f"   ADS test:  {len(ads_test)}  (severe={ads_test['severe'].sum()})")

# Tabular only
X_tr_tab, X_te_tab = build_X(ads_train, ads_test, TABULAR_FEATURES)
r_tab = run_lr(X_tr_tab, X_te_tab, ads_train["severe"], ads_test["severe"],
               "ADS — tabular only")

# Tabular + narrative flags
combined_features = TABULAR_FEATURES + NAV_FEATURES
X_tr_comb, X_te_comb = build_X(ads_train, ads_test, combined_features)
r_comb = run_lr(X_tr_comb, X_te_comb, ads_train["severe"], ads_test["severe"],
                "ADS — tabular + narrative")

# Narrative flags only (ablation)
X_tr_nav, X_te_nav = build_X(ads_train, ads_test, NAV_FEATURES)
r_nav = run_lr(X_tr_nav, X_te_nav, ads_train["severe"], ads_test["severe"],
               "ADS — narrative flags only")

ads_results = pd.DataFrame([r_tab, r_comb, r_nav])
ads_results.to_csv(OUT_DIR / "narrative_ads_model_comparison.csv", index=False)
print(f"\n   Saved narrative_ads_model_comparison.csv")

# Top narrative feature coefficients
lr_final = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="lbfgs")
lr_final.fit(X_tr_comb, ads_train["severe"])
coef_df = pd.DataFrame({
    "feature": X_tr_comb.columns,
    "coefficient": lr_final.coef_[0],
}).sort_values("coefficient", ascending=False)
nav_coefs = coef_df[coef_df["feature"].str.startswith("nav_")]
print("\n   Narrative feature coefficients (combined ADS model):")
print(nav_coefs.to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Pooled model comparison (ADS + L2 together, without automation_level)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. Pooled model: tabular (no automation_level) vs + narrative features")
print("=" * 70)

TABULAR_NO_LEVEL = [f for f in TABULAR_FEATURES if f != "automation_level"]

X_tr_p, X_te_p = build_X(train_df, test_df, TABULAR_NO_LEVEL)
r_pooled_tab = run_lr(X_tr_p, X_te_p, train_df["severe"], test_df["severe"],
                      "Pooled — tabular (no automation_level)")

X_tr_pn, X_te_pn = build_X(train_df, test_df, TABULAR_NO_LEVEL + NAV_FEATURES)
r_pooled_comb = run_lr(X_tr_pn, X_te_pn, train_df["severe"], test_df["severe"],
                       "Pooled — tabular + narrative (no automation_level)")

pooled_results = pd.DataFrame([r_pooled_tab, r_pooled_comb])
pooled_results.to_csv(OUT_DIR / "narrative_pooled_model_comparison.csv", index=False)
print(f"\n   Saved narrative_pooled_model_comparison.csv")

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. Summary")
print("=" * 70)
print(f"\n  ADS model improvement (ROC-AUC):")
print(f"    Tabular only:           {r_tab['roc_auc']:.3f}")
print(f"    Tabular + narrative:    {r_comb['roc_auc']:.3f}  "
      f"(Δ = {r_comb['roc_auc'] - r_tab['roc_auc']:+.3f})")
print(f"    Narrative flags only:   {r_nav['roc_auc']:.3f}")
print(f"\n  ADS false-negative rate:")
print(f"    Tabular only:           {r_tab['fn_rate']:.1%}")
print(f"    Tabular + narrative:    {r_comb['fn_rate']:.1%}")
print(f"\n  Pooled model (no automation_level) ROC-AUC:")
print(f"    Tabular only:           {r_pooled_tab['roc_auc']:.3f}")
print(f"    Tabular + narrative:    {r_pooled_comb['roc_auc']:.3f}")
print("\nDone.")
