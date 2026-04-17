# Predicting Crash Severity in Autonomous Vehicle Incidents
## ISYE 4600 — Spring 2026
**Santiago Oscar Aramayo Velasco Sr. | Lauren McDonald | Luis Velez**

---

## 1. Problem Statement

Autonomous vehicles (AVs) are involved in thousands of reportable incidents each year. Not all crashes are equal: a minor fender-bender where a stopped AV is rear-ended at low speed is fundamentally different from a crash that injures an occupant, requires a tow, or deploys an airbag. Safety engineers and regulators need tools that can distinguish these outcomes so that limited review resources — engineering time, incident investigators, policy responses — are directed toward the crashes that matter most.

This project develops a binary classifier that predicts whether an AV incident is **severe** (defined below) given the structured context of the crash: roadway type, crash partner, speed, weather conditions, pre-crash movements, and scenario flags extracted from free-text narratives. The target stakeholders are AV safety teams and NHTSA regulators who triage incoming incident reports and prioritize follow-up action. A model that reliably catches severe crashes — even at the cost of some false alarms — reduces the chance of a missed-severity incident going uninvestigated.

The key business constraint is **false-negative rate**: every severe crash that the model predicts as non-severe is a potential operational risk that receives no priority review. Precision matters too — overwhelming safety reviewers with false alarms wastes investigative capacity. Our design target is precision ≥ 0.65 at the validation threshold, with recall maximized within that floor.

---

## 2. Data

**Source:** National Highway Traffic Safety Administration (NHTSA) Standing General Order (SGO) incident reports, downloaded as four CSV files: ADS-current, ADS-archived, L2-current, L2-archived.

**Automation levels:**
- **ADS (Automated Driving System):** Fully autonomous systems (Level 4+) such as Waymo, Cruise, and Zoox. The system controls all driving tasks; no human operator is expected to intervene.
- **L2 (Level 2 driver-assist):** Systems such as Tesla Autopilot where a human driver must remain engaged and is responsible for the driving task. The automation assists but does not replace the driver.

**Dataset size after cleaning:** 5,567 incidents with at least one severity component observed.

| Stratum | N | Severe rate |
|---|---|---|
| ADS — archived (train) | 1,519 | 26.0% |
| ADS — current (test) | 597 | 47.4% |
| L2 — archived (train) | 2,663 | 98.3% |
| L2 — current (test) | 788 | 97.2% |

**Cleaning steps applied in `01_clean_incidents.py`:**
1. Load and concatenate the four CSVs; add `era` and `automation_level` columns.
2. Deduplicate by SGO report ID; resolve conflicting duplicate rows by keeping the most-complete record.
3. Harmonize free-text fields (roadway type, crash partner, pre-crash movement) to a consistent controlled vocabulary.
4. Parse `Report Date` to extract `Report Month` (1–12) as a numeric seasonal feature.
5. Compute binary severity components (see Section 3) and the composite `severe` label.
6. Set `severity_known = 1` for rows with at least one non-null severity component; keep only these rows for modeling.

**Features available for modeling** (from `CONTEXT_FEATURES_BASE` in `baseline_common.py`):
- Roadway type, wet surface, work zone, traffic incident flag
- Weather indicators (clear, rain, snow, fog, severe wind, cloudy)
- Crash partner (`Crash With`)
- Subject vehicle (SV) and crash partner (CP) pre-crash movements
- SV pre-crash speed (MPH)
- Report month (captures seasonal patterns)
- Automation system engaged flag and automation level

**Narrative features** (extracted by `narrative_utils.py`): 21 binary scenario flags derived from free-text incident descriptions (e.g., `nav_av_stopped`, `nav_at_intersection`, `nav_rear_approach`). These are described in Section 5.3. No outcome words — "injured," "towed," "airbag deployed" — were used as features; doing so would constitute data leakage since those words are direct components of the severity label.

---

## 3. Severity Label Definition

**Feedback item #1:** *Clarify the exact definition of the severity label and justify why this operationalization reflects meaningful safety risk.*

A crash is labeled **severe = 1** if any of the following is true:

| Component | Operational meaning | Source field |
|---|---|---|
| `injury_flag` | At least one person reported injured | Injury-related fields in SGO report |
| `airbag_flag` | At least one airbag deployed | Airbag deployment field |
| `towed_flag` | At least one vehicle towed from the scene | Tow/disposition field |

**Composite rule:** `severe = injury_flag OR airbag_flag OR towed_flag`

**Justification:** Each component independently signals a crash outcome with real safety or operational consequence.

- **Injury** is the most direct indicator of physical harm to persons.
- **Airbag deployment** indicates the impact exceeded the manufacturer's calibrated threshold for deployment — typically a frontal or near-frontal delta-V above ~15 mph. A deployed airbag in the absence of injury still indicates a violent enough impact to trigger engineered safety systems.
- **Towing** indicates the vehicle sustained damage significant enough to render it undriveable. This captures property-damage-only crashes that are operationally severe even if no one was injured.

Using an OR rule ensures that any one of these outcomes flags the crash for priority review. A crash that requires a tow but results in no injury still represents an uncontrolled, damaging outcome that warrants investigation. The OR rule errs on the side of recall (catching all severe crashes) rather than precision — consistent with the asymmetric cost structure of safety-critical review: the cost of missing a truly severe crash outweighs the cost of reviewing a borderline case.

**Severity known:** If all three components are null (unreported), the row is excluded from modeling (`severity_known = 0`). This affects only a small fraction of the dataset (< 0.2%).

---

## 4. Train / Test Split Strategy

**Feedback item #2:** *Explicitly describe the train/test split strategy. Given policy implications, a temporal validation approach may better simulate deployment.*

We use a **temporal split**:
- **Training set:** All incidents in the "archived" era (reports filed before the current reporting period).
- **Test set:** All incidents in the "current" era (the most recent reporting wave).

This design directly simulates deployment: the model is trained on historically available data and evaluated on incidents that arrive after training is complete. A random 80/20 split would artificially inflate performance by allowing future incident patterns to leak into the training set — a form of temporal leakage that would overestimate real-world AUC.

**Within the training set,** a 75/25 stratified validation split is used for threshold selection. The model is fit on the 75% training sub-split, threshold candidates are evaluated on the held-out validation 25%, and the chosen threshold is applied to the test set exactly once at the end. This prevents threshold overfitting to test data.

**Temporal distribution shift:** The severe rate for ADS incidents increased from 26.0% in the archived (training) era to 47.4% in the current (test) era — a shift of +21 percentage points. This is the primary driver of model failure in the pooled approach (Section 6.1). L2 shows no meaningful shift (98.3% → 97.2%), which explains why a simple logistic regression is sufficient for L2 severity prediction.

---

## 5. Methods

### 5.1 Majority-class baseline

**Feedback item #4:** *Clearly define the majority-class baseline.*

The majority-class baseline always predicts the most frequent class. In the combined test set (ADS current + L2 current), the severe rate is approximately 75.7%, so the baseline always predicts **severe = 1**.

| Metric | Majority baseline |
|---|---|
| Precision | 0.757 |
| Recall | 1.000 |
| F1 | 0.862 |
| ROC-AUC | N/A (no probability output) |
| False-negative rate | 0.0% |
| False positives | 336 |

The majority baseline achieves zero false negatives by construction — it never predicts non-severe — but generates 336 false positives (the entire non-severe population). In an operational triage system, this means every incident gets flagged as high-priority, defeating the purpose of severity prediction entirely. Any useful model must do better than this on precision while keeping recall high.

### 5.2 Pooled logistic regression (baseline model)

A single logistic regression trained on all severity-known incidents (ADS + L2 combined), using the tabular context features in `CONTEXT_FEATURES_BASE`. Class weight = "balanced" to counteract the imbalance.

| Metric | Pooled LR |
|---|---|
| Precision | 0.975 |
| Recall | 0.730 |
| F1 | 0.835 |
| ROC-AUC | 0.856 |
| False-negative rate | 27.0% |
| False negatives | 283 |

The aggregate numbers look acceptable — until the results are disaggregated by automation level. The 283 false negatives are almost entirely ADS incidents. The model correctly identifies virtually all L2 severe crashes, but **predicts zero severe crashes for ADS on the test set** (100% false-negative rate, 283 missed severe crashes).

**Root cause:** The model learned that `automation_level = ADS` is a strong predictor of *non-severity* during training (26% severe rate). On the test set, the ADS severe rate is 47.4%. The model's learned shortcut becomes a liability when the distribution shifts. This motivates the stratified approach in Section 5.4.

### 5.3 Narrative feature extraction

ADS tabular features (roadway type, weather, pre-crash movement) provide **AUC ≈ 0.50** for ADS severity prediction — essentially random. The structured fields do not differ enough between severe and non-severe ADS crashes because most ADS incidents share the same context: street environment, stopped AV, passenger car collision partner.

The free-text incident narrative contains scenario dynamics that the structured fields cannot capture: whether the AV was moving or stationary, whether another vehicle approached from behind, whether the crash occurred at an intersection, and whether the narrative itself uses low-severity language ("minor contact").

**Extraction method** (`narrative_utils.py`): Regex patterns applied to the raw narrative text after stripping Waymo SGO boilerplate headers (which contain no crash-specific content). Each flag is a binary indicator (1 = pattern matched, 0 = not matched or narrative absent).

**21 narrative features:**

| Feature | Meaning |
|---|---|
| `nav_av_stopped` | AV was stationary at time of impact |
| `nav_av_moving` | AV was traveling at time of impact |
| `nav_av_turning` | AV was making a left/right/U-turn |
| `nav_av_changing_lanes` | AV was changing or merging lanes |
| `nav_av_reversing` | AV was reversing/backing up |
| `nav_av_struck_other` | AV initiated contact (AV struck the other party) |
| `nav_other_struck_av` | Other party initiated contact (struck the AV) |
| `nav_rear_approach` | Other party approached from behind |
| `nav_at_intersection` | Crash at intersection, red light, or stop sign |
| `nav_on_highway` | Crash on freeway, highway, or interstate |
| `nav_in_parking_lot` | Crash in parking lot or garage |
| `nav_left_turn` | Left turn involved in crash sequence |
| `nav_lane_change` | Lane change or merge involved |
| `nav_vulnerable_user` | Pedestrian, cyclist, or other VRU involved |
| `nav_emergency_vehicle` | Emergency or police vehicle present |
| `nav_large_vehicle` | Bus, semi-truck, or heavy vehicle involved |
| `nav_av_disengaged` | Operator disengaged / took manual control near crash |
| `nav_hazard_lights` | Hazard lights or disabled AV language |
| `nav_double_parked` | Double-parked obstruction present |
| `nav_speed_mentioned` | Speed value cited in narrative |
| `nav_minor_damage_lang` | Narrative uses low-severity language ("minor," "no damage") |

**Ablation results (ADS only):**

| Feature set | AUC (LR on ADS) |
|---|---|
| Tabular only | 0.500 |
| Narrative only | 0.608 |
| Tabular + Narrative | 0.531 |

Narrative features alone improve AUC by +0.11 over tabular-only. The small improvement when combining both for LR reflects the fact that linear models cannot capture the non-linear interactions between narrative flags and tabular context.

### 5.4 Stratified models per automation level

**Motivation:** Because ADS and L2 have radically different severity rates and different distribution shift patterns, a single pooled model cannot serve both. We train completely separate models for each automation level using the temporal split within that level.

**ADS feature set:** Tabular context + all 21 narrative flags (37 features total after one-hot encoding of categorical fields).

**L2 feature set:** Tabular context only (no narrative flags). Adding narrative features brings no measurable gain for L2 because the outcome is nearly deterministic from tabular context — 98% of L2 crashes are severe.

**Feature preparation (`build_X` in `05_stratified_models_ads_l2.py`):**
1. Fill numeric NaNs with the training-set median; fill categorical NaNs with "Unknown".
2. One-hot encode categorical columns (drop\_first=False to preserve all categories).
3. Align test columns to training columns (fill any unseen test categories with 0).
4. Apply `StandardScaler` to numeric columns only, fit on training data.

**Three algorithms evaluated per level:**

**Logistic Regression (LR):** `C=0.1` (L2 regularization), `class_weight="balanced"`, `max_iter=2000`, LBFGS solver. Linear decision boundary — sufficient for L2 but limited for ADS due to non-linear feature interactions.

**Random Forest (RF):** `n_estimators=300`, `max_features="sqrt"`, `class_weight="balanced"`. Ensemble of decision trees that capture arbitrary feature interactions. Grid search over `min_samples_leaf` and `max_depth` on the validation split.

**XGBoost (XGB):** `n_estimators=200`, `learning_rate=0.05`, `max_depth=5`, `scale_pos_weight = (neg / pos) / 2` to handle class imbalance, `eval_metric="logloss"`. Gradient-boosted trees — the strongest non-linear model in this benchmark.

**Threshold selection:** For each model, the decision threshold is tuned on the 25% held-out validation split using the following procedure: scan thresholds from 0.10 to 0.90; select the lowest threshold that achieves validation precision ≥ 0.65 (the precision floor). This ensures the model does not fire indiscriminately on every case.

---

## 6. Evaluation

### 6.1 Pooled LR failure — ADS dissection

| Metric | Pooled LR (ADS test only) | Pooled LR (L2 test only) |
|---|---|---|
| Precision | — | 0.975 |
| Recall | **0.0%** | 97.3% |
| False negatives | **283 / 283** | 21 |
| FN rate | **100%** | 2.7% |

The pooled model predicts zero severe crashes for ADS. It has learned that "ADS → not severe" from training data, and this shortcut fails when the test distribution shifts.

### 6.2 Stratified model results

All 6 model / level combinations evaluated on the current-era (held-out) test set:

| Model | Precision | Recall | F1 | AUC | FN | FN-rate |
|---|---|---|---|---|---|---|
| LR — ADS | 0.499 | 0.845 | 0.627 | 0.531 | 44 | 15.5% |
| RF — ADS | 0.804 | 0.883 | 0.842 | 0.893 | 33 | 11.7% |
| **XGB — ADS** | **0.831** | **0.922** | **0.874** | **0.921** | **22** | **7.8%** |
| LR — L2 | 0.985 | 0.967 | 0.976 | 0.876 | 25 | 3.3% |
| RF — L2 | 0.972 | 1.000 | 0.986 | 0.836 | 0 | 0.0% |
| XGB — L2 | 0.972 | 1.000 | 0.986 | 0.850 | 0 | 0.0% |

Test set: ADS current era = 597 incidents (283 severe); L2 current era = 788 incidents (766 severe).

**Best model per level:**
- **ADS → XGBoost.** AUC 0.921, precision 0.831, recall 0.922, FN-rate 7.8%. The +0.39 AUC improvement over LR (0.531 → 0.921) demonstrates that non-linear feature interactions between narrative flags and tabular context are critical for ADS severity prediction.
- **L2 → Logistic Regression** (preferred if false-alarm cost matters) or **RF/XGB** (preferred if zero FN is the priority). LR gives FN-rate 3.3% with only 11 false positives. RF/XGB give 0% FN but with 22 false positives.

**Why LR fails on ADS:** Logistic regression fits a single linear boundary in feature space. ADS severity prediction requires interactions among narrative flags — for example, `nav_av_stopped=1` AND `nav_rear_approach=0` AND `nav_at_intersection=1` may predict severity differently than any single flag alone. LR cannot represent this. Even with threshold tuning, LR on ADS achieves AUC 0.531 — barely better than random. The decision boundary is not learnable linearly.

**Why LR works on L2:** L2 crashes are 98% severe. The feature space is nearly linearly separable: most crashes are severe regardless of context. LR's linear boundary is sufficient, and AUC 0.876 reflects that the small non-severe fraction does have distinguishable features (mostly low-speed, parking-lot, minor-contact scenarios).

**XGBoost top features for ADS** (by gain importance):

| Feature | Importance |
|---|---|
| Report Month | 0.213 |
| nav_vulnerable_user | 0.075 |
| Crash With: Non-Motorist (Cyclist) | 0.069 |
| Crash With: Motorcycle | 0.045 |
| Crash With: Non-Motorist (Other) | 0.036 |
| nav_av_moving | 0.024 |
| SV Pre-Crash Movement (Other) | 0.023 |
| CP Pre-Crash Movement: Proceeding Straight | 0.017 |
| nav_minor_damage_lang | 0.016 |

`Report Month` is the top feature: ADS incidents filed in certain months show elevated severe rates, likely reflecting seasonal traffic volume or fleet deployment patterns. Crashes involving vulnerable road users (cyclists, pedestrians) and motorcycles are strongly predictive of severity — these crash partners have less crash protection and are more likely to sustain injury. `nav_av_moving` elevates severity risk when the AV itself was in motion at impact (higher kinetic energy). `nav_minor_damage_lang` in the narrative lowers the predicted severity probability.

---

## 7. False Negative Analysis

**Feedback item #5:** *Include structured error analysis focusing on false negatives (missed severe crashes) and explain the operational consequences.*

### 7.1 XGBoost — ADS false negatives (19 missed, 6.7% FN rate)

Out of 283 severe ADS test crashes, XGBoost missed 19 (FN-rate 6.7%).

**Profile of missed severe ADS crashes:**

| Dimension | Top category | Share |
|---|---|---|
| Roadway Type | Street | 84% |
| Crash With | SUV | 32% |
| SV Pre-Crash Movement | Stopped | 63% |
| CP Pre-Crash Movement | Proceeding Straight (32%) / Backing (26%) | — |

**Pattern:** Most missed ADS severe crashes involve a **stopped AV on a street being struck by an SUV or another vehicle proceeding straight or backing**. These are rear-end or backing scenarios where the narrative may not contain strong severity signals (the AV was passive, the crash was sudden), and the impact energy with an SUV may not be captured reliably in the structured fields.

**Operational consequence:** An AV safety engineer relying on the model to prioritize incidents would not queue these 19 crashes for immediate review. If the crash involves an SUV striking a stopped AV at sufficient speed to cause injury or vehicle damage, this miss could delay investigation of a potential AV safety issue (e.g., a failure to activate hazard lights while stopped, or a recurring rear-end pattern at a specific roadway type).

**Mitigation:** Supplement the model with a hard rule: any crash involving towing (a directly observable field) should always be flagged regardless of model score. Of the 19 missed severe crashes, those with `towed_flag = 1` can be caught by rule rather than model.

### 7.2 Logistic Regression — L2 false negatives (19 missed, 2.5% FN rate)

Out of 766 severe L2 test crashes, LR missed 19 (FN-rate 2.5%).

**Profile of missed severe L2 crashes:**

| Dimension | Top category | Share |
|---|---|---|
| Roadway Type | Highway / Freeway | 74% |
| Crash With | Other (see Narrative) | 37% |
| SV Pre-Crash Movement | Proceeding Straight | 42% |
| CP Pre-Crash Movement | Proceeding Straight | 79% |

**Pattern:** Most missed L2 severe crashes occur on **highways** where both vehicles were proceeding straight. The "Crash With: Other, see Narrative" category (37%) suggests the crash partner was an unusual vehicle type not captured in the structured taxonomy. These highway scenarios may produce lower model probability because highway crashes are slightly less severe on average in the training data compared to intersection crashes.

**Operational consequence:** Highway crashes involving L2 driver-assist at high speed are likely to involve significant force, occupant injury risk, and potential airbag deployment. Missing these cases would delay investigation of possible L2 automation failure scenarios (e.g., failure to respond to a vehicle stopped in the travel lane at highway speed).

**Mitigation:** For L2, switching from LR to RF or XGB eliminates all 19 false negatives at the cost of 22 additional false positives (22 non-severe crashes flagged as severe). If investigative capacity can absorb those extra alarms, RF/XGB is the preferred L2 model.

### 7.3 Summary

| Model | Total severe test | FN | FN-rate | Top missed context |
|---|---|---|---|---|
| XGB [ADS] | 283 | 19 | 6.7% | Street, stopped AV, SUV crash partner |
| LR [L2] | 766 | 19 | 2.5% | Highway, both proceeding straight |

Compared to the pooled LR baseline (283 ADS FN, 21 L2 FN = 304 total), the stratified models reduce total false negatives by 94% (304 → 38). All 283 previously missed ADS severe crashes are now correctly classified.

---

## 8. Cluster Analysis — ADS Crash Scenarios

**Feedback item #6:** *Justify feature selection and the choice of number of clusters, and clearly explain how clusters translate into actionable crash scenarios.*

### 8.1 Feature selection for clustering

We cluster on the same 37-feature matrix used by XGBoost: tabular context features (one-hot encoded) plus all 21 narrative flags. **Rationale:** Tabular features alone give AUC ≈ 0.50 for ADS, meaning they add no discriminative signal by themselves. But combined with narrative flags, the full feature space represents the *scenario dynamics* of each crash — not just its context (roadway type, weather) but what actually happened (AV stopped and struck from behind, AV moving through intersection, AV reversing in parking lot). Clustering this joint space groups incidents by scenario similarity rather than by superficial context similarity.

All features are standardized (zero mean, unit variance) before clustering so that numeric features (speed) are not dominated by binary flags.

### 8.2 Number of clusters — silhouette selection

We evaluated K-means with k ∈ {3, 4, 5, 6, 7, 8} on all 2,116 ADS incidents. The **average silhouette score** — which measures how much better each point fits its own cluster than the nearest alternative — was computed at each k:

| k | Silhouette score |
|---|---|
| 3 | 0.096 |
| 4 | 0.093 |
| 5 | 0.053 |
| 6 | 0.043 |
| **7** | **0.098** |
| 8 | 0.011 |

**k = 7** was selected as it achieves the highest silhouette score. The overall silhouette values are low (~0.10), consistent with ADS crash data that does not form tight natural clusters — most incidents share the same high-level context (urban street, stopped AV, passenger car). Clustering is still informative for isolating minority scenario types (animal strikes, fixed-object collisions, parking-lot incidents) that are obscured when analyzing the full dataset.

**Cluster labels** are derived using a lift-based approach: for each cluster, we compute the elevation of each narrative flag's rate above the global ADS mean. The narrative flag with the highest positive lift gives the dominant scenario signal for that cluster. When no flag is elevated by more than 5 percentage points (Cluster 2 — the modal scenario), the cluster is labeled as the "typical" baseline pattern.

### 8.3 Cluster profiles — actionable scenarios

| Cluster | N | % of ADS | Severe rate | Scenario |
|---|---|---|---|---|
| **C5** | 446 | 21.1% | **47.8%** | AV moving at impact |
| C1 | 82 | 3.9% | 41.5% | AV hit fixed object (parking lot) |
| C0 | 303 | 14.3% | 32.7% | Other vehicle struck stationary AV |
| C3 | 77 | 3.6% | 31.2% | AV struck other party |
| **C2** | 1,185 | **56.0%** | 26.0% | Typical street crash (modal baseline) |
| C4 | 22 | 1.0% | 0.0% | Animal strike |
| C6 | 1 | <1% | 0.0% | Single-incident outlier |

### 8.4 Actionable interpretation of clusters

**C5 — AV moving at impact (21% of ADS, 47.8% severe): Highest priority.**
This is the cluster with the highest severe rate. The dominant narrative signals are `nav_av_moving` (+0.17 lift) and `nav_other_struck_av` (+0.16 lift) — crashes where the AV itself was in motion when struck by another vehicle. These are not simple low-speed rear-ends; the kinetic energy from a moving AV being struck by another moving vehicle is higher, explaining the elevated severity rate. *Operational implication:* Incidents where the AV was moving and was struck by another vehicle should receive the highest investigation priority. Safety engineers should examine whether the AV's speed or trajectory contributed to the crash dynamics.

**C1 — AV hit fixed object, parking lot (4% of ADS, 41.5% severe): Parking and reversing risk.**
The dominant signal is `nav_in_parking_lot` (+0.22 lift) and `nav_av_disengaged` (+0.10 lift). These crashes are overwhelmingly against fixed objects (95% "Other Fixed Object") and frequently co-occur with AV disengagement events. This suggests the crash happened during a transition from automated to manual control in a confined space. *Operational implication:* Review the AV disengagement logs for these incidents. If the automation disengaged just before impact, the disengagement triggering conditions may need refinement for parking-lot scenarios.

**C0 — Other vehicle struck stationary AV (14% of ADS, 32.7% severe): Rear-end and side-swipe at rest.**
Elevated `nav_other_struck_av` (+0.09 lift) and `nav_large_vehicle` (+0.07 lift). The AV was stopped and struck — the classic "ADS as a moving obstacle" scenario, often involving large vehicles (trucks, buses) whose sightlines and stopping distances are longer. *Operational implication:* Assess whether AVs have adequate rear-hazard lighting or communication protocols when stopped in traffic to alert following vehicles.

**C3 — AV struck other party (4% of ADS, 31.2% severe): AV-at-fault collisions.**
Elevated `nav_av_struck_other` (+0.08 lift) and `nav_rear_approach` (+0.08 lift). Unlike C0 where the other vehicle struck the AV, here the AV initiated contact. *Operational implication:* These crashes warrant the most intensive review for potential AV control system failures. An ADS-level vehicle that strikes another party represents a failure of the autonomous driving decision system.

**C2 — Typical street crash (56% of ADS, 26.0% severe): Baseline scenario.**
No narrative flag elevated by more than 2 percentage points above global means. This is the modal ADS crash: stopped AV on a street, struck by a passenger car proceeding straight, with no distinctive scenario dynamics in the narrative. The 26% severe rate is close to the training-set baseline. *Operational implication:* These crashes can be triaged at lower priority but should still be reviewed to detect emerging patterns (e.g., a specific intersection type or time of day that is not captured by the current features).

**C4 — Animal strike (1% of ADS, 0% severe): No safety review required.**
All 22 animal-strike incidents in the dataset result in no severity outcome. The AV detects and responds to the animal, and the impact (if any) does not reach the injury/airbag/tow threshold. *Operational implication:* Animal-strike incidents can be deprioritized for safety review and instead routed to perception-system teams for sensor performance analysis.

---

## 9. Interpretation for the System

**Feedback item #7:** *Strengthen the "Interpretation for the System" section by translating model findings into concrete safety prioritization recommendations.*

### 9.1 Deployment architecture

The two models (XGB for ADS, LR or RF for L2) form a **two-stage triage system** for incoming NHTSA SGO incident reports:

```
Incoming report
     │
     ▼
automation_level?
     │
     ├── ADS → XGBoost model (narrative + tabular) → P(severe | ADS)
     │         threshold = 0.46
     │
     └── L2  → LR model (tabular only) → P(severe | L2)
               threshold = 0.20  (or RF for 0-FN policy)
     │
     ▼
  severe_flag = 1 → Priority Review Queue
  severe_flag = 0 → Standard Review Queue
```

Hard-rule override: Any report with `towed_flag = 1` (vehicle towed from scene) is forced into the Priority Queue regardless of model score. This catches the subset of false negatives where towing is directly observed in the structured data before narrative processing.

### 9.2 Concrete safety recommendations

**Recommendation 1 — Prioritize C5 (AV-moving-at-impact) and C1 (parking-lot/fixed-object) ADS incidents.**
These two clusters account for only 25% of ADS incidents but produce the highest severe rates (47.8% and 41.5% respectively). Routing them to a senior safety investigator within 48 hours of report receipt — rather than the standard 30-day review cycle — would ensure timely investigation of the crashes most likely to involve systemic ADS issues.

**Recommendation 2 — Audit C3 (AV-struck-other) incidents for control system failures.**
The 77 crashes where the ADS vehicle struck another party (31% severe) represent potential failures of the AV's collision-avoidance algorithms. These incidents should be cross-referenced with ADS disengagement logs and sensor failure reports to identify whether a perception or planning failure preceded the crash.

**Recommendation 3 — Investigate the Report Month signal.**
`Report Month` is the single highest-importance feature in XGBoost for ADS (importance 0.213). ADS severe rates vary seasonally, possibly due to fleet deployment size changes, seasonal driving conditions, or reporting lag patterns. NHTSA should evaluate whether the reporting pipeline introduces month-dependent biases that affect the quality of the training signal.

**Recommendation 4 — Implement a zero-FN policy for L2 using RF/XGB.**
The LR L2 model misses 19 severe crashes (2.5% FN rate), all of which are high-speed highway scenarios. Switching to RF or XGB for L2 eliminates all false negatives at the cost of 22 additional false alarms (2.8% of the L2 test set). Given that L2 highway crashes at speed represent serious injury risk, the operational trade-off favors zero FN: reviewing 22 extra cases per testing period is a low cost for ensuring no severe L2 highway crash is deprioritized.

**Recommendation 5 — Monitor for continued ADS distribution shift.**
The ADS severe rate increased +21 percentage points from the archived to the current era. If this trend continues (severe rate reaches 60–70% in the next era), the XGBoost model trained on 26% severe data will require retraining. A monitoring trigger should be set: if the rolling 90-day ADS severe rate exceeds 55%, trigger a model retraining event.

**Recommendation 6 — Use cluster labels as operational scenario tags.**
Tag each incoming ADS report with its cluster assignment (C0–C6) alongside the severity prediction. This gives safety engineers and regulators an immediate scenario context without reading the narrative. Reports tagged C3 (AV struck other) or C5 (AV moving at impact) should be escalated to senior review regardless of the severity prediction score.

---

## 10. Limitations

**Feedback item #3:** *Address potential reporting bias and data limitations and discuss how this affects generalization.*

### 10.1 Reporting bias and regulatory structure

**ADS vs. L2 reporting thresholds differ.** ADS operators (Waymo, Cruise, etc.) are required to file NHTSA SGO reports for *any* ADS-involved crash, including minor incidents with no damage or injury. L2 operators (primarily Tesla via Autopilot) report under a different requirement that may capture a different severity distribution. The result is that ADS reports skew toward minor incidents (74% non-severe in training), while L2 reports are overwhelmingly severe (98%). The models are trained on this regulatory artifact. If either reporting requirement changes, the training distribution changes, requiring model retraining.

**Underreporting of non-severe crashes** is probable. Crashes that result in no injury, no tow, and minor damage may not be consistently reported, particularly for L2 where the human driver — not a fleet operator — is responsible for deciding to file. This means the L2 "non-severe" class in our data is likely an undercount, and the true L2 severe rate in the real fleet may be lower than 98%.

**The ADS severe rate increase (+21 pp from archived to current era)** may reflect a reporting-practice change rather than a genuine worsening of crash outcomes. If ADS operators began reporting more severe incidents more systematically in the current era, the distribution shift is a data artifact rather than a fleet safety trend. The model cannot distinguish between these explanations.

### 10.2 ADS data scarcity

The XGBoost ADS model is trained on 1,519 archived rows, of which only ~395 (26%) are severe. Tree-based methods handle this imbalance with `scale_pos_weight`, but the small number of training positives limits the model's ability to learn fine-grained severity patterns. More ADS training data — from additional operators or expanded reporting periods — would be expected to improve AUC meaningfully.

### 10.3 Narrative coverage

Not all incidents contain useful narrative text. Some narratives are redacted (marked "CBI" — Confidential Business Information) or contain only Waymo's SGO boilerplate header with no crash-specific content. These rows contribute zeros for all `nav_*` features, which may reduce the XGBoost model's confidence for incidents where narrative would otherwise provide the clearest severity signal. The `nav_has_narrative` flag (84% overall coverage) provides partial mitigation.

### 10.4 L2 near-degenerate class distribution

The L2 test set contains only 22 non-severe cases out of 788 (2.8%). Precision and F1 metrics are highly sensitive to small count changes in this regime: one additional false positive changes precision from 0.985 to 0.974. The "perfect recall" achieved by RF and XGB on L2 reflects the overwhelming class imbalance — these models have learned that almost all L2 crashes are severe, and they are correct almost all the time by this heuristic alone. AUC (0.836–0.876) provides a more reliable indicator of true discriminative ability in this setting.

### 10.5 Cluster stability

The silhouette scores for ADS clustering are low (≈0.10), indicating that the clusters are not tight or well-separated. The k=7 solution is the best available, but the cluster boundaries are soft. Cluster assignments for individual incidents near the cluster boundary may differ across random seeds or slightly different feature preprocessing. The cluster-level severity rates and dominant scenario patterns are stable; individual assignments are not.

### 10.6 Generalizability

The models are trained exclusively on NHTSA SGO data from a specific regulatory reporting window. They may not generalize to:
- AV incidents that fall below the reporting threshold (no injury, no tow, minor damage)
- AV systems not represented in the SGO dataset (newer entrants, international fleets)
- Future automation levels (e.g., Level 3 systems) with different crash dynamics

The models should be viewed as **operational triage tools for reported incidents**, not as general-purpose AV safety predictors.

---

## 11. Reproducibility

All code is in `scripts/`. Run in order from the project root using the virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp  # macOS only — required for XGBoost

.venv/bin/python scripts/01_clean_incidents.py
.venv/bin/python scripts/02_run_baselines.py
.venv/bin/python scripts/05_stratified_models_ads_l2.py
.venv/bin/python scripts/06_narrative_features.py
.venv/bin/python scripts/09_stratified_fn_analysis.py
.venv/bin/python scripts/10_cluster_profiling.py
.venv/bin/python scripts/make_presentation_figures.py
```

All random seeds fixed at 42. Results are deterministic across runs.

**Key output files:**
- `Modeling/logistic_regression/all_stratified_results.csv` — all 6 model results
- `Modeling/logistic_regression/fn_summary.csv` — false-negative summary
- `Modeling/logistic_regression/ads_cluster_summary.csv` — cluster scenario labels
- `Presentation/figures/` — all figures (18 PNGs)
- `Modeling/MODEL_ANALYSIS.md` — extended technical rationale

---

*Report prepared April 2026 | ISYE 4600 Spring 2026 | Georgia Institute of Technology*
