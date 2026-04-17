# Model Analysis — AV Crash Severity Prediction
## ISYE 4600 | Spring 2026

---

## 1. Why We Split ADS and L2

The data contains two automation levels: **ADS** (fully autonomous systems, e.g. Waymo, Cruise)
and **L2** (driver-assist systems, e.g. Tesla Autopilot). Treating them as a single dataset
causes a critical failure.

### The distribution shift problem

When we trained a single pooled logistic regression on both groups, the model
correctly identified that L2 crashes are almost always severe (98% severe rate in
training) and that ADS crashes are mostly not severe (26% severe rate in training).
It learned this as a shortcut: **"ADS → not severe."**

On the test set, ADS crashes had a 47% severe rate — nearly double the training rate.
The pooled model, locked into its learned shortcut, predicted 0 severe crashes for
ADS: **100% false-negative rate, 283 missed severe crashes.**

| Slice | Train severe rate | Test severe rate | Shift |
|---|---|---|---|
| ADS | 26.0% | 47.4% | +21 pp |
| L2  | 98.3% | 97.2% | −1 pp  |

L2 has no meaningful shift. ADS has a large one. A pooled model cannot handle both.

**Fix:** train completely separate models on ADS-only and L2-only data, using a
temporal split within each group (archived era → train, current era → test).

---

## 2. Feature Sets

### ADS: narrative + tabular features

Tabular context features (roadway type, weather, pre-crash movement, speed) give
**AUC ≈ 0.50** for ADS — essentially random discrimination. The structured fields
don't differ enough between severe and non-severe ADS crashes.

Narrative scenario flags extracted from the free-text incident description
(`nav_av_stopped`, `nav_on_highway`, `nav_at_intersection`, `nav_minor_damage_lang`, etc.)
carry more signal because they capture crash *dynamics* rather than just context.
Used alone they give AUC ≈ 0.61. Combined with tabular features they give the RF
a richer feature space to find non-linear interactions.

**Design constraint:** no outcome words (injured, towed, airbag deployed) were used
as features — those are components of the label itself and would be data leakage.

### L2: tabular features only

L2 crashes are 97–98% severe. The near-single-class nature of the problem means
even simple features (roadway type, pre-crash speed, crash partner) are sufficient.
Adding narrative flags brings no measurable gain and adds noise.

---

## 3. Algorithm Choice

### Why LR fails on ADS

Logistic regression fits a linear decision boundary in feature space. The relationship
between narrative scenario flags and ADS crash severity is non-linear — for example,
the combination of `nav_av_stopped=1` AND `nav_rear_approach=0` AND `nav_on_highway=1`
may predict severity differently than any single flag alone. LR cannot capture this.

The result: even with threshold tuning, LR on ADS achieves AUC ≈ 0.53 and
precision ≈ 0.50 — barely better than random. The model fires on roughly every other
case regardless of how the threshold is tuned, because the underlying probabilities
are not well calibrated.

### Why RF works on ADS

Random Forest builds an ensemble of decision trees, each of which can learn arbitrary
feature interactions (if `nav_av_stopped=1` AND `nav_rear_approach=0` → go right in
the tree). This non-linear capacity is exactly what ADS severity prediction requires.

RF on ADS achieves AUC 0.893 — a jump of +0.36 over LR — with precision 0.804 and
recall 0.883. The precision floor of 0.65 is met comfortably on the validation set.

### Why LR works on L2

L2 crashes are 98% severe in training. The feature space is nearly linearly separable:
almost anything that looks like a crash predicts severity. LR's linear boundary is
sufficient. AUC 0.876, F1 0.976.

### Why RF also works on L2 (and gets perfect recall)

RF on L2 achieves 100% recall (0 false negatives) by learning that essentially all
L2 crashes in this dataset are severe. It trades slightly more false alarms (22 vs 11
for LR) for zero misses. Both models are strong here; the choice between them is a
policy decision about whether false alarms or missed cases are more costly.

---

## 4. Results

| Model | Precision | Recall | F1 | AUC | FN-rate | FP | FN |
|---|---|---|---|---|---|---|---|
| LR — ADS | 0.499 | 0.845 | 0.627 | 0.531 | 15.5% | 240 | 44 |
| **RF — ADS** | **0.804** | **0.883** | **0.842** | **0.893** | **11.7%** | **61** | **33** |
| LR — L2 | 0.985 | 0.967 | 0.976 | 0.876 | 3.3% | 11 | 25 |
| RF — L2 | 0.972 | **1.000** | **0.986** | 0.836 | **0.0%** | 22 | 0 |

Test set is the current era only (deployment-style evaluation).
Threshold selected on a 25% held-out validation split of the archived training data.

**Best model per level:** RF for ADS, either for L2 (RF if zero FN is the priority,
LR if false alarms are more costly).

---

## 5. Limitations

**ADS data scarcity.** Training the ADS model on 1,519 archived rows with only 26%
severe rate means the model sees ~395 severe examples. Tree-based methods handle
this with class weighting, but more data would improve stability.

**Temporal distribution shift.** ADS severe rate nearly doubled from archived (26%)
to current (47%) era. This is likely a real shift in reporting patterns or fleet
behavior, not sampling noise. Models trained on archived data may continue to
underperform as the distribution evolves.

**Narrative coverage.** Not all incidents have usable narratives — some are redacted
(CBI/confidential) or contain only boilerplate SGO header text. Rows with no usable
narrative contribute zeros for all `nav_*` features, which may hurt ADS precision
at the margin.

**L2 near-degenerate.** L2 test set has only 22 non-severe cases out of 788 (2.8%).
Metrics like precision and F1 are sensitive to small count changes in this regime.
The "perfect recall" of RF on L2 reflects the overwhelming class imbalance as much
as model quality.

---

## 6. Reproducibility

Run in order from the project root (`.venv/bin/python scripts/<name>`):

```
01_clean_incidents.py          # data cleaning
02_run_baselines.py            # pooled majority + LR baselines
05_stratified_models_ads_l2.py # all 4 stratified combinations (this analysis)
06_narrative_features.py       # narrative flag ablation study
make_presentation_figures.py   # all presentation figures
```

Key outputs:
- `Modeling/logistic_regression/all_stratified_results.csv` — all 4 model results
- `Modeling/logistic_regression/lr_stratified_by_level_results.csv` — best per level
- `Presentation/figures/13_2x2_model_comparison.png` — 2×2 results figure
