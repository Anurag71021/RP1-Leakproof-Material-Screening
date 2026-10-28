# Interpretability limits

**Applies to:** permutation importance and any coefficient-based reading of the
benchmark models.
**Status:** binding interpretation rules, not suggestions.

---

## 1. What was actually computed

| Item | Implementation |
| --- | --- |
| Primary method | `sklearn.inspection.permutation_importance` on each official held-out fold |
| Scoring | `neg_mean_absolute_error` (decrease in MAE when a feature is shuffled) |
| Repeats | 5 per feature per fold |
| Sample cap | up to 1,500 held-out rows per fold |
| Aggregation | mean across the five folds; SD across folds reported |
| Features | 132 Magpie composition descriptors |
| SHAP | **not installed, not run.** No SHAP values are reported or implied |

Permutation importance is computed on the *transformed, imputed, scaled* feature
matrix after the pipeline is fitted inside the fold. The held-out fold is used only
for the importance measurement, never for fitting.

## 2. Association is not causation

A high permutation importance means: **this fitted model's held-out MAE gets worse
when this column is shuffled.** It does not mean:

- the descriptor causes the shear modulus;
- the descriptor has a physical mechanism;
- changing an alloy to increase that descriptor will increase stiffness;
- the descriptor is relevant to biocompatibility, corrosion or clinical outcome.

Magpie descriptors are summaries of elemental property tables. They are correlated
with each other and with composition in complex ways. Importance is a property of
*this model on this dataset*, not of nature.

## 3. Correlated features dilute and share importance

Permutation importance assigns low importance to a feature if another correlated
feature can substitute for it. Conversely, it can assign importance to a feature
that is merely a proxy for composition. Consequences:

- A low-ranked feature is **not** evidence that the underlying element property is
  irrelevant.
- A high-ranked feature is **not** evidence that it should be optimised.
- Rankings can change if the feature set, model class or random seed changes.
- Grouped or conditional importance would give different answers; neither is
  reported here.

## 4. Model-specific and fold-specific

- Importance is only meaningful for the selected `RandomForestRegressor`
  configuration. It does not transfer to Ridge or to another model class.
- Fold-to-fold SD is reported because a single fold can produce a different order.
  Treat differences within the SD as ties.
- The sample cap (1,500 rows/fold) adds sampling noise. It was chosen to bound
  runtime and is recorded in the artefact metadata.

## 5. Extrapolation and applicability domain

- Importance says nothing about behaviour outside the training distribution.
- The applicability domain is a nearest-neighbour distance in standardized
  descriptor space. It is a support indicator, not a confidence probability.
- Compositions with no close training neighbour can receive a plausible-looking
  prediction. The app labels these `NEAR_BOUNDARY` or `OUT_OF_DOMAIN`, and such
  predictions must not be compared or ranked.

## 6. Calibration limits

- Split-conformal intervals are calibrated on **out-of-fold absolute residuals**.
  They describe average error magnitude, not a posterior distribution.
- Deployment half-widths are computed from all OOF residuals and are therefore
  slightly optimistic relative to the leave-one-fold-out coverage check.
- Coverage is marginal: it can under-cover specific composition regions,
  especially near the domain boundary.
- No interval is a guarantee for an individual candidate.

## 7. Failure modes to state in any write-up

1. **Proxy learning.** The model may rely on composition proxies for dataset
   membership rather than on physical elastic behaviour.
2. **Distribution shift.** Implant alloys (Ti-6Al-4V, CoCr, 316L) are a tiny and
   atypical part of a broad inorganic dataset. Predictions for them are not
   validated by the benchmark score alone.
3. **Surface blindness.** Composition-only features cannot represent surface
   treatment, roughness, porosity, texture or coating state, all of which dominate
   implant performance.
4. **Computed target.** The target is a DFT elastic descriptor. Agreement with DFT
   is not agreement with experiment.
5. **No clinical endpoint.** Nothing in this project is validated against patient
   outcomes.

## 8. Mandatory wording

Use:

> "Permutation importance for the fitted Random Forest under official folds;
> associational only, not causal."

Do not use:

> "The model shows that X controls stiffness", "X drives implant success",
> "important for biocompatibility", or any causal or clinical phrasing.
