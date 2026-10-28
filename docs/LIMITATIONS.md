# Limitations

**Version:** 1.0 (post-run)
**Scope:** every claim in this repository is bounded by the items below.
**Rule:** if a statement conflicts with this file, this file wins.

---

## 1. Intended use and hard non-use

**Intended:** research demonstration of leakage-aware benchmarking, uncertainty
reporting, provenance-preserving curation and transparent screening.

**Not intended / not validated for:**

- clinical decisions, patient-specific advice, or implant selection;
- regulatory submission, device qualification, or design allowables;
- biocompatibility, cytotoxicity, infection, wear, corrosion-in-service or fatigue
  claims;
- any statement that one material is "better for implants" than another.

## 2. The target is computed, not experimental

`log10(G_VRH)` is derived from DFT elastic tensors in Matbench v0.1. It is not:

- a measured shear modulus of a real specimen;
- an implant alloy design allowable;
- a hardness, yield strength, fatigue limit or fracture toughness;
- a biological or clinical outcome.

A model that predicts this descriptor well has demonstrated **statistical learning
of a computed property**, not materials qualification.

## 3. Benchmark distribution and interpolation

- 10,987 inorganic materials, 1–5 elements, dominated by simple crystalline
  compounds. Implant alloys are a small, atypical subregion.
- Official folds are random with respect to composition. The benchmark measures
  interpolation, not extrapolation to a new alloy family.
- The official hash and cache checksum pin one dataset snapshot. A Matbench update
  would invalidate the exact numbers.
- No structure features are used. This is deliberate, so the deployment path for a
  hypothetical composition matches the benchmark path, but it discards potentially
  useful structural information.

## 4. Model selection and tuning

- Three fixed configurations were compared: Dummy (mean), Ridge (α = 1.0), Random
  Forest (300 trees, `max_features=0.33`, `min_samples_leaf=1`).
- No hyperparameter search was performed. This avoids selection optimism but may
  understate achievable performance.
- Selection used mean OOF MAE only. RMSE and R² are reported for context; a
  different selection rule could select a different model.
- One random seed (42) was used. Seed sensitivity was not quantified.

## 5. Y-randomization: what it does and does not prove

- Random Forest randomized R² max -0.1116, MAE 0.3093; Ridge randomized R² max
  -0.0068, MAE 0.2952. Both collapse to Dummy-level performance.
- This is strong evidence against **label leakage through the evaluated pipeline**.
- It does not detect leakage that preserves the label relationship (e.g. near-duplicate
  structures across folds), and it does not validate the scientific meaningfulness of
  the target.
- Only 2 Random Forest permutations were run (runtime); Ridge used 5. More
  permutations would tighten the null distribution.

## 6. Uncertainty is marginal and post hoc

| Nominal | Empirical (leave-one-fold-out) | Mean half-width (log10 GPa) |
| --- | --- | --- |
| 0.80 | 0.8001 | 0.1570 |
| 0.90 | 0.9001 | 0.2349 |
| 0.95 | 0.9502 | 0.3190 |

- Split conformal assumes exchangeability between calibration and target rows. A new
  alloy family may violate this.
- Coverage is **marginal**: it can under-cover specific regions, especially near the
  applicability-domain boundary.
- Deployment half-widths use all OOF residuals and are slightly optimistic relative
  to the leave-one-fold-out coverage check.
- Intervals describe average error magnitude, not a posterior probability and not a
  guarantee for one candidate.

## 7. Applicability domain is support, not confidence

- Method: mean Euclidean distance to 5 nearest training neighbours in standardized
  Magpie space; thresholds at the 95th and 99.5th percentiles of training-fold
  self-distances.
- OOF counts: 10,433 `IN_DOMAIN`, 511 `NEAR_BOUNDARY`, 43 `OUT_OF_DOMAIN`.
- Deployment reference counts (all data): 10,870 / 106 / 11.
- By construction, ~95% of training rows are in-domain, so the label is relative to
  the training distribution, not an absolute physical judgement.
- `OUT_OF_DOMAIN` means "no close training neighbour", not "bad material".
- The distance is computed on composition descriptors only; two alloys with the same
  composition but different processing map to the same point.

## 8. Interpretability limits

See `docs/INTERPRETABILITY_LIMITS.md`. In short:

- permutation importance is associational, model-specific and fold-specific;
- correlated Magpie descriptors share/dilute importance;
- SHAP was not installed and no SHAP values are reported;
- importance does not identify a mechanism or a design variable to optimise.

## 9. Literature evidence limitations

- **Purposive curation, not a systematic review.** No PRISMA claim; publication and
  reporting bias are unquantified.
- 40 rows from 5 studies. Ra: 23 rows from 2 studies, 87% from one study.
- The locked endpoint cannot support a study-aware model; Phase 7 is
  `CURATION_ONLY`. This is the correct outcome, not a hidden failure.
- Values are laboratory-specific: instrument, cutoff, probe, drop volume, electrolyte
  and temperature differ across studies. Cross-study values are displayed separately,
  never pooled into a mean.
- Sa, Rz and Rq are not Ra; water and saline contact angles are not pooled;
  potentials are converted only when the reference electrode is stated.
- Dispersion, replicate counts and treatment parameters are often missing (`NaN`).
- Two dental-only studies were excluded; this reduces the evidence base but preserves
  the orthopaedic scope.
- Extraction is **AI-assisted** and requires independent human verification against
  the PDFs before thesis submission.

## 10. Screening limitations

- Constraints (modulus 1.40–2.00 log10 GPa, Ra ≤ 2.00 µm, contact angle ≤ 90°) are
  **research demonstration thresholds**, not clinical acceptance limits.
- Ascending predicted modulus reflects a stress-shielding research hypothesis (lower
  stiffness closer to bone), not a clinical requirement or a proven benefit.
- Of 14 candidates: 5 `RANKED`, 1 `CONSTRAINT_FAILED`, 8 `INDETERMINATE`.
- All Ti-6Al-4V variants receive the **same** predicted modulus because composition
  is identical; the model cannot distinguish surface states. This is a feature of the
  honest design, not a bug, and is why surface evidence is a separate layer.
- Priority sensitivity flips the top candidate: Ti-15Mo-5Zr-3Al (modulus-first) versus
  316L (roughness-first). No single "best" material is supported.
- `RANKED` means "all critical criteria known and within user constraints". It does
  not mean suitable, safe, manufacturable or clinically preferable.
- Non-critical contact-angle failure does not block ranking; it remains visible in the
  criterion matrix.
- No cost, availability, manufacturability, sterilization, wear or corrosion-in-service
  criteria are included.

## 11. App and reproducibility

- The app is a local demonstrator with no authentication, audit logging or clinical
  governance.
- It reads persisted artefacts; if an artefact is missing it shows a warning rather
  than a placeholder number.
- Seeds are fixed where feasible. Random Forest training is deterministic for a fixed
  seed and data, but floating-point results can differ at the last decimal across
  BLAS/hardware versions.
- `matbench==0.6` was installed with `--no-deps` because of an Apple Silicon pin
  conflict. This is a packaging deviation; dataset and folds are still official.
- Early log files label local time with a trailing `Z`; the formatter was corrected to
  UTC after the first runs. File artefacts use UTC timestamps correctly.

## 12. Hypotheses: final verdicts

| Hypothesis | Verdict | Actual evidence |
| --- | --- | --- |
| H1: Ridge beats the mean baseline | **Supported** | Ridge MAE 0.1568 vs Dummy 0.2931 |
| H2: Random Forest does not beat Ridge by a large margin | **Not supported** | RF MAE 0.1024 vs Ridge 0.1568 (gap 0.0544) |
| H3: Y-randomized models have R² ≤ 0.10 | **Supported** | RF randomized R² max -0.1116; Ridge max -0.0068 |
| H4: Surface evidence is insufficient for a model | **Supported** | 2 studies / 23 rows / 87% dominance → `CURATION_ONLY` |

## 13. What would change the conclusions

- An experimental modulus dataset for the same alloys, with processing and surface
  state recorded.
- ≥ 3 independent Ra studies with ≥ 25 compatible rows and no study above 40%.
- External validation on alloy families absent from Matbench.
- A prospective study linking surface descriptors to device outcomes.

Until then, treat every output as a **hypothesis for research**, not a finding for
practice.
