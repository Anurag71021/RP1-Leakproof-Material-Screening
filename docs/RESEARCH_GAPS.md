# Research gaps

**Derived from:** the actual benchmark and evidence artefacts generated on
2026-10-28, not from a general literature survey.

---

## 1. Benchmark-side gaps

| Gap | Evidence in this project | Consequence |
| --- | --- | --- |
| Computed target only | `log10(G_VRH)` is a DFT-derived descriptor; no experimental modulus exists in Matbench | A good score is not evidence about a real specimen |
| Broad inorganic distribution | 10,987 materials, 1–5 elements, dominated by simple crystalline compounds | Implant alloys are a small, atypical subregion; benchmark MAE is not an implant-AM accuracy claim |
| Random official folds | Official folds are random with respect to composition | Measures interpolation, not extrapolation to a new alloy family |
| Composition-only features | 132 Magpie descriptors; structure deliberately unused | Cannot represent processing, microstructure, texture, porosity, residual stress or surface state |
| Moderate absolute error | Best OOF MAE 0.1024 log10(GPa) ≈ a factor of 1.27 in GPa | Ranking near-ties is not supportable |
| No uncertainty in the benchmark fit | Split-conformal intervals are post hoc | Intervals are average-width statements, not per-material guarantees |
| Single dataset version | Matbench v0.1 snapshot with one official hash | Results are tied to that snapshot; a dataset update invalidates exact numbers |

## 2. Evidence-layer gaps

| Gap | Observed value | What would close it |
| --- | --- | --- |
| Too few studies for the locked endpoint | 2 studies for `surface_roughness_ra_um` | ≥ 3 independent studies, preferably ≥ 4 |
| Too few rows | 23 deduplicated Ra rows | ≥ 25 compatible rows |
| Single-study dominance | 87% of Ra rows from one study | No study > 40% of rows |
| Study-aware split infeasible | 2 groups; LOGO/GroupKFold would not be meaningful | ≥ 3 independent groups |
| Endpoint heterogeneity | Sa and Rz commonly reported instead of Ra | A pre-registered conversion or a separate endpoint model with justified conversion |
| Missing dispersion | Many rows have no SD/SEM or replicate count | Reported dispersion and n in primary studies |
| Missing conditions | Treatment parameters often absent or free text | Structured reporting of electrolyte, voltage, time, temperature, instrument |
| Reference electrodes | Only one study stated Ag/AgCl for Ecorr | Mandatory reference-electrode reporting |
| Dental/orthopaedic mixing | Two otherwise usable studies were dental-only | Clearly scoped orthopaedic surface studies |
| No clinical linkage | No curated study links a surface value to a clinical outcome | Prospective studies with device-level outcomes |

## 3. Methodological gaps

1. **No nested hyperparameter tuning.** Model settings were fixed a priori to avoid
   optimism. A nested CV study could test whether tuning helps, but must remain
   inside each official training fold.
2. **No structure-based comparison.** Structure features could improve benchmark
   accuracy, but would break the deployment path for hypothetical compositions.
   A dual-model design would need separate validation and explicit scope labels.
3. **No grouped importance.** Correlated Magpie descriptors share importance; grouped
   or conditional importance would be more interpretable.
4. **Learning curves on one fold only.** Runtime was bounded by using official fold 0;
   five-fold curves would give a stability estimate.
5. **No experimental validation of candidate ranking.** The screening output is a
   hypothesis list. Any ranking would require synthesis, surface processing,
   profilometry/wettability measurement and biological testing.
6. **No cost or manufacturability criteria.** Screening ignores powder availability,
   machining, heat treatment, sterilization and regulatory status.
7. **Threshold sensitivity is only two priority orders.** A fuller sensitivity
   analysis would vary modulus and Ra thresholds and report rank stability regions.

## 4. Highest-value next studies

1. **Standardised Ra round-robin.** Measure the same Ti-6Al-4V surface states with
   profilometry and AFM across laboratories, reporting Ra, Rz, Sa, instrument,
   cutoff and n. This directly addresses the 2-study, 87%-dominance problem.
2. **Orthopaedic-only surface corpus.** A preregistered, double-extracted corpus of
   orthopaedic implant surfaces with a fixed endpoint vocabulary.
3. **Experimental validation of one screening candidate.** Pick one `RANKED` or
   near-boundary candidate, manufacture it, measure Ra and contact angle, and compare
   with the curated values.
4. **Benchmark-to-device calibration.** Compare predicted `log10(G_VRH)` with measured
   shear modulus for a small set of implant alloys, reporting both DFT and
   experimental values rather than substituting one for the other.

## 5. What would **not** close a gap

- Adding more rows from the same study or laboratory (it worsens dominance).
- Merging Sa/Rz into Ra, or pooling water and saline contact angles.
- Training a surface model on 23 rows and reporting cross-validated R² as if it were
  generalisation.
- Adding a composite "suitability" score to hide missing evidence.
