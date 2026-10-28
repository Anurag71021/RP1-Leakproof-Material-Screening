# Screening output - reading guide

**Generated:** 2026-10-28T18:10:44+00:00
**Model:** `random_forest` (deployment fit on all labelled rows)
**Target:** `log10(G_VRH)` in `log10(GPa)` - computed DFT shear modulus

> RESEARCH DEMONSTRATOR - Not for clinical use. No patient-specific advice. No implant recommendation implied.

## Constraint thresholds used

| Constraint | Value |
| --- | --- |
| Modulus (log10 GPa) | 1.4 - 2.0 |
| Ra max (um) | 2.0 |
| Contact angle max (deg, non-critical) | 90.0 |

These thresholds are **user research constraints**, not clinical acceptance
limits.

## Candidate status counts

| Status | Candidates |
| --- | --- |
| INDETERMINATE | 8 |
| RANKED | 5 |
| CONSTRAINT_FAILED | 1 |

## Ranked candidates

Lexicographic order: `predicted_log_gvrh -> surface_roughness_ra_um`. Ties are ties;
no weighted aggregate is computed. Ascending modulus reflects a stress-shielding
research hypothesis (lower stiffness is closer to bone), **not** a clinical
requirement; ascending Ra reflects the user's surface constraint.

| Rank | Candidate | Predicted log10(GPa) | Ra (um) |
| --- | --- | --- | --- |
| 1 | Ti-15Mo-5Zr-3Al - polishing level 2 | 1.568 | 0.170 |
| 2 | Ti-6Al-4V ELI - vibration tumbled | 1.570 | 0.280 |
| 3 | Ti-6Al-4V ELI - MAO + blasting | 1.570 | 0.500 |
| 4 | Ti-6Al-4V ELI - MAO (alkaline silicate) | 1.570 | 1.500 |
| 5 | 316L stainless steel - polishing level 2 | 1.872 | 0.060 |

## Indeterminate candidates

| Candidate | Why |
| --- | --- |
| Ti-6Al-4V AM - chemically polished (uncoated) | surface_roughness_ra_um unknown |
| Ti-6Al-4V AM - anodized nanoporous | surface_roughness_ra_um unknown |
| Ti-6Al-4V AM - anodized nanotubular | surface_roughness_ra_um unknown |
| Ti-6Al-4V - laser textured (P29) | surface_roughness_ra_um unknown |
| Ti-6Al-4V - laser textured + cyclohexane 70 C | surface_roughness_ra_um unknown |
| Ti-13Nb-13Zr - no surface evidence | surface_roughness_ra_um unknown |
| cp-Ti grade 4 - no surface evidence | surface_roughness_ra_um unknown |
| Mg AZ31 - no surface evidence | surface_roughness_ra_um unknown |

## Files

| File | Contents |
| --- | --- |
| `candidate_predictions.csv` | Prediction, 90% conformal interval, AD status, KNN distance |
| `candidates.csv` | Candidate identities and alloy family |
| `criterion_matrix.csv` | Long-format criterion-level evidence with source and evidence level |
| `candidate_summary.csv` | Status and transparent counts (not a score) |
| `ranking.csv` | Lexicographic ranking of fully supported candidates only |
| `sensitivity.json` | Whether the top candidate changes when priority order changes |

## Mandatory caveats

1. Modulus is predicted from **composition only**, using a model trained on a
   broad inorganic materials dataset. It is not measured on the candidate.
2. Ra and contact angle values come from specific laboratory protocols. Values
   from different studies are not interchangeable.
3. A candidate marked `RANKED` means "all critical criteria are known and meet
   the user's constraints". It does not mean "suitable for implantation".
4. `INDETERMINATE` means the evidence is missing, not that the candidate is bad.
5. Sensitivity: top candidates under alternative priorities were
   `ss316l_polished_2, ti15mo_polished_2`;
   flips observed: `True`.
