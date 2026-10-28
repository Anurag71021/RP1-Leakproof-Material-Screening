# Surface evidence audit

**Generated:** 2026-10-28T18:03:18+00:00 by `scripts/build_literature.py` from
`data/literature/surface_evidence_raw.csv`.

**Scope:** metallic orthopaedic implant materials only. Dental-only studies and
non-metallic controls are excluded.

## 1. Counts

| Metric | Value |
| --- | --- |
| Raw extracted rows | 40 |
| Duplicates removed | 0 |
| Clean rows | 40 |
| Distinct studies | 5 |
| Rows per study (min / median / max) | 3 / 6.0 / 20 |
| Top study share | 50.0% |

### Rows per study

| study_id | rows |
| --- | --- |
| `10.3390/ma14133702` | 20 |
| `10.3390/ma18050955` | 8 |
| `10.1038/s41598-020-76448-w` | 6 |
| `10.1016/j.heliyon.2024.e37808` | 3 |
| `10.3390/ijms27104335` | 3 |

## 2. Endpoints and units

| endpoint_key | rows | harmonised units |
| --- | --- | --- |
| `surface_roughness_ra_um` | 23 | um |
| `water_contact_angle_deg` | 9 | degrees |
| `corrosion_ecorr_v` | 8 | V |

Locked endpoint for any modelling attempt: **`surface_roughness_ra_um`**.
Rows for the locked endpoint: **23** across
**2** studies; largest study share
**87.0%**.

## 3. Missingness (not reported, never imputed)

| Field | Rows missing |
| --- | --- |
| `coating_chemistry` | 30 |
| `treatment_parameters` | 20 |
| `sd_or_sem` | 11 |
| `dispersion_type` | 11 |
| `n_replicates` | 34 |
| `measurement_method` | 0 |
| `reference_electrode` | 32 |
| `conditions_detail` | 0 |

## 4. Deduplication

Key: `(study_id, material_grade_or_formula, surface_treatment, coating_chemistry, endpoint_key, conditions_key)`.

0 duplicate row(s) removed. Full
removal log is in `results/validation/surface_gate.json`.

## 5. Exclusions

| DOI | PMCID | Reason |
| --- | --- | --- |
| `10.1016/j.heliyon.2020.e04762` | PMC7475236 | Dental-implant application context; out of the locked metallic orthopaedic scope. |
| `10.3389/fbioe.2025.1640122` | PMC12364894 | Dental-implant focus (sandblasting/acid etching of AM and machined titanium, osteoblast adhesion); out of scope. It also reports Sa, not Ra. |

Additional exclusions at screening: polystyrene control rows from PMC7666130
(not a metal), and any value available only as a figure without a numeric
table (not digitised).

## 6. Unit harmonisation

- Ra: nm ÷ 1000 → µm; all included Ra values were already in µm.
- Contact angle: degrees, no conversion.
- Ecorr: mV ÷ 1000 → V, only because the reference electrode (Ag/AgCl) is stated.
- Sa/Rz/Rq are **not** converted to Ra and are not pooled with Ra.
- Saline contact angles are **not** pooled with water contact angles.

## 7. Extraction method and its limits

Values were read from open-access full text via the Europe PMC full-text XML
service on 2026-10-28. Every row records the table/section and a note. Extraction
was AI-assisted. **Human verification against the PDF is required before thesis
submission.** Figure-only values were not digitised. The search was a targeted
purposive sample, not a systematic review; no PRISMA claim is made.

## 8. Phase 7 modelling gate

| Check | Result |
| --- | --- |
| a_min_studies | FAIL |
| b_min_rows | FAIL |
| c_no_dominant_study | FAIL |
| d_study_aware_split_feasible | FAIL |
| e_phase4_gate_passed | PASS |

Observed: 2 studies, 23 rows,
top study share 87.0%.

**Decision: `CURATION_ONLY`**

Reasons:

- only 2 distinct studies for surface_roughness_ra_um (< 3)
- only 23 deduplicated rows for surface_roughness_ra_um (< 25)
- largest study contributes 87% of rows (> 40%)
- fewer than 3 groups; LOGO/GroupKFold would not be meaningful

## 9. Honest interpretation

The curated table is an evidence catalogue. It is **not** a clinical dataset, it
does not establish biocompatibility, and it cannot support a generalisable
surface-response model at this sample size. The table is intentionally kept
small and traceable rather than padded with unverifiable numbers.
