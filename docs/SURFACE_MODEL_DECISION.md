# Surface model decision (Phase 7)

**Generated:** 2026-10-28T18:00:48+00:00
**Locked endpoint:** `surface_roughness_ra_um`

| Gate condition | Result |
| --- | --- |
| a_min_studies | FAIL |
| b_min_rows | FAIL |
| c_no_dominant_study | FAIL |
| d_study_aware_split_feasible | FAIL |
| e_phase4_gate_passed | PASS |

Observed: 2 studies, 23 rows,
largest study share 87.0%.

## Decision

`CURATION_ONLY`. No surface-response model was trained.

The gate was re-evaluated from the persisted evidence table at
2026-10-28T18:00:48+00:00; it was not inherited from an earlier run.

### Failed conditions

- only 2 distinct studies for surface_roughness_ra_um (< 3)
- only 23 deduplicated rows for surface_roughness_ra_um (< 25)
- largest study contributes 87% of rows (> 40%)
- fewer than 3 groups; LOGO/GroupKFold would not be meaningful

### What is delivered

A provenance-preserving evidence catalogue with explicit units, conditions and
unknowns. The screening app displays observed values where they exist and
`INDETERMINATE` where they do not. It does not invent a surface score.


## Non-negotiable limits

- Surface values are **experimental** and laboratory-specific; they are not
  predictions from the benchmark model.
- `log10(G_VRH)` from the benchmark is **computed DFT**, not an implant modulus.
- Neither layer establishes biocompatibility, corrosion resistance in service,
  wear performance, or clinical success.
