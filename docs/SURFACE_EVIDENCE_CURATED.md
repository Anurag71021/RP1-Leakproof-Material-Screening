# Surface evidence - curation-only decision

**Decision:** `CURATION_ONLY`

The Phase 7 gate was applied before any surface model was trained.

## Why no surface model

- only 2 distinct studies for surface_roughness_ra_um (< 3)
- only 23 deduplicated rows for surface_roughness_ra_um (< 25)
- largest study contributes 87% of rows (> 40%)
- fewer than 3 groups; LOGO/GroupKFold would not be meaningful

## What is delivered instead

1. `data/literature/surface_evidence_raw.csv` - 40 extracted rows with DOI,
   table/figure location and extraction notes.
2. `data/literature/surface_evidence_clean.csv` - 40 harmonised,
   deduplicated rows with originals retained.
3. `docs/SURFACE_EVIDENCE_AUDIT.md` - counts, missingness, exclusions, units.
4. `data_cards/surface_evidence_data_card.md` - scope and limitations.
5. A screening layer that reports observed values, evidence level and
   `INDETERMINATE` status rather than a fabricated surface score.

## What would change the decision

- At least 3 independent studies (preferably 4+) reporting the same locked
  endpoint with compatible protocols.
- At least 25 deduplicated compatible rows.
- No single study contributing more than 40% of rows.
- A feasible leave-one-study-out split and a passing Phase 4 gate.

Until then, training a surface model would be overfitting a handful of
laboratory-specific measurements. Curation-only is the correct outcome.
