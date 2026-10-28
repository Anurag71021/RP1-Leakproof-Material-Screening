# Data card - surface evidence layer

**Status:** generated 2026-10-28T18:03:18+00:00 from `data/literature/surface_evidence_clean.csv`.

## 1. What this is

A hand-curated, provenance-preserving table of **measured surface properties**
reported in open-access studies of metallic orthopaedic implant materials.

| Item | Value |
| --- | --- |
| Raw rows | 40 |
| Clean rows | 40 |
| Distinct studies | 5 |
| Endpoints | surface_roughness_ra_um, water_contact_angle_deg, corrosion_ecorr_v |
| Locked endpoint | `surface_roughness_ra_um` (23 rows) |
| Modelling decision | `CURATION_ONLY` |

## 2. What this is not

- Not a clinical dataset.
- Not a biocompatibility or safety dataset.
- Not a systematic review; no PRISMA claim.
- Not a source of implant recommendations.
- Not imputed: missing values stay missing.

## 3. Provenance

Every row has a DOI, PMCID (where available), source URL, access status,
extraction location, extraction note, extractor and UTC timestamp. The raw table
is never edited after extraction; harmonisation writes a separate clean table
and preserves `value_original` / `unit_original`.

## 4. Scope and exclusions

Included: metallic orthopaedic implant materials (Ti alloys, Co-Cr alloys,
316L stainless steel). Excluded: dental-only studies, polymeric controls,
soft-tissue scaffolds, drug-eluting systems, and figure-only values that could
not be read numerically.

## 5. Known limitations

1. Small purposive sample; publication and reporting bias are unquantified.
2. Different laboratories, instruments and protocols; pooled values are not
   directly comparable.
3. Ra is not Rz/Sa; water contact angle is not a biological outcome.
4. Some dispersion values or replicate counts are not reported (`NaN`).
5. Extraction is AI-assisted and requires human verification before submission.
6. The evidence cannot support a generalisable predictive model
   (`CURATION_ONLY`).

## 6. Intended use

Decision-support display of what has actually been measured for a material and
surface treatment, with explicit evidence levels and unknowns.
