# Literature review and evidence protocol

**Version:** 1.0
**Search/extraction date:** 2026-10-28 (UTC)
**Scope:** metallic orthopaedic implant materials and their surface states
**Status:** purposive curation, **not** a systematic review. No PRISMA claim is made.

---

## 1. Question this layer answers

> For metallic orthopaedic implant materials, what surface properties have actually
> been **measured and reported** in publicly accessible literature, under what
> conditions, and is the evidence sufficient to support a study-aware predictive
> model?

It does **not** answer whether a material is biocompatible, safe, or clinically
superior. Measured surface descriptors are not clinical outcomes.

## 2. Search strategy (as performed)

| Item | Detail |
| --- | --- |
| Indexes | NCBI PMC E-utilities (`esearch`/`esummary`) and Europe PMC full-text XML |
| Core query | `"Ti-6Al-4V" AND "contact angle" AND "roughness"` |
| Supplementary terms | `Ra`, `surface treatment`, `anodization`, `micro-arc oxidation`, `sandblasting`, `laser texturing`, `Ecorr` |
| Date filter | none applied; retrieved studies span 2020–2026 |
| Language | English |
| Access | open-access full text only |
| Screening | title/abstract first, then full text for numeric values in tables or text |
| Stopping rule | purposive: stop when the in-scope endpoint space was covered by the accessible studies found; **not** an exhaustive census |

Europe PMC's search endpoint returned HTTP 503 during the session, so discovery was
performed with NCBI E-utilities and full text was retrieved from the Europe PMC
`fullTextXML` endpoint by PMCID. Search dates and endpoint behaviour are recorded in
`docs/PROJECT_LOG.md`.

## 3. Inclusion and exclusion criteria

**Included**

- Metallic materials used in orthopaedic implants: Ti alloys, Co–Cr alloys,
  316L stainless steel, biodegradable Mg alloys.
- At least one numeric surface measurement readable from an open-access source.
- Clear material, surface treatment and test conditions.
- Orthopaedic or general implant-biomaterials context, not dental-only.

**Excluded**

- Dental-only studies (both identified studies are excluded, with reasons recorded).
- Polymeric or ceramic controls (e.g. polystyrene discs).
- Soft-tissue scaffolds, drug-eluting systems, purely in-silico studies.
- Values available only as an unlabeled figure, or where units/conditions cannot be
  established. Figure-only values were **not** digitised.
- Studies whose application context could not be verified from full text.

## 4. Extraction template

Every row in `data/literature/surface_evidence_raw.csv` records:

| Group | Fields |
| --- | --- |
| Identity | `study_id`, `doi`, `pmcid`, `title`, `year`, `source_url`, `access_status` |
| Provenance | `extraction_location`, `extraction_note`, `extractor`, `extracted_utc` |
| Material | `alloy_family`, `material_grade_or_formula`, `substrate_form` |
| Surface | `surface_treatment`, `coating_chemistry`, `treatment_parameters`, `series_note` |
| Endpoint | `endpoint_key`, `endpoint_label`, `value_original`, `unit_original`, `value`, `unit` |
| Statistics | `sd_or_sem`, `dispersion_type`, `n_replicates` |
| Conditions | `conditions_key`, `conditions_detail`, `measurement_method`, `reference_electrode` |
| Classification | `evidence_level`, `biological_assay`, `in_vivo`, `included_in_modelling`, `exclusion_reason` |

`NaN` means **not reported**, never zero.

## 5. Included studies

All titles below were read from the article XML. DOIs resolve to the cited articles.

| study_id (DOI) | Year | Journal | Material / surface | Extracted rows | Endpoints |
| --- | --- | --- | --- | --- | --- |
| `10.3390/ma14133702` (PMC8269856) | 2021 | Materials | CoCr, 316L (SUS), Ti-6Al-4V, Ti-15Mo-5Zr-3Al; five polishing levels | 20 | Ra |
| `10.1038/s41598-020-76448-w` (PMC7666130) | 2020 | Scientific Reports | Ti-6Al-4V ELI; vibration tumbling, micro-arc oxidation, MAO + blasting | 6 | Ra, water contact angle |
| `10.3390/ijms27104335` (PMC13207260) | 2026 | Int. J. Mol. Sci. | Ti-6Al-4V; uncoated, anodized nanoporous, anodized nanotubular | 3 | water contact angle |
| `10.3390/ma18050955` (PMC11901143) | 2025 | Materials | Ti-6Al-4V; as-received and heat-treated, with/without PEO | 8 | Ecorr |
| `10.1016/j.heliyon.2024.e37808` (PMC11417309) | 2024 | Heliyon | Ti-6Al-4V; polished, laser-textured, laser + cyclohexane | 3 | water contact angle |

### What each study reports (only what was extracted)

- **PMC8269856** reports Ra for four orthopaedic stem alloys at five polishing
  levels, with SD. The paper's focus is the stem–bone-cement interface; the title
  links increased polishing to excessive taper-slip. Extracted Ra values range from
  0.01 to 0.56 µm across alloys and levels.
- **PMC7666130** reports Ra and water contact angle for a tumbled control, a
  micro-arc-oxidized (MAO) surface, and MAO with the outer layer removed by
  blasting. Extracted Ra: 0.28, 1.50, 0.50 µm; water contact angle: 71.8°, 15.6°,
  35.4°. The polystyrene control rows were excluded.
- **PMC13207260** reports water contact angle for uncoated (63.0°), nanoporous
  (26.1°) and nanotubular (12.9°) Ti-6Al-4V surfaces in a femoral-defect model.
  Only the contact-angle endpoint was extracted; no dispersion was reported for it.
- **PMC11901143** reports Ecorr in mV vs Ag/AgCl for as-received and heat-treated
  Ti-6Al-4V with and without plasma electrolytic oxidation, in 0.9% NaCl. Values
  were converted to V only because the reference electrode is stated.
- **PMC11417309** reports water contact angle for polished, laser-textured, and
  laser + cyclohexane surfaces (100.3 ± 9.5°, 18.5 ± 5.2°, 151.1 ± 3.6°). It
  reports **Sa**, not Ra, so no roughness row was extracted. The abstract also
  reports a maximum of 155.1° for the treated sample; the mean ± SD is used.

## 6. Excluded studies

| DOI | PMCID | Reason |
| --- | --- | --- |
| `10.1016/j.heliyon.2020.e04762` | PMC7475236 | Dental-implant context (School of Dentistry; keywords list dental implants). Out of the locked orthopaedic scope. |
| `10.3389/fbioe.2025.1640122` | PMC12364894 | Dental-implant context (prosthetic dentistry; dental implant introduction). Also reports Sa, not Ra. |

## 7. Evidence map and gaps

| Endpoint | Rows | Studies | Largest study share | Modelling status |
| --- | --- | --- | --- | --- |
| `surface_roughness_ra_um` | 23 | 2 | 87% | **Locked endpoint; gate failed** |
| `water_contact_angle_deg` | 9 | 3 | 33% | Not modelled; descriptive only |
| `corrosion_ecorr_v` | 8 | 1 | 100% | Not modelled; descriptive only |

The locked endpoint has 23 rows, two studies, and one study contributing 87% of
rows. Any model would largely learn one laboratory's polishing protocol. The gate
therefore returns `CURATION_ONLY`.

Other gaps observed:

1. Most studies report **Sa** or **Rz**, not Ra, so they cannot contribute to the
   locked endpoint without an invalid unit merge.
2. Dispersion and replicate counts are often missing.
3. Surface treatments are described in free text with different parameter names,
   preventing reliable harmonisation.
4. Corrosion potentials are rarely reported with a stated reference electrode;
   only one study here did.
5. No study in the curated set reports a clinical outcome linked to the measured
   surface value.
6. Orthopaedic and dental literature are frequently mixed; strict scope filtering
   removes otherwise usable surface measurements.

## 8. Extraction quality statement

Extraction was AI-assisted and every row records the table/section used. **Human
verification against the PDF is required before thesis submission.** The numbers
were checked against the article XML tables during extraction; this is not a
substitute for independent double extraction.

No value in this layer was imputed, averaged across incompatible protocols, or
taken from a figure without a numeric label.

## 9. How this layer is used

- The app displays measured values with DOI-level provenance.
- The screening engine compares candidates against user constraints using the
  locked endpoint only where it exists.
- Missing evidence produces `INDETERMINATE`, not a low score.
- No biological compatibility score is produced from these physical measurements.
