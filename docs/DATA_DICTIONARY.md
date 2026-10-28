# Data dictionary - RP-1

All tables are CSV/JSON in this repository except the immutable raw pickle.
`NaN` means **not reported or not measurable**, never zero.

---

## 1. Benchmark table (`data/raw/matbench_log_gvrh.pkl.gz`)

| Column | Type | Units | Description | Missing |
| --- | --- | --- | --- | --- |
| `structure` | pymatgen `Structure` (pickle) | Å, degrees | Relaxed crystal structure from the Materials Project | none |
| `log10(G_VRH)` | float | log10(GPa) | Base-10 log of the Voigt–Reuss–Hill average shear modulus | none |
| *index* | string | – | Matbench material ID, e.g. `mb-log-gvrh-00001`. Used for tracing only; **never a feature** | none |

Derived bookkeeping (not features): `row_index` (integer position), `fold_id`
(0–4), `y_true`, `y_pred`.

Target facts at ingestion: n = 10987, min = 0.0000, max = 2.7185,
mean = 1.5536, std = 0.3716 (log10 GPa; 10^x spans 1 to 523 GPa). See
`reports/data_validation.json` for the authoritative generated values.

---

## 2. Surface evidence table

Files:

- `data/literature/surface_evidence_raw.csv` - exactly as extracted, one row per
  reported value.
- `data/literature/surface_evidence_clean.csv` - harmonised, deduplicated,
  analysis-ready. Original values retained in `*_original` columns.

### 2.1 Identifiers and provenance

| Column | Type | Allowed / units | Required | Description |
| --- | --- | --- | --- | --- |
| `study_id` | string | DOI lower-case, optional `_S1` suffix | yes | Stable study identifier. One DOI = one study unless independent series are justified in `series_note`. |
| `doi` | string | DOI | yes | Digital Object Identifier |
| `pmcid` | string | `PMC...` or NaN | no | Open-access full-text identifier when available |
| `title` | string | – | yes | Study title |
| `year` | integer | 1990–2026 | yes | Publication year |
| `source_url` | string | https URL | yes | Publicly accessible location used for extraction |
| `access_status` | enum | `open_access`, `abstract_only` | yes | Whether values came from full text or abstract |
| `extraction_location` | string | e.g. `Table 2`, `Fig. 3b`, `Section 3.2` | yes | Exact place the value was read |
| `extraction_note` | string | short quote / paraphrase | yes | What the source states, in our words |
| `extractor` | string | initials | yes | Who extracted the row |
| `extracted_utc` | ISO-8601 | UTC | yes | When the row was created |

### 2.2 Material and surface

| Column | Type | Allowed / units | Required | Description |
| --- | --- | --- | --- | --- |
| `alloy_family` | enum | `Ti_alloy`, `CoCr_alloy`, `stainless_steel`, `Mg_alloy`, `other_metal` | yes | Broad metallurgical family |
| `material_grade_or_formula` | string | e.g. `Ti-6Al-4V ELI`, `316L` | yes | Grade or formula as reported |
| `substrate_form` | enum | `bulk`, `coating`, `porous`, `additive` | yes | Physical form |
| `surface_treatment` | string | controlled-ish text | yes | e.g. `anodization`, `sandblasting`, `SLA`, `electropolishing`, `plasma spray` |
| `coating_chemistry` | string | formula or NaN | no | Coating composition if any; NaN for uncoated |
| `treatment_parameters` | string | free text | no | Voltage, time, temperature, electrolyte, etc. |
| `series_note` | string | free text or NaN | no | Justification for splitting one DOI into multiple study_ids |

### 2.3 Endpoint and value

| Column | Type | Allowed / units | Required | Description |
| --- | --- | --- | --- | --- |
| `endpoint_key` | enum | see §2.5 | yes | Normalised endpoint name |
| `endpoint_label` | string | as reported | yes | Exact source wording |
| `value_original` | float | – | yes | Value exactly as reported |
| `unit_original` | string | as reported | yes | e.g. `nm`, `µm`, `degrees`, `mV vs SCE` |
| `value` | float | harmonised unit | yes | Converted value, only if conversion is unambiguous |
| `unit` | string | harmonised unit | yes | Target unit for the endpoint |
| `sd_or_sem` | float | same as `value` | no | Dispersion, only if reported and identified |
| `dispersion_type` | enum | `SD`, `SEM`, `CI95`, `range`, NaN | no | Which dispersion was reported |
| `n_replicates` | integer | ≥ 1 or NaN | no | Number of replicates/samples |
| `conditions_key` | string | e.g. `SBF_37C_7d`, `air_25C` | yes | Normalised test condition |
| `conditions_detail` | string | free text | no | Electrolyte, temperature, time, cell line, etc. |
| `measurement_method` | string | e.g. `profilometry`, `AFM`, `sessile drop` | no | How the value was measured |
| `reference_electrode` | enum | `SCE`, `Ag/AgCl`, `SHE`, NaN | no | Required for electrochemical potentials |

### 2.4 Evidence classification

| Column | Type | Allowed | Required | Description |
| --- | --- | --- | --- | --- |
| `evidence_level` | enum | `observed_literature`, `literature_extracted`, `benchmark_predicted`, `unknown` | yes | Strength/type of evidence |
| `biological_assay` | string | e.g. `MG-63 viability 24 h` or NaN | no | Biological endpoint, kept separate from physical endpoints |
| `in_vivo` | boolean or NaN | `TRUE`, `FALSE`, NaN | no | Whether an animal/human study is involved |
| `included_in_modelling` | boolean | `TRUE`, `FALSE` | derived | Assigned by the Phase 7 gate, not by hand |
| `exclusion_reason` | string | free text or NaN | no | Why a row is excluded from modelling |

### 2.5 Endpoint vocabulary (locked)

| `endpoint_key` | Harmonised unit | Conversion rule |
| --- | --- | --- |
| `surface_roughness_ra_um` | µm | nm ÷ 1000; otherwise only if source states Ra |
| `surface_roughness_rz_um` | µm | kept separate from Ra; never merged |
| `water_contact_angle_deg` | degrees | no conversion |
| `corrosion_ecorr_v` | V | mV ÷ 1000, only when reference electrode is stated |
| `corrosion_icorr_ua_cm2` | µA/cm² | no conversion; different units are not merged |
| `oxide_layer_thickness_nm` | nm | µm × 1000 |
| `hardness_hv` | HV | no conversion |
| `elastic_modulus_gpa` | GPa | only experimental values; never Matbench predictions |

Rules:

- Ra and Rz are different quantities. They are never averaged or merged.
- Potentials without a stated reference electrode are not converted.
- Different cell lines, assays, or exposure times are not merged into one
  " biocompatibility" endpoint.
- A value read only from a figure is flagged in `extraction_note` as
  `digitised_from_figure` and is not used for modelling.

### 2.6 Deduplication key

```
(study_id, material_grade_or_formula, surface_treatment,
 coating_chemistry, endpoint_key, conditions_key)
```

When duplicates exist, the first complete row is kept and the dropped rows are
logged with their original values in `docs/SURFACE_EVIDENCE_AUDIT.md`.

---

## 3. Screening output (`results/screening/`)

### 3.1 `criterion_matrix.csv`

Long format, one row per (candidate, criterion):

| Column | Type | Description |
| --- | --- | --- |
| `candidate_id` | string | Stable candidate key |
| `candidate_label` | string | Human-readable alloy/grade |
| `criterion` | string | e.g. `predicted_log_gvrh`, `surface_roughness_ra_um` |
| `value` | float or NaN | Value, NaN if unknown |
| `units` | string | Units of the value |
| `source` | string | DOI, OOF file, or provenance record |
| `evidence_level` | enum | observed / literature / predicted / unknown |
| `applicability` | enum | `IN_DOMAIN`, `NEAR_BOUNDARY`, `OUT_OF_DOMAIN`, `NOT_APPLICABLE` |
| `constraint_met` | enum | `YES`, `NO`, `NA` |
| `notes` | string | Interpretation limit or caveat |

### 3.2 `candidate_summary.csv`

| Column | Type | Description |
| --- | --- | --- |
| `candidate_id` | string | Candidate key |
| `candidate_label` | string | Label |
| `status` | enum | `RANKED` or `INDETERMINATE` |
| `indeterminate_reasons` | string | Semicolon-separated missing critical criteria |
| `criteria_met` | integer | Count of `YES` constraints across all criteria (not a percentage score) |
| `criteria_failed` | integer | Count of failed **critical** constraints (non-critical failures stay visible in the matrix) |
| `criteria_unknown` | integer | Count of unknown **critical** criteria |
| `evidence_completeness` | string | e.g. `3 of 4 criteria reported` |

`criteria_met` is a count for transparency only. It is **not** a suitability
score and is never used to rank across candidates.
