#!/usr/bin/env python
"""Phase 6/7: curate surface evidence, audit it, and apply the modelling gate.

Values were read from the open-access full text of each DOI on 2026-10-28.
Every row records the exact table/section and a short extraction note. The
script refuses to run if any required provenance field is missing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import PATHS
from src.data.literature import (
    LOCKED_ENDPOINT,
    audit,
    deduplicate,
    harmonise_units,
    modelling_gate,
    validate_evidence,
)
from src.utils.io import utc_now, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")
EXTRACTOR = "AI-assisted extraction (OpenCode agent); human verification required before submission"

STUDIES = {
    "10.3390/ma14133702": {
        "doi": "10.3390/ma14133702",
        "pmcid": "PMC8269856",
        "title": (
            "Relationship between the Surface Roughness of Material and Bone Cement: "
            "An Increased 'Polished' Stem May Result in the Excessive Taper-Slip"
        ),
        "year": 2021,
        "journal": "Materials",
        "license": "CC BY 4.0",
        "source_url": "https://doi.org/10.3390/ma14133702",
        "access_status": "open_access",
    },
    "10.1038/s41598-020-76448-w": {
        "doi": "10.1038/s41598-020-76448-w",
        "pmcid": "PMC7666130",
        "title": (
            "Preparation of highly wettable coatings on Ti-6Al-4V ELI alloy for traumatological "
            "implants using micro-arc oxidation in an alkaline electrolyte"
        ),
        "year": 2020,
        "journal": "Scientific Reports",
        "license": "CC BY 4.0",
        "source_url": "https://doi.org/10.1038/s41598-020-76448-w",
        "access_status": "open_access",
    },
    "10.3390/ijms27104335": {
        "doi": "10.3390/ijms27104335",
        "pmcid": "PMC13207260",
        "title": (
            "Feasibility of Ti-6Al-4V Alloys with Nanoporous and Nanotubular Surfaces "
            "for the Treatment of Femoral Defects"
        ),
        "year": 2026,
        "journal": "International Journal of Molecular Sciences",
        "license": "CC BY 4.0",
        "source_url": "https://doi.org/10.3390/ijms27104335",
        "access_status": "open_access",
    },
    "10.3390/ma18050955": {
        "doi": "10.3390/ma18050955",
        "pmcid": "PMC11901143",
        "title": (
            "Effect of Bulk Phase Composition on the Growth of PEO Coatings on the "
            "Biomedical Ti-6Al-4V Alloy"
        ),
        "year": 2025,
        "journal": "Materials",
        "license": "CC BY 4.0",
        "source_url": "https://doi.org/10.3390/ma18050955",
        "access_status": "open_access",
    },
    "10.1016/j.heliyon.2024.e37808": {
        "doi": "10.1016/j.heliyon.2024.e37808",
        "pmcid": "PMC11417309",
        "title": (
            "Fast fabrication of superhydrophobic Ti-6Al-4V surface using Q-switched "
            "nanosecond pulsed laser at 1064 nm and cyclohexane"
        ),
        "year": 2024,
        "journal": "Heliyon",
        "license": "CC BY-NC-ND 4.0",
        "source_url": "https://doi.org/10.1016/j.heliyon.2024.e37808",
        "access_status": "open_access",
    },
}

EXCLUDED = [
    {
        "doi": "10.1016/j.heliyon.2020.e04762",
        "pmcid": "PMC7475236",
        "reason": "Dental-implant application context; out of the locked metallic orthopaedic scope.",
    },
    {
        "doi": "10.3389/fbioe.2025.1640122",
        "pmcid": "PMC12364894",
        "reason": (
            "Dental-implant focus (sandblasting/acid etching of AM and machined titanium, "
            "osteoblast adhesion); out of scope. It also reports Sa, not Ra."
        ),
    },
]


def _row(
    study_key: str,
    material: str,
    family: str,
    form: str,
    treatment: str,
    endpoint: str,
    label: str,
    value: float,
    unit: str,
    *,
    sd: float | None = None,
    dispersion: str | None = None,
    n: int | None = None,
    conditions: str = "air_25C",
    detail: str | None = None,
    method: str | None = None,
    location: str = "Table",
    note: str = "",
    coating: str | None = None,
    params: str | None = None,
    reference: str | None = None,
    biological: str | None = None,
) -> dict:
    study = STUDIES[study_key]
    return {
        **study,
        "study_id": study_key,
        "extraction_location": location,
        "extraction_note": note,
        "extractor": EXTRACTOR,
        "extracted_utc": NOW,
        "alloy_family": family,
        "material_grade_or_formula": material,
        "substrate_form": form,
        "surface_treatment": treatment,
        "coating_chemistry": coating,
        "treatment_parameters": params,
        "series_note": np.nan,
        "endpoint_key": endpoint,
        "endpoint_label": label,
        "value_original": value,
        "unit_original": unit,
        "sd_or_sem": sd,
        "dispersion_type": dispersion,
        "n_replicates": n,
        "conditions_key": conditions,
        "conditions_detail": detail,
        "measurement_method": method,
        "reference_electrode": reference,
        "evidence_level": "observed_literature",
        "biological_assay": biological,
        "in_vivo": False,
        "exclusion_reason": np.nan,
    }


def build_rows() -> list[dict]:
    rows: list[dict] = []

    # ---- Study 1: orthopaedic stem alloys, five polishing levels (Ra) ----
    s1 = "10.3390/ma14133702"
    s1_specs = {
        "CoCr (Co-28Cr-6Mo)": ("CoCr_alloy", [0.02, 0.06, 0.12, 0.29, 0.39], [0.002, 0.002, 0.009, 0.014, 0.017]),
        "316L stainless steel (SUS 316)": ("stainless_steel", [0.01, 0.06, 0.08, 0.40, 0.56], [0.002, 0.003, 0.005, 0.018, 0.030]),
        "Ti-6Al-4V": ("Ti_alloy", [0.10, 0.13, 0.19, 0.42, 0.44], [0.004, 0.007, 0.019, 0.017, 0.027]),
        "Ti-15Mo-5Zr-3Al": ("Ti_alloy", [0.11, 0.17, 0.27, 0.34, 0.41], [0.007, 0.023, 0.017, 0.009, 0.018]),
    }
    for material, (family, values, sds) in s1_specs.items():
        for level, (value, sd) in enumerate(zip(values, sds), start=1):
            rows.append(
                _row(
                    s1,
                    material,
                    family,
                    "bulk",
                    f"polishing_level_{level}",
                    "surface_roughness_ra_um",
                    "Ra",
                    value,
                    "um",
                    sd=sd,
                    dispersion="SD",
                    conditions="air_22_23C_RH50_55",
                    detail="Dry discs at 22-23 C, 50-55% RH, cemented-stem interface study",
                    method="laser scanning wide-area 3D measurement system (KEYENCE VR-5000)",
                    location="Table 2",
                    note=(
                        f"Table 2, {material}, polishing level ({level}); value and SD transcribed exactly. "
                        "Data are presented as mean ± standard deviation in the source."
                    ),
                )
            )

    # ---- Study 2: MAO on Ti-6Al-4V ELI (Ra and water contact angle) ----
    s2 = "10.1038/s41598-020-76448-w"
    s2_groups = [
        ("vibration tumbling (control)", 0.28, 0.01, 71.8, 5.8, None),
        ("micro-arc oxidation (MAO, alkaline silicate)", 1.50, 0.04, 15.6, 4.6, "TiO2 (Si-incorporated)"),
        ("MAO + blasting (outer layer removed)", 0.50, 0.02, 35.4, 9.3, "TiO2 (Si-incorporated, outer layer removed)"),
    ]
    for treatment, ra, ra_sd, ca, ca_sd, coating in s2_groups:
        rows.append(
            _row(
                s2,
                "Ti-6Al-4V ELI",
                "Ti_alloy",
                "bulk",
                treatment,
                "surface_roughness_ra_um",
                "Ra",
                ra,
                "um",
                sd=ra_sd,
                dispersion="SD",
                conditions="air_25C",
                detail="Dry, room temperature; contact profilometry",
                method="contact profilometer (Taylor Hobson Talysurf 50)",
                location="Table 1",
                note="Table 1, roughness Ra row; value and dispersion as reported.",
                coating=coating,
                params="Pulsed MAO in pH>13 Na2SiO3.9H2O + NaOH; optional blasting removes outer rutile-rich layer.",
            )
        )
        rows.append(
            _row(
                s2,
                "Ti-6Al-4V ELI",
                "Ti_alloy",
                "bulk",
                treatment,
                "water_contact_angle_deg",
                "water contact angle (H2O)",
                ca,
                "degrees",
                sd=ca_sd,
                dispersion="SD",
                conditions="air_25C_sessile_water",
                detail="Sessile drop, water, room temperature",
                method="sessile drop, Kruss DSA100 (Owens-Wendt-Rabel-Kaelble)",
                location="Table 1",
                note="Table 1, Contact angle (H2O) row; value and dispersion as reported.",
                coating=coating,
                params="Pulsed MAO in pH>13 Na2SiO3.9H2O + NaOH; optional blasting removes outer rutile-rich layer.",
            )
        )

    # ---- Study 3: anodized additively manufactured femoral implants (water contact angle) ----
    s3 = "10.3390/ijms27104335"
    s3_groups = [
        ("uncoated (chemically polished)", 63.0, None),
        ("anodized nanoporous", 26.1, "TiO2 nanoporous"),
        ("anodized nanotubular", 12.9, "TiO2 nanotubular"),
    ]
    for treatment, ca, coating in s3_groups:
        rows.append(
            _row(
                s3,
                "Ti-6Al-4V (additively manufactured porous implant)",
                "Ti_alloy",
                "additive",
                treatment,
                "water_contact_angle_deg",
                "theta_H2O",
                ca,
                "degrees",
                n=5,
                conditions="air_25C_sessile_water",
                detail="Deionized water, 0.7 uL drop, room temperature, air phase; n=5",
                method="sessile drop, KSV Attension CAM 200, Young-Laplace",
                location="Table 2",
                note="Table 2, theta_H2O column; no dispersion reported. Five measurements per condition.",
                coating=coating,
                params="Chemical polishing 120 s (HF/HNO3/H2O); anodization to nanoporous or nanotubular oxide.",
            )
        )

    # ---- Study 4: PEO on Ti-6Al-4V, corrosion potential (mV -> V) ----
    s4 = "10.3390/ma18050955"
    s4_groups = [
        ("as-received", -580, None),
        ("PEO (as-received substrate)", -490, "TiO2 PEO (Ca/P-rich)"),
        ("heat treatment 600 C", -600, None),
        ("heat treatment 600 C + PEO", -50, "TiO2 PEO (Ca/P-rich)"),
        ("heat treatment 800 C", -780, None),
        ("heat treatment 800 C + PEO", -290, "TiO2 PEO (Ca/P-rich)"),
        ("heat treatment 1000 C", -720, None),
        ("heat treatment 1000 C + PEO", -340, "TiO2 PEO (Ca/P-rich)"),
    ]
    for treatment, ecorr_mv, coating in s4_groups:
        rows.append(
            _row(
                s4,
                "Ti-6Al-4V",
                "Ti_alloy",
                "bulk",
                treatment,
                "corrosion_ecorr_v",
                "Ecorr",
                ecorr_mv,
                "mV",
                conditions="0.9pct_NaCl_25C",
                detail="0.9% NaCl aqueous electrolyte, room temperature",
                method="potentiodynamic polarization (Autolab PGSTAT128N); Ag/AgCl reference",
                location="Table 2",
                note="Table 2, Ecorr column in mV vs Ag/AgCl; converted to V by dividing by 1000.",
                coating=coating,
                params="Heat treatments at 600/800/1000 C; PEO in Ca- and P-rich electrolyte.",
                reference="Ag/AgCl",
            )
        )

    # ---- Study 5: laser-textured Ti-6Al-4V (water contact angle; Sa not Ra) ----
    s5 = "10.1016/j.heliyon.2024.e37808"
    s5_groups = [
        ("mechanical polishing (raw)", 100.3, 9.5),
        ("laser texturing (P29, 29 um pitch)", 18.5, 5.2),
        ("laser texturing + cyclohexane 70 C 2 h", 151.1, 3.6),
    ]
    for treatment, ca, sd in s5_groups:
        rows.append(
            _row(
                s5,
                "Ti-6Al-4V",
                "Ti_alloy",
                "bulk",
                treatment,
                "water_contact_angle_deg",
                "water contact angle (WCA)",
                ca,
                "degrees",
                sd=sd,
                dispersion="SD",
                n=3,
                conditions="air_25C_sessile_water",
                detail="Deionized water, ~4 uL drop, room temperature; three measurements averaged",
                method="sessile drop, SANYO VCC-5755P camera, ImageJ",
                location="Highlights and Abstract",
                note=(
                    "Mean ± SD from the Highlights. The Abstract additionally reports a maximum of "
                    "155.1 degrees for the best cyclohexane-treated sample; the mean 151.1 ± 3.6 is used here. "
                    "The source reports area roughness Sa, not Ra, so no roughness row is extracted."
                ),
                params="Q-switched nanosecond laser, 1064 nm, 4.7 mJ/pulse, pitch 29 um; cyclohexane 70 C for 2 h.",
            )
        )

    return rows


def _write_audit_doc(payload: dict, gate: dict, clean: pd.DataFrame) -> None:
    per_study = "\n".join(
        f"| `{study}` | {n} |" for study, n in payload["rows_per_study"]["by_study"].items()
    )
    endpoints = "\n".join(
        f"| `{endpoint}` | {count} | {', '.join(payload['units_by_endpoint'].get(endpoint, []))} |"
        for endpoint, count in payload["endpoint_counts"].items()
    )
    missing = "\n".join(f"| `{k}` | {v} |" for k, v in payload["missing_counts"].items())
    excluded = "\n".join(
        f"| `{item['doi']}` | {item['pmcid']} | {item['reason']} |" for item in payload["excluded_studies"]
    )
    checks = "\n".join(
        f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in gate["checks"].items()
    )
    reasons = "\n".join(f"- {r}" for r in gate["reasons"]) or "- none"

    text = f"""# Surface evidence audit

**Generated:** {utc_now()} by `scripts/build_literature.py` from
`data/literature/surface_evidence_raw.csv`.

**Scope:** metallic orthopaedic implant materials only. Dental-only studies and
non-metallic controls are excluded.

## 1. Counts

| Metric | Value |
| --- | --- |
| Raw extracted rows | {payload['n_raw']} |
| Duplicates removed | {payload['n_duplicates_removed']} |
| Clean rows | {payload['n_clean']} |
| Distinct studies | {payload['n_distinct_studies']} |
| Rows per study (min / median / max) | {payload['rows_per_study']['min']} / {payload['rows_per_study']['median']} / {payload['rows_per_study']['max']} |
| Top study share | {payload['top_study_share']:.1%} |

### Rows per study

| study_id | rows |
| --- | --- |
{per_study}

## 2. Endpoints and units

| endpoint_key | rows | harmonised units |
| --- | --- | --- |
{endpoints}

Locked endpoint for any modelling attempt: **`{LOCKED_ENDPOINT}`**.
Rows for the locked endpoint: **{payload['locked_endpoint_rows']}** across
**{payload['locked_endpoint_studies']}** studies; largest study share
**{payload['locked_endpoint_top_study_share']:.1%}**.

## 3. Missingness (not reported, never imputed)

| Field | Rows missing |
| --- | --- |
{missing}

## 4. Deduplication

Key: `({', '.join(payload['dedup_log']['key'])})`.

{payload['dedup_log']['n_duplicates_removed']} duplicate row(s) removed. Full
removal log is in `results/validation/surface_gate.json`.

## 5. Exclusions

| DOI | PMCID | Reason |
| --- | --- | --- |
{excluded}

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
{checks}

Observed: {gate['observed']['studies']} studies, {gate['observed']['rows']} rows,
top study share {gate['observed']['top_study_share']:.1%}.

**Decision: `{gate['decision']}`**

Reasons:

{reasons}

## 9. Honest interpretation

The curated table is an evidence catalogue. It is **not** a clinical dataset, it
does not establish biocompatibility, and it cannot support a generalisable
surface-response model at this sample size. The table is intentionally kept
small and traceable rather than padded with unverifiable numbers.
"""
    (PATHS["docs"] / "SURFACE_EVIDENCE_AUDIT.md").write_text(text, encoding="utf-8")

    curated = f"""# Surface evidence - curation-only decision

**Decision:** `{gate['decision']}`

The Phase 7 gate was applied before any surface model was trained.

## Why no surface model

{chr(10).join('- ' + r for r in gate['reasons']) if gate['reasons'] else '- All gate conditions were met.'}

## What is delivered instead

1. `data/literature/surface_evidence_raw.csv` - {payload['n_raw']} extracted rows with DOI,
   table/figure location and extraction notes.
2. `data/literature/surface_evidence_clean.csv` - {payload['n_clean']} harmonised,
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
"""
    (PATHS["docs"] / "SURFACE_EVIDENCE_CURATED.md").write_text(curated, encoding="utf-8")


def _write_data_card(payload: dict, gate: dict) -> None:
    text = f"""# Data card - surface evidence layer

**Status:** generated {utc_now()} from `data/literature/surface_evidence_clean.csv`.

## 1. What this is

A hand-curated, provenance-preserving table of **measured surface properties**
reported in open-access studies of metallic orthopaedic implant materials.

| Item | Value |
| --- | --- |
| Raw rows | {payload['n_raw']} |
| Clean rows | {payload['n_clean']} |
| Distinct studies | {payload['n_distinct_studies']} |
| Endpoints | {', '.join(payload['endpoint_counts'].keys())} |
| Locked endpoint | `{payload['locked_endpoint']}` ({payload['locked_endpoint_rows']} rows) |
| Modelling decision | `{gate['decision']}` |

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
   (`{gate['decision']}`).

## 6. Intended use

Decision-support display of what has actually been measured for a material and
surface treatment, with explicit evidence levels and unknowns.
"""
    (PATHS["data_cards"] / "surface_evidence_data_card.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows = build_rows()
    raw = pd.DataFrame(rows)

    # Required optional columns
    for column in [
        "coating_chemistry",
        "treatment_parameters",
        "sd_or_sem",
        "dispersion_type",
        "n_replicates",
        "measurement_method",
        "reference_electrode",
        "biological_assay",
        "in_vivo",
    ]:
        if column not in raw.columns:
            raw[column] = np.nan

    validate_evidence(raw)
    clean, dedup_log = deduplicate(harmonise_units(raw))

    gate_path = PATHS["docs"] / "LEAKAGE_AUDIT.md"
    phase4_passed = gate_path.exists() and "LEAKAGE_GATE: PASSED" in gate_path.read_text(encoding="utf-8")

    audit_payload = audit(raw, clean, dedup_log, EXCLUDED)
    gate = modelling_gate(audit_payload, phase4_gate_passed=phase4_passed)

    clean["included_in_modelling"] = False
    clean["exclusion_reason"] = clean["exclusion_reason"].astype(object)
    if gate["passed"]:
        clean.loc[clean["endpoint_key"] == LOCKED_ENDPOINT, "included_in_modelling"] = True
    else:
        clean.loc[clean["endpoint_key"] != LOCKED_ENDPOINT, "exclusion_reason"] = (
            "Not the single locked endpoint for Phase 7 modelling."
        )

    PATHS["data_literature"].mkdir(parents=True, exist_ok=True)
    raw.to_csv(PATHS["data_literature"] / "surface_evidence_raw.csv", index=False)
    clean.to_csv(PATHS["data_literature"] / "surface_evidence_clean.csv", index=False)

    payload = {**audit_payload, "gate": gate, "created_utc": utc_now()}
    write_json(PATHS["validation"] / "surface_gate.json", payload)

    _write_audit_doc(audit_payload, gate, clean)
    _write_data_card(audit_payload, gate)

    print(json.dumps({
        "n_raw": audit_payload["n_raw"],
        "n_clean": audit_payload["n_clean"],
        "n_studies": audit_payload["n_distinct_studies"],
        "endpoints": audit_payload["endpoint_counts"],
        "locked_endpoint_rows": audit_payload["locked_endpoint_rows"],
        "locked_endpoint_studies": audit_payload["locked_endpoint_studies"],
        "decision": gate["decision"],
        "reasons": gate["reasons"],
    }, indent=2))


if __name__ == "__main__":
    main()
