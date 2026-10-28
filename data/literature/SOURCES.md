# Literature source retrieval

The full-text XML files used for extraction are **not committed**, because the
licences vary (some sources are CC BY-NC-ND) and third-party full text should not be
redistributed from this repository.

Each file can be retrieved by PMCID from the Europe PMC full-text endpoint:

```bash
curl -o data/literature/sources/<PMCID>.xml \
  "https://www.ebi.ac.uk/europepmc/webservices/rest/<PMCID>/fullTextXML"
```

## Included studies

| study_id (DOI) | PMCID |
| --- | --- |
| `10.3390/ma14133702` | PMC8269856 |
| `10.1038/s41598-020-76448-w` | PMC7666130 |
| `10.3390/ijms27104335` | PMC13207260 |
| `10.3390/ma18050955` | PMC11901143 |
| `10.1016/j.heliyon.2024.e37808` | PMC11417309 |

## Screened and excluded (dental-only)

| DOI | PMCID | Reason |
| --- | --- | --- |
| `10.1016/j.heliyon.2020.e04762` | PMC7475236 | Dental-implant context; out of orthopaedic scope |
| `10.3389/fbioe.2025.1640122` | PMC12364894 | Dental-implant context; also reports Sa, not Ra |

All extracted values, their table/section location and conditions are recorded in
`data/literature/surface_evidence_raw.csv` and `surface_evidence_clean.csv`.
