# Benchmark comparison - matbench_log_gvrh

Target: `log10(G_VRH)` in log10(GPa), computed DFT shear modulus.
Official Matbench 5-fold splits; composition-only descriptors.

| Model         |   Features |   MAE (mean) |   MAE (std) |   RMSE (mean) |   RMSE (std) |   R2 (mean) |   R2 (std) |
|:--------------|-----------:|-------------:|------------:|--------------:|-------------:|------------:|-----------:|
| random_forest |        132 |       0.1024 |      0.0016 |        0.152  |       0.0046 |      0.8325 |     0.01   |
| ridge         |        132 |       0.1568 |      0.0011 |        0.2144 |       0.0028 |      0.6671 |     0.0069 |
| dummy         |        132 |       0.2931 |      0.0035 |        0.3716 |       0.0042 |     -0.0002 |     0.0002 |

**Selected model (lowest mean OOF MAE):** `random_forest`
