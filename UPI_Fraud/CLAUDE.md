# UPI_Fraud — Raw NPCI Source Data

Official NPCI monthly UPI transaction statistics used to calibrate synthetic data generation in `upi_fraud_detection/`.

## Files

| File | What | When to read |
|---|---|---|
| `Product-Statistics-UPI-Upi-monthly-statistics-2021-22-monthly.xlsx` | NPCI monthly stats for FY 2021–22: Volume (Mn), Avg Daily Volume, Value (Cr), Avg Daily Value — 10 months | Tracing calibration parameters, verifying monthly data completeness |
| `Product-Statistics-UPI-Upi-monthly-statistics-2022-23-monthly.xlsx` | NPCI monthly stats for FY 2022–23 — 10 months | Tracing calibration parameters, verifying monthly data completeness |
| `Product-Statistics-UPI-Upi-monthly-statistics-2023-24-monthly.xlsx` | NPCI monthly stats for FY 2023–24 — 10 months | Tracing calibration parameters, verifying monthly data completeness |

## Usage Note

These files are **read-only source data** consumed by `../upi_fraud_detection/data_prep.py`. They are not modified by any script. The key derived parameter is: avg transaction value = total Value (Cr × 1e7) ÷ total Volume (Mn × 1e6) ≈ Rs. 1,612 across 2021–2024.
