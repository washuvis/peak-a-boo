# Synthetic demo data

This directory contains a fully synthetic public demonstration dataset.

- `synthetic_chromatograms.h5` contains six generated time–intensity traces (`1001`–`1006`).
- `synthetic_reference.xlsx` contains generated reference intervals in the `reference_peaks` worksheet.

The signals were produced from Gaussian peaks, baseline drift, heteroscedastic noise, and a few unlabeled artifacts. The reference workbook contains only the generated peak definitions. No source research chromatograms, labels, or trained model records are included.

The application reads the HDF5 signal and uses the reference intervals for TP/FP/FN comparison. Analyst actions are written to the local `synthetic_reference.xlsx` workbook in these additional sheets:

- `ReviewActions`
- `ReviewAuditTrail`

Regenerate the bundled demo files with:

```bash
python generate_synthetic_data.py
python train_synthetic_ml.py
```

On Streamlit Community Cloud, local writes may reset when the app restarts. Use the download buttons to retain review records.
