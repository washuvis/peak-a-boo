# Validation Record — Modern UI Edition

Validated again on 2026-08-31 using GitHub Actions on Python 3.11.

## Static checks

- Python compilation passed for `app.py`, `chromato_peak/`, `tests/`, `generate_synthetic_data.py`, and `train_synthetic_ml.py`.
- No syntax errors were reported by the CI compilation step.

## Automated tests

Command:

```bash
pytest -q
```

Result:

- **11 passed in 6.95 s**

Coverage includes:

- synthetic data and model presence
- synthetic data schema and manifest
- pipeline and queue generation
- trace consistency between overview and zoom
- review persistence and report export
- core application controls
- plain selected-region evidence rows
- side-by-side Sandbox layout
- live Sandbox chart updates
- modern visual shell and circular stability UI

## Continuous integration

`.github/workflows/tests.yml` now runs the test suite and Python compilation on pushes, pull requests, and manual workflow dispatch.

## Prior interaction/runtime checks

The earlier validation record also confirmed through Streamlit application testing that:

- the application loads without exceptions
- queue cues remain selectable
- automatic focus behavior remains available
- `Accept as reviewed` persists and displays the resolved state
- Sandbox parameter changes update the live chart
- navigation and export controls remain available

The application was previously launched with:

```bash
streamlit run app.py --server.headless true --server.port 8611
```

with a healthy endpoint response.

## Packaging checks

- synthetic HDF5 file included
- synthetic reference workbook included
- synthetic classifier included
- requirements and Streamlit configuration included
- temporary files and virtual environments excluded

## Note

The bundled synthetic classifier may emit a scikit-learn version warning if loaded under a different patch/minor release than the one used to serialize it. The current automated tests still pass.
