# Validation Record — Modern UI Edition

## Static checks

- Python compilation passed for `app.py` and all `chromato_peak` modules.
- No syntax or import errors were found.

## Automated tests

Command:

```bash
pytest -q
```

Result:

- **11 passed**

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

## Interaction checks

Validated through Streamlit application testing:

- application loads without exceptions
- queue cues remain selectable
- automatic focus behavior remains available
- `Accept as reviewed` persists and displays the resolved state
- Sandbox parameter changes update the live chart
- navigation and export controls remain available

## Runtime check

The application was launched with:

```bash
streamlit run app.py --server.headless true --server.port 8611
```

Health endpoint result:

```text
ok
```

## Packaging checks

- synthetic HDF5 file included
- synthetic reference workbook included
- synthetic classifier included
- requirements and Streamlit configuration included
- temporary files and virtual environments excluded
- ZIP integrity verified after packaging

## Note

The bundled synthetic classifier was created with a newer scikit-learn patch version than the validation environment. Loading it produces a version warning in tests, but the synthetic-model metadata and application workflow remain valid.
