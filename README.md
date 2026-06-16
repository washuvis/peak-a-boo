# Peak-a-boo Review Workbench — Modern UI Edition

This is a public-facing Streamlit demonstration of the Peak-a-boo chromatography review workflow. It uses only synthetic chromatogram signals, synthetic labeled intervals, and a classifier trained only on the synthetic demo data.

The analytical and review functionality is preserved from the previous version. This release focuses on a full visual redesign and interaction polish.

## What is preserved

- Labeled Data, AI Review, Sandbox, and Audit Log views
- synthetic HDF5 signal loading
- synthetic labeled-interval loading
- automatic zoom to a selected review cue
- raw, smoothed, baseline, uncertainty, label, and peak layers
- comparison-run preview
- grouped Sandbox parameters with side-by-side live chart updates
- ML-enhanced Sandbox preview
- accept, flag, resolve, reopen, and analyst-note actions
- workbook-backed review persistence
- audit report and reviewed-workbook downloads

## Modern interface improvements

- glass-style cards with subtle depth and gradient accents
- centered product header with a clear synthetic-workspace indicator
- gradient primary actions and stronger hover/focus feedback
- modern pill navigation and toolbar controls
- review cards with severity-colored edge indicators
- clearer selected-card treatment in the queue
- upgraded chromatogram colors and chart surface
- circular stability indicator in the selected-region panel
- improved information hierarchy for provenance and evidence
- modern feature cards, explanation cards, audit entries, and status messages
- sticky Sandbox controls so parameters and the live chart remain side by side
- responsive styling for smaller desktop widths

## Included synthetic assets

```text
data/synthetic_chromatograms.h5
data/synthetic_reference.xlsx
models/synthetic_peak_classifier.joblib
```

None of these files are derived from the original research dataset.

## Project structure

```text
peak-a-boo-workbench-modern-ui/
├── app.py
├── chromato_peak/
├── data/
├── models/
├── tests/
├── assets/
├── .streamlit/config.toml
├── requirements.txt
├── IMPLEMENTATION_AUDIT.md
└── VALIDATION.md
```

## Run locally

### Windows PowerShell

```powershell
cd peak-a-boo-workbench-modern-ui
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### macOS or Linux

```bash
cd peak-a-boo-workbench-modern-ui
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Extract the ZIP.
2. Upload the complete folder to a GitHub repository.
3. Create a Streamlit app from that repository.
4. Select `app.py` as the entry point.
5. Deploy without adding secrets or external data paths.

## Main workflow

1. Select a review cue from the left queue.
2. The chromatogram automatically focuses on its interval.
3. Inspect the selected evidence and provenance.
4. Accept the cue, flag an exception, or add a note.
5. Review the audit trail or download the report/workbook.

## Tests

```bash
pytest -q
```

The validation record documents all completed checks.
