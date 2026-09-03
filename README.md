# Peak-a-boo

Peak-a-boo is an uncertainty-aware visual analytics workbench for reviewing chromatographic peak detections. This public repository uses only synthetic chromatograms, synthetic reference intervals, and a classifier trained on the synthetic demo data.

Live demo: https://peakaboo.streamlit.app/

## Introduction

Chromatographic peak detectors usually return a peak/no-peak decision even when that decision may depend on smoothing, local noise, overlapping signals, or parameter choices. Peak-a-boo turns each candidate into a reviewable case. It shows the local chromatogram together with separate evidence such as detectability, perturbation stability, reference correspondence, and processing context. A Sandbox lets analysts test parameter changes without changing the main review state, and an Audit Log records review decisions and notes. The public release is intended for demonstration, interface testing, and reproducible development without distributing the original research data.

## Repo Structure

- `app.py` — main Streamlit application and review interface.
- `chromato_peak/` — core package.
  - `config.py` — default preprocessing, segmentation, detection, scoring, and stability settings.
  - `data_io.py` — data loading and validation.
  - `preprocessing.py` — smoothing and local-noise estimation.
  - `segmentation.py` — local analysis regions.
  - `detection.py` — global and segment-dependent peak detection and duplicate handling.
  - `scoring.py` — detectability and perturbation-stability measures.
  - `evaluation.py` — reference matching and summary measures.
  - `pipeline.py` — end-to-end analysis pipeline.
  - `review.py` — review-case construction and review-state logic.
  - `persistence.py` — review actions, notes, and audit records.
  - `visualization.py` — Plotly chromatogram views.
- `data/` — public synthetic HDF5 and reference workbook plus data notes.
- `models/` — synthetic classifier used by the demo.
- `tests/` — pipeline and interface tests.
- `generate_synthetic_data.py` — rebuilds the synthetic chromatograms and reference workbook.
- `train_synthetic_ml.py` — retrains the synthetic classifier.
- `requirements.txt` — supported Python package ranges.
- `run_app.sh`, `run_app.bat` — convenience launch scripts.
- `.streamlit/config.toml` — Streamlit settings.
- `IMPLEMENTATION_AUDIT.md` — implementation notes for the public release.
- `VALIDATION.md` — current validation record.

## Getting Started

### Prerequisites & Needed Materials

You need Python, `pip`, Git, and a terminal or command prompt. The current CI workflow validates this repository with **Python 3.11**.

Install package versions from `requirements.txt`:

```text
streamlit>=1.57,<2
plotly>=5.20,<7
pandas>=2.0,<3
numpy>=1.24,<3
scipy>=1.10,<2
h5py>=3.9,<4
openpyxl>=3.1,<4
pytest>=8,<9
joblib>=1.3,<2
scikit-learn>=1.3,<2
```

The synthetic HDF5 file, reference workbook, and classifier needed for the public demo are already included. Access to the original research data is not required.

### Installation

```bash
git clone https://github.com/washuvis/peak-a-boo.git
cd peak-a-boo
python -m venv .venv
```

Activate the environment.

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Usage

Start the workbench:

```bash
python -m streamlit run app.py
```

Typical workflow:

1. Select a review case.
2. Inspect the raw and smoothed signal around the candidate.
3. Review the available peak-level evidence.
4. Use the Sandbox to test alternative settings without changing the baseline review state.
5. Accept, flag, reopen, or annotate the case.
6. Review or export the audit record.

Rebuild the synthetic demo data:

```bash
python generate_synthetic_data.py
```

Retrain the synthetic classifier:

```bash
python train_synthetic_ml.py
```

Run the tests:

```bash
pytest -q
```

The current validation record reports **11 passing tests**. See `VALIDATION.md` for details.

Main application inputs:

```text
data/synthetic_chromatograms.h5
data/synthetic_reference.xlsx
models/synthetic_peak_classifier.joblib
```

Review actions may be written to `ReviewActions` and `ReviewAuditTrail` sheets in the synthetic workbook. On Streamlit Community Cloud, local file changes may be temporary, so use the application's export options when records need to be kept.

## Related Repositories

The public Peak-a-boo repositories are maintained under the VIBE Lab `washuvis` GitHub organization:

- [`washuvis/peak-a-boo`](https://github.com/washuvis/peak-a-boo) — this public synthetic workbench.
- [`washuvis/peakaboo-expert-study`](https://github.com/washuvis/peakaboo-expert-study) — expert-study application, case-bank workflow, response logging, and analysis code.
- [`washuvis/peak-detection`](https://github.com/washuvis/peak-detection) — historical early scaffold that predates the current package structure.

Anyone continuing the project should first identify which repository matches the task before making changes.

## Main Technical Libraries

- [Streamlit](https://streamlit.io/) — web application and state.
- [Plotly](https://plotly.com/python/) — interactive charts.
- [SciPy](https://scipy.org/) — peak detection and signal processing.
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) — data processing.
- [h5py](https://www.h5py.org/) — HDF5 files.
- [scikit-learn](https://scikit-learn.org/) — synthetic classifier.

## Data Use

This repository contains only synthetic demonstration data. The HDF5 file, reference workbook, and classifier are not derived from the original research chromatograms or original trained model. Reference intervals are comparison evidence and should not be treated as universal chemical ground truth.

## Validation and Maintenance

Before a major change:

1. Run `pytest -q`.
2. Start the Streamlit application and confirm that it loads without errors.
3. Check queue selection, linked zoom, review actions, Sandbox updates, and exports.
4. Update `VALIDATION.md` when validated behavior changes.
5. Update this README when the repository structure, setup, related repositories, or workflow changes.

For major interface changes, update `IMPLEMENTATION_AUDIT.md` when the documented public-release architecture changes.
