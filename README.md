# Peak-a-boo

Peak-a-boo is an uncertainty-aware visual analytics workbench for reviewing chromatographic peak detections. This repository is the public-facing version of the project and uses only synthetic chromatograms, synthetic reference intervals, and a classifier trained on the synthetic demo data.

Live demo: https://peakaboo.streamlit.app/

## Status

This repository is an in-progress public demonstration of the broader Peak-a-boo research project. The current version supports uncertainty-aware peak review, parameter exploration, review actions, and audit records using synthetic data. The broader project is still being developed and evaluated, including an expert-study version maintained in a separate repository.

## Introduction

Chromatographic peak detectors usually return a peak or no-peak decision, even when that decision may change because of smoothing, local noise, overlapping signals, or parameter choices. Peak-a-boo makes these decisions easier to inspect by showing the chromatogram together with separate evidence about a detected peak, including local uncertainty, detectability, perturbation stability, reference correspondence, and processing history. The workbench also provides a Sandbox for testing parameter changes without changing the main review state, and an Audit Log for recording analyst decisions and notes. This public repository is designed for demonstration, interface testing, and reproducible development without exposing the original research data.

This repository is associated with the ongoing research project currently titled **“Peak-a-boo: Making Binary Chromatographic Peak Detections Reviewable.”** A peer-reviewed publication is not yet linked to this repository. If a publication becomes available, add the final citation here.

## Repository Structure

- `app.py`
  - Main Streamlit application.
  - Builds the review interface, navigation, controls, linked views, Sandbox, and audit workflow.

- `chromato_peak/`
  - Core Python package used by the application.
  - `config.py`: default settings for preprocessing, segmentation, detection, scoring, and perturbation stability.
  - `data_io.py`: loads chromatogram signals and reference tables.
  - `preprocessing.py`: smooths the signal and estimates local noise and uncertainty bands.
  - `segmentation.py`: divides a chromatogram into local analysis regions.
  - `detection.py`: runs global and segment-dependent peak detection and removes duplicate detections.
  - `scoring.py`: computes peak-level detectability and perturbation-stability measures.
  - `evaluation.py`: compares detected peaks with reference intervals and produces evaluation labels and summary measures.
  - `pipeline.py`: runs the end-to-end preprocessing, detection, scoring, stability, and evaluation pipeline.
  - `review.py`: creates review cases and supports review-state logic.
  - `persistence.py`: stores analyst actions, notes, and audit records in the workbook.
  - `visualization.py`: builds the Plotly chromatogram and related visual layers.
  - `__init__.py`: package entry file.

- `data/`
  - Public synthetic data used by the demo.
  - `synthetic_chromatograms.h5`: six generated chromatogram traces.
  - `synthetic_reference.xlsx`: generated reference intervals and review-record sheets.
  - `README.md`: details about how the synthetic data were created and used.

- `models/`
  - `synthetic_peak_classifier.joblib`: optional classifier trained only on the bundled synthetic data.

- `tests/`
  - `conftest.py`: shared test setup.
  - `test_pipeline_smoke.py`: checks the core data and pipeline workflow.
  - `test_app_ux.py`: checks key application controls and interface behavior.

- `generate_synthetic_data.py`
  - Recreates the public synthetic chromatograms and reference workbook from fixed mathematical functions and random seeds.

- `train_synthetic_ml.py`
  - Trains the optional Random Forest classifier on the bundled synthetic demo data.

- `requirements.txt`
  - Python package requirements and supported version ranges.

- `run_app.sh`
  - Starts the Streamlit application on macOS or Linux.

- `run_app.bat`
  - Starts the Streamlit application on Windows.

- `.streamlit/config.toml`
  - Streamlit theme and application settings.

- `assets/`
  - Static design reference material used during interface development.

- `IMPLEMENTATION_AUDIT.md`
  - Records the major interface and implementation changes made for the public release.

- `VALIDATION.md`
  - Records completed static checks, automated tests, interaction checks, and packaging checks.

## Getting Started

### Prerequisites and Needed Materials

You need:

- Python 3
- `pip`
- Git
- a terminal or command prompt

This repository does not currently pin one Python interpreter version. Python package versions are controlled in `requirements.txt`.

Current package ranges are:

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

The repository already includes the synthetic HDF5 file, synthetic reference workbook, and synthetic classifier needed for the public demo. You do not need access to the original research data.

### Installation

Clone the repository:

```bash
git clone https://github.com/washuvis/peak-a-boo.git
cd peak-a-boo
```

Create a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

### Usage

Start the application with:

```bash
python -m streamlit run app.py
```

You can also use the included launch scripts.

Windows:

```text
run_app.bat
```

macOS or Linux:

```bash
bash run_app.sh
```

The main review workflow is:

1. Select a review case from the queue.
2. Inspect the raw and smoothed chromatogram around that case.
3. Turn evidence layers on or off as needed.
4. Review peak-level evidence and processing information.
5. Use the Sandbox to test alternative parameter settings without changing the main review state.
6. Accept a detection, flag an exception, reopen a case, or add an analyst note.
7. Review the Audit Log or export the available review records.

### Rebuild the Synthetic Demo Data

To recreate the bundled synthetic chromatograms and reference workbook:

```bash
python generate_synthetic_data.py
```

This script creates six synthetic channels, numbered `1001` through `1006`. The signals are generated from mathematical peak shapes, baseline drift, changing noise, and selected unlabeled artifacts. It does not read or derive values from the original research data.

### Retrain the Synthetic Classifier

After generating the synthetic data, retrain the optional classifier with:

```bash
python train_synthetic_ml.py
```

The model is saved to:

```text
models/synthetic_peak_classifier.joblib
```

The classifier is trained only on the synthetic demo data.

### Run the Tests

Run:

```bash
pytest -q
```

The current validation record reports 11 passing tests. See `VALIDATION.md` for the checks that were completed for the public release.

### Review Data and Outputs

The application reads:

```text
data/synthetic_chromatograms.h5
data/synthetic_reference.xlsx
models/synthetic_peak_classifier.joblib
```

Analyst review actions can be written to additional sheets in `synthetic_reference.xlsx`, including:

```text
ReviewActions
ReviewAuditTrail
```

On Streamlit Community Cloud, local file changes may be lost when the application restarts. Use the application's download options if review records need to be kept.

## Related Repositories

Peak-a-boo has been developed through several related repositories. They serve different purposes and should not be treated as separate research projects.

- [`washuvis/peak-a-boo`](https://github.com/washuvis/peak-a-boo)
  - This repository.
  - Public synthetic demonstration and current public workbench.

- [`washuvis/chromato-peak-app`](https://github.com/washuvis/chromato-peak-app)
  - Private/internal Peak-a-boo development repository.
  - Contains an internal version of the chromatography peak-analysis application.
  - Access depends on VIBE Lab permissions.

- [`ghoshsaurav/peakaboo-expert-study`](https://github.com/ghoshsaurav/peakaboo-expert-study)
  - Expert-study version of Peak-a-boo.
  - Contains the study flow, case bank, participant response logging, analysis scripts, and study-specific documentation.

- [`ghoshsaurav/peak-detection`](https://github.com/ghoshsaurav/peak-detection)
  - Earlier peak-detection research prototype.
  - Contains exploratory code used before the current Peak-a-boo workbench architecture.

Anyone continuing this work should first identify which repository matches the task they are working on before making changes.

## Main Technical Libraries

The application is built with the following main libraries:

- [Streamlit](https://streamlit.io/) for the web application.
- [Plotly](https://plotly.com/python/) for interactive charts.
- [SciPy](https://scipy.org/) for peak detection and signal-processing functions.
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data processing.
- [h5py](https://www.h5py.org/) for HDF5 chromatogram files.
- [scikit-learn](https://scikit-learn.org/) for the optional synthetic classifier.

Exact supported package ranges are listed in `requirements.txt`.

## Data Use

This public repository contains only synthetic demonstration data. The bundled HDF5 file, reference workbook, and classifier are not derived from the original research chromatograms or original trained models.

Reference intervals are used as comparison evidence in the public demo. They should not be interpreted as universal chemical ground truth.

## Future Work

Useful next steps for this line of work include:

- evaluate how domain experts use separate forms of uncertainty evidence when reviewing difficult peak detections;
- study when evidence disagreement leads analysts to accept, reject, or defer an automated decision;
- improve the study workflow and analysis for human-AI reliance and review behavior;
- test the approach on additional chromatography data when data-sharing and research permissions allow;
- continue testing which peak-level evidence is most useful for identifying cases that need human review; and
- keep the public demo, internal development version, and expert-study version synchronized where appropriate.

## Validation and Maintenance

Before committing a major change:

1. Run `pytest -q`.
2. Start the Streamlit application and check that it loads without errors.
3. Test queue selection, linked zoom, review actions, Sandbox updates, and exports.
4. Update `VALIDATION.md` if the validated behavior changes.
5. Update this README if the directory structure, setup steps, related repositories, or main workflow changes.

For major interface changes, also update `IMPLEMENTATION_AUDIT.md` when the change affects the documented public-release architecture.
