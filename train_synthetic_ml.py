"""Train the optional sandbox classifier on the bundled synthetic demo only."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from chromato_peak.data_io import list_h5_channels
from chromato_peak.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent
H5_PATH = ROOT / "data" / "synthetic_chromatograms.h5"
REFERENCE_PATH = ROOT / "data" / "synthetic_reference.xlsx"
MODEL_PATH = ROOT / "models" / "synthetic_peak_classifier.joblib"

FEATURE_COLUMNS = [
    "height", "prominence", "width_samples", "sigma_at_peak", "local_noise",
    "local_baseline", "prominence_threshold", "weber_score", "weber_margin",
    "abs_weber_margin", "decision_uncertainty", "stability_filled", "relative_time",
    "distance_to_prev", "distance_to_next", "prominence_ratio_to_prev",
    "prominence_ratio_to_next", "height_ratio_to_prev", "height_ratio_to_next",
    "candidate_density_segment", "relative_position_in_segment",
]


def feature_table(result) -> pd.DataFrame:
    peaks = result.final_peaks.copy().sort_values("sample_idx").reset_index(drop=True)
    if peaks.empty:
        return peaks
    peaks["abs_weber_margin"] = pd.to_numeric(peaks.get("weber_margin"), errors="coerce").abs()
    peaks["stability_filled"] = pd.to_numeric(peaks.get("stability"), errors="coerce").fillna(0.0)
    time_span = max(float(result.time.max() - result.time.min()), 1e-12)
    peaks["relative_time"] = (pd.to_numeric(peaks["time"], errors="coerce") - float(result.time.min())) / time_span
    sample_idx = pd.to_numeric(peaks["sample_idx"], errors="coerce").to_numpy(dtype=float)
    peaks["distance_to_prev"] = np.r_[len(result.time), np.diff(sample_idx)]
    peaks["distance_to_next"] = np.r_[np.diff(sample_idx), len(result.time)]
    prom = np.maximum(pd.to_numeric(peaks.get("prominence"), errors="coerce").to_numpy(dtype=float), 1e-12)
    height = np.maximum(np.abs(pd.to_numeric(peaks.get("height"), errors="coerce").to_numpy(dtype=float)), 1e-12)
    peaks["prominence_ratio_to_prev"] = prom / np.maximum(np.r_[prom[0], prom[:-1]], 1e-12)
    peaks["prominence_ratio_to_next"] = prom / np.maximum(np.r_[prom[1:], prom[-1]], 1e-12)
    peaks["height_ratio_to_prev"] = height / np.maximum(np.r_[height[0], height[:-1]], 1e-12)
    peaks["height_ratio_to_next"] = height / np.maximum(np.r_[height[1:], height[-1]], 1e-12)
    peaks["candidate_density_segment"] = peaks.groupby("segment_id")["peak_id"].transform("count")
    seg_start = pd.to_numeric(peaks["segment_time_start"], errors="coerce").fillna(float(result.time.min()))
    seg_end = pd.to_numeric(peaks["segment_time_end"], errors="coerce").fillna(float(result.time.max()))
    seg_span = np.maximum((seg_end - seg_start).to_numpy(dtype=float), 1e-12)
    peaks["relative_position_in_segment"] = (
        pd.to_numeric(peaks["time"], errors="coerce").to_numpy(dtype=float) - seg_start.to_numpy(dtype=float)
    ) / seg_span
    peaks["target"] = (peaks["match_status"] == "TP").astype(int)
    peaks["channel_id"] = int(result.channel_id)
    for col in FEATURE_COLUMNS:
        if col not in peaks:
            peaks[col] = 0.0
    return peaks


def main() -> None:
    tables = [feature_table(run_pipeline(H5_PATH, REFERENCE_PATH, channel)) for channel in list_h5_channels(H5_PATH)]
    data = pd.concat(tables, ignore_index=True)
    X = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = data["target"].astype(int)
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=2026,
    )
    model.fit(X, y)
    probability = model.predict_proba(X)[:, 1]
    prediction = probability >= 0.5
    metrics = {
        "dataset": "public synthetic demo only",
        "n_samples": int(len(data)),
        "n_features": len(FEATURE_COLUMNS),
        "positive_rate": float(y.mean()),
        "roc_auc_train": float(roc_auc_score(y, probability)),
        "classification_report_train": classification_report(y, prediction, output_dict=True, zero_division=0),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_type": "random_forest_synthetic_demo",
            "feature_columns": FEATURE_COLUMNS,
            "probability_threshold": 0.5,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    print(f"Saved {MODEL_PATH}")
    print(metrics)


if __name__ == "__main__":
    main()
