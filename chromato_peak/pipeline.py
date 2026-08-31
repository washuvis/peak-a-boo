"""End-to-end uncertainty-aware peak review pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from .config import PipelineConfig
from .data_io import load_chromatogram_h5, load_label_table, labels_for_channel
from .detection import detect_peaks_global, detect_peaks_segmented_from_smooth
from .evaluation import evaluate_peaks
from .preprocessing import estimate_local_noise_arrays, moving_average
from .scoring import add_weber_scores, compute_peak_stability, stability_label
from .segmentation import Segment


@dataclass(slots=True)
class PipelineResult:
    """Collect all signal arrays, detections, evidence, and metrics from one pipeline run."""

    channel_id: Optional[int]
    time: np.ndarray
    raw_signal: np.ndarray
    smoothed_signal: np.ndarray
    residual: np.ndarray
    sigma_t: np.ndarray
    band_low: np.ndarray
    band_high: np.ndarray
    segments: list[Segment]
    segment_table: pd.DataFrame
    global_peaks: pd.DataFrame
    final_peaks: pd.DataFrame
    label_table: pd.DataFrame
    global_metrics: dict[str, float]
    final_metrics: dict[str, float]
    config: PipelineConfig


def run_arrays(t: np.ndarray, y_raw: np.ndarray, labels: pd.DataFrame | None = None, *, channel_id: Optional[int] = None, config: PipelineConfig | None = None) -> PipelineResult:
    """Run preprocessing, detection, evidence scoring, stability, and reference comparison.

    The same input signal is used throughout the run. Perturbation stability is
    computed by rerunning the segment-dependent detector on noise-scaled copies
    of the raw signal; these copies are used only for the stability calculation.
    """
    cfg = config or PipelineConfig()
    t = np.asarray(t, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    if len(t) != len(y_raw) or len(t) == 0:
        raise ValueError("Chromatogram time and intensity arrays must be non-empty and aligned.")
    y_smooth = moving_average(y_raw, cfg.preprocessing.smooth_window)
    residual, sigma_t, low, high = estimate_local_noise_arrays(y_raw, y_smooth, cfg.preprocessing.noise_window, cfg.preprocessing.uncertainty_sigma)
    _, global_peaks = detect_peaks_global(t, y_raw, cfg.preprocessing, cfg.detection, sigma_t)
    global_peaks = add_weber_scores(global_peaks, cfg.scoring)
    segments, final_peaks, segment_table = detect_peaks_segmented_from_smooth(t, y_raw, y_smooth, sigma_t, cfg.segmentation, cfg.detection)
    final_peaks = add_weber_scores(final_peaks, cfg.scoring)
    if not final_peaks.empty:
        def detector(perturbed: np.ndarray) -> pd.DataFrame:
            """Rerun the local detector for one perturbed signal during stability estimation."""
            smooth = moving_average(perturbed, cfg.preprocessing.smooth_window)
            _, pert_sigma, _, _ = estimate_local_noise_arrays(perturbed, smooth, cfg.preprocessing.noise_window, cfg.preprocessing.uncertainty_sigma)
            _, p, _ = detect_peaks_segmented_from_smooth(t, perturbed, smooth, pert_sigma, cfg.segmentation, cfg.detection)
            return p
        final_peaks["stability"] = compute_peak_stability(y_raw, sigma_t, final_peaks, detector, cfg.stability)
        final_peaks["stability_status"] = final_peaks["stability"].map(stability_label)
    else:
        final_peaks["stability"] = pd.Series(dtype=float)
        final_peaks["stability_status"] = pd.Series(dtype=str)
    global_metrics, global_peaks, _ = evaluate_peaks(global_peaks, labels)
    final_metrics, final_peaks, label_table = evaluate_peaks(final_peaks, labels)
    return PipelineResult(channel_id, t, y_raw, y_smooth, residual, sigma_t, low, high, segments, segment_table, global_peaks, final_peaks, label_table, global_metrics, final_metrics, cfg)


def run_pipeline(h5_path: str | Path, labels_path: str | Path, channel_id: int, *, config: PipelineConfig | None = None) -> PipelineResult:
    """Load one channel and its references from disk, then run the full pipeline."""
    t, y = load_chromatogram_h5(h5_path, channel_id)
    labels = labels_for_channel(load_label_table(labels_path), channel_id)
    return run_arrays(t, y, labels, channel_id=channel_id, config=config)
