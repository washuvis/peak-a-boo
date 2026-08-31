"""Global and segment-dependent peak detection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths
from .config import DetectionConfig, PreprocessingConfig, SegmentationConfig
from .preprocessing import moving_average, segment_noise_summary
from .segmentation import Segment, make_segments

PEAK_COLUMNS = ["peak_id", "sample_idx", "time", "height", "prominence", "width_samples", "segment_id", "segment_label", "segment_time_start", "segment_time_end", "local_baseline", "local_noise", "sigma_at_peak", "prominence_threshold", "height_threshold", "detector"]


def empty_peak_table() -> pd.DataFrame:
    """Return an empty peak table with the columns expected by the rest of the pipeline."""
    return pd.DataFrame(columns=PEAK_COLUMNS)


def _peak_features(t: np.ndarray, y_smooth: np.ndarray, peak_idx: np.ndarray, props: dict) -> pd.DataFrame:
    """Compute time, height, prominence, and width for detected peak indexes."""
    peak_idx = np.asarray(peak_idx, dtype=int)
    if peak_idx.size == 0:
        return empty_peak_table()
    prominence = props.get("prominences")
    if prominence is None:
        prominence = peak_prominences(y_smooth, peak_idx)[0]
    widths = peak_widths(y_smooth, peak_idx, rel_height=0.5)[0]
    return pd.DataFrame({"sample_idx": peak_idx, "time": t[peak_idx], "height": y_smooth[peak_idx], "prominence": prominence, "width_samples": widths})


def deduplicate_peaks(peaks: pd.DataFrame, tolerance: int) -> pd.DataFrame:
    """Merge nearby duplicate detections by keeping the most prominent candidate in each group."""
    if peaks.empty:
        return peaks.copy()
    peaks = peaks.sort_values("sample_idx").reset_index(drop=True)
    selected: list[int] = []
    group: list[int] = [0]
    for i in range(1, len(peaks)):
        if int(peaks.loc[i, "sample_idx"]) - int(peaks.loc[i-1, "sample_idx"]) <= tolerance:
            group.append(i)
        else:
            selected.append(int(peaks.loc[group, "prominence"].idxmax()))
            group = [i]
    selected.append(int(peaks.loc[group, "prominence"].idxmax()))
    out = peaks.loc[selected].sort_values("sample_idx").reset_index(drop=True)
    out["peak_id"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def detect_peaks_global(t: np.ndarray, y_raw: np.ndarray, preprocessing: PreprocessingConfig, detection: DetectionConfig, sigma_t: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Run the fixed global-prominence baseline detector and return its peak table."""
    y_smooth = moving_average(y_raw, preprocessing.smooth_window)
    kwargs: dict[str, object] = {"distance": max(1, int(detection.distance))}
    if detection.global_prominence is not None:
        kwargs["prominence"] = float(detection.global_prominence)
    idx, props = find_peaks(y_smooth, **kwargs)
    table = _peak_features(t, y_smooth, idx, props)
    if table.empty:
        return y_smooth, empty_peak_table()
    table["peak_id"] = np.arange(1, len(table)+1)
    table["segment_id"] = -1
    table["segment_label"] = "Official global threshold"
    table["segment_time_start"] = float(t[0])
    table["segment_time_end"] = float(t[-1])
    table["local_baseline"] = float(np.median(y_smooth))
    table["local_noise"] = float(np.median(sigma_t))
    table["sigma_at_peak"] = sigma_t[table["sample_idx"].to_numpy(dtype=int)]
    table["prominence_threshold"] = float(detection.global_prominence or 0.0)
    table["height_threshold"] = np.nan
    table["detector"] = "official_global"
    return y_smooth, table[PEAK_COLUMNS]


def detect_peaks_segmented_from_smooth(t: np.ndarray, y_raw: np.ndarray, y_smooth: np.ndarray, sigma_t: np.ndarray, segmentation: SegmentationConfig, detection: DetectionConfig) -> tuple[list[Segment], pd.DataFrame, pd.DataFrame]:
    """Detect peaks in overlapping local regions using prominence scaled by local noise."""
    segments = make_segments(t, y_smooth, segmentation)
    peak_rows: list[dict] = []
    segment_rows: list[dict] = []
    for seg in segments:
        start, end = seg.start_idx, seg.end_idx
        local_stats = segment_noise_summary(y_raw[start:end], y_smooth[start:end], sigma_t[start:end])
        local_threshold = max(float(detection.prominence_floor), float(detection.prominence_k) * local_stats["noise"])
        idx, props = find_peaks(y_smooth[start:end], distance=max(1, int(detection.distance)), prominence=local_threshold)
        global_idx = idx + start
        features = _peak_features(t, y_smooth, global_idx, props)
        for _, row in features.iterrows():
            p = int(row["sample_idx"])
            peak_rows.append({**row.to_dict(), "segment_id": seg.segment_id, "segment_label": seg.label, "segment_time_start": seg.time_start, "segment_time_end": seg.time_end, "local_baseline": local_stats["baseline"], "local_noise": local_stats["noise"], "sigma_at_peak": float(sigma_t[p]), "prominence_threshold": local_threshold, "height_threshold": np.nan, "detector": "segment_dependent"})
        segment_rows.append({"segment_id": seg.segment_id, "label": seg.label, "time_start": seg.time_start, "time_end": seg.time_end, "local_noise": local_stats["noise"], "prominence_threshold": local_threshold, "n_raw_detections": int(len(features))})
    peaks = pd.DataFrame(peak_rows) if peak_rows else empty_peak_table()
    if not peaks.empty:
        peaks = deduplicate_peaks(peaks, max(0, int(detection.dedup_tolerance)))
        peaks = peaks[PEAK_COLUMNS]
    segments_df = pd.DataFrame(segment_rows)
    return segments, peaks, segments_df
