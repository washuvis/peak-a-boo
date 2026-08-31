"""Local signal-region segmentation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import SegmentationConfig


@dataclass(frozen=True)
class Segment:
    """Describe one overlapping local region used by the segment-dependent detector."""

    segment_id: int
    start_idx: int
    end_idx: int
    time_start: float
    time_end: float
    label: str


def make_segments(t: np.ndarray, y_smooth: np.ndarray, config: SegmentationConfig) -> list[Segment]:
    """Create overlapping fixed-width regions and record each region's time range and baseline."""
    n = len(t)
    if n == 0:
        return []
    window = min(n, max(20, int(config.window_points)))
    overlap = min(window - 1, max(0, int(config.overlap_points)))
    step = max(1, window - overlap)
    starts = list(range(0, max(1, n - window + 1), step))
    if not starts or starts[-1] + window < n:
        starts.append(max(0, n - window))
    unique_starts = sorted(set(starts))
    segments: list[Segment] = []
    for i, start in enumerate(unique_starts):
        end = min(n, start + window)
        local_level = float(np.median(y_smooth[start:end]))
        label = f"Local region {i + 1} | baseline {local_level:.4g}"
        segments.append(Segment(i, start, end, float(t[start]), float(t[end - 1]), label))
    return segments
