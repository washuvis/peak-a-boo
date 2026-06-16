"""Configuration objects for the Peak-a-boo workbench pipeline."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PreprocessingConfig:
    smooth_window: int = 2
    noise_window: int = 51
    uncertainty_sigma: float = 2.0


@dataclass(frozen=True)
class SegmentationConfig:
    mode: str = "fixed"
    window_points: int = 500
    overlap_points: int = 100
    nbands: int = 3
    min_band_run: int = 80


@dataclass(frozen=True)
class DetectionConfig:
    distance: int = 22
    prominence_floor: float = 5e-5
    prominence_k: float = 6.0
    global_prominence: Optional[float] = 0.00018
    dedup_tolerance: int = 8
    width: Optional[tuple[float, float]] = None
    height_k: Optional[float] = None


@dataclass(frozen=True)
class ScoringConfig:
    weber_k: float = 15.96


@dataclass(frozen=True)
class StabilityConfig:
    n_runs: int = 10
    perturbation_scale: float = 1.0
    match_tolerance_samples: int = 14
    seed: int = 17


@dataclass(frozen=True)
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)

    def to_dict(self) -> dict:
        return asdict(self)
