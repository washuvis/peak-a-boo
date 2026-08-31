"""Decision-uncertainty and Monte Carlo stability scores."""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd
from .config import ScoringConfig, StabilityConfig


def add_weber_scores(peaks: pd.DataFrame, config: ScoringConfig) -> pd.DataFrame:
    """Add prominence-to-noise detectability, boundary margin, and uncertainty columns."""
    out = peaks.copy()
    if out.empty:
        for col in ["weber_score", "weber_margin", "decision_uncertainty"]:
            out[col] = pd.Series(dtype=float)
        return out
    sigma = np.maximum(pd.to_numeric(out["sigma_at_peak"], errors="coerce").fillna(1e-12).to_numpy(dtype=float), 1e-12)
    score = out["prominence"].to_numpy(dtype=float) / sigma
    margin = score - float(config.weber_k)
    uncertainty = 1.0 / (1.0 + np.abs(margin) / max(float(config.weber_k), 1e-12))
    out["weber_score"] = score
    out["weber_margin"] = margin
    out["decision_uncertainty"] = uncertainty
    return out


def compute_peak_stability(y_raw: np.ndarray, sigma_t: np.ndarray, reference_peaks: pd.DataFrame, detector: Callable[[np.ndarray], pd.DataFrame], config: StabilityConfig) -> np.ndarray:
    """Estimate how often each candidate is recovered after noise-scaled perturbations.

    For each perturbation run, Gaussian noise is scaled by the local ``sigma_t``
    estimate. A candidate counts as recovered when a rerun detection falls within
    ``match_tolerance_samples`` of its original sample index.
    """
    if reference_peaks.empty:
        return np.array([], dtype=float)
    if config.n_runs <= 0:
        return np.full(len(reference_peaks), np.nan)
    ref = reference_peaks["sample_idx"].to_numpy(dtype=int)
    hits = np.zeros(len(ref), dtype=int)
    rng = np.random.default_rng(config.seed)
    for _ in range(int(config.n_runs)):
        perturbation = rng.normal(0.0, sigma_t * float(config.perturbation_scale), size=len(y_raw))
        observed = detector(y_raw + perturbation)
        found = observed["sample_idx"].to_numpy(dtype=int) if not observed.empty else np.array([], dtype=int)
        if found.size:
            for i, idx in enumerate(ref):
                hits[i] += int(np.min(np.abs(found - idx)) <= int(config.match_tolerance_samples))
    return hits / float(config.n_runs)


def stability_label(value: float) -> str:
    """Convert a numeric perturbation-recovery rate into a plain-language label."""
    if not np.isfinite(value):
        return "Not computed"
    if value >= 0.80:
        return "Stable"
    if value >= 0.50:
        return "Borderline"
    return "Fragile"
