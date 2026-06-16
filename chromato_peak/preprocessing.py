"""Signal smoothing and local uncertainty estimation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    window = max(1, int(window))
    if window == 1:
        return y.copy()
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def estimate_local_noise_arrays(y_raw: np.ndarray, y_smooth: np.ndarray, noise_window: int, uncertainty_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual = np.asarray(y_raw, dtype=float) - np.asarray(y_smooth, dtype=float)
    window = max(5, int(noise_window))
    sigma = pd.Series(residual).rolling(window=window, center=True, min_periods=max(3, window // 4)).std(ddof=0).to_numpy(dtype=float)
    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    fallback = float(np.median(positive)) if positive.size else max(float(np.std(residual)), 1e-12)
    sigma = np.nan_to_num(sigma, nan=fallback, posinf=fallback, neginf=fallback)
    sigma = np.maximum(sigma, max(fallback * 1e-6, 1e-12))
    multiplier = float(uncertainty_sigma)
    return residual, sigma, y_smooth - multiplier * sigma, y_smooth + multiplier * sigma


def segment_noise_summary(y_raw: np.ndarray, y_smooth: np.ndarray, sigma_t: np.ndarray) -> dict[str, float]:
    sigma = np.asarray(sigma_t, dtype=float)
    return {
        "baseline": float(np.median(np.asarray(y_smooth, dtype=float))),
        "noise": float(np.median(sigma[np.isfinite(sigma)])) if sigma.size else 0.0,
        "residual_rms": float(np.sqrt(np.mean((np.asarray(y_raw) - np.asarray(y_smooth)) ** 2))),
    }
