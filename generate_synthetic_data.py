"""Generate the public synthetic demo dataset used by Peak-a-boo.

This script creates new chromatogram signals and reference intervals from a
fixed random seed. It does not read or derive values from any source research dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
H5_PATH = DATA_DIR / "synthetic_chromatograms.h5"
REFERENCE_PATH = DATA_DIR / "synthetic_reference.xlsx"


@dataclass(frozen=True)
class PeakSpec:
    center: float
    amplitude: float
    sigma: float
    label: bool = True


def gaussian(t: np.ndarray, center: float, amplitude: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def make_channel(channel_id: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 16.0, 3201)

    baseline = (
        0.035
        + 0.0025 * np.sin(2 * np.pi * t / 7.5 + 0.17 * seed)
        + 0.0007 * t
        + 0.0012 * np.sin(2 * np.pi * t / 1.9)
    )

    shift = 0.06 * (channel_id - 1001)
    scale = 1.0 + 0.05 * (channel_id - 1001)
    peaks = [
        PeakSpec(1.20 + shift, 0.016 * scale, 0.050),
        PeakSpec(2.55 - 0.3 * shift, 0.028 * scale, 0.070),
        PeakSpec(3.90 + 0.2 * shift, 0.012 * scale, 0.040),
        PeakSpec(4.72 + 0.15 * shift, 0.105 * scale, 0.085),
        PeakSpec(4.98 + 0.1 * shift, 0.155 * scale, 0.070),
        PeakSpec(5.24 + 0.1 * shift, 0.060 * scale, 0.105),
        PeakSpec(6.85 - 0.1 * shift, 0.020 * scale, 0.060),
        PeakSpec(8.18 + 0.2 * shift, 0.014 * scale, 0.045),
        PeakSpec(9.42 - 0.1 * shift, 0.032 * scale, 0.090),
        PeakSpec(10.70 + 0.25 * shift, 0.095 * scale, 0.120),
        PeakSpec(12.15 - 0.15 * shift, 0.018 * scale, 0.050),
        PeakSpec(12.56 + 0.10 * shift, 0.014 * scale, 0.043),
        PeakSpec(13.85 + 0.05 * shift, 0.040 * scale, 0.085),
        PeakSpec(14.72 - 0.08 * shift, 0.022 * scale, 0.055),
        # Unlabeled artifacts to create realistic detector-only candidates.
        PeakSpec(7.45 + 0.03 * shift, 0.010 * scale, 0.026, label=False),
        PeakSpec(11.62 - 0.02 * shift, 0.008 * scale, 0.022, label=False),
    ]

    clean = baseline.copy()
    labels: list[dict] = []
    peak_id_base = (channel_id - 1000) * 100
    label_index = 0
    for spec in peaks:
        clean += gaussian(t, spec.center, spec.amplitude, spec.sigma)
        if spec.label:
            label_index += 1
            start = max(0.0, spec.center - 2.35 * spec.sigma)
            end = min(16.0, spec.center + 2.35 * spec.sigma)
            labels.append(
                {
                    "PeakId": peak_id_base + label_index,
                    "ChannelId": channel_id,
                    "StartTime": start,
                    "EndTime": end,
                    "RetentionTime": spec.center,
                    "Height": spec.amplitude,
                    "Area": spec.amplitude * spec.sigma * np.sqrt(2.0 * np.pi),
                    "Width": 2.355 * spec.sigma,
                    "Compound": f"Synthetic compound {label_index:02d}",
                    "Source": "Generated synthetic reference",
                }
            )

    local_noise = 0.00075 + 0.00045 * (1 + np.sin(2 * np.pi * t / 5.2 + channel_id)) / 2
    local_noise += 0.0009 * np.exp(-0.5 * ((t - 5.0) / 0.65) ** 2)
    noise = rng.normal(0.0, local_noise)
    ripple = 0.00055 * np.sin(2 * np.pi * t * (5.5 + 0.15 * (channel_id - 1001)))
    y = clean + noise + ripple

    # A deterministic injection artifact, visually useful but intentionally unlabeled.
    y[80] -= 0.055 + 0.004 * (channel_id - 1001)
    return t.astype(float), y.astype(float), labels


def generate() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_labels: list[dict] = []
    with h5py.File(H5_PATH, "w") as h5:
        h5.attrs["dataset_name"] = "Peak-a-boo public synthetic demo"
        h5.attrs["generated"] = "deterministic; fixed seeds; no source research data"
        for offset, channel_id in enumerate(range(1001, 1007)):
            t, y, labels = make_channel(channel_id, seed=8400 + offset)
            group = h5.create_group(str(channel_id))
            group.create_dataset("time", data=t, compression="gzip")
            group.create_dataset("values", data=y, compression="gzip")
            group.attrs["description"] = "Synthetic chromatogram generated from Gaussian peaks, drift, and noise"
            all_labels.extend(labels)

    reference = pd.DataFrame(all_labels).sort_values(["ChannelId", "StartTime"]).reset_index(drop=True)
    with pd.ExcelWriter(REFERENCE_PATH, engine="openpyxl") as writer:
        reference.to_excel(writer, sheet_name="reference_peaks", index=False)
        pd.DataFrame(
            [
                {"Field": "Dataset", "Value": "Peak-a-boo public synthetic demo"},
                {"Field": "Origin", "Value": "Generated from fixed mathematical functions and random seeds"},
                {"Field": "Privacy", "Value": "Contains no source research data"},
                {"Field": "Channels", "Value": "1001-1006"},
                {"Field": "License note", "Value": "Use for demonstrations and interface testing"},
            ]
        ).to_excel(writer, sheet_name="about", index=False)

    print(f"Created {H5_PATH}")
    print(f"Created {REFERENCE_PATH}")
    print(f"Reference rows: {len(reference)}")


if __name__ == "__main__":
    generate()
