"""Data loading and validation utilities for the public synthetic demo."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple
import h5py
import numpy as np
import pandas as pd

H5_NAME = "synthetic_chromatograms.h5"
REFERENCE_XLSX_NAME = "synthetic_reference.xlsx"
REFERENCE_CSV_NAME = "synthetic_reference.csv"
REFERENCE_SHEET = "reference_peaks"


def resolve_existing_file(filename: str, explicit_path: str | Path | None = None, extra_candidates: Sequence[str | Path] | None = None) -> Optional[Path]:
    candidates: list[Path] = []
    if explicit_path:
        p = Path(explicit_path).expanduser()
        candidates.append(p / filename if p.is_dir() else p)
    if extra_candidates:
        for candidate in extra_candidates:
            p = Path(candidate).expanduser()
            candidates.append(p / filename if p.is_dir() else p)
    roots = [Path.cwd(), Path.cwd() / "data", Path("/mnt/data")]
    candidates.extend(root / filename for root in roots)
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve(strict=False))
        if key not in seen and p.exists() and p.is_file():
            return p
        seen.add(key)
    return None


def list_h5_channels(h5_path: str | Path) -> list[int]:
    channels: list[int] = []
    with h5py.File(h5_path, "r") as h5:
        for key in h5.keys():
            try:
                group = h5[key]
                if "time" in group and "values" in group:
                    channels.append(int(key))
            except (ValueError, TypeError):
                continue
    return sorted(channels)


def load_chromatogram_h5(h5_path: str | Path, channel_id: int) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        key = str(int(channel_id))
        if key not in h5:
            raise KeyError(f"Channel {channel_id} was not found in {h5_path}.")
        group = h5[key]
        required = {"time", "values"}
        if not required.issubset(group.keys()):
            raise KeyError(f"Channel {channel_id} must include datasets: time and values.")
        t = np.asarray(group["time"][:], dtype=float)
        y = np.asarray(group["values"][:], dtype=float)
    if not len(t) or len(t) != len(y):
        raise ValueError(f"Channel {channel_id} contains empty or mismatched time/value arrays.")
    if not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError(f"Channel {channel_id} contains non-finite values.")
    order = np.argsort(t)
    return t[order], y[order]


def load_label_table(path: str | Path) -> pd.DataFrame:
    """Load synthetic reference intervals while retaining workbook row identity."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        workbook = pd.ExcelFile(path)
        sheet = REFERENCE_SHEET if REFERENCE_SHEET in workbook.sheet_names else workbook.sheet_names[0]
        labels = pd.read_excel(path, sheet_name=sheet)
        labels["_SourceExcelRow"] = np.arange(2, len(labels) + 2, dtype=int)
    elif path.suffix.lower() == ".csv":
        labels = pd.read_csv(path)
        labels["_SourceExcelRow"] = np.arange(2, len(labels) + 2, dtype=int)
    else:
        raise ValueError(f"Unsupported reference file extension: {path.suffix}")
    needed = ["ChannelId", "StartTime", "EndTime", "RetentionTime"]
    missing = [c for c in needed if c not in labels.columns]
    if missing:
        raise ValueError(f"Reference table is missing required columns: {missing}")
    labels = labels.copy()
    numeric = ["ChannelId", "StartTime", "EndTime", "RetentionTime", "Area", "Height", "Width", "PeakId"]
    for col in numeric:
        if col in labels:
            labels[col] = pd.to_numeric(labels[col], errors="coerce")
    labels = labels.dropna(subset=["ChannelId", "StartTime", "EndTime", "RetentionTime"])
    labels = labels[(labels["StartTime"] >= 0) & (labels["EndTime"] > labels["StartTime"])]
    labels["ChannelId"] = labels["ChannelId"].astype(int)
    sort_cols = [c for c in ["ChannelId", "StartTime", "EndTime"] if c in labels]
    return labels.sort_values(sort_cols).reset_index(drop=True)


def labels_for_channel(labels: pd.DataFrame, channel_id: int) -> pd.DataFrame:
    return labels.loc[labels["ChannelId"] == int(channel_id)].copy().reset_index(drop=True)


def channel_label_counts(labels: pd.DataFrame) -> pd.DataFrame:
    return (labels.groupby("ChannelId", as_index=False).size().rename(columns={"size": "n_labels"})
            .sort_values(["n_labels", "ChannelId"], ascending=[False, True]).reset_index(drop=True))


def data_manifest(h5_path: str | Path, labels_path: str | Path) -> dict:
    channels = list_h5_channels(h5_path)
    labels = load_label_table(labels_path)
    common = sorted(set(channels).intersection(labels["ChannelId"].unique()))
    return {
        "signal_path": str(h5_path),
        "reference_path": str(labels_path),
        "n_signal_channels": len(channels),
        "n_reference_rows": int(len(labels)),
        "n_reference_channels": int(labels["ChannelId"].nunique()),
        "n_common_channels": len(common),
        "common_channels": common,
    }
