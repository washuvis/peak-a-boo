"""Comparison of predicted peak apexes against reference intervals."""
from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate_peaks(peaks: pd.DataFrame, labels: pd.DataFrame | None) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    pred = peaks.copy()
    label_table = labels.copy().reset_index(drop=True) if labels is not None else pd.DataFrame(columns=["StartTime", "EndTime"])
    pred["match_status"] = "UNLABELED"
    pred["reference_index"] = np.nan
    label_table["matched_peak_id"] = np.nan
    if label_table.empty:
        return {"TP": 0, "FP": int(len(pred)), "FN": 0, "precision": 0.0, "recall": 0.0, "F1": 0.0}, pred, label_table
    used_pred: set[int] = set()
    for lab_idx, label in label_table.iterrows():
        inside = pred[(pred["time"] >= label["StartTime"]) & (pred["time"] <= label["EndTime"])]
        if inside.empty:
            continue
        # Retain the strongest candidate as the one-to-one match; additional peaks are review exceptions.
        chosen_idx = int(inside["prominence"].idxmax())
        used_pred.add(chosen_idx)
        pred.loc[chosen_idx, "match_status"] = "TP"
        pred.loc[chosen_idx, "reference_index"] = lab_idx
        label_table.loc[lab_idx, "matched_peak_id"] = pred.loc[chosen_idx, "peak_id"]
        extras = [idx for idx in inside.index if idx != chosen_idx]
        pred.loc[extras, "match_status"] = "FP_OVERSEG"
        pred.loc[extras, "reference_index"] = lab_idx
    remaining = pred["match_status"].eq("UNLABELED")
    pred.loc[remaining, "match_status"] = "FP"
    tp = int(pred["match_status"].eq("TP").sum())
    fp = int(pred["match_status"].str.startswith("FP").sum())
    fn = int(label_table["matched_peak_id"].isna().sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1}
    return metrics, pred, label_table
