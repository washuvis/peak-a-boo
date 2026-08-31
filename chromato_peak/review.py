"""Review-queue construction, data provenance, workflow state, explanations, and reporting."""
from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .persistence import RESOLVED_STATES, WORKFLOW_LABELS, workflow_status
from .pipeline import PipelineResult


def _case_key(result: PipelineResult, peak_id: int, start: float, end: float) -> str:
    """Return an identity tied to source analytical evidence, never to card order."""
    channel_id = "unknown" if result.channel_id is None else str(int(result.channel_id))
    if peak_id >= 0:
        return f"channel:{channel_id}:candidate:{peak_id}"
    return f"channel:{channel_id}:reference-miss:{start:.6f}:{end:.6f}"


def _finite_int(value: object) -> int | None:
    """Convert a finite numeric value to an integer, or return ``None``."""
    try:
        number = float(value)
        return int(number) if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _reference_excel_row(label: pd.Series | None) -> float:
    """Return the original Excel row retained for a reference interval."""
    if label is None:
        return np.nan
    value = label.get("_SourceExcelRow", np.nan)
    return float(value) if pd.notna(value) else np.nan


def _reference_identity(label: pd.Series | None) -> tuple[str, int | None, int | None]:
    """Return a stable user-facing ID from the source workbook when available."""
    if label is None:
        return "", None, None
    source_row = _finite_int(label.get("_SourceExcelRow", np.nan))
    peak_id = _finite_int(label.get("PeakId", np.nan))
    if peak_id is not None:
        return f"REF-{peak_id:03d}", peak_id, source_row
    if source_row is not None:
        return f"ROW-{source_row:04d}", None, source_row
    return "REFERENCE", None, None


def _source_fields(linked_label: pd.Series | None, peak_id: int, match_status: str) -> dict[str, Any]:
    """Build provenance fields for a reference-linked case or detector-only candidate."""
    region_id, reference_peak_id, source_row = _reference_identity(linked_label)
    if linked_label is not None:
        source_status = "reference interval missed" if match_status == "FN" else "reference-linked detection"
        return {
            "region_id": region_id,
            "reference_peak_id": reference_peak_id,
            "label_source_row": float(source_row) if source_row is not None else np.nan,
            "source_type": source_status,
            "data_origin": f"synthetic_reference.xlsx · PeakId {reference_peak_id if reference_peak_id is not None else 'n/a'} · row {source_row if source_row is not None else 'n/a'}",
            "interval_origin": "Reference interval from synthetic_reference.xlsx",
        }
    return {
        "region_id": f"CAND-{peak_id:03d}",
        "reference_peak_id": np.nan,
        "label_source_row": np.nan,
        "source_type": "unmatched detector candidate",
        "data_origin": f"synthetic_chromatograms.h5 · detected candidate #{peak_id}; outside all synthetic reference intervals",
        "interval_origin": "Candidate inspection window derived from detected apex/width",
    }


def _peak_region(result: PipelineResult, row: pd.Series, title: str, review_status: str, severity: str) -> dict[str, Any]:
    """Convert one detected candidate into a review-queue record with provenance."""
    time = float(row["time"])
    label_idx = row.get("reference_index", np.nan)
    linked_label: pd.Series | None = None
    if pd.notna(label_idx) and int(label_idx) < len(result.label_table):
        linked_label = result.label_table.loc[int(label_idx)]
        start, end = float(linked_label["StartTime"]), float(linked_label["EndTime"])
    else:
        width_time = max(float(np.median(np.diff(result.time))) * float(row.get("width_samples", 10)) * 2.2, 0.05)
        start, end = time - width_time, time + width_time
    peak_id = int(row["peak_id"])
    match_status = str(row.get("match_status", "UNLABELED"))
    source = _source_fields(linked_label, peak_id, match_status)
    return {
        "channel_id": result.channel_id,
        "case_key": _case_key(result, peak_id, start, end),
        "peak_id": peak_id,
        "time": time,
        "time_start": start,
        "time_end": end,
        "title": title,
        "review_status": review_status,
        "severity": severity,
        "stability": float(row.get("stability", np.nan)),
        "weber_score": float(row.get("weber_score", np.nan)),
        "weber_margin": float(row.get("weber_margin", np.nan)),
        "local_noise": float(row.get("sigma_at_peak", np.nan)),
        "prominence": float(row.get("prominence", np.nan)),
        "match_status": match_status,
        **source,
    }


def build_review_queue(result: PipelineResult, max_items: int = 10) -> pd.DataFrame:
    """Create a priority queue from synthetic signal decisions and synthetic reference intervals.

    Human-facing IDs are source-grounded: ``REF-###`` uses ``PeakId`` from
    ``synthetic_reference.xlsx`` and ``CAND-###`` uses the detector candidate ID
    computed from the synthetic HDF5 trace.
    """
    peaks = result.final_peaks.copy()
    rows: list[dict[str, Any]] = []
    used_keys: set[str] = set()

    def add_unique(item: dict[str, Any]) -> None:
        """Append a case once while respecting the queue-size limit."""
        key = str(item["case_key"])
        if key not in used_keys and len(rows) < max_items:
            rows.append(item)
            used_keys.add(key)

    if not peaks.empty:
        boundary = (
            peaks.assign(distance_to_boundary=peaks["weber_margin"].abs())
            .sort_values(["distance_to_boundary", "stability"])
            .head(2)
        )
        if len(boundary) > 0:
            add_unique(_peak_region(result, boundary.iloc[0], "Borderline small peak", "Needs review", "high"))
        if len(boundary) > 1:
            add_unique(_peak_region(result, boundary.iloc[1], "Baseline-sensitive boundary", "Analysis note pending", "medium"))
        stable_tp = peaks[(peaks["match_status"] == "TP") & (peaks["stability"] >= 0.8)].sort_values("prominence", ascending=False)
        if not stable_tp.empty:
            add_unique(_peak_region(result, stable_tp.iloc[0], "Stable major peak", "Ready to accept", "low"))

    missing = result.label_table[result.label_table["matched_peak_id"].isna()] if "matched_peak_id" in result.label_table else pd.DataFrame()
    if not missing.empty and len(rows) < max_items:
        lab = missing.iloc[0]
        start, end = float(lab["StartTime"]), float(lab["EndTime"])
        source = _source_fields(lab, -1, "FN")
        add_unique({
            "channel_id": result.channel_id,
            "case_key": _case_key(result, -1, start, end),
            "peak_id": -1,
            "time": float(lab["RetentionTime"]),
            "time_start": start,
            "time_end": end,
            "title": "Possible missed shoulder",
            "review_status": "Exception candidate",
            "severity": "high",
            "stability": np.nan,
            "weber_score": np.nan,
            "weber_margin": np.nan,
            "local_noise": float(np.interp(float(lab["RetentionTime"]), result.time, result.sigma_t)),
            "prominence": np.nan,
            "match_status": "FN",
            **source,
        })

    if not peaks.empty and len(rows) < max_items:
        used_peak_ids = {int(row["peak_id"]) for row in rows if int(row["peak_id"]) >= 0}
        extra = (
            peaks[~peaks["peak_id"].isin(used_peak_ids)]
            .assign(priority=peaks["decision_uncertainty"] + (1 - peaks["stability"].fillna(1)))
            .sort_values("priority", ascending=False)
        )
        for _, row in extra.iterrows():
            title = "Uncertain supported peak" if row["match_status"] == "TP" else "Unsupported candidate"
            add_unique(_peak_region(result, row, title, "Needs review", "medium" if row["match_status"] == "TP" else "high"))
            if len(rows) >= max_items:
                break

    return pd.DataFrame(rows)


def apply_review_state(queue: pd.DataFrame, records: Mapping[str, Mapping[str, Any]] | None = None) -> pd.DataFrame:
    """Overlay durable analyst dispositions onto immutable data-derived cases."""
    records = records or {}
    updated = queue.copy()
    if updated.empty:
        return updated
    updated["source_review_status"] = updated["review_status"]
    updated["workflow_state"] = "open"
    updated["analyst_note"] = ""
    updated["decision_at"] = ""
    for idx, row in updated.iterrows():
        record = records.get(str(row["case_key"]), {})
        state = str(record.get("state", "open"))
        if state not in WORKFLOW_LABELS:
            state = "open"
        note = str(record.get("note", "")).strip()
        updated.at[idx, "workflow_state"] = state
        updated.at[idx, "analyst_note"] = note
        updated.at[idx, "decision_at"] = str(record.get("updated_at", ""))
        updated.at[idx, "review_status"] = workflow_status(state, note) if state != "open" or note else row["review_status"]
    return updated


def review_queue_counts(queue: pd.DataFrame) -> dict[str, int]:
    """Count active, completed, and exception cases in the current review queue."""
    if queue.empty or "workflow_state" not in queue:
        return {"active": int(len(queue)), "reviewed": 0, "exceptions": 0, "total": int(len(queue))}
    state = queue["workflow_state"]
    resolved = state.isin(RESOLVED_STATES)
    return {
        "active": int((~resolved).sum()),
        "reviewed": int(resolved.sum()),
        "exceptions": int(state.str.startswith("exception").sum()),
        "total": int(len(queue)),
    }


def filter_review_queue(queue: pd.DataFrame, view: str) -> pd.DataFrame:
    """Filter queue into active or completed review work."""
    if queue.empty or "workflow_state" not in queue:
        return queue.copy()
    resolved = queue["workflow_state"].isin(RESOLVED_STATES)
    if view == "Active":
        filtered = queue[~resolved].copy()
    elif view == "Reviewed":
        filtered = queue[resolved].copy()
    else:
        filtered = queue.copy()
    rank = {"exception": 0, "open": 1, "accepted": 2, "exception_resolved": 3}
    filtered["_workflow_rank"] = filtered["workflow_state"].map(rank).fillna(1)
    return filtered.sort_values(["_workflow_rank", "time_start"]).drop(columns=["_workflow_rank"])


def pipeline_attribution(result: PipelineResult, selected: Mapping[str, Any] | pd.Series) -> pd.DataFrame:
    """Create a plain-language trace of the computed evidence for one selected case."""
    stability = float(selected.get("stability", np.nan))
    score = float(selected.get("weber_score", np.nan))
    margin = float(selected.get("weber_margin", np.nan))
    noise = float(selected.get("local_noise", np.nan))
    prominence = float(selected.get("prominence", np.nan))
    rows = [
        {"Stage": "Signal source", "Evidence": f"synthetic_chromatograms.h5 / channel {int(selected.get('channel_id'))}", "Interpretation": "Raw time-intensity samples used to compute this view."},
        {"Stage": "Reference source", "Evidence": str(selected.get("data_origin", "")), "Interpretation": str(selected.get("interval_origin", ""))},
        {"Stage": "Smoothing", "Evidence": f"Moving average window = {result.config.preprocessing.smooth_window}", "Interpretation": "Reduces high-frequency variation before detection."},
        {"Stage": "Local noise", "Evidence": f"σ(t) = {noise:.6g}" if np.isfinite(noise) else "No candidate apex", "Interpretation": "Sets the local uncertainty scale for this region."},
        {"Stage": "Prominence", "Evidence": f"{prominence:.6g}" if np.isfinite(prominence) else "No surviving apex", "Interpretation": "Measures vertical separation from nearby baseline."},
        {"Stage": "Weber margin", "Evidence": f"score={score:.2f}; margin={margin:.2f}" if np.isfinite(margin) else "Not available", "Interpretation": "Near-zero margin indicates decision-boundary sensitivity."},
        {"Stage": "Perturbation stability", "Evidence": f"{stability * 100:.0f}% of reruns" if np.isfinite(stability) else "Not evaluated for missed label", "Interpretation": "Low stability means small signal changes flip the detection."},
        {"Stage": "Reference comparison", "Evidence": str(selected.get("match_status", "")), "Interpretation": "TP/FP/FN status relative to the synthetic reference intervals."},
    ]
    return pd.DataFrame(rows)


def add_audit_event(log: list[dict[str, str]], action: str, detail: str) -> None:
    """Insert a timestamped audit event at the front of an in-memory audit list."""
    log.insert(0, {"time": datetime.now().strftime("%H:%M"), "action": action, "detail": detail})


def report_json(
    result: PipelineResult,
    queue: pd.DataFrame,
    audit_log: list[dict],
    decisions: Mapping[str, Any],
    channel_id: int,
    workbook_path: str | None = None,
    signal_path: str | None = None,
) -> str:
    """Serialize method settings, review cases, decisions, and audit history as JSON."""
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "channel_id": channel_id,
        "data_sources": {
            "synthetic_signal_h5": signal_path,
            "synthetic_reference_workbook": workbook_path,
            "review_id_rule": "REF-### is a synthetic reference PeakId; CAND-### is a detector-only synthetic candidate",
        },
        "method": result.config.to_dict(),
        "metrics": result.final_metrics,
        "workflow_summary": review_queue_counts(queue),
        "review_queue": queue.to_dict(orient="records"),
        "review_records": dict(decisions),
        "audit_log": audit_log,
    }
    return json.dumps(payload, indent=2, default=str)
