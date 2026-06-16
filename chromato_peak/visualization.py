"""Plotly chromatogram views for the Peak-a-boo review workbench.

Every view is a viewport over the same immutable ``PipelineResult`` arrays.
Selecting a review cue changes axis bounds and overlays only: raw/smoothed
samples are never recomputed, renormalized, or replaced by a different trace.

View semantics:
- Labeled Data: show detector apexes and optional synthetic labeled reference
  intervals as analytical evidence; do not hide detections.
- AI Review: add review-priority / uncertainty encodings on top of the same
  official evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .pipeline import PipelineResult

STATUS_STYLE = {
    "accepted": {"fill": "rgba(16,185,129,0.14)", "accent": "#10B981", "symbol": "circle-open"},
    "exception_resolved": {"fill": "rgba(16,185,129,0.14)", "accent": "#10B981", "symbol": "diamond-open"},
    "exception": {"fill": "rgba(244,63,94,0.15)", "accent": "#F43F5E", "symbol": "diamond"},
    "open-high": {"fill": "rgba(244,63,94,0.13)", "accent": "#E11D48", "symbol": "circle"},
    "open-medium": {"fill": "rgba(245,158,11,0.14)", "accent": "#D97706", "symbol": "circle"},
    "open-low": {"fill": "rgba(16,185,129,0.13)", "accent": "#059669", "symbol": "circle"},
}
OFFICIAL_SELECTION_STYLE = {"fill": "rgba(37,99,235,0.11)", "accent": "#2563EB", "symbol": "circle"}
OFFICIAL_PEAK_COLOR = "#2563EB"


def _style_for_region(region: pd.Series, *, use_ai_status_style: bool = True) -> dict[str, str]:
    if not use_ai_status_style:
        return OFFICIAL_SELECTION_STYLE
    state = str(region.get("workflow_state", "open"))
    if state in STATUS_STYLE:
        return STATUS_STYLE[state]
    return STATUS_STYLE[f"open-{region.get('severity', 'medium')}"]


def _finite_range(arrays: list[np.ndarray], *, lower_padding: float, upper_padding: float) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array, dtype=float).ravel() for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)
    y_min, y_max = float(values.min()), float(values.max())
    span = max(y_max - y_min, abs(y_max) * 0.02, 1e-8)
    return (y_min - lower_padding * span, y_max + upper_padding * span)


def full_trace_y_range(result: PipelineResult, compare_signal: np.ndarray | None = None) -> tuple[float, float]:
    """Return one stable vertical range for overview and focused review views."""
    arrays = [result.raw_signal, result.smoothed_signal, result.band_low, result.band_high]
    if compare_signal is not None:
        arrays.append(compare_signal)
    return _finite_range(arrays, lower_padding=0.06, upper_padding=0.08)


def selected_focus_ranges(
    result: PipelineResult,
    selected_region: pd.Series,
    focused: bool,
    *,
    magnify_y_axis: bool = False,
    compare_signal: np.ndarray | None = None,
) -> tuple[tuple[float, float] | None, tuple[float, float]]:
    """Return viewport ranges while preserving one underlying signal trace."""
    fixed_y = full_trace_y_range(result, compare_signal)
    if not focused:
        return None, fixed_y

    start = float(selected_region["time_start"])
    end = float(selected_region["time_end"])
    midpoint = (start + end) / 2.0
    interval = max(end - start, 0.015)
    padding = max(interval * 2.25, 0.12)
    half_span = interval / 2.0 + padding
    x_range = (
        max(float(result.time.min()), midpoint - half_span),
        min(float(result.time.max()), midpoint + half_span),
    )

    if not magnify_y_axis:
        return x_range, fixed_y

    mask = (result.time >= x_range[0]) & (result.time <= x_range[1])
    if not np.any(mask):
        return x_range, fixed_y
    arrays = [result.raw_signal[mask], result.smoothed_signal[mask], result.band_low[mask], result.band_high[mask]]
    if compare_signal is not None:
        arrays.append(compare_signal[mask])
    return x_range, _finite_range(arrays, lower_padding=0.18, upper_padding=0.28)


def _case_key_by_peak_id(queue: pd.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if queue.empty or "peak_id" not in queue:
        return mapping
    for _, row in queue.iterrows():
        try:
            peak_id = int(row["peak_id"])
        except (TypeError, ValueError):
            continue
        if peak_id >= 0:
            mapping[peak_id] = str(row["case_key"])
    return mapping


def chromatogram_figure(
    result: PipelineResult,
    queue: pd.DataFrame,
    selected_region: pd.Series,
    *,
    show_raw: bool,
    show_band: bool,
    show_labels: bool,
    show_official_peaks: bool = False,
    compare_signal: np.ndarray | None = None,
    ml_peaks: pd.DataFrame | None = None,
    zoom_selected: bool = False,
    show_review_regions: bool = True,
    use_ai_status_style: bool = True,
    magnify_y_axis: bool = False,
) -> go.Figure:
    """Build the main chromatogram view from a single official pipeline result.

    ``show_official_peaks`` renders the official detector output directly from
    ``result.final_peaks``. These markers are different from AI review flags:
    they are the output of the locked method itself and therefore belong in the
    Labeled Data view.
    """
    time = np.asarray(result.time, dtype=float)
    focused = bool(zoom_selected)
    selected_key = str(selected_region.get("case_key", ""))
    selected_style = _style_for_region(selected_region, use_ai_status_style=use_ai_status_style)
    baseline_y = np.full_like(time, float(np.median(result.smoothed_signal)), dtype=float)
    fig = go.Figure()

    if show_band:
        fig.add_trace(go.Scatter(
            x=time, y=np.asarray(result.band_high, dtype=float), mode="lines", line={"width": 0},
            hoverinfo="skip", showlegend=False, name="Uncertainty band high",
        ))
        fig.add_trace(go.Scatter(
            x=time, y=np.asarray(result.band_low, dtype=float), mode="lines", line={"width": 0},
            fill="tonexty", fillcolor="rgba(14,165,233,0.17)",
            hoverinfo="skip", showlegend=False, name="Uncertainty band",
        ))
    if show_raw:
        fig.add_trace(go.Scatter(
            x=time, y=np.asarray(result.raw_signal, dtype=float), mode="lines", name="Raw signal",
            line={"color": "#172033", "width": 1.55},
            hovertemplate="%{x:.6f} min<br>Raw: %{y:.8g}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=time, y=np.asarray(result.smoothed_signal, dtype=float), mode="lines", name="Smoothed signal",
        line={"color": "#5B5FEF", "width": 2.35},
        hovertemplate="%{x:.6f} min<br>Smoothed: %{y:.8g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=time, y=baseline_y, mode="lines", name="Baseline",
        line={"color": "#94A3B8", "width": 1.2, "dash": "dot"},
        hovertemplate="%{x:.6f} min<br>Baseline: %{y:.8g}<extra></extra>",
    ))
    if compare_signal is not None:
        fig.add_trace(go.Scatter(
            x=time, y=np.asarray(compare_signal, dtype=float), mode="lines", name="Comparison run",
            line={"color": "#0EA5E9", "width": 1.5, "dash": "dash"},
            hovertemplate="%{x:.6f} min<br>Comparison: %{y:.8g}<extra></extra>",
        ))

    if show_labels:
        for _, label in result.label_table.iterrows():
            fig.add_vrect(
                x0=float(label["StartTime"]), x1=float(label["EndTime"]),
                fillcolor="rgba(91,95,239,0.09)", line_width=0, layer="below",
            )

    # Locked SOP detections are analytical evidence, not AI cues. They should be
    # visible in the Labeled Data view rather than silently hidden.
    if show_official_peaks and not result.final_peaks.empty:
        official = result.final_peaks.copy()
        apex_indices = [int(np.argmin(np.abs(time - float(t)))) for t in official["time"]]
        y_values = [float(result.smoothed_signal[idx]) for idx in apex_indices]
        linked_cases = _case_key_by_peak_id(queue)
        customdata = [[linked_cases.get(int(peak_id), "")] for peak_id in official["peak_id"]]
        hover = [
            f"<b>Official detected apex #{int(peak_id):03d}</b><br>{float(apex):.6f} min<br>Prominence: {float(prom):.6g}<extra></extra>"
            for peak_id, apex, prom in zip(official["peak_id"], official["time"], official["prominence"])
        ]
        fig.add_trace(go.Scatter(
            x=np.asarray(official["time"], dtype=float), y=y_values,
            mode="markers", name="Official detected apexes", customdata=customdata,
            marker={"size": 7.5, "color": OFFICIAL_PEAK_COLOR, "symbol": "circle-open", "line": {"width": 1.2, "color": OFFICIAL_PEAK_COLOR}},
            hovertemplate=hover,
            showlegend=False,
        ))

    if show_review_regions:
        for _, region in queue.iterrows():
            is_selected = str(region["case_key"]) == selected_key
            style = _style_for_region(region, use_ai_status_style=True)
            start, end = float(region["time_start"]), float(region["time_end"])
            apex = float(region["time"])
            y_value = float(result.smoothed_signal[int(np.argmin(np.abs(time - apex)))])
            strong = focused and is_selected
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=style["fill"] if strong else style["fill"].replace("0.13", "0.09").replace("0.14", "0.09").replace("0.15", "0.09"),
                line_color=style["accent"] if strong else "rgba(0,0,0,0)",
                line_width=2.6 if strong else 0,
                layer="below",
            )
            if not strong:
                fig.add_trace(go.Scatter(
                    x=[apex], y=[y_value], mode="markers", showlegend=False,
                    name=str(region["region_id"]), customdata=[[str(region["case_key"])]],
                    marker={"size": 9.5, "color": style["accent"], "symbol": style["symbol"]},
                    hovertemplate=(
                        f"<b>{region['region_id']}</b><br>{region.get('title','')}<br>"
                        f"{region.get('source_type','')}<br>"
                        "%{x:.6f} min<br>Click its queue card to focus<extra></extra>"
                    ),
                ))

    if ml_peaks is not None and not ml_peaks.empty:
        ml = ml_peaks.copy()
        keep = ml[ml.get("ml_keep", False).astype(bool)] if "ml_keep" in ml else ml
        if not keep.empty:
            apex_indices = [int(np.argmin(np.abs(time - float(t)))) for t in keep["time"]]
            y_values = [float(result.smoothed_signal[idx]) for idx in apex_indices]
            fig.add_trace(go.Scatter(
                x=np.asarray(keep["time"], dtype=float), y=y_values, mode="markers", name="ML-filtered peaks",
                marker={"size": 11, "color": "#7C3AED", "symbol": "star", "line": {"width": 1, "color": "#FFFFFF"}},
                hovertemplate="<b>ML-filtered peak</b><br>%{x:.6f} min<br>p(real)=%{customdata:.3f}<extra></extra>",
                customdata=np.asarray(keep.get("p_ml_real", np.nan), dtype=float),
                showlegend=False,
            ))

    if focused:
        apex = float(selected_region["time"])
        y_selected = float(result.smoothed_signal[int(np.argmin(np.abs(time - apex)))])
        start = float(selected_region["time_start"])
        end = float(selected_region["time_end"])
        fig.add_vrect(
            x0=start, x1=end, fillcolor=selected_style["fill"],
            line_color=selected_style["accent"], line_width=2.7, layer="below",
        )
        fig.add_vline(x=apex, line_color=selected_style["accent"], line_width=1.4, line_dash="dot")
        fig.add_trace(go.Scatter(
            x=[apex], y=[y_selected], mode="markers", showlegend=False, hoverinfo="skip",
            marker={"size": 27, "color": "rgba(255,255,255,0)", "line": {"width": 3, "color": selected_style["accent"]}},
        ))
        fig.add_trace(go.Scatter(
            x=[apex], y=[y_selected], mode="markers", showlegend=False,
            name="Selected review case", customdata=[[selected_key]],
            marker={"size": 11, "color": selected_style["accent"], "symbol": selected_style["symbol"], "line": {"width": 1, "color": "#FFFFFF"}},
            hovertemplate=f"<b>{selected_region['region_id']} · selected</b><br>{selected_region.get('title','')}<br>{selected_region.get('source_type','')}<br>%{{x:.6f}} min<extra></extra>",
        ))
        fig.add_annotation(
            x=apex, y=y_selected, text=f"{selected_region['region_id']} · selected",
            showarrow=True, arrowhead=0, ax=0, ay=-40,
            arrowcolor=selected_style["accent"], bgcolor="rgba(255,255,255,0.96)", bordercolor=selected_style["accent"],
            borderwidth=1, borderpad=6, font={"size": 11, "color": "#111827"},
        )
        if magnify_y_axis:
            fig.add_annotation(
                x=0.01, y=0.985, xref="paper", yref="paper", xanchor="left", yanchor="top",
                text="Vertical magnification on · signal samples unchanged", showarrow=False,
                bgcolor="#FFFBEB", bordercolor="#FDE68A", borderwidth=1, borderpad=5,
                font={"size": 10, "color": "#B45309"},
            )

    x_range, y_range = selected_focus_ranges(
        result, selected_region, focused,
        magnify_y_axis=bool(magnify_y_axis), compare_signal=compare_signal,
    )
    if x_range:
        fig.update_xaxes(range=list(x_range))
    fig.update_yaxes(range=list(y_range))

    fig.update_layout(
        height=382,
        margin={"l": 12, "r": 12, "t": 18, "b": 38},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBFCFF",
        showlegend=False, hovermode="closest", clickmode="event+select", dragmode="zoom",
        font={"family": "Inter, Arial, sans-serif", "size": 12, "color": "#526581"},
        xaxis={
            "title": {"text": "Retention time (min)", "font": {"size": 11, "color": "#64748B"}},
            "gridcolor": "#E9EDF5", "zeroline": False, "showline": False, "ticks": "outside",
        },
        yaxis={
            "title": {"text": "Signal", "font": {"size": 11, "color": "#64748B"}},
            "gridcolor": "#E9EDF5", "zeroline": False, "showline": False,
        },
        hoverlabel={"bgcolor": "#FFFFFF", "font": {"color": "#111827", "size": 12}, "bordercolor": "#C7D2FE"},
        selectdirection="h",
    )
    return fig
