"""Peak-a-boo Review Workbench — reference-aligned linked-selection UI.

The app preserves the analytical pipeline and workbook-backed analyst state,
while the primary interaction is intentionally simple: select a review case and
the chromatogram immediately re-renders in a focused viewport for that case.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from chromato_peak.config import (
    DetectionConfig,
    PipelineConfig,
    PreprocessingConfig,
    ScoringConfig,
    SegmentationConfig,
    StabilityConfig,
)
from chromato_peak.data_io import (
    labels_for_channel,
    list_h5_channels,
    load_chromatogram_h5,
    load_label_table,
    resolve_existing_file,
)
from chromato_peak.persistence import (
    RESOLVED_STATES,
    ensure_working_label_workbook,
    load_persisted_audit,
    load_review_records,
    persist_review_record,
)
from chromato_peak.pipeline import PipelineResult, run_arrays
from chromato_peak.preprocessing import moving_average
from chromato_peak.review import (
    apply_review_state,
    build_review_queue,
    filter_review_queue,
    pipeline_attribution,
    report_json,
    review_queue_counts,
)
from chromato_peak.visualization import chromatogram_figure

st.set_page_config(
    page_title="Peak-a-boo Review Workbench",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_ROOT = Path(__file__).resolve().parent
LABELED_DATA_VIEW = "Labeled Data"
DATA_DIR = Path(os.environ.get("PEAK_A_BOO_DATA_DIR", str(APP_ROOT / "data"))).expanduser()
OFFICIAL_CONFIG = PipelineConfig(
    preprocessing=PreprocessingConfig(smooth_window=2, noise_window=51, uncertainty_sigma=2.0),
    segmentation=SegmentationConfig(window_points=500, overlap_points=100),
    detection=DetectionConfig(
        distance=22,
        prominence_floor=5e-5,
        prominence_k=2.0,
        global_prominence=0.00018,
        dedup_tolerance=8,
    ),
    scoring=ScoringConfig(weber_k=15.96),
    stability=StabilityConfig(n_runs=10, perturbation_scale=1.0, match_tolerance_samples=14, seed=17),
)

CSS = """
<style>
:root {
  --ink:#111827;
  --ink-2:#1F2937;
  --muted:#667085;
  --muted-2:#98A2B3;
  --line:rgba(148,163,184,.24);
  --line-strong:rgba(100,116,139,.35);
  --surface:rgba(255,255,255,.88);
  --surface-solid:#FFFFFF;
  --surface-soft:#F8FAFC;
  --page:#F3F6FB;
  --brand:#5B5FEF;
  --brand-2:#8B5CF6;
  --brand-soft:#EEF0FF;
  --cyan:#0EA5E9;
  --rose:#F43F5E;
  --amber:#F59E0B;
  --green:#10B981;
  --shadow-sm:0 2px 8px rgba(15,23,42,.06);
  --shadow-md:0 14px 38px rgba(30,41,59,.10), 0 2px 8px rgba(30,41,59,.05);
  --shadow-lg:0 24px 60px rgba(30,41,59,.14), 0 5px 16px rgba(30,41,59,.06);
}
html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  color:var(--ink);
}
.stApp {
  background:
    radial-gradient(circle at 6% -8%, rgba(91,95,239,.13), transparent 30%),
    radial-gradient(circle at 100% 0%, rgba(14,165,233,.10), transparent 24%),
    linear-gradient(180deg,#F8FAFD 0%,#F3F6FB 46%,#F6F8FC 100%);
}
.stApp:before {
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:radial-gradient(rgba(91,95,239,.12) .7px, transparent .7px);
  background-size:20px 20px; opacity:.22; mask-image:linear-gradient(to bottom,black,transparent 44%);
}
[data-testid="stHeader"], [data-testid="stSidebar"], #MainMenu, footer {display:none !important;}
.block-container {padding:1.15rem 1.28rem 1.35rem !important; max-width:100% !important; position:relative; z-index:1;}
div[data-testid="stHorizontalBlock"] {gap:1rem; align-items:flex-start;}
* {scrollbar-width:thin; scrollbar-color:#C7CDDA transparent;}
::-webkit-scrollbar {width:8px;height:8px;} ::-webkit-scrollbar-thumb {background:#C7CDDA;border-radius:20px;} ::-webkit-scrollbar-track {background:transparent;}

/* Premium glass surfaces */
.st-key-header_shell, .st-key-queue_shell, .st-key-method_shell, .st-key-chart_shell,
.st-key-selected_shell, .st-key-explanation_shell, .st-key-audit_shell, .st-key-sandbox_shell,
.st-key-detection_attribution_shell {
  background:var(--surface); border:1px solid rgba(255,255,255,.76); border-radius:22px;
  box-shadow:var(--shadow-md); backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
  position:relative; overflow:hidden;
}
.st-key-header_shell:after, .st-key-chart_shell:after, .st-key-selected_shell:after {
  content:""; position:absolute; left:18px; right:18px; top:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(91,95,239,.48),rgba(14,165,233,.42),transparent);
}
.st-key-header_shell {padding:1.02rem 1.22rem; margin-bottom:1.05rem; min-height:76px;}
.st-key-queue_shell {padding:1.05rem 1rem .88rem; margin-bottom:1rem;}
.st-key-method_shell {padding:1rem;}
.st-key-chart_shell {padding:1.05rem 1.1rem 1rem; margin-top:.86rem;}
.st-key-selected_shell, .st-key-explanation_shell, .st-key-audit_shell, .st-key-detection_attribution_shell {padding:1.05rem 1.08rem; margin-bottom:1rem;}
.st-key-sandbox_shell {padding:1rem; position:sticky; top:14px;}

/* Header */
.page-title {
  font-size:27px; line-height:1.06; letter-spacing:-.045em; font-weight:800; text-align:center;
  background:linear-gradient(110deg,#111827 5%,#3439B8 45%,#5B5FEF 72%,#0EA5E9 110%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.page-subtitle {font-size:11.5px;color:#7A8699;text-align:center;margin-top:5px;letter-spacing:.015em;}
.workspace-pill {display:inline-flex;align-items:center;gap:8px;border:1px solid rgba(91,95,239,.18);background:linear-gradient(135deg,#F5F3FF,#EEF6FF);color:#494FB9;border-radius:999px;padding:8px 11px;font-size:11.5px;font-weight:700;box-shadow:inset 0 1px 0 rgba(255,255,255,.9);white-space:nowrap;}
.workspace-dot {width:8px;height:8px;border-radius:50%;background:linear-gradient(135deg,#5B5FEF,#0EA5E9);box-shadow:0 0 0 4px rgba(91,95,239,.10);}
.header-spacer {min-height:2.45rem;}

/* Typography */
.panel-title {font-size:15.5px;font-weight:780;letter-spacing:-.025em;color:#111827;}
.panel-copy {font-size:12.2px;color:var(--muted);line-height:1.55;margin-top:5px;}
.queue-head {display:flex;justify-content:space-between;align-items:center;margin-bottom:13px;}
.counter {border-radius:999px;border:1px solid rgba(244,63,94,.22);color:#E11D48;background:linear-gradient(135deg,#FFF1F2,#FFF7F8);padding:6px 10px;font-size:10.8px;font-weight:800;white-space:nowrap;box-shadow:inset 0 1px 0 #fff;}

/* Buttons and controls */
.stButton > button, .stDownloadButton > button {
  border-radius:13px;border:1px solid rgba(148,163,184,.26);min-height:2.55rem;font-size:12.8px;font-weight:680;
  color:#243047;background:rgba(255,255,255,.90);box-shadow:0 3px 10px rgba(15,23,42,.055);
  transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease, background .16s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {transform:translateY(-1px);border-color:rgba(91,95,239,.40);color:#3439B8;background:#fff;box-shadow:0 8px 18px rgba(91,95,239,.12);}
.stButton > button:active, .stDownloadButton > button:active {transform:translateY(0);}
.stButton > button[kind="primary"] {
  background:linear-gradient(135deg,#222A49 0%,#151B31 48%,#101625 100%);border-color:#151B31;color:#fff;
  box-shadow:0 10px 24px rgba(15,23,42,.18),inset 0 1px 0 rgba(255,255,255,.10);
}
.stButton > button[kind="primary"]:hover {background:linear-gradient(135deg,#343B63,#171D34);color:#fff;border-color:#343B63;box-shadow:0 13px 28px rgba(15,23,42,.24);}
.st-key-header_download .stDownloadButton > button {
  border:0;color:#fff;background:linear-gradient(135deg,#5B5FEF,#7C3AED);padding-left:17px;padding-right:17px;
  box-shadow:0 10px 24px rgba(91,95,239,.24);
}
.st-key-header_download .stDownloadButton > button:hover {color:#fff;background:linear-gradient(135deg,#4F46E5,#6D28D9);box-shadow:0 13px 28px rgba(91,95,239,.30);}
div[data-testid="stTextInput"] {margin:0 0 .75rem;}
div[data-testid="stTextInput"] input {border-radius:13px;border:1px solid rgba(148,163,184,.24);min-height:2.65rem;background:rgba(248,250,252,.84);font-size:12.8px;box-shadow:inset 0 1px 2px rgba(15,23,42,.025);}
div[data-testid="stTextInput"] input:focus {border-color:#818CF8;box-shadow:0 0 0 3px rgba(99,102,241,.10);}
[data-testid="stSegmentedControl"] {margin:.02rem 0 0;background:rgba(255,255,255,.76);padding:4px;border:1px solid rgba(148,163,184,.23);border-radius:15px;box-shadow:var(--shadow-sm);}
[data-testid="stSegmentedControl"] button {border:0 !important;border-radius:11px !important;min-height:2.48rem;padding:.44rem .95rem !important;font-weight:680;font-size:12.5px;transition:all .15s ease;}
[data-testid="stSegmentedControl"] button[aria-checked="true"] {background:linear-gradient(135deg,#EEF0FF,#E8F3FF)!important;color:#4146C6!important;box-shadow:0 4px 12px rgba(91,95,239,.14)!important;}
.st-key-toolbar_shell {background:rgba(255,255,255,.64);border:1px solid rgba(255,255,255,.72);padding:7px;border-radius:18px;box-shadow:var(--shadow-sm);backdrop-filter:blur(14px);}
.st-key-toolbar_shell [data-testid="stHorizontalBlock"] {gap:.55rem;}
.st-key-toolbar_shell .stButton > button {min-height:2.56rem;}

/* Queue */
.st-key-queue_scroll {padding-right:4px;}
[class*="st-key-case_"] {margin-bottom:9px;position:relative;}
[class*="st-key-case_"] .stButton > button {
  width:100%;justify-content:flex-start;text-align:left;height:auto;min-height:5.35rem;border-radius:16px;padding:11px 12px 11px 15px;
  border:1px solid rgba(148,163,184,.22);box-shadow:0 3px 11px rgba(15,23,42,.045);background:rgba(255,255,255,.72);position:relative;overflow:hidden;
}
[class*="st-key-case_"] .stButton > button:before {content:"";position:absolute;left:0;top:12px;bottom:12px;width:3px;border-radius:0 4px 4px 0;background:#CBD5E1;}
[class*="st-key-case_high_"] .stButton > button:before {background:linear-gradient(#FB7185,#E11D48);}
[class*="st-key-case_medium_"] .stButton > button:before {background:linear-gradient(#FBBF24,#F59E0B);}
[class*="st-key-case_low_"] .stButton > button:before {background:linear-gradient(#34D399,#10B981);}
[class*="st-key-case_"] .stButton > button:hover {transform:translateX(3px);border-color:rgba(91,95,239,.32);box-shadow:0 10px 24px rgba(54,65,100,.10);}
[class*="st-key-case_"] .stButton > button p {white-space:pre-line;line-height:1.46;margin:0;font-size:11.8px;color:#65758B;}
[class*="st-key-case_"] .stButton > button p strong {font-size:13.1px;color:#172033;font-weight:760;}
[class*="st-key-case_"] .stButton > button[kind="primary"] {background:linear-gradient(135deg,#202945,#101727);border-color:#1B243D;box-shadow:0 13px 30px rgba(15,23,42,.22);}
[class*="st-key-case_"] .stButton > button[kind="primary"]:before {background:linear-gradient(#8B5CF6,#0EA5E9);width:4px;top:0;bottom:0;}
[class*="st-key-case_"] .stButton > button[kind="primary"] p,[class*="st-key-case_"] .stButton > button[kind="primary"] p strong {color:#fff;}
.reviewed-launch .stButton > button {font-size:12px;min-height:2.25rem;}

/* Reference / method panel */
.method-box {background:linear-gradient(145deg,#F7F8FF,#F3F7FF 55%,#F7FBFF);border:1px solid rgba(91,95,239,.14);border-radius:16px;padding:13px;margin-top:12px;box-shadow:inset 0 1px 0 rgba(255,255,255,.9);}
.method-title {font-size:13px;font-weight:740;color:#242C4E;margin-bottom:5px;}
.chips {display:flex;flex-wrap:wrap;gap:6px;margin-top:11px;}
.chip {padding:5px 9px;border-radius:999px;border:1px solid rgba(91,95,239,.13);background:rgba(255,255,255,.74);color:#53617A;font-size:10.9px;font-weight:680;box-shadow:0 2px 6px rgba(15,23,42,.035);}

/* Chart */
.chart-head {display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:5px;}
.chart-title {font-size:17px;font-weight:790;letter-spacing:-.025em;color:#121A2B;}
.chart-subtitle {font-size:12px;color:#6C7A91;margin-top:4px;line-height:1.45;}
.ai-pill,.sandbox-pill {display:inline-flex;align-items:center;border-radius:999px;padding:7px 11px;font-size:10.9px;font-weight:760;white-space:nowrap;box-shadow:inset 0 1px 0 rgba(255,255,255,.8);}
.ai-pill {border:1px solid rgba(245,158,11,.25);color:#B45309;background:linear-gradient(135deg,#FFFBEB,#FFF7D6);}
.sandbox-pill {border:1px solid rgba(91,95,239,.20);color:#4F46E5;background:linear-gradient(135deg,#EEF2FF,#EFF8FF);}
[data-testid="stPlotlyChart"] {border-radius:17px;overflow:hidden;margin-top:3px;border:1px solid rgba(148,163,184,.14);box-shadow:inset 0 1px 0 rgba(255,255,255,.9);}
.legend-row {display:flex;gap:7px;flex-wrap:wrap;padding-top:4px;}
.legend {display:inline-flex;align-items:center;gap:6px;border:1px solid rgba(148,163,184,.18);background:rgba(248,250,252,.88);color:#5C6B80;border-radius:999px;padding:6px 10px;font-size:10.8px;font-weight:680;}
.legend.raw:before {content:"";width:16px;height:6px;border-radius:6px;background:#172033;}
.legend.smooth:before {content:"";width:16px;height:6px;border-radius:6px;background:linear-gradient(90deg,#5B5FEF,#0EA5E9);}
.legend.base:before {content:"";width:16px;height:0;border-top:2px dotted #A6B2C4;}
.legend.band:before {content:"";width:16px;height:8px;border-radius:6px;background:linear-gradient(90deg,#93C5FD,#60A5FA);}
.legend.flag {border-color:rgba(244,63,94,.18);background:#FFF4F6;color:#D92D52;}.legend.flag:before {content:"";width:8px;height:8px;border-radius:50%;background:#F43F5E;}
.legend.stable {border-color:rgba(16,185,129,.18);background:#EFFCF7;color:#087E61;}.legend.stable:before {content:"";width:8px;height:8px;border-radius:50%;background:#10B981;}

/* Feature cards */
.feature-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin-top:13px;}
.feature-card {background:rgba(255,255,255,.82);border:1px solid rgba(148,163,184,.18);border-radius:18px;padding:14px;min-height:108px;box-shadow:0 5px 16px rgba(15,23,42,.045);transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease;position:relative;overflow:hidden;}
.feature-card:before {content:"";position:absolute;left:0;top:0;width:100%;height:3px;background:linear-gradient(90deg,#5B5FEF,#0EA5E9);opacity:.75;}
.feature-card:hover {transform:translateY(-2px);border-color:rgba(91,95,239,.26);box-shadow:0 14px 30px rgba(54,65,100,.10);}
.feature-title {font-size:13px;font-weight:760;color:#1B2440;margin-bottom:7px;}
.feature-copy {font-size:11.9px;line-height:1.5;color:#627189;}

/* Selected region */
.severity {float:right;border-radius:999px;padding:6px 10px;font-size:10.5px;font-weight:800;border:1px solid rgba(244,63,94,.20);color:#E11D48;background:#FFF2F4;}
.severity.mid {border-color:rgba(245,158,11,.24);color:#B45309;background:#FFF8E8;}.severity.done,.severity.low {border-color:rgba(16,185,129,.22);color:#087E61;background:#ECFDF5;}
.decision-hero {display:flex;align-items:center;justify-content:space-between;gap:12px;border-radius:17px;border:1px solid rgba(91,95,239,.13);background:linear-gradient(145deg,#F9FAFF,#F3F7FF);padding:13px;margin:13px 0 14px;}
.decision-copy {min-width:0;}.decision-title {font-size:14px;font-weight:760;color:#18213A;}.decision-subtitle {font-size:11.8px;color:#68778E;line-height:1.45;margin-top:5px;}
.stability-ring {--value:0deg;--gauge:#5B5FEF;width:66px;height:66px;min-width:66px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--gauge) var(--value),#E9EDF5 0);position:relative;box-shadow:0 6px 16px rgba(91,95,239,.14);}
.stability-ring:after {content:"";position:absolute;width:50px;height:50px;border-radius:50%;background:#fff;box-shadow:inset 0 1px 4px rgba(15,23,42,.05);}.stability-ring span {position:relative;z-index:2;font-size:12px;font-weight:820;color:#222B45;}
.focus-box {border-radius:15px;border:1px solid rgba(148,163,184,.19);background:linear-gradient(145deg,#F9FAFC,#F6F8FC);padding:12px;margin:12px 0 15px;}
.focus-title {font-size:13.5px;font-weight:740;color:#162038;}.focus-copy {font-size:12px;color:#617088;line-height:1.5;margin-top:6px;}
.workflow-note {border-radius:13px;padding:10px 11px;font-size:11.8px;line-height:1.45;margin:9px 0;}.workflow-note.done {background:#ECFDF5;border:1px solid rgba(16,185,129,.24);color:#087E61;}.workflow-note.exception {background:#FFF1F2;border:1px solid rgba(244,63,94,.22);color:#BE123C;}
.explain-box {background:linear-gradient(145deg,#FAFBFD,#F6F8FC);border:1px solid rgba(148,163,184,.15);border-radius:15px;padding:12px;margin-top:11px;transition:transform .15s ease;}.explain-box:hover {transform:translateX(2px);}
.explain-title {font-size:12.7px;font-weight:730;color:#1E293B;margin-bottom:6px;}.explain-copy {font-size:11.9px;color:#627189;line-height:1.5;}
.audit-row {font-size:11.9px;line-height:1.48;color:#627189;margin:9px 0;padding-left:11px;border-left:2px solid rgba(91,95,239,.26);}.audit-time {font-weight:780;color:#1B2440;margin-right:6px;}
.saved {background:linear-gradient(135deg,#ECFDF5,#F0FDF4);border:1px solid rgba(16,185,129,.24);color:#087E61;padding:8px 10px;font-size:11.7px;font-weight:680;border-radius:12px;margin:9px 0;}
.note-caption {font-size:11px;color:#7A879A;line-height:1.47;margin-top:9px;}
.source-box {background:linear-gradient(145deg,#F9FAFC,#F5F7FB);border:1px solid rgba(148,163,184,.16);border-radius:14px;padding:11px 12px;margin:12px 0;}
.source-title {font-size:10.3px;text-transform:uppercase;letter-spacing:.08em;color:#7B879A;font-weight:800;margin-bottom:6px;}.source-copy {font-size:11.6px;color:#53627A;line-height:1.56;}.source-copy strong {color:#1D2842;}
.evidence-list {border:1px solid rgba(148,163,184,.16);border-radius:14px;background:rgba(255,255,255,.85);overflow:hidden;margin-top:12px;box-shadow:inset 0 1px 0 rgba(255,255,255,.9);}.evidence-row {display:flex;justify-content:space-between;align-items:center;gap:12px;padding:9px 11px;border-bottom:1px solid rgba(226,232,240,.80);font-size:11.8px;color:#66758C;}.evidence-row:last-child {border-bottom:none;}.evidence-row strong {color:#1B2440;font-weight:730;text-align:right;}

/* Sandbox */
.sandbox-group {border:1px solid rgba(148,163,184,.17);border-radius:16px;background:linear-gradient(145deg,#FBFCFE,#F6F8FC);padding:12px;margin:9px 0 12px;}.sandbox-group-title {font-size:13px;font-weight:760;color:#17213A;margin-bottom:4px;}.sandbox-hint {font-size:11.6px;color:#738096;line-height:1.45;margin-bottom:7px;}
.ml-status {border:1px solid rgba(14,165,233,.24);background:linear-gradient(135deg,#EFF8FF,#EEF6FF);color:#0369A1;border-radius:13px;padding:9px 10px;font-size:11.8px;line-height:1.46;margin-top:9px;}.ml-status.warn {border-color:rgba(245,158,11,.24);background:#FFF9EB;color:#92400E;}
.st-key-sandbox_controls_scroll {padding-right:7px;}
.st-key-sandbox_chart_note {background:linear-gradient(135deg,#EEF2FF,#EFF8FF);border:1px solid rgba(91,95,239,.18);color:#3F46B7;border-radius:14px;padding:10px 12px;font-size:11.7px;line-height:1.47;margin-bottom:10px;box-shadow:0 5px 14px rgba(91,95,239,.08);}
[data-testid="stExpander"] {border:1px solid rgba(148,163,184,.17)!important;border-radius:15px!important;background:rgba(255,255,255,.64)!important;overflow:hidden;margin-bottom:10px;box-shadow:0 3px 10px rgba(15,23,42,.035);}
[data-testid="stExpander"] summary {font-weight:710;color:#27324A;}
[data-testid="stSlider"] [role="slider"] {background:#5B5FEF!important;box-shadow:0 0 0 4px rgba(91,95,239,.12)!important;}

/* Action accents */
.st-key-accept_review_action .stButton > button {background:linear-gradient(135deg,#5B5FEF,#7C3AED);border:0;color:#fff;box-shadow:0 12px 26px rgba(91,95,239,.25);}
.st-key-accept_review_action .stButton > button:hover {color:#fff;background:linear-gradient(135deg,#4F46E5,#6D28D9);box-shadow:0 15px 30px rgba(91,95,239,.31);}
.st-key-flag_exception_action .stButton > button {border-color:rgba(244,63,94,.20);color:#D92D52;background:#FFF7F8;}
.st-key-flag_exception_action .stButton > button:hover {border-color:rgba(244,63,94,.35);color:#BE123C;background:#FFF1F2;}

@media (max-width:1250px) {.feature-grid{grid-template-columns:1fr}.page-title{font-size:23px}.workspace-pill{display:none}.st-key-sandbox_shell{position:relative;top:auto;}}
@media (max-width:980px) {.block-container{padding:.75rem!important}.st-key-header_shell{border-radius:17px}.page-title{font-size:21px}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def resolved_data_paths() -> tuple[Optional[Path], Optional[Path]]:
    h5_override = Path(str(st.session_state.get("h5_override", "")).strip()).expanduser()
    labels_override = Path(str(st.session_state.get("labels_override", "")).strip()).expanduser()
    h5 = h5_override if str(h5_override) not in {"", "."} and h5_override.exists() else resolve_existing_file("synthetic_chromatograms.h5", extra_candidates=[DATA_DIR])
    supplied = labels_override if str(labels_override) not in {"", "."} and labels_override.exists() else resolve_existing_file("synthetic_reference.xlsx", extra_candidates=[DATA_DIR])
    if supplied is None:
        return h5, None
    return h5, ensure_working_label_workbook(supplied, DATA_DIR)


@st.cache_data(show_spinner=False)
def load_dataset_state(h5_path: str, labels_path: str) -> tuple[list[int], pd.DataFrame]:
    return list_h5_channels(h5_path), load_label_table(labels_path)


@st.cache_data(show_spinner=False)
def run_cached(h5_path: str, labels_path: str, channel_id: int, config_dict: str) -> PipelineResult:
    parsed = json.loads(config_dict)
    cfg = PipelineConfig(
        preprocessing=PreprocessingConfig(**parsed["preprocessing"]),
        segmentation=SegmentationConfig(**parsed["segmentation"]),
        detection=DetectionConfig(**parsed["detection"]),
        scoring=ScoringConfig(**parsed["scoring"]),
        stability=StabilityConfig(**parsed["stability"]),
    )
    t, y = load_chromatogram_h5(h5_path, channel_id)
    labels = labels_for_channel(load_label_table(labels_path), channel_id)
    return run_arrays(t, y, labels, channel_id=channel_id, config=cfg)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "workflow_view": "AI Review",
        "selected_case_key": None,
        "viewport_mode": "overview",  # screenshot-compatible initial view; selecting a case switches to focus.
        "plot_revision": 0,
        "show_raw": True,
        "show_band": True,
        "show_labels": False,
        "show_official_peaks": True,
        "compare_runs": False,
        "show_explain": False,
        "magnify_y_axis": False,
        "last_saved": "",
        "channel_id": None,
        "data_source": "Synthetic demo dataset",
        "quick_channel": "Manual",
        "h5_override": "",
        "labels_override": "",
        "sb_smooth": 2,
        "sb_noise_window": 51,
        "sb_uncertainty_sigma": 2.0,
        "sb_seg_mode": "fixed",
        "sb_window_points": 500,
        "sb_overlap_points": 100,
        "sb_nbands": 3,
        "sb_min_band_run": 80,
        "sb_distance": 22,
        "sb_prom_k": 6.0,
        "sb_prom_floor": 0.00005,
        "sb_global_prominence": 0.00018,
        "sb_dedup_tolerance": 8,
        "sb_weber_k": 15.96,
        "sb_runs": 0,
        "sb_perturbation_scale": 1.0,
        "sb_ml_enable": False,
        "sb_ml_model_path": "models/synthetic_peak_classifier.joblib",
        "sb_ml_threshold": 0.50,
        "sb_ml_display": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.get("workflow_view") == "Official SOP":
        st.session_state.workflow_view = LABELED_DATA_VIEW


def config_for_sandbox() -> PipelineConfig:
    return PipelineConfig(
        preprocessing=PreprocessingConfig(
            smooth_window=int(st.session_state.get("sb_smooth", 2)),
            noise_window=int(st.session_state.get("sb_noise_window", 51)),
            uncertainty_sigma=float(st.session_state.get("sb_uncertainty_sigma", 2.0)),
        ),
        segmentation=SegmentationConfig(
            mode=str(st.session_state.get("sb_seg_mode", "fixed")),
            window_points=int(st.session_state.get("sb_window_points", 500)),
            overlap_points=int(st.session_state.get("sb_overlap_points", 100)),
            nbands=int(st.session_state.get("sb_nbands", 3)),
            min_band_run=int(st.session_state.get("sb_min_band_run", 80)),
        ),
        detection=DetectionConfig(
            distance=int(st.session_state.get("sb_distance", 22)),
            prominence_floor=float(st.session_state.get("sb_prom_floor", 0.00005)),
            prominence_k=float(st.session_state.get("sb_prom_k", 6.0)),
            global_prominence=float(st.session_state.get("sb_global_prominence", 0.00018)),
            dedup_tolerance=int(st.session_state.get("sb_dedup_tolerance", 8)),
        ),
        scoring=ScoringConfig(weber_k=float(st.session_state.get("sb_weber_k", 15.96))),
        stability=StabilityConfig(
            n_runs=int(st.session_state.get("sb_runs", 0)),
            perturbation_scale=float(st.session_state.get("sb_perturbation_scale", 1.0)),
            match_tolerance_samples=OFFICIAL_CONFIG.stability.match_tolerance_samples,
            seed=OFFICIAL_CONFIG.stability.seed,
        ),
    )




def resolve_model_path(model_path: str) -> Path | None:
    raw = str(model_path or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = APP_ROOT / candidate
    return candidate if candidate.exists() else None


def ml_preview_predictions(result: PipelineResult, model_path: str, threshold: float) -> tuple[pd.DataFrame, str]:
    """Return sandbox-only ML probabilities for current candidate peaks.

    The synthetic reference workbook and official detector output are not modified. When the model
    cannot be loaded, the UI reports a warning instead of silently fabricating
    probabilities.
    """
    if result.final_peaks.empty:
        return pd.DataFrame(), "No candidate peaks are available for ML scoring under the current sandbox settings."
    resolved = resolve_model_path(model_path)
    if resolved is None:
        return pd.DataFrame(), f"Model file not found: {model_path}"
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        import joblib  # type: ignore
        artifact = joblib.load(resolved)
        model = artifact.get("model", artifact) if isinstance(artifact, dict) else artifact
        feature_columns = artifact.get("feature_columns", []) if isinstance(artifact, dict) else list(getattr(model, "feature_names_in_", []))
        if not feature_columns:
            return pd.DataFrame(), "Model artifact does not expose feature columns."
        peaks = result.final_peaks.copy().sort_values("sample_idx").reset_index(drop=True)
        n = len(peaks)
        peaks["abs_weber_margin"] = pd.to_numeric(peaks.get("weber_margin"), errors="coerce").abs()
        peaks["stability_filled"] = pd.to_numeric(peaks.get("stability"), errors="coerce").fillna(0.0)
        time_span = max(float(result.time.max() - result.time.min()), 1e-12)
        peaks["relative_time"] = (pd.to_numeric(peaks["time"], errors="coerce") - float(result.time.min())) / time_span
        sample_idx = pd.to_numeric(peaks["sample_idx"], errors="coerce").to_numpy(dtype=float)
        peaks["distance_to_prev"] = np.r_[np.inf, np.diff(sample_idx)]
        peaks["distance_to_next"] = np.r_[np.diff(sample_idx), np.inf]
        peaks[["distance_to_prev", "distance_to_next"]] = peaks[["distance_to_prev", "distance_to_next"]].replace([np.inf, -np.inf], len(result.time))
        prom = np.maximum(pd.to_numeric(peaks.get("prominence"), errors="coerce").to_numpy(dtype=float), 1e-12)
        height = np.maximum(np.abs(pd.to_numeric(peaks.get("height"), errors="coerce").to_numpy(dtype=float)), 1e-12)
        prom_prev = np.r_[prom[0], prom[:-1]]; prom_next = np.r_[prom[1:], prom[-1]]
        height_prev = np.r_[height[0], height[:-1]]; height_next = np.r_[height[1:], height[-1]]
        peaks["prominence_ratio_to_prev"] = prom / np.maximum(prom_prev, 1e-12)
        peaks["prominence_ratio_to_next"] = prom / np.maximum(prom_next, 1e-12)
        peaks["height_ratio_to_prev"] = height / np.maximum(height_prev, 1e-12)
        peaks["height_ratio_to_next"] = height / np.maximum(height_next, 1e-12)
        if "segment_id" in peaks:
            density = peaks.groupby("segment_id")["peak_id"].transform("count")
            peaks["candidate_density_segment"] = density
        else:
            peaks["candidate_density_segment"] = n
        seg_start = pd.to_numeric(peaks.get("segment_time_start"), errors="coerce").fillna(float(result.time.min()))
        seg_end = pd.to_numeric(peaks.get("segment_time_end"), errors="coerce").fillna(float(result.time.max()))
        seg_span = np.maximum((seg_end - seg_start).to_numpy(dtype=float), 1e-12)
        peaks["relative_position_in_segment"] = (pd.to_numeric(peaks["time"], errors="coerce").to_numpy(dtype=float) - seg_start.to_numpy(dtype=float)) / seg_span
        for col in feature_columns:
            if col not in peaks:
                peaks[col] = 0.0
        X = peaks[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X)[:, 1]
        else:
            score = np.asarray(model.decision_function(X), dtype=float)
            probability = 1.0 / (1.0 + np.exp(-score))
        peaks["p_ml_real"] = probability
        peaks["ml_keep"] = peaks["p_ml_real"] >= float(threshold)
        out = peaks[["peak_id", "time", "prominence", "weber_score", "weber_margin", "stability", "p_ml_real", "ml_keep"]].copy()
        kept = int(out["ml_keep"].sum())
        return out, f"ML preview loaded from {resolved.name}: {kept}/{len(out)} candidates pass p ≥ {float(threshold):.2f}."
    except Exception as exc:  # pragma: no cover - defensive UI path
        return pd.DataFrame(), f"ML preview unavailable: {exc}"

def focus_case(case_key: str) -> None:
    """Select a data-derived review case and force the Plotly viewport to its interval."""
    st.session_state.selected_case_key = str(case_key)
    st.session_state.viewport_mode = "focus"
    # A review cue is an AI-review item; selecting it always reveals the review overlays.
    st.session_state.workflow_view = "AI Review"
    st.session_state.plot_revision += 1


def show_overview() -> None:
    st.session_state.viewport_mode = "overview"
    st.session_state.plot_revision += 1


def toggle_comparison() -> None:
    st.session_state.compare_runs = not bool(st.session_state.compare_runs)
    st.session_state.plot_revision += 1


def record_action(labels_path: Path, row: pd.Series, *, state: str, note: str, action: str, detail: str) -> None:
    persist_review_record(labels_path, row, state=state, note=note, action=action, detail=detail)
    st.session_state.last_saved = f"{row['region_id']} · {action} saved"
    st.session_state.selected_case_key = str(row["case_key"])
    st.session_state.plot_revision += 1


def case_from_chart_event(event: Any) -> str | None:
    try:
        points = event.selection.points
    except AttributeError:
        try:
            points = event.get("selection", {}).get("points", [])
        except AttributeError:
            points = []
    if not points:
        return None
    point = points[0]
    custom = point.get("customdata") if isinstance(point, dict) else None
    if isinstance(custom, (list, tuple)):
        custom = custom[0] if custom else None
    return str(custom) if custom else None


def queue_button_label(item: pd.Series) -> str:
    state = str(item.get("workflow_state", "open"))
    icon = "✓" if state in RESOLVED_STATES else "!" if state == "exception" else "•"
    source = "synthetic reference" if str(item.get("source_type", "")).startswith("reference") else "synthetic candidate"
    return (
        f"**{item['region_id']}**                                      {item['time_start']:.2f}–{item['time_end']:.2f} min\n"
        f"{item['title']}\n"
        f"{icon}  {item['review_status']}  ·  {source}"
    )


def render_sandbox_controls(
    *,
    channels: list[int],
    labels_all: pd.DataFrame,
    channel_id: int,
    h5_path: Path,
    labels_path: Path,
) -> None:
    """Render categorized Sandbox controls in a scrollable side panel."""
    with st.container(key="sandbox_shell"):
        st.markdown(
            '<div class="panel-title">Sandbox controls</div>'
            '<div class="panel-copy">Change a parameter and inspect the chart beside it. '
            'Sandbox settings never overwrite the labeled-data baseline or audit record.</div>',
            unsafe_allow_html=True,
        )
        with st.container(height=690, border=False, key="sandbox_controls_scroll"):
            with st.expander("Data", expanded=False):
                st.selectbox("Source", ["Synthetic demo dataset"], key="data_source")
                quick_options = ["Manual", "Channel 1001", "First available", "Most references"]
                quick_choice = st.selectbox("Quick channels", quick_options, key="quick_channel")
                proposed_channel = channel_id
                if quick_choice == "Channel 1001" and 1001 in channels:
                    proposed_channel = 1001
                elif quick_choice == "First available":
                    proposed_channel = int(channels[0])
                elif quick_choice == "Most references":
                    counts_by_channel = labels_all.groupby("ChannelId").size().sort_values(ascending=False)
                    usable = [int(ch) for ch in counts_by_channel.index if int(ch) in channels]
                    proposed_channel = usable[0] if usable else channel_id
                selected_sandbox_channel = st.selectbox(
                    "ChannelId",
                    channels,
                    index=channels.index(proposed_channel),
                    key="sandbox_channel_select",
                )
                if int(selected_sandbox_channel) != channel_id:
                    st.session_state.channel_id = int(selected_sandbox_channel)
                    st.session_state.selected_case_key = None
                    show_overview()
                    st.rerun()
                with st.expander("Optional path overrides", expanded=False):
                    st.text_input("Synthetic HDF5 path override", key="h5_override", placeholder=str(h5_path))
                    st.text_input("Synthetic reference path override", key="labels_override", placeholder=str(labels_path))
                    st.caption("Leave blank to use the bundled synthetic data files.")

            with st.expander("Preprocessing", expanded=True):
                st.slider("Smoothing window", 1, 51, key="sb_smooth", step=1)
                st.slider("Local noise window", 3, 151, key="sb_noise_window", step=2)
                st.slider(
                    "Uncertainty band σ multiplier",
                    0.50,
                    5.00,
                    key="sb_uncertainty_sigma",
                    step=0.10,
                    format="%.2f",
                )

            with st.expander("Segment-dependent detector", expanded=True):
                st.selectbox("Segmentation mode", ["fixed", "intensity"], key="sb_seg_mode")
                st.slider("Segment window points", 100, 2000, key="sb_window_points", step=50)
                st.slider("Segment overlap points", 0, 500, key="sb_overlap_points", step=25)
                st.slider("Intensity bands", 1, 8, key="sb_nbands", step=1)
                st.slider("Minimum band run", 10, 500, key="sb_min_band_run", step=10)
                st.slider("Minimum peak distance", 1, 100, key="sb_distance", step=1)
                st.slider(
                    "Segment prominence k × local noise",
                    0.50,
                    20.00,
                    key="sb_prom_k",
                    step=0.25,
                    format="%.2f",
                )
                st.number_input(
                    "Prominence floor",
                    min_value=0.0,
                    max_value=0.01,
                    key="sb_prom_floor",
                    step=0.00001,
                    format="%.6f",
                )
                st.number_input(
                    "Baseline global prominence",
                    min_value=0.0,
                    max_value=0.01,
                    key="sb_global_prominence",
                    step=0.00001,
                    format="%.6f",
                )
                st.slider("Dedup tolerance samples", 0, 100, key="sb_dedup_tolerance", step=1)

            with st.expander("Weber / stability", expanded=True):
                st.slider("Weber threshold k*", 0.50, 60.00, key="sb_weber_k", step=0.10, format="%.2f")
                st.slider("Monte Carlo stability runs", 0, 100, key="sb_runs", step=5)
                st.slider("Perturbation scale", 0.10, 3.00, key="sb_perturbation_scale", step=0.10, format="%.2f")

            with st.expander("ML-enhanced filtering", expanded=False):
                st.checkbox("Enable ML classifier", key="sb_ml_enable")
                st.checkbox("Display ML-filtered peaks", key="sb_ml_display")
                st.text_input("ML model path", key="sb_ml_model_path")
                st.slider("ML probability threshold", 0.00, 1.00, key="sb_ml_threshold", step=0.01, format="%.2f")


def render_chart_panel(
    *,
    view: str,
    preview_result: PipelineResult,
    official_result: PipelineResult,
    queue_all: pd.DataFrame,
    selected: pd.Series,
    channel_id: int,
    compare_signal: np.ndarray | None,
    ml_preview_table: pd.DataFrame,
    ml_preview_message: str,
) -> None:
    """Render the chromatogram panel for any workflow view."""
    mode_pill = (
        '<span class="sandbox-pill">Sandbox preview · not saved</span>' if view == "Sandbox"
        else '<span class="ai-pill">✧ AI suggestions visible</span>' if view == "AI Review"
        else '<span class="ai-pill">Labeled regions + detector output</span>'
    )
    with st.container(key="chart_shell"):
        st.markdown(
            f'<div class="chart-head"><div><div class="chart-title">Chromatogram inspection</div>'
            f'<div class="chart-subtitle">Raw and processed traces from synthetic_chromatograms.h5; '
            f'labeled regions from synthetic_reference.xlsx. Zoom preserves the source trace.</div></div>{mode_pill}</div>',
            unsafe_allow_html=True,
        )
        focused = st.session_state.viewport_mode == "focus"
        fig = chromatogram_figure(
            preview_result,
            queue_all,
            selected,
            show_raw=bool(st.session_state.show_raw),
            show_band=bool(st.session_state.show_band and view in {"AI Review", "Sandbox", "Audit Log"}),
            show_labels=bool(st.session_state.show_labels or view in {LABELED_DATA_VIEW, "Audit Log"}),
            show_official_peaks=bool(view in {LABELED_DATA_VIEW, "Audit Log"} and st.session_state.show_official_peaks),
            compare_signal=compare_signal,
            ml_peaks=(
                ml_preview_table
                if view == "Sandbox" and bool(st.session_state.sb_ml_enable) and bool(st.session_state.sb_ml_display)
                else None
            ),
            zoom_selected=focused,
            show_review_regions=view != LABELED_DATA_VIEW,
            use_ai_status_style=view != LABELED_DATA_VIEW,
            magnify_y_axis=bool(st.session_state.magnify_y_axis and focused),
        )
        chart_key = (
            f"chart_{channel_id}_{view}_{st.session_state.viewport_mode}_"
            f"{st.session_state.plot_revision}_{selected['case_key']}"
        )
        event = st.plotly_chart(
            fig,
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": True},
            key=chart_key,
            on_select="rerun",
            selection_mode="points",
        )
        clicked_key = case_from_chart_event(event)
        if clicked_key and clicked_key in queue_all["case_key"].tolist() and clicked_key != st.session_state.selected_case_key:
            focus_case(clicked_key)
            st.rerun()

        if view == "Sandbox" and bool(st.session_state.sb_ml_enable):
            status_class = "ml-status" if not ml_preview_table.empty else "ml-status warn"
            st.markdown(f'<div class="{status_class}">{esc(ml_preview_message)}</div>', unsafe_allow_html=True)
            if not ml_preview_table.empty:
                st.dataframe(ml_preview_table, hide_index=True, width="stretch")

        if view == LABELED_DATA_VIEW:
            st.markdown(
                '<div class="legend-row"><span class="legend raw">Raw signal</span><span class="legend smooth">Smoothed</span>'
                '<span class="legend base">Baseline</span><span class="legend stable">Detected apex</span>'
                '<span class="legend band">Labeled interval</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Labeled-data view: {len(official_result.final_peaks)} detected apexes and "
                f"{len(official_result.label_table)} synthetic labeled intervals. Labels are comparison evidence, not forced detections."
            )
        else:
            st.markdown(
                '<div class="legend-row"><span class="legend raw">Raw signal</span><span class="legend smooth">Smoothed</span>'
                '<span class="legend base">Baseline</span><span class="legend band">Uncertainty band</span>'
                '<span class="legend flag">High-priority flag</span><span class="legend stable">Stable peak</span></div>',
                unsafe_allow_html=True,
            )


init_state()
h5_path, labels_path = resolved_data_paths()
if not h5_path or not labels_path:
    st.error("Required data were not found. Place `synthetic_chromatograms.h5` and `synthetic_reference.xlsx` in the project `data/` directory.")
    st.stop()

channels, labels_all = load_dataset_state(str(h5_path), str(labels_path))
default_channel = 1001 if 1001 in channels else channels[0]
if st.session_state.channel_id not in channels:
    st.session_state.channel_id = default_channel
channel_id = int(st.session_state.channel_id)

official_result = run_cached(str(h5_path), str(labels_path), channel_id, json.dumps(OFFICIAL_CONFIG.to_dict(), sort_keys=True))
base_queue = build_review_queue(official_result, max_items=10)
persisted_records = load_review_records(labels_path, channel_id)
queue_all = apply_review_state(base_queue, persisted_records)
if queue_all.empty:
    st.warning("No reviewable regions were detected for this channel under the baseline detector.")
    st.stop()

if st.session_state.selected_case_key not in queue_all["case_key"].tolist():
    st.session_state.selected_case_key = str(queue_all.iloc[0]["case_key"])
selected = queue_all.loc[queue_all["case_key"] == st.session_state.selected_case_key].iloc[0]
counts = review_queue_counts(queue_all)
persisted_audit = load_persisted_audit(labels_path, channel_id)
report_blob = report_json(official_result, queue_all, persisted_audit, persisted_records, channel_id, str(labels_path), str(h5_path))

# Header — modern product shell; functionality remains unchanged.
with st.container(key="header_shell"):
    h_left, h_title, h_report = st.columns([0.23, 0.54, 0.23], vertical_alignment="center")
    with h_left:
        st.markdown(
            '<div class="workspace-pill"><span class="workspace-dot"></span>Synthetic review workspace</div>',
            unsafe_allow_html=True,
        )
    with h_title:
        st.markdown(
            '<div class="page-title">Peak-a-boo Review Workbench</div>'
            '<div class="page-subtitle">Inspect evidence · resolve uncertainty · preserve provenance</div>',
            unsafe_allow_html=True,
        )
    with h_report:
        with st.container(key="header_download"):
            st.download_button(
                "▣  Generate report",
                report_blob.encode("utf-8"),
                file_name=f"peak_a_boo_channel_{channel_id}_review_report.json",
                mime="application/json",
                width="stretch",
            )

left, center, right = st.columns([0.205, 0.54, 0.255], gap="medium")

with left:
    with st.container(key="queue_shell"):
        st.markdown(
            f'<div class="queue-head"><div class="panel-title">Review queue</div>'
            f'<span class="counter">{counts["active"]} cues</span></div>',
            unsafe_allow_html=True,
        )
        search = st.text_input("Search", placeholder="Search regions, issues…", label_visibility="collapsed")
        active_queue = filter_review_queue(queue_all, "Active")
        if search.strip():
            needle = search.strip().lower()
            active_queue = active_queue[active_queue.astype(str).apply(lambda row: row.str.lower().str.contains(needle).any(), axis=1)]
        with st.container(height=386, border=False, key="queue_scroll"):
            if active_queue.empty:
                st.info("No active review cases.")
            for index, item in active_queue.reset_index(drop=True).iterrows():
                current = str(item["case_key"]) == str(st.session_state.selected_case_key)
                severity_key = str(item.get("severity", "medium")).lower().replace(" ", "_")
                with st.container(key=f"case_{severity_key}_{index}"):
                    st.button(
                        queue_button_label(item),
                        key=f"select_{item['case_key']}",
                        type="primary" if current else "secondary",
                        width="stretch",
                        on_click=focus_case,
                        args=(str(item["case_key"]),),
                    )
        if counts["reviewed"] > 0:
            with st.popover(f"Reviewed records ({counts['reviewed']})", width="stretch"):
                for index, item in filter_review_queue(queue_all, "Reviewed").reset_index(drop=True).iterrows():
                    st.button(
                        queue_button_label(item),
                        key=f"reviewed_{item['case_key']}",
                        width="stretch",
                        on_click=focus_case,
                        args=(str(item["case_key"]),),
                    )

    with st.container(key="method_shell"):
        st.markdown('<div class="panel-title">Reference data status</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="method-box"><div class="method-title">♧ &nbsp;Synthetic labeled dataset</div>'
            '<div class="panel-copy">Reference intervals are fixed for review. Detector settings remain unchanged outside the Sandbox.</div>'
            '<div class="chips"><span class="chip">Prominence: 0.00018</span><span class="chip">Smoothing: 2</span>'
            '<span class="chip">Distance: 22</span><span class="chip">Labels: synthetic</span></div></div>'
            f'<div class="source-box"><div class="source-title">Loaded analytical sources</div>'
            f'<div class="source-copy"><strong>Signal:</strong> synthetic_chromatograms.h5 / channel {channel_id} · {len(official_result.time):,} time-intensity samples<br>'
            f'<strong>References:</strong> synthetic_reference.xlsx · {len(labels_for_channel(labels_all, channel_id))} labeled intervals</div></div>',
            unsafe_allow_html=True,
        )

selected = queue_all.loc[queue_all["case_key"] == st.session_state.selected_case_key].iloc[0]
workflow_state = str(selected.get("workflow_state", "open"))
resolved = workflow_state in RESOLVED_STATES

with center:
    with st.container(key="toolbar_shell"):
        nav, zoom_col, layers_col, compare_col = st.columns([1.82, 0.67, 0.72, 1.03], vertical_alignment="center")
        with nav:
            view = st.segmented_control(
                "Workflow view",
                [LABELED_DATA_VIEW, "AI Review", "Sandbox", "Audit Log"],
                key="workflow_view",
                label_visibility="collapsed",
                width="stretch",
            )
        with zoom_col:
            st.button("⌕  Zoom", width="stretch", on_click=focus_case, args=(str(selected["case_key"]),))
        with layers_col:
            with st.popover("▱  Layers", width="stretch"):
                st.checkbox("Raw signal", key="show_raw")
                st.checkbox("Uncertainty band", key="show_band")
                st.checkbox("Detected apexes", key="show_official_peaks")
                if view in {LABELED_DATA_VIEW, "Audit Log"}:
                    st.checkbox("Synthetic labeled intervals", value=True, disabled=True, key="fixed_reference_intervals")
                    st.caption("Labeled intervals are always visible in this evidence view.")
                else:
                    st.checkbox("Synthetic labeled intervals", key="show_labels")
                st.checkbox(
                    "Magnify Y-axis in focused view",
                    key="magnify_y_axis",
                    help="Optional axis magnification only. Raw and smoothed trace samples remain unchanged.",
                )
                st.button("Show full trace", width="stretch", on_click=show_overview)
                selected_channel = st.selectbox("Channel", channels, index=channels.index(channel_id), key="channel_select_hidden")
                if int(selected_channel) != channel_id:
                    st.session_state.channel_id = int(selected_channel)
                    st.session_state.selected_case_key = None
                    show_overview()
                    st.rerun()
        with compare_col:
            compare_label = "⇄  Hide compare" if st.session_state.compare_runs else "⇄  Compare runs"
            st.button(compare_label, width="stretch", on_click=toggle_comparison)

    preview_result = official_result
    ml_preview_table = pd.DataFrame()
    ml_preview_message = ""

    if view == "Sandbox":
        controls_col, live_chart_col = st.columns([0.36, 0.64], gap="medium")
        with controls_col:
            render_sandbox_controls(
                channels=channels,
                labels_all=labels_all,
                channel_id=channel_id,
                h5_path=Path(h5_path),
                labels_path=Path(labels_path),
            )

        sandbox_cfg = config_for_sandbox()
        preview_result = run_cached(
            str(h5_path),
            str(labels_path),
            channel_id,
            json.dumps(sandbox_cfg.to_dict(), sort_keys=True),
        )
        if bool(st.session_state.sb_ml_enable):
            ml_preview_table, ml_preview_message = ml_preview_predictions(
                preview_result,
                st.session_state.sb_ml_model_path,
                float(st.session_state.sb_ml_threshold),
            )

        compare_signal = None
        if st.session_state.compare_runs:
            rng = np.random.default_rng(2026 + channel_id)
            perturbed = preview_result.raw_signal + rng.normal(
                0,
                preview_result.sigma_t,
                size=len(preview_result.raw_signal),
            )
            compare_signal = moving_average(perturbed, preview_result.config.preprocessing.smooth_window)

        with live_chart_col:
            st.markdown(
                '<div class="st-key-sandbox_chart_note"><strong>Live preview:</strong> '
                'parameter changes update this chart immediately. Nothing here changes the labeled-data baseline.</div>',
                unsafe_allow_html=True,
            )
            render_chart_panel(
                view=view,
                preview_result=preview_result,
                official_result=official_result,
                queue_all=queue_all,
                selected=selected,
                channel_id=channel_id,
                compare_signal=compare_signal,
                ml_preview_table=ml_preview_table,
                ml_preview_message=ml_preview_message,
            )
    else:
        compare_signal = None
        if st.session_state.compare_runs:
            rng = np.random.default_rng(2026 + channel_id)
            perturbed = preview_result.raw_signal + rng.normal(
                0,
                preview_result.sigma_t,
                size=len(preview_result.raw_signal),
            )
            compare_signal = moving_average(perturbed, preview_result.config.preprocessing.smooth_window)

        render_chart_panel(
            view=view,
            preview_result=preview_result,
            official_result=official_result,
            queue_all=queue_all,
            selected=selected,
            channel_id=channel_id,
            compare_signal=compare_signal,
            ml_preview_table=ml_preview_table,
            ml_preview_message=ml_preview_message,
        )

        if st.session_state.show_explain or view == "Audit Log":
            if view == "Audit Log":
                with st.container(key="detection_attribution_shell"):
                    st.markdown(
                        '<div class="panel-title">Review audit records</div>'
                        '<div class="panel-copy">Persisted analyst actions stored in the synthetic review workbook.</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        pd.DataFrame(persisted_audit)
                        if persisted_audit
                        else pd.DataFrame(columns=["time", "action", "detail"]),
                        hide_index=True,
                        width="stretch",
                    )
            else:
                with st.container(key="detection_attribution_shell"):
                    st.markdown(
                        '<div class="panel-title">Detection attribution</div>'
                        '<div class="panel-copy">Baseline detector evidence for the selected region.</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(pipeline_attribution(official_result, selected), hide_index=True, width="stretch")
        else:
            st.markdown(
                '<div class="feature-grid"><div class="feature-card"><div class="feature-title">◉ &nbsp;Explain detection</div>'
                '<div class="feature-copy">Shows which pipeline stage affected the selected region: smoothing, local noise, prominence, Weber margin, or stability.</div></div>'
                '<div class="feature-card"><div class="feature-title">☷ &nbsp;Sandbox only</div>'
                '<div class="feature-copy">Alternative parameters can be previewed beside the chart without changing the labeled-data baseline.</div></div>'
                '<div class="feature-card"><div class="feature-title">▣ &nbsp;Audit-ready output</div>'
                '<div class="feature-copy">Flags, comments, scores, and decisions are compiled into an exportable review report.</div></div></div>',
                unsafe_allow_html=True,
            )
            a, b, c = st.columns(3)
            with a:
                if st.button("Open explanation", width="stretch"):
                    st.session_state.show_explain = True
                    st.rerun()
            with b:
                if st.button("Open sandbox", width="stretch"):
                    st.session_state.workflow_view = "Sandbox"
                    st.rerun()
            with c:
                st.download_button(
                    "Download output",
                    report_blob.encode("utf-8"),
                    file_name=f"peak_a_boo_channel_{channel_id}_review_report.json",
                    mime="application/json",
                    width="stretch",
                )

with right:
    stability = float(selected.get("stability", np.nan))
    stability_pct = 0 if not np.isfinite(stability) else int(round(stability * 100))
    severity_class = "done" if resolved else "mid" if selected["severity"] == "medium" else "low" if selected["severity"] == "low" else ""
    severity_text = "Reviewed" if resolved else "Moderate" if selected["severity"] == "medium" else "Low" if selected["severity"] == "low" else "High"
    gauge_color = "#10B981" if resolved or selected["severity"] == "low" else "#F59E0B" if selected["severity"] == "medium" else "#F43F5E"
    score = float(selected.get("weber_score", np.nan))
    margin = float(selected.get("weber_margin", np.nan))
    score_text = f"{score:.1f}×" if np.isfinite(score) else "n/a"
    noise_text = "elevated" if float(selected.get("local_noise", 0)) > float(np.median(official_result.sigma_t)) else "typical"
    stability_tag = "Stability fail" if stability_pct < 50 else "Stability watch" if stability_pct < 80 else "Stability pass"
    detection_copy = "Reference interval has no surviving official-method apex." if selected["match_status"] == "FN" else f"Detected in {stability_pct}% of perturbation runs"

    with st.container(key="selected_shell"):
        st.markdown(
            f'<span class="severity {severity_class}">{severity_text}</span><div class="panel-title">Selected region</div>'
            f'<div class="panel-copy">{esc(selected["region_id"])} · {selected["time_start"]:.2f}–{selected["time_end"]:.2f} min</div>'
            f'<div class="decision-hero"><div class="decision-copy"><div class="decision-title">{esc(selected["title"])}</div>'
            f'<div class="decision-subtitle">{esc(detection_copy)}</div></div>'
            f'<div class="stability-ring" style="--value:{max(0, min(100, stability_pct)) * 3.6:.1f}deg;--gauge:{gauge_color}"><span>{stability_pct}%</span></div></div>'
            f'<div class="source-box"><div class="source-title">Data provenance</div>'
            f'<div class="source-copy"><strong>Signal:</strong> synthetic_chromatograms.h5 / channel {channel_id}<br>'
            f'<strong>Region:</strong> {esc(selected.get("interval_origin", ""))}<br>'
            f'<strong>Record:</strong> {esc(selected.get("data_origin", ""))}<br>'
            f'<strong>Apex:</strong> {float(selected["time"]):.6f} min · <strong>Match:</strong> {esc(selected["match_status"])}</div></div>'
            f'<div class="evidence-list">'
            f'<div class="evidence-row"><span>Local contrast</span><strong>{score_text}</strong></div>'
            f'<div class="evidence-row"><span>Noise level</span><strong>{noise_text.title()}</strong></div>'
            f'<div class="evidence-row"><span>Prominence</span><strong>{"No detected apex" if selected["match_status"] == "FN" else "Passed"}</strong></div>'
            f'<div class="evidence-row"><span>Stability check</span><strong>{stability_tag.replace("Stability ", "").title()}</strong></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.last_saved:
            st.markdown(f'<div class="saved">✓ {esc(st.session_state.last_saved)}</div>', unsafe_allow_html=True)
        if workflow_state == "accepted":
            st.markdown('<div class="workflow-note done"><strong>Accepted / reviewed.</strong> This case is resolved and retained in the audit record.</div>', unsafe_allow_html=True)
        elif workflow_state == "exception":
            st.markdown('<div class="workflow-note exception"><strong>Exception flagged.</strong> This case stays active until explicitly resolved.</div>', unsafe_allow_html=True)
        elif workflow_state == "exception_resolved":
            st.markdown('<div class="workflow-note done"><strong>Exception resolved.</strong> The completed escalation remains in the audit record.</div>', unsafe_allow_html=True)

        note_value = str(selected.get("analyst_note", ""))
        if not resolved:
            with st.container(key="accept_review_action"):
                if st.button("◉  Accept as reviewed", type="primary", width="stretch", key=f"accept_{selected['case_key']}"):
                    record_action(labels_path, selected, state="accepted", note=note_value, action="Accept as reviewed", detail=f"{selected['region_id']} accepted as reviewed.")
                    st.rerun()
            if workflow_state != "exception":
                with st.container(key="flag_exception_action"):
                    if st.button("⚑  Flag exception", width="stretch", key=f"flag_{selected['case_key']}"):
                        record_action(labels_path, selected, state="exception", note=note_value, action="Flag exception", detail=f"{selected['region_id']} flagged as exception.")
                        st.rerun()
            if workflow_state == "exception" and st.button("✓  Resolve exception", width="stretch", key=f"resolve_{selected['case_key']}"):
                record_action(labels_path, selected, state="exception_resolved", note=note_value, action="Resolve exception", detail=f"{selected['region_id']} exception resolved.")
                st.rerun()
        else:
            if st.button("↺  Reopen for review", width="stretch", key=f"reopen_{selected['case_key']}"):
                record_action(labels_path, selected, state="open", note=note_value, action="Reopen review", detail=f"{selected['region_id']} reopened for review.")
                st.rerun()
        with st.popover("▢  Add analyst note", width="stretch"):
            with st.form(f"note_{selected['case_key']}"):
                note = st.text_area("Review note", value=note_value, placeholder="Document the rationale for this disposition.")
                if st.form_submit_button("Save note", width="stretch"):
                    record_action(labels_path, selected, state=workflow_state, note=note, action="Save analyst note", detail=f"Analyst note saved for {selected['region_id']}.")
                    st.rerun()

    if selected["match_status"] == "FN":
        cause = "a reference interval without a surviving detected apex"
    elif stability_pct < 50:
        cause = "smoothing and local noise"
    elif np.isfinite(margin) and abs(margin) < 5:
        cause = "proximity to the Weber decision boundary"
    else:
        cause = "peak morphology and prominence"
    margin_text = f"{margin:.2f}" if np.isfinite(margin) else "not available"
    next_step = "Decision saved. Review the audit trail or export the report." if resolved else "Document as an exception candidate or accept after reviewing the focused evidence."
    with st.container(key="explanation_shell"):
        st.markdown(
            f'<div class="panel-title">Explanation panel</div>'
            f'<div class="explain-box"><div class="explain-title">⚙ &nbsp;Pipeline attribution</div>'
            f'<div class="explain-copy">This region is sensitive to {cause}. Baseline detector Weber margin: {margin_text}.</div></div>'
            f'<div class="explain-box"><div class="explain-title">ⓘ &nbsp;Recommended next step</div>'
            f'<div class="explain-copy">{next_step}</div></div>',
            unsafe_allow_html=True,
        )

    with st.container(key="audit_shell"):
        audit_rows = persisted_audit[:3]
        audit_html = "".join(
            f'<div class="audit-row"><span class="audit-time">{esc(row["time"])}</span>{esc(row["detail"])}</div>'
            for row in audit_rows
        ) or '<div class="audit-row">No review actions recorded yet.</div>'
        st.markdown(f'<div class="panel-title">◔ &nbsp;Audit trail</div>{audit_html}', unsafe_allow_html=True)
        st.download_button(
            "Download synthetic review workbook",
            Path(labels_path).read_bytes(),
            file_name="synthetic_reference_reviewed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        st.markdown(
            f'<div class="note-caption">Sources: <strong>{esc(Path(h5_path).name)}</strong> and <strong>{esc(Path(labels_path).name)}</strong>. '
            f'Channel {channel_id}: {len(labels_for_channel(labels_all, channel_id))} reference intervals. IDs shown as <strong>REF-###</strong> are synthetic reference PeakIds; <strong>CAND-###</strong> are detector-only synthetic candidates.</div>',
            unsafe_allow_html=True,
        )
