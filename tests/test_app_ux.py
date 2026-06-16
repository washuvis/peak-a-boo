from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def run_app() -> AppTest:
    app = AppTest.from_file(ROOT / "app.py", default_timeout=90)
    app.run(timeout=90)
    assert not app.exception
    return app


def test_app_loads_synthetic_demo_and_core_controls():
    app = run_app()
    assert any("Peak-a-boo Review Workbench" in str(block.value) for block in app.markdown)
    assert any("synthetic_chromatograms.h5" in str(block.value) for block in app.markdown)
    assert any("synthetic_reference.xlsx" in str(block.value) for block in app.markdown)
    assert any(button.label == "⌕  Zoom" for button in app.button)
    assert any(button.label == "◉  Accept as reviewed" for button in app.button)
    workflow = next(control for control in app.segmented_control if control.label == "Workflow view")
    assert "Labeled Data" in workflow.options
    assert "Official SOP" not in workflow.options


def test_selected_metrics_are_plain_information_rows():
    app = run_app()
    selected_panel = next(block for block in app.markdown if "Local contrast" in str(block.value))
    html = str(selected_panel.value)
    assert "evidence-list" in html
    assert "evidence-row" in html
    assert "chip violet" not in html
    assert "chip amber" not in html


def test_sandbox_uses_side_by_side_live_preview_and_grouped_controls():
    app = run_app()
    workflow = next(control for control in app.segmented_control if control.label == "Workflow view")
    workflow.set_value("Sandbox")
    app.run(timeout=90)
    assert not app.exception
    assert any(select.label == "Source" and select.value == "Synthetic demo dataset" for select in app.selectbox)
    assert any(slider.label == "Smoothing window" for slider in app.slider)
    assert any(slider.label == "Weber threshold k*" for slider in app.slider)
    assert any(text.label == "ML model path" and "synthetic_peak_classifier.joblib" in text.value for text in app.text_input)
    assert any("Live preview:" in str(block.value) for block in app.markdown)
    assert len(app.get("plotly_chart")) == 1


def test_sandbox_parameter_change_updates_live_chart():
    app = run_app()
    workflow = next(control for control in app.segmented_control if control.label == "Workflow view")
    workflow.set_value("Sandbox")
    app.run(timeout=90)
    before = app.get("plotly_chart")[0].proto.spec
    smoothing = next(slider for slider in app.slider if slider.label == "Smoothing window")
    smoothing.set_value(15)
    app.run(timeout=90)
    assert not app.exception
    after = app.get("plotly_chart")[0].proto.spec
    assert before != after


def test_modern_visual_shell_is_present_without_changing_workflow():
    app = run_app()
    assert any("Synthetic review workspace" in str(block.value) for block in app.markdown)
    assert any("decision-hero" in str(block.value) for block in app.markdown)
    css = str(app.markdown[0].value)
    assert "radial-gradient" in css
    assert "stability-ring" in css
    assert "accept_review_action" in css
