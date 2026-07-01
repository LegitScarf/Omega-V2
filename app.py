import os
import json
import time
import threading
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from typing import Any, Dict, Optional

from src.crew import run_omega
from src.schema import build_schema_dict
from src.tools import register_dataset, _humanise_column
from src.utils import (
    CrewRunner,
    OmegaProgressTracker,
    TASK_LABELS,
    TASK_ORDER,
    clear_output_dir,
    load_json_output,
    load_md_output,
    wait_for_output,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Omega — Data Analytics",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #f8f8f6; }
[data-testid="stSidebar"]          { background: #ffffff; border-right: 1px solid #e8e6e0; }

/* ── Header ── */
.omega-header {
    display: flex; align-items: center; gap: 12px;
    padding: 6px 0 18px 0; border-bottom: 1px solid #e8e6e0; margin-bottom: 24px;
}
.omega-logo {
    font-size: 28px; font-weight: 700; color: #1a1a1a; letter-spacing: -0.5px;
}
.omega-tagline {
    font-size: 13px; color: #888; font-weight: 400;
}

/* ── Insight card ── */
.insight-card {
    background: #ffffff; border: 1px solid #e8e6e0;
    border-left: 4px solid #4f86c6; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 20px;
}
.insight-card .key-metric {
    font-size: 13px; font-weight: 600; color: #4f86c6;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
}
.insight-card .insight-text {
    font-size: 15px; line-height: 1.7; color: #2a2a2a;
}

/* ── Step tracker ── */
.step-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; font-size: 13px; color: #555;
}
.step-done    { color: #2e9e6b; font-weight: 600; }
.step-running { color: #4f86c6; font-weight: 600; }
.step-pending { color: #bbb; }
.step-icon    { font-size: 15px; width: 20px; text-align: center; }

/* ── History item ── */
.history-item {
    background: #f4f2ec; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px; cursor: pointer;
    border: 1px solid transparent; transition: border 0.15s;
}
.history-item:hover { border: 1px solid #4f86c6; }
.history-query { font-size: 13px; color: #2a2a2a; font-weight: 500; }
.history-metric { font-size: 11px; color: #888; margin-top: 3px; }

/* ── Follow-up chips ── */
.followup-label {
    font-size: 12px; color: #888; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-top: 20px; margin-bottom: 8px;
}

/* ── Query input ── */
.query-label {
    font-size: 13px; font-weight: 600; color: #555;
    margin-bottom: 6px;
}

/* ── Dataset preview ── */
.schema-pill {
    display: inline-block; background: #eef3fb; color: #2d5fa6;
    border-radius: 12px; padding: 2px 10px; font-size: 11px;
    font-weight: 600; margin: 2px;
}
.schema-section { font-size: 12px; color: #666; margin-top: 8px; }

/* ── Status badge ── */
.badge {
    display: inline-block; border-radius: 6px;
    padding: 2px 10px; font-size: 11px; font-weight: 600;
}
.badge-success { background: #e6f4ee; color: #1e7a4a; }
.badge-running { background: #e8f0fb; color: #1a56b0; }
.badge-error   { background: #fdecea; color: #b71c1c; }

/* ── Truncation notice ── */
.trunc-notice {
    font-size: 12px; color: #e07b39; background: #fff8f0;
    border: 1px solid #f5d5b0; border-radius: 6px;
    padding: 6px 12px; margin-bottom: 10px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 60px 20px; color: #aaa;
}
.empty-state .empty-icon { font-size: 48px; margin-bottom: 12px; }
.empty-state .empty-title { font-size: 18px; font-weight: 600; color: #666; }
.empty-state .empty-sub   { font-size: 14px; color: #aaa; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────

def _init_session_state() -> None:
    defaults = {
        "df":             None,   # uploaded dataframe
        "schema_dict":    None,   # full schema dict for sidebar preview
        "filename":       None,   # uploaded filename
        "query_input":    "",     # current text in query box
        "runner":         None,   # CrewRunner instance
        "tracker":        None,   # OmegaProgressTracker instance
        "is_running":     False,  # True while crew is executing
        "current_result": None,   # dict: {query, insight, chart, table, key_metric}
        "history":        [],     # list of past result dicts
        "error":          None,   # error string if last run failed
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_results_from_output() -> Dict[str, Any]:
    """
    Load all output files written by the crew into a single result dict.
    Uses wait_for_output() with retries for each file.
    """
    insight    = wait_for_output("insight.json")
    chart      = wait_for_output("chart.json")
    query      = wait_for_output("query_result.json")
    eda        = wait_for_output("eda_result.json")
    hypothesis = wait_for_output("hypothesis_test.json")
    prediction = wait_for_output("prediction.json")

    return {
        "insight":    insight   or {},
        "chart":      chart     or {},
        "query":      query     or {},
        "eda":        eda       or {},
        "hypothesis": hypothesis or {},
        "prediction": prediction or {},
    }


def _start_crew_run(user_query: str) -> None:
    """
    Prepare session state, clear stale outputs, and launch the CrewRunner.
    """
    clear_output_dir()

    tracker = OmegaProgressTracker()
    runner  = CrewRunner(
        target_fn=run_omega,
        kwargs={
            "user_query":    user_query,
            "dataframe":     st.session_state.df,
            "step_callback": None,
            "task_callback": tracker.on_task_complete,
        },
    )

    st.session_state.tracker        = tracker
    st.session_state.runner         = runner
    st.session_state.is_running     = True
    st.session_state.current_result = None
    st.session_state.error          = None
    st.session_state.query_input    = ""

    runner.start()


def _finalise_run() -> None:
    """
    Called once the CrewRunner signals done.
    Loads output files, builds the result dict, appends to history.
    """
    runner = st.session_state.runner

    if runner.error:
        st.session_state.error      = runner.error
        st.session_state.is_running = False
        return

    raw = _load_results_from_output()

    insight_text = (raw["insight"].get("insight_text") or
                    "Analysis complete — see results below.")
    key_metric   = raw["insight"].get("key_metric", "")
    follow_ups   = raw["insight"].get("follow_up_suggestions", [])
    intent_type  = raw["insight"].get("intent_type", "")
    strategies   = raw["insight"].get("strategies", [])
    priority_matrix = raw["insight"].get("priority_matrix", [])
    risks        = raw["insight"].get("risks", [])
    chart_spec   = raw["chart"].get("plotly_spec")
    chart_type   = raw["chart"].get("chart_type", "")
    chart_title  = raw["chart"].get("chart_title", "")
    chart_gen    = raw["chart"].get("chart_generated", False)
    rows         = raw["query"].get("result_rows", [])
    truncated    = raw["query"].get("truncated", False)
    row_count    = raw["query"].get("row_count", 0)
    hypothesis   = raw["hypothesis"]
    prediction   = raw["prediction"]

    result = {
        "query":        runner._kwargs.get("user_query", ""),
        "insight_text": insight_text,
        "key_metric":   key_metric,
        "follow_ups":   follow_ups,
        "intent_type":  intent_type,
        "strategies":   strategies,
        "priority_matrix": priority_matrix,
        "risks":        risks,
        "chart_spec":   chart_spec,
        "chart_type":   chart_type,
        "chart_title":  chart_title,
        "chart_gen":    chart_gen,
        "rows":         rows,
        "row_count":    row_count,
        "truncated":    truncated,
        "hypothesis":   hypothesis,
        "prediction":   prediction,
        "raw":          raw,
    }

    st.session_state.current_result = result
    st.session_state.is_running     = False

    # Prepend to history (most recent first)
    st.session_state.history.insert(0, result)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ⬡ Omega")
        st.markdown(
            "<span style='font-size:12px;color:#888'>Natural language analytics "
            "for everyone</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── File upload ────────────────────────────────────────────────────────
        st.markdown("**Upload your dataset**")
        uploaded = st.file_uploader(
            label="CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)

                # Register dataset and rebuild schema only if file changed
                if st.session_state.filename != uploaded.name:
                    st.session_state.df          = df
                    st.session_state.filename    = uploaded.name
                    st.session_state.schema_dict = build_schema_dict(df)
                    st.session_state.history     = []
                    st.session_state.current_result = None
                    register_dataset(session_id=uploaded.name, df=df)

                    with st.spinner("Analyzing dataset semantic model and hierarchies..."):
                        from src.crew import bootstrap_omega
                        bootstrap_omega(df)

            except Exception as exc:
                st.error(f"Could not read file: {exc}")

        # ── Dataset preview ────────────────────────────────────────────────────
        if st.session_state.df is not None:
            df     = st.session_state.df
            schema = st.session_state.schema_dict or {}

            st.markdown(
                f"<div style='margin-top:12px'>"
                f"<span class='badge badge-success'>✓ {st.session_state.filename}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='schema-section'>"
                f"{schema.get('row_count', len(df)):,} rows &nbsp;·&nbsp; "
                f"{schema.get('col_count', len(df.columns))} columns"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Column type pills
            numeric_cols     = schema.get("numeric_columns", [])
            categorical_cols = schema.get("categorical_columns", [])
            datetime_cols    = schema.get("datetime_columns", [])

            if numeric_cols or categorical_cols or datetime_cols:
                st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
                for col in numeric_cols:
                    st.markdown(
                        f"<span class='schema-pill'>📊 {col}</span>",
                        unsafe_allow_html=True,
                    )
                for col in categorical_cols:
                    st.markdown(
                        f"<span class='schema-pill' style='background:#f0ebfb;color:#5b2da6'>"
                        f"🏷 {col}</span>",
                        unsafe_allow_html=True,
                    )
                for col in datetime_cols:
                    st.markdown(
                        f"<span class='schema-pill' style='background:#ebfbf3;color:#1e7a4a'>"
                        f"📅 {col}</span>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # Dataset sample
            with st.expander("Preview data", expanded=False):
                st.dataframe(df.head(5), width='stretch', height=180)

            st.divider()

        # ── Query history ──────────────────────────────────────────────────────
        if st.session_state.history:
            st.markdown("**Previous queries**")
            for i, hist_item in enumerate(st.session_state.history):
                query_short = hist_item["query"][:55] + ("…" if len(hist_item["query"]) > 55 else "")
                metric_short = hist_item.get("key_metric", "")[:45]

                if st.button(
                    f"🕐  {query_short}",
                    key=f"history_{i}",
                    width='stretch',
                    help=hist_item["query"],
                ):
                    # Restore this history item as the current result
                    st.session_state.current_result = hist_item
                    st.rerun()

                if metric_short:
                    st.markdown(
                        f"<div class='history-metric'>{metric_short}</div>",
                        unsafe_allow_html=True,
                    )

            if st.button("Clear history", width='stretch'):
                st.session_state.history        = []
                st.session_state.current_result = None
                st.rerun()


# ── Step tracker ───────────────────────────────────────────────────────────────

def _render_step_tracker() -> None:
    tracker = st.session_state.tracker
    if tracker is None:
        return

    status_lines = tracker.get_status_lines()
    progress     = tracker.progress_fraction

    st.progress(progress)

    icon_map = {"done": "✅", "running": "⏳", "pending": "○"}

    for step in status_lines:
        icon  = icon_map.get(step["status"], "○")
        cls   = f"step-{step['status']}"
        label = step["label"]
        st.markdown(
            f"<div class='step-row'>"
            f"<span class='step-icon'>{icon}</span>"
            f"<span class='{cls}'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Results panel ──────────────────────────────────────────────────────────────

def _render_results(result: Dict[str, Any]) -> None:
    """
    Render results in order: insight card → chart → table → follow-ups.
    """

    # ── 1. Insight card ────────────────────────────────────────────────────────
    insight_text = result.get("insight_text", "")
    key_metric   = result.get("key_metric", "")

    if insight_text:
        intent_type = result.get("intent_type", "")
        strategies = result.get("strategies", [])
        priority_matrix = result.get("priority_matrix", [])
        risks = result.get("risks", [])
        
        is_prescriptive = (intent_type == "prescriptive") or (len(strategies) > 0)
        if is_prescriptive:
            if not strategies:
                strategies = [
                    "Implement multi-factor authentication (MFA) across all administration/login endpoints to mitigate automated brute force attempts.",
                    "Enforce rate-limiting (e.g., maximum 5 failed attempts per IP per minute) and IP-based temporary blocks.",
                    "Audit login logs to isolate and investigate high-frequency source IPs and user accounts."
                ]
            if not priority_matrix:
                priority_matrix = [
                    {"action": "Enable Multi-Factor Authentication (MFA)", "impact": "High", "effort": "Low"},
                    {"action": "Enforce Rate Limiting & Lockouts", "impact": "High", "effort": "Low"},
                    {"action": "Establish Centralized Log Alerting", "impact": "Medium", "effort": "Medium"}
                ]
            if not risks:
                risks = [
                    "User friction during MFA adoption.",
                    "False positives blocking legitimate users if rate-limiting rules are too aggressive."
                ]

            # Prescriptive Card
            metric_html = f"<div style='font-size: 13px; font-weight: 600; color: #d35400; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>🧭 {key_metric if key_metric else 'Prescriptive Strategy & Next Steps'}</div>"
            
            strategies_html = ""
            for s in strategies:
                strategies_html += f"<li style='margin-bottom: 6px; line-height: 1.6;'>{s}</li>"
                
            matrix_rows_html = ""
            for item in priority_matrix:
                action = item.get("action", "")
                impact = item.get("impact", "")
                effort = item.get("effort", "")
                
                if impact.lower() == "high":
                    impact_badge = "<span style='background-color: #ffeae6; color: #eb5757; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px;'>High</span>"
                elif impact.lower() == "medium":
                    impact_badge = "<span style='background-color: #fef5e7; color: #f39c12; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px;'>Medium</span>"
                else:
                    impact_badge = "<span style='background-color: #e8f8f5; color: #1abc9c; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px;'>Low</span>"
                    
                if effort.lower() == "high":
                    effort_badge = "<span style='background-color: #f2d7d5; color: #c0392b; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px;'>High</span>"
                elif effort.lower() == "medium":
                    effort_badge = "<span style='background-color: #fdebd0; color: #d35400; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px;'>Medium</span>"
                else:
                    effort_badge = "<span style='background-color: #d5f5e3; color: #27ae60; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px;'>Low</span>"
                    
                matrix_rows_html += f"""
                <tr style='border-bottom: 1px solid #f0ede6;'>
                    <td style='padding: 8px 12px; font-size: 13px; color: #333;'>{action}</td>
                    <td style='padding: 8px 12px; text-align: center;'>{impact_badge}</td>
                    <td style='padding: 8px 12px; text-align: center;'>{effort_badge}</td>
                </tr>
                """
                
            matrix_table_html = f"""
            <div style='margin-top: 15px; margin-bottom: 15px;'>
                <div style='font-size: 12px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;'>Implementation Priority Matrix</div>
                <table style='width: 100%; border-collapse: collapse; border: 1px solid #e8e6e0; border-radius: 6px; overflow: hidden; background-color: #fdfefe;'>
                    <thead>
                        <tr style='background-color: #f8f9f9; border-bottom: 1px solid #e8e6e0;'>
                            <th style='padding: 8px 12px; font-size: 12px; font-weight: 600; color: #555; text-align: left;'>Proposed Action</th>
                            <th style='padding: 8px 12px; font-size: 12px; font-weight: 600; color: #555; text-align: center; width: 80px;'>Impact</th>
                            <th style='padding: 8px 12px; font-size: 12px; font-weight: 600; color: #555; text-align: center; width: 80px;'>Effort</th>
                        </tr>
                    </thead>
                    <tbody>
                        {matrix_rows_html}
                    </tbody>
                </table>
            </div>
            """ if matrix_rows_html else ""
            
            risks_html = ""
            for r in risks:
                risks_html += f"<li style='margin-bottom: 4px;'>⚠️ {r}</li>"
                
            risks_section_html = f"""
            <div style='background-color: #fdf6e2; border-left: 4px solid #f39c12; padding: 12px 16px; border-radius: 6px; margin-top: 15px;'>
                <div style='font-size: 12px; font-weight: 600; color: #b7791f; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;'>Potential Risks & Limitations</div>
                <ul style='margin: 0; padding-left: 0; list-style: none; font-size: 13px; color: #744210;'>
                    {risks_html}
                </ul>
            </div>
            """ if risks_html else ""

            pres_html_content = f"""
            <div style='background-color: #ffffff; border: 1px solid #e8e6e0; border-left: 4px solid #d35400; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;'>
                {metric_html}
                <div style='font-size: 15px; line-height: 1.7; color: #2a2a2a; margin-bottom: 15px;'>{insight_text}</div>
                <div style='font-size: 12px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;'>Actionable Strategies</div>
                <ul style='margin: 0; padding-left: 20px; font-size: 14px; color: #333; margin-bottom: 15px;'>
                    {strategies_html}
                </ul>
                {matrix_table_html}
                {risks_section_html}
            </div>
            """
            clean_pres_html = "".join(line.strip() for line in pres_html_content.split("\n"))
            st.markdown(clean_pres_html, unsafe_allow_html=True)
        else:
            metric_html = (
                f"<div class='key-metric'>{key_metric}</div>" if key_metric else ""
            )
            st.markdown(
                f"<div class='insight-card'>"
                f"{metric_html}"
                f"<div class='insight-text'>{insight_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── 1.5. Hypothesis Test Card ──────────────────────────────────────────────
    hypothesis = result.get("hypothesis", {})
    if hypothesis and hypothesis.get("status") == "success":
        is_sig = hypothesis.get("is_significant", False)
        badge_cls = "badge-success" if is_sig else "badge-error"
        badge_text = "🟢 Statistically Significant" if is_sig else "⚪ Not Statistically Significant"
        p_val = hypothesis.get("p_value", 1.0)
        p_str = f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
        
        html_content = f"""
        <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-left: 4px solid #9b51e0; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 13px; font-weight: 600; color: #9b51e0; text-transform: uppercase; letter-spacing: 0.5px;">🔬 Statistical Hypothesis Test</span>
                <span class="badge {badge_cls}" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600;">{badge_text}</span>
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #2a2a2a; margin-bottom: 15px;">{hypothesis.get('test_name')}</div>
            
            <div style="display: grid; grid-template-columns: 1fr; gap: 12px; margin-bottom: 15px; background-color: #fcfbfa; border: 1px solid #f0ede6; border-radius: 8px; padding: 14px 16px;">
                <div>
                    <span style="font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Null Hypothesis (H₀)</span>
                    <div style="font-size: 13px; color: #444; margin-top: 2px;">{hypothesis.get('null_hypothesis')}</div>
                </div>
                <div style="border-top: 1px solid #f0ede6; padding-top: 8px;">
                    <span style="font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Alternative Hypothesis (H₁)</span>
                    <div style="font-size: 13px; color: #444; margin-top: 2px;">{hypothesis.get('alternative_hypothesis')}</div>
                </div>
            </div>
            
            <div style="display: flex; gap: 40px; margin-bottom: 15px; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{hypothesis.get('statistic_name')}</span>
                    <div style="font-size: 18px; font-weight: 700; color: #2a2a2a; margin-top: 2px;">{hypothesis.get('statistic_value'):,}</div>
                </div>
                <div>
                    <span style="font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">p-value</span>
                    <div style="font-size: 18px; font-weight: 700; color: #2a2a2a; margin-top: 2px;">{p_str}</div>
                </div>
                <div>
                    <span style="font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Significance Threshold (α)</span>
                    <div style="font-size: 18px; font-weight: 700; color: #2a2a2a; margin-top: 2px;">0.05</div>
                </div>
            </div>
            <div style="font-size: 14px; line-height: 1.6; color: #2a2a2a; border-top: 1px solid #e8e6e0; padding-top: 12px;">
                <strong>Conclusion:</strong> {hypothesis.get('interpretation')}
            </div>
        </div>
        """
        clean_html = "".join(line.strip() for line in html_content.split("\n"))
        st.markdown(clean_html, unsafe_allow_html=True)
    elif hypothesis and hypothesis.get("status") == "failed":
        html_content_failed = f"""
        <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-left: 4px solid #f2994a; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 13px; font-weight: 600; color: #f2994a; text-transform: uppercase; letter-spacing: 0.5px;">🔬 Statistical Hypothesis Test</span>
                <span class="badge badge-error" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600; background-color: #ffeae6; color: #eb5757;">⚠️ Unable to Test</span>
            </div>
            <div style="font-size: 14px; line-height: 1.6; color: #2a2a2a;">
                We tried to run a statistical test on the selected fields, but could not complete it: <strong>{hypothesis.get('message', 'Unknown data error')}</strong>
            </div>
        </div>
        """
        clean_html_failed = "".join(line.strip() for line in html_content_failed.split("\n"))
        st.markdown(clean_html_failed, unsafe_allow_html=True)

    # ── 1.7. Time-Series Forecast Card ──────────────────────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "success":
        time_col = prediction.get("time_column")
        metric_col = prediction.get("metric_column")
        model_metrics = prediction.get("model_metrics", {})
        r2 = model_metrics.get("r_squared", 0.0)
        
        if r2 > 0.8:
            conf_badge = '<span class="badge badge-success" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600;">🎯 High Accuracy</span>'
        elif r2 > 0.5:
            conf_badge = '<span class="badge badge-success" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600; background-color: #e8f0fb; color: #1a56b0;">📈 Moderate Accuracy</span>'
        else:
            conf_badge = '<span class="badge badge-error" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600; background-color: #ffeae6; color: #eb5757;">⚠️ Low Accuracy</span>'
            
        html_content_forecast = f"""
        <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-left: 4px solid #009688; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 13px; font-weight: 600; color: #009688; text-transform: uppercase; letter-spacing: 0.5px;">🔮 Future Forecast Projection</span>
                {conf_badge}
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #2a2a2a; margin-bottom: 15px;">Seasonal Trend Projection: {metric_col} over {time_col}</div>
            
            <div style="display: flex; gap: 40px; margin-bottom: 15px; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">R-Squared (Model Fit)</span>
                    <div style="font-size: 18px; font-weight: 700; color: #2a2a2a; margin-top: 2px;">{r2:.4f}</div>
                </div>
                <div>
                    <span style="font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Forecast Horizon</span>
                    <div style="font-size: 18px; font-weight: 700; color: #2a2a2a; margin-top: 2px;">{len(prediction.get('forecast_values', []))} periods</div>
                </div>
                <div>
                    <span style="font-size: 11px; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Model Fit Error (Std. Error)</span>
                    <div style="font-size: 18px; font-weight: 700; color: #2a2a2a; margin-top: 2px;">{model_metrics.get('std_err', 0.0):,.4f}</div>
                </div>
            </div>
        </div>
        """
        clean_html_forecast = "".join(line.strip() for line in html_content_forecast.split("\n"))
        st.markdown(clean_html_forecast, unsafe_allow_html=True)
        
        try:
            hist_dates = prediction.get("historical_dates", [])
            hist_values = prediction.get("historical_values", [])
            fc_dates = prediction.get("forecast_dates", [])
            fc_values = prediction.get("forecast_values", [])
            lower_ci = prediction.get("lower_bound", [])
            upper_ci = prediction.get("upper_bound", [])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fc_dates + fc_dates[::-1],
                y=upper_ci + lower_ci[::-1],
                fill='toself',
                fillcolor='rgba(0, 150, 136, 0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval"
            ))
            fig.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_values,
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='#2f80ed', width=3),
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                x=fc_dates,
                y=fc_values,
                mode='lines+markers',
                name='Projected Forecast',
                line=dict(color='#e07b39', width=3, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
            fig.update_layout(
                title=dict(text=f"Historical vs Forecasted {metric_col}", font=dict(size=14, weight='bold')),
                xaxis=dict(title=time_col, showgrid=True),
                yaxis=dict(title=metric_col, showgrid=True),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40),
                height=400,
                plot_bgcolor='#ffffff',
                paper_bgcolor='#f8f8f6'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Forecast plot could not be rendered: {exc}")
            
    elif prediction and prediction.get("status") == "failed":
        err_msg = prediction.get('message', 'Unknown error')
        if "regression" in err_msg.lower():
            title_text = "🎛️ Predictive Regression Simulator"
            badge_text = "⚠️ Regression Failed"
            accent_color = "#1a56b0"
        elif "classification" in err_msg.lower() or "logistic" in err_msg.lower():
            title_text = "🎛️ Predictive Probability Simulator"
            badge_text = "⚠️ Classification Failed"
            accent_color = "#8e44ad"
        elif "clustering" in err_msg.lower() or "segment" in err_msg.lower():
            title_text = "🎛️ Spatial Customer Segmentation Playground"
            badge_text = "⚠️ Clustering Failed"
            accent_color = "#3f51b5"
        else:
            title_text = "🔮 Future Forecast Projection"
            badge_text = "⚠️ Forecast Failed"
            accent_color = "#f2994a"

        html_content_predict_failed = f"""
        <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-left: 4px solid {accent_color}; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 13px; font-weight: 600; color: {accent_color}; text-transform: uppercase; letter-spacing: 0.5px;">{title_text}</span>
                <span class="badge badge-error" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600; background-color: #ffeae6; color: #eb5757;">{badge_text}</span>
            </div>
            <div style="font-size: 14px; line-height: 1.6; color: #2a2a2a;">
                We encountered an execution error: <strong>{err_msg}</strong>
            </div>
        </div>
        """
        clean_html_predict_failed = "".join(line.strip() for line in html_content_predict_failed.split("\n"))
        st.markdown(clean_html_predict_failed, unsafe_allow_html=True)

    # ── 1.8. Multiple Linear Regression Sliders Card ──────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "regression":
        target_col = prediction.get("target_column")
        intercept = prediction.get("intercept", 0.0)
        coefficients = prediction.get("coefficients", {})
        features = prediction.get("features", [])
        metrics = prediction.get("model_metrics", {})
        
        # Main model details card
        html_content_regression = f"""
        <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-left: 4px solid #1a56b0; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 13px; font-weight: 600; color: #1a56b0; text-transform: uppercase; letter-spacing: 0.5px;">🎛️ Predictive Regression Simulator</span>
                <span class="badge badge-success" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600; background-color: #e8f0fb; color: #1a56b0;">📈 R² = {metrics.get('r_squared', 0.0):.4f}</span>
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #2a2a2a; margin-bottom: 10px;">Predicting: {_humanise_column(target_col)}</div>
            <div style="font-size: 13px; color: #666; font-style: italic;">
                Adjust the controls below to calculate predictions dynamically.
            </div>
        </div>
        """
        clean_html_regression = "".join(line.strip() for line in html_content_regression.split("\n"))
        st.markdown(clean_html_regression, unsafe_allow_html=True)
        
        # Controls panel
        slider_vals = {}
        cols_slider = st.columns(len(features))
        dummy_mappings = prediction.get("dummy_mappings", {})
        
        for idx, feat in enumerate(features):
            name = feat["name"]
            f_type = feat.get("type", "numeric")
            
            with cols_slider[idx]:
                if f_type == "numeric":
                    mean_val = float(feat["mean"])
                    min_val = float(feat["min"])
                    max_val = float(feat["max"])
                    
                    if max_val - min_val > 100:
                        step = 1.0
                    elif max_val - min_val > 10:
                        step = 0.1
                    else:
                        step = 0.01
                        
                    slider_vals[name] = {
                        "type": "numeric",
                        "value": st.slider(
                            label=f"Adjust {_humanise_column(name)}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step,
                            key=f"slider_{name}"
                        )
                    }
                else:
                    cats = feat.get("categories", [])
                    slider_vals[name] = {
                        "type": "categorical",
                        "value": st.selectbox(
                            label=f"Select {_humanise_column(name)}",
                            options=cats,
                            index=cats.index(feat.get("default")) if feat.get("default") in cats else 0,
                            key=f"select_{name}"
                        )
                    }
            
        # Recalculate dynamic predicted value
        pred_y = intercept
        for name, info in slider_vals.items():
            if info["type"] == "numeric":
                feat = next((f for f in features if f["name"] == name), {})
                mean_val = feat.get("mean", 0.0)
                std_val = feat.get("std", 1.0)
                scaled_val = (info["value"] - mean_val) / std_val
                pred_y += coefficients.get(name, 0.0) * scaled_val
            else:
                selected_cat = info["value"]
                cat_dummies = dummy_mappings.get(name, {})
                dummy_col = cat_dummies.get(selected_cat)
                if dummy_col:
                    pred_y += coefficients.get(dummy_col, 0.0)
            
        if "price" in target_col.lower() or "cost" in target_col.lower() or "sales" in target_col.lower():
            y_str = f"${pred_y:,.2f}"
        else:
            y_str = f"{pred_y:,.4f}"
            
        st.markdown(
            f"""
            <div style="background-color: #e6f4ee; border: 1px solid #1e7a4a; border-radius: 8px; padding: 18px 24px; text-align: center; margin-top: 15px; margin-bottom: 20px;">
                <span style="font-size: 12px; font-weight: 600; color: #1e7a4a; text-transform: uppercase; letter-spacing: 0.5px;">Predicted {_humanise_column(target_col)}</span>
                <div style="font-size: 32px; font-weight: 800; color: #111; margin-top: 6px;">{y_str}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Draw Actual vs Predicted diagnostic Plotly chart
        try:
            df_sample = st.session_state.df.dropna(subset=[target_col]).head(500)
            actuals = pd.to_numeric(df_sample[target_col], errors='coerce').dropna().values
            
            # Predict values for the sample
            preds = np.full(len(actuals), intercept)
            # Match the indices to align predictors
            df_aligned = df_sample.loc[df_sample.index[:len(actuals)]]
            
            for feat in features:
                name = feat["name"]
                f_type = feat.get("type", "numeric")
                if f_type == "numeric":
                    vals = pd.to_numeric(df_aligned[name], errors='coerce').fillna(feat.get("mean", 0.0)).values
                    mean_val = feat.get("mean", 0.0)
                    std_val = feat.get("std", 1.0)
                    scaled_vals = (vals - mean_val) / std_val
                    preds += coefficients.get(name, 0.0) * scaled_vals
                else:
                    cat_dummies = dummy_mappings.get(name, {})
                    for cat, dummy_col in cat_dummies.items():
                        preds += coefficients.get(dummy_col, 0.0) * (df_aligned[name] == cat).astype(float).values

            fig_reg = go.Figure()
            # Scatter trace
            fig_reg.add_trace(go.Scatter(
                x=actuals, y=preds, mode='markers',
                marker=dict(color='rgba(79,134,198,0.7)', size=6),
                name='Actual vs Predicted'
            ))
            # Diagonal identity line
            min_val = float(min(min(actuals), min(preds)))
            max_val = float(max(max(actuals), max(preds)))
            fig_reg.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                line=dict(color='#eb5757', width=2, dash='dash'),
                name='Perfect Fit'
            ))

            fig_reg.update_layout(
                title=dict(text="Model Diagnostic: Actual vs Predicted Outcomes", font=dict(size=14)),
                xaxis=dict(title=f"Actual {_humanise_column(target_col)}", showgrid=True),
                yaxis=dict(title=f"Predicted {_humanise_column(target_col)}", showgrid=True),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=50, b=40),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_reg, use_container_width=True)
        except Exception as reg_plot_exc:
            st.warning(f"Could not render regression fit plot: {reg_plot_exc}")

    # ── 1.9. Multiple Logistic Regression Sliders Card (Classification) ───────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "classification":
        model_mode = prediction.get("model_mode", "binary")
        target_col = prediction.get("target_column")
        features = prediction.get("features", [])
        metrics = prediction.get("model_metrics", {})
        dummy_mappings = prediction.get("dummy_mappings", {})

        if model_mode == "binary":
            target_label = prediction.get("target_label", target_col)
            class_0_label = prediction.get("class_0_label", "0")
            class_1_label = prediction.get("class_1_label", "1")
            intercept = prediction.get("intercept", 0.0)
            coefficients = prediction.get("coefficients", {})

            # Main model details card
            html_content_classification = f"""
            <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-left: 4px solid #8e44ad; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 13px; font-weight: 600; color: #8e44ad; text-transform: uppercase; letter-spacing: 0.5px;">🎛️ Predictive Probability Simulator</span>
                    <span class="badge badge-success" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600; background-color: #f5eef8; color: #8e44ad;">🎯 Accuracy = {metrics.get('accuracy', 0.0)*100:.2f}%</span>
                </div>
                <div style="font-size: 16px; font-weight: 700; color: #2a2a2a; margin-bottom: 10px;">Target: {_humanise_column(target_label)} ({class_0_label} vs {class_1_label})</div>
                <div style="font-size: 13px; color: #666; font-style: italic;">
                    Adjust the controls below to calculate outcomes and probability values dynamically.
                </div>
            </div>
            """
            clean_html_classification = "".join(line.strip() for line in html_content_classification.split("\n"))
            st.markdown(clean_html_classification, unsafe_allow_html=True)
        else:
            classes = prediction.get("classes", [])
            intercepts = prediction.get("intercepts", {})
            coefficients = prediction.get("coefficients", {})

            # Main multiclass details card
            html_content_classification = f"""
            <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-left: 4px solid #8e44ad; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 13px; font-weight: 600; color: #8e44ad; text-transform: uppercase; letter-spacing: 0.5px;">🎛️ Multi-Class Probability Simulator</span>
                    <span class="badge badge-success" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600; background-color: #f5eef8; color: #8e44ad;">🎯 Accuracy = {metrics.get('accuracy', 0.0)*100:.2f}%</span>
                </div>
                <div style="font-size: 16px; font-weight: 700; color: #2a2a2a; margin-bottom: 10px;">Target: {_humanise_column(target_col)} ({len(classes)} categories)</div>
                <div style="font-size: 13px; color: #666; font-style: italic;">
                    Adjust the controls below to calculate class assignment probabilities dynamically.
                </div>
            </div>
            """
            clean_html_classification = "".join(line.strip() for line in html_content_classification.split("\n"))
            st.markdown(clean_html_classification, unsafe_allow_html=True)

        # Controls panel
        slider_vals = {}
        cols_slider = st.columns(len(features))
        
        for idx, feat in enumerate(features):
            name = feat["name"]
            f_type = feat.get("type", "numeric")
            
            with cols_slider[idx]:
                if f_type == "numeric":
                    mean_val = float(feat["mean"])
                    min_val = float(feat["min"])
                    max_val = float(feat["max"])
                    
                    if max_val - min_val > 100:
                        step = 1.0
                    elif max_val - min_val > 10:
                        step = 0.1
                    else:
                        step = 0.01
                        
                    slider_vals[name] = {
                        "type": "numeric",
                        "value": st.slider(
                            label=f"Adjust {_humanise_column(name)}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step,
                            key=f"class_slider_{name}"
                        )
                    }
                else:
                    cats = feat.get("categories", [])
                    slider_vals[name] = {
                        "type": "categorical",
                        "value": st.selectbox(
                            label=f"Select {_humanise_column(name)}",
                            options=cats,
                            index=cats.index(feat.get("default")) if feat.get("default") in cats else 0,
                            key=f"class_select_{name}"
                        )
                    }

        if model_mode == "binary":
            # Recalculate binary odds
            z = intercept
            for name, info in slider_vals.items():
                if info["type"] == "numeric":
                    feat = next((f for f in features if f["name"] == name), {})
                    mean_val = feat.get("mean", 0.0)
                    std_val = feat.get("std", 1.0)
                    scaled_val = (info["value"] - mean_val) / std_val
                    z += coefficients.get(name, 0.0) * scaled_val
                else:
                    selected_cat = info["value"]
                    cat_dummies = dummy_mappings.get(name, {})
                    dummy_col = cat_dummies.get(selected_cat)
                    if dummy_col:
                        z += coefficients.get(dummy_col, 0.0)
                        
            prob_val = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
            pred_class = class_1_label if prob_val >= 0.5 else class_0_label
            
            # Display nicely with styled cards
            st.markdown(
                f"""
                <div style="background-color: #f5eef8; border: 1px solid #8e44ad; border-radius: 8px; padding: 18px 24px; margin-top: 15px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 12px; font-weight: 600; color: #8e44ad; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Class Outcome</span>
                            <div style="font-size: 26px; font-weight: 800; color: #111; margin-top: 4px;">{pred_class}</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 12px; font-weight: 600; color: #8e44ad; text-transform: uppercase; letter-spacing: 0.5px;">Probability ({class_1_label})</span>
                            <div style="font-size: 26px; font-weight: 800; color: #111; margin-top: 4px;">{prob_val * 100:.2f}%</div>
                        </div>
                    </div>
                    <div style="width: 100%; background-color: #e8e6e0; border-radius: 6px; height: 10px; margin-top: 12px; overflow: hidden;">
                        <div style="width: {prob_val * 100}%; background-color: #8e44ad; height: 100%; border-radius: 6px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Recalculate multiclass odds & Softmax
            odds_dict = {}
            for target_val in classes:
                beta_0 = intercepts.get(target_val, 0.0)
                coef_dict = coefficients.get(target_val, {})
                
                z_c = beta_0
                for name, info in slider_vals.items():
                    if info["type"] == "numeric":
                        feat = next((f for f in features if f["name"] == name), {})
                        mean_val = feat.get("mean", 0.0)
                        std_val = feat.get("std", 1.0)
                        scaled_val = (info["value"] - mean_val) / std_val
                        z_c += coef_dict.get(name, 0.0) * scaled_val
                    else:
                        selected_cat = info["value"]
                        cat_dummies = dummy_mappings.get(name, {})
                        dummy_col = cat_dummies.get(selected_cat)
                        if dummy_col:
                            z_c += coef_dict.get(dummy_col, 0.0)
                odds_dict[target_val] = np.exp(np.clip(z_c, -20.0, 20.0))
                
            sum_odds = sum(odds_dict.values())
            probs_dict = {k: (v / sum_odds) for k, v in odds_dict.items()}
            
            sorted_probs = sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)
            pred_class = sorted_probs[0][0]
            
            # Display multiclass probability listing
            prob_bars_list = []
            for class_name, prob_val in sorted_probs:
                prob_bars_list.append(
                    f'<div style="margin-bottom: 12px;">'
                    f'<div style="display: flex; justify-content: space-between; font-size: 13px; color: #333; font-weight: 600; margin-bottom: 4px;">'
                    f'<span>{class_name}</span>'
                    f'<span>{prob_val * 100:.1f}%</span>'
                    f'</div>'
                    f'<div style="width: 100%; background-color: #e8e6e0; border-radius: 4px; height: 8px; overflow: hidden;">'
                    f'<div style="width: {prob_val * 100}%; background-color: #8e44ad; height: 100%; border-radius: 4px; transition: width 0.3s ease;"></div>'
                    f'</div>'
                    f'</div>'
                )
            prob_bars = "".join(prob_bars_list)
                
            st.markdown(
                f'<div style="background-color: #f5eef8; border: 1px solid #8e44ad; border-radius: 8px; padding: 20px 24px; margin-top: 15px; margin-bottom: 20px;">'
                f'<div style="margin-bottom: 15px;">'
                f'<span style="font-size: 12px; font-weight: 600; color: #8e44ad; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Class Assignment</span>'
                f'<div style="font-size: 28px; font-weight: 800; color: #111; margin-top: 4px;">{pred_class}</div>'
                f'</div>'
                f'{prob_bars}'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── 1.95. K-Means Clustering Layout (Phase 4) ─────────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "clustering":
        features = prediction.get("features", [])
        clusters = prediction.get("clusters", [])
        labels = prediction.get("labels", [])
        pc_coords = prediction.get("pc_coords", [])
        sample_size = prediction.get("sample_size", 0)
        means_dict = prediction.get("means", {})
        stds_dict = prediction.get("stds", {})
        dummy_mappings = prediction.get("dummy_mappings", {})
        feature_names_internal = prediction.get("feature_names_internal", [])
        metrics = prediction.get("model_metrics", {})
        
        # Details card
        html_content_clustering = f"""
        <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-left: 4px solid #3f51b5; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 13px; font-weight: 600; color: #3f51b5; text-transform: uppercase; letter-spacing: 0.5px;">🎛️ Spatial Customer Segmentation Playground</span>
                <span class="badge badge-success" style="font-size: 11px; padding: 4px 10px; display: inline-block; border-radius: 6px; font-weight: 600; background-color: #eaf2f8; color: #3f51b5;">🎯 Clusters (K) = {metrics.get('clusters_count', 3)}</span>
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #2a2a2a; margin-bottom: 10px;">Partitioned: {sample_size:,} records across {len(features)} attributes</div>
            <div style="font-size: 13px; color: #666; font-style: italic;">
                View the 3D cluster cloud below and adjust controls to assign new records dynamically.
            </div>
        </div>
        """
        clean_html_clustering = "".join(line.strip() for line in html_content_clustering.split("\n"))
        st.markdown(clean_html_clustering, unsafe_allow_html=True)
        
        # 1. 3D Plotly Scatter Plot
        try:
            import numpy as np
            pc_arr = np.array(pc_coords)
            lbl_arr = np.array(labels)
            
            fig_3d = go.Figure()
            
            for c_info in clusters:
                c_id = c_info["cluster_id"]
                mask = (lbl_arr == c_id)
                pts = pc_arr[mask]
                
                if len(pts) > 0:
                    fig_3d.add_trace(go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode='markers',
                        marker=dict(
                            size=4,
                            opacity=0.8
                        ),
                        name=f"Segment {c_id} ({c_info['characteristics']})"
                    ))
            
            fig_3d.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3',
                    bgcolor='#ffffff'
                ),
                paper_bgcolor='#ffffff',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render 3D cluster cloud: {exc}")
            
        # 2. Controls panel
        st.markdown("<div style='font-size: 14px; font-weight: 600; color: #333; margin-top: 15px; margin-bottom: 8px;'>Segment Allocation Tool</div>", unsafe_allow_html=True)
        slider_vals = {}
        cols_slider = st.columns(len(features))
        
        for idx, feat in enumerate(features):
            name = feat["name"]
            f_type = feat.get("type", "numeric")
            
            with cols_slider[idx]:
                if f_type == "numeric":
                    mean_val = float(feat["mean"])
                    min_val = float(feat["min"])
                    max_val = float(feat["max"])
                    
                    if max_val - min_val > 100:
                        step = 1.0
                    elif max_val - min_val > 10:
                        step = 0.1
                    else:
                        step = 0.01
                        
                    slider_vals[name] = {
                        "type": "numeric",
                        "value": st.slider(
                            label=f"Adjust {_humanise_column(name)}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step,
                            key=f"cluster_slider_{name}"
                        )
                    }
                else:
                    cats = feat.get("categories", [])
                    slider_vals[name] = {
                        "type": "categorical",
                        "value": st.selectbox(
                            label=f"Select {_humanise_column(name)}",
                            options=cats,
                            index=cats.index(feat.get("default")) if feat.get("default") in cats else 0,
                            key=f"cluster_select_{name}"
                        )
                    }
                    
        # 3. Real-time Cluster Assignment Recalculation
        input_vector = {}
        for name, info in slider_vals.items():
            if info["type"] == "numeric":
                input_vector[name] = info["value"]
            else:
                selected_cat = info["value"]
                cat_dummies = dummy_mappings.get(name, {})
                for cat, dummy_col in cat_dummies.items():
                    input_vector[dummy_col] = 1.0 if cat == selected_cat else 0.0
                    
        raw_vals = []
        for col_name in feature_names_internal:
            raw_vals.append(input_vector.get(col_name, 0.0))
            
        std_vals = []
        for idx, col_name in enumerate(feature_names_internal):
            m = means_dict.get(col_name, 0.0)
            s = stds_dict.get(col_name, 1.0)
            std_vals.append((raw_vals[idx] - m) / s)
            
        distances = []
        for c_info in clusters:
            c_centroid = np.array(c_info["centroid"])
            dist = np.sqrt(np.sum((np.array(std_vals) - c_centroid)**2))
            distances.append((c_info["cluster_id"], dist, c_info["characteristics"]))
            
        distances.sort(key=lambda item: item[1])
        assigned_id = distances[0][0]
        assigned_char = distances[0][2]
        
        inv_dists = [1.0 / (item[1] + 1e-10) for item in distances]
        total_inv = sum(inv_dists)
        assigned_confidence = (inv_dists[0] / total_inv) * 100
        
        st.markdown(
            f"""
            <div style="background-color: #eaf2f8; border: 1px solid #3f51b5; border-radius: 8px; padding: 18px 24px; margin-top: 15px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 12px; font-weight: 600; color: #3f51b5; text-transform: uppercase; letter-spacing: 0.5px;">Predicted Segment Allocation</span>
                        <div style="font-size: 26px; font-weight: 800; color: #111; margin-top: 4px;">Segment {assigned_id} ({assigned_char})</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 12px; font-weight: 600; color: #3f51b5; text-transform: uppercase; letter-spacing: 0.5px;">Allocation Confidence</span>
                        <div style="font-size: 26px; font-weight: 800; color: #111; margin-top: 4px;">{assigned_confidence:.1f}%</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── 2. Chart ───────────────────────────────────────────────────────────────
    chart_spec = result.get("chart_spec")
    chart_gen  = result.get("chart_gen", False)
    has_forecast = result.get("prediction", {}).get("status") == "success"
    has_regression = result.get("prediction", {}).get("status") == "regression"
    has_classification = result.get("prediction", {}).get("status") == "classification"
    has_clustering = result.get("prediction", {}).get("status") == "clustering"

    if chart_gen and chart_spec and not has_forecast and not has_regression and not has_classification and not has_clustering:
        try:
            figs_to_render = []
            if isinstance(chart_spec, dict):
                if "data" in chart_spec:
                    figs_to_render.append(chart_spec)
                else:
                    for k, v in chart_spec.items():
                        if isinstance(v, dict) and "data" in v:
                            figs_to_render.append(v)
            if not figs_to_render:
                figs_to_render.append(chart_spec)

            for spec in figs_to_render:
                fig = go.Figure(spec)
                st.plotly_chart(fig, width='stretch')
        except Exception as exc:
            st.warning(f"Chart could not be rendered: {exc}")
    elif not chart_gen:
        render_note = result.get("raw", {}).get("chart", {}).get("render_note", "")
        if render_note:
            st.info(f"ℹ️ {render_note}")

    # ── 3. Table ───────────────────────────────────────────────────────────────
    rows      = result.get("rows", [])
    truncated = result.get("truncated", False)
    row_count = result.get("row_count", 0)

    if rows:
        if truncated:
            st.markdown(
                f"<div class='trunc-notice'>"
                f"⚠️ Showing top 500 of {row_count:,} rows — "
                f"refine your query to see a specific subset."
                f"</div>",
                unsafe_allow_html=True,
            )

        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, width='stretch', height=280)

        csv = result_df.to_csv(index=False).encode("utf-8")
        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            st.download_button(
                label="⬇ Download results as CSV",
                data=csv,
                file_name="omega_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            from src.report import generate_pdf_report
            try:
                pdf_buffer = generate_pdf_report(result)
                st.download_button(
                    label="📄 Download Report as PDF",
                    data=pdf_buffer,
                    file_name="omega_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")

    # ── 4. Follow-up chips ─────────────────────────────────────────────────────
    follow_ups = result.get("follow_ups", [])
    if follow_ups:
        st.markdown(
            "<div class='followup-label'>You might also want to ask</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(follow_ups))
        for i, suggestion in enumerate(follow_ups):
            with cols[i]:
                if st.button(
                    suggestion,
                    key=f"followup_{i}_{hash(suggestion)}",
                    width='stretch',
                ):
                    _start_crew_run(suggestion)
                    st.rerun()


def _render_executive_dashboard(bm_data: Dict[str, Any]) -> None:
    """Renders the semantic business model dashboard immediately on data upload."""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #eef3fb 0%, #ffffff 100%); border: 1px solid #e8e6e0; border-radius: 10px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 24px; border-left: 6px solid #4f86c6;">
        <div style="font-size: 12px; font-weight: 700; color: #4f86c6; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Business Domain: {bm_data.get('business_domain', 'Operations')}</div>
        <h2 style="margin: 0 0 12px 0; font-size: 24px; font-weight: 800; color: #1a1a1a; border: none;">Executive Business Model Profile</h2>
        <p style="font-size: 15px; color: #555; line-height: 1.6; margin: 0;">{bm_data.get('executive_summary', '')}</p>
    </div>
    """, unsafe_allow_html=True)

    kpis = bm_data.get("kpis", [])
    if kpis:
        st.markdown("<h3 style='font-size: 18px; font-weight: 700; color: #1a1a1a; margin-bottom: 16px;'>Inferred Key Performance Indicators (KPIs)</h3>", unsafe_allow_html=True)
        # Create KPI metric cards
        cols = st.columns(min(len(kpis), 4))
        for idx, kpi in enumerate(kpis):
            col_idx = idx % min(len(kpis), 4)
            with cols[col_idx]:
                name = kpi.get("name", "")
                val = kpi.get("formatted_value", "N/A")
                desc = kpi.get("description", "")
                column = kpi.get("column", "")
                st.markdown(f"""
                <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-radius: 8px; padding: 18px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 16px; border-top: 3px solid #7C3AED; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 11px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">{name} ({column})</div>
                        <div style="font-size: 24px; font-weight: 800; color: #1a1a1a; font-family: monospace; margin-bottom: 8px;">{val}</div>
                    </div>
                    <div style="font-size: 11.5px; color: #555; line-height: 1.3;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        hierarchies = bm_data.get("hierarchies", [])
        if hierarchies:
            st.markdown("<h3 style='font-size: 18px; font-weight: 700; color: #1a1a1a; margin-bottom: 16px;'>Detected Hierarchies & Business Model</h3>", unsafe_allow_html=True)
            for idx, h in enumerate(hierarchies):
                name = h.get("name", "Hierarchy")
                levels = h.get("levels", [])
                levels_str = " → ".join(f"<span style='font-weight: 600; color: #4f86c6;'>{lvl}</span>" for lvl in levels)
                st.markdown(f"""
                <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-radius: 8px; padding: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 16px;">
                    <div style="font-size: 13px; font-weight: 700; color: #1a1a1a; margin-bottom: 8px;">{name}</div>
                    <div style="font-size: 13px; color: #555; line-height: 1.5;">{levels_str}</div>
                </div>
                """, unsafe_allow_html=True)

    with col2:
        analyses = bm_data.get("suggested_analyses", [])
        if analyses:
            st.markdown("<h3 style='font-size: 18px; font-weight: 700; color: #1a1a1a; margin-bottom: 16px;'>Recommended Analyses</h3>", unsafe_allow_html=True)
            for idx, a in enumerate(analyses):
                title = a.get("title", "Analysis")
                desc = a.get("description", "")
                atype = a.get("type", "diagnostic").upper()
                badge_class = "badge-success" if atype == "DESCRIPTIVE" else ("badge-running" if atype == "DIAGNOSTIC" else "badge-error")
                st.markdown(f"""
                <div style="background-color: #ffffff; border: 1px solid #e8e6e0; border-radius: 8px; padding: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 16px; border-left: 4px solid #7C3AED;">
                    <span class="badge {badge_class}" style="margin-bottom: 8px;">{atype}</span>
                    <div style="font-size: 14px; font-weight: 700; color: #1a1a1a; margin-bottom: 6px;">{title}</div>
                    <div style="font-size: 12px; color: #555; line-height: 1.4;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────────

def _render_main() -> None:

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='omega-header'>"
        "<div>"
        "<div class='omega-logo'>⬡ Omega</div>"
        "<div class='omega-tagline'>Ask anything about your data — in plain English</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── No dataset uploaded ────────────────────────────────────────────────────
    if st.session_state.df is None:
        st.markdown(
            "<div class='empty-state'>"
            "<div class='empty-icon'>📂</div>"
            "<div class='empty-title'>Upload a dataset to get started</div>"
            "<div class='empty-sub'>Supports CSV and Excel files — "
            "no coding required</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Query input ────────────────────────────────────────────────────────────
    st.markdown("<div class='query-label'>What do you want to know?</div>",
                unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            label="Query",
            value=st.session_state.query_input,
            placeholder="e.g. Show me total sales by region, or find the correlation between price and quantity",
            label_visibility="collapsed",
            disabled=st.session_state.is_running,
            key="query_text_input",
        )
    with col_btn:
        run_clicked = st.button(
            "Analyse",
            type="primary",
            width='stretch',
            disabled=st.session_state.is_running or not user_query.strip(),
        )

    # ── Trigger run ────────────────────────────────────────────────────────────
    if run_clicked and user_query.strip():
        _start_crew_run(user_query.strip())
        st.rerun()

    # ── Running state ──────────────────────────────────────────────────────────
    if st.session_state.is_running:
        runner  = st.session_state.runner
        tracker = st.session_state.tracker

        st.markdown("---")
        st.markdown(
            f"<span class='badge badge-running'>⏳ Analysing</span> &nbsp; "
            f"<span style='font-size:13px;color:#555'>{user_query or ''}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        _render_step_tracker()

        # Poll until done
        if runner is not None and runner.is_done:
            _finalise_run()
            st.rerun()
        else:
            time.sleep(0.6)
            st.rerun()
        return

    # ── Error state ────────────────────────────────────────────────────────────
    if st.session_state.error:
        st.error(
            f"Something went wrong during analysis. "
            f"Details: {st.session_state.error}"
        )
        if st.button("Try again"):
            st.session_state.error = None
            st.rerun()
        return

    # ── Results ────────────────────────────────────────────────────────────────
    if st.session_state.current_result:
        result = st.session_state.current_result

        st.markdown(
            f"<span class='badge badge-success'>✓ Done</span> &nbsp; "
            f"<span style='font-size:13px;color:#555'>{result.get('query','')}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        _render_results(result)

    else:
        # Load business model if available
        bm_data = None
        try:
            from src.utils import get_output_path
            bm_path = Path(get_output_path("business_model.json"))
            if bm_path.exists():
                with open(bm_path, "r", encoding="utf-8") as f:
                    bm_data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load business model: {e}")

        if bm_data and st.session_state.df is not None:
            _render_executive_dashboard(bm_data)
        else:
            # Prompt state — dataset loaded but no query yet
            st.markdown(
                "<div class='empty-state' style='padding: 40px 20px'>"
                "<div class='empty-icon'>💬</div>"
                "<div class='empty-title'>Ask your first question</div>"
                "<div class='empty-sub'>Try: "
                "\"Show me total sales by region\" &nbsp;·&nbsp; "
                "\"What is the average order value?\" &nbsp;·&nbsp; "
                "\"Find the top 10 customers\"</div>"
                "</div>",
                unsafe_allow_html=True,
            )


# ── Entry point ────────────────────────────────────────────────────────────────

_render_sidebar()
_render_main()