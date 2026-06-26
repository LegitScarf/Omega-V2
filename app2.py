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

# ── Design System & Custom CSS ─────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Design Tokens ── */
:root {
    --bg-base:        #FAFAFA;
    --bg-surface:     #FFFFFF;
    --bg-subtle:      #F5F5F5;
    --bg-hover:       #F0F0F0;

    --border-subtle:  #EBEBEB;
    --border-default: #E0E0E0;
    --border-strong:  #C8C8C8;

    --text-primary:   #111111;
    --text-secondary: #555555;
    --text-tertiary:  #999999;
    --text-inverse:   #FFFFFF;

    --accent:         #2563EB;
    --accent-light:   #EFF4FF;
    --accent-hover:   #1D4ED8;

    --success:        #16A34A;
    --success-light:  #F0FDF4;
    --warning:        #D97706;
    --warning-light:  #FFFBEB;
    --error:          #DC2626;
    --error-light:    #FEF2F2;
    --purple:         #7C3AED;
    --purple-light:   #F5F3FF;
    --teal:           #0D9488;
    --teal-light:     #F0FDFA;
    --indigo:         #4338CA;
    --indigo-light:   #EEF2FF;
    --orange:         #EA580C;
    --orange-light:   #FFF7ED;

    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  14px;
    --radius-xl:  20px;

    --shadow-sm:  0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:  0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg:  0 8px 24px rgba(0,0,0,0.10), 0 4px 8px rgba(0,0,0,0.04);

    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'DM Mono', 'Fira Code', monospace;
}

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: var(--bg-base);
    font-family: var(--font-sans);
}

[data-testid="stSidebar"] {
    background: var(--bg-surface);
    border-right: 1px solid var(--border-subtle);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 24px;
}

/* hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar Brand ── */
.omega-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 4px 20px 4px;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 20px;
}
.omega-brand-mark {
    width: 36px; height: 36px;
    background: var(--text-primary);
    border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; color: white; font-weight: 700;
    flex-shrink: 0;
}
.omega-brand-name {
    font-size: 16px; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.3px;
}
.omega-brand-sub {
    font-size: 11px; color: var(--text-tertiary);
    font-weight: 400; margin-top: 1px;
}

/* ── Sidebar section labels ── */
.sidebar-label {
    font-size: 10px; font-weight: 700;
    color: var(--text-tertiary); letter-spacing: 0.8px;
    text-transform: uppercase; margin-bottom: 10px;
    padding: 0 2px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--bg-subtle);
    border: 1.5px dashed var(--border-default);
    border-radius: var(--radius-md);
    padding: 4px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

/* ── Dataset status chip ── */
.dataset-chip {
    display: flex; align-items: center; gap: 8px;
    background: var(--success-light);
    border: 1px solid #BBF7D0;
    border-radius: var(--radius-md);
    padding: 10px 14px;
    margin: 12px 0 8px 0;
}
.dataset-chip-dot {
    width: 7px; height: 7px;
    background: var(--success);
    border-radius: 50%; flex-shrink: 0;
}
.dataset-chip-name {
    font-size: 12px; font-weight: 600;
    color: var(--success); flex: 1;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.dataset-meta {
    font-size: 11px; color: var(--text-tertiary);
    padding: 0 2px; margin-bottom: 12px;
    display: flex; gap: 6px; align-items: center;
}
.dataset-meta-sep { color: var(--border-default); }

/* ── Schema pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 14px; }
.pill {
    display: inline-flex; align-items: center; gap: 4px;
    border-radius: 20px; padding: 3px 9px;
    font-size: 10px; font-weight: 600; letter-spacing: 0.2px;
}
.pill-numeric  { background: #EFF4FF; color: #2563EB; }
.pill-category { background: #F5F3FF; color: #7C3AED; }
.pill-datetime { background: #F0FDFA; color: #0D9488; }

/* ── History buttons ── */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 9px 12px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--border-default) !important;
    box-shadow: var(--shadow-sm) !important;
}
.hist-meta {
    font-size: 10px; color: var(--text-tertiary);
    padding: 2px 2px 8px 2px; margin-top: -4px;
    font-family: var(--font-mono);
}

/* ── Main header ── */
.main-header {
    padding: 32px 0 28px 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 32px;
}
.main-wordmark {
    font-size: 26px; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.8px;
    display: flex; align-items: center; gap: 10px;
}
.main-wordmark-hex {
    display: inline-flex; width: 34px; height: 34px;
    background: var(--text-primary); color: white;
    border-radius: 8px; align-items: center;
    justify-content: center; font-size: 18px;
}
.main-tagline {
    font-size: 14px; color: var(--text-tertiary);
    margin-top: 4px; font-weight: 400;
}

/* ── Query bar ── */
.query-wrap {
    background: var(--bg-surface);
    border: 1.5px solid var(--border-default);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: 6px 6px 6px 18px;
    display: flex; align-items: center; gap: 10px;
    transition: border-color 0.2s, box-shadow 0.2s;
    margin-bottom: 20px;
}
.query-wrap:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(37,99,235,0.10), var(--shadow-sm);
}

/* override streamlit text_input inside query-wrap */
.query-wrap [data-testid="stTextInput"] {
    flex: 1;
}
.query-wrap [data-testid="stTextInput"] > div {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.query-wrap [data-testid="stTextInput"] input {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 15px !important;
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
    padding: 8px 0 !important;
}
.query-wrap [data-testid="stTextInput"] input::placeholder {
    color: var(--text-tertiary) !important;
}

/* ── Primary Analyse button ── */
.stButton > button[kind="primary"] {
    background: var(--text-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.15s ease !important;
    letter-spacing: -0.1px !important;
}
.stButton > button[kind="primary"]:hover:not(:disabled) {
    background: #333 !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: var(--border-default) !important;
    color: var(--text-tertiary) !important;
}

/* ── Follow-up chips ── */
.followup-section { margin-top: 28px; }
.followup-heading {
    font-size: 10px; font-weight: 700; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 10px;
}
.stButton > button[data-followup="true"],
div[data-followup-container] .stButton > button {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: 20px !important;
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 7px 14px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
}
.stButton > button[data-followup="true"]:hover,
div[data-followup-container] .stButton > button:hover {
    background: var(--accent-light) !important;
    border-color: #BFDBFE !important;
    color: var(--accent) !important;
}

/* ── Result header ── */
.result-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 24px;
}
.result-query-text {
    font-size: 13px; color: var(--text-secondary);
    flex: 1; font-weight: 400;
}
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 20px; padding: 4px 12px;
    font-size: 11px; font-weight: 600;
    flex-shrink: 0;
}
.badge-success {
    background: var(--success-light);
    color: var(--success);
    border: 1px solid #BBF7D0;
}
.badge-running {
    background: var(--accent-light);
    color: var(--accent);
    border: 1px solid #BFDBFE;
}
.badge-error {
    background: var(--error-light);
    color: var(--error);
    border: 1px solid #FECACA;
}

/* ── Unified result card ── */
.omega-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: 24px 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.omega-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--card-accent, var(--accent));
    border-radius: 3px 0 0 3px;
}
.card-eyebrow {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--card-accent, var(--accent));
    margin-bottom: 8px;
    display: flex; align-items: center; gap: 6px;
}
.card-title {
    font-size: 18px; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.3px;
    margin-bottom: 12px; line-height: 1.3;
}
.card-body {
    font-size: 14px; line-height: 1.75;
    color: var(--text-secondary);
}

/* ── Key metric callout ── */
.metric-callout {
    display: inline-block;
    background: var(--bg-subtle);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 3px 10px;
    font-size: 12px; font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 12px;
    font-family: var(--font-mono);
}

/* ── Stat group ── */
.stat-group {
    display: flex; gap: 32px; flex-wrap: wrap;
    padding: 16px 20px;
    background: var(--bg-subtle);
    border-radius: var(--radius-md);
    margin: 16px 0;
    border: 1px solid var(--border-subtle);
}
.stat-item {}
.stat-label {
    font-size: 10px; font-weight: 700; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px;
}
.stat-value {
    font-size: 22px; font-weight: 700; color: var(--text-primary);
    letter-spacing: -0.5px; font-family: var(--font-mono);
}

/* ── Hypothesis section ── */
.hypothesis-section {
    background: var(--bg-subtle);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin: 14px 0;
}
.hyp-label {
    font-size: 10px; font-weight: 700; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px;
}
.hyp-text { font-size: 13px; color: var(--text-secondary); line-height: 1.5; }
.hyp-divider { border: none; border-top: 1px solid var(--border-subtle); margin: 12px 0; }

/* ── Priority matrix table ── */
.omega-table {
    width: 100%; border-collapse: collapse;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md); overflow: hidden;
    font-size: 13px; margin: 14px 0;
}
.omega-table thead tr {
    background: var(--bg-subtle);
    border-bottom: 1px solid var(--border-default);
}
.omega-table th {
    padding: 10px 14px; font-size: 10px; font-weight: 700;
    color: var(--text-tertiary); text-transform: uppercase;
    letter-spacing: 0.6px; text-align: left;
}
.omega-table td {
    padding: 10px 14px; color: var(--text-secondary);
    border-bottom: 1px solid var(--border-subtle);
    vertical-align: middle;
}
.omega-table tbody tr:last-child td { border-bottom: none; }
.omega-table tbody tr:hover td { background: var(--bg-subtle); }

/* ── Impact/effort badges ── */
.tag {
    display: inline-flex; align-items: center;
    border-radius: 20px; padding: 2px 9px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.2px;
}
.tag-high-impact  { background: #FEF2F2; color: #DC2626; }
.tag-med-impact   { background: #FFFBEB; color: #D97706; }
.tag-low-impact   { background: #F0FDF4; color: #16A34A; }
.tag-high-effort  { background: #FEE2E2; color: #B91C1C; }
.tag-med-effort   { background: #FFEDD5; color: #C2410C; }
.tag-low-effort   { background: #DCFCE7; color: #15803D; }

/* ── Risk box ── */
.risk-box {
    background: var(--warning-light);
    border: 1px solid #FDE68A;
    border-left: 3px solid var(--warning);
    border-radius: var(--radius-md);
    padding: 14px 18px; margin-top: 14px;
}
.risk-box-label {
    font-size: 10px; font-weight: 700; color: var(--warning);
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;
}
.risk-box ul {
    margin: 0; padding-left: 0; list-style: none;
    font-size: 13px; color: #92400E;
}
.risk-box ul li { margin-bottom: 4px; }
.risk-box ul li::before { content: '↳ '; opacity: 0.6; }

/* ── Step tracker ── */
.tracker-wrap {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin: 20px 0;
    box-shadow: var(--shadow-sm);
}
.tracker-title {
    font-size: 12px; font-weight: 700; color: var(--text-tertiary);
    text-transform: uppercase; letter-spacing: 0.6px;
    margin-bottom: 16px;
}
.step-row {
    display: flex; align-items: center; gap: 12px;
    padding: 6px 0; font-size: 13px;
}
.step-icon-wrap {
    width: 24px; height: 24px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 12px;
}
.step-done .step-icon-wrap    { background: var(--success-light); color: var(--success); }
.step-running .step-icon-wrap { background: var(--accent-light); color: var(--accent); }
.step-pending .step-icon-wrap { background: var(--bg-subtle); color: var(--text-tertiary); }
.step-done    .step-label { color: var(--text-primary); font-weight: 500; }
.step-running .step-label { color: var(--accent); font-weight: 600; }
.step-pending .step-label { color: var(--text-tertiary); font-weight: 400; }

/* ── Pulse animation for running step ── */
@keyframes pulse-ring {
    0%   { transform: scale(0.95); opacity: 0.7; }
    50%  { transform: scale(1.08); opacity: 1;   }
    100% { transform: scale(0.95); opacity: 0.7; }
}
.step-running .step-icon-wrap {
    animation: pulse-ring 1.6s ease-in-out infinite;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: var(--accent) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
    background: var(--border-subtle) !important;
    border-radius: 4px !important;
    height: 4px !important;
}

/* ── Truncation notice ── */
.trunc-notice {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: var(--warning);
    background: var(--warning-light);
    border: 1px solid #FDE68A;
    border-radius: var(--radius-sm);
    padding: 8px 14px; margin-bottom: 12px;
    font-weight: 500;
}

/* ── Empty states ── */
.empty-wrap {
    text-align: center; padding: 72px 24px;
}
.empty-icon-ring {
    width: 64px; height: 64px;
    border: 2px dashed var(--border-default);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; margin: 0 auto 16px auto;
    color: var(--text-tertiary);
}
.empty-title {
    font-size: 17px; font-weight: 600;
    color: var(--text-secondary); margin-bottom: 6px;
}
.empty-sub { font-size: 13px; color: var(--text-tertiary); line-height: 1.6; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    font-size: 12px !important; font-weight: 600 !important;
    padding: 8px 16px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.15s ease !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: var(--bg-subtle) !important;
    border-color: var(--border-strong) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: none !important;
}

/* ── Divider ── */
[data-testid="stDivider"] { border-color: var(--border-subtle) !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {
    background-color: var(--accent) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background: var(--accent) !important;
}

</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────

def _init_session_state() -> None:
    defaults = {
        "df":             None,
        "schema_dict":    None,
        "filename":       None,
        "query_input":    "",
        "runner":         None,
        "tracker":        None,
        "is_running":     False,
        "current_result": None,
        "history":        [],
        "error":          None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_results_from_output() -> Dict[str, Any]:
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
    clear_output_dir()

    tracker = OmegaProgressTracker()
    
    # Grab context from history list
    chat_history = []
    if "history" in st.session_state and st.session_state.history:
        # Pass the last 5 turns to keep context window light but highly informative
        chat_history = st.session_state.history[:5]

    runner  = CrewRunner(
        target_fn=run_omega,
        kwargs={
            "user_query":    user_query,
            "dataframe":     st.session_state.df,
            "step_callback": None,
            "task_callback": tracker.on_task_complete,
            "chat_history":  chat_history,
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

    components   = raw["insight"].get("components", [])

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
        "components":   components,
        "raw":          raw,
    }

    st.session_state.current_result = result
    st.session_state.is_running     = False
    st.session_state.history.insert(0, result)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:

        # Brand
        st.markdown("""
        <div class="omega-brand">
            <div class="omega-brand-mark">⬡</div>
            <div>
                <div class="omega-brand-name">Omega</div>
                <div class="omega-brand-sub">Natural language analytics</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Upload ──
        st.markdown("<div class='sidebar-label'>Dataset</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            label="Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)

                if st.session_state.filename != uploaded.name:
                    st.session_state.df          = df
                    st.session_state.filename    = uploaded.name
                    st.session_state.schema_dict = build_schema_dict(df)
                    st.session_state.history     = []
                    st.session_state.current_result = None
                    register_dataset(session_id=uploaded.name, df=df)

            except Exception as exc:
                st.error(f"Could not read file: {exc}")

        # ── Dataset info ──
        if st.session_state.df is not None:
            df     = st.session_state.df
            schema = st.session_state.schema_dict or {}

            fname = st.session_state.filename or "dataset"
            st.markdown(f"""
            <div class="dataset-chip">
                <div class="dataset-chip-dot"></div>
                <div class="dataset-chip-name">{fname}</div>
            </div>
            <div class="dataset-meta">
                <span>{schema.get('row_count', len(df)):,} rows</span>
                <span class="dataset-meta-sep">·</span>
                <span>{schema.get('col_count', len(df.columns))} columns</span>
            </div>
            """, unsafe_allow_html=True)

            numeric_cols     = schema.get("numeric_columns", [])
            categorical_cols = schema.get("categorical_columns", [])
            datetime_cols    = schema.get("datetime_columns", [])

            pills_html = "<div class='pill-row'>"
            for col in numeric_cols:
                pills_html += f"<span class='pill pill-numeric'>📊 {col}</span>"
            for col in categorical_cols:
                pills_html += f"<span class='pill pill-category'>🏷 {col}</span>"
            for col in datetime_cols:
                pills_html += f"<span class='pill pill-datetime'>📅 {col}</span>"
            pills_html += "</div>"
            st.markdown(pills_html, unsafe_allow_html=True)

            with st.expander("Preview data", expanded=False):
                st.dataframe(df.head(5), use_container_width=True, height=180)

            st.divider()

        # ── History ──
        if st.session_state.history:
            st.markdown("<div class='sidebar-label'>Recent queries</div>", unsafe_allow_html=True)
            for i, hist_item in enumerate(st.session_state.history):
                q_short = hist_item["query"][:52] + ("…" if len(hist_item["query"]) > 52 else "")
                metric  = hist_item.get("key_metric", "")[:42]

                if st.button(f"{q_short}", key=f"history_{i}", use_container_width=True, help=hist_item["query"]):
                    st.session_state.current_result = hist_item
                    st.rerun()

                if metric:
                    st.markdown(f"<div class='hist-meta'>{metric}</div>", unsafe_allow_html=True)

            if st.button("Clear history", use_container_width=True):
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

    st.markdown("<div class='tracker-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='tracker-title'>Analysis in progress</div>", unsafe_allow_html=True)
    st.progress(progress)

    icon_map = {"done": "✓", "running": "◉", "pending": "○"}

    for step in status_lines:
        icon  = icon_map.get(step["status"], "○")
        cls   = f"step-{step['status']}"
        label = step["label"]
        st.markdown(
            f"<div class='step-row {cls}'>"
            f"<div class='step-icon-wrap'>{icon}</div>"
            f"<span class='step-label'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ── Results panel ──────────────────────────────────────────────────────────────

def _tag(text: str, level: str, kind: str) -> str:
    css = f"tag-{level.lower()}-{kind}"
    return f"<span class='tag {css}'>{text}</span>"


def _render_results(result: Dict[str, Any]) -> None:

    components = result.get("components", [])

    if components:
        # Render dynamic components layout
        for idx, comp in enumerate(components):
            c_type = comp.get("type")
            if c_type == "markdown":
                content = comp.get("content", "")
                if content:
                    st.markdown(content)
            elif c_type == "metric_grid":
                metrics = comp.get("metrics", [])
                if metrics:
                    cols = st.columns(len(metrics))
                    for m_idx, metric in enumerate(metrics):
                        with cols[m_idx]:
                            label = metric.get("label", "")
                            value = metric.get("value", "")
                            # If value is a float-like string or float, try rounding it to 2 decimal places
                            try:
                                float_val = float(value)
                                if not float_val.is_integer():
                                    value = f"{float_val:.2f}"
                                else:
                                    value = f"{int(float_val)}"
                            except (ValueError, TypeError):
                                pass
                            # Use custom aesthetic container aligned with app2's styling
                            st.markdown(
                                f"""
                                <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 16px; text-align: center; box-shadow: var(--shadow-sm); margin-bottom: 12px;">
                                    <div style="font-size: 11px; font-weight: 600; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">{label}</div>
                                    <div style="font-size: 22px; font-weight: 800; color: var(--text-primary); font-family: var(--font-mono);">{value}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            elif c_type == "table":
                headers = comp.get("headers", [])
                rows = comp.get("rows", [])
                if rows:
                    df_comp = pd.DataFrame(rows, columns=headers if headers else None)
                    st.dataframe(df_comp, use_container_width=True, height=280)
            elif c_type == "chart":
                plotly_spec = comp.get("plotly_spec")
                if plotly_spec:
                    try:
                        # Deep sanitize
                        import copy
                        def deep_sanitize(val):
                            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                                return None
                            elif isinstance(val, dict):
                                # If it's a binary-encoded array from Plotly, decode it
                                if "bdata" in val and "dtype" in val:
                                    import base64
                                    import gzip
                                    try:
                                        bdata = val["bdata"]
                                        dtype = val["dtype"]
                                        raw_bytes = base64.b64decode(bdata)
                                        # Detect gzip magic bytes (0x1f 0x8b)
                                        if raw_bytes.startswith(b"\x1f\x8b"):
                                            raw_bytes = gzip.decompress(raw_bytes)
                                        arr = np.frombuffer(raw_bytes, dtype=dtype)
                                        return arr.tolist()
                                    except Exception:
                                        pass
                                return {k: deep_sanitize(v) for k, v in val.items()}
                            elif isinstance(val, list):
                                return [deep_sanitize(v) for v in val]
                            return val
                        import json
                        spec_to_sanitize = plotly_spec
                        if isinstance(spec_to_sanitize, str):
                            try:
                                spec_to_sanitize = json.loads(spec_to_sanitize)
                            except Exception:
                                pass
                        sanitized_spec = deep_sanitize(copy.deepcopy(spec_to_sanitize))
                        fig = go.Figure(sanitized_spec)
                        fig.update_layout(
                            plot_bgcolor='#FFFFFF',
                            paper_bgcolor='#FAFAFA',
                            font=dict(family='Inter, sans-serif', size=12, color='#555'),
                            margin=dict(l=40, r=40, t=48, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"comp_chart_{idx}")
                    except Exception as e:
                        st.warning(f"Component chart could not be rendered: {e}")
            st.write("") # small spacer

    else:
        # ── 1. Insight card ────────────────────────────────────────────────────────
        insight_text = result.get("insight_text", "")
        key_metric   = result.get("key_metric", "")
        intent_type  = result.get("intent_type", "")
        strategies   = result.get("strategies", [])
        priority_matrix = result.get("priority_matrix", [])
        risks        = result.get("risks", [])

        if insight_text:
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

                # Prescriptive variant
                strat_items = "".join(f"<li style='margin-bottom:6px'>{s}</li>" for s in strategies)

                matrix_rows = ""
                for item in priority_matrix:
                    imp = item.get("impact", "").lower()
                    eff = item.get("effort", "").lower()
                    imp_tag = _tag(imp.capitalize(), imp, "impact")
                    eff_tag = _tag(eff.capitalize(), eff, "effort")
                    matrix_rows += f"""
                    <tr>
                        <td>{item.get('action','')}</td>
                        <td style="text-align:center">{imp_tag}</td>
                        <td style="text-align:center">{eff_tag}</td>
                    </tr>"""

                matrix_html = f"""
                <div style="margin-top:16px">
                    <div class="hyp-label">Implementation Priority Matrix</div>
                    <table class="omega-table">
                        <thead><tr>
                            <th>Proposed Action</th>
                            <th style="width:80px;text-align:center">Impact</th>
                            <th style="width:80px;text-align:center">Effort</th>
                        </tr></thead>
                        <tbody>{matrix_rows}</tbody>
                    </table>
                </div>""" if matrix_rows else ""

                risk_items = "".join(f"<li>{r}</li>" for r in risks)
                risks_html = f"""
                <div class="risk-box">
                    <div class="risk-box-label">Potential Risks & Limitations</div>
                    <ul>{risk_items}</ul>
                </div>""" if risk_items else ""

                metric_chip = f"<div class='metric-callout'>{key_metric}</div>" if key_metric else ""

                pres_html = f"""
                <div class="omega-card" style="--card-accent: var(--orange)">
                    <div class="card-eyebrow">🧭 Prescriptive Strategy</div>
                    {metric_chip}
                    <div class="card-body" style="margin-bottom:14px">{insight_text}</div>
                    <div class="hyp-label" style="margin-top:4px">Actionable Strategies</div>
                    <ul style="margin:6px 0 0 0; padding-left:20px; font-size:14px; color:var(--text-secondary); line-height:1.8">
                        {strat_items}
                    </ul>
                    {matrix_html}
                    {risks_html}
                </div>
                """
                clean_pres_html = "".join(line.strip() for line in pres_html.split("\n"))
                st.markdown(clean_pres_html, unsafe_allow_html=True)

            else:
                # Standard insight
                metric_chip = f"<div class='metric-callout'>{key_metric}</div>" if key_metric else ""
                st.markdown(f"""
                <div class="omega-card" style="--card-accent: var(--accent)">
                    <div class="card-eyebrow">💡 Key Insight</div>
                    {metric_chip}
                    <div class="card-body">{insight_text}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── 1.5. Hypothesis Test ───────────────────────────────────────────────────
    hypothesis = result.get("hypothesis", {})
    if hypothesis and hypothesis.get("status") == "success":
        is_sig    = hypothesis.get("is_significant", False)
        p_val     = hypothesis.get("p_value", 1.0)
        p_str     = f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
        sig_badge = (
            "<span class='badge badge-success'>● Significant</span>"
            if is_sig else
            "<span class='badge badge-error'>● Not Significant</span>"
        )

        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--purple)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>🔬 Statistical Hypothesis Test</span>
                {sig_badge}
            </div>
            <div class="card-title">{hypothesis.get('test_name','')}</div>
            <div class="hypothesis-section">
                <div class="hyp-label">Null Hypothesis (H₀)</div>
                <div class="hyp-text">{hypothesis.get('null_hypothesis','')}</div>
                <hr class="hyp-divider">
                <div class="hyp-label">Alternative Hypothesis (H₁)</div>
                <div class="hyp-text">{hypothesis.get('alternative_hypothesis','')}</div>
            </div>
            <div class="stat-group">
                <div class="stat-item">
                    <div class="stat-label">{hypothesis.get('statistic_name','Statistic')}</div>
                    <div class="stat-value">{hypothesis.get('statistic_value'):,}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">p-value</div>
                    <div class="stat-value">{p_str}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Threshold (α)</div>
                    <div class="stat-value">0.05</div>
                </div>
            </div>
            <div class="card-body" style="border-top:1px solid var(--border-subtle); padding-top:14px; margin-top:4px">
                <strong>Conclusion:</strong> {hypothesis.get('interpretation','')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif hypothesis and hypothesis.get("status") == "failed":
        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--warning)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>🔬 Statistical Hypothesis Test</span>
                <span class="badge badge-error">Unable to Test</span>
            </div>
            <div class="card-body">
                Could not complete the statistical test:
                <strong>{hypothesis.get('message','Unknown error')}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 1.7. Forecast ─────────────────────────────────────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "success":
        time_col  = prediction.get("time_column")
        metric_col = prediction.get("metric_column")
        model_metrics = prediction.get("model_metrics", {})
        r2 = model_metrics.get("r_squared", 0.0)

        if r2 > 0.8:
            acc_badge = "<span class='badge badge-success'>🎯 High Accuracy</span>"
        elif r2 > 0.5:
            acc_badge = "<span class='badge badge-running'>📈 Moderate Accuracy</span>"
        else:
            acc_badge = "<span class='badge badge-error'>⚠ Low Accuracy</span>"

        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--teal)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>🔮 Future Forecast Projection</span>
                {acc_badge}
            </div>
            <div class="card-title">Seasonal Trend: {metric_col} over {time_col}</div>
            <div class="stat-group">
                <div class="stat-item">
                    <div class="stat-label">R-Squared</div>
                    <div class="stat-value">{r2:.4f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Forecast Horizon</div>
                    <div class="stat-value">{len(prediction.get('forecast_values', []))} periods</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Std. Error</div>
                    <div class="stat-value">{model_metrics.get('std_err', 0.0):,.4f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            hist_dates = prediction.get("historical_dates", [])
            hist_values = prediction.get("historical_values", [])
            fc_dates  = prediction.get("forecast_dates", [])
            fc_values = prediction.get("forecast_values", [])
            lower_ci  = prediction.get("lower_bound", [])
            upper_ci  = prediction.get("upper_bound", [])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fc_dates + fc_dates[::-1],
                y=upper_ci + lower_ci[::-1],
                fill='toself', fillcolor='rgba(13,148,136,0.08)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True,
                name="95% Confidence Interval"
            ))
            fig.add_trace(go.Scatter(
                x=hist_dates, y=hist_values,
                mode='lines+markers', name='Historical',
                line=dict(color='#2563EB', width=2.5),
                marker=dict(size=5)
            ))
            fig.add_trace(go.Scatter(
                x=fc_dates, y=fc_values,
                mode='lines+markers', name='Forecast',
                line=dict(color='#EA580C', width=2.5, dash='dash'),
                marker=dict(size=5, symbol='diamond')
            ))
            fig.update_layout(
                xaxis=dict(title=time_col, showgrid=True, gridcolor='#F0F0F0'),
                yaxis=dict(title=metric_col, showgrid=True, gridcolor='#F0F0F0'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40), height=380,
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FAFAFA',
                font=dict(family='Inter, sans-serif', size=12, color='#555'),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Forecast plot could not be rendered: {exc}")

    elif prediction and prediction.get("status") == "failed":
        err_msg = prediction.get('message', 'Unknown error')
        if "regression" in err_msg.lower():
            title_text = "🎛️ Predictive Regression Simulator"
            badge_text = "Regression Failed"
            accent_var = "var(--indigo)"
        elif "classification" in err_msg.lower() or "logistic" in err_msg.lower():
            title_text = "🎛️ Predictive Probability Simulator"
            badge_text = "Classification Failed"
            accent_var = "var(--purple)"
        elif "clustering" in err_msg.lower() or "segment" in err_msg.lower():
            title_text = "🎛️ Spatial Customer Segmentation Playground"
            badge_text = "Clustering Failed"
            accent_var = "var(--indigo)"
        else:
            title_text = "🔮 Future Forecast Projection"
            badge_text = "Forecast Failed"
            accent_var = "var(--warning)"

        st.markdown(f"""
        <div class="omega-card" style="--card-accent: {accent_var}">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>{title_text}</span>
                <span class="badge badge-error">{badge_text}</span>
            </div>
            <div class="card-body">
                We encountered an execution error:
                <strong>{err_msg}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 1.8. Regression Simulator ─────────────────────────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "regression":
        target_col   = prediction.get("target_column")
        intercept    = prediction.get("intercept", 0.0)
        coefficients = prediction.get("coefficients", {})
        features     = prediction.get("features", [])
        metrics      = prediction.get("model_metrics", {})

        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--indigo)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>🎛️ Predictive Regression Simulator</span>
                <span class="badge badge-running" style="background:var(--indigo-light);color:var(--indigo);border-color:#C7D2FE">
                    R² = {metrics.get('r_squared', 0.0):.4f}
                </span>
            </div>
            <div class="card-title">Predicting: {_humanise_column(target_col)}</div>
            <div class="card-body">Adjust the controls below to calculate predictions dynamically.</div>
        </div>
        """, unsafe_allow_html=True)

        slider_vals = {}
        dummy_mappings = prediction.get("dummy_mappings", {})
        cols_slider = st.columns(len(features))

        for idx, feat in enumerate(features):
            name   = feat["name"]
            f_type = feat.get("type", "numeric")
            with cols_slider[idx]:
                if f_type == "numeric":
                    mean_val = float(feat["mean"])
                    min_val  = float(feat["min"])
                    max_val  = float(feat["max"])
                    step     = 1.0 if max_val - min_val > 100 else (0.1 if max_val - min_val > 10 else 0.01)
                    slider_vals[name] = {"type": "numeric", "value": st.slider(
                        label=f"Adjust {_humanise_column(name)}",
                        min_value=min_val, max_value=max_val, value=mean_val,
                        step=step, key=f"slider_{name}"
                    )}
                else:
                    cats = feat.get("categories", [])
                    slider_vals[name] = {"type": "categorical", "value": st.selectbox(
                        label=f"Select {_humanise_column(name)}", options=cats,
                        index=cats.index(feat.get("default")) if feat.get("default") in cats else 0,
                        key=f"select_{name}"
                    )}

        pred_y = intercept
        for name, info in slider_vals.items():
            if info["type"] == "numeric":
                feat = next((f for f in features if f["name"] == name), {})
                mean_val = feat.get("mean", 0.0)
                std_val = feat.get("std", 1.0)
                scaled_val = (info["value"] - mean_val) / std_val
                pred_y += coefficients.get(name, 0.0) * scaled_val
            else:
                cat_dummies = dummy_mappings.get(name, {})
                dummy_col   = cat_dummies.get(info["value"])
                if dummy_col:
                    pred_y += coefficients.get(dummy_col, 0.0)

        y_str = (f"${pred_y:,.2f}" if any(k in target_col.lower() for k in ("price","cost","sales"))
                 else f"{pred_y:,.4f}")

        st.markdown(f"""
        <div style="background:var(--success-light); border:1px solid #BBF7D0;
                    border-radius:var(--radius-md); padding:18px 24px;
                    text-align:center; margin:16px 0">
            <div class="stat-label">Predicted {_humanise_column(target_col)}</div>
            <div style="font-size:36px; font-weight:800; color:var(--text-primary);
                        letter-spacing:-1px; font-family:var(--font-mono)">{y_str}</div>
        </div>
        """, unsafe_allow_html=True)

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
                marker=dict(color='rgba(99,102,241,0.6)', size=6),
                name='Actual vs Predicted'
            ))
            # Diagonal identity line
            min_val = float(min(min(actuals), min(preds)))
            max_val = float(max(max(actuals), max(preds)))
            fig_reg.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                line=dict(color='var(--error)', width=2, dash='dash'),
                name='Perfect Fit'
            ))

            fig_reg.update_layout(
                title=dict(text="Model Diagnostic: Actual vs Predicted Outcomes", font=dict(size=14, family='Inter, sans-serif')),
                xaxis=dict(
                    title=f"Actual {_humanise_column(target_col)}",
                    showgrid=True, gridcolor='var(--border-subtle)',
                    tickfont=dict(color='var(--text-secondary)')
                ),
                yaxis=dict(
                    title=f"Predicted {_humanise_column(target_col)}",
                    showgrid=True, gridcolor='var(--border-subtle)',
                    tickfont=dict(color='var(--text-secondary)')
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=50, b=40),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_reg, use_container_width=True)
        except Exception as reg_plot_exc:
            st.warning(f"Could not render regression fit plot: {reg_plot_exc}")

    # ── 1.9. Classification Simulator ─────────────────────────────────────────
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

            st.markdown(f"""
            <div class="omega-card" style="--card-accent: var(--purple)">
                <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                    <span>🎛️ Predictive Probability Simulator</span>
                    <span class="badge" style="background:var(--purple-light);color:var(--purple);border:1px solid #DDD6FE">
                        Accuracy {metrics.get('accuracy', 0.0)*100:.1f}%
                    </span>
                </div>
                <div class="card-title">Target: {_humanise_column(target_label)}</div>
                <div class="card-body">{class_0_label} vs {class_1_label} — adjust controls to calculate probability dynamically.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            classes = prediction.get("classes", [])
            intercepts = prediction.get("intercepts", {})
            coefficients = prediction.get("coefficients", {})

            st.markdown(f"""
            <div class="omega-card" style="--card-accent: var(--purple)">
                <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                    <span>🎛️ Multi-Class Probability Simulator</span>
                    <span class="badge" style="background:var(--purple-light);color:var(--purple);border:1px solid #DDD6FE">
                        Accuracy {metrics.get('accuracy', 0.0)*100:.1f}%
                    </span>
                </div>
                <div class="card-title">Target: {_humanise_column(target_col)}</div>
                <div class="card-body">{len(classes)} categories — adjust controls to calculate assignment probabilities dynamically.</div>
            </div>
            """, unsafe_allow_html=True)

        slider_vals = {}
        cols_slider = st.columns(len(features))

        for idx, feat in enumerate(features):
            name   = feat["name"]
            f_type = feat.get("type", "numeric")
            with cols_slider[idx]:
                if f_type == "numeric":
                    mean_val = float(feat["mean"])
                    min_val  = float(feat["min"])
                    max_val  = float(feat["max"])
                    step     = 1.0 if max_val - min_val > 100 else (0.1 if max_val - min_val > 10 else 0.01)
                    slider_vals[name] = {"type": "numeric", "value": st.slider(
                        label=f"Adjust {_humanise_column(name)}",
                        min_value=min_val, max_value=max_val, value=mean_val,
                        step=step, key=f"class_slider_{name}"
                    )}
                else:
                    cats = feat.get("categories", [])
                    slider_vals[name] = {"type": "categorical", "value": st.selectbox(
                        label=f"Select {_humanise_column(name)}", options=cats,
                        index=cats.index(feat.get("default")) if feat.get("default") in cats else 0,
                        key=f"class_select_{name}"
                    )}

        if model_mode == "binary":
            z = intercept
            for name, info in slider_vals.items():
                if info["type"] == "numeric":
                    feat = next((f for f in features if f["name"] == name), {})
                    mean_val = feat.get("mean", 0.0)
                    std_val = feat.get("std", 1.0)
                    scaled_val = (info["value"] - mean_val) / std_val
                    z += coefficients.get(name, 0.0) * scaled_val
                else:
                    cat_dummies = dummy_mappings.get(name, {})
                    dummy_col   = cat_dummies.get(info["value"])
                    if dummy_col:
                        z += coefficients.get(dummy_col, 0.0)

            prob_val   = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
            pred_class = class_1_label if prob_val >= 0.5 else class_0_label
            bar_color = "var(--success)" if prob_val >= 0.5 else "var(--error)"

            st.markdown(f"""
            <div style="background:var(--purple-light); border:1px solid #DDD6FE;
                        border-radius:var(--radius-md); padding:20px 24px; margin:16px 0">
                <div style="display:flex; justify-content:space-between; margin-bottom:14px">
                    <div>
                        <div class="stat-label">Predicted Class</div>
                        <div style="font-size:26px; font-weight:800; color:var(--text-primary);
                                    font-family:var(--font-mono)">{pred_class}</div>
                    </div>
                    <div style="text-align:right">
                        <div class="stat-label">Probability ({class_1_label})</div>
                        <div style="font-size:26px; font-weight:800; color:var(--text-primary);
                                    font-family:var(--font-mono)">{prob_val*100:.1f}%</div>
                    </div>
                </div>
                <div style="width:100%; background:var(--border-subtle); border-radius:4px; height:8px; overflow:hidden">
                    <div style="width:{prob_val*100}%; background:{bar_color}; height:100%;
                                border-radius:4px; transition:width 0.3s ease"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
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
            
            prob_bars_list = []
            for class_name, prob_val in sorted_probs:
                prob_bars_list.append(
                    f'<div style="margin-bottom:12px;">'
                    f'<div style="display:flex; justify-content:space-between; font-size:13px; color:var(--text-secondary); font-weight:600; margin-bottom:4px;">'
                    f'<span>{class_name}</span>'
                    f'<span>{prob_val * 100:.1f}%</span>'
                    f'</div>'
                    f'<div style="width:100%; background:var(--border-subtle); border-radius:4px; height:8px; overflow:hidden">'
                    f'<div style="width:{prob_val * 100}%; background:var(--purple); height:100%; border-radius:4px; transition:width 0.3s ease;"></div>'
                    f'</div>'
                    f'</div>'
                )
            prob_bars = "".join(prob_bars_list)
                
            st.markdown(
                f'<div style="background:var(--purple-light); border:1px solid #DDD6FE; border-radius:var(--radius-md); padding:20px 24px; margin:16px 0">'
                f'<div style="margin-bottom:16px;">'
                f'<div class="stat-label">Predicted Class Assignment</div>'
                f'<div style="font-size:28px; font-weight:800; color:var(--text-primary); font-family:var(--font-mono); margin-top:4px;">{pred_class}</div>'
                f'</div>'
                f'{prob_bars}'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── 1.95. Clustering ──────────────────────────────────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "clustering":
        features     = prediction.get("features", [])
        clusters     = prediction.get("clusters", [])
        labels       = prediction.get("labels", [])
        pc_coords    = prediction.get("pc_coords", [])
        sample_size  = prediction.get("sample_size", 0)
        means_dict   = prediction.get("means", {})
        stds_dict    = prediction.get("stds", {})
        dummy_mappings       = prediction.get("dummy_mappings", {})
        feature_names_internal = prediction.get("feature_names_internal", [])
        metrics      = prediction.get("model_metrics", {})

        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--indigo)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>🎛️ Spatial Customer Segmentation</span>
                <span class="badge" style="background:var(--indigo-light);color:var(--indigo);border:1px solid #C7D2FE">
                    K = {metrics.get('clusters_count', 3)} Clusters
                </span>
            </div>
            <div class="card-title">
                {sample_size:,} records partitioned across {len(features)} attributes
            </div>
            <div class="card-body">Adjust controls below to assign new records to a segment.</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            pc_arr  = np.array(pc_coords)
            lbl_arr = np.array(labels)
            fig_3d  = go.Figure()
            for c_info in clusters:
                c_id = c_info["cluster_id"]
                mask = (lbl_arr == c_id)
                pts  = pc_arr[mask]
                if len(pts) > 0:
                    fig_3d.add_trace(go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode='markers',
                        marker=dict(size=4, opacity=0.75),
                        name=f"Segment {c_id} ({c_info['characteristics']})"
                    ))
            fig_3d.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3',
                    bgcolor='#FFFFFF'
                ),
                paper_bgcolor='#FAFAFA',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                font=dict(family='Inter, sans-serif', size=11, color='#555'),
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render cluster cloud: {exc}")

        st.markdown("<div class='sidebar-label' style='margin:16px 0 10px'>Segment Allocation Tool</div>", unsafe_allow_html=True)
        slider_vals = {}
        cols_slider = st.columns(len(features))

        for idx, feat in enumerate(features):
            name   = feat["name"]
            f_type = feat.get("type", "numeric")
            with cols_slider[idx]:
                if f_type == "numeric":
                    mean_val = float(feat["mean"])
                    min_val  = float(feat["min"])
                    max_val  = float(feat["max"])
                    step     = 1.0 if max_val - min_val > 100 else (0.1 if max_val - min_val > 10 else 0.01)
                    slider_vals[name] = {"type": "numeric", "value": st.slider(
                        label=f"Adjust {_humanise_column(name)}",
                        min_value=min_val, max_value=max_val, value=mean_val,
                        step=step, key=f"cluster_slider_{name}"
                    )}
                else:
                    cats = feat.get("categories", [])
                    slider_vals[name] = {"type": "categorical", "value": st.selectbox(
                        label=f"Select {_humanise_column(name)}", options=cats,
                        index=cats.index(feat.get("default")) if feat.get("default") in cats else 0,
                        key=f"cluster_select_{name}"
                    )}

        input_vector = {}
        for name, info in slider_vals.items():
            if info["type"] == "numeric":
                input_vector[name] = info["value"]
            else:
                cat_dummies = dummy_mappings.get(name, {})
                for cat, dummy_col in cat_dummies.items():
                    input_vector[dummy_col] = 1.0 if cat == info["value"] else 0.0

        raw_vals  = [input_vector.get(col, 0.0) for col in feature_names_internal]
        std_vals  = [(raw_vals[i] - means_dict.get(col, 0.0)) / stds_dict.get(col, 1.0)
                     for i, col in enumerate(feature_names_internal)]

        distances = []
        for c_info in clusters:
            dist = np.sqrt(np.sum((np.array(std_vals) - np.array(c_info["centroid"]))**2))
            distances.append((c_info["cluster_id"], dist, c_info["characteristics"]))
        distances.sort(key=lambda x: x[1])

        inv_dists         = [1.0 / (d[1] + 1e-10) for d in distances]
        assigned_confidence = (inv_dists[0] / sum(inv_dists)) * 100

        st.markdown(f"""
        <div style="background:var(--indigo-light); border:1px solid #C7D2FE;
                    border-radius:var(--radius-md); padding:20px 24px; margin:16px 0;
                    display:flex; justify-content:space-between; align-items:center">
            <div>
                <div class="stat-label">Predicted Segment</div>
                <div style="font-size:22px; font-weight:800; color:var(--text-primary);
                            font-family:var(--font-mono)">
                    Segment {distances[0][0]} ({distances[0][2]})
                </div>
            </div>
            <div style="text-align:right">
                <div class="stat-label">Confidence</div>
                <div style="font-size:22px; font-weight:800; color:var(--text-primary);
                            font-family:var(--font-mono)">{assigned_confidence:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 2. Chart ───────────────────────────────────────────────────────────────
    chart_spec = result.get("chart_spec")
    chart_gen  = result.get("chart_gen", False)
    has_forecast = result.get("prediction", {}).get("status") == "success"
    has_regression = result.get("prediction", {}).get("status") == "regression"
    has_classification = result.get("prediction", {}).get("status") == "classification"
    has_clustering = result.get("prediction", {}).get("status") == "clustering"

    if chart_gen and chart_spec and not has_forecast and not has_regression and not has_classification and not has_clustering:
        try:
            import copy
            def deep_sanitize(val):
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    return None
                elif isinstance(val, dict):
                    if "bdata" in val and "dtype" in val:
                        import base64
                        import gzip
                        try:
                            bdata = val["bdata"]
                            dtype = val["dtype"]
                            raw_bytes = base64.b64decode(bdata)
                            if raw_bytes.startswith(b"\x1f\x8b"):
                                raw_bytes = gzip.decompress(raw_bytes)
                            arr = np.frombuffer(raw_bytes, dtype=dtype)
                            return arr.tolist()
                        except Exception:
                            pass
                    return {k: deep_sanitize(v) for k, v in val.items()}
                elif isinstance(val, list):
                    return [deep_sanitize(v) for v in val]
                return val
            import json
            spec_to_sanitize = chart_spec
            if isinstance(spec_to_sanitize, str):
                try:
                    spec_to_sanitize = json.loads(spec_to_sanitize)
                except Exception:
                    pass
            sanitized_spec = deep_sanitize(copy.deepcopy(spec_to_sanitize))
            fig = go.Figure(sanitized_spec)
            fig.update_layout(
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FAFAFA',
                font=dict(family='Inter, sans-serif', size=12, color='#555'),
                margin=dict(l=40, r=40, t=48, b=40),
            )
            st.plotly_chart(fig, use_container_width=True, key="main_layout_chart")
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
                f"⚠ Showing top 500 of {row_count:,} rows — refine your query to see a specific subset."
                f"</div>",
                unsafe_allow_html=True,
            )
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True, height=280)
        csv = result_df.to_csv(index=False).encode("utf-8")
        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            st.download_button(
                label="⬇ Download results as CSV",
                data=csv, file_name="omega_results.csv", mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            from src.report import generate_pdf_report
            try:
                pdf_buffer = generate_pdf_report(result)
                st.download_button(
                    label="📄 Download Report as PDF",
                    data=pdf_buffer, file_name="omega_report.pdf", mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")

    # ── 4. Follow-ups ──────────────────────────────────────────────────────────
    follow_ups = result.get("follow_ups", [])
    if follow_ups:
        st.markdown(
            "<div class='followup-heading' style='margin-top:28px'>You might also ask</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(follow_ups))
        for i, suggestion in enumerate(follow_ups):
            with cols[i]:
                if st.button(suggestion, key=f"followup_{i}_{hash(suggestion)}", use_container_width=True):
                    _start_crew_run(suggestion)
                    st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────

def _render_main() -> None:

    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-wordmark">
            <span class="main-wordmark-hex">⬡</span>
            Omega
        </div>
        <div class="main-tagline">Ask anything about your data — in plain English</div>
    </div>
    """, unsafe_allow_html=True)

    # No dataset
    if st.session_state.df is None:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-icon-ring">📂</div>
            <div class="empty-title">Upload a dataset to get started</div>
            <div class="empty-sub">Supports CSV and Excel files<br>No code required — just ask your question</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Query bar
    st.markdown("<div class='query-label' style='font-size:12px;font-weight:600;color:var(--text-tertiary);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.6px'>Ask a question</div>", unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            label="Query",
            value=st.session_state.query_input,
            placeholder="e.g. Show total sales by region, or find top 10 customers by revenue",
            label_visibility="collapsed",
            disabled=st.session_state.is_running,
            key="query_text_input",
        )
    with col_btn:
        run_clicked = st.button(
            "Analyse →",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_running or not user_query.strip(),
        )

    if run_clicked and user_query.strip():
        _start_crew_run(user_query.strip())
        st.rerun()

    # Running state
    if st.session_state.is_running:
        runner  = st.session_state.runner
        tracker = st.session_state.tracker

        st.markdown(
            f"<div style='margin:12px 0 4px 0'>"
            f"<span class='badge badge-running'>⏳ Analysing</span>"
            f"&nbsp; <span style='font-size:13px;color:var(--text-tertiary)'>{user_query or ''}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        _render_step_tracker()

        if runner is not None and runner.is_done:
            _finalise_run()
            st.rerun()
        else:
            time.sleep(0.6)
            st.rerun()
        return

    # Error state
    if st.session_state.error:
        st.error(f"Something went wrong: {st.session_state.error}")
        if st.button("Try again"):
            st.session_state.error = None
            st.rerun()
        return

    # Results
    if st.session_state.current_result:
        result = st.session_state.current_result
        st.markdown(
            f"<div class='result-header'>"
            f"<span class='badge badge-success'>✓ Done</span>"
            f"<span class='result-query-text'>{result.get('query','')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border:none;border-top:1px solid var(--border-subtle);margin-bottom:24px'>",
                    unsafe_allow_html=True)
        _render_results(result)

    else:
        # Prompt state
        st.markdown("""
        <div class="empty-wrap" style="padding:48px 24px">
            <div class="empty-icon-ring">💬</div>
            <div class="empty-title">Ask your first question</div>
            <div class="empty-sub">
                Try: "Show me total sales by region"<br>
                or "What is the average order value?"<br>
                or "Find the top 10 customers"
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Entry point ────────────────────────────────────────────────────────────────

_render_sidebar()
_render_main()