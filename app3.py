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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ────────────────────────────────────────────────
   DESIGN TOKENS
──────────────────────────────────────────────── */
:root {
    --white:          #FFFFFF;
    --off-white:      #FAFAFA;
    --gray-50:        #F7F7F7;
    --gray-100:       #F0F0F0;
    --gray-150:       #E8E8E8;
    --gray-200:       #E0E0E0;
    --gray-300:       #CACACA;
    --gray-400:       #AAAAAA;
    --gray-500:       #888888;
    --gray-600:       #666666;
    --gray-700:       #444444;
    --gray-800:       #222222;
    --gray-900:       #111111;

    --ink:            #0A0A0A;
    --ink-muted:      #525252;
    --ink-faint:      #9B9B9B;

    /* Accent — electric indigo */
    --accent:         #4F46E5;
    --accent-mid:     #6366F1;
    --accent-light:   #EEF2FF;
    --accent-border:  #C7D2FE;
    --accent-glow:    rgba(79,70,229,0.12);

    /* Semantic */
    --emerald:        #059669;
    --emerald-light:  #ECFDF5;
    --emerald-border: #A7F3D0;
    --amber:          #D97706;
    --amber-light:    #FFFBEB;
    --amber-border:   #FDE68A;
    --rose:           #E11D48;
    --rose-light:     #FFF1F2;
    --rose-border:    #FECDD3;
    --violet:         #7C3AED;
    --violet-light:   #F5F3FF;
    --violet-border:  #DDD6FE;
    --teal:           #0D9488;
    --teal-light:     #F0FDFA;
    --teal-border:    #99F6E4;
    --orange:         #EA580C;
    --orange-light:   #FFF7ED;
    --orange-border:  #FED7AA;
    --cyan:           #0891B2;
    --cyan-light:     #ECFEFF;
    --cyan-border:    #A5F3FC;

    --radius-xs:  4px;
    --radius-sm:  8px;
    --radius-md:  12px;
    --radius-lg:  16px;
    --radius-xl:  24px;
    --radius-2xl: 32px;

    --shadow-xs:  0 1px 2px rgba(0,0,0,0.04);
    --shadow-sm:  0 1px 4px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04);
    --shadow-md:  0 4px 16px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04);
    --shadow-lg:  0 8px 32px rgba(0,0,0,0.10), 0 2px 8px rgba(0,0,0,0.04);
    --shadow-xl:  0 16px 48px rgba(0,0,0,0.12), 0 4px 16px rgba(0,0,0,0.06);

    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ────────────────────────────────────────────────
   KEYFRAME ANIMATIONS
──────────────────────────────────────────────── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}
@keyframes spin-ring {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes progress-pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}
@keyframes card-in {
    from { opacity: 0; transform: translateY(8px) scale(0.995); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes border-glow {
    0%, 100% { box-shadow: 0 0 0 0 var(--accent-glow); }
    50%       { box-shadow: 0 0 0 6px var(--accent-glow); }
}

/* ────────────────────────────────────────────────
   GLOBAL RESET & BASE
──────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--white) !important;
    font-family: var(--font-sans);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
}

/* Streamlit's main padding */
.main .block-container {
    padding-top: 0 !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1120px !important;
}

/* Hide chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ────────────────────────────────────────────────
   SIDEBAR
──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--gray-150) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 28px 20px 24px 20px;
}

/* ── Brand mark ── */
.omega-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 24px;
    margin-bottom: 24px;
    border-bottom: 1px solid var(--gray-150);
}
.omega-logo {
    width: 38px;
    height: 38px;
    background: var(--ink);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    color: white;
    flex-shrink: 0;
    position: relative;
    overflow: hidden;
}
.omega-logo::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 60%);
}
.omega-brand-text {}
.omega-brand-name {
    font-size: 15px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.4px;
    line-height: 1.1;
}
.omega-brand-sub {
    font-size: 11px;
    color: var(--ink-faint);
    margin-top: 2px;
    font-weight: 400;
}

/* ── Sidebar section label ── */
.s-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--ink-faint);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding: 0 2px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--gray-50) !important;
    border: 1.5px dashed var(--gray-200) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-mid) !important;
    background: var(--accent-light) !important;
}
[data-testid="stFileUploader"] label {
    font-size: 12px !important;
    color: var(--ink-muted) !important;
}

/* ── Dataset chip ── */
.ds-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--emerald-light);
    border: 1px solid var(--emerald-border);
    border-radius: var(--radius-md);
    padding: 11px 14px;
    margin: 14px 0 8px 0;
    animation: fadeSlideUp 0.3s ease;
}
.ds-dot {
    width: 8px;
    height: 8px;
    background: var(--emerald);
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse-dot 2s ease-in-out infinite;
}
.ds-name {
    font-size: 12px;
    font-weight: 600;
    color: var(--emerald);
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ds-meta {
    font-size: 11px;
    color: var(--ink-faint);
    padding: 0 2px;
    margin-bottom: 14px;
    display: flex;
    gap: 6px;
    align-items: center;
}
.ds-sep { color: var(--gray-300); }

/* ── Column pills ── */
.pill-row { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 16px; }
.pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border-radius: 20px;
    padding: 3px 9px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1px;
    transition: transform 0.15s;
}
.pill:hover { transform: translateY(-1px); }
.pill-n { background: var(--accent-light);  color: var(--accent); }
.pill-c { background: var(--violet-light);  color: var(--violet); }
.pill-d { background: var(--teal-light);    color: var(--teal); }

/* ── History buttons ── */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: var(--gray-50) !important;
    border: 1px solid var(--gray-150) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 9px 12px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    font-family: var(--font-sans) !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    background: var(--gray-100) !important;
    border-color: var(--gray-300) !important;
    transform: translateX(2px) !important;
}
.hist-meta {
    font-size: 10px;
    color: var(--ink-faint);
    padding: 2px 2px 10px 2px;
    margin-top: -4px;
    font-family: var(--font-mono);
}

/* ────────────────────────────────────────────────
   MAIN HEADER
──────────────────────────────────────────────── */
.main-header {
    padding: 40px 0 32px 0;
    border-bottom: 1px solid var(--gray-150);
    margin-bottom: 36px;
    animation: fadeIn 0.5s ease;
}
.header-eyebrow {
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.header-eyebrow::before {
    content: '';
    display: inline-block;
    width: 20px;
    height: 1.5px;
    background: var(--accent);
}
.main-wordmark {
    font-size: 32px;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -1.2px;
    line-height: 1;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.wordmark-hex {
    width: 40px;
    height: 40px;
    background: var(--ink);
    color: white;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}
.main-tagline {
    font-size: 15px;
    color: var(--ink-muted);
    font-weight: 400;
    line-height: 1.5;
}

/* ────────────────────────────────────────────────
   QUERY BAR
──────────────────────────────────────────────── */
.query-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--ink-faint);
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* Streamlit text_input overrides */
[data-testid="stTextInput"] > div > div > input {
    font-size: 15px !important;
    font-family: var(--font-sans) !important;
    color: var(--ink) !important;
    border: 1.5px solid var(--gray-200) !important;
    border-radius: var(--radius-md) !important;
    background: var(--white) !important;
    padding: 12px 16px !important;
    height: 48px !important;
    box-shadow: var(--shadow-xs) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
[data-testid="stTextInput"] > div > div > input::placeholder {
    color: var(--gray-400) !important;
}

/* ── Analyse button ── */
.stButton > button[kind="primary"] {
    background: var(--ink) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-sans) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0 24px !important;
    height: 48px !important;
    letter-spacing: -0.2px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button[kind="primary"]::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, transparent 60%);
}
.stButton > button[kind="primary"]:hover:not(:disabled) {
    background: var(--gray-800) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active:not(:disabled) {
    transform: translateY(0) !important;
    box-shadow: var(--shadow-xs) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: var(--gray-200) !important;
    color: var(--gray-400) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ────────────────────────────────────────────────
   STATUS BADGES
──────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1px;
    flex-shrink: 0;
}
.badge-ok {
    background: var(--emerald-light);
    color: var(--emerald);
    border: 1px solid var(--emerald-border);
}
.badge-running {
    background: var(--accent-light);
    color: var(--accent);
    border: 1px solid var(--accent-border);
    animation: progress-pulse 2s ease-in-out infinite;
}
.badge-err {
    background: var(--rose-light);
    color: var(--rose);
    border: 1px solid var(--rose-border);
}

/* ────────────────────────────────────────────────
   RESULT HEADER
──────────────────────────────────────────────── */
.result-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
    animation: fadeSlideUp 0.4s ease;
}
.result-query-text {
    font-size: 13px;
    color: var(--ink-muted);
    flex: 1;
}

/* ────────────────────────────────────────────────
   OMEGA CARDS — universal result container
──────────────────────────────────────────────── */
.omega-card {
    background: var(--white);
    border: 1px solid var(--gray-150);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: 24px 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    animation: card-in 0.4s ease;
    transition: box-shadow 0.2s, border-color 0.2s;
}
.omega-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--gray-200);
}
/* Accent left-rail */
.omega-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 3px;
    height: 100%;
    background: var(--card-accent, var(--accent));
    border-radius: 3px 0 0 3px;
}
/* Top sheen */
.omega-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent);
}

.card-eyebrow {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--card-accent, var(--accent));
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.card-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.5px;
    margin-bottom: 14px;
    line-height: 1.3;
}
.card-body {
    font-size: 14px;
    line-height: 1.8;
    color: var(--ink-muted);
}

/* Metric monospace chip */
.metric-callout {
    display: inline-block;
    background: var(--gray-50);
    border: 1px solid var(--gray-150);
    border-radius: var(--radius-sm);
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--ink-muted);
    margin-bottom: 14px;
    font-family: var(--font-mono);
    letter-spacing: -0.2px;
}

/* ────────────────────────────────────────────────
   STAT GROUP
──────────────────────────────────────────────── */
.stat-group {
    display: flex;
    gap: 0;
    flex-wrap: wrap;
    margin: 18px 0;
    border: 1px solid var(--gray-150);
    border-radius: var(--radius-md);
    overflow: hidden;
}
.stat-item {
    flex: 1;
    padding: 16px 20px;
    background: var(--gray-50);
    border-right: 1px solid var(--gray-150);
}
.stat-item:last-child { border-right: none; }
.stat-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.8px;
    font-family: var(--font-mono);
}

/* ────────────────────────────────────────────────
   HYPOTHESIS SECTION
──────────────────────────────────────────────── */
.hypothesis-section {
    background: var(--gray-50);
    border: 1px solid var(--gray-150);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin: 16px 0;
}
.hyp-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.hyp-text {
    font-size: 13px;
    color: var(--ink-muted);
    line-height: 1.6;
}
.hyp-divider {
    border: none;
    border-top: 1px solid var(--gray-150);
    margin: 14px 0;
}

/* ────────────────────────────────────────────────
   PRIORITY / TABLE
──────────────────────────────────────────────── */
.omega-table {
    width: 100%;
    border-collapse: collapse;
    border: 1px solid var(--gray-150);
    border-radius: var(--radius-md);
    overflow: hidden;
    font-size: 13px;
    margin: 16px 0;
}
.omega-table thead tr {
    background: var(--gray-50);
    border-bottom: 1px solid var(--gray-200);
}
.omega-table th {
    padding: 11px 16px;
    font-size: 10px;
    font-weight: 700;
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    text-align: left;
}
.omega-table td {
    padding: 11px 16px;
    color: var(--ink-muted);
    border-bottom: 1px solid var(--gray-100);
    vertical-align: middle;
}
.omega-table tbody tr:last-child td { border-bottom: none; }
.omega-table tbody tr {
    transition: background 0.1s;
}
.omega-table tbody tr:hover td {
    background: var(--gray-50);
    color: var(--ink);
}

/* ── Impact/effort tags ── */
.tag {
    display: inline-flex;
    align-items: center;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.tag-high-impact  { background: var(--rose-light);    color: var(--rose); }
.tag-med-impact   { background: var(--amber-light);   color: var(--amber); }
.tag-low-impact   { background: var(--emerald-light); color: var(--emerald); }
.tag-high-effort  { background: var(--rose-light);    color: var(--rose); }
.tag-med-effort   { background: var(--amber-light);   color: var(--amber); }
.tag-low-effort   { background: var(--emerald-light); color: var(--emerald); }

/* ── Risk box ── */
.risk-box {
    background: var(--amber-light);
    border: 1px solid var(--amber-border);
    border-left: 3px solid var(--amber);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-top: 16px;
}
.risk-box-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--amber);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
}
.risk-box ul {
    margin: 0;
    padding-left: 0;
    list-style: none;
    font-size: 13px;
    color: #78350F;
}
.risk-box ul li { margin-bottom: 5px; }
.risk-box ul li::before { content: '↳ '; opacity: 0.5; }

/* ────────────────────────────────────────────────
   STEP TRACKER
──────────────────────────────────────────────── */
.tracker-wrap {
    background: var(--white);
    border: 1px solid var(--gray-150);
    border-radius: var(--radius-lg);
    padding: 24px 28px;
    margin: 24px 0;
    box-shadow: var(--shadow-sm);
    animation: fadeSlideUp 0.3s ease;
}
.tracker-title {
    font-size: 11px;
    font-weight: 700;
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 20px;
}

/* Custom progress bar */
.omega-progress-track {
    height: 3px;
    background: var(--gray-150);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 20px;
}
.omega-progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
    position: relative;
}
.omega-progress-fill::after {
    content: '';
    position: absolute;
    right: 0;
    top: 0;
    height: 100%;
    width: 40px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6));
    animation: shimmer 1.5s ease-in-out infinite;
    background-size: 200% 100%;
}

.step-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
    font-size: 13px;
    transition: opacity 0.2s;
}
.step-pending { opacity: 0.45; }

.step-icon-wrap {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 11px;
    font-weight: 700;
    transition: all 0.3s;
}
.step-done .step-icon-wrap {
    background: var(--emerald-light);
    color: var(--emerald);
    border: 1px solid var(--emerald-border);
}
.step-running .step-icon-wrap {
    background: var(--accent-light);
    color: var(--accent);
    border: 1px solid var(--accent-border);
    animation: border-glow 2s ease-in-out infinite;
}
.step-pending .step-icon-wrap {
    background: var(--gray-50);
    color: var(--gray-300);
    border: 1px solid var(--gray-150);
}

.step-done .step-label    { color: var(--ink); font-weight: 500; }
.step-running .step-label { color: var(--accent); font-weight: 600; }
.step-pending .step-label { color: var(--gray-400); font-weight: 400; }

/* Spinning icon for running */
.spin-ring {
    width: 12px;
    height: 12px;
    border: 2px solid var(--accent-border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin-ring 0.8s linear infinite;
    display: inline-block;
}

/* ────────────────────────────────────────────────
   FOLLOW-UP CHIPS
──────────────────────────────────────────────── */
.followup-heading {
    font-size: 10px;
    font-weight: 700;
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    margin-top: 32px;
}
div[data-followup-container] .stButton > button,
.stButton > button[data-followup="true"] {
    background: var(--white) !important;
    border: 1px solid var(--gray-200) !important;
    border-radius: 20px !important;
    color: var(--ink-muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 7px 16px !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all 0.15s ease !important;
    font-family: var(--font-sans) !important;
}
div[data-followup-container] .stButton > button:hover,
.stButton > button[data-followup="true"]:hover {
    background: var(--accent-light) !important;
    border-color: var(--accent-border) !important;
    color: var(--accent) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ────────────────────────────────────────────────
   EMPTY STATES
──────────────────────────────────────────────── */
.empty-wrap {
    text-align: center;
    padding: 80px 24px;
    animation: fadeIn 0.5s ease;
}
.empty-ring {
    width: 72px;
    height: 72px;
    border: 1.5px dashed var(--gray-200);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin: 0 auto 20px auto;
    color: var(--gray-300);
    background: var(--gray-50);
}
.empty-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.3px;
    margin-bottom: 8px;
}
.empty-sub {
    font-size: 14px;
    color: var(--ink-faint);
    line-height: 1.7;
    max-width: 360px;
    margin: 0 auto;
}

/* ── Example query pills ── */
.example-queries {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin-top: 24px;
}
.eq-pill {
    display: inline-block;
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: 20px;
    padding: 7px 16px;
    font-size: 12px;
    color: var(--ink-muted);
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}
.eq-pill:hover {
    background: var(--accent-light);
    border-color: var(--accent-border);
    color: var(--accent);
}

/* ────────────────────────────────────────────────
   TRUNCATION NOTICE
──────────────────────────────────────────────── */
.trunc-notice {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 12px;
    color: var(--amber);
    background: var(--amber-light);
    border: 1px solid var(--amber-border);
    border-radius: var(--radius-sm);
    padding: 9px 14px;
    margin-bottom: 14px;
    font-weight: 500;
}

/* ────────────────────────────────────────────────
   DATAFRAME
──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--gray-150) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-xs) !important;
}

/* ────────────────────────────────────────────────
   DOWNLOAD BUTTON
──────────────────────────────────────────────── */
[data-testid="stDownloadButton"] button {
    background: var(--white) !important;
    border: 1px solid var(--gray-200) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--ink-muted) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 9px 18px !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all 0.15s ease !important;
    font-family: var(--font-sans) !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: var(--gray-50) !important;
    border-color: var(--gray-300) !important;
    box-shadow: var(--shadow-sm) !important;
    color: var(--ink) !important;
}

/* ────────────────────────────────────────────────
   EXPANDER
──────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--gray-150) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: none !important;
    transition: border-color 0.2s !important;
}
[data-testid="stExpander"]:hover {
    border-color: var(--gray-300) !important;
}

/* ────────────────────────────────────────────────
   STREAMLIT PROGRESS OVERRIDE
──────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: var(--accent) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
    background: var(--gray-150) !important;
    border-radius: 4px !important;
    height: 3px !important;
}

/* ────────────────────────────────────────────────
   STREAMLIT SLIDER
──────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {
    background-color: var(--accent) !important;
}

/* ────────────────────────────────────────────────
   STREAMLIT SELECT / RADIO
──────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    border: 1px solid var(--gray-200) !important;
    border-radius: var(--radius-sm) !important;
}

/* ────────────────────────────────────────────────
   DIVIDER
──────────────────────────────────────────────── */
[data-testid="stDivider"] { border-color: var(--gray-150) !important; }
hr { border-color: var(--gray-150) !important; }

/* ────────────────────────────────────────────────
   METRIC GRID (component type)
──────────────────────────────────────────────── */
.m-card {
    background: var(--white);
    border: 1px solid var(--gray-150);
    border-radius: var(--radius-md);
    padding: 18px 20px;
    text-align: center;
    box-shadow: var(--shadow-xs);
    margin-bottom: 12px;
    transition: all 0.2s;
    animation: card-in 0.4s ease both;
}
.m-card:hover {
    border-color: var(--gray-300);
    box-shadow: var(--shadow-sm);
    transform: translateY(-2px);
}
.m-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--ink-faint);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}
.m-value {
    font-size: 26px;
    font-weight: 800;
    color: var(--ink);
    font-family: var(--font-mono);
    letter-spacing: -1px;
}

/* ────────────────────────────────────────────────
   PREDICTION / SIMULATOR COMPONENTS
──────────────────────────────────────────────── */
.pred-output {
    border-radius: var(--radius-md);
    padding: 20px 24px;
    margin: 16px 0;
    animation: card-in 0.35s ease;
}
.pred-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.pred-value {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -1.5px;
    font-family: var(--font-mono);
}

/* Probability bar */
.prob-bar-wrap {
    width: 100%;
    background: var(--gray-150);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-top: 14px;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s cubic-bezier(0.4,0,0.2,1);
}

/* ────────────────────────────────────────────────
   RUNNING STATE HEADER
──────────────────────────────────────────────── */
.running-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 20px;
    background: var(--accent-light);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-md);
    margin: 16px 0;
    animation: fadeSlideUp 0.3s ease;
}
.running-text {
    font-size: 13px;
    color: var(--accent);
    font-weight: 500;
    flex: 1;
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
    
    chat_history = []
    if "history" in st.session_state and st.session_state.history:
        chat_history = st.session_state.history[:5]

    runner = CrewRunner(
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

    insight_text    = (raw["insight"].get("insight_text") or "Analysis complete — see results below.")
    key_metric      = raw["insight"].get("key_metric", "")
    follow_ups      = raw["insight"].get("follow_up_suggestions", [])
    intent_type     = raw["insight"].get("intent_type", "")
    strategies      = raw["insight"].get("strategies", [])
    priority_matrix = raw["insight"].get("priority_matrix", [])
    risks           = raw["insight"].get("risks", [])
    chart_spec      = raw["chart"].get("plotly_spec")
    chart_type      = raw["chart"].get("chart_type", "")
    chart_title     = raw["chart"].get("chart_title", "")
    chart_gen       = raw["chart"].get("chart_generated", False)
    rows            = raw["query"].get("result_rows", [])
    truncated       = raw["query"].get("truncated", False)
    row_count       = raw["query"].get("row_count", 0)
    hypothesis      = raw["hypothesis"]
    prediction      = raw["prediction"]
    components      = raw["insight"].get("components", [])

    result = {
        "query":          runner._kwargs.get("user_query", ""),
        "insight_text":   insight_text,
        "key_metric":     key_metric,
        "follow_ups":     follow_ups,
        "intent_type":    intent_type,
        "strategies":     strategies,
        "priority_matrix": priority_matrix,
        "risks":          risks,
        "chart_spec":     chart_spec,
        "chart_type":     chart_type,
        "chart_title":    chart_title,
        "chart_gen":      chart_gen,
        "rows":           rows,
        "row_count":      row_count,
        "truncated":      truncated,
        "hypothesis":     hypothesis,
        "prediction":     prediction,
        "components":     components,
        "raw":            raw,
    }

    st.session_state.current_result = result
    st.session_state.is_running     = False
    st.session_state.history.insert(0, result)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div class="omega-brand">
            <div class="omega-logo">⬡</div>
            <div class="omega-brand-text">
                <div class="omega-brand-name">Omega</div>
                <div class="omega-brand-sub">Natural language analytics</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='s-label'>Dataset</div>", unsafe_allow_html=True)
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

        if st.session_state.df is not None:
            df     = st.session_state.df
            schema = st.session_state.schema_dict or {}

            fname = st.session_state.filename or "dataset"
            st.markdown(f"""
            <div class="ds-chip">
                <div class="ds-dot"></div>
                <div class="ds-name">{fname}</div>
            </div>
            <div class="ds-meta">
                <span>{schema.get('row_count', len(df)):,} rows</span>
                <span class="ds-sep">·</span>
                <span>{schema.get('col_count', len(df.columns))} columns</span>
            </div>
            """, unsafe_allow_html=True)

            numeric_cols     = schema.get("numeric_columns", [])
            categorical_cols = schema.get("categorical_columns", [])
            datetime_cols    = schema.get("datetime_columns", [])

            pills_html = "<div class='pill-row'>"
            for col in numeric_cols:
                pills_html += f"<span class='pill pill-n'>#{col}</span>"
            for col in categorical_cols:
                pills_html += f"<span class='pill pill-c'>@{col}</span>"
            for col in datetime_cols:
                pills_html += f"<span class='pill pill-d'>⏱ {col}</span>"
            pills_html += "</div>"
            st.markdown(pills_html, unsafe_allow_html=True)

            with st.expander("Preview data", expanded=False):
                st.dataframe(df.head(5), use_container_width=True, height=180)

            st.divider()

        if st.session_state.history:
            st.markdown("<div class='s-label'>Recent queries</div>", unsafe_allow_html=True)
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
    pct          = int(progress * 100)

    st.markdown(f"""
    <div class="tracker-wrap">
        <div class="tracker-title">Analysis in progress</div>
        <div class="omega-progress-track">
            <div class="omega-progress-fill" style="width:{pct}%"></div>
        </div>
    """, unsafe_allow_html=True)

    for step in status_lines:
        status = step["status"]
        label  = step["label"]

        if status == "done":
            icon = "✓"
        elif status == "running":
            icon = '<span class="spin-ring"></span>'
        else:
            icon = "○"

        st.markdown(
            f"<div class='step-row step-{status}'>"
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


def _render_chart(chart_spec, chart_key="chart"):
    """Sanitize and render a Plotly chart spec."""
    try:
        import copy, json, base64, gzip

        def deep_sanitize(val):
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return None
            elif isinstance(val, dict):
                if "bdata" in val and "dtype" in val:
                    try:
                        raw_bytes = base64.b64decode(val["bdata"])
                        if raw_bytes.startswith(b"\x1f\x8b"):
                            raw_bytes = gzip.decompress(raw_bytes)
                        return np.frombuffer(raw_bytes, dtype=val["dtype"]).tolist()
                    except Exception:
                        pass
                return {k: deep_sanitize(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [deep_sanitize(v) for v in val]
            return val

        spec = chart_spec
        if isinstance(spec, str):
            spec = json.loads(spec)

        sanitized = deep_sanitize(copy.deepcopy(spec))
        fig = go.Figure(sanitized)
        fig.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FAFAFA',
            font=dict(family='Inter, -apple-system, sans-serif', size=12, color='#525252'),
            margin=dict(l=44, r=44, t=52, b=44),
            xaxis=dict(gridcolor='#F0F0F0', linecolor='#E0E0E0', zerolinecolor='#E0E0E0'),
            yaxis=dict(gridcolor='#F0F0F0', linecolor='#E0E0E0', zerolinecolor='#E0E0E0'),
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
    except Exception as e:
        st.warning(f"Chart could not be rendered: {e}")


def _render_results(result: Dict[str, Any]) -> None:

    components = result.get("components", [])

    if components:
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
                            try:
                                fv = float(value)
                                value = f"{int(fv)}" if fv.is_integer() else f"{fv:.2f}"
                            except (ValueError, TypeError):
                                pass
                            st.markdown(f"""
                            <div class="m-card" style="animation-delay:{m_idx * 0.06}s">
                                <div class="m-label">{label}</div>
                                <div class="m-value">{value}</div>
                            </div>
                            """, unsafe_allow_html=True)
            elif c_type == "table":
                headers = comp.get("headers", [])
                rows    = comp.get("rows", [])
                if rows:
                    df_comp = pd.DataFrame(rows, columns=headers if headers else None)
                    st.dataframe(df_comp, use_container_width=True, height=280)
            elif c_type == "chart":
                if comp.get("plotly_spec"):
                    _render_chart(comp["plotly_spec"], chart_key=f"comp_chart_{idx}")
            st.write("")

    else:
        # ── Insight card ─────────────────────────────────────────────────────
        insight_text    = result.get("insight_text", "")
        key_metric      = result.get("key_metric", "")
        intent_type     = result.get("intent_type", "")
        strategies      = result.get("strategies", [])
        priority_matrix = result.get("priority_matrix", [])
        risks           = result.get("risks", [])

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

                strat_items = "".join(f"<li style='margin-bottom:8px'>{s}</li>" for s in strategies)

                matrix_rows = ""
                for item in priority_matrix:
                    imp = item.get("impact", "").lower()
                    eff = item.get("effort", "").lower()
                    matrix_rows += f"""
                    <tr>
                        <td>{item.get('action','')}</td>
                        <td style="text-align:center">{_tag(imp.capitalize(), imp, "impact")}</td>
                        <td style="text-align:center">{_tag(eff.capitalize(), eff, "effort")}</td>
                    </tr>"""

                matrix_html = f"""
                <div style="margin-top:18px">
                    <div class="hyp-label" style="margin-bottom:10px">Implementation priority matrix</div>
                    <table class="omega-table">
                        <thead><tr>
                            <th>Proposed action</th>
                            <th style="width:90px;text-align:center">Impact</th>
                            <th style="width:90px;text-align:center">Effort</th>
                        </tr></thead>
                        <tbody>{matrix_rows}</tbody>
                    </table>
                </div>""" if matrix_rows else ""

                risk_items  = "".join(f"<li>{r}</li>" for r in risks)
                risks_html  = f"""
                <div class="risk-box">
                    <div class="risk-box-label">Potential risks</div>
                    <ul>{risk_items}</ul>
                </div>""" if risk_items else ""

                metric_chip = f"<div class='metric-callout'>{key_metric}</div>" if key_metric else ""

                st.markdown(f"""
                <div class="omega-card" style="--card-accent: var(--orange)">
                    <div class="card-eyebrow">Prescriptive strategy</div>
                    {metric_chip}
                    <div class="card-body" style="margin-bottom:16px">{insight_text}</div>
                    <div class="hyp-label">Actionable strategies</div>
                    <ul style="margin:8px 0 0 0; padding-left:20px; font-size:14px;
                               color:var(--ink-muted); line-height:1.9">{strat_items}</ul>
                    {matrix_html}
                    {risks_html}
                </div>
                """, unsafe_allow_html=True)

            else:
                metric_chip = f"<div class='metric-callout'>{key_metric}</div>" if key_metric else ""
                st.markdown(f"""
                <div class="omega-card" style="--card-accent: var(--accent)">
                    <div class="card-eyebrow">Key insight</div>
                    {metric_chip}
                    <div class="card-body">{insight_text}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Hypothesis test ───────────────────────────────────────────────────────
    hypothesis = result.get("hypothesis", {})
    if hypothesis and hypothesis.get("status") == "success":
        is_sig  = hypothesis.get("is_significant", False)
        p_val   = hypothesis.get("p_value", 1.0)
        p_str   = f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}"
        sig_badge = (
            "<span class='badge badge-ok'>Significant</span>"
            if is_sig else
            "<span class='badge badge-err'>Not significant</span>"
        )

        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--violet)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>Statistical hypothesis test</span>
                {sig_badge}
            </div>
            <div class="card-title">{hypothesis.get('test_name','')}</div>
            <div class="hypothesis-section">
                <div class="hyp-label">Null hypothesis (H₀)</div>
                <div class="hyp-text">{hypothesis.get('null_hypothesis','')}</div>
                <hr class="hyp-divider">
                <div class="hyp-label">Alternative hypothesis (H₁)</div>
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
            <div class="card-body" style="border-top:1px solid var(--gray-150);
                padding-top:16px; margin-top:4px">
                <strong style="color:var(--ink)">Conclusion:</strong> {hypothesis.get('interpretation','')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif hypothesis and hypothesis.get("status") == "failed":
        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--amber)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>Statistical hypothesis test</span>
                <span class="badge badge-err">Unable to test</span>
            </div>
            <div class="card-body">
                Could not complete the statistical test:
                <strong style="color:var(--ink)">{hypothesis.get('message','Unknown error')}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Forecast ──────────────────────────────────────────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "success":
        time_col      = prediction.get("time_column")
        metric_col    = prediction.get("metric_column")
        model_metrics = prediction.get("model_metrics", {})
        r2            = model_metrics.get("r_squared", 0.0)

        if r2 > 0.8:
            acc_badge = "<span class='badge badge-ok'>High accuracy</span>"
        elif r2 > 0.5:
            acc_badge = "<span class='badge badge-running'>Moderate accuracy</span>"
        else:
            acc_badge = "<span class='badge badge-err'>Low accuracy</span>"

        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--teal)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>Forecast projection</span>
                {acc_badge}
            </div>
            <div class="card-title">Seasonal trend: {metric_col} over {time_col}</div>
            <div class="stat-group">
                <div class="stat-item">
                    <div class="stat-label">R-squared</div>
                    <div class="stat-value">{r2:.4f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Horizon</div>
                    <div class="stat-value">{len(prediction.get('forecast_values', []))}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Std. error</div>
                    <div class="stat-value">{model_metrics.get('std_err', 0.0):,.4f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            hist_dates = prediction.get("historical_dates", [])
            hist_values = prediction.get("historical_values", [])
            fc_dates   = prediction.get("forecast_dates", [])
            fc_values  = prediction.get("forecast_values", [])
            lower_ci   = prediction.get("lower_bound", [])
            upper_ci   = prediction.get("upper_bound", [])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fc_dates + fc_dates[::-1],
                y=upper_ci + lower_ci[::-1],
                fill='toself', fillcolor='rgba(13,148,136,0.07)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True,
                name="95% confidence interval"
            ))
            fig.add_trace(go.Scatter(
                x=hist_dates, y=hist_values,
                mode='lines+markers', name='Historical',
                line=dict(color='#4F46E5', width=2.5),
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
                margin=dict(l=44, r=44, t=44, b=44), height=380,
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FAFAFA',
                font=dict(family='Inter, sans-serif', size=12, color='#525252'),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"Forecast plot could not be rendered: {exc}")

    elif prediction and prediction.get("status") == "failed":
        err_msg = prediction.get('message', 'Unknown error')
        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--amber)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>Forecast projection</span>
                <span class="badge badge-err">Failed</span>
            </div>
            <div class="card-body">
                <strong style="color:var(--ink)">{err_msg}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Regression simulator ───────────────────────────────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "regression":
        target_col   = prediction.get("target_column")
        intercept    = prediction.get("intercept", 0.0)
        coefficients = prediction.get("coefficients", {})
        features     = prediction.get("features", [])
        metrics      = prediction.get("model_metrics", {})
        dummy_mappings = prediction.get("dummy_mappings", {})

        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--accent)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>Regression simulator</span>
                <span class="badge badge-running">R² = {metrics.get('r_squared', 0.0):.4f}</span>
            </div>
            <div class="card-title">Predicting: {_humanise_column(target_col)}</div>
            <div class="card-body">Adjust the controls below to see predictions update live.</div>
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
                        label=f"{_humanise_column(name)}",
                        min_value=min_val, max_value=max_val, value=mean_val,
                        step=step, key=f"slider_{name}"
                    )}
                else:
                    cats = feat.get("categories", [])
                    slider_vals[name] = {"type": "categorical", "value": st.selectbox(
                        label=f"{_humanise_column(name)}", options=cats,
                        index=cats.index(feat.get("default")) if feat.get("default") in cats else 0,
                        key=f"select_{name}"
                    )}

        pred_y = intercept
        for name, info in slider_vals.items():
            if info["type"] == "numeric":
                feat     = next((f for f in features if f["name"] == name), {})
                mean_val = feat.get("mean", 0.0)
                std_val  = feat.get("std", 1.0)
                pred_y  += coefficients.get(name, 0.0) * ((info["value"] - mean_val) / std_val)
            else:
                cat_dummies = dummy_mappings.get(name, {})
                dummy_col   = cat_dummies.get(info["value"])
                if dummy_col:
                    pred_y += coefficients.get(dummy_col, 0.0)

        y_str = (f"${pred_y:,.2f}" if any(k in target_col.lower() for k in ("price","cost","sales"))
                 else f"{pred_y:,.4f}")

        st.markdown(f"""
        <div style="background:var(--emerald-light); border:1px solid var(--emerald-border);
                    border-radius:var(--radius-md); padding:20px 28px;
                    text-align:center; margin:16px 0">
            <div class="stat-label" style="margin-bottom:8px">
                Predicted {_humanise_column(target_col)}
            </div>
            <div style="font-size:40px; font-weight:800; color:var(--ink);
                        letter-spacing:-2px; font-family:var(--font-mono)">{y_str}</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            df_sample = st.session_state.df.dropna(subset=[target_col]).head(500)
            actuals   = pd.to_numeric(df_sample[target_col], errors='coerce').dropna().values
            preds     = np.full(len(actuals), intercept)
            df_aligned = df_sample.loc[df_sample.index[:len(actuals)]]

            for feat in features:
                name   = feat["name"]
                f_type = feat.get("type", "numeric")
                if f_type == "numeric":
                    vals     = pd.to_numeric(df_aligned[name], errors='coerce').fillna(feat.get("mean", 0.0)).values
                    mean_val = feat.get("mean", 0.0)
                    std_val  = feat.get("std", 1.0)
                    preds   += coefficients.get(name, 0.0) * ((vals - mean_val) / std_val)
                else:
                    cat_dummies = dummy_mappings.get(name, {})
                    for cat, dummy_col in cat_dummies.items():
                        preds += coefficients.get(dummy_col, 0.0) * (df_aligned[name] == cat).astype(float).values

            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=actuals, y=preds, mode='markers',
                marker=dict(color='rgba(79,70,229,0.5)', size=5),
                name='Actual vs predicted'
            ))
            mn = float(min(min(actuals), min(preds)))
            mx = float(max(max(actuals), max(preds)))
            fig_reg.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx], mode='lines',
                line=dict(color='#E11D48', width=2, dash='dash'),
                name='Perfect fit'
            ))
            fig_reg.update_layout(
                title=dict(text="Model diagnostic — actual vs predicted",
                           font=dict(size=13, family='Inter, sans-serif')),
                xaxis=dict(title=f"Actual {_humanise_column(target_col)}",
                           showgrid=True, gridcolor='#F0F0F0'),
                yaxis=dict(title=f"Predicted {_humanise_column(target_col)}",
                           showgrid=True, gridcolor='#F0F0F0'),
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FAFAFA',
                margin=dict(l=44, r=44, t=52, b=44), height=360,
                font=dict(family='Inter, sans-serif', size=12, color='#525252'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_reg, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render regression fit plot: {exc}")

    # ── Classification simulator ───────────────────────────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "classification":
        model_mode    = prediction.get("model_mode", "binary")
        target_col    = prediction.get("target_column")
        features      = prediction.get("features", [])
        metrics       = prediction.get("model_metrics", {})
        dummy_mappings = prediction.get("dummy_mappings", {})

        if model_mode == "binary":
            target_label  = prediction.get("target_label", target_col)
            class_0_label = prediction.get("class_0_label", "0")
            class_1_label = prediction.get("class_1_label", "1")
            intercept     = prediction.get("intercept", 0.0)
            coefficients  = prediction.get("coefficients", {})

            st.markdown(f"""
            <div class="omega-card" style="--card-accent: var(--violet)">
                <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                    <span>Probability simulator</span>
                    <span class="badge" style="background:var(--violet-light);
                        color:var(--violet);border:1px solid var(--violet-border)">
                        Accuracy {metrics.get('accuracy', 0.0)*100:.1f}%
                    </span>
                </div>
                <div class="card-title">Target: {_humanise_column(target_label)}</div>
                <div class="card-body">{class_0_label} vs {class_1_label} — adjust controls to see probability live.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            classes      = prediction.get("classes", [])
            intercepts   = prediction.get("intercepts", {})
            coefficients = prediction.get("coefficients", {})

            st.markdown(f"""
            <div class="omega-card" style="--card-accent: var(--violet)">
                <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                    <span>Multi-class simulator</span>
                    <span class="badge" style="background:var(--violet-light);
                        color:var(--violet);border:1px solid var(--violet-border)">
                        Accuracy {metrics.get('accuracy', 0.0)*100:.1f}%
                    </span>
                </div>
                <div class="card-title">Target: {_humanise_column(target_col)}</div>
                <div class="card-body">{len(classes)} categories — adjust controls to calculate probabilities.</div>
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
                        label=f"{_humanise_column(name)}",
                        min_value=min_val, max_value=max_val, value=mean_val,
                        step=step, key=f"class_slider_{name}"
                    )}
                else:
                    cats = feat.get("categories", [])
                    slider_vals[name] = {"type": "categorical", "value": st.selectbox(
                        label=f"{_humanise_column(name)}", options=cats,
                        index=cats.index(feat.get("default")) if feat.get("default") in cats else 0,
                        key=f"class_select_{name}"
                    )}

        if model_mode == "binary":
            z = intercept
            for name, info in slider_vals.items():
                if info["type"] == "numeric":
                    feat     = next((f for f in features if f["name"] == name), {})
                    mean_val = feat.get("mean", 0.0)
                    std_val  = feat.get("std", 1.0)
                    z       += coefficients.get(name, 0.0) * ((info["value"] - mean_val) / std_val)
                else:
                    cat_dummies = dummy_mappings.get(name, {})
                    dummy_col   = cat_dummies.get(info["value"])
                    if dummy_col:
                        z += coefficients.get(dummy_col, 0.0)

            prob_val   = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
            pred_class = class_1_label if prob_val >= 0.5 else class_0_label
            bar_color  = "var(--emerald)" if prob_val >= 0.5 else "var(--rose)"

            st.markdown(f"""
            <div style="background:var(--violet-light); border:1px solid var(--violet-border);
                        border-radius:var(--radius-md); padding:22px 26px; margin:16px 0">
                <div style="display:flex; justify-content:space-between; margin-bottom:16px">
                    <div>
                        <div class="stat-label">Predicted class</div>
                        <div style="font-size:28px; font-weight:800; color:var(--ink);
                                    font-family:var(--font-mono); margin-top:4px">{pred_class}</div>
                    </div>
                    <div style="text-align:right">
                        <div class="stat-label">P({class_1_label})</div>
                        <div style="font-size:28px; font-weight:800; color:var(--ink);
                                    font-family:var(--font-mono); margin-top:4px">{prob_val*100:.1f}%</div>
                    </div>
                </div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{prob_val*100}%; background:{bar_color}"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            classes      = prediction.get("classes", [])
            intercepts   = prediction.get("intercepts", {})
            coefficients = prediction.get("coefficients", {})
            odds_dict    = {}

            for target_val in classes:
                beta_0    = intercepts.get(target_val, 0.0)
                coef_dict = coefficients.get(target_val, {})
                z_c       = beta_0
                for name, info in slider_vals.items():
                    if info["type"] == "numeric":
                        feat     = next((f for f in features if f["name"] == name), {})
                        mean_val = feat.get("mean", 0.0)
                        std_val  = feat.get("std", 1.0)
                        z_c     += coef_dict.get(name, 0.0) * ((info["value"] - mean_val) / std_val)
                    else:
                        cat_dummies = dummy_mappings.get(name, {})
                        dummy_col   = cat_dummies.get(info["value"])
                        if dummy_col:
                            z_c += coef_dict.get(dummy_col, 0.0)
                odds_dict[target_val] = np.exp(np.clip(z_c, -20.0, 20.0))

            sum_odds   = sum(odds_dict.values())
            probs_dict = {k: v / sum_odds for k, v in odds_dict.items()}
            sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
            pred_class = sorted_probs[0][0]

            prob_bars = "".join(
                f'<div style="margin-bottom:10px">'
                f'<div style="display:flex; justify-content:space-between; font-size:13px;'
                f'color:var(--ink-muted); font-weight:500; margin-bottom:5px">'
                f'<span>{cn}</span><span>{pv*100:.1f}%</span></div>'
                f'<div class="prob-bar-wrap"><div class="prob-bar-fill" '
                f'style="width:{pv*100}%; background:var(--violet)"></div></div>'
                f'</div>'
                for cn, pv in sorted_probs
            )

            st.markdown(
                f'<div style="background:var(--violet-light); border:1px solid var(--violet-border);'
                f'border-radius:var(--radius-md); padding:22px 26px; margin:16px 0">'
                f'<div style="margin-bottom:18px">'
                f'<div class="stat-label">Predicted assignment</div>'
                f'<div style="font-size:30px; font-weight:800; color:var(--ink);'
                f'font-family:var(--font-mono); margin-top:4px">{pred_class}</div></div>'
                f'{prob_bars}</div>',
                unsafe_allow_html=True
            )

    # ── Clustering ─────────────────────────────────────────────────────────────
    prediction = result.get("prediction", {})
    if prediction and prediction.get("status") == "clustering":
        features             = prediction.get("features", [])
        clusters             = prediction.get("clusters", [])
        labels               = prediction.get("labels", [])
        pc_coords            = prediction.get("pc_coords", [])
        sample_size          = prediction.get("sample_size", 0)
        means_dict           = prediction.get("means", {})
        stds_dict            = prediction.get("stds", {})
        dummy_mappings       = prediction.get("dummy_mappings", {})
        feature_names_internal = prediction.get("feature_names_internal", [])
        metrics              = prediction.get("model_metrics", {})

        st.markdown(f"""
        <div class="omega-card" style="--card-accent: var(--cyan)">
            <div class="card-eyebrow" style="justify-content:space-between; display:flex">
                <span>Customer segmentation</span>
                <span class="badge" style="background:var(--cyan-light);
                    color:var(--cyan);border:1px solid var(--cyan-border)">
                    K = {metrics.get('clusters_count', 3)} clusters
                </span>
            </div>
            <div class="card-title">{sample_size:,} records across {len(features)} attributes</div>
            <div class="card-body">Adjust controls below to assign a new record to a segment.</div>
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
                        marker=dict(size=4, opacity=0.72),
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
                font=dict(family='Inter, sans-serif', size=11, color='#525252'),
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render cluster cloud: {exc}")

        st.markdown("<div class='s-label' style='margin:18px 0 12px'>Segment allocation tool</div>",
                    unsafe_allow_html=True)
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
                        label=f"{_humanise_column(name)}",
                        min_value=min_val, max_value=max_val, value=mean_val,
                        step=step, key=f"cluster_slider_{name}"
                    )}
                else:
                    cats = feat.get("categories", [])
                    slider_vals[name] = {"type": "categorical", "value": st.selectbox(
                        label=f"{_humanise_column(name)}", options=cats,
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

        inv_dists           = [1.0 / (d[1] + 1e-10) for d in distances]
        assigned_confidence = (inv_dists[0] / sum(inv_dists)) * 100

        st.markdown(f"""
        <div style="background:var(--cyan-light); border:1px solid var(--cyan-border);
                    border-radius:var(--radius-md); padding:20px 26px; margin:16px 0;
                    display:flex; justify-content:space-between; align-items:center">
            <div>
                <div class="stat-label">Predicted segment</div>
                <div style="font-size:24px; font-weight:800; color:var(--ink);
                            font-family:var(--font-mono); margin-top:4px">
                    Segment {distances[0][0]} ({distances[0][2]})
                </div>
            </div>
            <div style="text-align:right">
                <div class="stat-label">Confidence</div>
                <div style="font-size:24px; font-weight:800; color:var(--ink);
                            font-family:var(--font-mono); margin-top:4px">
                    {assigned_confidence:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────────────────────
    chart_spec = result.get("chart_spec")
    chart_gen  = result.get("chart_gen", False)
    pred_status = result.get("prediction", {}).get("status")
    blocks_chart = pred_status in ("success", "regression", "classification", "clustering")

    if chart_gen and chart_spec and not blocks_chart:
        _render_chart(chart_spec, chart_key="main_layout_chart")
    elif not chart_gen:
        render_note = result.get("raw", {}).get("chart", {}).get("render_note", "")
        if render_note:
            st.info(f"ℹ️ {render_note}")

    # ── Table ─────────────────────────────────────────────────────────────────
    rows      = result.get("rows", [])
    truncated = result.get("truncated", False)
    row_count = result.get("row_count", 0)

    if rows:
        if truncated:
            st.markdown(
                f"<div class='trunc-notice'>"
                f"⚠ Showing top 500 of {row_count:,} rows"
                f"</div>",
                unsafe_allow_html=True,
            )
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True, height=280)
        csv = result_df.to_csv(index=False).encode("utf-8")
        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            st.download_button(
                label="⬇ Download CSV",
                data=csv, file_name="omega_results.csv", mime="text/csv",
                use_container_width=True
            )
        with col_dl2:
            from src.report import generate_pdf_report
            try:
                pdf_buffer = generate_pdf_report(result)
                st.download_button(
                    label="📄 Download PDF report",
                    data=pdf_buffer, file_name="omega_report.pdf", mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")

    # ── Follow-ups ─────────────────────────────────────────────────────────────
    follow_ups = result.get("follow_ups", [])
    if follow_ups:
        st.markdown(
            "<div class='followup-heading'>You might also ask</div>",
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

    st.markdown("""
    <div class="main-header">
        <div class="header-eyebrow">AI-powered analytics</div>
        <div class="main-wordmark">
            <span class="wordmark-hex">⬡</span>
            Omega
        </div>
        <div class="main-tagline">Ask anything about your data — in plain English</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-ring">📂</div>
            <div class="empty-title">Upload a dataset to begin</div>
            <div class="empty-sub">Supports CSV and Excel files. No code required — just ask your question and Omega does the rest.</div>
            <div class="example-queries">
                <span class="eq-pill">Show total sales by region</span>
                <span class="eq-pill">Find top 10 customers</span>
                <span class="eq-pill">Is there a trend in revenue?</span>
                <span class="eq-pill">Segment customers by behaviour</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Query bar
    st.markdown("<div class='query-label'>Ask a question</div>", unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            label="Query",
            value=st.session_state.query_input,
            placeholder="e.g. Show total sales by region, or find top 10 customers",
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

        st.markdown(f"""
        <div class="running-bar">
            <span class="spin-ring"></span>
            <span class="running-text">Analysing — {user_query or ''}</span>
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown(f"""
        <div class="result-header">
            <span class="badge badge-ok">Done</span>
            <span class="result-query-text">{result.get('query','')}</span>
        </div>
        <hr style="border:none;border-top:1px solid var(--gray-150);margin-bottom:28px">
        """, unsafe_allow_html=True)
        _render_results(result)

    else:
        st.markdown("""
        <div class="empty-wrap" style="padding:56px 24px">
            <div class="empty-ring">💬</div>
            <div class="empty-title">Ask your first question</div>
            <div class="empty-sub">Type a question above and Omega will analyse your data, generate charts, and surface key insights automatically.</div>
            <div class="example-queries">
                <span class="eq-pill">Show me total sales by region</span>
                <span class="eq-pill">What is the average order value?</span>
                <span class="eq-pill">Find the top 10 customers</span>
                <span class="eq-pill">Is there a seasonal trend?</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Entry point ────────────────────────────────────────────────────────────────

_render_sidebar()
_render_main()