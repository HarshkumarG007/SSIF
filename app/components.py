"""
components.py — Visual Design System & Component Library for SSIF Dashboard
Student Success Intelligence Framework (SSIF)

Implements HSL research-lab dark styling, metric cards, limitation banners,
and Plotly chart themes per docs/design.md.
"""
from __future__ import annotations

from typing import Any
import plotly.graph_objects as go
import streamlit as st

# Color Palette (HSL-aligned)
COLOR_PRIMARY = "#3B82F6"      # Trust blue
COLOR_SUCCESS = "#10B981"      # Teal green
COLOR_WARNING = "#F59E0B"      # Amber warning
COLOR_DANGER = "#EF4444"       # Risk red
COLOR_NEUTRAL = "#94A3B8"      # Cool grey
COLOR_BG_DARK = "#0F172A"      # Main dark background
COLOR_SURFACE = "#1E293B"      # Card surface
COLOR_BORDER = "#334155"       # Panel border


def apply_custom_css():
    """Inject research observatory CSS typography, cards, and theme overrides."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        h1, h2 {{
            font-family: 'Playfair Display', Georgia, serif !important;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        h3, h4, h5, h6 {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 600;
        }}

        .stMetric {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }}

        .research-context-box {{
            background: rgba(30, 41, 59, 0.7);
            border-left: 4px solid {COLOR_PRIMARY};
            border-top: 1px solid {COLOR_BORDER};
            border-right: 1px solid {COLOR_BORDER};
            border-bottom: 1px solid {COLOR_BORDER};
            padding: 12px 18px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 0.90rem;
        }}

        .limitation-banner {{
            background: rgba(245, 158, 11, 0.12);
            border-left: 4px solid {COLOR_WARNING};
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-left-width: 4px;
            padding: 10px 16px;
            border-radius: 6px;
            margin-bottom: 18px;
            color: #FCD34D;
            font-size: 0.88rem;
        }}

        .kpi-card {{
            background: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }}

        .code-annotation {{
            font-family: 'JetBrains Mono', monospace;
            background: #020617;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #38BDF8;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_research_context(dataset_info: str, method_info: str, limitation_info: str):
    """Render standardized research context banner."""
    st.markdown(
        f"""
        <div class="research-context-box">
            <div>📊 <b>DATASET:</b> {dataset_info}</div>
            <div>🔬 <b>METHODOLOGY:</b> {method_info}</div>
            <div>⚠️ <b>LIMITATION:</b> {limitation_info}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_limitation_banner(message: str | None = None):
    """Render scientific limitations banner for predictive pages."""
    msg = message or (
        "<b>STATISTICAL ESTIMATES ONLY:</b> Model outputs represent observational associations "
        "and counterfactual risk rankings. Confidence intervals reflect empirical sampling variation. "
        "Outputs must inform advisory human judgment rather than automated institutional gating."
    )
    st.markdown(
        f"""
        <div class="limitation-banner">
            ⚠️ {msg}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    """Apply research-lab dark styling to Plotly figures."""
    fig.update_layout(
        paper_bgcolor=COLOR_BG_DARK,
        plot_bgcolor=COLOR_SURFACE,
        margin=dict(l=40, r=30, t=50 if title else 20, b=40),
        height=height,
        font=dict(family="Inter, sans-serif", color="#F1F5F9", size=12),
        title=dict(
            text=title or "",
            font=dict(family="Playfair Display, serif", size=16, color="#F8FAFC"),
        ),
        xaxis=dict(
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color="#94A3B8"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            font=dict(color="#CBD5E1"),
        ),
        hoverlabel=dict(
            bgcolor="#020617",
            bordercolor=COLOR_PRIMARY,
            font=dict(family="JetBrains Mono, monospace", size=12, color="#F8FAFC"),
        ),
    )
    return fig
