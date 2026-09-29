"""
components.py — Visual Design System & Component Library for SSIF Dashboard
Student Success Intelligence Framework (SSIF)

Implements HSL research-lab dark styling, dimensional depth (glassmorphism & elevation),
WebGL 3D interactive visualizations with 2D flatten toggles, and Plotly chart themes
per docs/design.md v2.0.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Color System (HSL-calibrated per docs/design.md §2) ─────────────────────
COLOR_PRIMARY = "#3B82F6"      # Trust blue (hsl(220, 70%, 50%))
COLOR_PRIMARY_DARK = "#1D4ED8" # Dark blue (hsl(220, 70%, 35%))
COLOR_SUCCESS = "#10B981"      # Teal green (hsl(155, 60%, 40%))
COLOR_WARNING = "#F59E0B"      # Amber warning (hsl(38, 90%, 50%))
COLOR_DANGER = "#EF4444"       # Risk red (hsl(4, 75%, 50%))
COLOR_NEUTRAL = "#94A3B8"      # Cool grey (hsl(220, 15%, 60%))

# Background & Surface System (Dark research-lab aesthetic)
COLOR_BG_DARK = "#0F172A"      # Near-black main background (hsl(220, 20%, 10%))
COLOR_SURFACE = "#1E293B"      # Resting card surface (hsl(220, 18%, 14%))
COLOR_ELEVATED = "#334155"     # Elevated card surface (hsl(220, 16%, 18%))
COLOR_BORDER = "#334155"       # Panel border (hsl(220, 14%, 22%))

# Dimensional Tokens (docs/design.md §2 & §5)
DEPTH = {
    "glass_surface": "hsla(220, 18%, 18%, 0.55)",
    "glass_border": "hsla(220, 30%, 70%, 0.12)",
    "glass_highlight": "hsla(220, 40%, 85%, 0.06)",
    "shadow_1": "0 1px 2px hsla(220, 60%, 2%, 0.4)",
    "shadow_2": "0 4px 12px hsla(220, 60%, 2%, 0.45), 0 1px 2px hsla(220, 60%, 2%, 0.3)",
    "shadow_3": "0 12px 32px hsla(220, 60%, 2%, 0.5), 0 2px 6px hsla(220, 60%, 2%, 0.35)",
    "glow_primary": "0 0 24px hsla(220, 70%, 55%, 0.25)",
}


def apply_custom_css():
    """Inject research observatory CSS typography, elevation, glassmorphism, and transitions."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        :root {{
            --primary: {COLOR_PRIMARY};
            --surface: {COLOR_SURFACE};
            --border: {COLOR_BORDER};
            --glass-surface: {DEPTH['glass_surface']};
            --glass-border: {DEPTH['glass_border']};
            --glass-highlight: {DEPTH['glass_highlight']};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: {COLOR_BG_DARK};
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

        /* Level 1 & 2 Elevation: Resting & Hover Cards */
        .stMetric, .metric-card, .kpi-card {{
            background: {COLOR_SURFACE} !important;
            border: 1px solid {COLOR_BORDER} !important;
            padding: 14px 18px;
            border-radius: 8px;
            box-shadow: {DEPTH['shadow_1']};
            transition: transform 160ms ease, box-shadow 160ms ease;
        }}

        .stMetric:hover, .metric-card:hover, .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: {DEPTH['shadow_2']};
        }}

        /* Level 3 Elevation: Glassmorphism Floating Surface */
        .glass-panel {{
            background: var(--glass-surface);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: {DEPTH['shadow_3']};
            position: relative;
            margin-bottom: 20px;
        }}

        .glass-panel::before {{
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 1px;
            background: var(--glass-highlight);
        }}

        /* Research Context Header */
        .research-context-box {{
            background: rgba(30, 41, 59, 0.65);
            backdrop-filter: blur(8px);
            border-left: 4px solid {COLOR_PRIMARY};
            border-top: 1px solid {COLOR_BORDER};
            border-right: 1px solid {COLOR_BORDER};
            border-bottom: 1px solid {COLOR_BORDER};
            padding: 12px 18px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 0.90rem;
            box-shadow: {DEPTH['shadow_1']};
        }}

        /* Scientific Limitation Banner */
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

        .code-annotation {{
            font-family: 'JetBrains Mono', monospace;
            background: #020617;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #38BDF8;
        }}

        /* Accessibility: Reduce Motion */
        @media (prefers-reduced-motion: reduce) {{
            .stMetric, .metric-card, .kpi-card {{
                transition: none !important;
                transform: none !important;
            }}
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


def apply_plotly_theme(fig: go.Figure, title: str | None = None, height: int = 440) -> go.Figure:
    """Apply research-lab dark styling to 2D and 3D Plotly figures per docs/design.md."""
    layout_update: dict[str, Any] = {
        "paper_bgcolor": COLOR_BG_DARK,
        "plot_bgcolor": COLOR_SURFACE,
        "margin": dict(l=40, r=30, t=50 if title else 25, b=40),
        "height": height,
        "font": dict(family="Inter, sans-serif", color="#F1F5F9", size=12),
        "title": dict(
            text=title or "",
            font=dict(family="Playfair Display, serif", size=16, color="#F8FAFC"),
        ),
        "legend": dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            font=dict(color="#CBD5E1"),
        ),
        "hoverlabel": dict(
            bgcolor="#020617",
            bordercolor=COLOR_PRIMARY,
            font=dict(family="JetBrains Mono, monospace", size=12, color="#F8FAFC"),
        ),
    }

    # 2D Axes Styling
    if hasattr(fig.layout, "xaxis"):
        layout_update["xaxis"] = dict(
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color="#94A3B8"),
        )
    if hasattr(fig.layout, "yaxis"):
        layout_update["yaxis"] = dict(
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            tickfont=dict(color="#94A3B8"),
        )

    # 3D WebGL Scene Styling (docs/design.md §4.1 & §6)
    scene_config = dict(
        bgcolor=COLOR_BG_DARK,
        xaxis=dict(
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            backgroundcolor=COLOR_SURFACE,
            showbackground=True,
            tickfont=dict(color="#94A3B8"),
            title=dict(font=dict(color="#F1F5F9")),
        ),
        yaxis=dict(
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            backgroundcolor=COLOR_SURFACE,
            showbackground=True,
            tickfont=dict(color="#94A3B8"),
            title=dict(font=dict(color="#F1F5F9")),
        ),
        zaxis=dict(
            gridcolor=COLOR_BORDER,
            linecolor=COLOR_BORDER,
            backgroundcolor=COLOR_SURFACE,
            showbackground=True,
            tickfont=dict(color="#94A3B8"),
            title=dict(font=dict(color="#F1F5F9")),
        ),
    )

    if hasattr(fig.layout, "scene") and fig.layout.scene:
        fig.layout.scene.update(scene_config)
    else:
        layout_update["scene"] = scene_config

    fig.update_layout(**layout_update)
    return fig


# ─── 3D WebGL Chart Builders (§6 in docs/design.md v2.0) ────────────────────

def build_risk_surface_figure(flatten_2d: bool = False) -> go.Figure:
    """
    §6.1 Risk Surface — GPA × Attendance → Predicted Dropout Probability.
    Shows the nonlinear interaction between GPA slide and attendance erosion.
    Includes floor contour projection (project_z=True) and 2D heatmap toggle.
    """
    gpa_grid = np.linspace(1.5, 4.0, 35)
    att_grid = np.linspace(50.0, 100.0, 35)
    GPA, ATT = np.meshgrid(gpa_grid, att_grid)

    # Calibrated empirical logistic risk function
    logit = -0.8 - 1.6 * (GPA - 2.8) - 0.04 * (ATT - 82)
    Z = 1.0 / (1.0 + np.exp(-logit))

    colorscale = [
        [0.0, COLOR_SUCCESS],  # Persistent low risk (< 15%)
        [0.35, COLOR_WARNING], # Elevated risk (15–35%)
        [1.0, COLOR_DANGER],   # High risk (> 35%)
    ]

    if flatten_2d:
        # 2D Contour Heatmap Flat Projection
        fig = go.Figure(data=[
            go.Contour(
                x=gpa_grid,
                y=att_grid,
                z=Z,
                colorscale=colorscale,
                colorbar=dict(title="P(Dropout)", tickformat=".0%"),
                contours=dict(coloring="heatmap", showlabels=True),
            )
        ])
        fig.update_layout(
            xaxis_title="Recent GPA (1.5 – 4.0)",
            yaxis_title="Attendance Rate (%)",
            title="Predicted Dropout Risk Surface (2D Contour Projection)",
        )
    else:
        # 3D WebGL Surface with Floor Contour
        fig = go.Figure(data=[
            go.Surface(
                x=gpa_grid,
                y=att_grid,
                z=Z,
                colorscale=colorscale,
                colorbar=dict(title="P(Dropout)", tickformat=".0%"),
                contours_z=dict(show=True, usecolormap=True, project_z=True),
            )
        ])
        fig.update_layout(
            scene=dict(
                xaxis_title="Recent GPA",
                yaxis_title="Attendance (%)",
                zaxis_title="P(Dropout next sem.)",
                camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.9)),
            ),
            title="Predicted Dropout Risk Surface — GPA × Attendance (WebGL 3D)",
        )

    return apply_plotly_theme(fig, height=500)


def build_trajectory_ribbons_figure(
    df: pd.DataFrame, flatten_2d: bool = False, n_students: int = 40
) -> go.Figure:
    """
    §6.2 Trajectory Ribbons — Per-student path through (Semester, GPA, Attendance) space.
    Stratified sample of persisted vs dropped-out students.
    """
    persist_ids = df[df["Target_Dropout_Next_Sem"] == 0]["Student_ID"].unique()[:n_students // 2]
    dropout_ids = df[df["Target_Dropout_Next_Sem"] == 1]["Student_ID"].unique()[:n_students // 2]
    selected_ids = list(persist_ids) + list(dropout_ids)

    fig = go.Figure()

    if flatten_2d:
        # 2D Multi-Series: Semester vs GPA with Attendance encoded
        for sid in selected_ids:
            s = df[df["Student_ID"] == sid].sort_values("Semester")
            is_dropout = s["Target_Dropout_Next_Sem"].iloc[-1] == 1
            color = COLOR_DANGER if is_dropout else COLOR_SUCCESS
            fig.add_trace(
                go.Scatter(
                    x=s["Semester"],
                    y=s["Sem_GPA"],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=s["Attendance"] / 14.0, color=color),
                    name=f"{'Dropout' if is_dropout else 'Persisted'} ({sid})",
                    opacity=0.70,
                    hoverinfo="text",
                    text=[
                        f"Student: {sid}<br>Sem {sem}: GPA {gpa:.2f}<br>Att: {att:.1f}%"
                        for sem, gpa, att in zip(s["Semester"], s["Sem_GPA"], s["Attendance"])
                    ],
                )
            )
        fig.update_layout(
            xaxis_title="Semester (1–8)",
            yaxis_title="Semester GPA",
            title=f"Individual Trajectories (2D Projection, N={len(selected_ids)} Students)",
            showlegend=False,
        )
    else:
        # 3D WebGL Ribbons (Semester × GPA × Attendance)
        for sid in selected_ids:
            s = df[df["Student_ID"] == sid].sort_values("Semester")
            is_dropout = s["Target_Dropout_Next_Sem"].iloc[-1] == 1
            color = COLOR_DANGER if is_dropout else COLOR_SUCCESS
            fig.add_trace(
                go.Scatter3d(
                    x=s["Semester"],
                    y=s["Sem_GPA"],
                    z=s["Attendance"],
                    mode="lines+markers",
                    line=dict(color=color, width=4),
                    marker=dict(size=3, color=color),
                    name=f"Student {sid} ({'Dropout' if is_dropout else 'Persisted'})",
                    opacity=0.75,
                )
            )
        fig.update_layout(
            scene=dict(
                xaxis_title="Semester",
                yaxis_title="GPA",
                zaxis_title="Attendance (%)",
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
            ),
            title=f"Individual Academic Trajectories (Green: Persisted, Red: Dropped Out, N={len(selected_ids)})",
            showlegend=False,
        )

    return apply_plotly_theme(fig, height=520)


def build_risk_scatter_3d_figure(
    df: pd.DataFrame, flatten_2d: bool = False, max_points: int = 1500
) -> go.Figure:
    """
    §6.3 Multivariate Scatter — GPA velocity × Attendance × Failed Courses.
    Inspects cluster separation in 3D feature space with 2D pair projection toggle.
    """
    sub = df.sample(min(len(df), max_points), random_state=42)

    x_val = sub["Sem_GPA"]
    y_val = sub["Attendance"]
    z_val = sub["Failed_Courses"] if "Failed_Courses" in sub.columns else sub["LMS_Logins"]
    z_label = "Failed Courses" if "Failed_Courses" in sub.columns else "LMS Logins"

    colors = sub["Target_Dropout_Next_Sem"].map({0: COLOR_SUCCESS, 1: COLOR_DANGER})

    if flatten_2d:
        # 2D Scatter Projection (GPA vs Attendance, marker size by Failed Courses)
        fig = go.Figure(data=[
            go.Scatter(
                x=x_val,
                y=y_val,
                mode="markers",
                marker=dict(
                    size=(z_val + 1) * 3.5,
                    color=colors,
                    opacity=0.60,
                    line=dict(width=0.5, color="#1E293B"),
                ),
                text=[
                    f"Student: {sid}<br>GPA: {g:.2f}<br>Attendance: {a:.1f}%<br>{z_label}: {z}<br>Status: {'Dropout' if y==1 else 'Persisted'}"
                    for sid, g, a, z, y in zip(sub["Student_ID"], x_val, y_val, z_val, sub["Target_Dropout_Next_Sem"])
                ],
                hoverinfo="text",
            )
        ])
        fig.update_layout(
            xaxis_title="Semester GPA",
            yaxis_title="Attendance Rate (%)",
            title=f"Feature Risk Space (2D Projection, Bubble Size={z_label}, N={len(sub)})",
        )
    else:
        # 3D WebGL Scatter
        fig = go.Figure(data=[
            go.Scatter3d(
                x=x_val,
                y=y_val,
                z=z_val,
                mode="markers",
                marker=dict(size=3.5, color=colors, opacity=0.55),
                text=[
                    f"Student: {sid}<br>GPA: {g:.2f}<br>Attendance: {a:.1f}%<br>{z_label}: {z}<br>Status: {'Dropout' if y==1 else 'Persisted'}"
                    for sid, g, a, z, y in zip(sub["Student_ID"], x_val, y_val, z_val, sub["Target_Dropout_Next_Sem"])
                ],
                hoverinfo="text",
            )
        ])
        fig.update_layout(
            scene=dict(
                xaxis_title="GPA",
                yaxis_title="Attendance (%)",
                zaxis_title=z_label,
                camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8)),
            ),
            title=f"Three-Feature Risk Space (WebGL 3D, Rotate to Inspect Clustering, N={len(sub)})",
        )

    return apply_plotly_theme(fig, height=520)
