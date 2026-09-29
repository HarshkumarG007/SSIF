# design.md — Visual Design System
# Student Success Intelligence Framework (SSIF)

**Version:** 2.0
**Date:** 2026-09-29
**Design Philosophy:** Research Observatory, rendered with real depth — not a consumer app, not a flat spreadsheet either

**Changelog from v1.0:** Sections 0–3 and 6–10 are the same design language as before (color, typography, research-context panels, language rules, accessibility — all of that was already right). What's new is a dimensional layer: glassmorphism and elevation for the *interface*, and three genuine WebGL 3D charts for the specific places SSIF actually has three real variables worth looking at together (§4.1, §5). The v1.0 rule against decorative 3D charts is kept, sharpened, not dropped — see §4.1 for exactly where the line sits and why.

---

## 0. Core Design Identity

The SSIF dashboard must communicate:

> **"This is a computational research laboratory, not a toy. Every number has uncertainty. Every model has limitations. This system helps you investigate — it does not tell you what to do."**

**New in v2.0:** that identity is stated in flat panels today. It should be *felt* — through layered depth, real shadow, and charts you can rotate to actually explore structure — without ever letting the depth apply to a number it doesn't belong to. A researcher should be able to tell, at a glance, which parts of the screen are decorative surface and which are data, because they're rendered with deliberately different depth languages (§4.1).

The UI must feel like: **Nature/Science journal meets a modern analytics tool that happens to be genuinely fun to explore.**

It must NOT feel like: generic dashboard, consumer fintech app, or "AI predicts your students' futures."

---

## 1. Design Principles

| Principle | Implementation |
|---|---|
| Research First | Scientific context accompanies every chart |
| Uncertainty Visible | Confidence intervals, error bands always shown |
| Information Dense but Readable | Dense data layouts with breathing room via whitespace |
| Trustworthy | Muted, authoritative palette; motion serves navigation, never spectacle |
| Honest Limitations | Limitations banner present on every predictive page |
| Accessible | WCAG AA minimum; colorblind-safe; no info by colour alone |
| **Dimensional, not distorted** *(new)* | Depth communicates interface hierarchy (what's a surface, what floats above it); depth in a *chart* is used only when a third real variable earns it |

---

## 2. Color System

Unchanged base palette from v1.0 (still correct — keep it):

```python
COLORS = {
    "primary": "hsl(220, 70%, 50%)", "primary_dark": "hsl(220, 70%, 35%)", "primary_light": "hsl(220, 70%, 90%)",
    "success": "hsl(155, 60%, 40%)", "warning": "hsl(38, 90%, 50%)", "danger": "hsl(4, 75%, 50%)", "neutral": "hsl(220, 15%, 60%)",
    "retention": "hsl(220, 65%, 52%)", "placement": "hsl(32, 85%, 52%)", "dlsm": "hsl(270, 55%, 52%)", "cross": "hsl(155, 55%, 42%)",
    "bg_primary": "hsl(220, 20%, 10%)", "bg_surface": "hsl(220, 18%, 14%)", "bg_elevated": "hsl(220, 16%, 18%)", "bg_border": "hsl(220, 14%, 22%)",
    "text_primary": "hsl(220, 15%, 93%)", "text_secondary": "hsl(220, 12%, 65%)", "text_disabled": "hsl(220, 10%, 40%)",
}
```

**New — Elevation & Glass tokens**, added alongside the above, not replacing it:

```python
DEPTH = {
    # Glass surfaces — translucent panels with blur, used for floating UI chrome
    # (nav rail, filter drawers, the KPI card row) — never for chart backgrounds,
    # which stay opaque per the SSIF_TEMPLATE in §4 so data stays maximally legible.
    "glass_surface":   "hsla(220, 18%, 18%, 0.55)",
    "glass_border":    "hsla(220, 30%, 70%, 0.12)",
    "glass_highlight": "hsla(220, 40%, 85%, 0.06)",   # top-edge light catch

    # Elevation shadow stack (Material-style, tuned dark-mode: shadows glow faintly
    # rather than going pure black, or they'd disappear against bg_primary)
    "shadow_1": "0 1px 2px hsla(220,60%,2%,0.4)",                                   # resting card
    "shadow_2": "0 4px 12px hsla(220,60%,2%,0.45), 0 1px 2px hsla(220,60%,2%,0.3)",  # hovered card
    "shadow_3": "0 12px 32px hsla(220,60%,2%,0.5), 0 2px 6px hsla(220,60%,2%,0.35)", # modal / active panel
    "glow_primary": "0 0 24px hsla(220,70%,55%,0.25)",   # focus ring on the active risk tier / selected point
}
```

---

## 3. Typography

Unchanged from v1.0 — Playfair Display for titles/metric emphasis, Inter for UI, JetBrains Mono for data/code. Still correct; no changes.

---

## 4. Chart Standards

Rules 1–5 from v1.0 stand exactly as written (data-ink ratio, one research question per chart, uncertainty visible, axes with units, sample sizes on every chart). Rule 6 is rewritten to be precise instead of blanket:

### 4.1 The 3D Decision Rule *(replaces v1.0's "No decorative 3D. All charts are 2D.")*

> **A chart gets a third dimension only when a third real, continuous variable is being encoded by it — never to make a 2D quantity (a bar height, a pie slice) look more dramatic.**

| Chart type | 3D allowed? | Why |
|---|---|---|
| Bar chart, any KPI comparison | **Never** | Perspective makes a "back" bar of equal value read as shorter than a "front" one. This is the single most-cited bad-practice example in data visualization for exactly that reason. |
| Pie / donut chart | **Never**, and prefer a bar chart to a pie regardless | Tilted slices of equal value read as unequal areas. |
| Line chart over time, single or few series | **Never** | Depth adds nothing a 2D line doesn't already show clearly. |
| A relationship genuinely involving 3 continuous variables (e.g., GPA × Attendance × predicted risk) | **Yes — this is what 3D is for** | The third axis carries real information you'd otherwise need two separate 2D charts and a mental cross-reference to see. |
| A per-student trajectory over time with two behavioral metrics | **Yes** | Time is a natural third axis alongside two co-evolving metrics; a 3D ribbon shows the *shape* of decline or recovery in a way small-multiples of 2D lines can't as directly. |
| Decorative hero graphic (landing/empty-state) | **Yes, and encouraged** | It's explicitly not data — see §5.3 — so it carries no risk of misreading a quantity. |

All charts still render through the same opaque `SSIF_TEMPLATE` (v1.0 §4, unchanged) — 3D charts use the same dark background, Inter/Playfair fonts, and colorway, they just get a `scene` block instead of flat `xaxis`/`yaxis`. See §5 for the three specific charts this unlocks in SSIF.

---

## 5. Dimensional Design System *(new)*

### 5.1 Elevation Levels — interface depth, not chart depth

Four levels, used consistently so a user learns the visual grammar once:

| Level | Use | Shadow | Surface |
|---|---|---|---|
| 0 — Base | Page background | none | `bg_primary` |
| 1 — Resting card | KPI cards, chart containers at rest | `shadow_1` | `bg_surface` |
| 2 — Hovered / active | Card under cursor, active nav item | `shadow_2`, translate `-2px` on Y | `bg_elevated` |
| 3 — Floating | Modals, the risk-tier detail popover, filter drawer | `shadow_3` | `glass_surface` + `backdrop-filter: blur(16px)` |

```css
.card {
    background: hsl(220, 18%, 14%);
    border: 1px solid hsl(220, 14%, 22%);
    border-radius: 10px;
    box-shadow: 0 1px 2px hsla(220,60%,2%,0.4);
    transition: transform 160ms ease, box-shadow 160ms ease;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px hsla(220,60%,2%,0.45), 0 1px 2px hsla(220,60%,2%,0.3);
}
.glass-panel {
    background: hsla(220, 18%, 18%, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid hsla(220, 30%, 70%, 0.12);
    border-radius: 14px;
    box-shadow: 0 12px 32px hsla(220,60%,2%,0.5), 0 2px 6px hsla(220,60%,2%,0.35);
}
.glass-panel::before {
    /* top-edge light catch — the detail that reads as "glass" rather than "flat grey" */
    content: "";
    position: absolute; inset: 0 0 auto 0; height: 1px;
    background: hsla(220, 40%, 85%, 0.06);
}
```

### 5.2 Motion — earns "engaging" without touching data

- **Metric cards count up** on page load (0 → final value, ~600ms ease-out) instead of appearing instantly. Applies to KPI numbers only, never to axis values inside a chart.
- **Card hover-lift** per §5.1 — the only "3D-feeling" thing happening to 2D content, and it's a UI affordance (this is clickable/expandable), not a data encoding.
- **Risk-tier badge pulses once** (subtle scale 1 → 1.04 → 1, 400ms) when a student's predicted tier changes on the Early Warning page — draws the eye to a genuine state change, not decoration for its own sake.
- No page-load skeleton shimmer longer than 400ms, no auto-rotating carousels, no parallax scrolling on data pages — motion that delays getting to the number is the enemy of "research observatory."

### 5.3 Decorative 3D hero element

For the landing/cover page and page-transition empty states only (never behind live data): a slowly-rotating abstract node-link graph — points and edges with no specific data binding, rendered once via a lightweight Three.js scene or a pre-rendered WebM loop, in the `retention`/`placement`/`dlsm` colorway. Reads as "this is a systems-analysis tool" without claiming to represent anything measured. Static PNG fallback for users with `prefers-reduced-motion` set, and for the print/PDF export path.

---

## 6. The Three Legitimate 3D Charts

Real column names from the actual pipeline (`Student_ID`, `Semester`, `Sem_GPA`, `Attendance`, `LMS_Logins`, `Failed_Courses`) — not placeholders.

### 6.1 Risk Surface — GPA × Attendance → predicted dropout probability

Shows the *interaction*, which a 2D chart would need two separate slices to convey.

```python
import numpy as np
import plotly.graph_objects as go

def render_risk_surface(model, feature_medians: dict):
    gpa_grid = np.linspace(0.0, 4.0, 40)
    att_grid = np.linspace(0.0, 100.0, 40)
    GPA, ATT = np.meshgrid(gpa_grid, att_grid)
    X = pd.DataFrame({**feature_medians, "gpa_recent_mean": GPA.ravel(), "attendance_slope": ATT.ravel()})
    Z = model.predict_proba(X)[:, 1].reshape(GPA.shape)

    fig = go.Figure(data=[go.Surface(
        x=gpa_grid, y=att_grid, z=Z,
        colorscale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"]],  # RISK_TIERS-aligned
        colorbar=dict(title="P(dropout)", tickformat=".0%"),
        contours_z=dict(show=True, usecolormap=True, project_z=True),  # 2D contour "shadow" on the floor
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="Recent GPA", yaxis_title="Attendance Slope (%)", zaxis_title="P(dropout next sem.)",
            bgcolor="hsl(220, 20%, 10%)",
        ),
        title="Predicted Dropout Risk Surface — GPA × Attendance (holdout-calibrated model)",
    )
    return apply_plotly_theme(fig)  # existing helper, applies SSIF_TEMPLATE fonts/colors on top
```

The floor contour (`project_z=True`) gives you the familiar 2D heatmap "for free" at the base of the same figure — a skeptical viewer can flatten it mentally without losing anything, which is the honesty check for whether a 3D chart earns its dimension.

### 6.2 Trajectory Ribbons — per-student path through (Semester, GPA, Attendance) space

```python
def render_trajectory_ribbons(df: pd.DataFrame, student_ids: list[str], outcome_col="Target_Dropout_Next_Sem"):
    fig = go.Figure()
    for sid in student_ids:
        s = df[df["Student_ID"] == sid].sort_values("Semester")
        color = "#EF4444" if s[outcome_col].iloc[-1] == 1 else "#10B981"
        fig.add_trace(go.Scatter3d(
            x=s["Semester"], y=s["Sem_GPA"], z=s["Attendance"],
            mode="lines+markers", line=dict(color=color, width=4), marker=dict(size=3, color=color),
            name=f"{sid}", opacity=0.75,
        ))
    fig.update_layout(
        scene=dict(xaxis_title="Semester", yaxis_title="GPA", zaxis_title="Attendance (%)",
                   bgcolor="hsl(220, 20%, 10%)"),
        title="Individual Academic Trajectories (green = persisted, red = dropped out) — sample of N students",
        showlegend=False,
    )
    return apply_plotly_theme(fig)
```

Default to a random stratified sample of ~40 students (half each outcome) — plotting all 20,000 trajectories at once turns this into visual noise, which is its own kind of dishonesty (an unreadable chart hides pattern as effectively as a misleading one does).

### 6.3 Multivariate Scatter — GPA velocity × LMS engagement × Failed Courses, colored by outcome

```python
def render_risk_scatter_3d(df_features: pd.DataFrame):
    fig = go.Figure(data=[go.Scatter3d(
        x=df_features["gpa_velocity"], y=df_features["lms_slope"], z=df_features["cumulative_failed_courses"],
        mode="markers",
        marker=dict(
            size=4,
            color=df_features["Target_Dropout_Next_Sem"].map({0: "#10B981", 1: "#EF4444"}),
            opacity=0.55,
        ),
        text=df_features["Student_ID"], hovertemplate="%{text}<extra></extra>",
    )])
    fig.update_layout(
        scene=dict(xaxis_title="GPA Velocity", yaxis_title="LMS Engagement Slope", zaxis_title="Cumulative Failed Courses",
                   bgcolor="hsl(220, 20%, 10%)"),
        title="Three-Feature Risk Space (rotate to inspect cluster separation) — N=20,000 students, one point per student",
    )
    return apply_plotly_theme(fig)
```

Caption requirement (per the existing Chart Standards, unchanged): state N, and directly under the chart, one line on what rotating it can and can't tell you — e.g., *"Visual cluster separation here is suggestive, not a substitute for the holdout AUC reported on the Modeling page — a 3D scatter can look separable and still generalize poorly."* This is the same discipline as the rest of the dashboard; the chart being genuinely three-dimensional doesn't exempt it from the uncertainty/limitation language in §7 (v1.0) below.

---

## 7. Research Language Guidelines

Unchanged from v1.0 — still the correct table (association vs. causation, CI vs. bare accuracy, N-limited framing, DLSM compatibility language). Add one row:

| ✅ Acceptable | ❌ Forbidden |
|---|---|
| "Rotate to explore — cluster separation is suggestive, see holdout AUC for the real answer" | "AI visualizes your students' risk in 3D" |

---

## 8. Icons and Visual Cues

Unchanged from v1.0.

---

## 9. Streamlit CSS Overrides

v1.0's block stays; append the elevation/glass rules from §5.1 to the same injected stylesheet. Full combined block:

```css
:root {
    --primary: hsl(220, 70%, 50%);
    --surface: hsl(220, 18%, 14%);
    --text: hsl(220, 15%, 93%);
    --border: hsl(220, 14%, 22%);
    --glass-surface: hsla(220, 18%, 18%, 0.55);
    --glass-border: hsla(220, 30%, 70%, 0.12);
}
.stApp { background-color: hsl(220, 20%, 10%); font-family: 'Inter', sans-serif; }
.metric-card { background: hsl(220, 18%, 14%); border: 1px solid hsl(220, 14%, 22%); border-radius: 8px; padding: 1.25rem;
               box-shadow: 0 1px 2px hsla(220,60%,2%,0.4); transition: transform 160ms ease, box-shadow 160ms ease; }
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px hsla(220,60%,2%,0.45); }
.limitation-banner { background: hsl(38, 70%, 15%); border-left: 4px solid hsl(38, 90%, 50%); padding: 0.75rem 1rem; border-radius: 0 6px 6px 0; font-size: 0.88rem; }
.research-context { background: hsl(220, 18%, 12%); border: 1px solid hsl(220, 65%, 35%); border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; }
.glass-panel { background: var(--glass-surface); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
               border: 1px solid var(--glass-border); border-radius: 14px; box-shadow: 0 12px 32px hsla(220,60%,2%,0.5); }
code, .monospace { font-family: 'JetBrains Mono', monospace; background: hsl(220, 20%, 8%); padding: 2px 6px; border-radius: 3px; }
@media (prefers-reduced-motion: reduce) { .metric-card, .card { transition: none; } }
```

---

## 10. Accessibility Requirements

Unchanged from v1.0, plus: every 3D chart in §6 ships with a "flatten to 2D" toggle (renders the same data as the nearest 2D equivalent — the floor contour for §6.1, a small-multiples grid for §6.2, a 2D projection pair-plot for §6.3) for users who can't use mouse-drag rotation, and `prefers-reduced-motion` disables the §5.3 rotating hero in favor of its static fallback.

---

*Document owner: Lead UX/UI Designer + Research Lead*
*Update triggers: After any major design decision; after user testing; after any new 3D chart is proposed — run it through §4.1 first.*
