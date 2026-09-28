# design.md — Visual Design System
# Student Success Intelligence Framework (SSIF)

**Version:** 1.0  
**Date:** 2026-09-29  
**Design Philosophy:** Research Observatory — not a consumer app

---

## 0. Core Design Identity

The SSIF dashboard must communicate:

> **"This is a computational research laboratory. Every number has uncertainty. Every model has limitations. This system helps you investigate — it does not tell you what to do."**

The UI must feel like: **Nature/Science journal meets modern analytics tool.**

It must NOT feel like: generic dashboard, consumer fintech app, or "AI predicts your students' futures."

---

## 1. Design Principles

| Principle | Implementation |
|---|---|
| **Research First** | Scientific context accompanies every chart |
| **Uncertainty Visible** | Confidence intervals, error bands always shown |
| **Information Dense but Readable** | Dense data layouts with breathing room via whitespace |
| **Trustworthy** | Muted, authoritative palette; no gimmicky animations |
| **Honest Limitations** | Limitations banner present on every predictive page |
| **Accessible** | WCAG AA minimum; colorblind-safe; no info by colour alone |

---

## 2. Color System

### Primary Palette (HSL-calibrated)

```python
COLORS = {
    # Brand / UI
    "primary":        "hsl(220, 70%, 50%)",   # Calibrated blue — trust, science
    "primary_dark":   "hsl(220, 70%, 35%)",
    "primary_light":  "hsl(220, 70%, 90%)",

    # Semantic — Research outcomes
    "success":        "hsl(155, 60%, 40%)",   # Teal-green — positive outcome
    "warning":        "hsl(38, 90%, 50%)",    # Amber — caution / watch
    "danger":         "hsl(4, 75%, 50%)",     # Muted red — high risk
    "neutral":        "hsl(220, 15%, 60%)",   # Cool grey — neutral/uncertain

    # Dataset identity colours
    "retention":      "hsl(220, 65%, 52%)",   # Blue family — Dataset A
    "placement":      "hsl(32, 85%, 52%)",    # Amber family — Dataset B
    "dlsm":           "hsl(270, 55%, 52%)",   # Violet — DLSM (independent layer)
    "cross":          "hsl(155, 55%, 42%)",   # Teal — cross-dataset analysis

    # Background system (dark theme, research-lab aesthetic)
    "bg_primary":     "hsl(220, 20%, 10%)",   # Near-black — main background
    "bg_surface":     "hsl(220, 18%, 14%)",   # Card / panel surface
    "bg_elevated":    "hsl(220, 16%, 18%)",   # Slightly elevated panels
    "bg_border":      "hsl(220, 14%, 22%)",   # Subtle borders

    # Text
    "text_primary":   "hsl(220, 15%, 93%)",   # Near-white
    "text_secondary": "hsl(220, 12%, 65%)",   # Muted — captions, labels
    "text_disabled":  "hsl(220, 10%, 40%)",   # Greyed out
}
```

### Colorblind-Safe Sequential Palette (Plotly-ready)

```python
# Based on Okabe-Ito colorblind-safe palette, adapted
CB_SAFE = [
    "#3B82F6",   # Blue
    "#F59E0B",   # Amber
    "#10B981",   # Teal
    "#8B5CF6",   # Violet
    "#EF4444",   # Red
    "#06B6D4",   # Cyan
    "#EC4899",   # Pink
    "#84CC16",   # Lime
]

# For binary outcomes (Placed/Not Placed, Enrolled/Dropped_Out)
BINARY_PAIR = {
    "positive": "#10B981",    # Teal — Placed / Enrolled
    "negative": "#EF4444",    # Red — Not Placed / Dropped Out
    "censored": "#6B7280",    # Grey — Censored / Unknown
}

# For risk tiers (Early Warning System)
RISK_TIERS = {
    "low":       "#10B981",
    "watch":     "#F59E0B",
    "elevated":  "#F97316",
    "high":      "#EF4444",
}
```

**Rule:** No information is communicated by color alone. Every colored element also uses shape, pattern, or label.

---

## 3. Typography

```css
/* Google Fonts imports */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&display=swap');
```

| Element | Font | Weight | Size |
|---|---|---|---|
| Page title | Playfair Display | 700 | 2.2rem |
| Section heading | Inter | 600 | 1.4rem |
| Card title | Inter | 600 | 1.1rem |
| Body text | Inter | 400 | 0.95rem |
| Caption / annotation | Inter | 300 | 0.82rem |
| Metric value | Inter | 700 | 2.0rem |
| Metric label | Inter | 400 | 0.85rem |
| Code / data | JetBrains Mono | 400 | 0.88rem |
| Axis labels | Inter | 400 | 0.80rem |
| Table header | Inter | 600 | 0.90rem |

---

## 4. Chart Standards

All charts must follow **Tufte's principles of analytical design**:

1. **Data-ink ratio:** Remove all non-data ink. No gratuitous gridlines, borders, or decorations.
2. **Every chart answers exactly one research question.** The question must appear as a subtitle.
3. **Uncertainty must be visible.** Error bars, CI bands, or bootstrap ranges where applicable.
4. **Axes must include units.** "Sem_GPA" not "GPA." "AUC-ROC (5-fold GroupKFold)" not "AUC."
5. **Sample sizes must appear on charts.** "N=20,000 students, 79,239 observations."
6. **No decorative 3D.** All charts are 2D.

### Plotly Template

```python
import plotly.graph_objects as go
import plotly.io as pio

SSIF_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor="hsl(220, 20%, 10%)",
        plot_bgcolor="hsl(220, 18%, 14%)",
        font=dict(family="Inter", color="hsl(220, 15%, 93%)", size=12),
        title=dict(font=dict(family="Playfair Display", size=18)),
        xaxis=dict(
            gridcolor="hsl(220, 14%, 22%)",
            linecolor="hsl(220, 14%, 22%)",
            zerolinecolor="hsl(220, 14%, 22%)",
        ),
        yaxis=dict(
            gridcolor="hsl(220, 14%, 22%)",
            linecolor="hsl(220, 14%, 22%)",
            zerolinecolor="hsl(220, 14%, 22%)",
        ),
        colorway=["#3B82F6", "#F59E0B", "#10B981", "#8B5CF6", "#EF4444"],
        legend=dict(
            bgcolor="hsl(220, 18%, 17%)",
            bordercolor="hsl(220, 14%, 25%)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="hsl(220, 20%, 8%)",
            bordercolor="hsl(220, 14%, 30%)",
            font=dict(family="JetBrains Mono", size=12),
        ),
    )
)
pio.templates["ssif"] = SSIF_TEMPLATE
pio.templates.default = "ssif"
```

---

## 5. Component Library

### Metric Card (KPI display)

```python
def render_metric_card(title: str, value: str, delta: str = None,
                        uncertainty: str = None, source: str = None):
    """
    Display a research metric with optional delta and uncertainty annotation.

    Args:
        title: Metric name (e.g., "Dropout AUC-ROC")
        value: Formatted value string (e.g., "0.814")
        delta: Change vs baseline (e.g., "+0.031 vs logistic")
        uncertainty: CI string (e.g., "95% CI [0.798, 0.829]")
        source: Dataset/model source tag
    """
    # Rendered as Streamlit metric with custom CSS overlay
```

### Research Context Panel

Every analytical page must begin with:

```
┌──────────────────────────────────────────────────────────────────┐
│  📊 DATASET: Academic Persistence (N=20,000 students, 79,239 obs) │
│  🔬 METHOD: GroupKFold (k=5, groups=Student_ID)                   │
│  ⚠️  LIMITATION: Observational data — associations, not causation │
└──────────────────────────────────────────────────────────────────┘
```

### Scientific Limitation Banner

All pages with predictive models must include a persistent amber banner:

```
⚠️  MODEL OUTPUTS ARE STATISTICAL ESTIMATES — NOT DECISIONS
    These results reflect associations in observational data.
    Confidence intervals are shown. No causal claims are warranted.
    See Page 9 for full scientific limitations.
```

### Uncertainty Indicator

When displaying probability estimates:

```
Placement Probability: 0.74
[████████████████░░░░] 74%
                95% CI: [0.68 — 0.81]
                Calibration Error: 0.032
```

---

## 6. Dashboard Page Templates

### Template A — Analysis Page

```
Header: [Dataset pill] [Page title] [N indicator]
├── Research Context Panel (dataset, method, limitation)
├── [Left column — primary chart]    [Right column — metrics/stats]
├── [Expandable: Model Details / Assumptions]
└── [Footer: experiment_id | mlflow_run | timestamp]
```

### Template B — Comparison Page

```
Header: [Cross-Dataset Comparison]
├── Research Context Panel
├── [LEFT — Dataset A results]    [RIGHT — Dataset B results]
├── [CENTER — Statistical comparison]
└── [Interpretation: what is defensible / what is NOT]
```

### Template C — Explainability Page

```
Header: [XAI / Explainability]
├── ⚠️  Reminder: SHAP = attribution, not causation
├── [Global SHAP summary plot]
├── [Feature importance bar chart]
├── [Dependence plots — top 3 features]
├── [Local waterfall plots — 3 example students]
└── [Interpretation notes]
```

---

## 7. Research Language Guidelines

| ✅ Acceptable | ❌ Forbidden |
|---|---|
| "The model estimated..." | "The model knows..." |
| "Statistically associated with dropout" | "Causes dropout" |
| "Elevated risk probability" | "Will drop out" |
| "95% CI [0.70, 0.85]" | "Accuracy: 85%" (alone) |
| "Exploratory finding — N=215 limits confidence" | "Proven that..." |
| "DLSM compatibility: NO-GO — variables absent" | "DLSM applied" |
| "Representation comparison (different populations)" | "Dataset merge" |
| "Observational associations" | "Causal pathway" |

---

## 8. Icons and Visual Cues

| Element | Icon | Color |
|---|---|---|
| Dataset A (Retention) | 📚 | Blue |
| Dataset B (Placement) | 💼 | Amber |
| DLSM | 🧠 | Violet |
| Cross-dataset | 🔗 | Teal |
| Warning / limitation | ⚠️ | Amber |
| Success / positive | ✅ | Teal |
| Error / incompatible | ❌ | Red |
| Uncertainty / CI | 〰️ | Grey |
| Research finding | 🔬 | Blue |
| Sample size | N= | Monospace |

---

## 9. Streamlit CSS Overrides

```css
/* Streamlit Global Overrides (injected via st.markdown) */
:root {
    --primary: hsl(220, 70%, 50%);
    --surface: hsl(220, 18%, 14%);
    --text: hsl(220, 15%, 93%);
    --border: hsl(220, 14%, 22%);
}

.stApp {
    background-color: hsl(220, 20%, 10%);
    font-family: 'Inter', sans-serif;
}

.metric-card {
    background: hsl(220, 18%, 14%);
    border: 1px solid hsl(220, 14%, 22%);
    border-radius: 8px;
    padding: 1.25rem;
}

.limitation-banner {
    background: hsl(38, 70%, 15%);
    border-left: 4px solid hsl(38, 90%, 50%);
    padding: 0.75rem 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
}

.research-context {
    background: hsl(220, 18%, 12%);
    border: 1px solid hsl(220, 65%, 35%);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
}

code, .monospace {
    font-family: 'JetBrains Mono', monospace;
    background: hsl(220, 20%, 8%);
    padding: 2px 6px;
    border-radius: 3px;
}
```

---

## 10. Accessibility Requirements

- All contrast ratios must meet WCAG AA: text ≥ 4.5:1, large text ≥ 3:1
- All charts must have `alt` text descriptions for screen readers
- Error states must use both colour and icon (not colour alone)
- Risk tiers must use shape + label in addition to colour
- Font sizes must not go below 12px in any rendered element

---

*Document owner: Lead UX/UI Designer + Research Lead*  
*Update triggers: After any major design decision; after user testing*
