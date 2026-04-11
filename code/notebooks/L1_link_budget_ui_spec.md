# UI Specification: L1 Link Budget Calculator

**Agent:** sip-ui-architect
**Date:** 2026-04-11
**Design system version:** 1.0
**Artifact:** `code/notebooks/L1_link_budget_calculator.ipynb`

---

## Color palette

| Role | Hex | Applied to |
|------|-----|-----------|
| Background | `#000000` | Figure facecolor |
| Surface | `#111111` | Axes facecolor, legend background |
| Primary text | `#ffffff` | Titles, axis labels, annotations |
| Secondary text | `#888888` | Tick labels |
| Grid | `#333333` | Grid lines (linewidth 0.5) |
| Optical/Photonic | `#06b6d4` | All optical loss bars, link margin lines |
| Electrical/EIC | `#f59e0b` | Not used in L1 notebook |
| Pass | `#22c55e` | PASS status, positive margin annotation |
| Fail / RAS | `#ef4444` | FAIL status, negative margin, BER floor violation |
| Reference | `#64748b` | Zero-margin reference line, KP4 BER threshold line |
| Neutral bar | `#475569` | Fiber loss bar (not photonic component — distinct color) |

---

## Chart inventory

| # | Output | Chart type | Cell | Title |
|---|--------|-----------|------|-------|
| 1 | `budget_breakdown` | Horizontal waterfall bar | Cell 7 | "Optical Link Budget — Loss Breakdown" |
| 2 | `link_margin_dB` vs `coupler_loss_dB` | Line plot | Cell 8 | "Link Margin vs. Fiber-Chip Coupling Loss" |
| 3 | `link_margin_dB` vs `fiber_length_km` | Line plot | Cell 9 | "Link Margin vs. Fiber Span Length" |
| 4 | All outputs + pass/fail | Styled DataFrame table | Cell 10 | "Link Budget Summary" |

---

## Layout — notebook cell order

```
Cell 0  [Markdown]  Title block
Cell 1  [Code]      Imports
Cell 2  [Code]      Constants (corpus-sourced)
Cell 3  [Code]      rcParams block (MANDATORY — copy verbatim from below)
Cell 4  [Code]      Import optical_math + COLORS dict
Cell 5  [Code]      ipywidgets parameter sliders
Cell 6  [Code]      Compute outputs (interactive_output or observe)
Cell 7  [Code]      Chart 1: Waterfall loss breakdown
Cell 8  [Code]      Chart 2: Margin vs. coupler loss sweep
Cell 9  [Code]      Chart 3: Margin vs. fiber length sweep
Cell 10 [Code]      Summary table
Cell 11 [Markdown]  References
```

---

## Chart specifications

### Chart 1 — Waterfall Loss Breakdown (Cell 7)
- Type: Horizontal bar chart
- Figure size: 10 × 5 inches
- Y-axis: component names (bottom to top: Fiber-chip coupling ×2, Waveguide routing, Modulator IL, MUX/DEMUX, Fiber span)
- X-axis: Loss (dB), starts at 0
- Bar colors: optical components → `#06b6d4`; fiber span → `#475569`
- Overlay: vertical dashed line at total loss value, color `#ef4444`, label "Total: {X} dB"
- Annotation: each bar labeled with its dB value (white text, right-aligned inside bar if > 0.5 dB, else right of bar)
- Grid: vertical grid lines only, `#333333`, linewidth 0.5
- No legend needed — bars are self-labeled

### Chart 2 — Margin vs. Coupler Loss Sweep (Cell 8)
- Type: Line plot
- Figure size: 10 × 5 inches
- X-axis: Coupler loss per facet (dB), range 0.08 to 4.5, 100 points
- Y-axis: Link margin (dB)
- Line color: `#06b6d4` (optical domain)
- Reference line: y = 0, color `#64748b`, linestyle `--`, linewidth 1.0, label "Margin = 0 (link closure threshold)"
- Marker: vertical dotted line at current widget value of `coupler_loss_dB`, color `#f59e0b`, label "Current setting"
- Fill: `fill_between` above y=0 with `#22c55e` alpha 0.08 (PASS region); below y=0 with `#ef4444` alpha 0.08 (FAIL region)
- Annotation: mark the intersection point (coupler loss at margin = 0) with a dot + label "Closure limit: {X} dB"
- Grid: both axes, `#333333`, linewidth 0.5

### Chart 3 — Margin vs. Fiber Length Sweep (Cell 9)
- Type: Line plot
- Figure size: 10 × 5 inches
- X-axis: Fiber length (km), range 0 to 10, 100 points
- Y-axis: Link margin (dB)
- Line color: `#06b6d4`
- Reference line: y = 0, same style as Chart 2
- Marker: vertical dotted line at current `fiber_length_km` widget value, color `#f59e0b`
- Fill: PASS/FAIL regions same as Chart 2
- Annotation: max reach at margin = 0, labeled "Max reach: {X} km"
- Secondary annotation: vertical line at 10 km (400G-LR8 spec limit), color `#64748b`, label "LR8 spec limit"

### Chart 4 — Summary Table (Cell 10)
- Type: `pandas.DataFrame.style`
- Columns: Parameter | Value | Units | Confidence | Source
- Rows: all inputs + computed outputs
- Cell styling:
  - `link_status` row: background `#22c55e` (PASS) or `#ef4444` (FAIL), text `#000000`
  - MEDIUM-confidence rows: left border `#f59e0b` 2px
  - PROVISIONAL rows: italic text, `#888888`
- Table background: `#111111`
- Header background: `#000000`, text `#ffffff`
- No index column

---

## rcParams block (MANDATORY — sip-code-writer must use this verbatim in Cell 3)

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'figure.facecolor':  '#000000',
    'axes.facecolor':    '#111111',
    'axes.edgecolor':    '#444444',
    'axes.labelcolor':   '#ffffff',
    'xtick.color':       '#888888',
    'ytick.color':       '#888888',
    'text.color':        '#ffffff',
    'grid.color':        '#333333',
    'grid.linewidth':    0.5,
    'legend.facecolor':  '#111111',
    'legend.edgecolor':  '#111111',
    'legend.labelcolor': '#ffffff',
    'figure.dpi':        150,
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.titlesize':    14,
    'axes.titleweight':  'bold',
    'axes.titlecolor':   '#ffffff',
})

COLORS = {
    'optical':    '#06b6d4',
    'electrical': '#f59e0b',
    'firmware':   '#10b981',
    'cmis':       '#8b5cf6',
    'ras':        '#ef4444',
    'pass':       '#22c55e',
    'reference':  '#64748b',
    'neutral':    '#475569',
    'text':       '#ffffff',
    'surface':    '#111111',
}
```

---

## Special cases

None. All chart types are in the standard rules table. No deviations.
