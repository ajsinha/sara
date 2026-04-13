# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""
experiments/results_charts.py
==============================
Generate publication-quality ReportLab charts from aggregated ablation results.

Charts produced:
1. Capacity Curve — A-B gap vs model size (THE headline figure)
2. Condition Performance — grouped bars: KD score per condition per model
3. Domain Comparison — RAG vs Code per condition
4. Enhancement Impact — E-A and F-A gaps per model
5. Citation Fidelity — A vs B citation scores
6. Variance Profile — mean ± std per condition

All functions return ReportLab Drawing objects ready to insert into the PDF.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor, white, black
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon, Group
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.legends import Legend
from reportlab.lib.colors import Color

# ── Palette ──────────────────────────────────────────────────────────────

CRIMSON    = HexColor("#A51C30")
GOLD       = HexColor("#C49B2A")
TEAL       = HexColor("#2E8B8B")
SLATE      = HexColor("#4A5568")
GREEN      = HexColor("#38A169")
ORANGE     = HexColor("#DD6B20")
PURPLE     = HexColor("#805AD5")
RED_SOFT   = HexColor("#E53E3E")
GRAY_BG    = HexColor("#F7F7F7")
GRAY_LINE  = HexColor("#D0D0D0")
CHARCOAL   = HexColor("#2D3748")

CONDITION_COLORS = {
    "D": SLATE,
    "C": RED_SOFT,
    "B": GOLD,
    "A": CRIMSON,
    "E": TEAL,
    "F": GREEN,
}

# Model size in billions (for capacity curve x-axis)
MODEL_SIZES = {
    "tinyllama:1.1b": 1.1,
    "stablelm2:1.6b": 1.6,
    "smollm2:1.7b": 1.7,
    "gemma2:2b": 2.0,
    "llama3.2:3b": 3.0,
    "qwen2.5:3b": 3.0,
    "phi3:3.8b": 3.8,
    "llama3.1:8b": 8.0,
    "qwen2.5:7b": 7.0,
}

DW = 480  # drawing width


def load_results(path: Path = None) -> dict:
    p = path or Path(__file__).parent / "results" / "aggregated_results.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ── Chart 1: Capacity Curve ─────────────────────────────────────────────

def chart_capacity_curve(agg: dict, width=DW, height=220) -> list:
    """A-B gap (y-axis) vs student model size (x-axis). THE headline figure."""
    configs = agg.get("configs", [])
    if not configs:
        return []

    d = Drawing(width, height)

    # Background
    d.add(Rect(0, 0, width, height, fillColor=GRAY_BG, strokeColor=None))

    # Collect data points: (size_B, ab_gap, label)
    points = []
    for cfg in configs:
        student = cfg.get("student", "")
        size = MODEL_SIZES.get(student, 0)
        gap = cfg.get("ab_gap_mean", 0)
        if size > 0 and cfg.get("conditions", {}).get("D", {}).get("mean_kd", 0) > 0:
            points.append((size, gap, student.split(":")[0]))

    if not points:
        return []

    points.sort(key=lambda x: x[0])

    # Chart area
    margin_l, margin_r, margin_b, margin_t = 60, 20, 45, 25
    cw = width - margin_l - margin_r
    ch = height - margin_b - margin_t

    # Scale
    x_min = min(p[0] for p in points) - 0.2
    x_max = max(p[0] for p in points) + 0.2
    gaps = [p[1] for p in points]
    y_min = min(min(gaps) - 0.01, -0.03)
    y_max = max(max(gaps) + 0.01, 0.03)

    def sx(v): return margin_l + (v - x_min) / (x_max - x_min) * cw
    def sy(v): return margin_b + (v - y_min) / (y_max - y_min) * ch

    # Zero line (hypothesis threshold)
    y0 = sy(0)
    d.add(Line(margin_l, y0, width - margin_r, y0,
               strokeColor=HexColor("#999999"), strokeWidth=1, strokeDashArray=[4, 3]))
    d.add(String(margin_l - 5, y0 - 3, "0", fontName="Helvetica", fontSize=7,
                 fillColor=CHARCOAL, textAnchor="end"))

    # Threshold line at +0.02
    y02 = sy(0.02)
    if y_min < 0.02 < y_max:
        d.add(Line(margin_l, y02, width - margin_r, y02,
                   strokeColor=GREEN, strokeWidth=0.8, strokeDashArray=[2, 2]))
        d.add(String(width - margin_r + 2, y02 - 3, "strong", fontName="Helvetica",
                     fontSize=6, fillColor=GREEN))

    # Plot points and connect
    prev_x, prev_y = None, None
    for size, gap, label in points:
        px, py = sx(size), sy(gap)
        # Connect
        if prev_x is not None:
            d.add(Line(prev_x, prev_y, px, py,
                       strokeColor=CRIMSON, strokeWidth=1.5))
        # Point
        color = GREEN if gap > 0.02 else (GOLD if gap > 0 else RED_SOFT)
        d.add(Rect(px - 4, py - 4, 8, 8, fillColor=color, strokeColor=white, strokeWidth=1))
        # Label
        d.add(String(px, py + 8, f"{gap:+.3f}", fontName="Helvetica-Bold",
                     fontSize=7, fillColor=color, textAnchor="middle"))
        d.add(String(px, margin_b - 18, label, fontName="Helvetica",
                     fontSize=7, fillColor=CHARCOAL, textAnchor="middle"))
        d.add(String(px, margin_b - 28, f"{size}B", fontName="Helvetica",
                     fontSize=6, fillColor=SLATE, textAnchor="middle"))
        prev_x, prev_y = px, py

    # Axes
    d.add(Line(margin_l, margin_b, margin_l, height - margin_t,
               strokeColor=CHARCOAL, strokeWidth=1))
    d.add(Line(margin_l, margin_b, width - margin_r, margin_b,
               strokeColor=CHARCOAL, strokeWidth=1))

    # Axis labels
    d.add(String(width / 2, 5, "Student Model Size (Billion Parameters)",
                 fontName="Helvetica", fontSize=8, fillColor=CHARCOAL, textAnchor="middle"))
    d.add(String(12, height / 2, "A\u2212B Gap",
                 fontName="Helvetica-Bold", fontSize=8, fillColor=CRIMSON, textAnchor="middle"))

    # Title
    d.add(String(width / 2, height - 10,
                 "Figure 20.1 \u2014 Capacity Curve: Self-Knowledge A\u2212B Gap vs Student Model Size",
                 fontName="Helvetica-Bold", fontSize=8, fillColor=CHARCOAL, textAnchor="middle"))

    return [d]


# ── Chart 2: Condition Performance Bars ──────────────────────────────────

def chart_condition_bars(agg: dict, width=DW, height=200) -> list:
    """Grouped bar chart: KD score per condition, grouped by student model."""
    configs = [c for c in agg.get("configs", [])
               if c.get("conditions", {}).get("D", {}).get("mean_kd", 0) > 0]
    if not configs:
        return []

    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=GRAY_BG, strokeColor=None))

    conditions = ["D", "C", "B", "A", "E", "F"]
    n_configs = len(configs)
    n_conds = len(conditions)

    margin_l, margin_r, margin_b, margin_t = 50, 15, 50, 25
    cw = width - margin_l - margin_r
    ch = height - margin_b - margin_t

    group_width = cw / max(n_configs, 1)
    bar_width = group_width / (n_conds + 1)

    # Find y range
    all_kd = []
    for cfg in configs:
        for c in conditions:
            kd = cfg.get("conditions", {}).get(c, {}).get("mean_kd", 0)
            all_kd.append(kd)
    y_max = min(max(all_kd) + 0.05, 1.0) if all_kd else 1.0
    y_min = max(min(all_kd) - 0.05, 0) if all_kd else 0

    def sy(v): return margin_b + (v - y_min) / (y_max - y_min) * ch

    # Y-axis gridlines
    for yv in [round(y_min + i * (y_max - y_min) / 5, 2) for i in range(6)]:
        yp = sy(yv)
        d.add(Line(margin_l, yp, width - margin_r, yp,
                   strokeColor=GRAY_LINE, strokeWidth=0.3))
        d.add(String(margin_l - 5, yp - 3, f"{yv:.2f}", fontName="Helvetica",
                     fontSize=6, fillColor=SLATE, textAnchor="end"))

    # Bars
    for gi, cfg in enumerate(configs):
        gx = margin_l + gi * group_width
        label = cfg.get("student", "").split(":")[0]

        for ci, cond in enumerate(conditions):
            kd = cfg.get("conditions", {}).get(cond, {}).get("mean_kd", 0)
            if kd == 0:
                continue
            bx = gx + ci * bar_width + bar_width * 0.1
            bw = bar_width * 0.8
            by = sy(y_min)
            bh = sy(kd) - by
            color = CONDITION_COLORS.get(cond, SLATE)
            d.add(Rect(bx, by, bw, max(bh, 1), fillColor=color, strokeColor=None))

        # Model label
        d.add(String(gx + group_width / 2, margin_b - 15, label,
                     fontName="Helvetica-Bold", fontSize=7, fillColor=CHARCOAL,
                     textAnchor="middle"))

    # Legend
    lx = width - margin_r - 120
    for i, cond in enumerate(conditions):
        ly = height - margin_t - 5 - i * 11
        d.add(Rect(lx, ly, 8, 8, fillColor=CONDITION_COLORS.get(cond, SLATE), strokeColor=None))
        labels = {"D": "Baseline", "C": "Random", "B": "External", "A": "KD-SPAR",
                  "E": "MetaKDSPAR", "F": "Enhanced"}
        d.add(String(lx + 12, ly, f"{cond}: {labels.get(cond, cond)}", fontName="Helvetica",
                     fontSize=6, fillColor=CHARCOAL))

    # Axes
    d.add(Line(margin_l, margin_b, margin_l, height - margin_t,
               strokeColor=CHARCOAL, strokeWidth=1))
    d.add(Line(margin_l, margin_b, width - margin_r, margin_b,
               strokeColor=CHARCOAL, strokeWidth=1))

    d.add(String(width / 2, height - 10,
                 "Figure 20.2 \u2014 KD Score by Condition and Student Model",
                 fontName="Helvetica-Bold", fontSize=8, fillColor=CHARCOAL, textAnchor="middle"))
    d.add(String(12, height / 2, "KD Score",
                 fontName="Helvetica-Bold", fontSize=8, fillColor=CRIMSON, textAnchor="middle"))

    return [d]


# ── Chart 3: Enhancement Impact ──────────────────────────────────────────

def chart_enhancement_impact(agg: dict, width=DW, height=180) -> list:
    """E-A gap and F-A gap per student model."""
    configs = [c for c in agg.get("configs", [])
               if c.get("conditions", {}).get("A", {}).get("mean_kd", 0) > 0]
    if not configs:
        return []

    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=GRAY_BG, strokeColor=None))

    margin_l, margin_r, margin_b, margin_t = 50, 15, 50, 25
    cw = width - margin_l - margin_r
    ch = height - margin_b - margin_t

    # Compute gaps
    data = []
    for cfg in configs:
        conds = cfg.get("conditions", {})
        a_kd = conds.get("A", {}).get("mean_kd", 0)
        e_kd = conds.get("E", {}).get("mean_kd", 0)
        f_kd = conds.get("F", {}).get("mean_kd", 0)
        b_kd = conds.get("B", {}).get("mean_kd", 0)
        label = cfg.get("student", "").split(":")[0]
        if a_kd > 0:
            data.append({
                "label": label,
                "ea": e_kd - a_kd,
                "fa": f_kd - a_kd,
                "fb": f_kd - b_kd,
            })

    if not data:
        return []

    n = len(data)
    group_w = cw / max(n, 1)
    bar_w = group_w / 4

    all_vals = [d2["ea"] for d2 in data] + [d2["fa"] for d2 in data] + [d2["fb"] for d2 in data]
    y_min = min(min(all_vals) - 0.01, -0.02)
    y_max = max(max(all_vals) + 0.01, 0.05)

    def sy(v): return margin_b + (v - y_min) / (y_max - y_min) * ch

    # Zero line
    y0 = sy(0)
    d.add(Line(margin_l, y0, width - margin_r, y0,
               strokeColor=HexColor("#999999"), strokeWidth=0.8, strokeDashArray=[3, 2]))

    for i, item in enumerate(data):
        gx = margin_l + i * group_w
        for j, (key, color, label) in enumerate([
            ("ea", TEAL, "E\u2212A"),
            ("fa", GREEN, "F\u2212A"),
            ("fb", GOLD, "F\u2212B"),
        ]):
            val = item[key]
            bx = gx + (j + 0.5) * bar_w
            by = sy(0)
            bh = sy(val) - by
            d.add(Rect(bx, min(by, by + bh), bar_w * 0.8, abs(bh),
                       fillColor=color, strokeColor=None))
            d.add(String(bx + bar_w * 0.4, max(by, by + bh) + 3,
                         f"{val:+.3f}", fontName="Helvetica", fontSize=6,
                         fillColor=color, textAnchor="middle"))

        d.add(String(gx + group_w / 2, margin_b - 15, item["label"],
                     fontName="Helvetica-Bold", fontSize=7, fillColor=CHARCOAL,
                     textAnchor="middle"))

    # Legend
    for i, (label, color) in enumerate([("E\u2212A", TEAL), ("F\u2212A", GREEN), ("F\u2212B", GOLD)]):
        lx = width - margin_r - 80
        ly = height - margin_t - 5 - i * 11
        d.add(Rect(lx, ly, 8, 8, fillColor=color, strokeColor=None))
        d.add(String(lx + 12, ly, label, fontName="Helvetica", fontSize=7, fillColor=CHARCOAL))

    d.add(Line(margin_l, margin_b, margin_l, height - margin_t,
               strokeColor=CHARCOAL, strokeWidth=1))

    d.add(String(width / 2, height - 10,
                 "Figure 20.3 \u2014 Enhancement Impact: E\u2212A, F\u2212A, and F\u2212B Gaps by Model",
                 fontName="Helvetica-Bold", fontSize=8, fillColor=CHARCOAL, textAnchor="middle"))

    return [d]


# ── Generate all charts ──────────────────────────────────────────────────

def generate_all_charts(agg: dict) -> dict:
    """Generate all chart Drawing objects. Returns dict of name -> [Drawing]."""
    return {
        "capacity_curve": chart_capacity_curve(agg),
        "condition_bars": chart_condition_bars(agg),
        "enhancement_impact": chart_enhancement_impact(agg),
    }


if __name__ == "__main__":
    agg = load_results()
    if not agg:
        print("No aggregated results found.")
    else:
        charts = generate_all_charts(agg)
        for name, drawings in charts.items():
            print(f"  {name}: {len(drawings)} drawing(s)")
