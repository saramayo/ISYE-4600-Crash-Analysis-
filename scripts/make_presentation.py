"""
Generate a professional, minimalistic PowerPoint presentation for ISYE 4600.
"""
from __future__ import annotations
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── palette ──────────────────────────────────────────────────────────────────
NAVY    = RGBColor(0x1A, 0x35, 0x5E)   # headings, accent bar
BLUE    = RGBColor(0x2E, 0x75, 0xB6)   # subheadings, table headers
LTBLUE  = RGBColor(0xD6, 0xE4, 0xF0)   # table row tint, divider
GOLD    = RGBColor(0xC5, 0x9A, 0x2E)   # accent bullet / highlight number
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1A, 0x1A, 0x2E)   # body text
MGRAY   = RGBColor(0x6B, 0x6B, 0x6B)   # captions / footnotes
LGRAY   = RGBColor(0xF2, 0xF4, 0xF7)   # slide background

W  = Inches(13.333)   # widescreen 16:9
H  = Inches(7.5)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "Presentation" / "ISYE4600_AV_Crash_Severity.pptx"

# ── helpers ───────────────────────────────────────────────────────────────────
def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs: Presentation):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)


def bg(slide, color: RGBColor = LGRAY):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def accent_bar(slide, color: RGBColor = NAVY, width: float = 0.07):
    """Left vertical accent strip."""
    bar = slide.shapes.add_shape(1, 0, 0, Inches(width), H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()


def txb(slide, text, l, t, w, h,
        size=20, bold=False, color=DARK, align=PP_ALIGN.LEFT,
        italic=False, wrap=True):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    box.word_wrap = wrap
    tf  = box.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size   = Pt(size)
    run.font.bold   = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return box


def heading(slide, title, subtitle=None):
    """Standard section heading with navy underline."""
    txb(slide, title, 0.55, 0.18, 12.3, 0.7, size=30, bold=True, color=NAVY)
    # underline rule
    rule = slide.shapes.add_shape(1, Inches(0.55), Inches(0.95),
                                  Inches(12.3), Pt(2))
    rule.fill.solid(); rule.fill.fore_color.rgb = GOLD
    rule.line.fill.background()
    if subtitle:
        txb(slide, subtitle, 0.55, 1.0, 12.0, 0.45, size=15,
            color=MGRAY, italic=True)


def bullet_box(slide, items: list[tuple[str, str]], l, t, w, h,
               size=17, gap=0.0):
    """
    items = list of (bullet_char, text).
    bullet_char can be "•", "–", a number string, or "" for sub-bullets.
    """
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    box.word_wrap = True
    tf  = box.text_frame
    tf.word_wrap = True
    first = True
    for bchar, text in items:
        if first:
            p = tf.paragraphs[0]; first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(gap)
        if bchar:
            r1 = p.add_run()
            r1.text = bchar + "  "
            r1.font.size  = Pt(size)
            r1.font.bold  = bchar not in ("–", "·")
            r1.font.color.rgb = GOLD if bchar not in ("–", "·") else BLUE
        r2 = p.add_run()
        r2.text = text
        r2.font.size  = Pt(size)
        r2.font.color.rgb = DARK
    return box


def table_slide(slide, headers: list[str], rows: list[list[str]],
                l=0.55, t=1.55, col_widths=None):
    n_cols = len(headers)
    n_rows = len(rows) + 1
    if col_widths is None:
        cw = (12.3 / n_cols)
        col_widths = [cw] * n_cols

    tbl = slide.shapes.add_table(
        n_rows, n_cols,
        Inches(l), Inches(t),
        Inches(sum(col_widths)), Inches(0.42 * n_rows)
    ).table

    for ci, (w, h) in enumerate(zip(col_widths, [None]*n_cols)):
        tbl.columns[ci].width = Inches(col_widths[ci])

    def cell_fmt(cell, text, header=False, shade=False, align=PP_ALIGN.LEFT, bold_val=False):
        cell.text = text
        p = cell.text_frame.paragraphs[0]
        p.alignment = align
        run = p.runs[0] if p.runs else p.add_run()
        run.font.size  = Pt(14 if header else 13)
        run.font.bold  = header or bold_val
        run.font.color.rgb = WHITE if header else DARK
        fill = cell.fill
        fill.solid()
        if header:
            fill.fore_color.rgb = NAVY
        elif shade:
            fill.fore_color.rgb = LTBLUE
        else:
            fill.fore_color.rgb = WHITE

    for ci, h in enumerate(headers):
        cell_fmt(tbl.cell(0, ci), h, header=True, align=PP_ALIGN.CENTER)

    for ri, row in enumerate(rows):
        shade = (ri % 2 == 0)
        for ci, val in enumerate(row):
            bold_val = (ci == 0)
            cell_fmt(tbl.cell(ri + 1, ci), val, shade=shade,
                     align=PP_ALIGN.CENTER if ci > 0 else PP_ALIGN.LEFT,
                     bold_val=bold_val)
    return tbl


def footnote(slide, text, t=7.1):
    txb(slide, text, 0.55, t, 12.5, 0.35, size=11, color=MGRAY, italic=True)


# ═══════════════════════════════════════════════════════════════════════════
# SLIDES
# ═══════════════════════════════════════════════════════════════════════════

def slide_title(prs):
    s = blank_slide(prs)
    # full navy background
    fill = s.background.fill; fill.solid(); fill.fore_color.rgb = NAVY
    # gold top strip
    strip = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    strip.fill.solid(); strip.fill.fore_color.rgb = GOLD
    strip.line.fill.background()
    # gold bottom strip
    strip2 = s.shapes.add_shape(1, 0, Inches(7.38), W, Inches(0.12))
    strip2.fill.solid(); strip2.fill.fore_color.rgb = GOLD
    strip2.line.fill.background()
    # title
    txb(s, "Predicting Crash Severity in\nAutonomous Vehicle Incidents",
        1.0, 1.8, 11.3, 1.9, size=38, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    # divider line
    div = s.shapes.add_shape(1, Inches(4.0), Inches(3.85), Inches(5.3), Pt(2))
    div.fill.solid(); div.fill.fore_color.rgb = GOLD; div.line.fill.background()
    # subtitle
    txb(s, "ISYE 4600  ·  Spring 2026",
        1.0, 4.0, 11.3, 0.5, size=18, color=LTBLUE, align=PP_ALIGN.CENTER)
    txb(s, "Santiago Aramayo Velasco  ·  Lauren McDonald  ·  Luis Velez",
        1.0, 4.55, 11.3, 0.5, size=15, color=LTBLUE, align=PP_ALIGN.CENTER)
    txb(s, "Georgia Institute of Technology",
        1.0, 5.1, 11.3, 0.4, size=13, color=MGRAY, align=PP_ALIGN.CENTER, italic=True)


def slide_agenda(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "Agenda")
    items = [
        ("01", "Problem Statement & Stakeholders"),
        ("02", "Data & Severity Label"),
        ("03", "Train / Test Strategy"),
        ("04", "Methods — Baselines to XGBoost"),
        ("05", "Results — Stratified Models"),
        ("06", "False Negative Analysis"),
        ("07", "Crash Scenario Clusters"),
        ("08", "System Recommendations"),
        ("09", "Limitations"),
    ]
    for i, (num, text) in enumerate(items):
        row = i % 5; col = i // 5
        lx = 0.55 + col * 6.5; ly = 1.55 + row * 1.05
        box = s.shapes.add_shape(1, Inches(lx), Inches(ly), Inches(5.9), Inches(0.82))
        box.fill.solid(); box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = LTBLUE; box.line.width = Pt(1)
        txb(s, num, lx + 0.12, ly - 0.02, 0.55, 0.82, size=26, bold=True, color=GOLD)
        txb(s, text, lx + 0.7, ly + 0.12, 5.1, 0.6, size=15, color=DARK)


def slide_problem(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "Problem Statement", "What are we predicting, and why does it matter?")
    # left column
    bullet_box(s, [
        ("•", "NHTSA requires AV operators to report every crash — but not all crashes are equal"),
        ("•", "Safety engineers must manually triage thousands of reports annually"),
        ("•", "Missed severe crashes delay investigation of potential AV system failures"),
        ("•", "Goal: automatically classify incoming reports as severe or non-severe"),
    ], l=0.55, t=1.55, w=6.2, h=3.5, size=16, gap=4)
    # right — stakeholder box
    box = s.shapes.add_shape(1, Inches(7.1), Inches(1.55), Inches(5.7), Inches(3.6))
    box.fill.solid(); box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = LTBLUE; box.line.width = Pt(1)
    txb(s, "Stakeholders", 7.25, 1.6, 5.4, 0.45, size=14, bold=True, color=NAVY)
    bullet_box(s, [
        ("–", "AV safety engineers — triage incoming SGO reports"),
        ("–", "NHTSA regulators — identify patterns and systemic risks"),
        ("–", "AV operators — reduce review burden on non-severe incidents"),
    ], l=7.25, t=2.1, w=5.3, h=2.5, size=14, gap=6)
    # bottom key metric
    kpi = s.shapes.add_shape(1, Inches(0.55), Inches(5.4), Inches(12.3), Inches(1.55))
    kpi.fill.solid(); kpi.fill.fore_color.rgb = NAVY
    kpi.line.fill.background()
    txb(s, "Key constraint:", 0.75, 5.48, 2.2, 0.5, size=14, bold=True, color=GOLD)
    txb(s,
        "False negatives (missed severe crashes) are the primary risk — a missed severe crash "
        "may go uninvestigated, masking systemic AV failures. Precision ≥ 0.65 floor, recall maximized.",
        2.8, 5.48, 9.8, 1.2, size=14, color=WHITE)


def slide_data(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "Data Overview", "NHTSA SGO incident reports — 4 files, 5,567 labeled incidents")
    table_slide(s,
        headers=["Stratum", "N", "Severe Rate", "Era"],
        rows=[
            ["ADS — archived", "1,519", "26.0%", "Training"],
            ["ADS — current",  "597",   "47.4%", "Test"],
            ["L2 — archived",  "2,663", "98.3%", "Training"],
            ["L2 — current",   "788",   "97.2%", "Test"],
        ],
        col_widths=[4.5, 2.2, 2.8, 2.8],
        t=1.55,
    )
    bullet_box(s, [
        ("•", "ADS (Level 4+): Waymo, Cruise, Zoox — system controls all driving tasks"),
        ("•", "L2 (driver-assist): Tesla Autopilot — human driver remains responsible"),
        ("•", "Temporal split: archived era → train  |  current era → test"),
    ], l=0.55, t=4.45, w=12.3, h=2.1, size=15, gap=5)
    footnote(s, "ADS severe rate increased +21 pp from training to test era — the core distribution shift problem.")


def slide_severity_label(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "Severity Label Definition", "Feedback #1 — OR rule combining three observable outcomes")

    # three boxes
    defs = [
        ("Injury",    "injury_flag",  "At least one person\nreported injured"),
        ("Airbag",    "airbag_flag",  "At least one airbag\ndeployed (Δv threshold)"),
        ("Tow",       "towed_flag",   "Vehicle towed from\nscene (non-driveable)"),
    ]
    for i, (label, flag, desc) in enumerate(defs):
        lx = 0.55 + i * 4.2
        box = s.shapes.add_shape(1, Inches(lx), Inches(1.55), Inches(3.9), Inches(2.3))
        box.fill.solid(); box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = BLUE; box.line.width = Pt(1.5)
        txb(s, label, lx + 0.15, 1.6, 3.6, 0.5, size=18, bold=True, color=NAVY)
        txb(s, flag,  lx + 0.15, 2.1, 3.6, 0.4, size=12, color=BLUE, italic=True)
        txb(s, desc,  lx + 0.15, 2.55, 3.6, 0.9, size=13, color=DARK)

    # OR formula
    txb(s, "severe  =  injury_flag  OR  airbag_flag  OR  towed_flag",
        0.55, 4.05, 12.3, 0.6, size=18, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    rule = s.shapes.add_shape(1, Inches(1.5), Inches(4.7), Inches(10.3), Pt(1.5))
    rule.fill.solid(); rule.fill.fore_color.rgb = LTBLUE; rule.line.fill.background()

    bullet_box(s, [
        ("•", "OR rule maximizes recall — any one severe indicator flags the crash for priority review"),
        ("•", "Asymmetric cost: missing a severe crash > unnecessary review of a borderline case"),
        ("•", "No outcome words used as features — those words ARE the label (data leakage prevention)"),
    ], l=0.55, t=4.8, w=12.3, h=2.0, size=14, gap=4)


def slide_split(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "Train / Test Split Strategy", "Feedback #2 — Temporal split simulates real deployment")

    # timeline visual
    tl_y = 2.6
    # archived bar
    bar_a = s.shapes.add_shape(1, Inches(0.55), Inches(tl_y), Inches(7.5), Inches(0.7))
    bar_a.fill.solid(); bar_a.fill.fore_color.rgb = BLUE; bar_a.line.fill.background()
    txb(s, "ARCHIVED ERA  →  Train", 0.65, tl_y + 0.05, 7.2, 0.6,
        size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    # current bar
    bar_c = s.shapes.add_shape(1, Inches(8.2), Inches(tl_y), Inches(4.65), Inches(0.7))
    bar_c.fill.solid(); bar_c.fill.fore_color.rgb = GOLD; bar_c.line.fill.background()
    txb(s, "CURRENT ERA  →  Test", 8.3, tl_y + 0.05, 4.4, 0.6,
        size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    # divider arrow
    txb(s, "▶", 7.85, tl_y + 0.08, 0.5, 0.55, size=22, bold=True, color=NAVY,
        align=PP_ALIGN.CENTER)

    # validation sub-split note
    txb(s, "Within training: 75 / 25 stratified validation split for threshold tuning",
        0.55, 3.5, 12.3, 0.45, size=13, color=MGRAY, italic=True, align=PP_ALIGN.CENTER)

    bullet_box(s, [
        ("•", "Random 80/20 split would allow future patterns to leak into training — artificial AUC inflation"),
        ("•", "Threshold tuned on validation split; applied to test set exactly once"),
        ("•", "ADS severe rate: 26% train → 47% test  (+21 pp shift) — core modeling challenge"),
        ("•", "L2 severe rate: 98% train → 97% test  (stable — temporal shift is an ADS-specific problem)"),
    ], l=0.55, t=4.1, w=12.3, h=2.9, size=15, gap=5)


def slide_methods(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "Methods", "Three algorithms × two automation levels = 6 models")

    # top row: why stratify
    kpi = s.shapes.add_shape(1, Inches(0.55), Inches(1.55), Inches(12.3), Inches(0.8))
    kpi.fill.solid(); kpi.fill.fore_color.rgb = NAVY; kpi.line.fill.background()
    txb(s, "Pooled LR (single model on ADS + L2):  0 severe ADS crashes predicted on test set — 100% false-negative rate",
        0.65, 1.6, 12.0, 0.6, size=14, bold=False, color=WHITE)

    # three algorithm cards
    cards = [
        ("Logistic Regression", "Linear boundary\nC = 0.1, class_weight = balanced\n\nWorks for L2 (near-linearly separable)\nFails for ADS (non-linear interactions)"),
        ("Random Forest",       "Ensemble of decision trees\n300 trees, sqrt features\nclass_weight = balanced\n\nCaptures non-linear flag interactions"),
        ("XGBoost",             "Gradient-boosted trees\n200 trees, lr = 0.05, depth = 5\nscale_pos_weight for imbalance\n\nBest ADS model — AUC 0.921"),
    ]
    for i, (title, body) in enumerate(cards):
        lx = 0.55 + i * 4.27
        box = s.shapes.add_shape(1, Inches(lx), Inches(2.55), Inches(4.0), Inches(3.3))
        box.fill.solid(); box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = LTBLUE; box.line.width = Pt(1.5)
        # header band
        hdr = s.shapes.add_shape(1, Inches(lx), Inches(2.55), Inches(4.0), Inches(0.48))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = BLUE if i < 2 else GOLD
        hdr.line.fill.background()
        txb(s, title, lx + 0.12, 2.57, 3.75, 0.44, size=14, bold=True, color=WHITE)
        txb(s, body, lx + 0.12, 3.1, 3.75, 2.55, size=12, color=DARK)

    footnote(s, "Feature sets: ADS uses tabular + 21 narrative flags. L2 uses tabular only. Threshold selected on validation split (precision floor ≥ 0.65).")


def slide_narrative(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "Narrative Feature Extraction", "21 binary scenario flags from free-text incident descriptions")

    bullet_box(s, [
        ("•", "Tabular features give AUC ≈ 0.50 for ADS — essentially random discrimination"),
        ("•", "Narrative captures crash dynamics: was the AV moving? Did another vehicle approach from behind?"),
        ("•", "No outcome words used (injured, towed, airbag) — those ARE the label"),
    ], l=0.55, t=1.55, w=12.3, h=1.4, size=15, gap=4)

    # ablation table
    table_slide(s,
        headers=["Feature Set", "AUC (LR on ADS)"],
        rows=[
            ["Tabular only",           "0.500"],
            ["Narrative only",         "0.608"],
            ["Tabular + Narrative",    "0.531"],
        ],
        col_widths=[7.5, 4.8],
        t=3.1,
    )

    # example flags
    txb(s, "Example flags:", 0.55, 4.95, 3.0, 0.4, size=13, bold=True, color=NAVY)
    bullet_box(s, [
        ("–", "nav_av_stopped  ·  nav_rear_approach  ·  nav_at_intersection"),
        ("–", "nav_vulnerable_user  ·  nav_av_disengaged  ·  nav_minor_damage_lang"),
    ], l=0.55, t=5.35, w=12.3, h=1.5, size=13, gap=3)
    footnote(s, "Regex patterns applied after stripping Waymo SGO boilerplate headers. 84% of ADS incidents have usable narrative coverage.")


def slide_results(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "Stratified Model Results", "XGBoost is the best ADS model — LR sufficient for L2")

    table_slide(s,
        headers=["Model", "Precision", "Recall", "F1", "AUC", "FN", "FN-rate"],
        rows=[
            ["LR — ADS",    "0.499", "0.845", "0.627", "0.531", "44", "15.5%"],
            ["RF — ADS",    "0.804", "0.883", "0.842", "0.893", "33", "11.7%"],
            ["XGB — ADS ★", "0.831", "0.922", "0.874", "0.921", "22",  "7.8%"],
            ["LR — L2  ★",  "0.985", "0.967", "0.976", "0.876", "25",  "3.3%"],
            ["RF — L2",     "0.972", "1.000", "0.986", "0.836",  "0",  "0.0%"],
            ["XGB — L2",    "0.972", "1.000", "0.986", "0.850",  "0",  "0.0%"],
        ],
        col_widths=[2.8, 1.7, 1.7, 1.6, 1.6, 1.5, 1.4],
        t=1.55,
    )

    bullet_box(s, [
        ("★", "XGB ADS: +0.39 AUC over LR (0.531 → 0.921) — non-linear narrative interactions are critical"),
        ("★", "LR L2: near-perfect due to 97–98% severe rate — linear boundary is sufficient"),
        ("–", "Pooled LR (baseline): 283 missed severe ADS crashes → 0% recall on ADS"),
    ], l=0.55, t=5.5, w=12.3, h=1.7, size=14, gap=4)


def slide_fn(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "False Negative Analysis", "Feedback #5 — Who did the model miss, and why?")

    # two panels
    for col, (title, color, rows) in enumerate([
        ("XGB — ADS (19 missed, 6.7% FN-rate)", BLUE,
         [("Roadway Type", "Street (84%)"),
          ("Crash With",   "SUV (32%)"),
          ("SV Movement",  "Stopped (63%)"),
          ("CP Movement",  "Straight / Backing")]),
        ("LR — L2 (19 missed, 2.5% FN-rate)", NAVY,
         [("Roadway Type", "Highway (74%)"),
          ("Crash With",   "Other / narrative (37%)"),
          ("SV Movement",  "Proceeding straight (42%)"),
          ("CP Movement",  "Proceeding straight (79%)")]),
    ]):
        lx = 0.55 + col * 6.5
        hdr = s.shapes.add_shape(1, Inches(lx), Inches(1.55), Inches(6.1), Inches(0.5))
        hdr.fill.solid(); hdr.fill.fore_color.rgb = color; hdr.line.fill.background()
        txb(s, title, lx + 0.1, 1.57, 5.9, 0.46, size=13, bold=True, color=WHITE)
        for ri, (dim, val) in enumerate(rows):
            ry = 2.2 + ri * 0.75
            shade = s.shapes.add_shape(1, Inches(lx), Inches(ry), Inches(6.1), Inches(0.65))
            shade.fill.solid()
            shade.fill.fore_color.rgb = LTBLUE if ri % 2 == 0 else WHITE
            shade.line.fill.background()
            txb(s, dim, lx + 0.12, ry + 0.08, 2.8, 0.5, size=13, bold=True, color=DARK)
            txb(s, val, lx + 3.0,  ry + 0.08, 3.0, 0.5, size=13, color=DARK)

    bullet_box(s, [
        ("•", "ADS: stopped AV rear-ended by SUV — no strong narrative severity signal → missed"),
        ("•", "L2: high-speed highway, both straight — low-severity training distribution → missed"),
        ("•", "Mitigation: hard rule — any towed_flag = 1 report enters Priority Queue regardless of model score"),
        ("•", "Stratified models reduced total FN by 94%  (304 pooled → 38 stratified)"),
    ], l=0.55, t=5.4, w=12.3, h=1.85, size=13, gap=3)


def slide_clusters(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "ADS Crash Scenario Clusters", "Feedback #6 — K-means (k=7) on narrative + tabular features")

    table_slide(s,
        headers=["Cluster", "% of ADS", "Severe Rate", "Scenario"],
        rows=[
            ["C5  ⬆ priority", "21.1%", "47.8%", "AV moving at impact — struck by other vehicle"],
            ["C1",             " 3.9%", "41.5%", "AV hit fixed object (parking lot / disengagement)"],
            ["C0",             "14.3%", "32.7%", "Other vehicle struck stationary AV"],
            ["C3",             " 3.6%", "31.2%", "AV struck other party (AV-at-fault)"],
            ["C2  baseline",   "56.0%", "26.0%", "Typical street crash (modal scenario)"],
            ["C4",             " 1.0%",  "0.0%", "Animal strike — no injuries"],
        ],
        col_widths=[2.5, 2.0, 2.2, 5.6],
        t=1.55,
    )

    bullet_box(s, [
        ("•", "k selected by silhouette score (k=7 = 0.098) — scanned k ∈ {3…8}"),
        ("•", "Labels derived from narrative flag lift above global mean (not just highest absolute rate)"),
        ("•", "C5 (21% of ADS, 48% severe): AV was moving — highest kinetic energy, highest risk"),
        ("•", "C4 animal strikes (1%): 0% severe — can be deprioritized / routed to perception team"),
    ], l=0.55, t=5.5, w=12.3, h=1.75, size=14, gap=4)


def slide_recommendations(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "System Recommendations", "Feedback #7 — Translating model findings into concrete safety actions")

    recs = [
        ("01", "Triage architecture",
         "Route incoming reports through automation-level detector first, then the appropriate model (XGB for ADS, LR/RF for L2). Hard rule: towed_flag = 1 → Priority Queue regardless of score."),
        ("02", "Prioritize C5 & C1 clusters",
         "AV-moving-at-impact (C5, 48% severe) and parking-lot/fixed-object (C1, 42% severe) get 48-hour senior review. All other clusters enter standard 30-day queue."),
        ("03", "Audit C3 (AV-at-fault) separately",
         "ADS vehicle struck another party in 77 incidents — cross-reference with disengagement logs and sensor failure reports for potential control-system failures."),
        ("04", "Zero-FN L2 policy",
         "Switch L2 model to RF or XGB: 0 false negatives at cost of 22 extra alarms (2.8% of test set). Highway L2 severe crashes are too high-risk to miss."),
        ("05", "Monitor ADS distribution shift",
         "ADS severe rate rose +21 pp in one era. Set retraining trigger at rolling-90-day severe rate > 55%."),
        ("06", "Cluster tags on every report",
         "Attach scenario cluster label (C0–C6) to each ADS report as metadata — gives reviewers instant context without reading the full narrative."),
    ]
    for i, (num, title, body) in enumerate(recs):
        row = i % 3; col = i // 3
        lx = 0.55 + col * 6.5; ly = 1.55 + row * 1.85
        box = s.shapes.add_shape(1, Inches(lx), Inches(ly), Inches(6.1), Inches(1.7))
        box.fill.solid(); box.fill.fore_color.rgb = WHITE
        box.line.color.rgb = LTBLUE; box.line.width = Pt(1)
        txb(s, num, lx + 0.12, ly + 0.05, 0.55, 0.55, size=20, bold=True, color=GOLD)
        txb(s, title, lx + 0.65, ly + 0.05, 5.3, 0.45, size=13, bold=True, color=NAVY)
        txb(s, body, lx + 0.12, ly + 0.55, 5.85, 1.05, size=11, color=DARK)


def slide_limitations(prs):
    s = blank_slide(prs); bg(s); accent_bar(s)
    heading(s, "Limitations", "Feedback #3 — Reporting bias, data scarcity, and generalization")

    table_slide(s,
        headers=["Limitation", "Impact", "Mitigation"],
        rows=[
            ["Reporting bias (ADS vs L2 thresholds differ)",
             "L2 non-severe likely undercounted; 98% severe rate is a regulatory artifact",
             "Report L2 metrics alongside distribution context"],
            ["ADS distribution shift (+21 pp)",
             "May be a data artifact, not a genuine safety trend",
             "Monitor rolling severe rate; retrain at > 55%"],
            ["ADS data scarcity (395 training positives)",
             "Limits fine-grained pattern learning",
             "More operators / reporting periods"],
            ["Narrative coverage (84%)",
             "CBI-redacted rows → zero nav_ features",
             "nav_has_narrative flag in model"],
            ["L2 near-degenerate (2.8% non-severe)",
             "Precision/F1 sensitive to single-digit count changes",
             "Report AUC as primary L2 metric"],
        ],
        col_widths=[3.5, 4.5, 4.3],
        t=1.55,
    )
    footnote(s, "Models are operational triage tools for reported SGO incidents — not general-purpose AV safety predictors.")


def slide_conclusion(prs):
    s = blank_slide(prs)
    fill = s.background.fill; fill.solid(); fill.fore_color.rgb = NAVY
    strip = s.shapes.add_shape(1, 0, 0, W, Inches(0.12))
    strip.fill.solid(); strip.fill.fore_color.rgb = GOLD; strip.line.fill.background()
    strip2 = s.shapes.add_shape(1, 0, Inches(7.38), W, Inches(0.12))
    strip2.fill.solid(); strip2.fill.fore_color.rgb = GOLD; strip2.line.fill.background()

    txb(s, "Key Takeaways", 1.0, 0.7, 11.3, 0.7, size=30, bold=True,
        color=WHITE, align=PP_ALIGN.CENTER)

    takeaways = [
        ("Pooled model fails on ADS",
         "100% false-negative rate — learned a shortcut that breaks on distribution shift"),
        ("Stratified XGBoost (ADS)",
         "AUC 0.921 | Precision 0.831 | Recall 0.922 | FN-rate 6.7% — non-linear narrative interactions are essential"),
        ("LR / RF sufficient for L2",
         "Near-perfect precision and recall on a 97%-severe distribution"),
        ("Clusters → actionable scenarios",
         "C5 (AV moving, 48% severe) is highest-risk; C4 (animal strikes) can be deprioritized"),
        ("Total FN reduction: 304 → 38",
         "Stratified pipeline recovers all 283 previously missed severe ADS crashes"),
    ]
    for i, (title, body) in enumerate(takeaways):
        ly = 1.55 + i * 1.06
        dot = s.shapes.add_shape(1, Inches(0.9), Inches(ly + 0.12), Inches(0.18), Inches(0.18))
        dot.fill.solid(); dot.fill.fore_color.rgb = GOLD; dot.line.fill.background()
        txb(s, title, 1.2, ly, 4.5, 0.48, size=14, bold=True, color=WHITE)
        txb(s, body,  5.6, ly, 7.3, 0.48, size=14, color=LTBLUE)

    txb(s, "Thank you  ·  Questions welcome",
        1.0, 7.0, 11.3, 0.38, size=13, color=MGRAY, italic=True, align=PP_ALIGN.CENTER)


# ─── build ────────────────────────────────────────────────────────────────────
def main():
    prs = new_prs()
    slide_title(prs)
    slide_agenda(prs)
    slide_problem(prs)
    slide_data(prs)
    slide_severity_label(prs)
    slide_split(prs)
    slide_methods(prs)
    slide_narrative(prs)
    slide_results(prs)
    slide_fn(prs)
    slide_clusters(prs)
    slide_recommendations(prs)
    slide_limitations(prs)
    slide_conclusion(prs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved: {OUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
