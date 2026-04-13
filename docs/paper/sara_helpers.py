"""
Knowledge Distillation – Harvard Crimson Edition v3
Author: Ashutosh Sinha  |  ajsinha@gmail.com
Fixes: cover text visibility, adds diagrams, authorship, full references, prompt-tuning KD section
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, NextPageTemplate,
    Paragraph, Spacer, Table, TableStyle, PageBreak,
    HRFlowable, KeepTogether, Flowable
)
from reportlab.platypus.flowables import BalancedColumns
from reportlab.graphics.shapes import (
    Drawing, Rect, String, Line, Polygon, Circle,
    Group, Path, PolyLine
)
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.colors import HexColor, white, black, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io, os

# ── Register Unicode fonts for Devanagari script ─────────────────────────────
# FreeSerif ships on Ubuntu/Pop!_OS and fully covers the Devanagari Unicode block.
_FREE_SERIF_PATH   = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
_FREE_SERIF_B_PATH = "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"
_FREE_SERIF_I_PATH = "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf"
_FREE_MONO_PATH    = "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
_FREE_MONO_B_PATH  = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"

for _name, _path in [
    ("FreeSerif",       _FREE_SERIF_PATH),
    ("FreeSerif-Bold",  _FREE_SERIF_B_PATH),
    ("FreeSerif-Italic",_FREE_SERIF_I_PATH),
    ("FreeMono",        _FREE_MONO_PATH),
    ("FreeMono-Bold",   _FREE_MONO_B_PATH),
]:
    if os.path.exists(_path):
        try:
            pdfmetrics.registerFont(TTFont(_name, _path))
        except Exception:
            pass  # already registered or unavailable

# Register font family mapping so <font name="FreeSerif"> works in Paragraphs
if os.path.exists(_FREE_SERIF_PATH):
    try:
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily(
            'FreeSerif',
            normal='FreeSerif',
            bold='FreeSerif-Bold' if os.path.exists(_FREE_SERIF_B_PATH) else 'FreeSerif',
            italic='FreeSerif-Italic' if os.path.exists(_FREE_SERIF_I_PATH) else 'FreeSerif',
            boldItalic='FreeSerif-Bold' if os.path.exists(_FREE_SERIF_B_PATH) else 'FreeSerif',
        )
    except Exception:
        pass

# ── Resolved font names (after registration) ─────────────────────────────────
_DEVA_FONT      = "FreeSerif"       if os.path.exists(_FREE_SERIF_PATH)   else "Times-Bold"
_BODY_FONT      = "FreeSerif"       if os.path.exists(_FREE_SERIF_PATH)   else "Times-Roman"
_BODY_FONT_BOLD = "FreeSerif-Bold"  if os.path.exists(_FREE_SERIF_B_PATH) else "Times-Bold"
_BODY_FONT_ITAL = "FreeSerif-Italic" if os.path.exists(_FREE_SERIF_I_PATH) else "Times-Italic"
_MONO_FONT      = "FreeMono"        if os.path.exists(_FREE_MONO_PATH)    else "Courier"
_MONO_FONT_BOLD = "FreeMono-Bold"   if os.path.exists(_FREE_MONO_B_PATH)  else "Courier-Bold"

# Resolve best available monospace Unicode font for pseudocode

# ── Palette ─────────────────────────────────────────────────────────────────
CRIMSON      = HexColor("#A51C30")
CRIMSON_DARK = HexColor("#6B0F1F")
CRIMSON_MID  = HexColor("#C23050")
CRIMSON_SOFT = HexColor("#F8EAED")
GOLD         = HexColor("#C4A035")
GOLD_LIGHT   = HexColor("#FBF5E6")
CHARCOAL     = HexColor("#1C1C1C")
GRAY_DARK    = HexColor("#444444")
GRAY_MED     = HexColor("#777777")
GRAY_LIGHT   = HexColor("#DDDDDD")
CODE_BG      = HexColor("#1A1A2E")
CODE_FG      = HexColor("#E0E0E0")
TEAL         = HexColor("#0D7377")
STEEL        = HexColor("#2E4057")
W, H = letter   # 612 × 792 pt

# ── Page chrome callbacks  (fire BEFORE flowable content) ────────────────────
def draw_cover(canvas, doc):
    canvas.saveState()
    # Full crimson fill
    canvas.setFillColor(CRIMSON)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Gold stripes
    canvas.setFillColor(GOLD)
    canvas.rect(0, H - 1.8*inch, W, 5, fill=1, stroke=0)
    canvas.rect(0, 1.05*inch, W, 5, fill=1, stroke=0)
    # White hairlines
    canvas.setFillColor(white)
    canvas.rect(0, H - 1.82*inch, W, 2, fill=1, stroke=0)
    canvas.rect(0, 1.08*inch, W, 2, fill=1, stroke=0)
    # Decorative side bar
    canvas.setFillColor(CRIMSON_DARK)
    canvas.rect(0, 0, 6, H, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(6, 0, 3, H, fill=1, stroke=0)
    canvas.restoreState()

def draw_body(canvas, doc):
    canvas.saveState()
    pn = doc.page
    # Top rule
    canvas.setStrokeColor(CRIMSON)
    canvas.setLineWidth(1.2)
    canvas.line(0.7*inch, H - 0.52*inch, W - 0.7*inch, H - 0.52*inch)
    # Header left: title — split so Devanagari uses FreeSerif
    canvas.setFillColor(GRAY_MED)
    _hx = 0.7*inch
    _hy = H - 0.42*inch
    _hsize = 7.5
    for _hfont, _htext in [
        ("Helvetica-Oblique", "Sara ("),
        (_DEVA_FONT,          "\u0938\u093e\u0930"),  # सार in Unicode escapes
        ("Helvetica-Oblique", ")  —  Knowledge Distillation Reference"),
    ]:
        canvas.setFont(_hfont, _hsize)
        canvas.drawString(_hx, _hy, _htext)
        _hx += canvas.stringWidth(_htext, _hfont, _hsize)
    # Header right: author
    canvas.drawRightString(W - 0.7*inch, H - 0.42*inch,
                           "Ashutosh Sinha · 2025")
    # Bottom rule
    canvas.setStrokeColor(CRIMSON)
    canvas.line(0.7*inch, 0.58*inch, W - 0.7*inch, 0.58*inch)
    # Footer
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GRAY_MED)
    canvas.drawString(0.7*inch, 0.40*inch, "© 2025 Ashutosh Sinha. All rights reserved.")
    canvas.drawRightString(W - 0.7*inch, 0.40*inch, f"Confidential")
    # Crimson left bar
    canvas.setFillColor(CRIMSON)
    canvas.rect(0, 0, 6, H, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(6, 0, 3, H, fill=1, stroke=0)
    canvas.restoreState()

def draw_body_end(canvas, doc):
    """Page numbers injected after content — requires second-pass total."""
    pass   # handled in NumberedCanvas below

# ── Two-pass canvas for page-n-of-N ─────────────────────────────────────────
class NumberedCanvas(pdfcanvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for i, state in enumerate(self._saved):
            self.__dict__.update(state)
            if i > 0:   # skip cover
                self.setFont("Helvetica", 7.5)
                self.setFillColor(GRAY_MED)
                self.drawCentredString(W/2, 0.40*inch,
                                       f"—  {i+1}  /  {total}  —")
            super().showPage()
        super().save()

# ── Doc template ─────────────────────────────────────────────────────────────
def build_doc(story, out_path):
    cover_frame = Frame(0.75*inch, 0.9*inch, W-1.5*inch, H-0.9*inch-1.5*inch,
                        id='cover', leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0)
    body_frame  = Frame(0.80*inch, 0.80*inch, W-1.6*inch, H-0.80*inch-0.90*inch,
                        id='body',  leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0)

    cover_tpl = PageTemplate(id='Cover', frames=[cover_frame], onPage=draw_cover)
    body_tpl  = PageTemplate(id='Body',  frames=[body_frame],  onPage=draw_body)

    doc = BaseDocTemplate(
        out_path,
        pagesize=letter,
        pageTemplates=[cover_tpl, body_tpl],
        title="Sara (सार) — Knowledge Distillation",
        author="Ashutosh Sinha",
        subject="Machine Learning · Model Compression · KD-SPAR",
        creator="ReportLab / Sara Edition",
        leftMargin=0.80*inch, rightMargin=0.80*inch,
        topMargin=0.80*inch, bottomMargin=0.80*inch,
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF written → {out_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STYLES
# ═══════════════════════════════════════════════════════════════════════════
def S(name, **kw):
    return ParagraphStyle(name, **kw)


# Cover (white text on crimson)
# Scov_title uses FreeSerif so the Devanagari सार glyph renders correctly.
# Inline <font> tags inside the Paragraph switch to FreeSerif for the Sanskrit.

Scov_title  = S("cov_title",  fontName=_DEVA_FONT,    fontSize=38, leading=46,
                 textColor=white, alignment=TA_CENTER, spaceAfter=8)
Scov_sub    = S("cov_sub",    fontName=_BODY_FONT_ITAL, fontSize=15, leading=21,
                 textColor=HexColor("#FAE0E4"), alignment=TA_CENTER, spaceAfter=6)
Scov_meta   = S("cov_meta",   fontName=_BODY_FONT,      fontSize=10, leading=15,
                 textColor=HexColor("#FADADD"), alignment=TA_CENTER)
Scov_lbl    = S("cov_lbl",    fontName="Helvetica-Bold", fontSize=9,
                 textColor=white)
Scov_val    = S("cov_val",    fontName="Helvetica",      fontSize=9,
                 textColor=HexColor("#FADADD"))
Scov_copy   = S("cov_copy",   fontName=_BODY_FONT_ITAL,      fontSize=8,
                 textColor=HexColor("#E8C4C9"), alignment=TA_CENTER)

# Body — FreeSerif is visually identical to Times-Roman but covers all
# Unicode: Greek (α δ ε κ τ), math (← → ∈ ≤ ≥ ✓ ₀), arrows, etc.
Sbody  = S("body",  fontName=_BODY_FONT,       fontSize=10.5, leading=16,
            textColor=CHARCOAL, alignment=TA_JUSTIFY, spaceAfter=8, spaceBefore=2)
Sh1    = S("sh1",   fontName=_BODY_FONT_BOLD,  fontSize=17, leading=22,
            textColor=CRIMSON, spaceBefore=26, spaceAfter=4)
Sh2    = S("sh2",   fontName=_BODY_FONT_BOLD,  fontSize=13, leading=18,
            textColor=CRIMSON_DARK, spaceBefore=16, spaceAfter=4)
Sh3    = S("sh3",   fontName="FreeSerif-Italic" if os.path.exists(_FREE_SERIF_I_PATH) else "Times-BoldItalic",
            fontSize=11, leading=16,
            textColor=CHARCOAL, spaceBefore=10, spaceAfter=3)
Scode  = S("scode", fontName=_MONO_FONT,        fontSize=7.8, leading=11.5,
            textColor=CODE_FG,  backColor=CODE_BG, spaceAfter=0, spaceBefore=0)
Sbul   = S("sbul",  fontName=_BODY_FONT,        fontSize=10.5, leading=15,
            textColor=CHARCOAL, bulletIndent=10, leftIndent=22, spaceAfter=4)
Scap   = S("scap",  fontName=_BODY_FONT_ITAL,   fontSize=8.5, leading=12,
            textColor=GRAY_MED, alignment=TA_CENTER, spaceAfter=10, spaceBefore=2)
Sth    = S("sth",   fontName=_BODY_FONT_BOLD,    fontSize=8.5, leading=12,
            textColor=white, alignment=TA_CENTER)
Std    = S("std",   fontName=_BODY_FONT,         fontSize=8.5, leading=12,
            textColor=CHARCOAL)
Stdc   = S("stdc",  fontName=_BODY_FONT,         fontSize=8.5, leading=12,
            textColor=CHARCOAL, alignment=TA_CENTER)
Sref   = S("sref",  fontName=_BODY_FONT,         fontSize=9.5, leading=14,
            textColor=CHARCOAL, leftIndent=22, firstLineIndent=-22, spaceAfter=5)
# Inline Devanagari — used with <font name="FreeSerif">सार</font> in Paragraphs
Sdeva  = S("sdeva", fontName=_DEVA_FONT,          fontSize=10.5, leading=16,
            textColor=CHARCOAL)
Scall_t= S("scall_t",fontName=_BODY_FONT_BOLD,    fontSize=9.5, leading=13,
            textColor=CRIMSON_DARK, spaceAfter=3)
Scall_b= S("scall_b",fontName=_BODY_FONT,         fontSize=9.5, leading=14,
            textColor=CHARCOAL, spaceAfter=0)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FLOWABLE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════
CW = W - 1.6*inch   # content width  ≈ 7.0 in

def h1(txt):
    return [Paragraph(txt, Sh1),
            HRFlowable(width="100%", thickness=1.5, color=CRIMSON,
                       spaceAfter=8, spaceBefore=0)]
def h2(txt): return [Paragraph(txt, Sh2)]
def h3(txt): return [Paragraph(txt, Sh3)]
def bp(txt):
    return Paragraph(f"<bullet>\u2022</bullet> {txt}", Sbul)
def body(txt):  return [Paragraph(txt, Sbody)]
def sp(n=8):    return [Spacer(1, n)]
def blist(items): return [bp(i) for i in items]
def pgbrk():    return [PageBreak()]
def divider():
    return [HRFlowable(width="100%", thickness=0.5, color=GRAY_LIGHT,
                       spaceAfter=10, spaceBefore=10)]
def caption(txt): return [Paragraph(txt, Scap)]

def callout(title, txt, bg=CRIMSON_SOFT, bdr=CRIMSON):
    t = Table([[Paragraph(title, Scall_t)],[Paragraph(txt, Scall_b)]],
              colWidths=[CW])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), bg),
        ("LINEBEFORE",  (0,0),(0,-1), 3, bdr),
        ("LINEAFTER",   (0,0),(0,-1), 0.3, bdr),
        ("LINEABOVE",   (0,0),(-1,0), 0.5, bdr),
        ("LINEBELOW",   (0,-1),(-1,-1),0.5, bdr),
        ("TOPPADDING",  (0,0),(-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING", (0,0),(-1,-1), 14),
        ("RIGHTPADDING",(0,0),(-1,-1), 10),
    ]))
    return [t, Spacer(1,10)]

def gold_callout(title, txt):
    return callout(title, txt, bg=GOLD_LIGHT, bdr=GOLD)

def code_block(lines):
    rows = [[Paragraph(ln.replace(" ","&nbsp;").replace("<","&lt;")
                         .replace(">","&gt;"), Scode)]
            for ln in lines]
    t = Table(rows, colWidths=[CW])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), CODE_BG),
        ("LINEBEFORE",   (0,0),(0,-1),  3, CRIMSON),
        ("LINEABOVE",    (0,0),(-1,0),  0.5, GRAY_DARK),
        ("LINEBELOW",    (0,-1),(-1,-1),0.5, GRAY_DARK),
        ("TOPPADDING",   (0,0),(-1,-1), 1),
        ("BOTTOMPADDING",(0,0),(-1,-1), 1),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
    ]))
    return [t, Spacer(1,10)]

def dtable(headers, rows, col_widths=None):
    if col_widths is None:
        cw = CW / len(headers)
        col_widths = [cw]*len(headers)
    data = [[Paragraph(h, Sth) for h in headers]]
    for i, row in enumerate(rows):
        data.append([Paragraph(str(c), Stdc if j>0 else Std)
                     for j,c in enumerate(row)])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    st = [
        ("BACKGROUND",   (0,0),(-1,0), CRIMSON),
        ("LINEBELOW",    (0,0),(-1,0), 1.5, CRIMSON_DARK),
        ("GRID",         (0,0),(-1,-1),0.4, GRAY_LIGHT),
        ("VALIGN",       (0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
    ]
    for i in range(1, len(rows)+1):
        st.append(("BACKGROUND",(0,i),(-1,i),
                   CRIMSON_SOFT if i%2==0 else white))
    t.setStyle(TableStyle(st))
    return [t, Spacer(1,12)]

# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM BUILDERS  — all elements stay within [0, DW] × [20, DH]
# DW=488 is safely within CW≈497; bottom padding of 20pt prevents caption overlap
# ═══════════════════════════════════════════════════════════════════════════
DW = 488   # safe drawing width — do NOT exceed this

def _box(d, x, y, w, h, fill, text, text_color=white, font="Helvetica-Bold",
         fsize=8.5, radius=4):
    d.add(Rect(x, y, w, h, rx=radius, ry=radius,
               fillColor=fill, strokeColor=GRAY_LIGHT, strokeWidth=0.5))
    cx, cy = x + w/2, y + h/2
    d.add(String(cx, cy - fsize/2 + 1, text,
                 fontName=font, fontSize=fsize,
                 fillColor=text_color, textAnchor="middle"))

def _arrow(d, x1, y1, x2, y2, color=GRAY_MED, label="", fsize=7):
    import math
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.2))
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 6
    p1x = x2 - size*math.cos(angle - 0.4)
    p1y = y2 - size*math.sin(angle - 0.4)
    p2x = x2 - size*math.cos(angle + 0.4)
    p2y = y2 - size*math.sin(angle + 0.4)
    d.add(Polygon([x2, y2, p1x, p1y, p2x, p2y],
                  fillColor=color, strokeColor=color, strokeWidth=0))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        d.add(String(mx, my + 3, label, fontName="FreeSerif-Italic",
                     fontSize=fsize, fillColor=GRAY_MED, textAnchor="middle"))

def _lbl(d, x, y, txt, color=CHARCOAL, fsize=7.5, bold=False):
    d.add(String(x, y, txt,
                 fontName="FreeSerif-Bold" if bold else "Helvetica",
                 fontSize=fsize, fillColor=color, textAnchor="middle"))

# ── Fig 1.1: Teacher → Student flow ─────────────────────────────────────────
def diag_teacher_student():
    dw, dh = DW, 155
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white,
               strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    # Four main-flow boxes (y=44..78, centre at 61)
    # box_w=90, gap≈37, x = 8, 135, 262, 389  → rightmost = 479 < 488 ✓
    BOX, GAP, MID = 90, 37, 61
    xs = [8, 8+BOX+GAP, 8+2*(BOX+GAP), 8+3*(BOX+GAP)]   # 8,135,262,389
    boxes = [
        (CRIMSON,     "TEACHER MODEL",  "(Large / Accurate)"),
        (STEEL,       "SOFT TARGETS",   "Temperature T > 1"),
        (TEAL,        "STUDENT MODEL",  "(Small / Fast)"),
        (CRIMSON_DARK,"DEPLOYED MODEL", "(Efficient / Aligned)"),
    ]
    for i, (x, (col, lbl, sub)) in enumerate(zip(xs, boxes)):
        _box(d, x, MID-17, BOX, 34, col, lbl, fsize=7.8)
        _lbl(d, x+BOX/2, MID-17-11, sub, color=GRAY_MED, fsize=6.8)

    # Arrows between boxes
    _arrow(d, xs[0]+BOX, MID, xs[1],     MID, CRIMSON, "logits")
    _arrow(d, xs[1]+BOX, MID, xs[2],     MID, GRAY_MED)
    _arrow(d, xs[2]+BOX, MID, xs[3],     MID, TEAL, "deploy")

    # Upper lane: TRAINING DATA + HARD LABELS (y=107..133)
    _box(d, xs[0], 107, BOX, 26, GOLD, "TRAINING DATA",
         text_color=CHARCOAL, fsize=7.5, radius=3)
    _box(d, xs[1], 107, BOX, 26, GRAY_MED, "HARD LABELS", fsize=7.5, radius=3)
    _arrow(d, xs[0]+BOX/2, 107, xs[0]+BOX/2, 78, GOLD)        # data → teacher
    _arrow(d, xs[1]+BOX/2, 107, xs[1]+BOX/2, 78, GRAY_MED)    # labels → soft targets
    _arrow(d, xs[0]+BOX, 120, xs[1],          120, GOLD)       # data → hard labels

    # Loss formula at top
    _lbl(d, dw/2, 146,
         "Loss = α · KL(student ‖ teacher_soft) + (1−α) · CE(student, hard_labels)",
         color=CRIMSON, fsize=7.2, bold=True)
    return d

# ── Fig 2.1: Temperature effect bar chart ────────────────────────────────────
def diag_temperature():
    dw, dh = DW, 140
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white,
               strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    classes   = ["Cat", "Dog", "Car", "Bird", "Fish"]
    probs_t1  = [0.87, 0.10, 0.01, 0.01, 0.01]
    probs_t4  = [0.53, 0.28, 0.09, 0.06, 0.04]
    bar_w, gap = 24, 7
    base_y, max_h = 30, 72    # bars start at y=30, max height 72 → top=102 < 140 ✓

    # Group 1: T=1  starts at x=18
    gx1 = 18
    _lbl(d, gx1 + 2.5*(bar_w+gap), base_y+max_h+14,
         "T = 1   (Sharp — poor dark-knowledge transfer)",
         color=CRIMSON, fsize=7.5, bold=True)
    for i, (cls, p) in enumerate(zip(classes, probs_t1)):
        x  = gx1 + i*(bar_w+gap)
        bh = max(3, int(p*max_h))
        d.add(Rect(x, base_y, bar_w, bh,
                   fillColor=CRIMSON, strokeColor=white, strokeWidth=0.5))
        d.add(String(x+bar_w/2, base_y+bh+4, f"{p:.2f}",
                     fontName="FreeSerif", fontSize=6.5,
                     fillColor=CHARCOAL, textAnchor="middle"))
        d.add(String(x+bar_w/2, base_y-10, cls,
                     fontName="FreeSerif", fontSize=7,
                     fillColor=GRAY_DARK, textAnchor="middle"))

    # Separator + arrow in the middle
    sep_x = gx1 + 5*(bar_w+gap) + 10          # ≈ 173
    _arrow(d, sep_x, 66, sep_x+38, 66, GOLD, "Higher T")

    # Group 2: T=4  starts after arrow  ≈ sep_x+48
    gx2 = sep_x + 48                           # ≈ 221
    _lbl(d, gx2 + 2.5*(bar_w+gap), base_y+max_h+14,
         "T = 4   (Soft — rich inter-class signal)",
         color=TEAL, fsize=7.5, bold=True)
    for i, (cls, p) in enumerate(zip(classes, probs_t4)):
        x  = gx2 + i*(bar_w+gap)
        bh = max(3, int(p*max_h))
        d.add(Rect(x, base_y, bar_w, bh,
                   fillColor=TEAL, strokeColor=white, strokeWidth=0.5))
        d.add(String(x+bar_w/2, base_y+bh+4, f"{p:.2f}",
                     fontName="FreeSerif", fontSize=6.5,
                     fillColor=CHARCOAL, textAnchor="middle"))
        d.add(String(x+bar_w/2, base_y-10, cls,
                     fontName="FreeSerif", fontSize=7,
                     fillColor=GRAY_DARK, textAnchor="middle"))
    # rightmost bar: gx2 + 4*(bar_w+gap) + bar_w = 221+4*31+24 = 221+148=369 < 488 ✓
    return d

# ── Fig 3.1: Taxonomy tree ────────────────────────────────────────────────────
def diag_taxonomy():
    dw, dh = DW, 185
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white,
               strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    # Root box centred
    root_cx = dw/2
    _box(d, root_cx-70, 152, 140, 24, CRIMSON,
         "KNOWLEDGE DISTILLATION", fsize=7.5)

    # 5 branch boxes — box_w=78, evenly spaced within DW
    # gap=(488-20-5*78)/4=(488-20-390)/4=19.5
    BW, BH, BY = 78, 26, 106
    BG = (dw - 20 - 5*BW) / 4    # ≈ 19.5
    bxs = [10 + i*(BW+BG) for i in range(5)]
    # rightmost: bxs[4]+BW = 10+4*97.5+78 = 10+390+78 = 478 ✓ (BW+BG≈97.5)

    branch_cfg = [
        (STEEL,        "Response\nBased"),
        (CRIMSON_DARK, "Feature\nBased"),
        (TEAL,         "Attention\nTransfer"),
        (GOLD,         "Relation\nBased"),
        (GRAY_DARK,    "Data-Free"),
    ]
    subs = [
        ["Soft logits", "Temp. scaling"],
        ["FitNets",     "Hidden layers"],
        ["Attn maps",   "Spatial align"],
        ["RKD",         "Graph dist."],
        ["GAN synth",   "No orig. data"],
    ]
    for i, (bx, (col, lbl)) in enumerate(zip(bxs, branch_cfg)):
        bcx = bx + BW/2
        # arrow from root to branch
        _arrow(d, root_cx, 152, bcx, BY+BH, GRAY_LIGHT)
        _box(d, bx, BY, BW, BH, col, lbl.replace("\n", " "), fsize=7)
        # two sub-chips below
        for si, sub in enumerate(subs[i]):
            sy = BY - 22 - si*22    # si=0: 84; si=1: 62  → min=62 > 20 ✓
            d.add(Rect(bx+4, sy, BW-8, 18, rx=2, ry=2,
                       fillColor=CRIMSON_SOFT,
                       strokeColor=GRAY_LIGHT, strokeWidth=0.4))
            d.add(String(bcx, sy+5, sub, fontName="FreeSerif",
                         fontSize=6.5, fillColor=CHARCOAL,
                         textAnchor="middle"))
            if si == 0:
                _arrow(d, bcx, BY, bcx, sy+18, GRAY_LIGHT)
    return d

# ── Fig 4.1: Training pipeline ────────────────────────────────────────────────
def diag_training_pipeline():
    dw, dh = DW, 100
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white,
               strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    # 5 steps — box_w=78, gap≈19.5, x: 10,107,204,301,398 → rightmost=476 ✓
    BW, GAP = 78, 19
    xs = [10 + i*(BW+GAP) for i in range(5)]
    steps = [
        (CRIMSON,     "1. Pretrain\nTeacher"),
        (GRAY_DARK,   "2. Freeze\nTeacher"),
        (STEEL,       "3. Distil\nStudent"),
        (TEAL,        "4. Evaluate\nStudent"),
        (CRIMSON_DARK,"5. Deploy\nStudent"),
    ]
    MID = 57
    for x, (col, lbl) in zip(xs, steps):
        _box(d, x, MID-18, BW, 36, col, lbl.replace("\n"," "), fsize=8)

    for i in range(4):
        _arrow(d, xs[i]+BW, MID, xs[i+1], MID, GRAY_MED)

    _lbl(d, dw/2, 86,
         "Teacher weights are frozen throughout steps 2–5",
         color=GRAY_MED, fsize=7.5)
    return d

# ── Fig 13.1: RAG migration flow ──────────────────────────────────────────────
def diag_rag_migration():
    dw, dh = DW, 170
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white,
               strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    # 5 phase boxes — box_w=80, gap≈17, x: 10,107,204,301,398 → rightmost=478 ✓
    BW, GAP, BH = 80, 17, 54
    xs = [10 + i*(BW+GAP) for i in range(5)]
    phases = [
        (CRIMSON,     "LIVE TRACE\nHARVEST"),
        (STEEL,       "DATASET\nFORMAT"),
        (TEAL,        "SFT + KL\nDISTILLATION"),
        (GOLD,        "EQUIVALENCE\nTESTING"),
        (CRIMSON_DARK,"SHADOW →\nCANARY → 100%"),
    ]
    MID = 120
    for x, (col, lbl) in zip(xs, phases):
        _box(d, x, MID-BH//2, BW, BH, col, lbl.replace("\n"," "), fsize=7.5)

    for i in range(4):
        _arrow(d, xs[i]+BW, MID, xs[i+1], MID, GRAY_MED)

    # Teacher lane label (left half)
    d.add(Rect(10, 30, 220, 24, rx=3, ry=3,
               fillColor=CRIMSON_SOFT, strokeColor=CRIMSON, strokeWidth=0.8))
    d.add(String(120, 39, "Departing Model (Teacher) — traces captured",
                 fontName="Helvetica-Oblique", fontSize=7,
                 fillColor=CRIMSON_DARK, textAnchor="middle"))

    # Student lane label (right half)
    d.add(Rect(248, 30, 230, 24, rx=3, ry=3,
               fillColor=GOLD_LIGHT, strokeColor=GOLD, strokeWidth=0.8))
    d.add(String(363, 39, "Incoming Model (Student) — distilled & validated",
                 fontName="Helvetica-Oblique", fontSize=7,
                 fillColor=CHARCOAL, textAnchor="middle"))

    # Vertical connectors from lane labels up to phase boxes
    _arrow(d, 120, 54, 120, MID-BH//2, CRIMSON)
    _arrow(d, 363, 54, 363, MID-BH//2, GOLD)
    return d

# ── Fig 14.1: KD-guided prompt optimisation loop ──────────────────────────────
def diag_prompt_kd_loop():
    dw, dh = DW, 180
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white,
               strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    # Central circle at (244, 100)
    CX, CY, CR = 244, 100, 42
    d.add(Circle(CX, CY, CR,
                 fillColor=CRIMSON, strokeColor=white, strokeWidth=1.5))
    d.add(String(CX, CY+7,  "PROMPT",
                 fontName="FreeSerif-Bold", fontSize=9,
                 fillColor=white, textAnchor="middle"))
    d.add(String(CX, CY-7, "OPTIMIZER",
                 fontName="FreeSerif-Bold", fontSize=9,
                 fillColor=white, textAnchor="middle"))

    # Surrounding nodes — kept well inside DW=488
    # Left column (x=8..112): Teacher, Query Set
    # Right column (x=376..480): KD Loss, Optimized Prompt
    # Bottom centre (x=164..324): Student Inference
    NODE_W = 104
    nodes = [
        (8,    142, NODE_W, 26, STEEL,        "Teacher: Gold Responses"),
        (8,     65, NODE_W, 26, GRAY_DARK,    "Query Set + RAG Context"),
        (376,  142, NODE_W, 26, TEAL,         "KD Loss  (KL Divergence)"),
        (376,   65, NODE_W, 26, GOLD,         "Optimized Prompt"),
        (172,   28, 144,   26, CRIMSON_DARK, "Student Inference"),
    ]
    for nx, ny, nw, nh, nc, nlbl in nodes:
        _box(d, nx, ny, nw, nh, nc, nlbl, fsize=7.2)

    # Arrows: circle edge ↔ nodes (approximate)
    _arrow(d, 112,  155, CX-CR,  CY+20, STEEL)      # Teacher → circle left
    _arrow(d, 112,   78, CX-CR,  CY-20, GRAY_DARK)  # Query → circle left
    _arrow(d, CX+CR, CY+20, 376, 155,   TEAL)        # circle right → KD Loss
    _arrow(d, CX+CR, CY-20, 376,  78,   GOLD)        # circle right → Optimized
    _arrow(d, CX,   CY-CR,  CX,   54,   CRIMSON_DARK)# circle bottom → Student
    # Feedback: Student back to Query
    d.add(Line(172, 41, 118, 41,
               strokeColor=GRAY_MED, strokeWidth=0.9,
               strokeDashArray=[4, 3]))
    d.add(Line(118, 41, 118, 65,
               strokeColor=GRAY_MED, strokeWidth=0.9,
               strokeDashArray=[4, 3]))
    return d

# ── Fig 14.2: Soft prompt architecture ───────────────────────────────────────
def diag_soft_prompt():
    dw, dh = DW, 130
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white,
               strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    MID = 62
    # Section 1: 5 soft-prefix tokens (x=8..142)
    for i in range(5):
        x = 8 + i*27
        d.add(Rect(x, MID-13, 22, 26, rx=3, ry=3,
                   fillColor=GOLD, strokeColor=white, strokeWidth=0.8))
        d.add(String(x+11, MID-3, f"p{i+1}",
                     fontName="FreeSerif-Bold", fontSize=7.5,
                     fillColor=CHARCOAL, textAnchor="middle"))

    _lbl(d, 73, MID+22, "Learnable prefix tokens", color=GOLD, fsize=7, bold=True)

    # Concat symbol
    _lbl(d, 155, MID-3, "+", color=GRAY_DARK, fsize=14)

    # Section 2: 5 input tokens (x=170..335)
    toks  = ["[SYS]", "Query", "Chunk", "Chunk", "[ANS]"]
    tcols = [STEEL, CRIMSON_DARK, TEAL, TEAL, GRAY_DARK]
    for i, (tok, col) in enumerate(zip(toks, tcols)):
        x = 168 + i*35
        d.add(Rect(x, MID-13, 30, 26, rx=3, ry=3,
                   fillColor=col, strokeColor=white, strokeWidth=0.8))
        d.add(String(x+15, MID-3, tok,
                     fontName="FreeSerif", fontSize=6,
                     fillColor=white, textAnchor="middle"))
    # rightmost token: 168+4*35+30 = 168+140+30 = 338 ✓

    # Arrow to LLM
    _arrow(d, 342, MID, 378, MID, GRAY_MED)

    # LLM box
    _box(d, 378, MID-20, 60, 40, CRIMSON, "LLM", fsize=11)
    # rightmost: 438 < 488 ✓

    # Output arrow
    _arrow(d, 438, MID, 472, MID, GRAY_MED)
    _lbl(d, 472+6, MID-3, "out", color=GRAY_MED, fsize=7)

    # Gradient backprop label (dashed line at top)
    d.add(Line(8, MID+30, 165, MID+30,
               strokeColor=CRIMSON, strokeWidth=0.9,
               strokeDashArray=[4, 3]))
    _lbl(d, 87, MID+40,
         "∇ KL loss → updates p1–p5 only  (model weights frozen)",
         color=CRIMSON, fsize=7, bold=True)
    return d

# ── Fig 14.3: APO evolutionary search ────────────────────────────────────────
def diag_apo():
    dw, dh = DW, 120
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white,
               strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    BW, GAP, MID = 78, 19, 72
    xs = [10 + i*(BW+GAP) for i in range(5)]
    steps = [
        (CRIMSON,     "Seed Prompt\nPool"),
        (STEEL,       "Student\nInference"),
        (TEAL,        "KD Score\nvs Teacher"),
        (GOLD,        "Select Top-K\n& Mutate"),
        (CRIMSON_DARK,"Next\nGeneration"),
    ]
    for x, (col, lbl) in zip(xs, steps):
        _box(d, x, MID-18, BW, 36, col, lbl.replace("\n"," "), fsize=8)

    for i in range(4):
        _arrow(d, xs[i]+BW, MID, xs[i+1], MID, GRAY_MED)

    # Feedback loop drawn at bottom — y=28 (well above 0) to y=MID-18
    last_right = xs[4] + BW    # 398+78=476 ✓
    loop_y = 30
    d.add(Line(last_right,  MID-18, last_right,  loop_y,
               strokeColor=GRAY_MED, strokeWidth=0.9))
    d.add(Line(last_right,  loop_y, xs[0],        loop_y,
               strokeColor=GRAY_MED, strokeWidth=0.9))
    d.add(Line(xs[0],       loop_y, xs[0],         MID-18,
               strokeColor=GRAY_MED, strokeWidth=0.9))
    _lbl(d, dw/2, 22,
         "Evolution loop — repeat for N generations",
         color=GRAY_MED, fsize=7)
    return d


# ── Fig 10.1: KD-SPAR loop — student rewrites its own prompt ─────────────────
def diag_spar_loop():
    dw, dh = DW, 200
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white,
               strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    # 6 nodes in a rectangle:  top-row (3) + bottom-row (3)
    # Top y=148, Bottom y=52  — all safely within [20, 180]
    NW, NH = 130, 30
    TY, BY = 148, 52    # top / bottom row box-bottom y
    # x centres evenly: gap=(488-3*130)/2 = (488-390)/2=49, starts at 24.5
    xs = [24, 179, 334]   # 3 per row; rightmost=334+130=464 < 488 ✓

    top_nodes = [
        (CRIMSON,     "QUERY + RAG CONTEXT"),
        (STEEL,       "CURRENT PROMPT  P(t)"),
        (TEAL,        "STUDENT INFERENCE"),
    ]
    bot_nodes = [
        (GOLD,        "UPDATED PROMPT P(t+1)"),
        (GRAY_DARK,   "AGGREGATE & VALIDATE"),
        (CRIMSON_DARK,"STUDENT SELF-INTERVIEW"),
    ]
    for (col, lbl), x in zip(top_nodes, xs):
        _box(d, x, TY, NW, NH, col, lbl, fsize=7.2)
    for (col, lbl), x in zip(bot_nodes, xs):
        _box(d, x, BY, NW, NH, col, lbl, fsize=7.2)

    # Top-row arrows (left→right)
    for i in range(2):
        _arrow(d, xs[i]+NW, TY+NH//2, xs[i+1], TY+NH//2, GRAY_MED)

    # Teacher traces feeding in from right (outside top-right node)
    d.add(Rect(dw-88, TY+NH+8, 84, 20, rx=2, ry=2,
               fillColor=CRIMSON_SOFT, strokeColor=CRIMSON, strokeWidth=0.7))
    d.add(String(dw-88+42, TY+NH+15, "Teacher Traces",
                 fontName="FreeSerif-Bold", fontSize=6.5,
                 fillColor=CRIMSON_DARK, textAnchor="middle"))
    _arrow(d, dw-46, TY+NH+8, dw-46, TY+NH, CRIMSON)

    # Right connector: Student Inference → Failure Diagnosis (down)
    _arrow(d, xs[2]+NW//2, TY, xs[2]+NW//2, BY+NH, CRIMSON_DARK, "KD divergence")

    # Bottom-row arrows (right→left)
    for i in range(2, 0, -1):
        _arrow(d, xs[i], BY+NH//2, xs[i-1]+NW, BY+NH//2, GRAY_MED)

    # Left connector: Aggregate → (up to) Query/RAG (closes the loop)
    _arrow(d, xs[0]+NW//2, BY, xs[0]+NW//2, TY+NH, GOLD, "iterate")

    # Labels beneath each bottom node
    sub_labels = ["Prompt improves → adopt", "Vote & score proposals", "Model writes its own fix"]
    for lbl, x in zip(sub_labels, xs):
        d.add(String(x+NW//2, BY-10, lbl, fontName="FreeSerif-Italic",
                     fontSize=6, fillColor=GRAY_MED, textAnchor="middle"))

    # Centre legend
    d.add(String(dw/2, dh-10, "KD-SPAR Loop — Student Drives Its Own Prompt Towards Teacher Alignment",
                 fontName="FreeSerif-Bold", fontSize=7.5,
                 fillColor=CRIMSON, textAnchor="middle"))
    return d



# ── Fig 11.1: Multi-Teacher KD-SPAR ─────────────────────────────────────────
def diag_multi_teacher():
    dw, dh = DW, 175
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white, strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    # Three teachers on left, student centre, combined prompt on right
    TW, TH = 90, 28
    SW, SH = 100, 40
    ty_positions = [132, 90, 48]
    teacher_labels = ["Teacher A\n(Citation Expert)", "Teacher B\n(Reasoning Depth)", "Teacher C\n(Calibration)"]
    teacher_cols   = [CRIMSON, STEEL, TEAL]
    for ty, lbl, col in zip(ty_positions, teacher_labels, teacher_cols):
        _box(d, 10, ty, TW, TH, col, lbl.replace("\n"," "), fsize=6.8)
        _arrow(d, 100, ty+TH//2, 155, 100, GRAY_MED)

    # Student centre
    _box(d, 155, 80, SW, SH, CRIMSON_DARK, "STUDENT\n(single model)".replace("\n"," "), fsize=8)

    # KD signals stacked
    d.add(Rect(270, 72, 95, 56, rx=4, ry=4,
               fillColor=GOLD_LIGHT, strokeColor=GOLD, strokeWidth=0.8))
    d.add(String(318, 122, "KD Scores",    fontName="FreeSerif-Bold", fontSize=7, fillColor=CRIMSON_DARK, textAnchor="middle"))
    d.add(String(318, 109, "per teacher", fontName="FreeSerif",       fontSize=6.5, fillColor=GRAY_DARK,   textAnchor="middle"))
    d.add(String(318,  96, "-> find worst", fontName="FreeSerif-Italic", fontSize=6.5, fillColor=CRIMSON, textAnchor="middle"))
    d.add(String(318,  83, "-> interview",  fontName="FreeSerif-Italic", fontSize=6.5, fillColor=STEEL,   textAnchor="middle"))
    _arrow(d, 255, 100, 270, 100, GRAY_MED)

    # Updated prompt
    _box(d, 380, 80, 100, 40, CRIMSON, "UPDATED\nPROMPT P(t+1)".replace("\n"," "), fsize=7.5)
    _arrow(d, 365, 100, 380, 100, GRAY_MED)

    # Validation note
    d.add(Rect(270, 30, 210, 24, rx=3, ry=3,
               fillColor=CRIMSON_SOFT, strokeColor=CRIMSON, strokeWidth=0.7))
    d.add(String(375, 39, "Validate: primary ↑  AND  no secondary regression > tol",
                 fontName="FreeSerif-Italic", fontSize=6.5,
                 fillColor=CRIMSON_DARK, textAnchor="middle"))

    d.add(String(dw/2, dh-8, "Multi-Teacher KD-SPAR — student must satisfy all teachers simultaneously",
                 fontName="FreeSerif-Bold", fontSize=7.5,
                 fillColor=CRIMSON, textAnchor="middle"))
    return d


# ── Fig 12.1: Adversarial KD-SPAR ────────────────────────────────────────────
def diag_adversarial():
    dw, dh = DW, 165
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white, strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    BW, GAP, MID = 78, 18, 95
    xs = [10 + i*(BW+GAP) for i in range(5)]

    steps = [
        (CRIMSON,     "Mine Hard\nExamples"),
        (STEEL,       "Generate\nAdv. Queries"),
        (TEAL,        "Adversarial\nDiagnosis"),
        (GOLD,        "Adversarial\nInterview"),
        (CRIMSON_DARK,"Dual-Objective\nValidation"),
    ]
    for x, (col, lbl) in zip(xs, steps):
        _box(d, x, MID-18, BW, 36, col, lbl.replace("\n"," "), fsize=7.5)
    for i in range(4):
        _arrow(d, xs[i]+BW, MID, xs[i+1], MID, GRAY_MED)

    # Gap-mining annotation
    d.add(Rect(10, 35, 78, 22, rx=2, ry=2, fillColor=CRIMSON_SOFT, strokeColor=CRIMSON, strokeWidth=0.6))
    d.add(String(49, 43, "Bottom decile KD scores", fontName="FreeSerif-Italic", fontSize=6, fillColor=CRIMSON_DARK, textAnchor="middle"))

    # Generation annotation
    d.add(Rect(10+BW+GAP, 35, 78, 22, rx=2, ry=2, fillColor=GOLD_LIGHT, strokeColor=GOLD, strokeWidth=0.6))
    d.add(String(49+BW+GAP, 43, "Teacher generates hard Qs", fontName="FreeSerif-Italic", fontSize=6, fillColor=CHARCOAL, textAnchor="middle"))

    # Dual validation box
    _box(d, xs[4], 35, BW, 20, GRAY_DARK, "Adv ↑ AND Std ≥ −tol", fsize=6.5, radius=2)

    d.add(String(dw/2, dh-8, "Adversarial KD-SPAR — focuses exclusively on hard examples to build robustness",
                 fontName="FreeSerif-Bold", fontSize=7.5, fillColor=CRIMSON, textAnchor="middle"))
    return d


# ── Fig 13.1: Federated KD-SPAR ──────────────────────────────────────────────
def diag_federated():
    dw, dh = DW, 190
    d = Drawing(dw, dh)
    d.add(Rect(0, 0, dw, dh, fillColor=white, strokeColor=GRAY_LIGHT, strokeWidth=0.3))

    # Central server
    _box(d, 169, 118, 150, 46, CRIMSON, "AGGREGATION SERVER\nCluster • Score • Broadcast".replace("\n"," "), fsize=7.5)

    # Three clients
    client_xs = [10, 169, 328]
    client_labels = ["CLIENT A\n(Hospital)", "CLIENT B\n(Branch)", "CLIENT C\n(Dept)"]
    client_cols   = [STEEL, TEAL, GOLD]
    for cx, lbl, col in zip(client_xs, client_labels, client_cols):
        _box(d, cx, 30, 120, 40, col, lbl.replace("\n"," "), fsize=7.5)
        # Up arrows: proposals (instructions only) to server
        _arrow(d, cx+60, 70, 244, 118, GRAY_MED)
        # Down arrows: global prompt from server
        _arrow(d, 244, 118, cx+60, 70, CRIMSON)

    # Privacy label
    d.add(Rect(10, dh-28, 290, 18, rx=3, ry=3,
               fillColor=CRIMSON_SOFT, strokeColor=CRIMSON, strokeWidth=0.7))
    d.add(String(155, dh-19, "↑ Only instruction strings cross boundary — no query/response data shared ↑",
                 fontName="FreeSerif-Italic", fontSize=6.5,
                 fillColor=CRIMSON_DARK, textAnchor="middle"))

    # Legend
    d.add(String(dw-80, 48, "-> global prompt",  fontName="FreeSerif", fontSize=6.5, fillColor=CRIMSON, textAnchor="middle"))
    d.add(String(dw-80, 36, "<- proposals only", fontName="FreeSerif", fontSize=6.5, fillColor=GRAY_MED, textAnchor="middle"))

    d.add(String(dw/2, dh-8, "Federated KD-SPAR — privacy-preserving distributed prompt optimisation",
                 fontName="FreeSerif-Bold", fontSize=7.5, fillColor=CRIMSON, textAnchor="middle"))
    return d


def embed_diagram(drw, cap=""):
    out = [renderPDF.GraphicsFlowable(drw), Spacer(1, 10)]
    if cap:
        out += caption(cap)
    out += [Spacer(1, 6)]
    return out


# ── Part-divider banner ───────────────────────────────────────────────────────
_Spart_n = ParagraphStyle("part_n", fontName="Helvetica-Bold",   fontSize=8,
                           textColor=GOLD,  spaceAfter=0)
_Spart_t = ParagraphStyle("part_t", fontName=_BODY_FONT_BOLD,     fontSize=14,
                           textColor=white, spaceAfter=0)
_Spart_s = ParagraphStyle("part_s", fontName=_BODY_FONT_ITAL,     fontSize=9,
                           textColor=HexColor("#FADADD"), spaceAfter=0)

def part_banner(num, title, subtitle=""):
    rows = [[
        Paragraph(f"PART {num}", _Spart_n),
        Paragraph(title, _Spart_t),
        Paragraph(subtitle, _Spart_s),
    ]]
    t = Table(rows, colWidths=[0.65*inch, 3.2*inch, 3.0*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CRIMSON),
        ("LINEBELOW",     (0,0), (-1,-1), 1.5, GOLD),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    return [Spacer(1, 18), t, Spacer(1, 14)]



# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# CHART GENERATORS — for Section 20 results visualisation
# ═══════════════════════════════════════════════════════════════════════════

_C_TEAL   = HexColor("#2E8B8B")
_C_GREEN  = HexColor("#38A169")
_C_RED    = HexColor("#E53E3E")
COND_COLORS = {"D": GRAY_MED, "C": _C_RED, "B": GOLD, "A": CRIMSON, "E": _C_TEAL, "F": _C_GREEN}
COND_NAMES  = {"D":"Baseline","C":"Random","B":"External","A":"KD-SPAR","E":"MetaKDSPAR","F":"Enhanced"}

def results_bar_chart(labels, cond_scores, title="", fig_num="20.1", width=DW, height=195):
    """Grouped bar chart: models on x, conditions as coloured bars."""
    d = Drawing(width, height)
    d.add(Rect(0,0,width,height,fillColor=HexColor("#F7F7F7"),strokeColor=None))
    conds = [c for c in ["D","C","B","A","E","F"] if c in cond_scores]
    nm, nc = len(labels), len(conds)
    if nm==0 or nc==0: return []
    ml,mr,mb,mt = 52,15,48,22
    cw2,ch2 = width-ml-mr, height-mb-mt
    gw = cw2/nm; bw = gw/(nc+1.5)
    av = [v for s in cond_scores.values() for v in s if v>0]
    if not av: return []
    ymin,ymax = max(min(av)-0.08,0), min(max(av)+0.05,1.0)
    sy = lambda v: mb+(v-ymin)/max(ymax-ymin,0.01)*ch2
    for i in range(6):
        yv = ymin+i*(ymax-ymin)/5; yp = sy(yv)
        d.add(Line(ml,yp,width-mr,yp,strokeColor=HexColor("#D0D0D0"),strokeWidth=0.3))
        d.add(String(ml-5,yp-3,f"{yv:.2f}",fontName="Helvetica",fontSize=6,fillColor=GRAY_MED,textAnchor="end"))
    for gi,model in enumerate(labels):
        gx = ml+gi*gw
        for ci,c in enumerate(conds):
            sc = cond_scores[c]
            if gi>=len(sc) or sc[gi]==0: continue
            bx = gx+(ci+0.5)*bw; by0 = sy(ymin); bh = sy(sc[gi])-by0
            d.add(Rect(bx,by0,bw*0.85,max(bh,1),fillColor=COND_COLORS.get(c,GRAY_MED),strokeColor=None))
        lbl = model.split(":")[0] if ":" in model else model
        d.add(String(gx+gw/2,mb-14,lbl,fontName="Helvetica-Bold",fontSize=6.5,fillColor=CHARCOAL,textAnchor="middle"))
        sz = model.split(":")[-1] if ":" in model else ""
        d.add(String(gx+gw/2,mb-24,sz,fontName="Helvetica",fontSize=5.5,fillColor=GRAY_MED,textAnchor="middle"))
    lx = width-mr-90
    for i,c in enumerate(conds):
        ly = height-mt-3-i*10
        d.add(Rect(lx,ly,7,7,fillColor=COND_COLORS.get(c,GRAY_MED),strokeColor=None))
        d.add(String(lx+10,ly,f"{c}: {COND_NAMES.get(c,c)}",fontName="Helvetica",fontSize=5.5,fillColor=CHARCOAL))
    d.add(Line(ml,mb,ml,height-mt,strokeColor=CHARCOAL,strokeWidth=0.8))
    d.add(Line(ml,mb,width-mr,mb,strokeColor=CHARCOAL,strokeWidth=0.8))
    ft = f"Figure {fig_num} \u2014 {title}" if title else f"Figure {fig_num}"
    d.add(String(width/2,height-10,ft,fontName="Helvetica-Bold",fontSize=7.5,fillColor=CHARCOAL,textAnchor="middle"))
    d.add(String(10,height/2,"KD Score",fontName="Helvetica-Bold",fontSize=7,fillColor=CRIMSON,textAnchor="middle"))
    return [Spacer(1,8),d,Spacer(1,10)]

def results_gap_chart(labels, gaps, title="", fig_num="20.2", threshold=0.02, width=DW, height=170):
    """Bar chart of gaps (pos/neg) with threshold line. gaps: {name: [val_per_model]}"""
    d = Drawing(width, height)
    d.add(Rect(0,0,width,height,fillColor=HexColor("#F7F7F7"),strokeColor=None))
    gnames = list(gaps.keys()); nm = len(labels); ng = len(gnames)
    if nm==0: return []
    ml,mr,mb,mt = 52,15,48,22
    cw2,ch2 = width-ml-mr, height-mb-mt
    gw = cw2/nm; bw = gw/(ng+1.5)
    av = [v for gv in gaps.values() for v in gv]
    ymin,ymax = min(min(av)-0.01,-0.03), max(max(av)+0.01,0.05)
    sy = lambda v: mb+(v-ymin)/max(ymax-ymin,0.01)*ch2
    y0 = sy(0)
    d.add(Line(ml,y0,width-mr,y0,strokeColor=HexColor("#888"),strokeWidth=0.8,strokeDashArray=[4,3]))
    if ymin<threshold<ymax:
        yt = sy(threshold)
        d.add(Line(ml,yt,width-mr,yt,strokeColor=_C_GREEN,strokeWidth=0.6,strokeDashArray=[2,2]))
        d.add(String(width-mr+2,yt-3,"strong",fontName="Helvetica",fontSize=5,fillColor=_C_GREEN))
    gcols = [CRIMSON,_C_TEAL,_C_GREEN,GOLD,HexColor("#805AD5")]
    for gi,model in enumerate(labels):
        gx = ml+gi*gw
        for ji,gn in enumerate(gnames):
            vals = gaps[gn]
            if gi>=len(vals): continue
            v = vals[gi]; bx = gx+(ji+0.5)*bw; by0 = sy(0); bh = sy(v)-by0
            col = gcols[ji%len(gcols)]
            d.add(Rect(bx,min(by0,by0+bh),bw*0.85,abs(bh),fillColor=col,strokeColor=None))
            d.add(String(bx+bw*0.4,max(by0,by0+bh)+3,f"{v:+.3f}",fontName="Helvetica",fontSize=5.5,fillColor=col,textAnchor="middle"))
        lbl = model.split(":")[0] if ":" in model else model
        d.add(String(gx+gw/2,mb-14,lbl,fontName="Helvetica-Bold",fontSize=6.5,fillColor=CHARCOAL,textAnchor="middle"))
    lx = width-mr-80
    for i,gn in enumerate(gnames):
        ly = height-mt-3-i*10
        d.add(Rect(lx,ly,7,7,fillColor=gcols[i%len(gcols)],strokeColor=None))
        d.add(String(lx+10,ly,gn,fontName="Helvetica",fontSize=6,fillColor=CHARCOAL))
    d.add(Line(ml,mb,ml,height-mt,strokeColor=CHARCOAL,strokeWidth=0.8))
    ft = f"Figure {fig_num} \u2014 {title}" if title else f"Figure {fig_num}"
    d.add(String(width/2,height-10,ft,fontName="Helvetica-Bold",fontSize=7.5,fillColor=CHARCOAL,textAnchor="middle"))
    return [Spacer(1,8),d,Spacer(1,10)]

