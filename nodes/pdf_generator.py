"""
PDF Generator — builds a professional PDF from a MiningReport JSON dict.
Uses fpdf2 (pure Python). Returns raw PDF bytes for Supabase Storage upload.

Design: Dark navy (#1A3A5F) headers, gold (#C4A04A) accents, clean tables.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional

from fpdf import FPDF

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY      = (26, 58, 95)
NAVY_LIGHT= (45, 90, 145)
GOLD      = (196, 160, 74)
WHITE     = (255, 255, 255)
LIGHT_BG  = (240, 245, 252)
ALT_ROW   = (230, 238, 248)
GRAY_TEXT = (90, 90, 90)
DARK_TEXT = (30, 30, 30)
GREEN     = (34, 139, 34)
AMBER     = (210, 140, 0)
RED_RISK  = (190, 50, 50)
GREEN_BG  = (220, 245, 220)
AMBER_BG  = (255, 244, 204)
RED_BG    = (252, 225, 225)

# ── Design B layout constants ──────────────────────────────────────────────────
SIDEBAR_X  = 140        # sidebar left edge (mm from page left)
SIDEBAR_W  = 55         # sidebar column width (mm)
CONTENT_W  = 124        # main text column width when sidebar is active (mm)
FULL_W     = 182        # full usable width when no sidebar (mm)
GOLD_BAR_H = 8          # thin gold bar height on cover (mm)


def _s(value, default="N/A") -> str:
    """Safe string — coerce to str, strip non-latin-1."""
    if value is None:
        return default
    return str(value).encode("latin-1", errors="replace").decode("latin-1")


def _num(value, fmt="{:,.0f}", default="N/A") -> str:
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return default


class MIPdf(FPDF):
    def __init__(self, project_name: str, material: str, generated_date: str):
        super().__init__()
        self._proj         = _s(project_name)
        self._mat          = _s(material)
        self._date         = _s(generated_date)
        self._section_name = ""  # updated by each renderer before add_page()

    def header(self):
        if self.page_no() == 1:
            return  # cover page manages its own layout
        # Design B running header: "PROJECT | RESOURCE MODELING REPORT  SECTION"
        section_suffix = f"  {self._section_name}" if self._section_name else ""
        left_text = f"{self._proj}  |  RESOURCE MODELING REPORT{section_suffix}"
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*NAVY)
        self.set_xy(14, 10)
        self.cell(140, 5, left_text, ln=False)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GRAY_TEXT)
        self.cell(42, 5, self._date, align="R", ln=True)
        self.set_draw_color(*GOLD)
        self.set_line_width(0.6)
        self.line(14, self.get_y(), 196, self.get_y())
        self.set_line_width(0.2)
        self.set_draw_color(0, 0, 0)
        self.ln(3)
        self.set_text_color(*DARK_TEXT)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GRAY_TEXT)
        self.cell(0, 6,
            f"Page {self.page_no()}  |  NOT NI 43-101 or JORC compliant  |  For internal use only  |  Mining Intellect",
            align="C")
        self.set_text_color(*DARK_TEXT)


# ── Section helpers ─────────────────────────────────────────────────────────────

def _section_header(pdf: MIPdf, title: str) -> None:
    """
    Design B section header: large section number left, title right, gold rule below.
    Parses section number from title string like "1. Executive Summary" → "01".
    """
    pdf.ln(4)
    parts = title.split(".", 1)
    num_str = parts[0].strip()
    if num_str.isdigit():
        section_num  = num_str.zfill(2)
        section_text = title
    else:
        section_num  = ""
        section_text = title

    if section_num:
        # Large "01" in navy, left-aligned
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(*NAVY)
        pdf.set_x(14)
        pdf.cell(20, 10, section_num, ln=False)
        # Section title to the right, vertically aligned to lower half of number
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_xy(34, pdf.get_y() + 3)
        pdf.cell(CONTENT_W - 20, 8, _s(section_text), ln=True)
    else:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*NAVY)
        pdf.set_x(14)
        pdf.cell(FULL_W, 8, _s(section_text), ln=True)

    # Gold horizontal rule
    rule_y = pdf.get_y() + 1
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.8)
    pdf.line(14, rule_y, 196, rule_y)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_text_color(*DARK_TEXT)
    pdf.ln(5)


def _subsection(pdf: MIPdf, title: str) -> None:
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 7, f"  {_s(title)}", ln=True, fill=True)
    pdf.set_text_color(*DARK_TEXT)
    pdf.ln(1)


def _body(pdf: MIPdf, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK_TEXT)
    pdf.multi_cell(0, 5, _s(text))
    pdf.set_x(pdf.l_margin)
    pdf.ln(2)


def _kv(pdf: MIPdf, label: str, value, label_w: int = 52) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(label_w, 5, _s(label) + ":", ln=False)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK_TEXT)
    pdf.multi_cell(0, 5, _s(value))
    pdf.set_x(pdf.l_margin)


def _bullet(pdf: MIPdf, text: str, indent: int = 4) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK_TEXT)
    pdf.cell(indent, 5, "", ln=False)
    pdf.cell(5, 5, "\x95", ln=False)  # bullet character
    pdf.multi_cell(0, 5, _s(text))
    pdf.set_x(pdf.l_margin)


# ── Narrow variants for sidebar-aware pages ──────────────────────────────────

def _body_narrow(pdf: MIPdf, text: str, width: float = CONTENT_W) -> None:
    """Like _body but constrained to width (leaves room for right sidebar)."""
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK_TEXT)
    pdf.multi_cell(width, 5, _s(text))
    pdf.set_x(pdf.l_margin)
    pdf.ln(2)


def _kv_narrow(pdf: MIPdf, label: str, value, label_w: int = 44, width: float = CONTENT_W) -> None:
    """Like _kv but value column constrained to width."""
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(label_w, 5, _s(label) + ":", ln=False)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK_TEXT)
    pdf.multi_cell(width - label_w, 5, _s(value))
    pdf.set_x(pdf.l_margin)


def _bullet_narrow(pdf: MIPdf, text: str, indent: int = 4, width: float = CONTENT_W) -> None:
    """Like _bullet but constrained to width."""
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK_TEXT)
    pdf.cell(indent, 5, "", ln=False)
    pdf.cell(5, 5, "\x95", ln=False)
    pdf.multi_cell(width - indent - 5, 5, _s(text))
    pdf.set_x(pdf.l_margin)


def _grade_fmt(grade_pct: float, material: str = "") -> str:
    """Format a grade value. Adds ppm equivalent when grade < 2% for base metals."""
    mat = (material or "").lower()
    precious = any(m in mat for m in ("gold", "au", "silver", "ag", "platinum", "palladium", "pgm"))
    if precious:
        return f"{grade_pct:.2f} g/t"
    if 0 < grade_pct < 2.0:
        ppm = int(round(grade_pct * 10_000))
        return f"{grade_pct:.3f}% ({ppm:,} ppm)"
    return f"{grade_pct:.3f}%"


def _count_lines(pdf: MIPdf, text: str, width: float) -> int:
    """Return the number of lines text will occupy when wrapped to width."""
    if not text:
        return 1
    try:
        # fpdf2 >= 2.5: split_only returns lines without rendering
        return max(1, len(pdf.multi_cell(width, 5, _s(text), split_only=True)))
    except TypeError:
        # Fallback: manual word-wrap estimate
        total = 0
        for para in _s(text).split("\n"):
            words = para.split()
            if not words:
                total += 1
                continue
            lc, lw = 1, 0.0
            for word in words:
                ww = pdf.get_string_width(word + " ")
                if lw + ww > width and lw > 0:
                    lc += 1
                    lw = ww
                else:
                    lw += ww
            total += lc
        return max(1, total)


def _mc_row(pdf: MIPdf, cells: List[Dict]) -> None:
    """
    Draw one table row where every cell auto-sizes its height to fit wrapped text.

    Each cell dict:
      val (str), width (int), align (str "L"|"C"|"R"),
      fill (RGB tuple | None), text_color (RGB tuple | None),
      font_style (str, default ""), font_size (int, default 8)
    """
    LINE_H = 5
    PAD    = 2   # left + right text padding inside cell

    # Pass 1 — measure: find the tallest cell to set a uniform row height
    row_h = LINE_H + PAD
    for c in cells:
        pdf.set_font("Helvetica", c.get("font_style", ""), c.get("font_size", 8))
        n = _count_lines(pdf, _s(c.get("val", "")), c["width"] - PAD * 2)
        needed = n * LINE_H + PAD
        if needed > row_h:
            row_h = needed

    # Pass 2 — draw: render every cell at the shared height
    x0, y0 = pdf.get_x(), pdf.get_y()
    x = x0
    for c in cells:
        w = c["width"]
        fill = c.get("fill")
        if fill:
            pdf.set_fill_color(*fill)
            pdf.rect(x, y0, w, row_h, style="F")
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.2)
        pdf.rect(x, y0, w, row_h, style="D")
        pdf.set_xy(x + PAD, y0 + 1)
        pdf.set_font("Helvetica", c.get("font_style", ""), c.get("font_size", 8))
        pdf.set_text_color(*(c.get("text_color") or DARK_TEXT))
        pdf.multi_cell(
            w - PAD * 2, LINE_H, _s(c.get("val", "")),
            border=0, align=c.get("align", "L"), fill=False,
        )
        x += w

    pdf.set_text_color(*DARK_TEXT)
    pdf.set_xy(pdf.l_margin, y0 + row_h)


def _table_header(pdf: MIPdf, cols: List[str], widths: List[int]) -> None:
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 8)
    for col, w in zip(cols, widths):
        pdf.cell(w, 7, _s(col), border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(*DARK_TEXT)


def _table_row(pdf: MIPdf, vals: List[str], widths: List[int], aligns: List[str],
               fill_color=None, font_style: str = "") -> None:
    _mc_row(pdf, [
        {"val": v, "width": w, "align": a, "fill": fill_color, "font_style": font_style}
        for v, w, a in zip(vals, widths, aligns)
    ])


# ── Metric highlight boxes ──────────────────────────────────────────────────────

def _metric_box(pdf: MIPdf, label: str, value: str, unit: str = "", x: float = None) -> None:
    """Draw a single metric box. If x given, position absolutely."""
    box_w, box_h = 42, 22
    if x is not None:
        pdf.set_xy(x, pdf.get_y())
    pdf.set_fill_color(*LIGHT_BG)
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.6)
    pdf.rect(pdf.get_x(), pdf.get_y(), box_w, box_h, style="FD")
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)

    # Value (large, navy)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*NAVY)
    cy = pdf.get_y() + 3
    pdf.set_xy(pdf.get_x() - box_w, cy)  # back to box start
    # We drew rect, now write text inside
    start_x = pdf.get_x()
    pdf.set_xy(start_x, cy)
    pdf.cell(box_w, 7, _s(value), align="C", ln=False)
    pdf.set_xy(start_x, cy + 7)
    if unit:
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(box_w, 4, _s(unit), align="C", ln=False)
        pdf.set_xy(start_x, cy + 11)
    else:
        pdf.set_xy(start_x, cy + 7)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(box_w, 4, _s(label), align="C", ln=False)
    pdf.set_text_color(*DARK_TEXT)


def _metric_row(pdf: MIPdf, metrics: List[tuple]) -> None:
    """Draw a row of metric boxes. Each tuple: (label, value, unit)."""
    start_y = pdf.get_y()
    start_x = pdf.get_x()
    box_w, box_h, gap = 42, 22, 3
    for i, (label, value, unit) in enumerate(metrics):
        x = start_x + i * (box_w + gap)
        pdf.set_xy(x, start_y)
        # Draw box
        pdf.set_fill_color(*LIGHT_BG)
        pdf.set_draw_color(*GOLD)
        pdf.set_line_width(0.6)
        pdf.rect(x, start_y, box_w, box_h, style="FD")
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        # Value
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*NAVY)
        pdf.set_xy(x, start_y + 2)
        pdf.cell(box_w, 7, _s(value), align="C")
        # Unit
        if unit:
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.set_xy(x, start_y + 9)
            pdf.cell(box_w, 4, _s(unit), align="C")
        # Label
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.set_xy(x, start_y + 14)
        pdf.cell(box_w, 4, _s(label), align="C")
    pdf.set_text_color(*DARK_TEXT)
    pdf.set_xy(start_x, start_y + box_h + 3)


# ── Risk colour helper ──────────────────────────────────────────────────────────

# ── Design B right-column sidebar ──────────────────────────────────────────────

def _sidebar_metric_box(
    pdf: MIPdf,
    label: str,
    value: str,
    unit: str = "",
    sub_label: str = "",
    x: float = None,
    y: float = None,
    primary: bool = False,
) -> float:
    """
    Draw one stacked sidebar metric box at absolute (x, y).
    Returns the bottom y of the box so callers can chain boxes vertically.
    primary=True → navy bg, white text, taller box (hero metric).
    """
    if x is None:
        x = SIDEBAR_X
    if y is None:
        y = pdf.get_y()

    w      = SIDEBAR_W - 2   # 1mm gutter each side
    box_h  = 30 if primary else 22

    if primary:
        bg          = NAVY
        val_color   = WHITE
        unit_color  = GOLD
        label_color = (180, 200, 230)
        val_size    = 18
    else:
        bg          = LIGHT_BG
        val_color   = NAVY
        unit_color  = GRAY_TEXT
        label_color = GRAY_TEXT
        val_size    = 13

    # Background + border
    pdf.set_fill_color(*bg)
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.5)
    pdf.rect(x, y, w, box_h, style="FD")
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)

    # Label (top small caps)
    pdf.set_font("Helvetica", "B", 6)
    pdf.set_text_color(*label_color)
    pdf.set_xy(x, y + 2)
    pdf.cell(w, 4, _s(label).upper(), align="C")

    # Value (large number)
    pdf.set_font("Helvetica", "B", val_size)
    pdf.set_text_color(*val_color)
    pdf.set_xy(x, y + 7)
    pdf.cell(w, 8, _s(value), align="C")

    # Unit
    if unit:
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*unit_color)
        pdf.set_xy(x, y + 15)
        pdf.cell(w, 4, _s(unit), align="C")

    # Secondary label (e.g. "175 Mt @ 1.5 g/t")
    if sub_label:
        pdf.set_font("Helvetica", "I", 6)
        pdf.set_text_color(*label_color)
        pdf.set_xy(x, y + box_h - 5)
        pdf.cell(w, 4, _s(sub_label), align="C")

    pdf.set_text_color(*DARK_TEXT)
    return y + box_h + 1   # next available y (1mm gap between boxes)


def _draw_sidebar(pdf: MIPdf, metrics: list, start_y: float) -> None:
    """
    Render a stacked column of sidebar metric boxes at SIDEBAR_X.
    Uses absolute positioning — does NOT move the main content cursor.
    metrics: list of dicts with keys: label, value, unit, sub_label (opt), primary (opt bool).
    """
    current_y = start_y
    for m in metrics:
        current_y = _sidebar_metric_box(
            pdf,
            label     = m.get("label", ""),
            value     = m.get("value", ""),
            unit      = m.get("unit", ""),
            sub_label = m.get("sub_label", ""),
            x         = SIDEBAR_X,
            y         = current_y,
            primary   = m.get("primary", False),
        )


def _risk_color(impact: str, probability: str):
    """Return background RGB for risk cell."""
    h = str(impact).lower()
    p = str(probability).lower()
    if "high" in h and "high" in p:
        return RED_BG
    if "high" in h or "high" in p:
        return AMBER_BG
    return GREEN_BG


# ── Cover page ──────────────────────────────────────────────────────────────────

def _cover_page(pdf: MIPdf, report_json: Dict, project_name: str) -> None:
    """Design B cover: white page, gold top bar, left content column, right sidebar."""
    meta   = report_json.get("metadata", {})
    exec_s = report_json.get("executive_summary", {})
    material     = _s(meta.get("material", ""))
    deposit_type = _s(meta.get("deposit_type", ""))
    country      = _s(meta.get("country", ""))
    stage        = _s(meta.get("project_stage", ""))
    date_str     = _s(str(meta.get("generated_at", ""))[:10])
    assessment   = exec_s.get("overall_assessment", "Cautious")

    # ── 1. White background ────────────────────────────────────────────────────
    pdf.set_fill_color(*WHITE)
    pdf.rect(0, 0, 210, 297, style="F")

    # ── 2. Thin gold top bar ──────────────────────────────────────────────────
    pdf.set_fill_color(*GOLD)
    pdf.rect(0, 0, 210, GOLD_BAR_H, style="F")
    pdf.set_xy(14, 1.5)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, GOLD_BAR_H - 3, "MINING INTELLECT AI PLATFORM", ln=True)

    # ── 3. Left column (x=14, w=122) ──────────────────────────────────────────
    left_y = GOLD_BAR_H + 7

    # Date | description small gray
    pdf.set_xy(14, left_y)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(122, 5, f"{date_str}  |  AI-Powered Resource Analysis", ln=True)

    # "RESOURCE MODELING REPORT"
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*NAVY)
    pdf.cell(122, 8, "RESOURCE MODELING REPORT", ln=True)

    # Gold rule (left column only)
    sep_y = pdf.get_y() + 1
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(1.0)
    pdf.line(14, sep_y, 136, sep_y)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)

    # Project name (22pt bold navy)
    pdf.set_xy(14, sep_y + 4)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(122, 10, _s(project_name))

    # Meta tags
    meta_line = "  |  ".join(filter(None, [material, deposit_type, country, stage]))
    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.multi_cell(122, 5, meta_line)

    # Assessment badge
    badge_colors = {
        "Positive": (GREEN, GREEN_BG),
        "Cautious":  (AMBER, AMBER_BG),
        "Negative": (RED_RISK, RED_BG),
    }
    badge_text_c, badge_bg_c = badge_colors.get(assessment, (NAVY, LIGHT_BG))
    pdf.set_x(14)
    pdf.set_fill_color(*badge_bg_c)
    pdf.set_text_color(*badge_text_c)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(50, 6, f"  Assessment: {assessment}  ", fill=True, ln=True, align="C")
    pdf.set_text_color(*DARK_TEXT)
    pdf.ln(4)

    # Compact table of contents on cover
    toc_y = pdf.get_y()
    pdf.set_x(14)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*NAVY)
    pdf.cell(122, 5, "CONTENTS", ln=True)
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.3)
    pdf.line(14, pdf.get_y(), 136, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)
    pdf.ln(1)

    toc_sections = [
        ("01", "Executive Summary"),
        ("02", "Project Overview"),
        ("03", "Geological Framework"),
        ("04", "Analog Comparison"),
        ("05", "Resource Models"),
        ("06", "Drilling & Sampling"),
        ("07", "Sensitivity Analysis"),
        ("08", "Risk Matrix"),
        ("09", "Recommendations"),
        ("10", "Acquisition Analysis"),
        ("11", "Conclusion"),
        ("12", "Disclaimer"),
    ]
    for num, name in toc_sections:
        pdf.set_x(14)
        pdf.set_text_color(*NAVY)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(12, 5, num, ln=False)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*DARK_TEXT)
        pdf.cell(110, 5, name, ln=True)

    # Key takeaway below TOC
    key_takeaway = exec_s.get("key_takeaway", "")
    if key_takeaway:
        pdf.ln(3)
        pdf.set_x(14)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*NAVY)
        pdf.multi_cell(122, 5, f'"{_s(key_takeaway)}"')

    # ── 4. Right sidebar metric boxes (absolute positioning) ──────────────────
    comp_table = (report_json.get("resource_estimates") or {}).get("comparison_table", [])
    best_model = next((r for r in comp_table if "Model 1" in str(r.get("model",""))), None)
    if not best_model and comp_table:
        best_model = comp_table[0]
    ind = (report_json.get("resource_estimates") or {}).get("independent_analysis", {})
    upd = (report_json.get("resource_estimates") or {}).get("updated_analysis", {})

    sb_y = GOLD_BAR_H + 5

    if best_model:
        total_kt  = float(best_model.get("total_tonnage_kt", 0) or 0)
        total_mt  = total_kt / 1000
        mi_kt     = float(best_model.get("mi_tonnage_kt", 0) or 0)
        mi_mt     = mi_kt / 1000
        mi_grade  = float(best_model.get("mi_grade_pct", 0) or 0)
        conv_pct  = float(ind.get("confidence_pct", 0) or 0)
        conv2_pct = float(upd.get("confidence_pct", 0) or conv_pct)

        sb_y = _sidebar_metric_box(pdf, "TOTAL RESOURCE",
            f"{total_mt:.1f}", "Mt", x=SIDEBAR_X, y=sb_y, primary=True)
        sb_y = _sidebar_metric_box(pdf, "CONFIDENCE",
            f"{conv_pct:.0f}", "%", x=SIDEBAR_X, y=sb_y)
        hg_sub = f"@ {_grade_fmt(mi_grade, material)}" if mi_grade else ""
        sb_y = _sidebar_metric_box(pdf, "HIGH-GRADE (M&I)",
            f"{mi_mt:.0f}", "Mt", sub_label=hg_sub, x=SIDEBAR_X, y=sb_y)
        sb_y = _sidebar_metric_box(pdf, "MI CONVICTION",
            f"{conv2_pct:.0f}", "%", x=SIDEBAR_X, y=sb_y)

        # Prepared by stamp at bottom of sidebar
        stamp_y = sb_y + 4
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*NAVY)
        pdf.set_xy(SIDEBAR_X, stamp_y)
        pdf.cell(SIDEBAR_W - 2, 4, "Prepared by:", align="C")
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.set_xy(SIDEBAR_X, stamp_y + 4)
        pdf.cell(SIDEBAR_W - 2, 4, "Mining Intellect AI Platform", align="C")

    # ── 5. Disclaimer amber box at bottom ─────────────────────────────────────
    pdf.set_xy(14, 258)
    pdf.set_fill_color(255, 248, 220)
    pdf.set_draw_color(*AMBER)
    pdf.set_line_width(0.4)
    pdf.rect(14, 258, 182, 22, style="FD")
    pdf.set_line_width(0.2)
    pdf.set_xy(16, 260)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_text_color(*AMBER)
    pdf.cell(0, 4, "DISCLAIMER", ln=True)
    pdf.set_xy(16, 264)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(100, 80, 0)
    pdf.multi_cell(178, 4,
        "This report is for informational purposes only. It does not replace a regulatory-compliant "
        "mineral resource estimate (NI 43-101, JORC). Not prepared by a Qualified Person. "
        "Not investment advice. Estimates may change with new data.")
    pdf.set_text_color(*DARK_TEXT)


# ── Table of Contents ───────────────────────────────────────────────────────────

def _toc_page(pdf: MIPdf, report_json: Dict) -> None:
    """Design B: TOC is embedded on the cover page — this is a no-op."""
    pass


# ── Section renderers ───────────────────────────────────────────────────────────

def _render_executive_summary(pdf: MIPdf, report_json: Dict) -> None:
    exec_s   = report_json.get("executive_summary", {})
    res_est  = report_json.get("resource_estimates", {})
    material = report_json.get("metadata", {}).get("material", "")

    pdf._section_name = "EXECUTIVE SUMMARY"
    _section_header(pdf, "1. Executive Summary")

    content_start_y = pdf.get_y()

    # Build and draw sidebar first (absolute positioning, no cursor movement)
    comp_table = res_est.get("comparison_table", [])
    best_model = next((r for r in comp_table if "Model 1" in str(r.get("model",""))), None)
    if not best_model and comp_table:
        best_model = comp_table[0]
    ind = res_est.get("independent_analysis", {})

    if best_model:
        total_mt  = float(best_model.get("total_tonnage_kt", 0) or 0) / 1000
        grade     = float(best_model.get("total_grade_pct", 0) or 0)
        mi_mt     = float(best_model.get("mi_tonnage_kt", 0) or 0) / 1000
        mi_grade  = float(best_model.get("mi_grade_pct", 0) or 0)
        conv_pct  = float(ind.get("confidence_pct", 0) or 0)
        contained = float(best_model.get("total_contained_mlb", 0) or 0)
        _draw_sidebar(pdf, [
            {"label": "TOTAL RESOURCE", "value": f"{total_mt:.1f}", "unit": "Mt", "primary": True},
            {"label": "GRADE",          "value": _grade_fmt(grade, material), "unit": ""},
            {"label": "HIGH-GRADE M&I", "value": f"{mi_mt:.0f}", "unit": "Mt",
             "sub_label": f"@ {_grade_fmt(mi_grade, material)}" if mi_grade else ""},
            {"label": "CONTAINED",      "value": _num(contained, "{:,.1f}"), "unit": "Mlb/Moz"},
            {"label": "CONVICTION",     "value": f"{conv_pct:.0f}", "unit": "%"},
        ], content_start_y)

    # Main content (constrained to CONTENT_W to stay left of sidebar)
    assessment = exec_s.get("overall_assessment", "Cautious")
    badge_colors = {"Positive": (GREEN, GREEN_BG), "Cautious": (AMBER, AMBER_BG), "Negative": (RED_RISK, RED_BG)}
    badge_text_c, badge_bg_c = badge_colors.get(assessment, (NAVY, LIGHT_BG))

    pdf.set_xy(14, content_start_y)
    if assessment:
        pdf.set_fill_color(*badge_bg_c)
        pdf.set_text_color(*badge_text_c)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(55, 7, f"  Overall: {assessment}", fill=True, ln=False)
        pdf.set_text_color(*DARK_TEXT)
        pdf.ln()

    pdf.ln(2)
    key_takeaway = exec_s.get("key_takeaway", "")
    if key_takeaway:
        pdf.set_x(14)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*NAVY)
        pdf.multi_cell(CONTENT_W, 6, f'"{_s(key_takeaway)}"')
        pdf.set_text_color(*DARK_TEXT)

    pdf.ln(2)
    sum_text = exec_s.get("summary_text", "")
    if sum_text:
        _body_narrow(pdf, sum_text, CONTENT_W)


def _render_project_overview(pdf: MIPdf, report_json: Dict) -> None:
    overview = report_json.get("project_overview", {})
    meta     = report_json.get("metadata", {})
    pdf._section_name = "PROJECT OVERVIEW"
    _section_header(pdf, "2. Project Overview")

    # Quick facts table
    facts = [
        ("Project", meta.get("project_name", "")),
        ("Material", meta.get("material", "")),
        ("Deposit Type", meta.get("deposit_type", "")),
        ("Country", meta.get("country", "")),
        ("Stage", meta.get("project_stage", "")),
    ]
    pdf.set_font("Helvetica", "", 9)
    for label, val in facts:
        if val:
            _kv(pdf, label, val, label_w=44)

    pdf.ln(3)
    proj_sum = overview.get("project_summary", "")
    if proj_sum:
        _body(pdf, proj_sum)

    chars = overview.get("key_characteristics", [])
    if chars:
        _subsection(pdf, "Key Characteristics")
        for c in chars:
            _bullet(pdf, str(c))
        pdf.ln(2)

    mre = overview.get("official_mre_summary")
    if mre:
        _subsection(pdf, "Official MRE")
        _body(pdf, mre)

    drill = overview.get("drilling_data_summary")
    if drill:
        _subsection(pdf, "Drilling Data")
        _body(pdf, drill)


def _render_analogs(pdf: MIPdf, report_json: Dict) -> None:
    analogs = report_json.get("analogs_comparison") or []
    if not analogs:
        return
    pdf._section_name = "ANALOG COMPARISON"
    _section_header(pdf, "4. Analog Comparison")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 5, f"  {len(analogs)} comparable projects used to calibrate the resource model.", ln=True)
    pdf.set_text_color(*DARK_TEXT)
    pdf.ln(2)

    cols   = ["Project Name", "Country", "Deposit Type", "Tonnage (Mt)", "Grade", "Similarity"]
    widths = [52, 28, 36, 24, 20, 22]
    aligns = ["L", "L", "L", "R", "R", "C"]
    _table_header(pdf, cols, widths)

    for i, a in enumerate(analogs):
        score = float(a.get("similarity_score", 0))
        score_norm = score if score > 1 else score * 100
        sim_bg = GREEN_BG if score_norm >= 70 else (AMBER_BG if score_norm >= 40 else RED_BG)
        fill   = ALT_ROW if i % 2 == 0 else WHITE
        tonnage    = _num(a.get("tonnage_mt"), "{:,.1f}")
        grade_val  = a.get("grade_value")
        grade_unit = a.get("grade_unit", "%")
        grade_str  = f"{grade_val:.3f} {grade_unit}" if grade_val is not None else "N/A"
        _mc_row(pdf, [
            {"val": a.get("name",""),        "width": widths[0], "align": "L", "fill": fill},
            {"val": a.get("country",""),     "width": widths[1], "align": "L", "fill": fill},
            {"val": a.get("deposit_type",""),"width": widths[2], "align": "L", "fill": fill},
            {"val": tonnage,                 "width": widths[3], "align": "R", "fill": fill},
            {"val": grade_str,               "width": widths[4], "align": "R", "fill": fill},
            {"val": f"{score_norm:.0f}%",    "width": widths[5], "align": "C", "fill": sim_bg, "font_style": "B"},
        ])

    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 5, "  Similarity >=70% green, 40-70% amber, <40% red.", ln=True)
    pdf.set_text_color(*DARK_TEXT)


def _render_resource_models(pdf: MIPdf, report_json: Dict) -> None:
    res_est  = report_json.get("resource_estimates", {})
    material = report_json.get("metadata", {}).get("material", "")
    pdf._section_name = "RESOURCE MODELS"
    _section_header(pdf, "5. Resource Models")

    content_start_y = pdf.get_y()

    # Build sidebar
    comp_table = res_est.get("comparison_table", [])
    best_model = next((r for r in comp_table if "Model 1" in str(r.get("model",""))), None)
    if not best_model and comp_table:
        best_model = comp_table[0]
    ind = res_est.get("independent_analysis", {})
    upd = res_est.get("updated_analysis", {})

    if best_model:
        mi_kt     = float(best_model.get("mi_tonnage_kt", 0) or 0)
        mi_mt     = mi_kt / 1000
        mi_grade  = float(best_model.get("mi_grade_pct", 0) or 0)
        inf_kt    = float(best_model.get("inferred_tonnage_kt", 0) or 0)
        inf_mt    = inf_kt / 1000
        total_kt  = float(best_model.get("total_tonnage_kt", 0) or 0)
        total_mt  = total_kt / 1000
        contained = float(best_model.get("total_contained_mlb", 0) or 0)
        conv_pct  = float(ind.get("confidence_pct", 0) or 0)
        conv2_pct = float(upd.get("confidence_pct", 0) or conv_pct)
        _draw_sidebar(pdf, [
            {"label": "M&I TONNAGE",     "value": f"{mi_mt:.1f}",  "unit": "Mt", "primary": True},
            {"label": "M&I GRADE",       "value": _grade_fmt(mi_grade, material), "unit": ""},
            {"label": "INFERRED",        "value": f"{inf_mt:.1f}", "unit": "Mt"},
            {"label": "TOTAL RESOURCE",  "value": f"{total_mt:.1f}", "unit": "Mt"},
            {"label": "CONTAINED METAL", "value": _num(contained, "{:,.1f}"), "unit": "Mlb/Moz"},
            {"label": "CONVICTION",      "value": f"{conv_pct:.0f}", "unit": "%"},
        ], content_start_y)

    # Table 1: M&I + Total (split into two tables to fit CONTENT_W)
    if comp_table:
        _subsection(pdf, "Model Comparison — Tonnage & Grade")
        # Table 1: Model | M&I Ton | M&I Grade | Total Ton | Total Grade
        t1_cols   = ["Model", "M&I Ton (kt)", "M&I Grade", "Total (kt)", "Total Grade"]
        t1_widths = [42, 22, 22, 22, 22]   # sum = 130 ≈ CONTENT_W + 6 (slight)
        t1_aligns = ["L", "R", "R", "R", "R"]
        _table_header(pdf, t1_cols, t1_widths)

        for i, row in enumerate(comp_table):
            is_official = "Official" in str(row.get("model", ""))
            fill = (255, 250, 220) if is_official else (ALT_ROW if i % 2 == 0 else WHITE)
            fs   = "B" if is_official else ""
            _table_row(pdf, [
                row.get("model", ""),
                _num(row.get("mi_tonnage_kt"), "{:,.0f}"),
                _grade_fmt(row.get("mi_grade_pct", 0), material),
                _num(row.get("total_tonnage_kt"), "{:,.0f}"),
                _grade_fmt(row.get("total_grade_pct", 0), material),
            ], t1_widths, t1_aligns, fill_color=fill, font_style=fs)

        pdf.ln(3)

        # Table 2: Contained metal + conviction
        pdf.set_x(14)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(CONTENT_W, 5, "Contained metal (Mlb base metals / Moz precious) and model conviction:", ln=True)
        pdf.set_text_color(*DARK_TEXT)

        t2_cols   = ["Model", "Contained Metal", "Unit", "Conviction %", "Description"]
        t2_widths = [38, 24, 14, 20, 34]   # sum = 130
        t2_aligns = ["L", "R", "C", "C", "L"]
        _table_header(pdf, t2_cols, t2_widths)

        for i, row in enumerate(comp_table):
            fill = ALT_ROW if i % 2 == 0 else WHITE
            is_m1 = "Model 1" in str(row.get("model",""))
            is_m2 = "Model 2" in str(row.get("model",""))
            row_conv = (ind.get("confidence_pct") or 0) if is_m1 else ((upd.get("confidence_pct") or 0) if is_m2 else 0)
            _table_row(pdf, [
                row.get("model",""),
                _num(row.get("total_contained_mlb"), "{:,.2f}"),
                "Mlb/Moz",
                f"{row_conv:.0f}%" if row_conv else "N/A",
                row.get("description","")[:45],
            ], t2_widths, t2_aligns, fill_color=fill)

    pdf.ln(3)
    compliance = res_est.get("compliance_summary", [])
    if compliance:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*GRAY_TEXT)
        for line in compliance:
            pdf.set_x(pdf.l_margin)
            pdf.cell(4, 4, "", ln=False)
            pdf.multi_cell(CONTENT_W - 4, 4, _s(line))
            pdf.set_x(pdf.l_margin)
        pdf.set_text_color(*DARK_TEXT)


def _render_economic_assumptions(pdf: MIPdf, report_json: Dict) -> None:
    econ = report_json.get("economic_assumptions")
    if not econ:
        return
    pdf._section_name = "ECONOMIC ASSUMPTIONS"
    _section_header(pdf, "11. Economic Assumptions")

    fields = [
        ("CuEq Formula", econ.get("cueq_formula")),
        ("Cut-off Grade", econ.get("cutoff_grade")),
        ("Block Model Size", econ.get("block_model_size")),
        ("Cost per Tonne", econ.get("cost_per_tonne")),
    ]
    for label, val in fields:
        if val:
            _kv(pdf, label, val)

    prices = econ.get("metal_prices", {})
    if prices.get("primary_metal"):
        _kv(pdf, "Primary Metal", f"{prices['primary_metal']} @ {prices.get('primary_price','N/A')}")

    rec = econ.get("recoveries", {})
    if rec:
        _kv(pdf, "Primary Recovery", f"{rec.get('primary_pct','N/A')}%  —  {rec.get('notes','')}")

    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.multi_cell(0, 5,
        "Metal prices, recoveries, and cut-off grades are assumptions used for MI internal modeling only. "
        "These do not constitute a formal economic assessment.")
    pdf.set_text_color(*DARK_TEXT)


def _render_sensitivity(pdf: MIPdf, report_json: Dict) -> None:
    sens = report_json.get("sensitivity_analysis")
    if not sens:
        return
    pdf._section_name = "SENSITIVITY ANALYSIS"
    _section_header(pdf, "12. Sensitivity Analysis")

    # Cut-off grade table
    ct = sens.get("cutoff_table", [])
    if ct:
        _subsection(pdf, "Cut-Off Grade Sensitivity")
        cols   = ["Cut-Off", "Tonnage (kt)", "Grade", "Contained Metal"]
        widths = [30, 40, 30, 40]
        aligns = ["C", "R", "R", "R"]
        _table_header(pdf, cols, widths)
        for i, row in enumerate(ct):
            is_base = row.get("cut_off_label") == "Base"
            fill = (255, 250, 220) if is_base else (ALT_ROW if i % 2 == 0 else WHITE)
            fs = "B" if is_base else ""
            _table_row(pdf, [
                row.get("cut_off_label",""),
                _num(row.get("tonnage_kt"), "{:,.0f}"),
                f"{row.get('grade',0):.3f} {row.get('grade_unit','')}",
                f"{row.get('contained_metal',0):.2f} {row.get('metal_unit','')}",
            ], widths, aligns, fill_color=fill, font_style=fs)
        pdf.ln(3)

    # Price sensitivity table
    pt = sens.get("price_table", [])
    if pt:
        _subsection(pdf, "Metal Price Sensitivity")
        cols   = ["Price Change", "Tonnage (kt)", "Contained Metal"]
        widths = [40, 50, 50]
        aligns = ["C", "R", "R"]
        _table_header(pdf, cols, widths)
        for i, row in enumerate(pt):
            is_base = row.get("price_label") == "Base"
            delta = row.get("price_delta_pct", 0)
            fill = (255, 250, 220) if is_base else (GREEN_BG if delta > 0 else (RED_BG if delta < 0 else WHITE))
            fill = fill if fill != WHITE else (ALT_ROW if i % 2 == 0 else WHITE)
            fs = "B" if is_base else ""
            _table_row(pdf, [
                row.get("price_label",""),
                _num(row.get("tonnage_kt"), "{:,.0f}"),
                f"{row.get('contained_metal',0):.2f} {row.get('metal_unit','')}",
            ], widths, aligns, fill_color=fill, font_style=fs)
        pdf.ln(3)

    # Scenario table
    sc = sens.get("scenario_table", [])
    if sc:
        _subsection(pdf, "Combined Scenarios")
        cols   = ["Scenario", "Cut-Off", "Metal Price", "Recovery", "Tonnage (kt)", "Contained Metal"]
        widths = [28, 20, 24, 20, 28, 42]
        aligns = ["L", "C", "C", "C", "R", "R"]
        _table_header(pdf, cols, widths)
        scenario_fills = {"Best Case": GREEN_BG, "Base Case": (255,250,220), "Worst Case": RED_BG}
        for row in sc:
            fill = scenario_fills.get(row.get("scenario",""), WHITE)
            fs = "B" if "Base" in str(row.get("scenario","")) else ""
            _table_row(pdf, [
                row.get("scenario",""),
                row.get("cut_off",""),
                row.get("metal_price",""),
                row.get("recovery",""),
                _num(row.get("tonnage_kt"), "{:,.0f}"),
                f"{row.get('contained_metal',0):.2f}",
            ], widths, aligns, fill_color=fill, font_style=fs)


def _render_risk_matrix(pdf: MIPdf, report_json: Dict) -> None:
    risks = report_json.get("risk_matrix") or []
    if not risks:
        return
    pdf._section_name = "RISK MATRIX"
    _section_header(pdf, "13. Risk Matrix")

    cols   = ["Risk Factor", "Probability", "Impact", "Mitigation Strategy"]
    widths = [46, 30, 22, 84]
    aligns = ["L", "C", "C", "L"]
    _table_header(pdf, cols, widths)

    def _label_color(txt: str):
        if "High" in txt:    return RED_RISK
        if "Moderate" in txt: return AMBER
        return GREEN

    for risk in risks:
        prob   = str(risk.get("probability", ""))
        impact = str(risk.get("impact", ""))
        fill   = _risk_color(impact, prob)
        _mc_row(pdf, [
            {"val": risk.get("risk_factor",""),  "width": widths[0], "align": "L", "fill": fill, "font_style": "B", "font_size": 8},
            {"val": prob,                        "width": widths[1], "align": "C", "fill": fill, "font_style": "B", "font_size": 7, "text_color": _label_color(prob)},
            {"val": impact,                      "width": widths[2], "align": "C", "fill": fill, "font_style": "B", "font_size": 7, "text_color": _label_color(impact)},
            {"val": risk.get("mitigation",""),   "width": widths[3], "align": "L", "fill": fill, "font_style": "",  "font_size": 8},
        ])

    pdf.set_text_color(*DARK_TEXT)
    pdf.ln(2)
    # Legend
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 5, "  Legend:  Red = High probability + High impact  |  Amber = Moderate  |  Green = Low", ln=True)
    pdf.set_text_color(*DARK_TEXT)


def _render_exploration_strategy(pdf: MIPdf, report_json: Dict) -> None:
    strategy = report_json.get("exploration_strategy") or []
    if not strategy:
        return
    pdf._section_name = "EXPLORATION STRATEGY"
    _section_header(pdf, "14. Exploration Strategy & Timeline")

    cols   = ["Activity", "Cost Estimate", "Timeline", "Priority", "Expected Outcome"]
    widths = [44, 26, 22, 18, 72]
    aligns = ["L", "R", "C", "C", "L"]
    _table_header(pdf, cols, widths)

    priority_fills = {"High": RED_BG, "Medium": AMBER_BG, "Low": GREEN_BG}
    for i, phase in enumerate(strategy):
        fill = ALT_ROW if i % 2 == 0 else WHITE
        prio = str(phase.get("priority", "Medium"))
        _mc_row(pdf, [
            {"val": phase.get("activity",""),         "width": widths[0], "align": "L", "fill": fill},
            {"val": phase.get("cost_estimate",""),    "width": widths[1], "align": "R", "fill": fill},
            {"val": phase.get("timeline",""),         "width": widths[2], "align": "C", "fill": fill},
            {"val": prio,                             "width": widths[3], "align": "C", "fill": priority_fills.get(prio, AMBER_BG), "font_style": "B", "font_size": 7},
            {"val": phase.get("expected_outcome",""), "width": widths[4], "align": "L", "fill": fill},
        ])

    pdf.set_text_color(*DARK_TEXT)


def _render_recommendations(pdf: MIPdf, report_json: Dict) -> None:
    recs = report_json.get("actionable_recommendations") or []
    if not recs:
        return
    pdf._section_name = "RECOMMENDATIONS"
    _section_header(pdf, "15. Actionable Recommendations")

    priority_colors = {"High": (RED_RISK, RED_BG), "Medium": (AMBER, AMBER_BG), "Low": (GREEN, GREEN_BG)}
    for i, rec in enumerate(recs):
        if not isinstance(rec, dict):
            continue
        priority = str(rec.get("priority","Medium"))
        text_c, bg_c = priority_colors.get(priority, (NAVY, LIGHT_BG))

        # Priority badge + recommendation
        pdf.set_x(pdf.l_margin)
        pdf.set_fill_color(*bg_c)
        pdf.set_text_color(*text_c)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(20, 6, f"[{priority}]", fill=True, ln=False, align="C")
        pdf.set_text_color(*DARK_TEXT)
        pdf.set_fill_color(*LIGHT_BG)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, f"  {_s(rec.get('recommendation',''))}", fill=True, ln=True)

        if rec.get("rationale"):
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.cell(6, 5, "", ln=False)
            pdf.multi_cell(0, 5, _s(rec["rationale"]))
            pdf.set_x(pdf.l_margin)
            pdf.set_text_color(*DARK_TEXT)

        details = []
        if rec.get("estimated_cost"):
            details.append(f"Cost: {rec['estimated_cost']}")
        if rec.get("timeline"):
            details.append(f"Timeline: {rec['timeline']}")
        if details:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.cell(6, 4, "", ln=False)
            pdf.cell(0, 4, "  " + "  |  ".join(details), ln=True)
            pdf.set_text_color(*DARK_TEXT)
        pdf.ln(2)


def _render_strengths_uncertainties(pdf: MIPdf, report_json: Dict) -> None:
    uands = report_json.get("key_uncertainties_and_strengths", {})
    lessons = report_json.get("lessons_summary", {})
    strengths     = uands.get("strengths", [])
    uncertainties = uands.get("uncertainties", [])
    if not strengths and not uncertainties:
        return
    pdf._section_name = "STRENGTHS & UNCERTAINTIES"
    _section_header(pdf, "16. Strengths & Uncertainties")

    if strengths:
        _subsection(pdf, "Strengths")
        for s in strengths:
            pdf.set_x(pdf.l_margin)
            pdf.set_text_color(*GREEN)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(6, 5, "+", ln=False)
            pdf.set_text_color(*DARK_TEXT)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _s(s))
            pdf.set_x(pdf.l_margin)
        pdf.ln(2)

    if uncertainties:
        _subsection(pdf, "Uncertainties & Risks")
        for u in uncertainties:
            pdf.set_x(pdf.l_margin)
            pdf.set_text_color(*AMBER)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(6, 5, "!", ln=False)
            pdf.set_text_color(*DARK_TEXT)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _s(u))
            pdf.set_x(pdf.l_margin)
        pdf.ln(2)

    if lessons:
        _kv(pdf, "Rules Applied", f"{lessons.get('total_lessons_applied',0)} total, {lessons.get('high_confidence_lessons',0)} high-confidence")


def _render_acquisition_analysis(pdf: MIPdf, report_json: Dict) -> None:
    acq = report_json.get("acquisition_analysis")
    if not acq:
        return
    pdf._section_name = "ACQUISITION ANALYSIS"
    _section_header(pdf, "17. Strategic Acquisition Analysis")

    tiers = [
        ("Junior / Emerging Producer", acq.get("junior", {})),
        ("Mid-Tier Producer", acq.get("mid_tier", {})),
        ("Major Producer (Top-Tier)", acq.get("major", {})),
    ]
    status_styles = {
        "green": (GREEN, GREEN_BG),
        "amber": (AMBER, AMBER_BG),
        "red":   (RED_RISK, RED_BG),
    }
    for tier_name, tier_data in tiers:
        if not tier_data:
            continue
        verdict = tier_data.get("verdict","")
        summary = tier_data.get("score_summary","")

        # Tier header
        verdict_colors = {
            "Well-suited":       (GREEN,     GREEN_BG),
            "Potentially suitable": (AMBER,  AMBER_BG),
            "Not suitable":      (RED_RISK,  RED_BG),
        }
        v_text_c, v_bg_c = verdict_colors.get(verdict, (NAVY, LIGHT_BG))
        pdf.set_fill_color(*LIGHT_BG)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(100, 7, f"  {tier_name}", fill=True, ln=False)
        pdf.set_fill_color(*v_bg_c)
        pdf.set_text_color(*v_text_c)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(82, 7, f"  {verdict}", fill=True, ln=True)
        pdf.set_text_color(*DARK_TEXT)

        if summary:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.cell(6, 5, "", ln=False)
            pdf.multi_cell(0, 5, _s(summary))
            pdf.set_x(pdf.l_margin)
            pdf.set_text_color(*DARK_TEXT)

        items = tier_data.get("items", [])
        if items:
            cols   = ["Criterion", "Status", "Comment"]
            widths = [60, 22, 100]
            aligns = ["L", "C", "L"]
            _table_header(pdf, cols, widths)
            for item in items:
                status = str(item.get("status", "amber")).lower()
                text_c, bg_c = status_styles.get(status, (NAVY, LIGHT_BG))
                _mc_row(pdf, [
                    {"val": item.get("criterion",""), "width": widths[0], "align": "L", "fill": ALT_ROW},
                    {"val": status.upper(),           "width": widths[1], "align": "C", "fill": bg_c, "font_style": "B", "text_color": text_c},
                    {"val": item.get("comment",""),   "width": widths[2], "align": "L", "fill": ALT_ROW},
                ])
        pdf.ln(4)


def _render_key_terms(pdf: MIPdf, report_json: Dict) -> None:
    terms = report_json.get("key_terms") or []
    if not terms:
        return
    pdf._section_name = "KEY TERMS"
    _section_header(pdf, "19. Key Terms Glossary")

    for i, item in enumerate(terms):
        pdf.set_x(pdf.l_margin)
        fill = ALT_ROW if i % 2 == 0 else WHITE
        pdf.set_fill_color(*fill)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(55, 6, _s(item.get("term","")), fill=True, ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK_TEXT)
        pdf.multi_cell(0, 6, _s(item.get("definition","")), fill=True)
        pdf.set_x(pdf.l_margin)
    pdf.ln(2)


def _render_geological_framework(pdf: MIPdf, report_json: Dict) -> None:
    geo = report_json.get("geological_framework")
    if not geo:
        return
    pdf._section_name = "GEOLOGICAL FRAMEWORK"
    _section_header(pdf, "3. Geological Framework")
    _body(pdf, geo.get("regional_setting", ""))
    for label, key in [
        ("Deposit Characteristics", "deposit_characteristics"),
        ("Mineralisation Description", "mineralization_description"),
        ("Structural Complexity", "structural_complexity"),
        ("Geological Continuity", "geological_continuity"),
        ("Logistics & Infrastructure", "logistics_and_infrastructure"),
    ]:
        if geo.get(key):
            _subsection(pdf, label)
            _body(pdf, geo[key])

    zones = geo.get("mineral_zones", [])
    if zones:
        _subsection(pdf, "Mineral Zones")
        cols   = ["Zone Name", "Description", "Grade Range"]
        widths = [40, 100, 42]
        aligns = ["L", "L", "C"]
        _table_header(pdf, cols, widths)
        for i, z in enumerate(zones):
            fill = ALT_ROW if i % 2 == 0 else WHITE
            _table_row(pdf, [z.get("zone_name",""), z.get("description",""), z.get("grade_range","")],
                       widths, aligns, fill_color=fill)


def _render_drilling_and_sampling(pdf: MIPdf, report_json: Dict) -> None:
    ds = report_json.get("drilling_and_sampling")
    if not ds:
        return
    pdf._section_name = "DRILLING & SAMPLING"
    _section_header(pdf, "6. Drilling & Sampling Data")
    if ds.get("total_holes_estimated"):
        _kv(pdf, "Estimated Drilling", ds["total_holes_estimated"])
    if ds.get("drillhole_strategy"):
        _body(pdf, ds["drillhole_strategy"])
    for label, key in [
        ("Assay QA/QC Protocol", "assay_qa_qc"),
        ("XRF & Geochemical Notes", "xrf_geochemical_notes"),
        ("Cost Efficiency", "cost_efficiency_notes"),
    ]:
        if ds.get(key):
            _subsection(pdf, label)
            _body(pdf, ds[key])
    if ds.get("data_quality_assessment"):
        _subsection(pdf, "Data Quality Assessment")
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.multi_cell(0, 5, _s(ds["data_quality_assessment"]))
        pdf.set_x(pdf.l_margin)
        pdf.set_text_color(*DARK_TEXT)
        pdf.ln(2)


def _render_drilling_efficiency_metrics(pdf: MIPdf, report_json: Dict) -> None:
    dm = report_json.get("drilling_efficiency_metrics")
    if not dm:
        return
    pdf._section_name = "DRILLING EFFICIENCY"
    _section_header(pdf, "7. Drilling Efficiency Metrics")
    if dm.get("narrative"):
        _body(pdf, dm["narrative"])
    rows = dm.get("metrics_table", [])
    if rows:
        cols   = ["Metric", "Project Value", "Peer Range", "Assessment"]
        widths = [58, 40, 40, 24]
        aligns = ["L", "L", "L", "C"]
        _table_header(pdf, cols, widths)
        assessment_fills = {"Above Peer": GREEN_BG, "In-Line": AMBER_BG, "Below Peer": RED_BG}
        for i, row in enumerate(rows):
            fill = ALT_ROW if i % 2 == 0 else WHITE
            assessment = str(row.get("assessment", "In-Line"))
            _mc_row(pdf, [
                {"val": row.get("metric",""),        "width": widths[0], "align": "L", "fill": fill},
                {"val": row.get("project_value",""), "width": widths[1], "align": "L", "fill": fill},
                {"val": row.get("peer_range",""),    "width": widths[2], "align": "L", "fill": fill},
                {"val": assessment,                  "width": widths[3], "align": "C", "fill": assessment_fills.get(assessment, AMBER_BG), "font_style": "B", "font_size": 7},
            ])
        pdf.ln(2)
    for label, key in [
        ("Shareholder Dilution Efficiency", "shareholder_dilution_efficiency"),
        ("Cost per Meter vs. Peers", "cost_per_meter_vs_peers"),
    ]:
        if dm.get(key):
            _subsection(pdf, label)
            _body(pdf, dm[key])


def _render_geophysical_integration(pdf: MIPdf, report_json: Dict) -> None:
    gp = report_json.get("geophysical_integration")
    if not gp:
        return
    pdf._section_name = "GEOPHYSICAL INTEGRATION"
    _section_header(pdf, "8. Geophysical Integration")
    surveys = gp.get("survey_types_recommended", [])
    if surveys:
        cols   = ["Survey Type", "Rationale", "Priority"]
        widths = [44, 108, 20]
        aligns = ["L", "L", "C"]
        _table_header(pdf, cols, widths)
        priority_fills = {"High": RED_BG, "Medium": AMBER_BG, "Low": GREEN_BG}
        for i, s in enumerate(surveys):
            fill = ALT_ROW if i % 2 == 0 else WHITE
            prio = str(s.get("priority", "Medium"))
            _mc_row(pdf, [
                {"val": s.get("survey_type",""), "width": widths[0], "align": "L", "fill": fill},
                {"val": s.get("rationale",""),   "width": widths[1], "align": "L", "fill": fill},
                {"val": prio,                    "width": widths[2], "align": "C", "fill": priority_fills.get(prio, AMBER_BG), "font_style": "B", "font_size": 7},
            ])
        pdf.ln(2)
    for label, key in [
        ("Continuity Thresholds", "continuity_thresholds"),
        ("Validation Triggers", "validation_triggers"),
        ("Existing Data Notes", "existing_data_notes"),
    ]:
        if gp.get(key):
            _subsection(pdf, label)
            _body(pdf, gp[key])


def _render_geostatistical_modeling(pdf: MIPdf, report_json: Dict) -> None:
    gm = report_json.get("geostatistical_modeling")
    if not gm:
        return
    pdf._section_name = "GEOSTATISTICAL MODELING"
    _section_header(pdf, "9. Geostatistical Modeling")
    if gm.get("variography_narrative"):
        _body(pdf, gm["variography_narrative"])
    if gm.get("estimation_method"):
        _kv(pdf, "Estimation Method", gm["estimation_method"])
        pdf.ln(2)
    params = gm.get("variogram_parameters", [])
    if params:
        _subsection(pdf, "Variogram Parameters by Zone")
        cols   = ["Zone", "Nugget", "Sill", "Range Major (m)", "Range Minor (m)", "Anisotropy"]
        widths = [38, 20, 20, 30, 30, 24]
        aligns = ["L", "C", "C", "C", "C", "C"]
        _table_header(pdf, cols, widths)
        for i, p in enumerate(params):
            fill = ALT_ROW if i % 2 == 0 else WHITE
            _table_row(pdf, [
                p.get("zone",""), p.get("nugget",""), p.get("sill",""),
                p.get("range_major_m",""), p.get("range_minor_m",""), p.get("anisotropy_ratio",""),
            ], widths, aligns, fill_color=fill)
        pdf.ln(2)
    for label, key in [
        ("Grade Capping Method", "grade_capping_method"),
        ("Extension Ranges", "extension_ranges"),
        ("By-Product Modeling", "byproduct_modeling"),
    ]:
        if gm.get(key):
            _subsection(pdf, label)
            _body(pdf, gm[key])


def _render_validation_and_qc(pdf: MIPdf, report_json: Dict) -> None:
    vq = report_json.get("validation_and_qc")
    if not vq:
        return
    pdf._section_name = "VALIDATION & QC"
    _section_header(pdf, "10. Validation & Quality Control")
    if vq.get("check_assay_protocol"):
        _subsection(pdf, "Check Assay Protocol")
        _body(pdf, vq["check_assay_protocol"])
    if vq.get("monte_carlo_summary"):
        _subsection(pdf, "Monte Carlo Simulation")
        _body(pdf, vq["monte_carlo_summary"])

    p10t = vq.get("p10_tonnage_kt", 0)
    p90t = vq.get("p90_tonnage_kt", 0)
    p10g = vq.get("p10_grade", 0)
    p90g = vq.get("p90_grade", 0)
    if any([p10t, p90t, p10g, p90g]):
        _metric_row(pdf, [
            ("P10 Tonnage", _num(p10t, "{:,.0f}"), "kt"),
            ("P90 Tonnage", _num(p90t, "{:,.0f}"), "kt"),
            (f"P10 Grade", f"{p10g:.4f}", ""),
            (f"P90 Grade", f"{p90g:.4f}", ""),
        ])
        pdf.ln(2)

    for label, key, italic in [
        ("Statistical Reconciliation", "statistical_reconciliation", False),
        ("Audit Trail Notes", "audit_trail_notes", True),
    ]:
        if vq.get(key):
            _subsection(pdf, label)
            if italic:
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(*GRAY_TEXT)
                pdf.multi_cell(0, 5, _s(vq[key]))
                pdf.set_x(pdf.l_margin)
                pdf.set_text_color(*DARK_TEXT)
                pdf.ln(2)
            else:
                _body(pdf, vq[key])


def _render_conclusion(pdf: MIPdf, report_json: Dict) -> None:
    con = report_json.get("conclusion")
    if not con:
        return
    pdf._section_name = "CONCLUSION"
    _section_header(pdf, "18. Conclusion")
    headline = con.get("headline_finding", "")
    if headline:
        # Gold-bordered callout box
        pdf.set_fill_color(*LIGHT_BG)
        pdf.set_draw_color(*GOLD)
        pdf.set_line_width(0.8)
        box_y = pdf.get_y()
        pdf.rect(14, box_y, 182, 14, style="FD")
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_xy(18, box_y + 3)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.multi_cell(174, 5, _s(headline))
        pdf.set_text_color(*DARK_TEXT)
        pdf.ln(4)
    if con.get("conclusion_text"):
        _body(pdf, con["conclusion_text"])
    if con.get("next_milestone"):
        _kv(pdf, "Next Milestone", con["next_milestone"])
    readiness = con.get("investment_readiness", "")
    if readiness:
        readiness_colors = {
            "Development-ready": (GREEN, GREEN_BG),
            "Resource-stage":    (AMBER, AMBER_BG),
            "Pre-resource":      (RED_RISK, RED_BG),
        }
        text_c, bg_c = readiness_colors.get(readiness, (NAVY, LIGHT_BG))
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(52, 5, "Investment Readiness:", ln=False)
        pdf.set_fill_color(*bg_c)
        pdf.set_text_color(*text_c)
        pdf.cell(50, 5, f"  {_s(readiness)}  ", fill=True, ln=True)
        pdf.set_text_color(*DARK_TEXT)
        pdf.ln(2)


def _render_appendices(pdf: MIPdf, report_json: Dict) -> None:
    app = report_json.get("appendices")
    if not app:
        return
    pdf._section_name = "APPENDICES"
    _section_header(pdf, "20. Appendices")

    # Appendix A: Analog Input Weighting
    rows_a = app.get("input_weighting_table", [])
    if rows_a:
        _subsection(pdf, "Appendix A: Analog Input Weighting")
        cols   = ["Analog", "Weight (%)", "Key Rationale"]
        widths = [60, 22, 100]
        aligns = ["L", "C", "L"]
        _table_header(pdf, cols, widths)
        for i, r in enumerate(rows_a):
            fill = ALT_ROW if i % 2 == 0 else WHITE
            _table_row(pdf, [r.get("analog_name",""), r.get("weight_pct",""), r.get("key_rationale","")],
                       widths, aligns, fill_color=fill)
        pdf.ln(3)

    # Appendix B: Variogram Parameters
    rows_b = app.get("variogram_parameters_table", [])
    if rows_b:
        _subsection(pdf, "Appendix B: Variogram Parameters")
        cols   = ["Zone", "Nugget", "Sill", "Range Major (m)", "Range Minor (m)"]
        widths = [46, 24, 24, 30, 30]
        aligns = ["L", "C", "C", "C", "C"]
        _table_header(pdf, cols, widths)
        for i, r in enumerate(rows_b):
            fill = ALT_ROW if i % 2 == 0 else WHITE
            _table_row(pdf, [r.get("zone",""), r.get("nugget",""), r.get("sill",""),
                             r.get("range_major_m",""), r.get("range_minor_m","")],
                       widths, aligns, fill_color=fill)
        pdf.ln(3)

    # Appendix C: Drilling Summary
    rows_c = app.get("drilling_summary_table", [])
    if rows_c:
        _subsection(pdf, "Appendix C: Drilling Summary")
        cols   = ["Hole Type", "Count", "Avg Depth (m)", "Purpose"]
        widths = [36, 22, 34, 90]
        aligns = ["L", "C", "C", "L"]
        _table_header(pdf, cols, widths)
        for i, r in enumerate(rows_c):
            fill = ALT_ROW if i % 2 == 0 else WHITE
            _table_row(pdf, [r.get("hole_type",""), r.get("count",""), r.get("avg_depth_m",""), r.get("purpose","")],
                       widths, aligns, fill_color=fill)
        pdf.ln(3)

    # Appendix D: References
    refs = app.get("references", [])
    if refs:
        _subsection(pdf, "Appendix D: References")
        for ref in refs:
            _bullet(pdf, str(ref))
        pdf.ln(2)


def _render_disclaimer_page(pdf: MIPdf) -> None:
    pdf._section_name = "DISCLAIMER"
    _section_header(pdf, "Disclaimer & Important Notices")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*DARK_TEXT)
    notices = [
        ("Not Regulatory Compliant",
         "This report was not prepared or reviewed by a Qualified Person (QP) as defined under NI 43-101 or JORC. "
         "It does not constitute a formal mineral resource estimate compliant with any regulatory standard."),
        ("Not Investment Advice",
         "This report is for informational and planning purposes only. It does not constitute investment advice "
         "and should not be relied upon for investment decisions."),
        ("Preliminary Estimates",
         "All resource estimates are preliminary and internal to Mining Intellect. Estimates are subject to change "
         "with the receipt of new data, revised interpretations, or updated metal prices."),
        ("Analog-Based Methodology",
         "The modeling methodology uses comparable projects (analogs) and statistical methods to estimate "
         "potential resources. This approach has inherent limitations for early-stage projects."),
        ("Independent Verification Required",
         "For any regulatory, financing, or public reporting purposes, an independent Qualified Person review "
         "is required. Mining Intellect estimates should be treated as supplementary information only."),
    ]
    for title, text in notices:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 6, title, ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK_TEXT)
        pdf.multi_cell(0, 5, text)
        pdf.ln(3)


# ── Main entry point ────────────────────────────────────────────────────────────

def generate_pdf(report_json: Dict, project_name: str) -> bytes:
    """
    Generate a professional PDF from a MiningReport JSON dict.
    Returns PDF file bytes.
    """
    meta         = report_json.get("metadata", {})
    material     = str(meta.get("material", ""))
    generated_at = str(meta.get("generated_at", ""))[:10]

    pdf = MIPdf(project_name=project_name, material=material, generated_date=generated_at)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(14, 16, 14)

    # ── Page 1: Cover (Design B — TOC embedded on cover) ───────────────────────
    pdf.add_page()
    _cover_page(pdf, report_json, project_name)

    # ── Content pages ───────────────────────────────────────────────────────────
    pdf.add_page()
    _render_executive_summary(pdf, report_json)
    pdf.add_page()
    _render_project_overview(pdf, report_json)

    if report_json.get("geological_framework"):
        pdf.add_page()
        _render_geological_framework(pdf, report_json)

    if report_json.get("analogs_comparison"):
        pdf.add_page()
        _render_analogs(pdf, report_json)

    pdf.add_page()
    _render_resource_models(pdf, report_json)

    if report_json.get("drilling_and_sampling"):
        pdf.add_page()
        _render_drilling_and_sampling(pdf, report_json)

    if report_json.get("drilling_efficiency_metrics"):
        _render_drilling_efficiency_metrics(pdf, report_json)

    if report_json.get("geophysical_integration"):
        pdf.add_page()
        _render_geophysical_integration(pdf, report_json)

    if report_json.get("geostatistical_modeling"):
        pdf.add_page()
        _render_geostatistical_modeling(pdf, report_json)

    if report_json.get("validation_and_qc"):
        pdf.add_page()
        _render_validation_and_qc(pdf, report_json)

    if report_json.get("economic_assumptions"):
        _render_economic_assumptions(pdf, report_json)

    if report_json.get("sensitivity_analysis"):
        pdf.add_page()
        _render_sensitivity(pdf, report_json)

    if report_json.get("risk_matrix"):
        pdf.add_page()
        _render_risk_matrix(pdf, report_json)

    if report_json.get("exploration_strategy"):
        _render_exploration_strategy(pdf, report_json)

    pdf.add_page()
    _render_recommendations(pdf, report_json)
    _render_strengths_uncertainties(pdf, report_json)

    if report_json.get("acquisition_analysis"):
        pdf.add_page()
        _render_acquisition_analysis(pdf, report_json)

    if report_json.get("conclusion"):
        pdf.add_page()
        _render_conclusion(pdf, report_json)

    if report_json.get("key_terms"):
        pdf.add_page()
        _render_key_terms(pdf, report_json)

    if report_json.get("appendices"):
        pdf.add_page()
        _render_appendices(pdf, report_json)

    pdf.add_page()
    _render_disclaimer_page(pdf)

    pages = pdf.page
    logger.info(f"[PDF] Generated {pages} page(s) for '{project_name}'")
    return bytes(pdf.output())
