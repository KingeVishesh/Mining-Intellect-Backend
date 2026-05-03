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
        self._proj = _s(project_name)
        self._mat  = _s(material)
        self._date = _s(generated_date)

    def header(self):
        if self.page_no() == 1:
            return  # cover page has its own design
        # Running header
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*NAVY)
        self.cell(130, 6, self._proj, ln=False)
        self.set_text_color(*GRAY_TEXT)
        self.set_font("Helvetica", "", 7)
        self.cell(0, 6, f"Mining Intellect Resource Report  |  {self._date}", align="R", ln=True)
        self.set_draw_color(*GOLD)
        self.set_line_width(0.5)
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
    """Navy section header bar with white text."""
    pdf.ln(4)
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 9, f"  {_s(title)}", ln=True, fill=True)
    pdf.set_text_color(*DARK_TEXT)
    pdf.ln(2)


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
    if fill_color:
        pdf.set_fill_color(*fill_color)
    pdf.set_font("Helvetica", font_style, 8)
    for v, w, a in zip(vals, widths, aligns):
        pdf.cell(w, 6, _s(v), border=1, fill=bool(fill_color), align=a)
    pdf.ln()


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
    meta   = report_json.get("metadata", {})
    exec_s = report_json.get("executive_summary", {})
    material     = _s(meta.get("material", ""))
    deposit_type = _s(meta.get("deposit_type", ""))
    country      = _s(meta.get("country", ""))
    stage        = _s(meta.get("project_stage", ""))
    date_str     = _s(str(meta.get("generated_at", ""))[:10])
    assessment   = exec_s.get("overall_assessment", "Cautious")

    # Full navy background (top third)
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 90, style="F")

    # MI branding in white
    pdf.set_xy(14, 16)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*GOLD)
    pdf.cell(0, 7, "MINING INTELLECT", ln=True)
    pdf.set_xy(14, 23)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 5, "Resource Modeling & Analytics Platform", ln=True)

    # Report label
    pdf.set_xy(14, 36)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GOLD)
    pdf.cell(0, 6, "RESOURCE MODELING REPORT", ln=True)

    # Gold separator line
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(1.0)
    pdf.line(14, 43, 140, 43)
    pdf.set_line_width(0.2)

    # Project name
    pdf.set_xy(14, 47)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*WHITE)
    # Long names need to wrap
    pdf.multi_cell(175, 10, _s(project_name))

    # Meta tags
    meta_line = "  |  ".join(filter(None, [material, deposit_type, country, stage]))
    pdf.set_xy(14, 76)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(200, 215, 235)
    pdf.cell(0, 6, meta_line, ln=True)

    # White content area
    pdf.set_fill_color(*WHITE)
    pdf.rect(0, 90, 210, 207, style="F")

    # Date and assessment badge
    pdf.set_xy(14, 96)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(80, 6, f"Generated: {date_str}", ln=False)

    # Assessment badge
    badge_colors = {
        "Positive":  (GREEN, GREEN_BG),
        "Cautious":  (AMBER, AMBER_BG),
        "Negative":  (RED_RISK, RED_BG),
    }
    badge_text_c, badge_bg_c = badge_colors.get(assessment, (NAVY, LIGHT_BG))
    pdf.set_fill_color(*badge_bg_c)
    pdf.set_text_color(*badge_text_c)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 6, f"  Assessment: {assessment}  ", fill=True, ln=True, align="C")
    pdf.set_text_color(*DARK_TEXT)

    # Gold separator under date
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.5)
    pdf.line(14, 104, 196, 104)
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)

    # Executive key takeaway
    key_takeaway = exec_s.get("key_takeaway", "")
    if key_takeaway:
        pdf.set_xy(14, 108)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*NAVY)
        pdf.multi_cell(182, 6, f'"{_s(key_takeaway)}"')

    # Metric boxes — pull from resource estimates
    comp_table = (report_json.get("resource_estimates") or {}).get("comparison_table", [])
    best_model = next((r for r in comp_table if "Model 1" in str(r.get("model",""))), None)
    if not best_model and comp_table:
        best_model = comp_table[0]

    pdf.set_xy(14, 122)
    metrics = []
    if best_model:
        total_kt = best_model.get("total_tonnage_kt", 0)
        grade    = best_model.get("total_grade_pct", 0)
        metal    = best_model.get("total_contained_mlb", 0)
        metal_label = "Contained (Mlb)" if material.lower() not in {"gold","silver","platinum","palladium"} else "Contained (Moz)"
        metrics = [
            ("Total Tonnage", _num(total_kt), "kt"),
            ("Grade", f"{grade:.3f}", material[:2].upper() + "%"),
            (metal_label, _num(metal, "{:,.1f}"), "Mlb/Moz"),
        ]
        ind = (report_json.get("resource_estimates") or {}).get("independent_analysis", {})
        conv = ind.get("confidence_pct", 0)
        metrics.append(("MI Conviction", f"{conv:.0f}%", "confidence"))

    if metrics:
        _metric_row(pdf, metrics)

    # Summary text excerpt
    sum_text = exec_s.get("summary_text", "")
    if sum_text:
        excerpt = sum_text[:400] + ("..." if len(sum_text) > 400 else "")
        pdf.set_xy(14, pdf.get_y() + 2)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK_TEXT)
        pdf.multi_cell(182, 5, _s(excerpt))

    # Disclaimer box at bottom of cover
    pdf.set_xy(14, 258)
    pdf.set_fill_color(255, 248, 220)
    pdf.set_draw_color(*AMBER)
    pdf.set_line_width(0.4)
    pdf.rect(14, 258, 182, 20, style="FD")
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
    _section_header(pdf, "Table of Contents")

    sections = [
        ("1",  "Executive Summary"),
        ("2",  "Project Overview"),
        ("3",  "Geological Framework"),
        ("4",  "Analog Comparison"),
        ("5",  "Resource Models"),
        ("6",  "Drilling & Sampling Data"),
        ("7",  "Drilling Efficiency Metrics"),
        ("8",  "Geophysical Integration"),
        ("9",  "Geostatistical Modeling"),
        ("10", "Validation & Quality Control"),
        ("11", "Economic Assumptions"),
        ("12", "Sensitivity Analysis"),
        ("13", "Risk Matrix"),
        ("14", "Exploration Strategy"),
        ("15", "Actionable Recommendations"),
        ("16", "Strengths & Uncertainties"),
        ("17", "Acquisition Analysis"),
        ("18", "Conclusion"),
        ("19", "Key Terms Glossary"),
        ("20", "Appendices"),
    ]
    pdf.set_font("Helvetica", "", 10)
    for num, title in sections:
        pdf.set_text_color(*NAVY)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(12, 7, num + ".", ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*DARK_TEXT)
        pdf.cell(0, 7, title, ln=True)
        # Dotted separator
        pdf.set_draw_color(200, 200, 200)
        pdf.set_line_width(0.2)
        pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.set_text_color(*DARK_TEXT)


# ── Section renderers ───────────────────────────────────────────────────────────

def _render_executive_summary(pdf: MIPdf, report_json: Dict) -> None:
    exec_s = report_json.get("executive_summary", {})
    _section_header(pdf, "1. Executive Summary")

    assessment = exec_s.get("overall_assessment", "Cautious")
    badge_colors = {"Positive": (GREEN, GREEN_BG), "Cautious": (AMBER, AMBER_BG), "Negative": (RED_RISK, RED_BG)}
    badge_text_c, badge_bg_c = badge_colors.get(assessment, (NAVY, LIGHT_BG))

    # Assessment + key takeaway side by side
    if assessment:
        pdf.set_fill_color(*badge_bg_c)
        pdf.set_text_color(*badge_text_c)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(60, 7, f"  Overall: {assessment}", fill=True, ln=False)
        pdf.set_text_color(*DARK_TEXT)
        pdf.cell(5, 7, "", ln=False)

    key_takeaway = exec_s.get("key_takeaway", "")
    if key_takeaway:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*NAVY)
        pdf.multi_cell(0, 7, _s(key_takeaway))
    else:
        pdf.ln()

    pdf.ln(2)
    sum_text = exec_s.get("summary_text", "")
    if sum_text:
        _body(pdf, sum_text)


def _render_project_overview(pdf: MIPdf, report_json: Dict) -> None:
    overview = report_json.get("project_overview", {})
    meta     = report_json.get("metadata", {})
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
        if score > 1:
            score_norm = score  # already 0-100
        else:
            score_norm = score * 100

        # Colour-code similarity
        if score_norm >= 70:
            sim_bg = GREEN_BG
        elif score_norm >= 40:
            sim_bg = AMBER_BG
        else:
            sim_bg = RED_BG

        fill = ALT_ROW if i % 2 == 0 else WHITE
        tonnage = _num(a.get("tonnage_mt"), "{:,.1f}")
        grade_val = a.get("grade_value")
        grade_unit = a.get("grade_unit", "%")
        grade_str = f"{grade_val:.3f} {grade_unit}" if grade_val is not None else "N/A"

        # Draw each cell; similarity cell gets special colour
        pdf.set_fill_color(*fill)
        pdf.set_font("Helvetica", "", 8)
        row_vals  = [a.get("name",""), a.get("country",""), a.get("deposit_type",""), tonnage, grade_str]
        row_widths = widths[:-1]
        row_aligns = aligns[:-1]
        for v, w, al in zip(row_vals, row_widths, row_aligns):
            pdf.cell(w, 6, _s(v), border=1, fill=True, align=al)

        # Similarity cell
        pdf.set_fill_color(*sim_bg)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(widths[-1], 6, f"{score_norm:.0f}%", border=1, fill=True, align="C")
        pdf.ln()

    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 5, "  Similarity >=70% green, 40-70% amber, <40% red.", ln=True)
    pdf.set_text_color(*DARK_TEXT)


def _render_resource_models(pdf: MIPdf, report_json: Dict) -> None:
    res_est = report_json.get("resource_estimates", {})
    _section_header(pdf, "5. Resource Models")

    comp_table = res_est.get("comparison_table", [])
    if comp_table:
        cols   = ["Model", "M&I Tonnage (kt)", "M&I Grade", "Inf Tonnage (kt)", "Inf Grade", "Total (kt)", "Total Grade"]
        widths = [44, 24, 18, 24, 18, 24, 30]
        aligns = ["L", "R", "R", "R", "R", "R", "R"]
        _table_header(pdf, cols, widths)

        for i, row in enumerate(comp_table):
            is_official = "Official" in str(row.get("model", ""))
            is_m2       = "Model 2" in str(row.get("model", ""))
            fill = (255, 250, 220) if is_official else (ALT_ROW if i % 2 == 0 else WHITE)
            font_style = "B" if is_official else ""

            vals = [
                row.get("model", ""),
                _num(row.get("mi_tonnage_kt"), "{:,.0f}"),
                f"{row.get('mi_grade_pct', 0):.3f}%",
                _num(row.get("inferred_tonnage_kt"), "{:,.0f}"),
                f"{row.get('inferred_grade_pct', 0):.3f}%",
                _num(row.get("total_tonnage_kt"), "{:,.0f}"),
                f"{row.get('total_grade_pct', 0):.3f}%",
            ]
            _table_row(pdf, vals, widths, aligns, fill_color=fill, font_style=font_style)

        pdf.ln(2)
        # Contained metal row
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(0, 5, "  Contained metal (Mlb for base metals, Moz for precious):", ln=True)
        pdf.set_text_color(*DARK_TEXT)

        cont_cols   = ["Model", "Total Contained Metal", "Unit", "Description"]
        cont_widths = [44, 36, 20, 82]
        cont_aligns = ["L", "R", "C", "L"]
        _table_header(pdf, cont_cols, cont_widths)
        for i, row in enumerate(comp_table):
            fill = ALT_ROW if i % 2 == 0 else WHITE
            _table_row(pdf, [
                row.get("model",""),
                _num(row.get("total_contained_mlb"), "{:,.2f}"),
                "Mlb/Moz",
                row.get("description",""),
            ], cont_widths, cont_aligns, fill_color=fill)

    pdf.ln(3)
    ind = res_est.get("independent_analysis", {})
    upd = res_est.get("updated_analysis", {})
    if ind.get("confidence_pct") is not None:
        _kv(pdf, "Model 1 Conviction", f"{ind['confidence_pct']:.1f}%  —  key analogs: {', '.join(ind.get('key_factors',[])[:3])}")
    if upd.get("confidence_pct"):
        _kv(pdf, "Model 2 Conviction", f"{upd['confidence_pct']:.1f}%")

    compliance = res_est.get("compliance_summary", [])
    if compliance:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*GRAY_TEXT)
        for line in compliance:
            pdf.set_x(pdf.l_margin)
            pdf.cell(4, 4, "", ln=False)
            pdf.multi_cell(0, 4, _s(line))
            pdf.set_x(pdf.l_margin)
        pdf.set_text_color(*DARK_TEXT)


def _render_economic_assumptions(pdf: MIPdf, report_json: Dict) -> None:
    econ = report_json.get("economic_assumptions")
    if not econ:
        return
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
    _section_header(pdf, "13. Risk Matrix")

    cols   = ["Risk Factor", "Probability", "Impact", "Mitigation Strategy"]
    widths = [46, 30, 22, 84]
    aligns = ["L", "C", "C", "L"]
    _table_header(pdf, cols, widths)

    for risk in risks:
        prob   = str(risk.get("probability", ""))
        impact = str(risk.get("impact", ""))
        fill   = _risk_color(impact, prob)
        pdf.set_fill_color(*fill)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(widths[0], 6, _s(risk.get("risk_factor","")), border=1, fill=True, align="L")
        # Probability with color text
        if "High" in prob:
            pdf.set_text_color(*RED_RISK)
        elif "Moderate" in prob:
            pdf.set_text_color(*AMBER)
        else:
            pdf.set_text_color(*GREEN)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(widths[1], 6, _s(prob), border=1, fill=True, align="C")
        # Impact
        if "High" in impact:
            pdf.set_text_color(*RED_RISK)
        elif "Moderate" in impact:
            pdf.set_text_color(*AMBER)
        else:
            pdf.set_text_color(*GREEN)
        pdf.cell(widths[2], 6, _s(impact), border=1, fill=True, align="C")
        pdf.set_text_color(*DARK_TEXT)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(widths[3], 6, _s(risk.get("mitigation","")), border=1, fill=True, align="L")
        pdf.ln()

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
    _section_header(pdf, "14. Exploration Strategy & Timeline")

    cols   = ["Activity", "Cost Estimate", "Timeline", "Priority", "Expected Outcome"]
    widths = [44, 26, 22, 18, 72]
    aligns = ["L", "R", "C", "C", "L"]
    _table_header(pdf, cols, widths)

    priority_fills = {"High": RED_BG, "Medium": AMBER_BG, "Low": GREEN_BG}
    for i, phase in enumerate(strategy):
        fill = ALT_ROW if i % 2 == 0 else WHITE
        prio = str(phase.get("priority","Medium"))
        pdf.set_fill_color(*fill)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(widths[0], 6, _s(phase.get("activity","")), border=1, fill=True, align="L")
        pdf.cell(widths[1], 6, _s(phase.get("cost_estimate","")), border=1, fill=True, align="R")
        pdf.cell(widths[2], 6, _s(phase.get("timeline","")), border=1, fill=True, align="C")
        # Priority cell
        pdf.set_fill_color(*priority_fills.get(prio, AMBER_BG))
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(widths[3], 6, _s(prio), border=1, fill=True, align="C")
        pdf.set_fill_color(*fill)
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(widths[4], 6, _s(phase.get("expected_outcome","")), border=1, fill=True, align="L")
        pdf.ln()

    pdf.set_text_color(*DARK_TEXT)


def _render_recommendations(pdf: MIPdf, report_json: Dict) -> None:
    recs = report_json.get("actionable_recommendations") or []
    if not recs:
        return
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
                status = str(item.get("status","amber")).lower()
                text_c, bg_c = status_styles.get(status, (NAVY, LIGHT_BG))
                pdf.set_fill_color(*ALT_ROW)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(widths[0], 6, _s(item.get("criterion","")), border=1, fill=True, align="L")
                pdf.set_fill_color(*bg_c)
                pdf.set_text_color(*text_c)
                pdf.set_font("Helvetica", "B", 8)
                pdf.cell(widths[1], 6, _s(status.upper()), border=1, fill=True, align="C")
                pdf.set_text_color(*DARK_TEXT)
                pdf.set_fill_color(*ALT_ROW)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(widths[2], 6, _s(item.get("comment","")), border=1, fill=True, align="L")
                pdf.ln()
        pdf.ln(4)


def _render_key_terms(pdf: MIPdf, report_json: Dict) -> None:
    terms = report_json.get("key_terms") or []
    if not terms:
        return
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
            pdf.set_fill_color(*fill)
            pdf.set_font("Helvetica", "", 8)
            for v, w, a in zip(
                [row.get("metric",""), row.get("project_value",""), row.get("peer_range","")],
                widths[:-1], aligns[:-1]
            ):
                pdf.cell(w, 6, _s(v), border=1, fill=True, align=a)
            assessment = str(row.get("assessment", "In-Line"))
            pdf.set_fill_color(*assessment_fills.get(assessment, AMBER_BG))
            pdf.set_font("Helvetica", "B", 7)
            pdf.cell(widths[-1], 6, _s(assessment), border=1, fill=True, align="C")
            pdf.ln()
        pdf.set_text_color(*DARK_TEXT)
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
            pdf.set_fill_color(*fill)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(widths[0], 6, _s(s.get("survey_type","")), border=1, fill=True, align="L")
            pdf.cell(widths[1], 6, _s(s.get("rationale","")), border=1, fill=True, align="L")
            prio = str(s.get("priority","Medium"))
            pdf.set_fill_color(*priority_fills.get(prio, AMBER_BG))
            pdf.set_font("Helvetica", "B", 7)
            pdf.cell(widths[2], 6, _s(prio), border=1, fill=True, align="C")
            pdf.ln()
        pdf.set_text_color(*DARK_TEXT)
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

    # ── Page 1: Cover ───────────────────────────────────────────────────────────
    pdf.add_page()
    _cover_page(pdf, report_json, project_name)

    # ── Page 2: Table of Contents ───────────────────────────────────────────────
    pdf.add_page()
    _toc_page(pdf, report_json)

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
