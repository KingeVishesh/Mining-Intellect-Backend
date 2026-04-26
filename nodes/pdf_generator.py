"""
PDF Generator — builds a PDF from a MiningReport JSON dict.
Uses fpdf2 (pure Python, no browser/HTML rendering required).
Returns raw PDF bytes for upload to storage.
"""
from __future__ import annotations
import logging
from typing import Dict, List

from fpdf import FPDF

logger = logging.getLogger(__name__)


def _safe(value) -> str:
    """Coerce to string and replace non-latin-1 characters."""
    if value is None:
        return "N/A"
    return str(value).encode("latin-1", errors="replace").decode("latin-1")


class _ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 7, "Mining Intellect  |  Confidential Resource Assessment", align="C")
        self.ln(3)
        self.set_draw_color(200, 200, 200)
        self.line(14, self.get_y(), 196, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 8,
            f"Page {self.page_no()}  |  Not NI 43-101 or JORC compliant  |  For internal use only",
            align="C",
        )
        self.set_text_color(0, 0, 0)


def _section(pdf: _ReportPDF, title: str) -> None:
    pdf.ln(4)
    pdf.set_fill_color(235, 240, 245)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, _safe(title), ln=True, fill=True)
    pdf.ln(2)


def _kv(pdf: _ReportPDF, label: str, value) -> None:
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(52, 5, _safe(label) + ":", ln=False)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _safe(value))


def _body(pdf: _ReportPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _safe(text))
    pdf.ln(2)


def _bullet(pdf: _ReportPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(6, 5, "-", ln=False)
    pdf.multi_cell(0, 5, _safe(text))


def generate_pdf(report_json: Dict, project_name: str) -> bytes:
    """
    Generate a PDF from a MiningReport JSON dict.
    Returns PDF file bytes.
    """
    pdf = _ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(14, 16, 14)
    pdf.add_page()

    meta          = report_json.get("metadata", {})
    exec_sum      = report_json.get("executive_summary", {})
    proj_overview = report_json.get("project_overview", {})
    resource_est  = report_json.get("resource_estimates", {})
    recommendations = report_json.get("actionable_recommendations", [])
    lessons       = report_json.get("lessons_summary", {})
    uands         = report_json.get("key_uncertainties_and_strengths", {})

    generated_date = str(meta.get("generated_at", ""))[:10]

    # ── Title block ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, _safe(project_name), ln=True)
    pdf.set_font("Helvetica", "", 10)
    subtitle = f"{_safe(meta.get('material', ''))}  |  {_safe(meta.get('report_type', 'Full Assessment')).title()}  |  {generated_date}"
    pdf.cell(0, 6, subtitle, ln=True)
    pdf.ln(4)
    pdf.set_draw_color(50, 100, 160)
    pdf.set_line_width(0.8)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.set_draw_color(0, 0, 0)
    pdf.ln(5)

    # ── Executive Summary ─────────────────────────────────────────────────────
    _section(pdf, "Executive Summary")
    if exec_sum.get("overall_assessment"):
        _kv(pdf, "Overall Assessment", exec_sum["overall_assessment"])
    if exec_sum.get("key_takeaway"):
        _kv(pdf, "Key Takeaway", exec_sum["key_takeaway"])
    if exec_sum.get("summary_text"):
        pdf.ln(2)
        _body(pdf, exec_sum["summary_text"])

    # ── Project Overview ──────────────────────────────────────────────────────
    _section(pdf, "Project Overview")
    proj_sum = proj_overview.get("project_summary", {})
    if isinstance(proj_sum, dict):
        for field in ("deposit_type", "stage", "location", "material"):
            if proj_sum.get(field):
                _kv(pdf, field.replace("_", " ").title(), proj_sum[field])
    elif isinstance(proj_sum, str) and proj_sum.strip():
        _body(pdf, proj_sum)

    chars = proj_overview.get("key_characteristics", [])
    if chars:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "Key Characteristics:", ln=True)
        for c in chars:
            if isinstance(c, dict):
                line = f"{c.get('category', '')}: {c.get('value', '')}"
                if c.get("details"):
                    line += f"  ({c['details']})"
                _bullet(pdf, line)
            else:
                _bullet(pdf, str(c))

    # ── Resource Estimates ────────────────────────────────────────────────────
    _section(pdf, "Resource Estimates")
    comparison_table: List[Dict] = resource_est.get("comparison_table", [])
    if comparison_table:
        col_w = [56, 26, 22, 26, 22, 22]
        headers = ["Model", "M&I Tonnage (kt)", "M&I Grade (%)", "Inf Tonnage (kt)", "Inf Grade (%)", "Total (kt)"]

        # Header row
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(40, 80, 140)
        pdf.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        # Data rows
        pdf.set_font("Helvetica", "", 8)
        for idx, row in enumerate(comparison_table):
            is_official = "Official" in str(row.get("model", ""))
            if is_official:
                pdf.set_fill_color(255, 250, 220)
            elif idx % 2 == 0:
                pdf.set_fill_color(248, 250, 252)
            else:
                pdf.set_fill_color(255, 255, 255)

            vals = [
                row.get("model", ""),
                f"{row.get('mi_tonnage_kt', 0):,.0f}",
                f"{row.get('mi_grade_pct', 0):.3f}",
                f"{row.get('inferred_tonnage_kt', 0):,.0f}",
                f"{row.get('inferred_grade_pct', 0):.3f}",
                f"{row.get('total_tonnage_kt', 0):,.0f}",
            ]
            for i, v in enumerate(vals):
                pdf.cell(col_w[i], 6, _safe(v), border=1, fill=True, align="L" if i == 0 else "R")
            pdf.ln()
        pdf.ln(3)

    ind = resource_est.get("independent_analysis", {})
    upd = resource_est.get("updated_analysis", {})
    if ind.get("confidence_pct") is not None:
        _kv(pdf, "Model 1 Conviction", f"{ind['confidence_pct']:.1f}%")
    if upd.get("confidence_pct") is not None and upd.get("confidence_pct", 0) > 0:
        _kv(pdf, "Model 2 Conviction", f"{upd['confidence_pct']:.1f}%")

    compliance = resource_est.get("compliance_summary", [])
    if compliance:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(120, 120, 120)
        for line in compliance:
            pdf.multi_cell(0, 4, _safe(line))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    # ── Recommendations ───────────────────────────────────────────────────────
    if recommendations:
        _section(pdf, "Actionable Recommendations")
        for rec in recommendations:
            if not isinstance(rec, dict):
                continue
            priority = rec.get("priority", "Medium")
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, _safe(f"[{priority}]  {rec.get('recommendation', '')}"), ln=True)
            if rec.get("rationale"):
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(80, 80, 80)
                pdf.multi_cell(0, 4, _safe(rec["rationale"]))
                pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

    # ── Lessons Summary ───────────────────────────────────────────────────────
    if lessons:
        _section(pdf, "Rules & Lessons Applied")
        _kv(pdf, "Total Rules Applied", str(lessons.get("total_lessons_applied", 0)))
        _kv(pdf, "High-Confidence Rules", str(lessons.get("high_confidence_lessons", 0)))

    # ── Uncertainties & Strengths ─────────────────────────────────────────────
    strengths = uands.get("strengths", [])
    uncertainties = uands.get("uncertainties", [])
    if strengths or uncertainties:
        _section(pdf, "Strengths & Uncertainties")
        if strengths:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, "Strengths:", ln=True)
            for s in strengths:
                _bullet(pdf, str(s))
            pdf.ln(2)
        if uncertainties:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, "Uncertainties:", ln=True)
            for u in uncertainties:
                _bullet(pdf, str(u))

    logger.info(f"[PDF] Generated {pdf.page} page(s) for '{project_name}'")
    return bytes(pdf.output())
