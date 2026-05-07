"""
PDF Generator — renders a MiningReport JSON dict into a styled PDF.

Pipeline: report_json (dict) → Jinja2 (HTML) → WeasyPrint (PDF bytes).

Templates live in nodes/templates/. Edit the HTML/CSS there to change layout.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML, CSS

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_CSS_DIR      = _TEMPLATE_DIR / "css"
_CSS_FILES    = ["base.css", "components.css", "print.css"]


# ── Jinja2 filters ────────────────────────────────────────────────────────────

def _filter_num(value: Any, fmt: str = "{:,.0f}", default: str = "N/A") -> str:
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return default


def _filter_grade(grade_pct: Any, material: str = "") -> str:
    """Format a grade value. Adds ppm equivalent when grade < 2% for base metals."""
    try:
        g = float(grade_pct)
    except (TypeError, ValueError):
        return "N/A"
    mat = (material or "").lower()
    precious = any(m in mat for m in ("gold", "au", "silver", "ag", "platinum", "palladium", "pgm"))
    if precious:
        return f"{g:.2f} g/t"
    if 0 < g < 2.0:
        ppm = int(round(g * 10_000))
        return f"{g:.3f}% ({ppm:,} ppm)"
    return f"{g:.3f}%"


def _filter_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _filter_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Jinja env (cached) ────────────────────────────────────────────────────────

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)
_env.filters["num"]   = _filter_num
_env.filters["grade"] = _filter_grade
_env.filters["pct"]   = _filter_pct
_env.filters["float"] = _filter_float


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf(report_json: Dict, project_name: str) -> bytes:
    """
    Render a MiningReport JSON dict to PDF bytes.

    Layout is defined entirely in nodes/templates/ — edit the Jinja templates
    or CSS there to change visuals. No coordinate math here.
    """
    template = _env.get_template("report.html")
    html_str = template.render(report=report_json, project_name=project_name or "Unknown Project")

    stylesheets = [CSS(filename=str(_CSS_DIR / f)) for f in _CSS_FILES]
    pdf_bytes = HTML(string=html_str, base_url=str(_TEMPLATE_DIR)).write_pdf(stylesheets=stylesheets)

    logger.info(f"[PDF] Generated PDF for '{project_name}' ({len(pdf_bytes)} bytes)")
    return pdf_bytes
