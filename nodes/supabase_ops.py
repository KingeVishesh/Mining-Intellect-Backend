"""
Supabase Operations Layer
All database reads and writes go through this module.
Uses supabase-py (REST API) — no SQLAlchemy, no SQLite.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from supabase import create_client, Client
from config import settings

logger = logging.getLogger(__name__)

_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    return _client


# ── Projects ──────────────────────────────────────────────────────────────────

def get_project(project_id: str) -> Optional[Dict]:
    """Fetch a single project by ID. Returns None if not found."""
    res = get_client().table("projects").select("*").eq("id", project_id).maybe_single().execute()
    return res.data


def upsert_project(data: Dict) -> Dict:
    """Insert or update a project row. `data` must include `id`."""
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    res = get_client().table("projects").upsert(data).execute()
    return res.data[0] if res.data else {}


def search_projects_by_criteria(
    material: str,
    deposit_type: Optional[str] = None,
    min_tonnage: Optional[float] = None,
    max_tonnage: Optional[float] = None,
    min_grade: Optional[float] = None,
    max_grade: Optional[float] = None,
    limit: int = 20,
) -> List[Dict]:
    """Find projects matching analog criteria."""
    q = get_client().table("projects").select("*").eq("material", material)
    if deposit_type:
        q = q.eq("deposit_type", deposit_type)
    if min_tonnage is not None:
        q = q.gte("tonnage_mt", min_tonnage)
    if max_tonnage is not None:
        q = q.lte("tonnage_mt", max_tonnage)
    if min_grade is not None:
        q = q.gte("grade_value", min_grade)
    if max_grade is not None:
        q = q.lte("grade_value", max_grade)
    res = q.limit(limit).execute()
    return res.data or []


# ── Reports ───────────────────────────────────────────────────────────────────

def get_report(project_id: str) -> Optional[Dict]:
    """Get the most recent report for a project."""
    res = (
        get_client()
        .table("reports")
        .select("*")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def upload_pdf(project_id: str, report_id: str, pdf_bytes: bytes) -> str:
    """
    Upload a PDF to Supabase Storage bucket 'reports'.
    Returns the public URL.
    """
    bucket = "reports"
    path = f"{project_id}/{report_id}.pdf"
    client = get_client()

    client.storage.from_(bucket).upload(
        path,
        pdf_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    url = client.storage.from_(bucket).get_public_url(path)
    logger.info(f"[Storage] PDF uploaded: {url}")
    return url


def save_report(
    project_id: str,
    report_json: Dict,
    meta: Dict,
    report_id: Optional[str] = None,
    pdf_url: Optional[str] = None,
) -> str:
    """
    Save a report to the `reports` table.
    Returns the report id.
    """
    if not report_id:
        report_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Extract top-level conviction/model numbers for indexed columns
    resource = report_json.get("resource_estimates", {})
    table = resource.get("comparison_table", [])
    m1 = next((r for r in table if "Independent" in r.get("model", "")), {})
    m2 = next((r for r in table if "Updated" in r.get("model", "")), {})

    row = {
        "id": report_id,
        "project_id": project_id,
        "report_type": meta.get("report_type", "full"),
        "material": meta.get("material"),
        "deposit_type": meta.get("deposit_type"),
        "status": "published",
        "content_json": report_json,
        "file_path": pdf_url,
        "model_1_tonnage": m1.get("total_tonnage_kt"),
        "model_1_grade": m1.get("total_grade_pct"),
        "model_1_conviction": resource.get("independent_analysis", {}).get("confidence_pct"),
        "model_2_tonnage": m2.get("total_tonnage_kt"),
        "model_2_grade": m2.get("total_grade_pct"),
        "model_2_conviction": resource.get("updated_analysis", {}).get("confidence_pct"),
        "created_at": now,
        "updated_at": now,
    }
    get_client().table("reports").insert(row).execute()
    logger.info(f"[DB] Report saved: {report_id} for project {project_id}")
    return report_id


# ── Analogs ───────────────────────────────────────────────────────────────────

def save_analogs(project_id: str, analogs: List[Dict]) -> None:
    """
    Persist approved analogs into `workflow_states` as a JSON blob
    (no dedicated analogs table in the schema — store in analogs_json column).
    """
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": str(uuid4()),
        "project_id": project_id,
        "phase": "analog_finder",
        "phase_number": 2,
        "status": "complete",
        "analogs_json": analogs,
        "analogs_count": len(analogs),
        "created_at": now,
    }
    get_client().table("workflow_states").insert(row).execute()
    logger.info(f"[DB] {len(analogs)} analogs saved for project {project_id}")


def get_analogs(project_id: str) -> List[Dict]:
    """Load the most recently saved analogs for a project."""
    res = (
        get_client()
        .table("workflow_states")
        .select("analogs_json")
        .eq("project_id", project_id)
        .eq("phase", "analog_finder")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if res.data:
        return res.data[0].get("analogs_json") or []
    return []


# ── Compiled Rules ─────────────────────────────────────────────────────────────

def get_compiled_rules(material: str) -> List[Dict]:
    """Load compiled rules for a given material."""
    res = (
        get_client()
        .table("compiled_rules")
        .select("*")
        .eq("source_material", material)
        .execute()
    )
    return res.data or []


# ── Workflow State ─────────────────────────────────────────────────────────────

def log_workflow_state(
    project_id: str,
    phase: str,
    phase_number: int,
    status: str,
    state_json: Optional[Dict] = None,
    error_message: Optional[str] = None,
) -> None:
    row = {
        "id": str(uuid4()),
        "project_id": project_id,
        "phase": phase,
        "phase_number": phase_number,
        "status": status,
        "state_json": state_json,
        "error_message": error_message,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    get_client().table("workflow_states").insert(row).execute()
