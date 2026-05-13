"""
Supabase Operations Layer
All database reads and writes go through this module.
Uses supabase-py (REST API) — no SQLAlchemy, no SQLite.
"""
from __future__ import annotations
import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
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


def get_approved_analogs(
    material: str,
    deposit_type: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    """Query report_analogs for previously approved analogs of this commodity.

    When deposit_type is None (unknown target), the library is skipped entirely.
    Library analogs were approved for specific projects with known deposit types —
    returning them without a deposit_type filter would mix incompatible families
    (e.g. nickel laterite vs magmatic sulphide) and poison the candidate pool.

    Returns candidates in the standard analog dict format (name, material, deposit_type,
    tonnage_mt, grade_value, grade_unit, country, source_url, source='library').
    """
    if not deposit_type:
        return []

    keys = _MATERIAL_TO_RULES_KEYS.get(material.strip().lower(), [material.strip().lower()])
    dep = deposit_type.strip().lower()
    # report_analogs stores the raw material string — try all mapped keys
    q = (
        get_client()
        .table("report_analogs")
        .select(
            "analog_name,analog_material,analog_deposit_type,analog_country,"
            "analog_tonnage_mt,analog_grade_value,analog_grade_unit,source_url,"
            "similarity_score,analog_project_stage,"
            "analog_host_rock,analog_mineralization_style,analog_district,"
            # Geological profile (cascading match)
            "analog_deposit_subtype,analog_mineralization_mode,analog_tectonic_belt,"
            "analog_metal_suite,analog_alteration_signature,analog_recovery_method,"
            # Pattern + host class
            "analog_mineralization_pattern,analog_host_rock_class,"
            # Stage / mining / category / vintage / compliance
            "analog_project_stage_class,analog_mining_method_class,"
            "analog_resource_category_class,analog_resource_compliance_standard,"
            "analog_resource_vintage_year"
        )
        .in_("analog_material", keys)
        .eq("status", "approved")
    )
    if deposit_type:
        # ilike for case-insensitive partial match — Supabase REST supports this
        q = q.ilike("analog_deposit_type", f"%{dep}%")

    res = q.limit(limit).execute()
    rows = res.data or []

    candidates = []
    seen_names: set = set()
    for r in rows:
        name = r.get("analog_name") or ""
        if not name:
            continue
        norm = name.lower().strip()
        if norm in seen_names:
            continue
        seen_names.add(norm)
        candidates.append({
            "name": name,
            "material": r.get("analog_material") or material,
            "deposit_type": r.get("analog_deposit_type"),
            "host_rock": r.get("analog_host_rock"),
            "mineralization_style": r.get("analog_mineralization_style"),
            "district": r.get("analog_district"),
            "country": r.get("analog_country"),
            "tonnage_mt": r.get("analog_tonnage_mt"),
            "grade_value": r.get("analog_grade_value"),
            "grade_unit": r.get("analog_grade_unit"),
            "source_url": r.get("source_url"),
            "project_stage": r.get("analog_project_stage"),
            # Geological profile (used by cascading match)
            "deposit_subtype":        r.get("analog_deposit_subtype"),
            "mineralization_mode":    r.get("analog_mineralization_mode"),
            "tectonic_belt":          r.get("analog_tectonic_belt"),
            "metal_suite":            r.get("analog_metal_suite"),
            "alteration_signature":   r.get("analog_alteration_signature"),
            "recovery_method":        r.get("analog_recovery_method"),
            "mineralization_pattern": r.get("analog_mineralization_pattern"),
            "host_rock_class":        r.get("analog_host_rock_class"),
            "project_stage_class":         r.get("analog_project_stage_class"),
            "mining_method_class":         r.get("analog_mining_method_class"),
            "resource_category_class":     r.get("analog_resource_category_class"),
            "resource_compliance_standard": r.get("analog_resource_compliance_standard"),
            "resource_vintage_year":       r.get("analog_resource_vintage_year"),
            "source": "library",
        })
    return candidates


# ── Companies ─────────────────────────────────────────────────────────────────

_LEGAL_SUFFIXES = re.compile(
    r'\b(Ltd\.?|Limited|Inc\.?|Corp\.?|Corporation|LLC|L\.L\.C\.?|'
    r'Plc\.?|PLC|NL|AG|SA|S\.A\.|BV|B\.V\.|Co\.?|Company|Group|Holdings?)\b',
    re.IGNORECASE,
)


def _canonical(name: str) -> str:
    """Strip legal suffixes and normalize whitespace for fuzzy comparison."""
    cleaned = _LEGAL_SUFFIXES.sub("", name)
    return " ".join(cleaned.split()).lower()


def _fetch_all_companies() -> List[Dict]:
    """Fetch all company rows, paginating past the 1000-row default limit."""
    rows, offset = [], 0
    while True:
        res = get_client().table("companies").select("id, name").range(offset, offset + 999).execute()
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return rows


def find_company_by_name(name: str) -> Optional[Dict]:
    """Find a company by exact canonical match or fuzzy name similarity (>=0.85)."""
    normalized = _canonical(name)
    for row in _fetch_all_companies():
        row_canonical = _canonical(row["name"])
        if row_canonical == normalized:
            return row
        if SequenceMatcher(None, row_canonical, normalized).ratio() >= 0.85:
            return row
    return None


def upsert_company(name: str) -> str:
    """Find an existing company by name or create a new one. Returns company_id (UUID string)."""
    existing = find_company_by_name(name)
    if existing:
        logger.debug(f"[Company] Matched '{name}' → existing id={existing['id']}")
        return existing["id"]
    now = datetime.now(timezone.utc).isoformat()
    try:
        res = get_client().table("companies").insert({
            "name": name.strip(),
            "created_at": now,
            "updated_at": now,
        }).execute()
        new_id = res.data[0]["id"]
        logger.info(f"[Company] Created '{name.strip()}' id={new_id}")
        return new_id
    except Exception:
        # Unique constraint race: another process just created this company — re-fetch
        refetch = find_company_by_name(name)
        if refetch:
            return refetch["id"]
        raise


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
    bucket = "Project Reports"
    path = f"{project_id}/{report_id}.pdf"
    client = get_client()

    client.storage.from_(bucket).upload(
        path,
        pdf_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    # Return the storage path — frontend generates signed URLs on demand
    logger.info(f"[Storage] PDF uploaded: {path}")
    return path


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

def save_analog_audit_events(
    project_id: str,
    events: List[Dict],
    report_id: Optional[str] = None,
) -> None:
    """
    Persist per-candidate audit decisions emitted by graphs/analog_finder.py.

    Each event is a structured record of why an analog candidate was passed
    or dropped, including the rule_id, the lesson IDs that drove the decision,
    the detected geological profile, and a human-readable reason. Used for
    LangSmith traces, the admin "Audit Trail" tab, and future ML feedback
    loops on rule quality.
    """
    if not events:
        return
    rows = []
    for e in events:
        rows.append({
            "project_id":      project_id,
            "report_id":       report_id,
            "candidate_name":  e.get("candidate_name") or "Unknown",
            "candidate_source": e.get("candidate_source"),
            "decision":        e.get("decision") or "UNKNOWN",
            "level":           e.get("level"),
            "rule_id":         e.get("rule_id"),
            "lessons":         e.get("lessons") or [],
            "detected_profile": e.get("detected_profile") or {},
            "reason":          e.get("reason"),
            "rank_pts":        e.get("rank_pts"),
            "similarity_score": e.get("similarity_score"),
        })
    # Insert in batches to stay under PostgREST size limits
    BATCH = 50
    for i in range(0, len(rows), BATCH):
        get_client().table("analog_audit_events").insert(rows[i : i + BATCH]).execute()


def save_analogs(project_id: str, analogs: List[Dict]) -> None:
    """
    Persist approved analogs into `workflow_states` as a JSON blob
    (no dedicated analogs table in the schema — store in analogs_json column).
    """
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": str(uuid4()),
        "project_id": project_id,
        "phase": "analogs_found",
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
        .eq("phase", "analogs_found")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if res.data:
        return res.data[0].get("analogs_json") or []
    return []


# ── Compiled Rules ─────────────────────────────────────────────────────────────

# Gold and Silver each load their own rules PLUS the shared gold_silver bucket.
# This way gold-specific (Carlin, orogenic) and silver-specific (CRD, manto) rules
# are kept separate, while shared epithermal/porphyry rules apply to both.
_MATERIAL_TO_RULES_KEYS: Dict[str, List[str]] = {
    "gold":      ["gold", "gold_silver"],
    "silver":    ["silver", "gold_silver"],
    "copper":    ["copper"],
    "nickel":    ["nickel"],
    "uranium":   ["uranium"],
    "pgm":       ["pgm"],
    "platinum":  ["pgm"],
    "palladium": ["pgm"],
    "iron":      ["iron"],
    "iron ore":  ["iron"],
}


def get_compiled_rules(material: str, rule_type: Optional[str] = None) -> List[Dict]:
    """Load compiled rules for a given material, normalising to DB keys.

    Args:
        material: project material (e.g. 'Gold', 'Copper')
        rule_type: optional filter — 'analog_selection', 'model_adjustment',
                   'confidence_adjustment', 'data_quality'
    """
    keys = _MATERIAL_TO_RULES_KEYS.get(material.strip().lower(), [material.strip().lower()])
    query = (
        get_client()
        .table("compiled_rules")
        .select("*")
        .in_("source_material", keys)
    )
    # Only filter by active if the column exists (added in schema migration)
    try:
        query = query.eq("active", True)
    except Exception:
        pass
    if rule_type:
        query = query.eq("rule_type", rule_type)
    res = query.execute()
    return res.data or []


# ── Geocoding ─────────────────────────────────────────────────────────────────

def save_coords(
    project_id: str,
    latitude: float,
    longitude: float,
    method: str,
    source_url: Optional[str],
    existing_data_sources: Optional[Dict],
) -> bool:
    """
    UPDATE projects SET latitude, longitude, data_sources, updated_at.
    Merges coord_source metadata into existing data_sources JSON.
    Does NOT touch enrichment_status.
    """
    data_sources = dict(existing_data_sources) if isinstance(existing_data_sources, dict) else {}
    data_sources["coord_source"] = {
        "method": method,
        "source_url": source_url,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        get_client().table("projects").update({
            "latitude": latitude,
            "longitude": longitude,
            "data_sources": data_sources,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", project_id).execute()
        logger.info(f"[DB] Coords saved for {project_id}: lat={latitude}, lng={longitude}, method={method}")
        return True
    except Exception as e:
        logger.error(f"[DB] save_coords failed for {project_id}: {e}")
        return False


# ── Pipeline Orchestrator ──────────────────────────────────────────────────────

def save_pipeline_state(
    project_id: str,
    orchestrator_stage: str,
    orchestrator_thread_id: Optional[str] = None,
    research_thread_id: Optional[str] = None,
    analogs_thread_id: Optional[str] = None,
    report_thread_id: Optional[str] = None,
) -> None:
    """Upsert pipeline orchestrator tracking row in workflow_states."""
    state_update: Dict = {"stage": orchestrator_stage}
    if orchestrator_thread_id:
        state_update["orchestrator_thread_id"] = orchestrator_thread_id
    if research_thread_id:
        state_update["research_thread_id"] = research_thread_id
    if analogs_thread_id:
        state_update["analogs_thread_id"] = analogs_thread_id
    if report_thread_id:
        state_update["report_thread_id"] = report_thread_id

    existing_res = (
        get_client()
        .table("workflow_states")
        .select("id, state_json")
        .eq("project_id", project_id)
        .eq("phase", "pipeline_orchestrator")
        .limit(1)
        .execute()
    )
    existing_row = existing_res.data[0] if existing_res.data else None

    if existing_row:
        merged = {**(existing_row.get("state_json") or {}), **state_update}
        get_client().table("workflow_states").update({
            "status": orchestrator_stage,
            "state_json": merged,
        }).eq("id", existing_row["id"]).execute()
    else:
        get_client().table("workflow_states").insert({
            "id": str(uuid4()),
            "project_id": project_id,
            "phase": "pipeline_orchestrator",
            "phase_number": 0,
            "status": orchestrator_stage,
            "state_json": state_update,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    logger.info(f"[DB] Pipeline state → {orchestrator_stage} for project {project_id}")


def get_analogs_for_review(project_id: str) -> List[Dict]:
    """Return the most recently saved analog list for a project (for human review)."""
    return get_analogs(project_id)


def save_approved_analogs(project_id: str, approved_analogs: List[Dict]) -> None:
    """
    Insert a new workflow_states row with the human-approved analog list.
    Because get_analogs orders by created_at DESC, this supersedes the previous list.
    """
    now = datetime.now(timezone.utc).isoformat()
    get_client().table("workflow_states").insert({
        "id": str(uuid4()),
        "project_id": project_id,
        "phase": "analogs_found",
        "phase_number": 2,
        "status": "approved",
        "analogs_json": approved_analogs,
        "analogs_count": len(approved_analogs),
        "created_at": now,
    }).execute()
    logger.info(f"[DB] {len(approved_analogs)} approved analogs saved for project {project_id}")


def save_report_analogs(
    report_id: str,
    project_id: str,
    approved: List[Dict],
    rejected: List[Dict],
) -> None:
    """Insert approved and rejected analogs into the report_analogs table."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []

    def _row(a: Dict, status: str) -> Dict:
        return {
            "report_id": report_id,
            "project_id": project_id,
            "analog_name": a.get("name") or a.get("project_name") or "Unknown",
            "analog_material": a.get("material"),
            "analog_deposit_type": a.get("deposit_type"),
            "analog_host_rock": a.get("host_rock"),
            "analog_mineralization_style": a.get("mineralization_style"),
            "analog_district": a.get("district"),
            "analog_country": a.get("country"),
            "analog_tonnage_mt": a.get("tonnage_mt"),
            "analog_grade_value": a.get("grade_value"),
            "analog_grade_unit": a.get("grade_unit"),
            "analog_project_stage": a.get("project_stage"),
            # Geological profile (cascading match)
            "analog_deposit_subtype":        a.get("deposit_subtype"),
            "analog_mineralization_mode":    a.get("mineralization_mode"),
            "analog_tectonic_belt":          a.get("tectonic_belt"),
            "analog_metal_suite":            a.get("metal_suite"),
            "analog_alteration_signature":   a.get("alteration_signature"),
            "analog_recovery_method":        a.get("recovery_method"),
            "analog_mineralization_pattern":      a.get("mineralization_pattern"),
            "analog_host_rock_class":             a.get("host_rock_class"),
            "analog_project_stage_class":         a.get("project_stage_class"),
            "analog_mining_method_class":         a.get("mining_method_class"),
            "analog_resource_category_class":     a.get("resource_category_class"),
            "analog_resource_compliance_standard": a.get("resource_compliance_standard"),
            "analog_resource_vintage_year":       a.get("resource_vintage_year"),
            "similarity_score": a.get("similarity_score"),
            "source": a.get("source"),
            "source_url": a.get("source_url"),
            "status": status,
            "created_at": now,
        }

    for a in approved:
        rows.append(_row(a, "approved"))
    for a in rejected:
        rows.append(_row(a, "rejected"))

    if rows:
        get_client().table("report_analogs").insert(rows).execute()
        logger.info(
            f"[DB] report_analogs: {len(approved)} approved + {len(rejected)} rejected "
            f"for report {report_id}"
        )


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
