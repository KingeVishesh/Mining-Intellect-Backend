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
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from supabase import create_client, Client
from config import settings
from nodes import geo_taxonomy

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


_LIBRARY_SELECT = (
    "analog_name,analog_material,analog_deposit_type,analog_country,"
    "analog_tonnage_mt,analog_grade_value,analog_grade_unit,source_url,"
    "similarity_score,analog_project_stage,"
    "analog_host_rock,analog_mineralization_style,analog_district,"
    "analog_deposit_subtype,analog_mineralization_mode,analog_tectonic_belt,"
    "analog_metal_suite,analog_alteration_signature,analog_recovery_method,"
    "analog_mineralization_pattern,analog_host_rock_class,"
    "analog_project_stage_class,analog_mining_method_class,"
    "analog_resource_category_class,analog_resource_compliance_standard,"
    "analog_resource_vintage_year"
)

_FAMILY_KEYWORDS = (
    "alkalic", "calc-alkalic", "carlin", "porphyry", "epithermal",
    "orogenic", "iocg", "skarn", "vms", "vhms", "sediment-hosted",
    "kupferschiefer", "manto", "crd", "sedex", "mvt", "merensky",
    "ug2", "platreef", "laterite", "magmatic", "komatiite", "bif",
    "unconformity", "roll-front", "rollfront", "iscr",
)


def get_approved_analogs(
    material: str,
    deposit_type: Optional[str] = None,
    limit: int = 200,
    deposit_subtype: Optional[str] = None,
    deposit_subtypes: Optional[List[str]] = None,
    target_tectonic_belt: Optional[str] = None,
) -> List[Dict]:
    """Query the `analogs` table for previously seen analogs of this commodity.

    The `analogs` table is populated automatically at the end of every
    `combine_filter_score_node` run (cascade-pass = quality gate), so the
    library grows organically without waiting for report finalization.

    Filter strategy (most specific first):
      1. analog_deposit_subtype == deposit_subtype  (exact slug match — preferred)
      2. analog_deposit_type ILIKE %<family-keyword>%  (e.g., %carlin%, %porphyry%)
         where the keyword is the first geological term in deposit_type,
         stripped of qualifiers like "-style", "-type", "sediment-hosted".

    When `target_tectonic_belt` is supplied, the row budget is spent
    intelligently using a three-pass query:
      Pass 1 — exact target-belt rows.  These are the highest-signal analogs
               and must not be pushed out by generic null-belt rows.
      Pass 2 — null-belt rows.  Null-belt candidates pass L2.5 through, so
               keeping them honors the cascade's contract.
      Pass 3 — sibling belts in the same compatibility group fill the
               remaining slots.  Out-of-group belts (Lachlan, Guiana,
               Newfoundland-Appalachian, Brazilian Shield, etc. for an
               archean_greenstone target) are never fetched — the cascade
               would drop them at L2.5 anyway.
    Net effect: the limit=200 budget is spent only on candidates the
    cascade can actually use.

    When deposit_type is None AND deposit_subtype is None, the library is
    skipped entirely (no rule = don't poison the pool).
    """
    accepted_subtypes: List[str] = list(deposit_subtypes or [])
    if deposit_subtype and deposit_subtype not in accepted_subtypes:
        accepted_subtypes.append(deposit_subtype)

    keys = _MATERIAL_TO_RULES_KEYS.get(material.strip().lower(), [material.strip().lower()])
    dep = (deposit_type or "").strip().lower()

    if not accepted_subtypes and not dep:
        return []

    def _base_query():
        """Fresh query with material + status + subtype/keyword filters applied."""
        q = (
            get_client()
            .table("analogs")
            .select(_LIBRARY_SELECT)
            .in_("analog_material", keys)
            .eq("status", "approved")
        )
        if accepted_subtypes:
            if len(accepted_subtypes) == 1:
                q = q.eq("analog_deposit_subtype", accepted_subtypes[0])
            else:
                q = q.in_("analog_deposit_subtype", accepted_subtypes)
        elif dep:
            keyword = next((kw for kw in _FAMILY_KEYWORDS if kw in dep), None)
            if keyword is None:
                keyword = next((w for w in dep.split() if len(w) >= 4), dep[:8])
            q = q.ilike("analog_deposit_type", f"%{keyword}%")
        return q

    rows: List[Dict] = []
    compatible = geo_taxonomy.compatible_belts(target_tectonic_belt) if target_tectonic_belt else []

    if compatible:
        # Pass 1: exact target-belt rows. Gold backtests showed that combining
        # exact + null in one OR query can let generic rows consume the cap
        # before district/belt peers are considered.
        pass1 = (
            _base_query()
            .eq("analog_tectonic_belt", target_tectonic_belt)
            .limit(limit)
            .execute()
        )
        rows = list(pass1.data or [])

        # Pass 2: null-belt rows.
        remaining = limit - len(rows)
        if remaining > 0:
            pass2 = (
                _base_query()
                .is_("analog_tectonic_belt", "null")
                .limit(remaining)
                .execute()
            )
            rows.extend(pass2.data or [])

        # Pass 3: sibling belts in the same compatibility group, filling
        # whatever budget Pass 1 didn't consume.
        remaining = limit - len(rows)
        siblings = [b for b in compatible if b != target_tectonic_belt]
        if remaining > 0 and siblings:
            pass3 = (
                _base_query()
                .in_("analog_tectonic_belt", siblings)
                .limit(remaining)
                .execute()
            )
            rows.extend(pass3.data or [])
    else:
        # No belt info on the target (or belt not in any compatibility group)
        # → fall back to the original single-query approach.
        rows = (_base_query().limit(limit).execute()).data or []

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


def save_analogs(
    project_id: str,
    analogs: List[Dict],
    thread_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """
    Persist analogs in three places:
      1. workflow_states (legacy — Analog Tester page reads from here)
      2. analogs_runs (time-series history, one row per run)
      3. projects.analogs (latest snapshot, for quick lookup + UI display)
    """
    now = datetime.now(timezone.utc).isoformat()
    client = get_client()

    client.table("workflow_states").insert({
        "id": str(uuid4()),
        "project_id": project_id,
        "phase": "analogs_found",
        "phase_number": 2,
        "status": "complete",
        "analogs_json": analogs,
        "analogs_count": len(analogs),
        "created_at": now,
    }).execute()

    _persist_analogs_run_and_latest(project_id, analogs, thread_id, run_id)
    logger.info(f"[DB] {len(analogs)} analogs saved for project {project_id}")


def _persist_analogs_run_and_latest(
    project_id: str,
    analogs: List[Dict],
    thread_id: Optional[str],
    run_id: Optional[str],
) -> None:
    """Write to analogs_runs (history) and projects.analogs (latest)."""
    client = get_client()
    try:
        client.table("analogs_runs").insert({
            "project_id": project_id,
            "thread_id": thread_id,
            "run_id": run_id,
            "analogs_json": analogs,
            "analogs_count": len(analogs),
        }).execute()
    except Exception as e:
        logger.warning(f"[DB] analogs_runs insert failed: {e}")
    try:
        client.table("projects").update({"analogs": analogs}).eq("id", project_id).execute()
    except Exception as e:
        logger.warning(f"[DB] projects.analogs update failed: {e}")


def _project_analog_row_to_candidate(row: Dict) -> Dict:
    return {
        "name": row.get("analog_name"),
        "material": row.get("analog_material"),
        "deposit_type": row.get("analog_deposit_type"),
        "host_rock": row.get("analog_host_rock"),
        "mineralization_style": row.get("analog_mineralization_style"),
        "district": row.get("analog_district"),
        "country": row.get("analog_country"),
        "tonnage_mt": row.get("analog_tonnage_mt"),
        "grade_value": row.get("analog_grade_value"),
        "grade_unit": row.get("analog_grade_unit"),
        "source_url": row.get("source_url"),
        "project_stage": row.get("analog_project_stage"),
        "deposit_subtype": row.get("analog_deposit_subtype"),
        "mineralization_mode": row.get("analog_mineralization_mode"),
        "tectonic_belt": row.get("analog_tectonic_belt"),
        "metal_suite": row.get("analog_metal_suite"),
        "alteration_signature": row.get("analog_alteration_signature"),
        "recovery_method": row.get("analog_recovery_method"),
        "mineralization_pattern": row.get("analog_mineralization_pattern"),
        "host_rock_class": row.get("analog_host_rock_class"),
        "project_stage_class": row.get("analog_project_stage_class"),
        "mining_method_class": row.get("analog_mining_method_class"),
        "resource_category_class": row.get("analog_resource_category_class"),
        "resource_compliance_standard": row.get("analog_resource_compliance_standard"),
        "resource_vintage_year": row.get("analog_resource_vintage_year"),
        "similarity_score": row.get("similarity_score"),
        "source": row.get("source") or "analogs_table",
    }


def _get_project_approved_analogs(project_id: str, limit: int = 10) -> List[Dict]:
    res = (
        get_client()
        .table("analogs")
        .select(_LIBRARY_SELECT)
        .eq("project_id", project_id)
        .eq("status", "approved")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    out: List[Dict] = []
    seen: set[str] = set()
    for row in res.data or []:
        cand = _project_analog_row_to_candidate(row)
        name = (cand.get("name") or "").strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(cand)
    return out


def get_analogs(project_id: str) -> List[Dict]:
    """Load the freshest saved analogs for a project.

    Prefer the latest workflow state for UI continuity, but fall back to the
    approved analog table when a later low-confidence run saved an empty/thin
    workflow cohort. This prevents model runs from being starved when the
    library still has cascade-approved analogs for the same project.
    """
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
    workflow = []
    if res.data:
        workflow = res.data[0].get("analogs_json") or []
    if workflow:
        return workflow
    table_analogs = _get_project_approved_analogs(project_id)
    if not table_analogs:
        return workflow
    merged: List[Dict] = []
    seen: set[str] = set()
    for analog in list(workflow) + table_analogs:
        name = (analog.get("name") or analog.get("analog_name") or "").strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        merged.append(analog)
    return merged


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


def save_approved_analogs(
    project_id: str,
    approved_analogs: List[Dict],
    thread_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """
    Insert a new workflow_states row with the human-approved analog list.
    Because get_analogs orders by created_at DESC, this supersedes the previous list.
    Also writes to analogs_runs (history) and projects.analogs (latest snapshot).
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
    _persist_analogs_run_and_latest(project_id, approved_analogs, thread_id, run_id)
    logger.info(f"[DB] {len(approved_analogs)} approved analogs saved for project {project_id}")


# ── Model runs ─────────────────────────────────────────────────────────────────

# Fields written to BOTH projects (latest snapshot) and model_runs (history).
# Subset that the `/projects-back` table relies on for rendering each row.
_PROJECT_MODEL_FIELDS = (
    # Measured + Indicated combined (M&I)
    "mi_tonnage_mt", "mi_grade", "mi_contained",
    # Inferred
    "inferred_resource_mt", "inferred_grade", "inferred_contained",
    # Totals
    "tonnage_mt", "grade_value", "total_contained",
    # Conviction
    "conviction_score",
)

# Fields written to model_runs ONLY. P1 adds posterior percentiles + CV; the
# matching projects-table columns are deferred to P2.
_MODEL_RUN_FIELDS = _PROJECT_MODEL_FIELDS + (
    "conviction_tier",
    "p10_tonnage_mt", "p50_tonnage_mt", "p90_tonnage_mt",
    "p10_grade",       "p50_grade",       "p90_grade",
    "p10_contained",   "p50_contained",   "p90_contained",
    "cv_contained",
)


def save_model_run(
    project_id: str,
    model_type: str,
    fields: Dict,
    model_output_json: Optional[Dict] = None,
    thread_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Insert a row into model_runs capturing one Model 1 / Model 2 snapshot.

    The `signal_contributions_json` column captures per-signal (μ, σ) for
    each fusion input so future recalibration passes can attribute residual
    error back to the analog pool vs. geometry vs. rules.
    """
    row = {
        "project_id": project_id,
        "model_type": model_type,
        "thread_id": thread_id,
        "run_id": run_id,
        "model_output_json": model_output_json or {},
        "signal_contributions_json": fields.get("signal_contributions_json"),
        "status": "complete",
        **{k: fields.get(k) for k in _MODEL_RUN_FIELDS},
    }
    get_client().table("model_runs").insert(row).execute()
    logger.info(
        f"[DB] model_run saved: project={project_id} type={model_type} "
        f"tonnage={fields.get('tonnage_mt')} conviction={fields.get('conviction_score')} "
        f"cv={fields.get('cv_contained')}"
    )


def update_project_latest_model(project_id: str, fields: Dict) -> None:
    """Overwrite the latest-model columns on projects so the /projects-back
    table reflects the most recent run. Percentile / CV / signal-trace fields
    are intentionally NOT mirrored to projects in P1 — they live only in
    model_runs to keep the projects schema stable until P2.
    """
    payload = {k: fields.get(k) for k in _PROJECT_MODEL_FIELDS}
    get_client().table("projects").update(payload).eq("id", project_id).execute()
    logger.info(f"[DB] projects.{{model fields}} updated for project {project_id}")


# ── MRE history (mre_runs) ────────────────────────────────────────────────────

# A 1% tolerance on tonnage / grade is the threshold for "MRE actually
# changed". Below that we treat repeated extractions as the same MRE and
# don't insert a new mre_runs row — saves the table from filling up with
# noise when project_research re-fetches the same NI 43-101 report.
_MRE_CHANGE_TOLERANCE = 0.01


def get_latest_mre_run(project_id: str) -> Optional[Dict]:
    """Return the most-recently-inserted mre_runs row for a project, or
    None when no MRE has ever been recorded.
    """
    res = (
        get_client().table("mre_runs")
        .select("*")
        .eq("project_id", project_id)
        .order("fetched_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None


def _mre_changed(latest: Optional[Dict], new: Dict) -> bool:
    """Whether a new extraction represents an actual MRE update.

    Considers the totals, the M&I and Inferred breakdowns when present,
    the resource_category text, and the effective_date. A new effective
    date (the MRE's reported cutoff) always counts as a new update, even
    if the numbers happen to match.
    """
    if latest is None:
        # First time we've recorded an MRE for this project — anything
        # numeric is worth saving.
        return any(
            new.get(k) is not None
            for k in ("total_tonnage_mt", "total_grade",
                      "mi_tonnage_mt", "inferred_tonnage_mt")
        )
    # Effective-date change always wins
    if new.get("effective_date") and new["effective_date"] != latest.get("effective_date"):
        return True
    # Category change is also a content change
    if (new.get("resource_category") or "").strip() != (latest.get("resource_category") or "").strip():
        return True
    # Numeric drift on any tracked field beyond 1%
    for field in (
        "total_tonnage_mt", "total_grade", "total_contained",
        "mi_tonnage_mt", "mi_grade", "mi_contained",
        "inferred_tonnage_mt", "inferred_grade", "inferred_contained",
    ):
        old_v = latest.get(field)
        new_v = new.get(field)
        if old_v is None and new_v is None:
            continue
        if (old_v is None) != (new_v is None):
            return True
        if old_v == 0 and new_v == 0:
            continue
        denom = abs(old_v) if old_v else 1.0
        if abs((new_v or 0) - (old_v or 0)) / denom > _MRE_CHANGE_TOLERANCE:
            return True
    return False


def save_mre_run_if_changed(
    project_id: str,
    extracted: Dict,
    thread_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Optional[Dict]:
    """Insert a new mre_runs row when the extracted MRE differs from the
    latest cached row; otherwise skip. Returns the inserted row dict on
    save, or None when no insert happened.

    Caller is responsible for also updating the projects.* mirror columns
    (tonnage_mt, grade_value, mre_mi_*, mre_inferred_*) — those reflect
    the LATEST values, while mre_runs preserves the history.
    """
    latest = get_latest_mre_run(project_id)
    if not _mre_changed(latest, extracted):
        logger.info(
            f"[mre_runs] No change vs latest for project {project_id} — "
            f"not inserting duplicate row"
        )
        return None
    row = {
        "project_id": project_id,
        "thread_id": thread_id,
        "run_id": run_id,
        # Fields below are pulled from `extracted` defensively so a caller
        # can pass a subset (e.g. totals only, no breakdown) and the
        # rest stay NULL.
        "effective_date":       extracted.get("effective_date"),
        "resource_category":    extracted.get("resource_category"),
        "total_tonnage_mt":     extracted.get("total_tonnage_mt"),
        "total_grade":          extracted.get("total_grade"),
        "total_contained":      extracted.get("total_contained"),
        "grade_unit":           extracted.get("grade_unit"),
        "mi_tonnage_mt":        extracted.get("mi_tonnage_mt"),
        "mi_grade":             extracted.get("mi_grade"),
        "mi_contained":         extracted.get("mi_contained"),
        "inferred_tonnage_mt":  extracted.get("inferred_tonnage_mt"),
        "inferred_grade":       extracted.get("inferred_grade"),
        "inferred_contained":   extracted.get("inferred_contained"),
        "source":               extracted.get("source", "ni_43_101"),
        "source_url":           extracted.get("source_url"),
        "notes":                extracted.get("notes"),
    }
    res = get_client().table("mre_runs").insert(row).execute()
    logger.info(
        f"[mre_runs] Inserted MRE update for project {project_id}: "
        f"total={extracted.get('total_tonnage_mt')} Mt @ "
        f"{extracted.get('total_grade')} {extracted.get('grade_unit','')} "
        f"(M&I={extracted.get('mi_tonnage_mt')}, "
        f"Inf={extracted.get('inferred_tonnage_mt')})"
    )
    return res.data[0] if res.data else row


def update_project_mre_mirror(project_id: str, extracted: Dict) -> None:
    """Mirror the latest MRE values to the projects table so existing
    queries continue to work. Only writes fields that have non-null
    values in `extracted` — preserves previously-known values when a
    fresh extraction is partial.
    """
    payload: Dict = {}
    # The legacy totals already live on projects.tonnage_mt / grade_value
    if extracted.get("total_tonnage_mt") is not None:
        payload["tonnage_mt"] = extracted["total_tonnage_mt"]
    if extracted.get("total_grade") is not None:
        payload["grade_value"] = extracted["total_grade"]
    if extracted.get("grade_unit"):
        payload["grade_unit"] = extracted["grade_unit"]
    if extracted.get("resource_category"):
        payload["resource_category"] = extracted["resource_category"]
    # New breakdown columns
    for src, dst in (
        ("mi_tonnage_mt",      "mre_mi_tonnage_mt"),
        ("mi_grade",           "mre_mi_grade"),
        ("mi_contained",       "mre_mi_contained"),
        ("inferred_tonnage_mt","mre_inferred_tonnage_mt"),
        ("inferred_grade",     "mre_inferred_grade"),
        ("inferred_contained", "mre_inferred_contained"),
    ):
        if extracted.get(src) is not None:
            payload[dst] = extracted[src]
    if not payload:
        return
    get_client().table("projects").update(payload).eq("id", project_id).execute()
    logger.info(
        f"[mre_mirror] project {project_id} updated with "
        f"{list(payload.keys())}"
    )


# ── Drilling evidence (Model 1 v2 P3) ──────────────────────────────────────────

def get_project_drilling_evidence(
    project_id: str,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Return (drilling_evidence, fetched_at_iso) for a project, or (None, None)
    when unset. Powers the cache check in fetch_drilling_evidence_node so we
    don't re-hit Exa on every model run.
    """
    res = (
        get_client().table("projects")
        .select("drilling_evidence,drilling_evidence_fetched_at")
        .eq("id", project_id).maybe_single().execute()
    )
    if not res.data:
        return None, None
    return res.data.get("drilling_evidence"), res.data.get("drilling_evidence_fetched_at")


def save_project_drilling_evidence(project_id: str, evidence: Dict) -> None:
    """Persist freshly-extracted drilling evidence to the projects row."""
    payload = {
        "drilling_evidence": evidence,
        "drilling_evidence_fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    get_client().table("projects").update(payload).eq("id", project_id).execute()
    logger.info(f"[DB] drilling_evidence saved for project {project_id}")


def get_analog_drilling_evidence(
    analog_name: str, material: str,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Look up cached drilling evidence by analog name + material. Returns
    (evidence, fetched_at_iso) or (None, None) when not yet extracted.
    """
    res = (
        get_client().table("analogs")
        .select("drilling_evidence,drilling_evidence_fetched_at")
        .eq("analog_name", analog_name)
        .ilike("analog_material", material)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None, None
    return rows[0].get("drilling_evidence"), rows[0].get("drilling_evidence_fetched_at")


def save_analog_drilling_evidence(
    analog_name: str, material: str, evidence: Dict,
) -> None:
    """Persist analog drilling evidence across every analogs row matching
    (name, material). The same library analog can appear in many reports;
    we want a single drilling_evidence value shared across them.
    """
    payload = {
        "drilling_evidence": evidence,
        "drilling_evidence_fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (
        get_client().table("analogs")
        .update(payload)
        .eq("analog_name", analog_name)
        .ilike("analog_material", material)
        .execute()
    )
    logger.info(f"[DB] drilling_evidence saved for analog '{analog_name}'")


def get_analog_inferred_data(
    analog_name: str, material: str,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Look up the cached Inferred breakdown for an analog by (name, material).
    Returns ({inferred_tonnage_mt, inferred_grade}, fetched_at_iso) or
    (None, None) when the analog hasn't been extracted yet. A row that
    HAS been extracted but reported null for both fields (analog has no
    Inferred or no M&I/Inferred breakdown) returns
    ({inferred_tonnage_mt: None, inferred_grade: None}, iso) so callers
    know not to refetch within the staleness window.
    """
    res = (
        get_client().table("analogs")
        .select("analog_inferred_tonnage_mt,analog_inferred_grade,inferred_extracted_at")
        .eq("analog_name", analog_name)
        .ilike("analog_material", material)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None, None
    row = rows[0]
    if row.get("inferred_extracted_at") is None:
        return None, None
    return (
        {
            "inferred_tonnage_mt": row.get("analog_inferred_tonnage_mt"),
            "inferred_grade":      row.get("analog_inferred_grade"),
        },
        row.get("inferred_extracted_at"),
    )


def save_analog_inferred_data(
    analog_name: str, material: str,
    inferred_tonnage_mt: Optional[float],
    inferred_grade: Optional[float],
) -> None:
    """Persist the analog's M&I + Inferred breakdown across every analogs
    row matching (name, material). The same library analog can appear in
    many reports; we want a single canonical value shared across them.

    NULLs are persisted (they mean "extraction attempted, no data found")
    so the staleness window suppresses re-fetches.
    """
    payload = {
        "analog_inferred_tonnage_mt": inferred_tonnage_mt,
        "analog_inferred_grade":      inferred_grade,
        "inferred_extracted_at":      datetime.now(timezone.utc).isoformat(),
    }
    (
        get_client().table("analogs")
        .update(payload)
        .eq("analog_name", analog_name)
        .ilike("analog_material", material)
        .execute()
    )
    logger.info(
        f"[DB] inferred data saved for analog '{analog_name}': "
        f"{inferred_tonnage_mt} Mt @ {inferred_grade}"
    )


def _analog_row(
    a: Dict,
    project_id: str,
    status: str,
    now: str,
    report_id: Optional[str] = None,
) -> Dict:
    """Build an `analogs` row from a candidate dict."""
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


def upsert_analog_library(
    project_id: str,
    approved: List[Dict],
    rejected: Optional[List[Dict]] = None,
    report_id: Optional[str] = None,
) -> None:
    """Upsert approved + rejected analogs into the `analogs` table (library).

    Conflict key is (project_id, analog_name) — re-runs of the same project
    update the existing row in place rather than inserting duplicates. Used
    by both:
      * the write-on-cascade hook in combine_filter_score_node (no report_id)
      * report finalization in pipeline_orchestrator (with report_id)
    """
    now = datetime.now(timezone.utc).isoformat()
    rows = [_analog_row(a, project_id, "approved", now, report_id) for a in (approved or [])]
    rows += [_analog_row(a, project_id, "rejected", now, report_id) for a in (rejected or [])]

    if not rows:
        return

    (
        get_client()
        .table("analogs")
        .upsert(rows, on_conflict="project_id,analog_name")
        .execute()
    )
    logger.info(
        f"[DB] analogs upsert: {len(approved or [])} approved + "
        f"{len(rejected or [])} rejected for project {project_id}"
        + (f" (report {report_id})" if report_id else "")
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
