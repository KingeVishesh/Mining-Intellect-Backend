"""
Bootstrap — backend startup hook that keeps `compiled_rules` in sync with
`scripts/seed_analog_rules.py`.

How it works
------------
On import (or explicit call), we compute a deterministic sha256 hash of the
rule data declared in code. We compare it to the `rules_version` row in the
`system_meta` table. If different, we upsert every rule via the existing
`scripts.seed_analog_rules.build_rows()` machinery and update the hash row.

Result: a senior geologist edits a rule in code, opens a PR, the merged build
boots → the DB matches the code within seconds. UI edits to `compiled_rules`
are overwritten on next boot. **Code is canonical, DB is a read-through cache.**
"""
from __future__ import annotations
import hashlib
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _compute_rules_hash() -> tuple[str, int]:
    """Return (sha256-hex-digest, rule_count) for the current code rule set."""
    # Lazy import to avoid heavy load when bootstrap is imported in test paths
    from scripts.seed_analog_rules import (
        ANALOG_SELECTION_RULES,
        CONFIDENCE_RULES,
        ALL_COMMODITIES,
    )

    # Confidence rules expand per-commodity inside build_rows; replicate the
    # expansion here so the hash matches what actually lands in the DB.
    expanded_conf = []
    for r in CONFIDENCE_RULES:
        for commodity in ALL_COMMODITIES:
            new_r = dict(r)
            new_r["rule_id"] = (
                r["rule_id"].replace("gold", commodity)
                if "gold" in r["rule_id"]
                else f"{r['rule_id']}_{commodity}"
            )
            new_r["source_material"] = commodity
            expanded_conf.append(new_r)

    rules_for_hash = sorted(
        ANALOG_SELECTION_RULES + expanded_conf,
        key=lambda r: r.get("rule_id", ""),
    )
    blob = json.dumps(rules_for_hash, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return digest, len(rules_for_hash)


def _read_rules_version(client) -> Optional[dict]:
    """Read the stored rules_version row from system_meta. None if missing."""
    try:
        res = (
            client.table("system_meta")
            .select("value")
            .eq("key", "rules_version")
            .maybe_single()
            .execute()
        )
        return (res.data or {}).get("value")
    except Exception as e:
        logger.warning(f"[bootstrap] system_meta read failed: {e}")
        return None


def _write_rules_version(client, digest: str, rules_count: int) -> None:
    """Persist the new rules_version hash."""
    from datetime import datetime, timezone
    client.table("system_meta").upsert(
        {
            "key": "rules_version",
            "value": {"hash": digest, "rules_count": rules_count},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="key",
    ).execute()


def bootstrap_rules(force: bool = False) -> dict:
    """
    Ensure compiled_rules in the database matches the current code rule set.

    Returns a status dict: {action, code_hash, db_hash, rules_count}.
    action ∈ {"in_sync", "reseeded", "skipped_error"}.

    Safe to call repeatedly; the no-op path is one read of system_meta.
    """
    from nodes.supabase_ops import get_client
    from scripts.seed_analog_rules import (
        build_rows, ANALOG_SELECTION_RULES, CONFIDENCE_RULES, ALL_COMMODITIES,
    )

    code_hash, code_count = _compute_rules_hash()
    try:
        client = get_client()
        stored = _read_rules_version(client) or {}
        db_hash = stored.get("hash")
    except Exception as e:
        logger.error(f"[bootstrap] Supabase unavailable: {e} — skipping reseed")
        return {"action": "skipped_error", "code_hash": code_hash,
                "db_hash": None, "rules_count": code_count, "error": str(e)}

    if not force and db_hash == code_hash:
        logger.info(f"[bootstrap] rules in sync (hash={code_hash[:12]}…, {code_count} rules)")
        return {"action": "in_sync", "code_hash": code_hash,
                "db_hash": db_hash, "rules_count": code_count}

    logger.info(
        f"[bootstrap] rules drift detected — reseeding "
        f"(code={code_hash[:12]}…, db={(db_hash or 'none')[:12]}…)"
    )
    analog_rows = build_rows(ANALOG_SELECTION_RULES, "analog_selection")
    conf_rows = build_rows(CONFIDENCE_RULES, "confidence_adjustment",
                            extra_commodities=ALL_COMMODITIES)
    all_rows = analog_rows + conf_rows

    saved = 0
    for i in range(0, len(all_rows), 20):
        batch = all_rows[i : i + 20]
        client.table("compiled_rules").upsert(batch, on_conflict="rule_id").execute()
        saved += len(batch)

    _write_rules_version(client, code_hash, code_count)
    logger.info(f"[bootstrap] reseeded {saved} rules; hash={code_hash[:12]}…")
    return {"action": "reseeded", "code_hash": code_hash,
            "db_hash": db_hash, "rules_count": code_count, "rows_written": saved}
