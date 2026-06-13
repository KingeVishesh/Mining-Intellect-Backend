from __future__ import annotations

from pathlib import Path

from nodes.gold_resource_storage import GOLD_TABLES


ROOT = Path(__file__).resolve().parent.parent
SQL = (ROOT / "sql" / "gold_resource_predictor_schema.sql").read_text()


def test_gold_schema_declares_required_tables():
    for table in (
        "gold_projects",
        "gold_mre_truths",
        "gold_pre_mre_evidence",
        "gold_analog_candidates",
        "gold_analog_decisions",
        "gold_prediction_runs",
        "gold_prediction_scores",
        "gold_backtest_batches",
        "gold_parallel_cache",
    ):
        assert f"create table if not exists public.{table}" in SQL
        assert table in set(GOLD_TABLES.values())


def test_gold_schema_enables_rls_on_every_gold_table():
    for table in GOLD_TABLES.values():
        assert f"alter table public.{table} enable row level security;" in SQL


def test_gold_schema_keeps_client_roles_closed_by_default():
    for table in GOLD_TABLES.values():
        assert f"revoke all on table public.{table} from anon, authenticated;" in SQL
        assert f"grant select, insert, update, delete on table public.{table} to service_role;" in SQL

    assert " to anon;" not in SQL.lower()
    assert " to authenticated;" not in SQL.lower()


def test_gold_schema_stores_rejections_and_parallel_cache():
    assert "rejection_reason text" in SQL
    assert "rejection_reasons text[]" in SQL
    assert "raw_parallel_output jsonb" in SQL
    assert "cache_key text not null unique" in SQL
    assert "task_kind in ('mre_truth', 'pre_mre_evidence', 'analog_research')" in SQL
