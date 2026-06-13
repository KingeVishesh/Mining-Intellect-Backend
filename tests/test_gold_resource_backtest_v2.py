from __future__ import annotations

import threading
import time
from datetime import date
from types import SimpleNamespace

import scripts.run_gold_resource_backtest_v2 as backtest_v2
from scripts.run_gold_resource_backtest_v2 import (
    analog_candidate_row,
    build_gold_project_row,
    build_truth_row,
    decision_rows_for_candidates,
    evidence_rows_from_payload,
    parallel_analog_prompt,
    truth_row_from_parallel,
)
from scripts.run_gold_resource_predictor_v2 import _audit_summary, _prediction_run_row


def test_truth_builder_selects_earliest_validated_full_split_mre():
    project = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "First MRE Gold",
        "mre_mi_tonnage_mt": 10,
        "mre_mi_grade": 2,
        "mre_inferred_tonnage_mt": 20,
        "mre_inferred_grade": 1,
    }
    truth, reason = build_truth_row(project, [
        {
            "id": "later",
            "effective_date": "2024-01-15",
            "mi_tonnage_mt": 12,
            "mi_grade": 2,
            "inferred_tonnage_mt": 24,
            "inferred_grade": 1,
            "source_url": "https://example.com/later",
        },
        {
            "id": "first",
            "effective_date": "2023-01-15",
            "mi_tonnage_mt": 10,
            "mi_grade": 2,
            "inferred_tonnage_mt": 20,
            "inferred_grade": 1,
            "source_url": "https://example.com/first",
        },
    ])

    assert reason is None
    assert truth is not None
    assert truth["publication_date"] == date(2023, 1, 15)
    assert truth["mi_tonnage_mt"] == 10
    assert truth["inferred_tonnage_mt"] == 20
    assert "cutoff_date" not in truth


def test_truth_builder_rejects_updated_mre_sources():
    project = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Updated MRE Gold",
        "mre_mi_tonnage_mt": 10,
        "mre_mi_grade": 2,
        "mre_inferred_tonnage_mt": 20,
        "mre_inferred_grade": 1,
    }
    truth, reason = build_truth_row(project, [
        {
            "id": "updated",
            "effective_date": "2025-05-21",
            "mi_tonnage_mt": 12,
            "mi_grade": 2,
            "inferred_tonnage_mt": 24,
            "inferred_grade": 1,
            "source_url": "https://example.com/updated-mre-technical-report",
        }
    ])

    assert truth is None
    assert reason is not None
    assert "non_first_or_updated_mre_source" in reason


def test_truth_builder_rejects_year_end_placeholder_dates():
    project = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Placeholder MRE Gold",
        "mre_mi_tonnage_mt": 10,
        "mre_mi_grade": 2,
        "mre_inferred_tonnage_mt": 20,
        "mre_inferred_grade": 1,
    }
    truth, reason = build_truth_row(project, [
        {
            "id": "placeholder",
            "effective_date": "2025-12-31",
            "mi_tonnage_mt": 12,
            "mi_grade": 2,
            "inferred_tonnage_mt": 24,
            "inferred_grade": 1,
            "source_url": "https://example.com/mineral-resource-estimate",
        }
    ])

    assert truth is None
    assert reason is not None
    assert "year_end_placeholder_mre_date" in reason


def test_truth_builder_rejects_year_start_placeholder_dates():
    project = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Placeholder MRE Gold",
        "mre_mi_tonnage_mt": 10,
        "mre_mi_grade": 2,
        "mre_inferred_tonnage_mt": 20,
        "mre_inferred_grade": 1,
    }
    truth, reason = build_truth_row(project, [
        {
            "id": "placeholder",
            "effective_date": "2025-01-01",
            "mi_tonnage_mt": 12,
            "mi_grade": 2,
            "inferred_tonnage_mt": 24,
            "inferred_grade": 1,
            "source_url": "https://example.com/mineral-resource-estimate",
        }
    ])

    assert truth is None
    assert reason is not None
    assert "year_start_placeholder_mre_date" in reason


def test_truth_builder_does_not_use_project_mre_mirror_as_fallback():
    project = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Mirror Only Gold",
        "resource_vintage_year": 2024,
        "mre_mi_tonnage_mt": 10,
        "mre_mi_grade": 2,
        "mre_inferred_tonnage_mt": 20,
        "mre_inferred_grade": 1,
    }
    truth, reason = build_truth_row(project, [])

    assert truth is None
    assert reason == "no_validated_first_mre_with_full_split_and_date"


def test_project_row_clears_stale_exclusion_reason_when_truth_validated():
    row = build_gold_project_row(
        {"id": "00000000-0000-0000-0000-000000000001", "name": "Recovered Gold"},
        data_status="truth_validated",
        exclusion_reason="old_failure_reason",
    )

    assert row["data_status"] == "truth_validated"
    assert row["exclusion_reason"] == ""


def test_parallel_truth_builder_accepts_valid_first_mre_response():
    project = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Parallel First MRE Gold",
        "mre_mi_tonnage_mt": 999,
        "mre_mi_grade": 99,
        "mre_inferred_tonnage_mt": 999,
        "mre_inferred_grade": 99,
    }
    truth, reason = truth_row_from_parallel(project, {
        "status": "validated",
        "effective_date": "2023-06-15",
        "publication_date": "2023-07-01",
        "source_url": "https://example.com/first-mineral-resource-estimate",
        "source_title": "First Mineral Resource Estimate",
        "source_publisher": "Example Mining",
        "resource_standard": "ni_43_101",
        "mi_tonnage_mt": 10,
        "mi_grade_gpt": 2,
        "inferred_tonnage_mt": 20,
        "inferred_grade_gpt": 1,
        "confidence": "high",
        "validation_notes": "First MRE press release.",
    })

    assert reason is None
    assert truth is not None
    assert truth["publication_date"] == date(2023, 7, 1)
    assert truth["effective_date"] == date(2023, 6, 15)
    assert truth["mi_tonnage_mt"] == 10
    assert truth["total_tonnage_mt"] == 30
    assert truth["raw_parallel_output"]["source"] == "parallel_mre_truth"


def test_parallel_truth_builder_rejects_updated_or_weak_response():
    project = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Parallel Weak MRE Gold",
        "mre_mi_tonnage_mt": 999,
        "mre_mi_grade": 99,
        "mre_inferred_tonnage_mt": 999,
        "mre_inferred_grade": 99,
    }
    truth, reason = truth_row_from_parallel(project, {
        "status": "validated",
        "effective_date": "2025-05-21",
        "publication_date": "2025-07-07",
        "source_url": "https://example.com/updated-mre-technical-report",
        "mi_tonnage_mt": 10,
        "mi_grade_gpt": 2,
        "inferred_tonnage_mt": 20,
        "inferred_grade_gpt": 1,
        "confidence": "high",
    })

    assert truth is None
    assert reason is not None
    assert "non_first_or_updated_mre_source" in reason


def test_parallel_cache_replays_when_paid_calls_are_disallowed(monkeypatch):
    payload = {"status": "validated"}
    monkeypatch.setattr(
        backtest_v2,
        "get_parallel_cache",
        lambda key: {"response_status": "complete", "response_payload": payload},
    )

    def fail_parallel_call(**kwargs):
        raise AssertionError("cached Parallel replay should not call provider")

    monkeypatch.setattr(backtest_v2, "_run_parallel_task", fail_parallel_call)

    response, paid_call = backtest_v2.run_parallel_cached(
        task_kind="mre_truth",
        project_id="project-1",
        cutoff_date=None,
        prompt="prompt",
        output_schema={"type": "object"},
        save=False,
        allow_paid=False,
    )

    assert response == payload
    assert paid_call is False


def test_parallel_cache_miss_does_not_spend_when_paid_calls_are_disallowed(monkeypatch):
    monkeypatch.setattr(backtest_v2, "get_parallel_cache", lambda key: None)
    monkeypatch.setattr(backtest_v2, "find_reusable_parallel_cache", lambda **kwargs: None)

    def fail_parallel_call(**kwargs):
        raise AssertionError("paid Parallel call should be gated off")

    monkeypatch.setattr(backtest_v2, "_run_parallel_task", fail_parallel_call)

    response, paid_call = backtest_v2.run_parallel_cached(
        task_kind="mre_truth",
        project_id="project-1",
        cutoff_date=None,
        prompt="prompt",
        output_schema={"type": "object"},
        save=False,
        allow_paid=False,
    )

    assert response is None
    assert paid_call is False


def test_parallel_cache_replays_equivalent_complete_cache_before_paid_call(monkeypatch):
    payload = {"total_holes": 176}
    monkeypatch.setattr(backtest_v2, "get_parallel_cache", lambda key: {"response_status": "failed"})
    monkeypatch.setattr(
        backtest_v2,
        "find_reusable_parallel_cache",
        lambda **kwargs: {"response_status": "complete", "response_payload": payload},
    )

    def fail_parallel_call(**kwargs):
        raise AssertionError("equivalent cached Parallel replay should not call provider")

    monkeypatch.setattr(backtest_v2, "_run_parallel_task", fail_parallel_call)

    response, paid_call = backtest_v2.run_parallel_cached(
        task_kind="pre_mre_evidence",
        project_id="project-1",
        cutoff_date=date(2021, 4, 1),
        prompt="prompt",
        output_schema={"type": "object"},
        save=True,
        allow_paid=True,
    )

    assert response == payload
    assert paid_call is False


def test_parallel_cache_failed_paid_call_counts_and_caches(monkeypatch):
    writes = []
    monkeypatch.setattr(backtest_v2, "get_parallel_cache", lambda key: None)
    monkeypatch.setattr(backtest_v2, "find_reusable_parallel_cache", lambda **kwargs: None)
    monkeypatch.setattr(backtest_v2, "find_recoverable_parallel_cache", lambda **kwargs: None)
    monkeypatch.setattr(backtest_v2, "upsert_parallel_cache", lambda row: writes.append(row) or row)

    def fail_parallel_call(**kwargs):
        raise RuntimeError("Parallel task did not complete within 300s (status=running, run_id=trun_timeout)")

    monkeypatch.setattr(backtest_v2, "_run_parallel_task", fail_parallel_call)

    response, paid_call = backtest_v2.run_parallel_cached(
        task_kind="analog_research",
        project_id="project-1",
        cutoff_date=date(2026, 5, 14),
        prompt="prompt",
        output_schema={"type": "object"},
        save=True,
        allow_paid=True,
    )

    assert response is None
    assert paid_call is True
    assert writes[0]["response_status"] == "failed"
    assert "Parallel task did not complete" in writes[0]["provider_error"]
    assert writes[0]["provider_task_id"] == "trun_timeout"


def test_parallel_cache_recovers_failed_provider_task_before_paid_call(monkeypatch):
    payload = {"status": "validated", "confidence": "high"}
    writes = []
    cached_row = {
        "task_kind": "mre_truth",
        "cache_key": "cache-1",
        "project_id": "project-1",
        "cutoff_date": None,
        "request_payload": {"prompt": "prompt"},
        "response_payload": {},
        "response_status": "failed",
        "provider_error": "Parallel task did not complete within 600s (status=running, run_id=trun_timeout)",
        "provider_task_id": "trun_timeout",
    }
    monkeypatch.setattr(backtest_v2, "get_parallel_cache", lambda key: cached_row)
    monkeypatch.setattr(backtest_v2, "recover_parallel_task_result", lambda run_id: payload)
    monkeypatch.setattr(backtest_v2, "upsert_parallel_cache", lambda row: writes.append(row) or row)

    def fail_parallel_call(**kwargs):
        raise AssertionError("recoverable provider task should not start a new paid call")

    monkeypatch.setattr(backtest_v2, "_run_parallel_task", fail_parallel_call)

    response, paid_call = backtest_v2.run_parallel_cached(
        task_kind="mre_truth",
        project_id="project-1",
        cutoff_date=None,
        prompt="prompt",
        output_schema={"type": "object"},
        save=True,
        allow_paid=True,
    )

    assert response == payload
    assert paid_call is False
    assert writes[0]["response_status"] == "complete"
    assert writes[0]["response_payload"] == payload
    assert writes[0]["provider_error"] == ""


def test_parallel_truth_prefetch_caps_concurrency_and_paid_budget(monkeypatch):
    projects = [
        {"id": f"project-{idx}", "name": f"Project {idx}", "material": "gold"}
        for idx in range(4)
    ]
    validated = {"project-0": {"id": "truth-0"}}
    started = []
    ensured = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    monkeypatch.setattr(backtest_v2, "parallel_truth_prompt", lambda project, legacy_runs: ("prompt", {"type": "object"}))
    monkeypatch.setattr(
        backtest_v2,
        "ensure_project_for_parallel_cache",
        lambda project, *, data_status: ensured.append((project["id"], data_status)),
    )

    def fake_run_parallel_cached(**kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            started.append(kwargs["project_id"])
            return {"status": "validated"}, True
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(backtest_v2, "run_parallel_cached", fake_run_parallel_cached)

    paid_count = backtest_v2.prefetch_parallel_truth_cache(
        projects,
        validated_gold_truths=validated,
        mre_runs_by_project={project["id"]: [] for project in projects},
        max_paid=3,
        max_workers=2,
        save=True,
    )

    assert paid_count == 3
    assert sorted(started) == ["project-1", "project-2", "project-3"]
    assert max_active <= 2
    assert ensured == [
        ("project-1", "candidate"),
        ("project-2", "candidate"),
        ("project-3", "candidate"),
    ]


def test_parallel_evidence_prefetch_caps_concurrency_and_paid_budget(monkeypatch):
    projects = [
        {"id": f"project-{idx}", "name": f"Project {idx}", "material": "gold"}
        for idx in range(4)
    ]
    truths = {
        project["id"]: {"id": f"truth-{idx}", "effective_date": date(2021, 1, 1)}
        for idx, project in enumerate(projects)
    }
    started = []
    ensured = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    monkeypatch.setattr(backtest_v2, "load_gold_case_bundle", lambda project_id: {"evidence": []})
    monkeypatch.setattr(backtest_v2, "parallel_evidence_prompt", lambda project, cutoff: ("prompt", {"type": "object"}))
    monkeypatch.setattr(
        backtest_v2,
        "ensure_project_for_parallel_cache",
        lambda project, *, data_status: ensured.append((project["id"], data_status)),
    )

    def fake_run_parallel_cached(**kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            started.append(kwargs["project_id"])
            return {"confidence": "high"}, True
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(backtest_v2, "run_parallel_cached", fake_run_parallel_cached)

    paid_count = backtest_v2.prefetch_parallel_evidence_cache(
        projects,
        validated_gold_truths=truths,
        max_paid=3,
        max_workers=2,
        save=True,
    )

    assert paid_count == 3
    assert sorted(started) == ["project-0", "project-1", "project-2"]
    assert max_active <= 2
    assert ensured == [
        ("project-0", "candidate"),
        ("project-1", "candidate"),
        ("project-2", "candidate"),
    ]


def test_parallel_evidence_prefetch_replays_db_evidence_before_paid_call(monkeypatch):
    project = {"id": "project-1", "name": "Project 1", "material": "gold"}
    truth = {"id": "truth-1", "effective_date": date(2021, 1, 1)}
    db_evidence = [
        {
            "id": "e-tonnage",
            "project_id": project["id"],
            "mre_truth_id": truth["id"],
            "cutoff_date": date(2021, 1, 1),
            "source_url": "https://example.com/pre-mre-field-estimate",
            "source_title": "Pre-MRE field estimate",
            "source_date": date(2020, 1, 1),
            "source_document_type": "news_release",
            "evidence_status": "accepted",
            "fact_type": "geometry_tonnage_mt",
            "value_num": 10,
            "confidence": "high",
            "is_mre_tainted": False,
            "fact_payload": {},
        },
        {
            "id": "e-grade",
            "project_id": project["id"],
            "mre_truth_id": truth["id"],
            "cutoff_date": date(2021, 1, 1),
            "source_url": "https://example.com/pre-mre-field-estimate",
            "source_title": "Pre-MRE field estimate",
            "source_date": date(2020, 1, 1),
            "source_document_type": "news_release",
            "evidence_status": "accepted",
            "fact_type": "weighted_grade_gpt",
            "value_num": 1.2,
            "confidence": "high",
            "is_mre_tainted": False,
            "fact_payload": {},
        },
    ]
    monkeypatch.setattr(backtest_v2, "load_gold_case_bundle", lambda project_id: {"evidence": db_evidence})

    def fail_parallel_call(**kwargs):
        raise AssertionError("DB evidence replay should prevent a paid evidence call")

    monkeypatch.setattr(backtest_v2, "run_parallel_cached", fail_parallel_call)

    paid_count = backtest_v2.prefetch_parallel_evidence_cache(
        [project],
        validated_gold_truths={project["id"]: truth},
        max_paid=1,
        max_workers=2,
        save=True,
    )

    assert paid_count == 0


def test_parallel_analog_prefetch_caps_concurrency_and_paid_budget(monkeypatch):
    projects = [
        {"id": f"project-{idx}", "name": f"Project {idx}", "material": "gold"}
        for idx in range(4)
    ]
    truths = {
        project["id"]: {"id": f"truth-{idx}", "effective_date": date(2021, 1, 1)}
        for idx, project in enumerate(projects)
    }
    started = []
    ensured = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    monkeypatch.setattr(
        backtest_v2,
        "gold_target_proxy_state",
        lambda project, truth: (date(2021, 1, 1), [], 10.0, 1.0),
    )
    monkeypatch.setattr(
        backtest_v2,
        "analog_candidate_rows_for_prefetch",
        lambda project: ({"id": project["id"]}, []),
    )
    monkeypatch.setattr(backtest_v2, "clean_analog_cohort", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(
        backtest_v2,
        "parallel_analog_prompt",
        lambda project, cutoff, *, target_tonnage_mt, target_grade_gpt: ("prompt", {"type": "object"}),
    )
    monkeypatch.setattr(
        backtest_v2,
        "ensure_project_for_parallel_cache",
        lambda project, *, data_status: ensured.append((project["id"], data_status)),
    )

    def fake_run_parallel_cached(**kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            started.append(kwargs["project_id"])
            return {"analogs": []}, True
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(backtest_v2, "run_parallel_cached", fake_run_parallel_cached)

    paid_count = backtest_v2.prefetch_parallel_analog_cache(
        projects,
        validated_gold_truths=truths,
        max_paid=3,
        max_workers=2,
        save=True,
    )

    assert paid_count == 3
    assert sorted(started) == ["project-0", "project-1", "project-2"]
    assert max_active <= 2
    assert ensured == [
        ("project-0", "candidate"),
        ("project-1", "candidate"),
        ("project-2", "candidate"),
    ]


def test_truth_parallel_research_ensures_project_before_cache_write(monkeypatch):
    project = {
        "id": "00000000-0000-0000-0000-000000000001",
        "name": "Missing First MRE Gold",
        "company_name": "Example Gold",
        "material": "gold",
        "mre_mi_tonnage_mt": 10,
        "mre_mi_grade": 1.0,
        "mre_inferred_tonnage_mt": 2,
        "mre_inferred_grade": 1.1,
    }
    events = []

    monkeypatch.setattr(backtest_v2, "gold_table_counts", lambda: {})
    def fake_fetch_projects(limit=None, project_ids=None, *, require_legacy_split=True):
        assert require_legacy_split is False
        return [project]

    monkeypatch.setattr(backtest_v2, "fetch_gold_truth_projects", fake_fetch_projects)
    monkeypatch.setattr(backtest_v2, "fetch_validated_gold_truths", lambda project_ids: {})
    monkeypatch.setattr(backtest_v2, "fetch_mre_runs", lambda project_ids: {project["id"]: []})
    monkeypatch.setattr(backtest_v2, "create_gold_backtest_batch", lambda row: {"id": "batch-1"})
    monkeypatch.setattr(backtest_v2, "update_gold_backtest_batch", lambda batch_id, patch: {})
    monkeypatch.setattr(backtest_v2, "upsert_gold_project", lambda row: {})
    monkeypatch.setattr(
        backtest_v2,
        "ensure_project_for_parallel_cache",
        lambda project, *, data_status: events.append(("ensure_project", data_status)),
    )

    def fake_run_parallel_cached(**kwargs):
        assert events == [("ensure_project", "candidate")]
        return {"status": "no_validated_first_mre", "confidence": "high"}, False

    monkeypatch.setattr(backtest_v2, "run_parallel_cached", fake_run_parallel_cached)

    summary = backtest_v2.run(SimpleNamespace(
        processor=None,
        poll_timeout_s=None,
        project_id=[project["id"]],
        limit=None,
        no_save=False,
        research_missing_truth=True,
        max_parallel_truth_projects=1,
        research_missing_evidence=False,
        max_parallel_projects=0,
        research_missing_analogs=False,
        max_parallel_analog_projects=0,
        threshold=0.05,
        run_label="test-run",
    ))

    assert summary["excluded"] == 1
    assert summary["parallel_truth_research_calls"] == 0


def test_run_replays_validated_gold_truth_before_legacy_placeholder(monkeypatch):
    project = {
        "id": "00000000-0000-0000-0000-000000000002",
        "name": "Validated DB Gold",
        "company_name": "Example Gold",
        "material": "gold",
        "country": "Australia",
        "deposit_type": "orogenic",
        "deposit_subtype": "greenstone_orogenic",
        "tectonic_belt": "yilgarn",
        "mining_method_class": "open_pit_bulk",
        "project_stage_class": "exploration",
        "mre_mi_tonnage_mt": 10,
        "mre_mi_grade": 1.0,
        "mre_inferred_tonnage_mt": 10,
        "mre_inferred_grade": 1.0,
        "resource_compliance_standard": "JORC 2012",
        "drilling_evidence": {
            "source_url": "https://example.com/drilling-results",
            "source_date": "2025-01-01",
            "geometry_tonnage_mt": 20,
            "weighted_grade_g_t": 1.0,
            "confidence": "high",
        },
    }
    db_truth = {
        "id": "truth-validated",
        "project_id": project["id"],
        "truth_status": "validated",
        "effective_date": date(2026, 5, 14),
        "publication_date": date(2026, 5, 14),
        "source_url": "https://example.com/validated-first-mre",
        "resource_standard": "JORC 2012",
        "mi_tonnage_mt": 10,
        "mi_grade_gpt": 1.0,
        "inferred_tonnage_mt": 10,
        "inferred_grade_gpt": 1.0,
        "total_tonnage_mt": 20,
        "total_grade_gpt": 1.0,
        "total_contained_oz": 643014.932,
    }
    analogs = [
        {
            "analog_name": f"Analog {idx}",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "open_pit_bulk",
            "project_stage_class": "resource_m_and_i",
            "analog_tonnage_mt": 20,
            "analog_grade_value": 1.0,
            "analog_mi_tonnage_mt": 10,
            "analog_mi_grade": 1.0,
            "analog_inferred_tonnage_mt": 10,
            "analog_inferred_grade": 1.0,
            "analog_resource_compliance_standard": "JORC 2012",
            "source_url": f"https://example.com/analog-{idx}",
            "source_date": "2025-06-01",
        }
        for idx in range(3)
    ]

    monkeypatch.setattr(backtest_v2, "gold_table_counts", lambda: {})
    def fake_fetch_projects(limit=None, project_ids=None, *, require_legacy_split=True):
        assert require_legacy_split is False
        return [project]

    monkeypatch.setattr(backtest_v2, "fetch_gold_truth_projects", fake_fetch_projects)
    monkeypatch.setattr(backtest_v2, "fetch_validated_gold_truths", lambda project_ids: {project["id"]: db_truth})
    monkeypatch.setattr(
        backtest_v2,
        "fetch_mre_runs",
        lambda project_ids: {
            project["id"]: [
                {
                    "effective_date": "2026-01-01",
                    "source_url": "https://example.com/legacy-placeholder",
                    "mi_tonnage_mt": 10,
                    "mi_grade": 1.0,
                    "inferred_tonnage_mt": 10,
                    "inferred_grade": 1.0,
                },
            ],
        },
    )
    monkeypatch.setattr(backtest_v2, "load_legacy_analogs", lambda project: analogs)

    summary = backtest_v2.run(SimpleNamespace(
        processor=None,
        poll_timeout_s=None,
        project_id=[project["id"]],
        limit=None,
        no_save=True,
        research_missing_truth=False,
        max_parallel_truth_projects=0,
        research_missing_evidence=False,
        max_parallel_projects=0,
        research_missing_analogs=False,
        max_parallel_analog_projects=0,
        threshold=0.05,
        run_label="test-db-truth-replay",
    ))

    assert summary["truth_validated"] == 1
    assert summary["excluded"] == 0
    assert summary["predicted"] == 1
    assert summary["production_like_pass"] == 1


def test_evidence_builder_stores_rejected_payloads_without_post_cutoff_source_date():
    rows = evidence_rows_from_payload(
        project_id="project-1",
        truth_id="truth-1",
        cutoff_date=date(2024, 1, 1),
        evidence={
            "source_url": "https://example.com/2024-resource-update",
            "source_date": "2024-02-01",
            "weighted_grade_g_t": 1.2,
            "confidence": "high",
        },
    )

    assert len(rows) == 1
    assert rows[0]["evidence_status"] == "rejected"
    assert rows[0]["source_date"] is None
    assert rows[0]["fact_payload"]["rejected_source_date"] == "2024-02-01"


def test_evidence_builder_normalizes_legacy_confidence_objects():
    rows = evidence_rows_from_payload(
        project_id="project-1",
        truth_id="truth-1",
        cutoff_date=date(2024, 1, 1),
        evidence={
            "source_url": "https://example.com/2023-drilling",
            "source_date": "2023-02-01",
            "weighted_grade_g_t": 1.2,
            "confidence": {"level": "High"},
        },
    )

    assert len(rows) == 1
    assert rows[0]["confidence"] == "high"


def test_evidence_builder_stores_direct_geometry_tonnage_proxy():
    rows = evidence_rows_from_payload(
        project_id="project-1",
        truth_id="truth-1",
        cutoff_date=date(2024, 1, 1),
        evidence={
            "source_url": "https://example.com/2023-drilling",
            "source_date": "2023-02-01",
            "geometry_tonnage_mt": 18.5,
            "grade_proxy_g_t": 1.7,
            "confidence": "medium",
        },
    )

    assert [row["fact_type"] for row in rows] == ["grade_proxy_gpt", "geometry_tonnage_mt"]
    assert all(row["evidence_status"] == "accepted" for row in rows)
    assert [row["value_num"] for row in rows] == [1.7, 18.5]


def test_analog_candidate_derives_mi_split_from_total_and_inferred():
    row = analog_candidate_row("target-1", {
        "analog_name": "Split Analog",
        "analog_tonnage_mt": 30,
        "analog_grade_value": 2,
        "analog_inferred_tonnage_mt": 10,
        "analog_inferred_grade": 1,
        "analog_resource_vintage_year": 2022,
        "analog_resource_compliance_standard": "ni_43_101",
        "source_url": "https://example.com/split-analog",
    })

    assert row is not None
    assert row["mi_tonnage_mt"] == 20
    assert row["mi_grade_gpt"] == 2.5
    assert row["source_date"] == date(2022, 12, 31)


def test_analog_candidate_normalizes_freeform_project_stage_classes():
    producing = analog_candidate_row("target-1", {
        "analog_name": "Producing Analog",
        "analog_tonnage_mt": 3,
        "analog_grade_value": 1.6,
        "project_stage_class": "producing",
        "source_url": "https://example.com/producing",
        "source_date": "2024-01-01",
    })
    development = analog_candidate_row("target-1", {
        "analog_name": "Development Analog",
        "analog_tonnage_mt": 4,
        "analog_grade_value": 1.8,
        "project_stage_class": "development",
        "source_url": "https://example.com/development",
        "source_date": "2024-01-01",
    })

    assert producing is not None
    assert producing["candidate_project_stage_class"] == "production"
    assert development is not None
    assert development["candidate_project_stage_class"] == "resource_m_and_i"


def test_parallel_analog_prompt_requests_auditable_split_ready_fields():
    prompt, schema = parallel_analog_prompt(
        {
            "name": "Target Gold",
            "country": "Australia",
            "deposit_subtype": "greenstone_orogenic",
            "tectonic_belt": "yilgarn",
            "mining_method_class": "open_pit_bulk",
            "project_stage_class": "exploration",
        },
        date(2026, 5, 14),
        target_tonnage_mt=3.3,
        target_grade_gpt=1.75,
    )

    analog_schema = schema["properties"]["analogs"]["items"]["properties"]
    assert "target MRE/resource information" in prompt
    assert "mi_tonnage_mt" in analog_schema
    assert "inferred_grade" in analog_schema
    assert "resource_compliance_standard" in schema["properties"]["analogs"]["items"]["required"]


def test_decision_builder_rejects_analogs_when_target_evidence_missing():
    project = {
        "id": "target-1",
        "project_name": "Target",
        "deposit_subtype": "orogenic_general",
        "tectonic_belt": "abitibi",
        "mining_method_class": "open_pit_bulk",
        "project_stage_class": "exploration",
    }
    candidate = {
        "id": "analog-1",
        "target_project_id": "target-1",
        "candidate_project_name": "Analog",
        "source_url": "https://example.com/analog",
        "source_date": date(2020, 1, 1),
        "resource_standard": "ni_43_101",
        "total_tonnage_mt": 10,
        "total_grade_gpt": 1,
    }

    decisions = decision_rows_for_candidates(project, [], [candidate], cutoff_date=date(2024, 1, 1))

    assert decisions[0]["decision"] == "rejected"
    assert "target_missing_pre_mre_tonnage_proxy" in decisions[0]["rejection_reasons"]
    assert "target_missing_pre_mre_grade_proxy" in decisions[0]["rejection_reasons"]


def test_decision_builder_reports_exploration_target_anchor_quality_reasons():
    project = {
        "id": "target-1",
        "project_name": "Target",
        "deposit_subtype": "greenstone_orogenic",
        "tectonic_belt": "yilgarn",
        "mining_method_class": "open_pit_bulk",
        "project_stage_class": "exploration",
    }
    evidence_context = {
        "project_id": "target-1",
        "cutoff_date": date(2026, 5, 14),
        "source_url": "https://example.com/revere-resource-definition.pdf",
        "source_date": date(2023, 10, 5),
        "source_title": "High Grade Reef for JORC Resource Definition",
        "evidence_status": "accepted",
        "confidence": "medium",
        "fact_payload": {
            "notes": "JORC Exploration Target of 2.5-4.1 Mt at 1-2.5 g/t Au.",
        },
    }
    candidate = {
        "id": "analog-1",
        "target_project_id": "target-1",
        "candidate_project_name": "Analog",
        "source_url": "https://example.com/analog",
        "source_date": date(2020, 1, 1),
        "resource_standard": "JORC 2012",
        "total_tonnage_mt": 3,
        "total_grade_gpt": 1.5,
    }

    decisions = decision_rows_for_candidates(
        project,
        [
            {**evidence_context, "id": "evidence-tonnage", "fact_type": "geometry_tonnage_mt", "value_num": 3.3},
            {**evidence_context, "id": "evidence-grade", "fact_type": "grade_proxy_gpt", "value_num": 1.75},
        ],
        [candidate],
        cutoff_date=date(2026, 5, 14),
    )

    assert decisions[0]["decision"] == "rejected"
    assert "exploration_target_tonnage_anchor_insufficient" in decisions[0]["rejection_reasons"]
    assert "exploration_target_grade_anchor_insufficient" in decisions[0]["rejection_reasons"]


def test_replay_audit_summary_reports_rejected_evidence_and_analog_reasons():
    bundle = {
        "all_evidence": [
            {"evidence_status": "accepted", "fact_type": "weighted_grade_gpt"},
            {
                "evidence_status": "rejected",
                "fact_type": "strike_length_m",
                "rejection_reason": "mre_tainted_source;low_confidence_weak_fact",
            },
        ],
        "rejected_evidence": [
            {
                "evidence_status": "rejected",
                "fact_type": "strike_length_m",
                "rejection_reason": "mre_tainted_source;low_confidence_weak_fact",
            },
        ],
        "analog_candidates": [{"id": "analog-1"}],
        "analog_decisions": [
            {
                "id": "decision-1",
                "decision": "rejected",
                "rejection_reasons": ["target_missing_pre_mre_tonnage_proxy"],
            }
        ],
    }

    audit = _audit_summary(bundle)

    assert audit["evidence"]["accepted_count"] == 1
    assert audit["evidence"]["rejected_count"] == 1
    assert audit["evidence"]["rejection_reasons"]["mre_tainted_source"] == 1
    assert audit["analogs"]["candidate_count"] == 1
    assert audit["analogs"]["decision_counts"]["rejected"] == 1
    assert audit["analogs"]["rejection_reasons"]["target_missing_pre_mre_tonnage_proxy"] == 1


def test_replay_prediction_row_persists_existing_analog_decision_ids():
    row = _prediction_run_row(
        "project-1",
        {"id": "truth-1"},
        {
            "run_status": "no_prediction",
            "input_hash": "hash-1",
            "cutoff_date": "2024-01-01",
            "predictor_version": "test",
            "no_prediction_reasons": ["insufficient_pre_mre_tonnage_evidence"],
            "calculator_trace": {},
        },
        [
            {
                "id": "decision-1",
                "decision": "rejected",
                "analog_candidate_id": "analog-1",
            },
            {
                "id": "decision-2",
                "decision": "accepted",
                "analog_candidate_id": "analog-2",
            },
        ],
    )

    assert row["analog_decision_ids"] == ["decision-1", "decision-2"]
    assert row["analog_candidate_ids"] == ["analog-2"]
