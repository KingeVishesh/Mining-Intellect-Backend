"""
test_graphs.py — Full end-to-end graph tests with mocked external calls.

Runs all 4 LangGraph graphs using fake but structurally correct data.
No real API keys needed. Use this to verify graph logic and state flow.

Usage:
    python scripts/test_graphs.py
    python scripts/test_graphs.py --graph project_research
"""
import sys, json, argparse, traceback
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.memory import MemorySaver

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

# ── Mock data ─────────────────────────────────────────────────────────────────

FAKE_PROJECT = {
    "id": "test-proj-001",
    "name": "Athabasca Basin Uranium Project",
    "material": "uranium",
    "company_name": "NexGen Energy",
    "country": "Canada",
    "region": "Saskatchewan",
    "deposit_type": "unconformity-related",
    "project_stage": "PFS",
    "tonnage_mt": 45.0,
    "grade_value": 2.37,
    "grade_unit": "% U3O8",
    "latitude": 58.4,
    "longitude": -109.2,
    "location_name": "Athabasca Basin, Saskatchewan, Canada",
    "mining_method": "underground",
    "processing_method": "acid leach",
    "recovery_rate": 98.5,
    "mine_life_years": 10,
    "npv_usd_millions": 3400.0,
    "capex_usd_millions": 1300.0,
    "irr_percent": 52.4,
    "enrichment_status": "complete",
    "field_statuses": {},
    "has_model_1": False,
    "updated_at": "2025-01-01T00:00:00Z",
}

FAKE_EXTRACTED_FIELDS = {
    "country": "Canada",
    "region": "Saskatchewan",
    "company_name": "NexGen Energy",
    "commodity": "uranium",
    "deposit_type": "unconformity-related",
    "project_stage": "PFS",
    "tonnage_mt": 45.0,
    "grade_value": 2.37,
    "grade_unit": "% U3O8",
    "resource_category": "M+I+Inf",
    "mining_method": "underground",
    "processing_method": "acid leach",
    "recovery_rate": 98.5,
    "mine_life_years": 10.0,
    "depth_meters": 500.0,
    "width_meters": 10.0,
    "strike_length_meters": 200.0,
    "npv_usd_millions": 3400.0,
    "capex_usd_millions": 1300.0,
    "irr_percent": 52.4,
    "opex_per_unit": 12.5,
    "payback_years": 2.5,
    "production_rate_per_year": 4500.0,
    "latitude": 58.4,
    "longitude": -109.2,
    "location_name": "Athabasca Basin, Saskatchewan, Canada",
}

FAKE_FIELD_STATUSES = {f: "found" for f in FAKE_EXTRACTED_FIELDS}

FAKE_ANALOGS = [
    {
        "id": "analog-001",
        "name": "Cigar Lake",
        "material": "uranium",
        "deposit_type": "unconformity-related",
        "tonnage_mt": 81.0,
        "grade_value": 17.9,
        "grade_unit": "% U3O8",
        "project_stage": "Production",
        "country": "Canada",
        "mining_method": "underground",
        "source": "db",
        "similarity_score": 85,
        "similarity_reasons": ["Same deposit type", "Same country"],
        "approved": True,
    },
    {
        "id": "analog-002",
        "name": "McArthur River",
        "material": "uranium",
        "deposit_type": "unconformity-related",
        "tonnage_mt": 200.0,
        "grade_value": 8.1,
        "grade_unit": "% U3O8",
        "project_stage": "Production",
        "country": "Canada",
        "mining_method": "underground",
        "source": "db",
        "similarity_score": 80,
        "similarity_reasons": ["Same deposit type", "Same region"],
        "approved": True,
    },
]

FAKE_RULES = [
    {
        "id": "rule-001",
        "rule_id": "high_grade_uranium_underground",
        "source_material": "uranium",
        "impact": "High-grade unconformity deposits benefit from selective underground mining",
        "risk": "High grade creates radiation handling challenges",
        "conditions_json": {"deposit_type": "unconformity-related"},
        "model_effects_json": {"tonnage_multiplier": 1.05, "grade_multiplier": 0.95},
        "confidence_modifier": 5.0,
        "weight": 0.8,
    }
]

FAKE_NARRATIVE = {
    "executive_summary": {
        "summary_text": "The Athabasca Basin Uranium Project is a world-class uranium deposit.",
        "overall_assessment": "Positive",
        "key_takeaway": "Exceptional grade and well-established processing make this a top-tier asset.",
    },
    "project_overview": {
        "project_summary": "Located in Saskatchewan's Athabasca Basin...",
        "key_characteristics": ["High-grade unconformity-type", "Proven underground mining method"],
        "official_mre_summary": "NI 43-101 compliant resource of 45 Mt at 2.37% U3O8.",
        "drilling_data_summary": None,
    },
    "actionable_recommendations": [
        {"recommendation": "Complete feasibility study", "priority": "High",
         "rationale": "PFS results support advancement"}
    ],
    "key_uncertainties_and_strengths": {
        "strengths": ["World-class grade", "Proven jurisdiction"],
        "uncertainties": ["Permitting timeline", "Uranium price volatility"],
    },
}


def run_test(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True))
    except Exception as e:
        print(f"{FAIL} {name}: {e}")
        tb = traceback.format_exc()
        lines = [l for l in tb.strip().splitlines() if l.strip()]
        for l in lines[-6:]:
            print(f"       {l}")
        results.append((name, False))


# ── Graph 1: project_research ─────────────────────────────────────────────────

def test_project_research():
    from graphs import project_research as pr_module

    # Compile a fresh instance with in-memory checkpointer
    g = pr_module.builder.compile(
        interrupt_before=["human_review"],
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "test-research-001"}}
    initial = {
        "project_id": "test-proj-001",
        "project_name": "Athabasca Basin Uranium Project",
        "material": "uranium",
        "company": "NexGen Energy",
    }

    # Patch all external calls at the function level
    with patch("nodes.supabase_ops.get_project", return_value=None) as _, \
         patch("nodes.supabase_ops.upsert_project", return_value=FAKE_PROJECT) as _, \
         patch("nodes.exa_search.search_project_data",
               return_value=("Exa result text about Arrow deposit...", ["https://nexgen.com"])) as _, \
         patch("nodes.exa_search.search_missing_fields",
               return_value=("", [])) as _, \
         patch("nodes.field_extractor.extract_fields",
               return_value=FAKE_EXTRACTED_FIELDS) as _, \
         patch("nodes.field_extractor.judge_fields",
               return_value=(FAKE_EXTRACTED_FIELDS, FAKE_FIELD_STATUSES)) as _, \
         patch("nodes.geocoder.geocode", return_value=(58.4, -109.2)) as _:

        # Run until interrupt_before="human_review"
        events = list(g.stream(initial, config, stream_mode="values"))
        assert len(events) > 0, "No events emitted from stream"

        state = g.get_state(config)
        next_nodes = list(state.next) if state.next else []
        print(f"\n   Interrupt at: {next_nodes}", end="")

        assert "human_review" in next_nodes, \
            f"Expected interrupt before human_review, got: {next_nodes}. error={state.values.get('error')}"

        # Verify state has expected data
        ef = state.values.get("extracted_fields")
        assert ef is not None, "extracted_fields not in state"
        assert ef.get("country") == "Canada", f"Wrong country: {ef.get('country')}"
        assert state.values.get("exa_sources") == ["https://nexgen.com"]

        # Auto-approve
        g.update_state(config, {"human_approved": True, "human_edits": {}}, as_node="human_review")
        list(g.stream(None, config, stream_mode="values"))

        final = g.get_state(config)
        assert not final.next, f"Graph still pending: {final.next}"
        assert final.values.get("saved") is True, \
            f"Project not saved. error={final.values.get('error')}"
        print(f", saved=True", end="")


# ── Graph 2: analog_finder ────────────────────────────────────────────────────

def test_analog_finder():
    from graphs import analog_finder as af_module

    g = af_module.builder.compile(
        interrupt_before=["human_review"],
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "test-analogs-001"}}

    with patch("nodes.supabase_ops.get_project", return_value=FAKE_PROJECT) as _, \
         patch("nodes.supabase_ops.search_projects_by_criteria", return_value=[
             {**FAKE_PROJECT, "id": "other-proj-1", "name": "Cigar Lake", "tonnage_mt": 81.0}
         ]) as _, \
         patch("nodes.supabase_ops.save_analogs", return_value=None) as _, \
         patch("nodes.exa_search.search_analog_projects",
               return_value=("Cigar Lake uranium mine 81Mt @ 17.9% U3O8...", ["https://cameco.com"])) as _, \
         patch("nodes.field_extractor.extract_analog_projects", return_value=[
             {"name": "Cigar Lake", "company": "Cameco", "country": "Canada",
              "deposit_type": "unconformity-related", "tonnage_mt": 81.0,
              "grade_value": 17.9, "grade_unit": "% U3O8",
              "project_stage": "Production", "mining_method": "underground",
              "source_url": "https://cameco.com"},
         ]) as _, \
         patch("nodes.field_extractor.score_analogs", return_value=[
             {**FAKE_ANALOGS[0], "similarity_score": 85},
             {**FAKE_ANALOGS[1], "similarity_score": 80},
         ]) as _:

        events = list(g.stream({"project_id": "test-proj-001"}, config, stream_mode="values"))
        assert len(events) > 0

        state = g.get_state(config)
        next_nodes = list(state.next) if state.next else []
        print(f"\n   Interrupt at: {next_nodes}", end="")

        assert "human_review" in next_nodes, \
            f"Expected human_review, got: {next_nodes}. error={state.values.get('error')}"

        scored = state.values.get("scored_analogs", [])
        assert len(scored) > 0, "No scored_analogs in state"
        print(f", analogs={len(scored)}", end="")

        # Auto-approve
        g.update_state(config, {"human_approved": True, "approved_analogs": scored}, as_node="human_review")
        list(g.stream(None, config, stream_mode="values"))

        final = g.get_state(config)
        assert not final.next, f"Graph still pending: {final.next}"
        assert final.values.get("saved") is True, \
            f"Analogs not saved. error={final.values.get('error')}"
        print(f", saved=True", end="")


# ── Graph 3: report_generator ─────────────────────────────────────────────────

def test_report_generator():
    from graphs import report_generator as rg_module

    g = rg_module.builder.compile(
        interrupt_before=["human_review_model"],
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "test-report-001"}}

    # LLM mock: returns JSON for activate_rules and narrative
    llm_mock = MagicMock()
    activate_resp = MagicMock()
    activate_resp.content = json.dumps({
        "activated_rule_ids": ["high_grade_uranium_underground"],
        "reasoning": "High-grade unconformity deposit benefits from this rule",
    })
    narrative_resp = MagicMock()
    narrative_resp.content = json.dumps(FAKE_NARRATIVE)
    llm_mock.invoke.side_effect = [activate_resp, narrative_resp]

    with patch("nodes.supabase_ops.get_project", return_value=FAKE_PROJECT) as _, \
         patch("nodes.supabase_ops.get_analogs", return_value=FAKE_ANALOGS) as _, \
         patch("nodes.rules_engine.get_compiled_rules", return_value=FAKE_RULES) as _, \
         patch("nodes.supabase_ops.save_report", return_value="report-001") as _, \
         patch("nodes.supabase_ops.upsert_project", return_value=FAKE_PROJECT) as _, \
         patch("nodes.rules_engine.get_llm", return_value=llm_mock) as _, \
         patch("nodes.model_builder.get_llm", return_value=llm_mock) as _:

        events = list(g.stream({"project_id": "test-proj-001"}, config, stream_mode="values"))
        assert len(events) > 0

        state = g.get_state(config)
        next_nodes = list(state.next) if state.next else []
        print(f"\n   Interrupt at: {next_nodes}", end="")

        assert "human_review_model" in next_nodes, \
            f"Expected human_review_model, got: {next_nodes}. error={state.values.get('error')}"

        m1 = state.values.get("model_1")
        assert m1 is not None, "model_1 not built"
        assert m1["total_tonnage_kt"] > 0, f"model_1 total_tonnage_kt=0: {m1}"
        print(f", model_1={m1['total_tonnage_kt']:.0f}kt@{m1['total_grade_pct']:.2f}", end="")

        # Auto-approve model
        g.update_state(config, {"human_approved": True, "human_model_edits": {}}, as_node="human_review_model")
        list(g.stream(None, config, stream_mode="values"))

        final = g.get_state(config)
        assert not final.next, f"Graph still pending: {final.next}"
        assert final.values.get("saved") is True, \
            f"Report not saved. error={final.values.get('error')}"

        report = final.values.get("report_json")
        assert report is not None, "No report_json in final state"
        assert "resource_estimates" in report, "Missing resource_estimates in report"
        table = report["resource_estimates"]["comparison_table"]
        assert len(table) >= 1, f"No rows in comparison table"
        assert table[0]["model"] == "Model 1 (Independent)"
        assert table[0]["mi_tonnage_kt"] > 0
        print(f", report_id=report-001, table_rows={len(table)}", end="")


# ── Graph 4: project_discovery ────────────────────────────────────────────────

def test_project_discovery():
    from graphs import project_discovery as pd_module

    # project_discovery has no interrupt, so no MemorySaver needed — but add it for consistency
    g = pd_module.builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "test-discovery-001"}}

    with patch("nodes.supabase_ops.get_project", return_value=None) as _, \
         patch("nodes.supabase_ops.upsert_project", return_value={"id": "new-proj-001"}) as _, \
         patch("nodes.exa_search.discover_new_projects",
               return_value=("NewGold announced a 10Mt copper project in Chile...", ["https://newgold.com"])) as _, \
         patch("nodes.field_extractor.extract_new_projects", return_value=[
             {"name": "NewGold Chile", "company_name": "NewGold", "country": "Chile",
              "material": "copper", "project_stage": "Exploration",
              "description": "New porphyry copper discovery"},
         ]) as _:

        events = list(g.stream({"materials": ["copper"]}, config, stream_mode="values"))
        assert len(events) > 0

        final = g.get_state(config)
        assert not final.next, f"Discovery graph still pending: {final.next}"
        saved_count = final.values.get("saved_count", 0)
        assert saved_count >= 0
        print(f"\n   saved_count={saved_count}", end="")


# ── Run all ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph",
                        choices=["project_research", "analog_finder", "report_generator", "project_discovery"])
    args = parser.parse_args()

    tests = [
        ("Graph 1: project_research (mock)", test_project_research),
        ("Graph 2: analog_finder (mock)", test_analog_finder),
        ("Graph 3: report_generator (mock)", test_report_generator),
        ("Graph 4: project_discovery (mock)", test_project_discovery),
    ]

    if args.graph:
        tests = [(n, f) for n, f in tests if args.graph in n]

    print("\n=== Mock Graph Tests ===\n")
    for name, fn in tests:
        run_test(name, fn)
        print()  # newline after inline print

    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)
    print(f"Passed: {passed}  Failed: {failed}")

    if failed:
        print("\nFailed tests:")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)
    else:
        print("All mock tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
