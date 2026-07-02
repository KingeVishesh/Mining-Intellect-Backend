from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nodes import supabase_ops


class _FakeQuery:
    def __init__(self, table_name: str, workflow):
        self.table_name = table_name
        self.workflow = workflow

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.table_name == "workflow_states":
            return SimpleNamespace(data=[{"analogs_json": self.workflow}])
        if self.table_name == "analogs":
            return SimpleNamespace(data=[{
                "analog_name": "BAM East Gold Deposit",
                "analog_material": "Gold",
                "analog_tonnage_mt": 18.0,
                "analog_grade_value": 1.2,
                "analog_grade_unit": "g/t Au",
                "status": "approved",
            }])
        return SimpleNamespace(data=[])


class _FakeClient:
    def __init__(self, workflow=None):
        self.workflow = [] if workflow is None else workflow

    def table(self, table_name: str):
        return _FakeQuery(table_name, self.workflow)


def test_get_analogs_falls_back_to_approved_table_rows(monkeypatch):
    monkeypatch.setattr(supabase_ops, "get_client", lambda: _FakeClient())

    analogs = supabase_ops.get_analogs("project-id")

    assert analogs == [{
        "name": "BAM East Gold Deposit",
        "material": "Gold",
        "deposit_type": None,
        "host_rock": None,
        "mineralization_style": None,
        "district": None,
        "country": None,
        "tonnage_mt": 18.0,
        "grade_value": 1.2,
        "mi_tonnage_mt": 18.0,
        "mi_grade": 1.2,
        "inferred_tonnage_mt": None,
        "inferred_grade": None,
        "grade_unit": "g/t Au",
        "source_url": None,
        "project_stage": None,
        "deposit_subtype": None,
        "mineralization_mode": None,
        "tectonic_belt": None,
        "metal_suite": None,
        "alteration_signature": None,
        "recovery_method": None,
        "mineralization_pattern": None,
        "host_rock_class": None,
        "project_stage_class": None,
        "mining_method_class": None,
        "resource_category_class": None,
        "resource_compliance_standard": None,
        "resource_vintage_year": None,
        "similarity_score": None,
        "legacy_mi_resource": True,
        "source": "analogs_table",
    }]


def test_get_analogs_respects_thin_latest_workflow(monkeypatch):
    latest = [{"name": "Brewery Creek", "tonnage_mt": 31.0, "grade_value": 1.0}]
    monkeypatch.setattr(supabase_ops, "get_client", lambda: _FakeClient(workflow=latest))

    analogs = supabase_ops.get_analogs("project-id")

    assert analogs == latest
