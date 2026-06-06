from __future__ import annotations

from nodes import inferred_extractor


def test_mi_basis_prefers_explicit_measured_plus_indicated():
    basis = inferred_extractor._consensus_mi_basis(
        {"mi_category_basis": "measured_only"},
        {"mi_category_basis": "Measured + Indicated"},
    )

    assert basis == "measured_plus_indicated"


def test_measured_only_basis_drops_mi_values(monkeypatch):
    passes = iter([
        {
            "mi_tonnage_mt": 69.7,
            "mi_grade": 1.41,
            "mi_category_basis": "measured_only",
            "inferred_tonnage_mt": 44.5,
            "inferred_grade": 0.62,
            "confidence": "medium",
        },
        {
            "mi_tonnage_mt": 69.7,
            "mi_grade": 1.41,
            "mi_category_basis": "unknown",
            "inferred_tonnage_mt": 44.5,
            "inferred_grade": 0.62,
            "confidence": "medium",
        },
    ])

    monkeypatch.setattr(inferred_extractor.settings, "exa_api_key", "test-key")
    monkeypatch.setattr(
        inferred_extractor,
        "_single_query",
        lambda *args, **kwargs: next(passes),
    )

    result = inferred_extractor.extract_inferred_breakdown("Rogue", "gold")

    assert result is not None
    assert result["mi_category_basis"] == "measured_only"
    assert result["mi_tonnage_mt"] is None
    assert result["mi_grade"] is None
    assert result["inferred_tonnage_mt"] == 44.5
    assert result["inferred_grade"] == 0.62
    assert result["confidence"] == "low"


def test_single_query_requests_mi_category_basis(monkeypatch):
    posted_payloads = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"answer": {"confidence": "medium"}}

    def fake_post(*args, **kwargs):
        posted_payloads.append(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr(inferred_extractor.requests, "post", fake_post)

    inferred_extractor._single_query("test-key", "query", "prompt", "Analog")

    schema = posted_payloads[0]["output_schema"]["properties"]
    assert "mi_category_basis" in schema
    assert "standalone Measured row" in schema["mi_category_basis"]["description"]
