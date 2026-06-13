from __future__ import annotations

from datetime import date, datetime, timezone

from nodes import gold_resource_storage as storage


def test_get_parallel_cache_handles_empty_maybe_single(monkeypatch):
    class Query:
        def select(self, *args, **kwargs):
            return self

        def eq(self, *args, **kwargs):
            return self

        def maybe_single(self):
            return self

        def execute(self):
            return None

    class Client:
        def table(self, name):
            return Query()

    monkeypatch.setattr(storage, "get_client", lambda: Client())

    assert storage.get_parallel_cache("missing-cache-key") is None


def test_gold_storage_retries_transient_supabase_transport_error(monkeypatch):
    attempts = []
    resets = []

    class Response:
        data = [{"id": "project-1"}]

    class Query:
        def upsert(self, *args, **kwargs):
            return self

        def execute(self):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("httpx.ReadError: [Errno 49] Can't assign requested address")
            return Response()

    class Client:
        def table(self, name):
            return Query()

    monkeypatch.setattr(storage, "get_client", lambda: Client())
    monkeypatch.setattr(storage, "reset_thread_client", lambda: resets.append(1))
    monkeypatch.setattr(storage.time, "sleep", lambda _seconds: None)

    assert storage.upsert_gold_project({"id": "project-1"}) == {"id": "project-1"}
    assert len(attempts) == 2
    assert resets == [1]


def test_truth_cutoff_date_prefers_persisted_cutoff_then_truth_dates():
    assert storage.truth_cutoff_date({"cutoff_date": date(2024, 1, 2)}) == "2024-01-02"
    assert storage.truth_cutoff_date({"effective_date": datetime(2024, 3, 4, tzinfo=timezone.utc)}) == "2024-03-04"
    assert storage.truth_cutoff_date({"publication_date": "2024-05-06"}) == "2024-05-06"
    assert storage.truth_cutoff_date({}) is None
