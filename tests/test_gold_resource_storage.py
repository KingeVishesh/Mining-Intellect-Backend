from __future__ import annotations

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
