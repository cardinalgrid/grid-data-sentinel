"""fetch_isd_lite gives up cleanly when the archive cannot be reached."""
import requests

from grid_sentinel import weather


def test_fetch_returns_none_when_the_archive_is_unreachable(tmp_path, monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise requests.ConnectTimeout("no route")

    monkeypatch.setattr(weather.requests, "get", boom)
    monkeypatch.setattr(weather.time, "sleep", lambda s: None)
    assert weather.fetch_isd_lite("724080", "13739", 2026, tmp_path) is None
    assert len(calls) == 3
    assert not list(tmp_path.iterdir())


def test_fetch_returns_none_on_404_without_retrying(tmp_path, monkeypatch):
    class R:
        status_code = 404
        content = b""

    calls = []
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: calls.append(1) or R())
    assert weather.fetch_isd_lite("724080", "13739", 2026, tmp_path) is None
    assert len(calls) == 1
