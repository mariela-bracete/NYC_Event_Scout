"""Offline unit tests for app/storage.py.

Every test points STORAGE_DIR at a pytest tmp_path so nothing touches the
real repo's storage/ directory (which holds the committed
test_user_001_profile.json fixture).
"""

from unittest.mock import patch

import app.storage as storage
from app.schemas.models import Event, Signal


def _signal(event_id="evt_1", action="accept", timestamp="2026-07-25T12:00:00+00:00") -> Signal:
    return Signal(event_id=event_id, action=action, timestamp=timestamp)


def _event(event_id="evt_1", **overrides) -> Event:
    base = dict(
        event_id=event_id,
        org_id="org_1",
        title="Late Night Jazz",
        date="2026-07-25",
        location="NYC",
        price="Free",
        link="https://example.com",
        description="intimate set",
        type="nightlife_bars",
        org="The Village Vanguard",
    )
    base.update(overrides)
    return Event(**base)


def test_append_signal_creates_and_appends(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        storage.append_signal("user-1", _signal(event_id="evt_1"))
        storage.append_signal("user-1", _signal(event_id="evt_2", action="skip"))

        logged = storage.load_signals("user-1")

    assert len(logged) == 2
    assert logged[0]["event_id"] == "evt_1"
    assert logged[1]["action"] == "skip"


def test_signals_are_scoped_per_user(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        storage.append_signal("user-a", _signal())
        assert storage.load_signals("user-b") == []
        assert len(storage.load_signals("user-a")) == 1


def test_load_signals_missing_file_returns_empty_list(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        assert storage.load_signals("nobody") == []


def test_save_event_cache_entries_upserts_by_event_id(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        storage.save_event_cache_entries("user-1", [_event(event_id="evt_1", title="First")])
        storage.save_event_cache_entries(
            "user-1", [_event(event_id="evt_1", title="Updated"), _event(event_id="evt_2")]
        )

        cache = storage.load_event_cache("user-1")

    assert set(cache.keys()) == {"evt_1", "evt_2"}
    assert cache["evt_1"]["title"] == "Updated"


def test_load_event_cache_missing_file_returns_empty_dict(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        assert storage.load_event_cache("nobody") == {}


def test_corrupted_signals_file_treated_as_empty_not_raised(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        path = storage._signals_path("user-1")
        path.write_text("{not valid json", encoding="utf-8")

        assert storage.load_signals("user-1") == []
        # Recovery: a subsequent append should still work, overwriting the
        # corrupted file rather than erroring.
        storage.append_signal("user-1", _signal())
        assert len(storage.load_signals("user-1")) == 1
