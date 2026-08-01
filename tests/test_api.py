"""Offline tests for the spec'd public API routes in app/main.py
(POST /preferences, /search, /similar, /feedback; GET /profile, /weather).

Same conventions as the rest of the suite: every agent function is patched at
the app.main (or module-under-test) seam, storage writes go to a tmp_path via
patched STORAGE_DIR, and nothing touches the network, an LLM, or ChromaDB.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main
import app.storage as storage
from app.schemas.models import (
    Category,
    Event,
    FinalFeed,
    Org,
    PreferenceProfile,
    RankedEvents,
)

client = TestClient(main.app)


def _profile(user_id="user-1") -> PreferenceProfile:
    return PreferenceProfile(
        user_id=user_id,
        categories=[Category(name="arts_culture", weight=0.8)],
        orgs=[Org(org_id="org_1", name="MoMA", category="arts_culture", source="seeded")],
        raw_text="I love jazz",
        profile_embedding_seed="arts jazz museums",
    )


def _ranked(user_id="user-1") -> RankedEvents:
    return RankedEvents(
        user_id=user_id,
        events=[
            Event(
                event_id="evt_1",
                org_id="org_1",
                title="Late Night Jazz",
                date="2026-08-01",
                location="NYC",
                price="Free",
                link="https://example.com",
                similarity_score=0.9,
            )
        ],
    )


def _feed(user_id="user-1") -> FinalFeed:
    return FinalFeed(
        user_id=user_id,
        generated_at="2026-07-29T12:00:00+00:00",
        feed=[],
        best_bets_this_weekend=[],
    )


# --- POST /preferences ------------------------------------------------------


def test_preferences_returns_and_persists_profile(tmp_path):
    with (
        patch.object(main, "build_preference_profile", return_value=_profile()) as build,
        patch.object(storage, "STORAGE_DIR", tmp_path),
    ):
        response = client.post(
            "/preferences",
            json={"raw_text": "I love jazz", "selected_categories": ["arts_culture"]},
        )

    assert response.status_code == 200
    assert response.json()["user_id"] == "user-1"
    build.assert_called_once_with("I love jazz", ["arts_culture"])
    # Persisted in the locked schema, loadable for later /search-by-user_id.
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        stored = storage.load_profile("user-1")
    assert stored is not None and stored.raw_text == "I love jazz"


# --- POST /search -----------------------------------------------------------


def test_search_with_inline_profile_runs_full_pipeline():
    with (
        patch.object(main, "get_ranked_events", return_value=_ranked()) as retrieve,
        patch.object(main, "get_final_feed", return_value=_feed()) as curate,
    ):
        response = client.post("/search", json={"profile": _profile().model_dump()})

    assert response.status_code == 200
    assert response.json()["user_id"] == "user-1"
    retrieve.assert_called_once()
    # Agent 3 gets Agent 2's output plus the profile, in that order.
    (ranked_arg, profile_arg) = curate.call_args.args
    assert ranked_arg.events[0].event_id == "evt_1"
    assert profile_arg.user_id == "user-1"


def test_search_by_user_id_loads_stored_profile(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        storage.save_profile(_profile(user_id="user-2"))
        with (
            patch.object(main, "get_ranked_events", return_value=_ranked("user-2")),
            patch.object(main, "get_final_feed", return_value=_feed("user-2")),
        ):
            response = client.post("/search", json={"user_id": "user-2"})

    assert response.status_code == 200
    assert response.json()["user_id"] == "user-2"


def test_search_without_profile_or_known_user_404s(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        response = client.post("/search", json={"user_id": "nobody"})
    assert response.status_code == 404


# --- POST /feedback ---------------------------------------------------------


def test_feedback_matches_signals_behavior():
    batch = {
        "user_id": "user-1",
        "signals": [
            {"event_id": "evt_1", "action": "accept", "timestamp": "2026-07-29T12:00:00Z"}
        ],
    }
    result = {"logged": 1, "embedding_updated": True}
    with patch.object(main, "apply_signal_batch", return_value=result) as apply:
        response = client.post("/feedback", json=batch)

    assert response.status_code == 200
    assert response.json() == result
    # Same embedding_id convention as POST /signals: f"pref_{user_id}".
    assert apply.call_args.args[1] == "pref_user-1"
    # The stored profile (None here) is passed so the RL loop can seed a
    # baseline vector for users who have no stored embedding yet.
    assert "profile" in apply.call_args.kwargs


def test_signals_seeds_rl_baseline_from_stored_profile(tmp_path):
    batch = {
        "user_id": "user-1",
        "signals": [
            {"event_id": "evt_1", "action": "accept", "timestamp": "2026-07-29T12:00:00Z"}
        ],
    }
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        storage.save_profile(_profile())
        with patch.object(main, "apply_signal_batch", return_value={}) as apply:
            client.post("/signals", json=batch)

    assert apply.call_args.kwargs["profile"].user_id == "user-1"


# --- GET /profile -----------------------------------------------------------


def test_profile_returns_taste_data(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        storage.save_profile(_profile())
        response = client.get("/profile", params={"user_id": "user-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["stats"]["orgs_followed"] == 1
    # Declared interests back the breakdown when there are no signals yet.
    assert body["category_breakdown"][0]["category"] == "arts_culture"
    assert body["taste_type"]  # always some label, even heuristic


def test_profile_unknown_user_is_empty_not_500(tmp_path):
    with patch.object(storage, "STORAGE_DIR", tmp_path):
        response = client.get("/profile", params={"user_id": "nobody"})

    assert response.status_code == 200
    assert response.json()["available"] is False


# --- POST /similar ----------------------------------------------------------


def test_similar_delegates_and_never_500s():
    result = {"event_id": "evt_1", "available": True, "similar": []}
    with patch.object(main, "find_similar_events", return_value=result) as find:
        response = client.post("/similar", json={"event_id": "evt_1", "top_k": 3})

    assert response.status_code == 200
    assert response.json() == result
    find.assert_called_once_with("evt_1", 3, None)


def test_similar_unavailable_store_degrades_gracefully(tmp_path, monkeypatch):
    # Unpatched path: chromadb either isn't installed or (pointed at an empty
    # tmp store) has no events collection — either way the endpoint reports
    # unavailable, not 500.
    monkeypatch.setenv("EVENTS_CHROMA_PATH", str(tmp_path))
    response = client.post("/similar", json={"event_id": "evt_x"})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["similar"] == []


# --- GET /weather -----------------------------------------------------------


def test_weather_returns_forecast_shape():
    forecast = {"available": True, "periods": [{"name": "Saturday"}]}
    with patch.object(main, "get_weekend_forecast", return_value=forecast):
        response = client.get("/weather")

    assert response.status_code == 200
    assert response.json() == forecast
