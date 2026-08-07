"""Offline unit tests for Agent 3 (app/agents/curator_ranker.py).

InferenceClient, the weather loader, and the event-cache writer are all
patched — these tests never hit the network, HF, or the real filesystem
storage directory.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app.agents.curator_ranker as cr
from app.schemas.models import Category, Event, FinalFeed, Org, PreferenceProfile, RankedEvents


def _fake_completion(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _event(**overrides) -> Event:
    base = dict(
        event_id="evt_1",
        org_id="org_1",
        title="Late Night Jazz Set",
        date="2026-07-25",
        location="The Village Vanguard, New York, NY",
        price="$35",
        link="https://villagevanguard.com/",
        similarity_score=0.8,
        description="Intimate late-night jazz.",
        type="nightlife_bars",
        org="The Village Vanguard",
    )
    base.update(overrides)
    return Event(**base)


def _ranked_events(events=None, **overrides) -> RankedEvents:
    base = dict(user_id="user-xyz", events=events if events is not None else [_event()])
    base.update(overrides)
    return RankedEvents(**base)


def _profile(**overrides) -> PreferenceProfile:
    base = dict(
        user_id="user-xyz",
        categories=[Category(name="nightlife_bars", weight=0.9)],
        orgs=[Org(org_id="org_1", name="The Village Vanguard", category="nightlife_bars", source="seeded")],
        raw_text="I love jazz but hate crowds",
        profile_embedding_seed="I love jazz but hate crowds | nightlife_bars",
    )
    base.update(overrides)
    return PreferenceProfile(**base)


_WEEKEND_WEATHER = {
    "available": True,
    "periods": [
        {"name": "Friday", "date": "2026-07-24", "short_forecast": "Clear", "temperature": 78, "temperature_unit": "F"},
        {"name": "Saturday", "date": "2026-07-25", "short_forecast": "Clear", "temperature": 82, "temperature_unit": "F"},
        {"name": "Sunday", "date": "2026-07-26", "short_forecast": "Rain", "temperature": 70, "temperature_unit": "F"},
    ],
}


@patch("app.agents.curator_ranker._cache_event_metadata")
@patch("app.agents.curator_ranker.InferenceClient")
def test_scores_and_ranks_events_via_llm(mock_inference_cls, mock_cache):
    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion(
        json.dumps(
            {
                "scored": [
                    {
                        "event_id": "evt_1",
                        "final_score": 0.95,
                        "reason": "Outdoor jazz on a clear night, right up your alley.",
                    }
                ],
                "best_bets_this_weekend": ["evt_1"],
            }
        )
    )

    result = cr.get_final_feed(
        _ranked_events(), _profile(), hf_token="test-token", weather=_WEEKEND_WEATHER
    )

    assert isinstance(result, FinalFeed)
    assert result.user_id == "user-xyz"
    assert len(result.feed) == 1
    assert result.feed[0].event_id == "evt_1"
    assert result.feed[0].final_score == 0.95
    assert "jazz" in result.feed[0].reason.lower()
    assert result.feed[0].category == "nightlife_bars"  # joined via org_1 -> profile.orgs
    assert result.best_bets_this_weekend == ["evt_1"]
    mock_cache.assert_called_once()


@patch("app.agents.curator_ranker._cache_event_metadata")
@patch("app.agents.curator_ranker.InferenceClient")
def test_drops_invented_event_ids(mock_inference_cls, mock_cache):
    """Grounding: an event_id the model made up that wasn't in the input is
    silently dropped rather than trusted."""
    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion(
        json.dumps(
            {
                "scored": [
                    {"event_id": "evt_1", "final_score": 0.7, "reason": "Good fit."},
                    {"event_id": "evt_made_up", "final_score": 0.99, "reason": "Invented."},
                ],
                "best_bets_this_weekend": [],
            }
        )
    )

    result = cr.get_final_feed(
        _ranked_events(), _profile(), hf_token="test-token", weather=_WEEKEND_WEATHER
    )

    assert len(result.feed) == 1
    assert result.feed[0].event_id == "evt_1"


@patch("app.agents.curator_ranker._cache_event_metadata")
def test_falls_back_when_no_hf_token(mock_cache):
    result = cr.get_final_feed(
        _ranked_events(), _profile(), hf_token=None, weather=_WEEKEND_WEATHER
    )

    assert isinstance(result, FinalFeed)
    assert len(result.feed) == 1
    assert result.feed[0].final_score == 0.8  # falls back to similarity_score
    assert "unavailable" in result.feed[0].reason.lower()
    assert result.feed[0].category == "nightlife_bars"  # category join happens on both paths


@patch("app.agents.curator_ranker._cache_event_metadata")
def test_unmatched_org_id_falls_back_to_default_category(mock_cache):
    """An event whose org_id isn't in the profile's orgs list (e.g. mock_events.json
    stub data) gets FALLBACK_CATEGORY rather than crashing or leaving category blank."""
    result = cr.get_final_feed(
        _ranked_events(events=[_event(org_id="org_unknown")]),
        _profile(),
        hf_token=None,
        weather=_WEEKEND_WEATHER,
    )

    assert result.feed[0].category == cr.FALLBACK_CATEGORY


@patch("app.agents.curator_ranker._cache_event_metadata")
@patch("app.agents.curator_ranker.InferenceClient")
def test_falls_back_when_llm_call_raises(mock_inference_cls, mock_cache):
    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.side_effect = RuntimeError("simulated API failure")

    result = cr.get_final_feed(
        _ranked_events(), _profile(), hf_token="test-token", weather=_WEEKEND_WEATHER
    )

    assert isinstance(result, FinalFeed)
    assert len(result.feed) == 1
    assert result.feed[0].final_score == 0.8


@patch("app.agents.curator_ranker._cache_event_metadata")
@patch("app.agents.curator_ranker.InferenceClient")
def test_falls_back_when_model_scores_nothing_valid(mock_inference_cls, mock_cache):
    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion(
        json.dumps({"scored": [], "best_bets_this_weekend": []})
    )

    result = cr.get_final_feed(
        _ranked_events(), _profile(), hf_token="test-token", weather=_WEEKEND_WEATHER
    )

    assert len(result.feed) == 1
    assert result.feed[0].final_score == 0.8


@patch("app.agents.curator_ranker._cache_event_metadata")
def test_empty_ranked_events_short_circuits_without_llm_call(mock_cache):
    with patch("app.agents.curator_ranker.InferenceClient") as mock_inference_cls:
        result = cr.get_final_feed(
            _ranked_events(events=[]), _profile(), hf_token="test-token", weather=_WEEKEND_WEATHER
        )
        mock_inference_cls.assert_not_called()

    assert result.feed == []
    assert result.best_bets_this_weekend == []


@patch("app.agents.curator_ranker._cache_event_metadata")
@patch("app.agents.curator_ranker.InferenceClient")
def test_best_bets_outside_weekend_window_are_rejected(mock_inference_cls, mock_cache):
    """The model picking a non-weekend event as a 'best bet' gets overridden by
    our own weekend-only shortlist computed from the scored feed."""
    events = [
        _event(event_id="evt_weekday", date="2026-07-29"),  # a Wednesday, not in weekend window
        _event(event_id="evt_weekend", date="2026-07-25", title="Saturday Jazz"),
    ]
    mock_client = MagicMock()
    mock_inference_cls.return_value = mock_client
    mock_client.chat_completion.return_value = _fake_completion(
        json.dumps(
            {
                "scored": [
                    {"event_id": "evt_weekday", "final_score": 0.9, "reason": "Great fit."},
                    {"event_id": "evt_weekend", "final_score": 0.6, "reason": "Also good."},
                ],
                "best_bets_this_weekend": ["evt_weekday"],
            }
        )
    )

    result = cr.get_final_feed(
        _ranked_events(events=events), _profile(), hf_token="test-token", weather=_WEEKEND_WEATHER
    )

    assert result.best_bets_this_weekend == ["evt_weekend"]


def test_parse_event_date_handles_known_caveats():
    assert cr._parse_event_date("2026-07-25T18:00:00-04:00").isoformat() == "2026-07-25"
    assert cr._parse_event_date("2023-07-29 to 2023-07-31").isoformat() == "2023-07-29"
    assert cr._parse_event_date("TBA") is None
    assert cr._parse_event_date("") is None
    assert cr._parse_event_date(None) is None
