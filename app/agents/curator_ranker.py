"""Agent 3 — Curator & Ranker.

Takes Agent 2's ``RankedEvents`` (already cosine-ranked against the user's
preference embedding) plus the ``PreferenceProfile`` and the NWS weekend
forecast, and produces the ``FinalFeed`` the frontend actually renders: a
``final_score``/plain-language ``reason`` per event, and a short
``best_bets_this_weekend`` shortlist.

Follows the same conventions as Agents 1/2 (``preference_profiler.py`` /
``live_events.py`` / ``event_retriever.py``): one Hugging Face chat completion
via ``huggingface_hub.InferenceClient`` (same ``provider``/``model`` env vars),
a strict "respond with only JSON" system prompt, a resilient parser (direct
``json.loads`` then a regex fallback), grounding by construction (never trust
an ``event_id`` back from the model that wasn't in the input), and a
never-raise/never-500 contract — any failure anywhere degrades to a
similarity-ranked fallback feed built directly from ``RankedEvents``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, timezone
from typing import Any, List, Optional

from huggingface_hub import InferenceClient

from app.schemas.models import Event, FeedItem, FinalFeed, PreferenceProfile, RankedEvents

logger = logging.getLogger(__name__)

# Same provider + model as Agents 1 and 2 (notebooks/agent1_preference_profiler.ipynb).
DEFAULT_HF_PROVIDER = "publicai"
DEFAULT_HF_MODEL = "swiss-ai/Apertus-70B-Instruct-2509"

BEST_BETS_LIMIT = 3

# Matches preference_profiler.py's own fallback-profile default — reused here
# for consistency rather than inventing a second "default category" constant.
# Triggers when an event's org_id can't be matched against profile.orgs, which
# in practice means mock_events.json's stub events (their org_ids, e.g.
# "org_moma", never appear in a real PreferenceProfile) rather than live
# events, whose org_id is the same org list Agent 2 searched from.
FALLBACK_CATEGORY = "community_nonprofits"

SYSTEM_PROMPT = """You are the Curator & Ranker for NYC Event Scout — the final step before a \
user sees their event feed.

You will be given: the user's interest categories with weights, their own free-text description \
of their tastes, an NYC weekend weather forecast, and a list of candidate events already ranked \
by semantic similarity to the user's taste profile (each with an id, title, date, price, type, \
org, description, and similarity score).

For events worth recommending, do two things:
1. Assign a final_score from 0.0 to 1.0 — combine the given similarity score with your own \
judgment of relevance to the user's stated interests, how reasonable the date and price are, and \
(for outdoor or weekend events) the weather. You do not need to score every event given to you — \
you may leave out ones that are clearly stale, malformed, or a poor fit rather than scoring them low.
2. Write a short, plain-language, second-person reason (max ~25 words) a friend might say, for \
example: "You haven't been to a jazz show in a while and this one's outdoors on a clear night." \
Base any weather-related reasoning ONLY on the forecast given to you — never invent conditions.

Then pick up to 3 event ids as "best_bets_this_weekend" — your strongest picks that fall on the \
Friday/Saturday/Sunday covered by the forecast given to you. Leave the list empty if nothing this \
weekend is a strong fit. Never pick an event outside those weekend dates just to fill the list.

Respond with ONLY a single JSON object and nothing else — no markdown fences, no commentary before \
or after it. It must have exactly this shape:

{
  "scored": [{"event_id": "<id from input>", "final_score": <float 0-1>, "reason": "<string>"}, ...],
  "best_bets_this_weekend": ["<event_id>", ...]
}

Never invent an event_id that wasn't given to you in the candidate list.
"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_DATE_PREFIX_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
# Despite "respond with ONLY JSON, no commentary", the model sometimes adds
# inline "// explanation" comments or a trailing comma before a closing
# brace/bracket — both invalid JSON. Negative lookbehind on ":" keeps this
# from mangling "http://"/"https://" URLs in event links.
_JS_COMMENT_RE = re.compile(r"(?<!:)//[^\n]*")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


# --- lazy loader (patched in tests; keeps the weather HTTP call out of import) -


def _load_weather() -> dict:
    from app.agents.weather import get_weekend_forecast

    try:
        return get_weekend_forecast()
    except Exception:
        logger.exception("weather fetch failed; continuing without a forecast")
        return {"available": False, "periods": []}


# --- helpers --------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_event_date(value: Optional[str]) -> Optional[date]:
    """Best-effort date parse: pulls the first YYYY-MM-DD out of the string.

    Handles the caveats already documented for the ``date`` field (README:
    "Date field isn't validated as real ISO 8601") — literal "TBA", date
    ranges like "2023-07-29 to 2023-07-31", etc. Returns None rather than
    raising when nothing parseable is found.
    """
    if not value:
        return None
    match = _DATE_PREFIX_RE.search(value)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError("no JSON object found in model response")
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", _JS_COMMENT_RE.sub("", candidate))
    return json.loads(cleaned)


def _collect_completion_text(completion: Any) -> str:
    choice = completion.choices[0]
    return (choice.message.content or "").strip()


def _format_weather(weather: dict) -> str:
    if not weather or not weather.get("available"):
        return "(forecast unavailable)"
    lines = [
        f"{p.get('name')}: {p.get('short_forecast')}, {p.get('temperature')}°{p.get('temperature_unit', 'F')}"
        for p in weather.get("periods", [])
    ]
    return "\n".join(lines) if lines else "(forecast unavailable)"


def _event_context_line(event: Event) -> str:
    price_display = event.price if isinstance(event.price, str) else f"${event.price}"
    return (
        f"- id={event.event_id} | title={event.title!r} | date={event.date} | "
        f"price={price_display} | type={event.type or 'n/a'} | org={event.org or 'n/a'} | "
        f"similarity={event.similarity_score:.2f} | description={event.description or 'n/a'}"
    )


def _build_user_message(ranked_events: RankedEvents, profile: PreferenceProfile, weather: dict) -> str:
    categories_text = ", ".join(f"{c.name} ({c.weight:.1f})" for c in profile.categories) or "(none)"
    events_text = "\n".join(_event_context_line(e) for e in ranked_events.events)
    return (
        f"User categories (weight): {categories_text}\n"
        f"User's own words: {profile.raw_text or '(nothing provided)'}\n\n"
        f"Weekend weather forecast:\n{_format_weather(weather)}\n\n"
        f"Candidate events (already similarity-ranked, highest first):\n{events_text}"
    )


def _weekend_date_strings(weather: dict) -> set:
    return {p.get("date") for p in weather.get("periods", []) if p.get("date")} if weather else set()


def _org_category_lookup(profile: PreferenceProfile) -> dict:
    """org_id -> category, built from the profile's own org list."""
    return {org.org_id: org.category for org in profile.orgs}


def _event_category(event: Event, org_category_by_id: dict) -> str:
    """Resolve one event's category by joining its org_id against the
    profile's org list (built once by the caller via _org_category_lookup).

    Event itself carries no reliably-a-category-slug field to fall back to
    first: Event.type only holds a real category slug for live-search-sourced
    events (live_events.py sets it from org.category) — for mock_events.json's
    stub fallback it's a freeform event-type string ("exhibition", "concert",
    "outdoors", ...), not one of the five category slugs, so using it as an
    intermediate fallback would silently produce an invalid category on stub
    data. Falls straight to FALLBACK_CATEGORY when org_id isn't found.
    """
    return org_category_by_id.get(event.org_id, FALLBACK_CATEGORY)


def _build_feed_items(
    parsed: dict, ranked_events: RankedEvents, org_category_by_id: dict
) -> List[FeedItem]:
    """Grounding: only event_ids that were actually in the input survive — an
    id the model invented is silently dropped, same philosophy as Agents 1/2's
    link/name grounding checks."""
    events_by_id = {e.event_id: e for e in ranked_events.events}
    feed: List[FeedItem] = []
    for item in parsed.get("scored", []) or []:
        if not isinstance(item, dict):
            continue
        event = events_by_id.get(item.get("event_id"))
        if event is None:
            continue
        try:
            final_score = float(item.get("final_score", event.similarity_score))
        except (TypeError, ValueError):
            final_score = event.similarity_score
        final_score = max(0.0, min(1.0, final_score))
        reason = str(item.get("reason") or "Matches your stated interests.").strip()
        feed.append(
            FeedItem(
                event_id=event.event_id,
                title=event.title,
                date=event.date,
                location=event.location,
                price=event.price,
                link=event.link,
                final_score=final_score,
                reason=reason,
                category=_event_category(event, org_category_by_id),
            )
        )
    feed.sort(key=lambda f: f.final_score, reverse=True)
    return feed


def _weekend_shortlist(feed: List[FeedItem], weekend_dates: set, limit: int = BEST_BETS_LIMIT) -> List[str]:
    shortlist = []
    for item in feed:
        event_date = _parse_event_date(item.date)
        if event_date and (not weekend_dates or event_date.isoformat() in weekend_dates):
            shortlist.append(item.event_id)
        if len(shortlist) >= limit:
            break
    return shortlist


def _valid_best_bets(parsed: dict, feed: List[FeedItem], weather: dict) -> List[str]:
    feed_ids = {f.event_id: f for f in feed}
    weekend_dates = _weekend_date_strings(weather)

    candidates = []
    for event_id in parsed.get("best_bets_this_weekend", []) or []:
        item = feed_ids.get(event_id)
        if item is None:
            continue
        event_date = _parse_event_date(item.date)
        if weekend_dates and event_date and event_date.isoformat() not in weekend_dates:
            continue  # model picked something outside the actual forecast window
        candidates.append(event_id)

    if candidates:
        return candidates[:BEST_BETS_LIMIT]

    # Model returned nothing valid — compute our own weekend shortlist from the
    # already-scored feed rather than surfacing an empty section unnecessarily.
    return _weekend_shortlist(feed, weekend_dates)


def _cache_event_metadata(ranked_events: RankedEvents) -> None:
    """Best-effort: cache event metadata so a later accept/skip signal (which
    only carries event_id, per the locked Signal schema) can still be
    re-embedded by the RL loop. Never lets a cache failure break the feed."""
    try:
        from app.storage import save_event_cache_entries

        save_event_cache_entries(ranked_events.user_id, ranked_events.events)
    except Exception:
        logger.exception("event metadata cache write failed; continuing without it")


def _fallback_feed(
    ranked_events: RankedEvents, weather: dict, org_category_by_id: dict
) -> FinalFeed:
    """Used whenever the LLM path is unavailable or fails: rank purely by the
    similarity score Agent 2 already computed, with a generic reason."""
    feed = [
        FeedItem(
            event_id=e.event_id,
            title=e.title,
            date=e.date,
            location=e.location,
            price=e.price,
            link=e.link,
            final_score=e.similarity_score,
            reason="Ranked by similarity to your interests (curator step unavailable).",
            category=_event_category(e, org_category_by_id),
        )
        for e in ranked_events.events
    ]
    weekend_dates = _weekend_date_strings(weather)
    best_bets = _weekend_shortlist(feed, weekend_dates)
    return FinalFeed(
        user_id=ranked_events.user_id,
        generated_at=_now_iso(),
        feed=feed,
        best_bets_this_weekend=best_bets,
    )


# --- public API ------------------------------------------------------------------


def get_final_feed(
    ranked_events: RankedEvents,
    profile: PreferenceProfile,
    hf_token: Optional[str] = None,
    weather: Optional[dict] = None,
) -> FinalFeed:
    """Run Agent 3 for real: one Hugging Face chat completion that scores,
    filters, and explains the already-ranked events, plus a weekend shortlist.

    Falls back to a similarity-ranked feed on any failure — missing token,
    network error, malformed model output, etc. — rather than raising, so the
    endpoint never 500s, matching Agents 1 and 2's contract.
    """
    weather = weather if weather is not None else _load_weather()
    _cache_event_metadata(ranked_events)
    org_category_by_id = _org_category_lookup(profile)

    if not ranked_events.events:
        return FinalFeed(
            user_id=ranked_events.user_id, generated_at=_now_iso(), feed=[], best_bets_this_weekend=[]
        )

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set; returning similarity-ranked fallback feed")
        return _fallback_feed(ranked_events, weather, org_category_by_id)

    try:
        provider = os.environ.get("HF_PROVIDER", DEFAULT_HF_PROVIDER)
        client = InferenceClient(token=hf_token, provider=provider)
        model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)

        completion = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(ranked_events, profile, weather)},
            ],
            max_tokens=2048,
        )

        raw_output = _collect_completion_text(completion)
        if not raw_output:
            raise ValueError("model response contained no text content")

        parsed = _extract_json_object(raw_output)
        feed = _build_feed_items(parsed, ranked_events, org_category_by_id)
        if not feed:
            raise ValueError("model scored no valid events")

        best_bets = _valid_best_bets(parsed, feed, weather)

        return FinalFeed(
            user_id=ranked_events.user_id,
            generated_at=_now_iso(),
            feed=feed,
            best_bets_this_weekend=best_bets,
        )

    except Exception:
        logger.exception("Agent 3 (curator/ranker) failed; returning similarity-ranked fallback feed")
        return _fallback_feed(ranked_events, weather, org_category_by_id)
