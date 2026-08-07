"""Pydantic schemas shared across all NYC Event Scout agents.

All four endpoints (preference-profiler, event-retriever, curator-ranker,
signals) are live as of this phase.

Only one addition vs. the previously locked version: `CuratorRankerRequest`
— a new request-body model, additive-only, following the exact same pattern
as `PreferenceProfilerRequest` below. Nothing existing was renamed or
restructured; every other class is untouched.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

Price = Union[str, float]


# --- Agent 1 output: preference profile ---------------------------------


class Category(BaseModel):
    name: str
    weight: float = Field(ge=0.0, le=1.0)


class Org(BaseModel):
    org_id: str
    name: str
    category: str
    source: Literal["seeded", "user_added"]


class PreferenceProfile(BaseModel):
    user_id: str
    categories: List[Category]
    orgs: List[Org]
    raw_text: str
    profile_embedding_seed: str
    # Optional pointer to a precomputed user-preference vector in the ChromaDB
    # `user_preferences` collection (e.g. "pref_test_user_001"). When present and
    # found, Agent 2 uses that stored vector; otherwise it embeds
    # profile_embedding_seed at query time. Optional keeps this backward-compatible
    # with Agent 1, which does not populate it.
    embedding_id: Optional[str] = None


# --- Agent 2 output: ranked events --------------------------------------


class Event(BaseModel):
    event_id: str
    org_id: str
    title: str
    date: str  # ISO 8601
    location: str
    price: Price
    link: str
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Fields Agent 2 embeds on (title + description + type + org). Optional with
    # empty defaults so existing mock/output events that omit them still validate.
    description: str = ""
    type: str = ""
    org: str = ""


class RankedEvents(BaseModel):
    user_id: str
    events: List[Event]


# --- Agent 3 output: final feed -------------------------------------------


class FeedItem(BaseModel):
    event_id: str
    title: str
    date: str  # ISO 8601
    location: str
    price: Price
    link: str
    final_score: float
    reason: str
    # One of the 5 category slugs (arts_culture, parks_outdoors, nightlife_bars,
    # food_restaurants, community_nonprofits) — resolved in curator_ranker.py by
    # joining the event's org_id back against the PreferenceProfile.orgs list
    # that produced it (Event itself carries no reliable category field; see
    # curator_ranker.py's _event_category for why). Falls back to
    # "community_nonprofits" when org_id can't be matched (e.g. the
    # mock_events.json stub, whose org_ids never appear in a real profile).
    category: str


class FinalFeed(BaseModel):
    user_id: str
    generated_at: str  # ISO 8601
    feed: List[FeedItem]
    best_bets_this_weekend: List[str]


# --- Accept/skip signals ----------------------------------------------------


class Signal(BaseModel):
    event_id: str
    action: Literal["accept", "skip"]
    timestamp: str  # ISO 8601


class SignalBatch(BaseModel):
    user_id: str
    signals: List[Signal]


# --- API request bodies ---------------------------------------------------


class PreferenceProfilerRequest(BaseModel):
    raw_text: str = ""
    selected_categories: List[str] = Field(default_factory=list)


class CuratorRankerRequest(BaseModel):
    """Request body for POST /agents/curator-ranker — Agent 3's input is the
    profile (for category weights + raw text) plus Agent 2's ranked events."""

    profile: PreferenceProfile
    ranked_events: RankedEvents


class SearchRequest(BaseModel):
    """Request body for POST /search — the full pipeline (Agent 2 retrieval +
    RAG ranking, then Agent 3 curation) in one call.

    Send either an inline ``profile`` (the object POST /preferences returned)
    or just a ``user_id`` whose profile was previously persisted by
    POST /preferences. Inline profile wins when both are present."""

    profile: Optional[PreferenceProfile] = None
    user_id: Optional[str] = None


class SimilarRequest(BaseModel):
    """Request body for POST /similar — nearest events to one the user liked.

    ``user_id`` is optional; when present, results are enriched with
    title/org/type from that user's event metadata cache."""

    event_id: str
    user_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=25)
