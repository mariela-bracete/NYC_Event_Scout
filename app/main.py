"""FastAPI entrypoint: serves the API plus the static frontend."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.agents.curator_ranker import get_final_feed
from app.agents.event_retriever import get_ranked_events
from app.agents.preference_profiler import build_preference_profile
from app.agents.rl_loop import apply_signal_batch
from app.agents.taste_profile import build_taste_profile
from app.agents.weather import get_weekend_forecast
from app.schemas.models import (
    CuratorRankerRequest,
    FinalFeed,
    PreferenceProfile,
    PreferenceProfilerRequest,
    RankedEvents,
    SearchRequest,
    SignalBatch,
    SimilarRequest,
)
from app.similar import find_similar_events
from app.storage import load_profile, save_profile

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="NYC Event Scout")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agents/preference-profiler", response_model=PreferenceProfile)
def preference_profiler(request: PreferenceProfilerRequest) -> PreferenceProfile:
    return build_preference_profile(request.raw_text, request.selected_categories)


@app.post("/agents/event-retriever", response_model=RankedEvents)
def event_retriever(profile: PreferenceProfile) -> RankedEvents:
    return get_ranked_events(profile)


@app.post("/agents/curator-ranker", response_model=FinalFeed)
def curator_ranker(request: CuratorRankerRequest) -> FinalFeed:
    return get_final_feed(request.ranked_events, request.profile)


@app.post("/signals")
def signals(batch: SignalBatch) -> dict:
    # Convention until Agent 1 persists a stable embedding_id per user (see
    # docs/agent3_integration_handoff.md): f"pref_{user_id}".
    embedding_id = f"pref_{batch.user_id}"
    return apply_signal_batch(batch, embedding_id)


# --- Spec'd public API (proposal endpoint names) ---------------------------
#
# The /agents/* routes above are the per-agent seams each owner integrates
# against (and what the current frontend calls). The routes below are the
# endpoint names promised in the proposal / project breakdown; they compose
# the same agent functions rather than duplicating logic.


@app.post("/preferences", response_model=PreferenceProfile)
def preferences(request: PreferenceProfilerRequest) -> PreferenceProfile:
    """Agent 1, plus persistence: the returned profile is saved to flat JSON
    so /search and /profile can later work from just a user_id."""
    profile = build_preference_profile(request.raw_text, request.selected_categories)
    save_profile(profile)
    return profile


@app.post("/search", response_model=FinalFeed)
def search(request: SearchRequest) -> FinalFeed:
    """Full pipeline: Agent 2 (retrieval + RAG ranking) -> Agent 3 (curation,
    weather-aware best bets) -> final ranked feed."""
    profile = request.profile
    if profile is None and request.user_id:
        profile = load_profile(request.user_id)
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="No profile: send one inline or POST /preferences first for this user_id.",
        )
    ranked = get_ranked_events(profile)
    return get_final_feed(ranked, profile)


@app.post("/similar")
def similar(request: SimilarRequest) -> dict:
    """Nearest events (cosine) to one the user liked, from the shared events
    vector store Agent 2 populates during /search."""
    return find_similar_events(request.event_id, request.top_k, request.user_id)


@app.post("/feedback")
def feedback(batch: SignalBatch) -> dict:
    """Proposal name for the accept/skip signal endpoint — same behavior as
    POST /signals (which the RL loop integration already targets)."""
    return signals(batch)


@app.get("/profile")
def profile(user_id: str) -> dict:
    """Taste-profile data (category breakdown, interest graph, activity
    heatmap, taste-type label) assembled from persisted interaction data."""
    return build_taste_profile(user_id)


@app.get("/weather")
def weather() -> dict:
    """NYC weekend forecast from the National Weather Service (no key needed) —
    the same data Agent 3 folds into its ranking."""
    return get_weekend_forecast()


# Mounted last so it never shadows the explicit API routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
