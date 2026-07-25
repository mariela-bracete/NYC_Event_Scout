"""FastAPI entrypoint: serves the API plus the static frontend."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.agents.curator_ranker import get_final_feed
from app.agents.event_retriever import get_ranked_events
from app.agents.preference_profiler import build_preference_profile
from app.agents.rl_loop import apply_signal_batch
from app.schemas.models import (
    CuratorRankerRequest,
    FinalFeed,
    PreferenceProfile,
    PreferenceProfilerRequest,
    RankedEvents,
    SignalBatch,
)

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


# Mounted last so it never shadows the explicit API routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
