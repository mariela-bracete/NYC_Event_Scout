"""Flat JSON storage helpers Agent 3 depends on ahead of Vivek's shared
session-storage layer landing. Minimal, one-file-per-user, easy to swap out
later for a real store — not meant to be final infra, just enough for the RL
loop and signals endpoint to work end-to-end today.

Two files per user under ``storage/`` (same directory the committed
``storage/test_user_001_profile.json`` fixture already lives in):

- ``signals_<user_id>.json`` — append-only list of accept/skip ``Signal``
  dicts. The audit trail / replay source for the RL loop.
- ``event_cache_<user_id>.json`` — ``event_id -> {title, description, type,
  org}``, refreshed every time ``curator_ranker.get_final_feed`` produces a
  feed. Needed because the locked ``Signal`` schema only carries
  ``event_id``/``action``/``timestamp`` — without this cache, a signal arriving
  later (after the search session that produced the event) would have nothing
  to re-embed against.

Judgment call: unlike the committed seed fixture, these per-user files are
generated interaction data, not seed data — they should probably be added to
``.gitignore`` (not done here since Agent 3 doesn't own that file; flagged in
``docs/agent3_integration_handoff.md`` for Vivek).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from typing import Optional

from app.schemas.models import Event, PreferenceProfile, Signal

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
STORAGE_DIR = REPO_ROOT / "storage"


def _signals_path(user_id: str) -> Path:
    return STORAGE_DIR / f"signals_{user_id}.json"


def _profile_path(user_id: str) -> Path:
    # Same naming pattern as the committed storage/users/test_user_001_profile.json
    # prototype fixture, but files written here use the locked PreferenceProfile
    # schema from app/schemas/models.py, not the fixture's earlier prototype shape.
    return STORAGE_DIR / "users" / f"{user_id}_profile.json"


def _event_cache_path(user_id: str) -> Path:
    return STORAGE_DIR / f"event_cache_{user_id}.json"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("failed reading %s; treating as empty", path)
        return default


def _write_json(path: Path, data) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_signal(user_id: str, signal: Signal) -> None:
    """Append one Signal to that user's flat JSON log. Never raises on a read
    failure of a corrupted existing file — starts a fresh log instead, since
    losing history is better than losing the ability to record new signals."""
    path = _signals_path(user_id)
    signals = _read_json(path, [])
    signals.append(signal.model_dump())
    _write_json(path, signals)


def load_signals(user_id: str) -> List[dict]:
    return _read_json(_signals_path(user_id), [])


def save_event_cache_entries(user_id: str, events: Iterable[Event]) -> None:
    """Upsert event metadata into that user's cache, keyed by event_id.
    Additive — never drops entries from prior feeds, since a signal for an
    older event may still arrive later."""
    path = _event_cache_path(user_id)
    cache: Dict[str, dict] = _read_json(path, {})
    for event in events:
        cache[event.event_id] = {
            "title": event.title,
            "description": event.description,
            "type": event.type,
            "org": event.org,
        }
    _write_json(path, cache)


def load_event_cache(user_id: str) -> Dict[str, dict]:
    return _read_json(_event_cache_path(user_id), {})


def save_profile(profile: PreferenceProfile) -> None:
    """Persist a user's PreferenceProfile (locked schema) to flat JSON so a
    later session can search or build a taste profile without re-running
    Agent 1. Called by POST /preferences."""
    path = _profile_path(profile.user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, indent=2)


def load_profile(user_id: str) -> Optional[PreferenceProfile]:
    """Load a persisted PreferenceProfile, or None if absent/unparseable.
    Files in the earlier prototype shape (e.g. the committed
    test_user_001_profile.json fixture) don't validate against the locked
    schema and are treated as absent rather than raising."""
    data = _read_json(_profile_path(user_id), None)
    if not data:
        return None
    try:
        return PreferenceProfile(**data)
    except Exception:
        logger.warning("stored profile for %s doesn't match locked schema; ignoring", user_id)
        return None
