"""Agent 3 — lightweight, RLHF-inspired preference loop.

No RL framework — just an embedding update weighted by a reward signal, per
the handoff brief: accepted events nudge the user's stored preference vector
toward the event's embedding, skipped events nudge it away, both scaled by a
learning rate, then renormalized (the vector space is cosine, so it should
stay on the unit sphere).

Storage split, mirroring Agent 2's read/write split for ``chroma_events/`` vs.
committed ``chroma/``:

- READ path (``USER_PREF_CHROMA_PATH``, defaults to committed ``chroma/``) —
  same env var Agent 2 already reads. This module treats it as read-only,
  same as ``docs/spoorthy_integration_handoff.md`` #12 instructs for Agent 2:
  the committed store holds only the seeded ``pref_test_user_001`` vector and
  should never be overwritten by app code.
- WRITE path (``USER_PREF_WRITE_CHROMA_PATH``, defaults to a new gitignored
  ``chroma_user_prefs/``) — where every real embedding update actually lands.

On a user's first update, if the writable store has nothing yet for their
``embedding_id``, this seeds it from whatever the read path already has
(e.g. the one seeded test vector), or — if nothing exists anywhere — embeds
``profile.profile_embedding_seed`` as the starting point.

**Coordination needed, not resolved unilaterally here**: for Agent 2's
``_resolve_user_vector`` to ever see an *updated* vector instead of always
falling back to embedding the seed text fresh, two things have to happen
outside this module: (1) Agent 1 needs to set ``PreferenceProfile.embedding_id``
to a stable per-user id — this module assumes the convention
``f"pref_{user_id}"`` until the team agrees on one — and (2) whichever
``USER_PREF_CHROMA_PATH`` Agent 2 reads from in a real deployment needs to
point at this module's writable store (or the two stores need to be merged/
synced), not the pristine committed one. Flagged in
``docs/agent3_integration_handoff.md``.

Event re-embedding: the locked ``Signal`` schema only carries
``event_id``/``action``/``timestamp`` (no title/description/etc.), so this
module leans on ``app.storage``'s event metadata cache — written by
``curator_ranker.get_final_feed`` every time a feed goes out — to look up
what to re-embed for a given ``event_id`` when a signal arrives later,
possibly in a separate request.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from app.schemas.models import PreferenceProfile, Signal, SignalBatch
from app.storage import append_signal, load_event_cache

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent

READ_CHROMA_PATH = REPO_ROOT / "chroma"  # committed, read-only seed store
DEFAULT_WRITE_CHROMA_PATH = REPO_ROOT / "chroma_user_prefs"  # gitignored, writable

USER_PREF_COLLECTION = "user_preferences"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LEARNING_RATE = 0.15

# Cached embedder; loaded once on first real use (mirrors event_retriever.py).
_embedder = None


def _read_chroma_path() -> str:
    return os.environ.get("USER_PREF_CHROMA_PATH", str(READ_CHROMA_PATH))


def _write_chroma_path() -> str:
    return os.environ.get("USER_PREF_WRITE_CHROMA_PATH", str(DEFAULT_WRITE_CHROMA_PATH))


# --- lazy loaders (patched in tests; keep heavy imports out of module import) -


def _load_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _load_chroma_client(path: str):
    import chromadb

    return chromadb.PersistentClient(path=str(path))


# --- helpers ------------------------------------------------------------------


def _get_stored_vector(client, embedding_id: str) -> Optional[List[float]]:
    try:
        collection = client.get_collection(USER_PREF_COLLECTION)
    except Exception:
        return None
    got = collection.get(ids=[embedding_id], include=["embeddings"])
    embeddings = got.get("embeddings")
    # ChromaDB can return embeddings as a numpy array — never use truthiness on
    # it directly (same "ambiguous truth value" pitfall event_retriever.py
    # documents), always check length explicitly.
    if embeddings is None or len(embeddings) == 0:
        return None
    vector = embeddings[0]
    if vector is None or len(vector) == 0:
        return None
    return [float(x) for x in vector]


def _l2_normalize(vector: List[float]) -> List[float]:
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def _resolve_baseline_vector(
    embedding_id: str, profile: Optional[PreferenceProfile], embedder
) -> List[float]:
    """Writable store first, then the read-only seed store, then embed the
    profile's seed text as a last resort (a brand-new user with nothing
    stored anywhere yet)."""
    write_client = _load_chroma_client(_write_chroma_path())
    vector = _get_stored_vector(write_client, embedding_id)
    if vector is not None:
        return vector

    try:
        read_client = _load_chroma_client(_read_chroma_path())
        vector = _get_stored_vector(read_client, embedding_id)
        if vector is not None:
            return vector
    except Exception:
        logger.exception("could not read seed vector for %r", embedding_id)

    if profile is not None:
        seed = profile.profile_embedding_seed or profile.raw_text or ""
        return embedder.encode(seed).tolist()

    raise ValueError(f"no baseline vector available for {embedding_id!r} and no profile to seed one")


def _event_vector(event_id: str, user_id: str, embedder) -> Optional[List[float]]:
    """Re-embeds the event a signal refers to, using the same text contract
    Agent 2 embeds on (title + description + type + org), sourced from the
    cache curator_ranker writes at feed-generation time."""
    from app.agents.event_retriever import event_embedding_text

    metadata = load_event_cache(user_id).get(event_id)
    if metadata is None:
        logger.warning("no cached metadata for event_id=%r; cannot update vector for it", event_id)
        return None
    text = event_embedding_text(metadata)
    if not text:
        return None
    return embedder.encode(text).tolist()


# --- public API ------------------------------------------------------------------


def apply_signal(
    signal: Signal,
    user_id: str,
    embedding_id: str,
    profile: Optional[PreferenceProfile] = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> bool:
    """Apply one accept/skip signal to the stored preference vector.

    Always logs the raw signal first (audit trail), independent of whether the
    embedding update itself succeeds — a failed vector update should never
    lose the signal record.

    Returns True if the embedding was actually updated, False if it only
    logged the signal (e.g. no cached event metadata to re-embed, or any
    other failure along the way — this never raises).
    """
    append_signal(user_id, signal)

    try:
        embedder = _load_embedder()
        event_vector = _event_vector(signal.event_id, user_id, embedder)
        if event_vector is None:
            return False

        baseline = _resolve_baseline_vector(embedding_id, profile, embedder)
        if len(baseline) != len(event_vector):
            logger.warning(
                "vector dim mismatch for %r (%d vs %d); skipping update",
                embedding_id,
                len(baseline),
                len(event_vector),
            )
            return False

        direction = 1.0 if signal.action == "accept" else -1.0
        updated = _l2_normalize(
            [b + direction * learning_rate * (e - b) for b, e in zip(baseline, event_vector)]
        )

        write_client = _load_chroma_client(_write_chroma_path())
        collection = write_client.get_or_create_collection(
            name=USER_PREF_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        collection.upsert(ids=[embedding_id], embeddings=[updated])
        return True

    except Exception:
        logger.exception("preference embedding update failed for user_id=%r", user_id)
        return False


def apply_signal_batch(
    batch: SignalBatch,
    embedding_id: str,
    profile: Optional[PreferenceProfile] = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict:
    """Apply every signal in a batch sequentially, each nudging the vector a
    little further. Simple and easy to reason about, at the cost of being
    order-sensitive — an alternative would average all deltas into one update;
    noted as a judgment call, not implemented, since sequential application
    matches the "lightweight, no RL framework" brief most directly.
    """
    updated_count = 0
    for signal in batch.signals:
        if apply_signal(signal, batch.user_id, embedding_id, profile, learning_rate):
            updated_count += 1
    return {"received": len(batch.signals), "embedding_updates": updated_count}
