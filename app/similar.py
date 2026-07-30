"""POST /similar — "more like this" over Agent 2's event vectors.

Given an event_id that Agent 2 has already embedded into the ChromaDB
``events`` collection (which persists across requests in ``chroma_events/``),
returns the nearest other events by cosine similarity. When a ``user_id`` is
provided, results are enriched with title/org/type from that user's event
metadata cache (written by Agent 3's curator on every feed).

Reuses Agent 2's lazy Chroma loader and collection constants rather than
duplicating the access pattern — same shared-store convention documented in
the README. Read-only: never writes to any collection.

Never 500s: any failure (chromadb not installed, collection missing, unknown
event_id) returns an ``available: False`` response with a reason.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.agents.event_retriever import (
    EVENTS_COLLECTION,
    _events_chroma_path,
    _load_chroma_client,
)
from app.storage import load_event_cache

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5


def _unavailable(event_id: str, reason: str) -> Dict:
    return {"event_id": event_id, "available": False, "similar": [], "reason": reason}


def find_similar_events(
    event_id: str, top_k: int = DEFAULT_TOP_K, user_id: Optional[str] = None
) -> Dict:
    """Nearest events to ``event_id`` in the shared ``events`` collection.

    Returns:
        {
          "event_id": str,
          "available": bool,
          "similar": [{"event_id": str, "similarity_score": float,
                        "title": str, "org": str, "type": str, "snippet": str}],
          "reason": str,  # only when available is False
        }
    """
    try:
        client = _load_chroma_client(_events_chroma_path())
        collection = client.get_collection(EVENTS_COLLECTION)

        got = collection.get(ids=[event_id], include=["embeddings"])
        embeddings = got.get("embeddings")
        # ChromaDB returns numpy arrays — no truthiness checks (ambiguous).
        if embeddings is None or len(embeddings) == 0:
            return _unavailable(event_id, f"event_id {event_id!r} not found in events collection")
        anchor = [float(x) for x in embeddings[0]]

        # +1 because the anchor event is always its own nearest neighbor.
        result = collection.query(query_embeddings=[anchor], n_results=top_k + 1)
        ids = result["ids"][0]
        distances = result["distances"][0]
        documents = (result.get("documents") or [[]])[0]

        cache = load_event_cache(user_id) if user_id else {}
        similar: List[dict] = []
        for i, (neighbor_id, distance) in enumerate(zip(ids, distances)):
            if str(neighbor_id) == event_id:
                continue
            meta = cache.get(str(neighbor_id), {})
            similar.append(
                {
                    "event_id": str(neighbor_id),
                    "similarity_score": max(0.0, min(1.0, 1.0 - float(distance))),
                    "title": meta.get("title", ""),
                    "org": meta.get("org", ""),
                    "type": meta.get("type", ""),
                    "snippet": documents[i] if i < len(documents) else "",
                }
            )
            if len(similar) >= top_k:
                break

        return {"event_id": event_id, "available": True, "similar": similar}

    except Exception:
        logger.exception("similar-events lookup failed for %r", event_id)
        return _unavailable(
            event_id, "events collection unavailable — run a search first so events get embedded"
        )
