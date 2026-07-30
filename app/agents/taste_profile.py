"""Taste Profile data builder — backs GET /profile.

Assembles everything the frontend needs to render the taste-profile feature
from data the system already persists (no LLM call, no network):

- the stored ``PreferenceProfile`` (``storage/users/<user_id>_profile.json``)
- the accept/skip signal log (``storage/signals_<user_id>.json``)
- the event metadata cache (``storage/event_cache_<user_id>.json``)

Per the project breakdown, the Taste Profile *feature* (LLM-generated
taste-type label like "The Cultural Forager") is Amy's, and optional. This
module is the backend data layer under it: deterministic structures for the
interest graph, category breakdown, and activity heatmap, plus a heuristic
placeholder ``taste_type``. Amy's LLM label generation can replace
``_heuristic_taste_type`` without touching the endpoint or response shape.

Never raises: a brand-new user with no stored data gets a valid, empty
response with ``available: False``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List

from app.storage import load_event_cache, load_profile, load_signals

# Placeholder labels keyed by dominant category keyword; Amy's LLM-generated
# label replaces this heuristic (see module docstring).
_TASTE_LABELS = {
    "art": "The Cultural Forager",
    "culture": "The Cultural Forager",
    "museum": "The Cultural Forager",
    "music": "The Sound Seeker",
    "jazz": "The Sound Seeker",
    "nightlife": "The Night Owl",
    "bar": "The Night Owl",
    "food": "The Neighborhood Taster",
    "restaurant": "The Neighborhood Taster",
    "park": "The Open-Air Optimist",
    "outdoor": "The Open-Air Optimist",
    "community": "The Civic Connector",
    "nonprofit": "The Civic Connector",
}
_DEFAULT_LABEL = "The NYC Explorer"


def _parse_day(timestamp: str) -> str:
    """ISO timestamp -> weekday name; unparseable timestamps bucket as 'Unknown'."""
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%A")
    except (ValueError, AttributeError):
        return "Unknown"


def _heuristic_taste_type(category_counts: Counter) -> str:
    if not category_counts:
        return _DEFAULT_LABEL
    dominant = category_counts.most_common(1)[0][0].lower()
    for keyword, label in _TASTE_LABELS.items():
        if keyword in dominant:
            return label
    return _DEFAULT_LABEL


def build_taste_profile(user_id: str) -> Dict:
    """All taste-profile data for one user, from persisted interaction data.

    Shape (stable contract for the frontend):
        {
          "user_id": str,
          "generated_at": ISO 8601 str,
          "available": bool,          # False when there's no data yet
          "taste_type": str,          # heuristic until Amy's LLM label lands
          "stats": {"accepts": int, "skips": int, "orgs_followed": int},
          "category_breakdown": [{"category": str, "count": int}],   # accepted events
          "interest_graph": {
            "nodes": [{"id": str, "label": str, "kind": "org"|"category", "size": int}],
            "edges": [{"source": str, "target": str, "weight": int}], # org—category co-occurrence
          },
          "activity_heatmap": [{"day": str, "accepts": int, "skips": int}],
        }
    """
    profile = load_profile(user_id)
    signals = load_signals(user_id)
    event_cache = load_event_cache(user_id)

    accepts = [s for s in signals if s.get("action") == "accept"]
    skips = [s for s in signals if s.get("action") == "skip"]

    # Category/org counts from accepted events, joined through the event cache
    # (the Signal schema only carries event_id/action/timestamp).
    category_counts: Counter = Counter()
    org_counts: Counter = Counter()
    edge_weights: Dict[tuple, int] = defaultdict(int)
    for signal in accepts:
        meta = event_cache.get(signal.get("event_id", ""), {})
        category = (meta.get("type") or "").strip()
        org = (meta.get("org") or "").strip()
        if category:
            category_counts[category] += 1
        if org:
            org_counts[org] += 1
        if category and org:
            edge_weights[(org, category)] += 1

    # Heatmap: accept/skip volume by weekday, only for days with activity.
    day_activity: Dict[str, Dict[str, int]] = defaultdict(lambda: {"accepts": 0, "skips": 0})
    for signal in accepts:
        day_activity[_parse_day(signal.get("timestamp", ""))]["accepts"] += 1
    for signal in skips:
        day_activity[_parse_day(signal.get("timestamp", ""))]["skips"] += 1

    nodes: List[dict] = [
        {"id": f"org::{org}", "label": org, "kind": "org", "size": count}
        for org, count in org_counts.most_common()
    ] + [
        {"id": f"cat::{cat}", "label": cat, "kind": "category", "size": count}
        for cat, count in category_counts.most_common()
    ]
    edges = [
        {"source": f"org::{org}", "target": f"cat::{cat}", "weight": weight}
        for (org, cat), weight in sorted(edge_weights.items(), key=lambda kv: -kv[1])
    ]

    # Fall back to the stored profile's declared interests when there's no
    # interaction data yet, so a new user still sees *something*.
    if not category_counts and profile is not None:
        category_counts = Counter(
            {c.name: max(1, round(c.weight * 10)) for c in profile.categories}
        )

    return {
        "user_id": user_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "available": bool(signals or profile is not None),
        "taste_type": _heuristic_taste_type(category_counts),
        "stats": {
            "accepts": len(accepts),
            "skips": len(skips),
            "orgs_followed": len(profile.orgs) if profile else 0,
        },
        "category_breakdown": [
            {"category": cat, "count": count} for cat, count in category_counts.most_common()
        ],
        "interest_graph": {"nodes": nodes, "edges": edges},
        "activity_heatmap": [
            {"day": day, **counts} for day, counts in day_activity.items()
        ],
    }
