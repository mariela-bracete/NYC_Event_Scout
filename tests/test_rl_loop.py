"""Offline unit tests for Agent 3's RL loop (app/agents/rl_loop.py).

The embedder, ChromaDB client, and app.storage functions are all faked/patched
— these tests never load a real model, hit ChromaDB, or touch the filesystem.
"""

from unittest.mock import patch

import pytest

import app.agents.rl_loop as rl
from app.schemas.models import Category, PreferenceProfile, Signal, SignalBatch


# --- fakes --------------------------------------------------------------------


class _FakeEmbedding:
    def __init__(self, vector):
        self._vector = vector

    def tolist(self):
        return list(self._vector)


class _FakeEmbedder:
    def __init__(self, vectors_by_text=None, default=(0.2, 0.2, 0.2, 0.2)):
        self.vectors_by_text = vectors_by_text or {}
        self.default = default
        self.encoded = []

    def encode(self, text):
        self.encoded.append(text)
        vector = self.vectors_by_text.get(text, self.default)
        return _FakeEmbedding(vector)


class _FakeCollection:
    def __init__(self, initial=None):
        self._store = dict(initial or {})
        self.upsert_calls = []

    def get(self, ids, include):
        return {"embeddings": [self._store.get(i) for i in ids]}

    def upsert(self, ids, embeddings):
        for i, e in zip(ids, embeddings):
            self._store[i] = list(e)
        self.upsert_calls.append((list(ids), [list(e) for e in embeddings]))


class _FakeChromaClient:
    def __init__(self, collection=None, missing_collection=False):
        self._collection = collection if collection is not None else _FakeCollection()
        self._missing = missing_collection

    def get_collection(self, name):
        if self._missing:
            raise ValueError("collection does not exist")
        return self._collection

    def get_or_create_collection(self, name, metadata=None):
        return self._collection


def _profile(**overrides) -> PreferenceProfile:
    base = dict(
        user_id="user-1",
        categories=[Category(name="nightlife_bars", weight=0.9)],
        orgs=[],
        raw_text="I love jazz",
        profile_embedding_seed="I love jazz | nightlife_bars",
    )
    base.update(overrides)
    return PreferenceProfile(**base)


def _patch_clients(write_client, read_client=None):
    read_client = read_client if read_client is not None else write_client

    def _side_effect(path):
        return write_client if path == rl._write_chroma_path() else read_client

    return patch.object(rl, "_load_chroma_client", side_effect=_side_effect)


# --- tests --------------------------------------------------------------------


def test_apply_signal_accept_moves_vector_toward_event():
    embedder = _FakeEmbedder(vectors_by_text={"cached text": (0.0, 1.0, 0.0, 0.0)})
    write_collection = _FakeCollection({"pref_user-1": [1.0, 0.0, 0.0, 0.0]})
    write_client = _FakeChromaClient(write_collection)

    cache = {"evt_1": {"title": "cached", "description": "text", "type": "", "org": ""}}

    with patch.object(rl, "_load_embedder", return_value=embedder), _patch_clients(
        write_client
    ), patch.object(rl, "append_signal") as mock_append, patch.object(
        rl, "load_event_cache", return_value=cache
    ):
        updated = rl.apply_signal(
            Signal(event_id="evt_1", action="accept", timestamp="2026-07-25T12:00:00+00:00"),
            user_id="user-1",
            embedding_id="pref_user-1",
            learning_rate=0.5,
        )

    assert updated is True
    mock_append.assert_called_once()
    new_vector = write_collection._store["pref_user-1"]
    # b=[1,0,0,0], e=[0,1,0,0], lr=0.5 -> raw=[0.5,0.5,0,0] -> L2-normalized.
    assert new_vector == pytest.approx([0.70710678, 0.70710678, 0.0, 0.0], abs=1e-6)


def test_apply_signal_skip_moves_vector_away():
    embedder = _FakeEmbedder(vectors_by_text={"cached text": (0.0, 1.0, 0.0, 0.0)})
    write_collection = _FakeCollection({"pref_user-1": [1.0, 0.0, 0.0, 0.0]})
    write_client = _FakeChromaClient(write_collection)
    cache = {"evt_1": {"title": "cached", "description": "text", "type": "", "org": ""}}

    with patch.object(rl, "_load_embedder", return_value=embedder), _patch_clients(
        write_client
    ), patch.object(rl, "append_signal"), patch.object(
        rl, "load_event_cache", return_value=cache
    ):
        updated = rl.apply_signal(
            Signal(event_id="evt_1", action="skip", timestamp="2026-07-25T12:00:00+00:00"),
            user_id="user-1",
            embedding_id="pref_user-1",
            learning_rate=0.5,
        )

    assert updated is True
    new_vector = write_collection._store["pref_user-1"]
    # raw = b - 0.5*(e-b) = [1.5, -0.5, 0, 0] -> L2-normalized.
    assert new_vector == pytest.approx([0.9486833, -0.31622777, 0.0, 0.0], abs=1e-6)


def test_apply_signal_logs_signal_even_without_cached_metadata():
    """No cached metadata for the event_id -> can't re-embed -> returns False,
    but the raw signal is still logged (never silently lost)."""
    with patch.object(rl, "_load_embedder", return_value=_FakeEmbedder()), patch.object(
        rl, "append_signal"
    ) as mock_append, patch.object(rl, "load_event_cache", return_value={}):
        updated = rl.apply_signal(
            Signal(event_id="evt_unknown", action="accept", timestamp="2026-07-25T12:00:00+00:00"),
            user_id="user-1",
            embedding_id="pref_user-1",
        )

    assert updated is False
    mock_append.assert_called_once()


def test_apply_signal_falls_back_to_read_store_when_write_store_empty():
    embedder = _FakeEmbedder(vectors_by_text={"cached text": (0.0, 1.0, 0.0, 0.0)})
    write_client = _FakeChromaClient(_FakeCollection())  # nothing stored yet
    read_client = _FakeChromaClient(_FakeCollection({"pref_test_user_001": [1.0, 0.0, 0.0, 0.0]}))
    cache = {"evt_1": {"title": "cached", "description": "text", "type": "", "org": ""}}

    with patch.object(rl, "_load_embedder", return_value=embedder), _patch_clients(
        write_client, read_client
    ), patch.object(rl, "append_signal"), patch.object(
        rl, "load_event_cache", return_value=cache
    ):
        updated = rl.apply_signal(
            Signal(event_id="evt_1", action="accept", timestamp="2026-07-25T12:00:00+00:00"),
            user_id="test_user_001",
            embedding_id="pref_test_user_001",
            learning_rate=0.5,
        )

    assert updated is True
    # The write client now holds the updated vector, seeded from the read store.
    assert "pref_test_user_001" in write_client._collection._store


def test_apply_signal_seeds_from_profile_when_nothing_stored_anywhere():
    embedder = _FakeEmbedder(
        vectors_by_text={
            "cached text": (0.0, 1.0, 0.0, 0.0),
            "new profile seed | nightlife_bars": (1.0, 0.0, 0.0, 0.0),
        }
    )
    empty_client = _FakeChromaClient(_FakeCollection())
    cache = {"evt_1": {"title": "cached", "description": "text", "type": "", "org": ""}}

    with patch.object(rl, "_load_embedder", return_value=embedder), _patch_clients(
        empty_client
    ), patch.object(rl, "append_signal"), patch.object(rl, "load_event_cache", return_value=cache):
        updated = rl.apply_signal(
            Signal(event_id="evt_1", action="accept", timestamp="2026-07-25T12:00:00+00:00"),
            user_id="brand-new-user",
            embedding_id="pref_brand-new-user",
            profile=_profile(profile_embedding_seed="new profile seed | nightlife_bars"),
            learning_rate=0.5,
        )

    assert updated is True
    assert "new profile seed | nightlife_bars" in embedder.encoded


def test_apply_signal_dim_mismatch_returns_false():
    embedder = _FakeEmbedder(vectors_by_text={"cached text": (0.0, 1.0, 0.0)})  # 3-dim
    write_collection = _FakeCollection({"pref_user-1": [1.0, 0.0, 0.0, 0.0]})  # 4-dim
    write_client = _FakeChromaClient(write_collection)
    cache = {"evt_1": {"title": "cached", "description": "text", "type": "", "org": ""}}

    with patch.object(rl, "_load_embedder", return_value=embedder), _patch_clients(
        write_client
    ), patch.object(rl, "append_signal"), patch.object(
        rl, "load_event_cache", return_value=cache
    ):
        updated = rl.apply_signal(
            Signal(event_id="evt_1", action="accept", timestamp="2026-07-25T12:00:00+00:00"),
            user_id="user-1",
            embedding_id="pref_user-1",
        )

    assert updated is False


def test_apply_signal_never_raises_on_chroma_failure():
    with patch.object(rl, "_load_embedder", return_value=_FakeEmbedder()), patch.object(
        rl, "_load_chroma_client", side_effect=RuntimeError("chroma unavailable")
    ), patch.object(rl, "append_signal") as mock_append, patch.object(
        rl, "load_event_cache", return_value={"evt_1": {"title": "x", "description": "", "type": "", "org": ""}}
    ):
        updated = rl.apply_signal(
            Signal(event_id="evt_1", action="accept", timestamp="2026-07-25T12:00:00+00:00"),
            user_id="user-1",
            embedding_id="pref_user-1",
        )

    assert updated is False
    mock_append.assert_called_once()  # signal still logged


def test_apply_signal_batch_applies_each_signal_and_counts_updates():
    calls = []

    def _fake_apply(signal, user_id, embedding_id, profile=None, learning_rate=rl.DEFAULT_LEARNING_RATE):
        calls.append(signal.event_id)
        return signal.event_id != "evt_no_cache"

    batch = SignalBatch(
        user_id="user-1",
        signals=[
            Signal(event_id="evt_1", action="accept", timestamp="2026-07-25T12:00:00+00:00"),
            Signal(event_id="evt_no_cache", action="skip", timestamp="2026-07-25T12:01:00+00:00"),
        ],
    )

    with patch.object(rl, "apply_signal", side_effect=_fake_apply):
        result = rl.apply_signal_batch(batch, embedding_id="pref_user-1")

    assert calls == ["evt_1", "evt_no_cache"]
    assert result == {"received": 2, "embedding_updates": 1}
