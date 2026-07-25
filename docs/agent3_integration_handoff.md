# Handoff: Agent 3 — Curator & Ranker + RL Preference Loop (@Mariela)

Mirrors the format of `docs/spoorthy_integration_handoff.md`. Agent 3 consumes
`RankedEvents` (Agent 2's output) + `PreferenceProfile` (Agent 1's output) and
produces `FinalFeed`; it also owns the accept/skip signal loop that nudges the
stored preference embedding over time.

## 1. What's new

| File | Status |
|---|---|
| `app/agents/curator_ranker.py` | new — Agent 3 core (`get_final_feed`) |
| `app/agents/weather.py` | new — NWS weekend forecast (`get_weekend_forecast`) |
| `app/agents/rl_loop.py` | new — signal application (`apply_signal`, `apply_signal_batch`) |
| `app/storage.py` | new — flat JSON signal log + event metadata cache |
| `app/schemas/models.py` | **additive only** — added `CuratorRankerRequest`; every existing class untouched |
| `app/main.py` | added `POST /agents/curator-ranker` and `POST /signals` routes |
| `tests/test_curator_ranker.py`, `test_weather.py`, `test_rl_loop.py`, `test_storage.py` | new, offline, patch-based (same style as the existing suite) |

No changes to `preference_profiler.py`, `live_events.py`, or `event_retriever.py`.

## 2. New endpoints

```
POST /agents/curator-ranker
  body: {"profile": PreferenceProfile, "ranked_events": RankedEvents}
  returns: FinalFeed

POST /signals
  body: SignalBatch
  returns: {"received": int, "embedding_updates": int}
```

## 3. Schema change — needs a quick team nod

`app/schemas/models.py` gained one new class, `CuratorRankerRequest`
(`{profile, ranked_events}`) — the request body `/agents/curator-ranker`
needs since `FinalFeed`'s inputs were never given their own request model.
Additive only, same pattern as `PreferenceProfilerRequest`. Nothing else in
the file changed. Per the existing convention ("locked contract,
additive-only, coordinate first") — flagging here rather than merging silently.

## 4. Curator/ranker behavior

- One HF `chat_completion` call, same `provider`/`model` env vars as Agents 1
  and 2. Prompt includes category weights, raw text, the NWS weekend
  forecast, and every candidate event (id, title, date, price, type, org,
  description, similarity score).
- Grounding: any `event_id` the model returns that wasn't in the input
  `RankedEvents` is dropped — same philosophy as Agents 1/2's link/name checks.
- `best_bets_this_weekend` is validated against the actual forecast dates; if
  the model's picks don't survive validation (or it returns none), Agent 3
  computes its own weekend shortlist from the scored feed rather than leaving
  the section unexplained-empty.
- **Never raises.** Missing `HF_TOKEN`, a failed HF call, or malformed model
  output all degrade to a similarity-ranked fallback feed
  (`final_score = similarity_score`, generic reason), same contract as
  Agents 1 and 2.
- Price isn't hard-filtered in code — the model weighs it holistically
  alongside relevance/date/weather per the prompt, since `price` is
  `Union[str, float]` with inconsistent formats (`"Free"`, `"$25"`, `"See
  website"`, numeric) and a hardcoded parser would silently misjudge one of
  those forms. Worth revisiting with an explicit price-tier filter if the
  team wants harder guarantees than "the LLM was told to consider it."

## 5. Weather

- `get_weekend_forecast()` — two-step NWS call (`/points/{lat},{lon}` →
  `forecast` URL), Central Park (40.7829, -73.9654) used as a single citywide
  anchor point (override via `NWS_LAT`/`NWS_LON`). No API key required, but
  NWS asks for an identifying `User-Agent` — set `NWS_CONTACT_EMAIL` in `.env`
  so it's not sent anonymous in production.
- Degrades to `{"available": False, "periods": []}` on any failure; the
  curator prompt is told to say "(forecast unavailable)" rather than invent
  conditions.
- Uses `httpx` (already in `requirements.txt` for Agent 2/tests) — no new
  dependency needed.

## 6. RL preference loop — read this before wiring `/signals` into production

- `apply_signal(signal, user_id, embedding_id, profile=None, learning_rate=0.15)`
  nudges the stored `user_preferences` vector toward (`accept`) or away from
  (`skip`) the event's embedding, then L2-renormalizes (cosine space).
- **The raw signal is always logged first**, independent of whether the
  embedding update succeeds — `storage/signals_<user_id>.json`.
- Event re-embedding uses a metadata cache
  (`storage/event_cache_<user_id>.json`) that `curator_ranker.get_final_feed`
  writes every time it produces a feed — necessary because the locked
  `Signal` schema only carries `event_id`/`action`/`timestamp`, not the event
  fields needed to re-embed it.
- **Chroma read/write split, mirroring Agent 2's `chroma/` vs.
  `chroma_events/` pattern:** this module treats the committed `chroma/`
  (`USER_PREF_CHROMA_PATH`) as **read-only**, per
  `spoorthy_integration_handoff.md` #12 ("committed `chroma/` ... do not
  modify"). All real updates instead go to a new, gitignored,
  writable store: `chroma_user_prefs/` (override via
  `USER_PREF_WRITE_CHROMA_PATH`). On a user's first-ever update, if the
  writable store has nothing yet, it seeds from the read-only store (e.g. the
  seeded `pref_test_user_001` vector), or from `profile.profile_embedding_seed`
  if nothing exists anywhere.

### Coordination needed (not resolved unilaterally here)

1. **Agent 1 (Amy):** `PreferenceProfile.embedding_id` is currently always
   `null` from real Agent 1 output (README caveat, still true). `/signals`
   assumes the convention `embedding_id = f"pref_{user_id}"` until the team
   agrees on one — Agent 1 needs to start setting this so a real user's
   profile and their RL-updated vector actually share a key.
2. **Agent 2 (Spoorthy/Masa):** even once (1) is fixed, `_resolve_user_vector`
   in `event_retriever.py` only reads from `USER_PREF_CHROMA_PATH` (defaults
   to the pristine committed `chroma/`) — it never sees this module's
   writable `chroma_user_prefs/` store. For updated preferences to actually
   change future search ranking, either point a shared `USER_PREF_CHROMA_PATH`
   env var at the writable store in deployment, or the two stores need an
   explicit sync step. Flagging for the team, not changing Agent 2's code.

## 7. `.gitignore` additions needed (Agent 3 doesn't own this file)

```
chroma_user_prefs/
storage/signals_*.json
storage/event_cache_*.json
```

Unlike the committed `storage/test_user_001_profile.json` seed fixture, these
are generated interaction data — probably shouldn't be committed.

## 8. `.env.example` additions (optional, all have working defaults)

```
# Agent 3 — NWS weather (no key required; contact info is polite, not enforced)
NWS_CONTACT_EMAIL=you@example.com
# NWS_LAT=40.7829
# NWS_LON=-73.9654

# Agent 3 — RL loop writable vector store (defaults to chroma_user_prefs/)
# USER_PREF_WRITE_CHROMA_PATH=chroma_user_prefs
```

## 9. Observed while testing (not caused by this work)

Running the existing `tests/test_live_events.py` with no `HF_TOKEN` in the
environment at all, `test_fetch_live_events_grounds_and_shapes_events` and
`test_fetch_live_events_falls_back_to_categories_when_no_orgs` fail — both
short-circuit on `fetch_live_events`'s own `if not hf_token: raise` before
the mocked `InferenceClient`/`DDGS` are ever exercised. They pass as soon as
any `HF_TOKEN` value (real or dummy) is present in the environment, so this
is presumably masked by a local `.env` or CI secret already being set in
normal dev — but it means the suite isn't actually token-independent as the
README states. Worth a `@patch.dict(os.environ, {"HF_TOKEN": "test-token"})`
on those two tests, or a session-scoped autouse fixture, so `pytest` is
reproducibly green on a totally clean checkout. Every new Agent 3 test in
this handoff avoids the issue by passing `hf_token=` explicitly rather than
relying on the ambient environment.

## 10. Test coverage added

- `test_weather.py` (7): weekend-window filtering, graceful degradation on
  each NWS failure mode, weekend-date-math edge cases (including the
  "today is already Saturday" case).
- `test_curator_ranker.py` (8): LLM scoring/reason generation, event_id
  grounding (invented ids dropped), fallback on no token / LLM failure /
  empty model output, empty-input short-circuit (no LLM call), best-bets
  weekend validation, date-parsing caveats (`"TBA"`, date ranges).
- `test_rl_loop.py` (8): accept moves the vector toward the event, skip moves
  it away (exact vector math asserted), signal always logged even without
  cached metadata, write-store-empty falls back to read-store, brand-new-user
  falls back to embedding the profile seed, vector-dimension mismatch guard,
  never raises on a Chroma failure, batch application + update counting.
- `test_storage.py` (6): per-user signal log append/read, per-user event
  cache upsert-by-id, missing-file and corrupted-file handling.

All new tests run with only `fastapi`, `pydantic`, `huggingface_hub`,
`python-dotenv`, `pytest`, `httpx` installed — `chromadb` and
`sentence-transformers` are never actually imported in tests (same lazy-
loader-plus-patch pattern `test_event_retriever.py` already uses), so the
suite doesn't need those heavy deps installed to pass. Full suite: 52 passed
(23 pre-existing + 29 new) with a dummy `HF_TOKEN` set, 50 passed without one
(see #9).

## 11. Files to avoid modifying

- `app/schemas/models.py` — only the one additive class above; nothing
  existing was touched.
- `app/agents/preference_profiler.py`, `live_events.py`, `event_retriever.py`
  — untouched.
- Committed `chroma/` and `storage/test_user_001_profile.json` — never
  written to by this module.

## 12. Integration checklist

- [x] `POST /agents/curator-ranker` wired, returns a valid `FinalFeed`.
- [x] `POST /signals` wired, returns `{"received", "embedding_updates"}`.
- [x] Never raises/500s on missing token, HF failure, weather failure, or
      malformed model output.
- [x] `pytest -q` → 52 passing (with a token set), new modules' own tests
      pass regardless of `HF_TOKEN` (pass it explicitly).
- [ ] Amy: start setting `PreferenceProfile.embedding_id = f"pref_{user_id}"`.
- [ ] Vivek: add the `.gitignore` entries in #7.
- [ ] Team: confirm the `USER_PREF_CHROMA_PATH` deployment story in #6.2, or
      accept that RL updates won't affect ranking until it's resolved.
- [ ] Frontend: wire `/agents/curator-ranker` and `/signals` calls into
      `app.js` (not done here — today's session was scoped to the backend
      agent + RL loop; happy to do the frontend wiring next).
