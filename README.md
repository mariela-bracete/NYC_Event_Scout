# NYC Event Scout

A multi-agent event discovery system for NYC. This phase implements:

- **Agent 1 — Preference Profiler** (`app/agents/preference_profiler.py`): real
  implementation, built on the approach already prototyped in
  `notebooks/agent1_preference_profiler.ipynb` (Hugging Face `InferenceClient`,
  `publicai` provider). It runs a live DuckDuckGo web search for NYC organizations,
  then calls that same Hugging Face-hosted LLM to turn a user's free text +
  selected interest categories into normalized category weights and a seeded list
  of real NYC organizations grounded in those search results, adapted to the
  locked `PreferenceProfile` schema below.
- **Agent 2 — Event Retriever** (`app/agents/event_retriever.py` +
  `app/agents/live_events.py`): **real, live**. For each org in the user's
  `PreferenceProfile` (or each selected category if Agent 1 fell back to no orgs),
  `live_events.py` runs a real DuckDuckGo search and asks the same Hugging Face
  model Agent 1 uses to extract grounded, real upcoming events (an event's `link`
  must be one of the URLs the search actually returned, or it's dropped).
  `event_retriever.py` then embeds each event on `title + description + type + org`
  with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) into the ChromaDB
  `events` collection, resolves the user's preference vector **hybrid**-style —
  the stored `user_preferences` vector when `PreferenceProfile.embedding_id` is
  set and found, otherwise an on-the-fly embedding of `profile_embedding_seed` —
  and returns `Event`s sorted by cosine `similarity_score`. If live retrieval
  fails (no token, nothing found, malformed event), or `chromadb` /
  `sentence-transformers` aren't installed, or the vector path fails, it degrades
  gracefully to the `app/mocks/mock_events.json` stub (`get_stub_events`).
- **Agent 3 — Curator & Ranker + RL Preference Loop** (`app/agents/curator_ranker.py`
  + `app/agents/weather.py` + `app/agents/rl_loop.py`): **real, live**. Takes
  Agent 2's `RankedEvents` plus the `PreferenceProfile`, fetches the NYC
  weekend forecast from the National Weather Service (`api.weather.gov`, no
  key required), and makes one more Hugging Face call (same model/provider as
  Agents 1 and 2) to score/filter events, write a plain-language `reason` per
  event, and shortlist up to 3 `best_bets_this_weekend` — validated against
  the actual forecast dates, never invented. Any failure (no token, HF error,
  malformed output, weather unavailable) degrades to a similarity-ranked feed,
  same never-500 contract as the other agents. `app/agents/rl_loop.py` then
  owns the accept/skip feedback loop: `POST /signals` logs each signal to a
  flat JSON file and nudges the user's stored preference vector toward
  accepted events / away from skipped ones (lightweight, RLHF-inspired — no RL
  framework). See `docs/agent3_integration_handoff.md` for the full detail,
  including two open coordination items with Agents 1 and 2.
- A minimal vanilla HTML/CSS/JS frontend that drives Agents 1 and 2 in
  sequence. It does **not** yet call `/agents/curator-ranker` or `/signals` —
  wiring the final feed + accept/skip buttons into `app.js` is the next step.

## Repo layout

```
app/
├── main.py                        # FastAPI app: API routes + static frontend
├── agents/
│   ├── preference_profiler.py     # Agent 1 — real
│   ├── event_retriever.py         # Agent 2 — RAG core (+ stub fallback)
│   ├── live_events.py             # Agent 2 — live retrieval (search + HF extraction)
│   ├── curator_ranker.py          # Agent 3 — scoring/filtering/reasons + best bets (+ stub fallback)
│   ├── weather.py                 # Agent 3 — NWS weekend forecast (no key required)
│   └── rl_loop.py                 # Agent 3 — accept/skip -> preference embedding update
├── schemas/
│   └── models.py                  # all shared pydantic schemas
├── storage.py                     # Agent 3 — flat JSON signal log + event metadata cache
└── mocks/
    └── mock_events.json           # stub-fallback events only (live path doesn't touch this)
frontend/
├── index.html
├── style.css
└── app.js
tests/
├── test_health.py
├── test_preference_profiler.py
├── test_event_retriever.py
├── test_live_events.py
├── test_curator_ranker.py
├── test_weather.py
├── test_rl_loop.py
└── test_storage.py
docs/
├── spoorthy_integration_handoff.md
├── agent2_rag_implementation.md
└── agent3_integration_handoff.md
```

`chroma/` at the repo root is a committed ChromaDB store: its `user_preferences`
collection holds the seeded 384-dim preference vector (`pref_test_user_001`) that
Agent 2 reads when a profile carries a matching `embedding_id` (read-only). Agent 2
writes its `events` collection to a separate, gitignored `chroma_events/` store at
query time, so the committed `chroma/` stays pristine. Both paths are overridable
via `USER_PREF_CHROMA_PATH` / `EVENTS_CHROMA_PATH`. Agent 3's RL loop follows the
same read/write split: it treats `chroma/` as read-only and writes every real
preference-vector update to a separate, gitignored `chroma_user_prefs/` store
instead (override via `USER_PREF_WRITE_CHROMA_PATH`) — see
`docs/agent3_integration_handoff.md` for why that store isn't wired into Agent 2's
read path yet. `notebooks/`, `storage/*_profile.json`, and `agents/prompts/` remain
earlier prototype artifacts not wired into the `app/` service; `storage/signals_*.json`
and `storage/event_cache_*.json` are new, generated per-user by Agent 3 at runtime.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and set:
#   HF_TOKEN=hf_...

uvicorn app.main:app --reload
```

Then open **http://localhost:8000** — type some interests, check a few categories,
and click "Find Events". You should see a real Agent-1-generated org list (drawn
from a live web search, not invented), followed by real, live events for those
orgs (also grounded in search results, not invented) ranked by similarity to your
profile. The live event pipeline runs one search + one HF call per org, so expect
this step to take significantly longer than Agent 1 — tens of seconds for a
profile with several orgs.

`GET http://localhost:8000/health` should return `{"status": "ok"}`.

The frontend doesn't call Agent 3 yet (see above), but both new endpoints work
directly:

```bash
curl -X POST http://localhost:8000/agents/curator-ranker \
  -H "Content-Type: application/json" \
  -d '{"profile": <PreferenceProfile JSON>, "ranked_events": <RankedEvents JSON>}'

curl -X POST http://localhost:8000/signals \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user_001", "signals": [{"event_id": "evt_001", "action": "accept", "timestamp": "2026-07-25T12:00:00Z"}]}'
```

## Running tests

```bash
pytest
```

All Hugging Face, search, weather, and vector-store calls are mocked across the
suite — nothing hits the network, a real API, or requires `chromadb` /
`sentence-transformers` to actually be installed (Agent 3's tests use the same
lazy-loader-plus-patch pattern `test_event_retriever.py` established). 52 tests
total:

- `test_preference_profiler.py` (4): profile matches schema shape and merges the
  model's categories/orgs; graceful fallback on an LLM exception, a search
  failure, and a missing `HF_TOKEN`.
- `test_live_events.py` (6): events are grounded/shaped correctly; an event whose
  `link` wasn't actually in the search results is dropped; falls back to
  per-category search when a profile has no orgs; raises (→ stub fallback) with
  no token, nothing to search, or no events extracted.
- `test_event_retriever.py` (12): the RAG ranking core (embedding, hybrid vector
  resolution, cosine scoring, sorting) — unchanged by the live-events work, now
  fed via `_load_live_events`/`_load_raw_events()` depending on the test; plus
  live-events flow through with a **computed** `similarity_score`, malformed live
  events fall back to stub, `_normalize` field validation, and event_id dedupe.
- `test_curator_ranker.py` (8): LLM scoring/reason generation, event_id
  grounding (invented ids dropped), fallback on no token / LLM failure / empty
  model output, empty-input short-circuit, best-bets weekend validation,
  date-parsing caveats (`"TBA"`, date ranges).
- `test_weather.py` (7): weekend-window filtering, graceful degradation on each
  NWS failure mode, weekend-date-math edge cases.
- `test_rl_loop.py` (8): accept moves the preference vector toward the event,
  skip moves it away (exact vector math asserted), signal always logged even
  without cached metadata, write-store-empty falls back to read-store,
  brand-new-user falls back to embedding the profile seed, vector-dimension
  mismatch guard, never raises on a Chroma failure, batch application.
- `test_storage.py` (6): per-user signal log append/read, per-user event cache
  upsert-by-id, missing-file and corrupted-file handling.
- `test_health.py` (1).

Note: two of the existing `test_live_events.py` tests only pass with an
`HF_TOKEN` present in the environment (even a dummy one) — see
`docs/agent3_integration_handoff.md` #9 for detail. All new Agent 3 tests avoid
this by passing `hf_token=` explicitly rather than relying on the ambient
environment.

## Where the API key goes

Put it in `.env` at the repo root (never commit this file — it's already in
`.gitignore`):

```
HF_TOKEN=hf_...
```

Get one at https://huggingface.co/settings/tokens. `app/main.py` calls
`load_dotenv()` before anything else reads the environment, so
`uvicorn app.main:app --reload` picks it up automatically.

Optional `HF_PROVIDER` / `HF_MODEL` env vars override the defaults (`publicai` /
`swiss-ai/Apertus-70B-Instruct-2509` — the same combo already validated in the
prototype notebook) if either isn't available under your HF plan.

Agent 3's weather call (`api.weather.gov`) needs **no API key** — NWS just asks
consumers to identify themselves via a `User-Agent` header. Set
`NWS_CONTACT_EMAIL` in `.env` so that header isn't sent anonymous in production;
`NWS_LAT`/`NWS_LON` optionally override the single citywide anchor point
(defaults to Central Park). None of these are required to run the app or tests.

## What's next (later phases)

- **Date validation for live events**: `live_events.py` asks the model for ISO
  8601 dates but doesn't validate the format — verified live, a non-ISO value
  (`"TBA"`) got through once. Worth adding a real parse-or-drop check.
- **Live event freshness**: since retrieval goes through generic web search
  rather than a live events API/calendar, some extracted events come from stale,
  cached search results (verified live: a couple of 2022/2023-dated events slipped
  through alongside genuinely current ones). A recency check on the date field
  would help.
- **Latency**: live retrieval is one search + one HF call per org (up to
  `MAX_TARGETS = 5`), sequential — a full request can take a minute or more.
  Worth parallelizing (e.g. `concurrent.futures`) if this needs to feel snappier.
- **Agent 1 JSON robustness**: found live — the model occasionally wraps its JSON
  in a ```` ```json ```` fence plus trailing prose commentary despite being told
  not to, which can break the regex-based parser. Same underlying pattern is used
  in `live_events.py`'s extraction, so it's worth hardening both at once (e.g.
  strip fences explicitly before parsing) rather than just retrying.
- **Persisting `embedding_id`**: Agent 1 still doesn't write to the
  `user_preferences` ChromaDB collection or set `PreferenceProfile.embedding_id`,
  so the hybrid "stored vector" path in Agent 2 (and now Agent 3's RL loop) is
  only exercised by the one seeded `pref_test_user_001` profile, never a real
  live user. Agent 3 assumes the convention `embedding_id = f"pref_{user_id}"`
  until the team agrees on one and Agent 1 starts setting it.
- **RL updates not yet visible to search**: Agent 3's RL loop writes updated
  preference vectors to a separate, gitignored `chroma_user_prefs/` store
  rather than the committed read-only `chroma/` — but Agent 2's
  `_resolve_user_vector` still only reads from `USER_PREF_CHROMA_PATH`
  (defaults to committed `chroma/`). Until a deployment points that env var at
  the writable store (or the two are synced), accept/skip signals get logged
  and the embedding gets updated, but future searches won't reflect it yet.
  See `docs/agent3_integration_handoff.md` #6.
- **Frontend wiring for Agent 3**: `app.js` still only calls
  `/agents/preference-profiler` and `/agents/event-retriever` in sequence —
  `/agents/curator-ranker` and `/signals` (accept/skip buttons) aren't wired
  into the UI yet.
- **Price as a hard filter**: Agent 3's prompt asks the model to weigh price
  holistically rather than hard-filtering on it in code, since `price` is
  `Union[str, float]` with inconsistent formats (`"Free"`, `"$25"`, `"See
  website"`, numeric). Worth an explicit price-tier filter if the team wants a
  harder guarantee than "the LLM was told to consider it."
- **`.gitignore` additions still needed**: `chroma_user_prefs/`,
  `storage/signals_*.json`, `storage/event_cache_*.json` — these are Agent 3's
  generated per-user data, not seed fixtures, and probably shouldn't be
  committed (not added yet since Agent 3 doesn't own `.gitignore`).

## Judgment calls made this phase

- **LLM provider**: Hugging Face's Inference API doesn't offer a built-in,
  autonomous "web_search" tool the model can call itself the way some other
  providers do. So the search step is done directly in Python — a real, keyless
  DuckDuckGo query via the `ddgs` package — and the results are handed to the LLM
  as grounding context, with an explicit instruction to only select organizations
  that actually appear in those results (not invent names). This is still a real
  search + a real LLM call, just orchestrated client-side rather than via
  agentic tool-calling.
- **Provider + model**: `provider="publicai"` with `swiss-ai/Apertus-70B-Instruct-2509`,
  matching what was already validated working in
  `notebooks/agent1_preference_profiler.ipynb` rather than picking a different
  default. Override with `HF_PROVIDER`/`HF_MODEL` in `.env` if either becomes
  unavailable for your account.
- **Model output parsing**: Agent 1 asks the model to return a single raw JSON
  object rather than relying on structured-output/JSON-schema enforcement, since
  that support varies across HF Inference Providers. The parser first tries a
  direct `json.loads`, then falls back to extracting the first `{...}` block via
  regex.
- **IDs**: `user_id` and each `org_id` are generated server-side with `uuid4` —
  never trusted from the model.
- **Grounding**: for Agent 1's orgs, the prompt is backed by a real keyword filter
  (`_looks_nyc_related`) that drops search results with no NYC signal before the
  model ever sees them. For Agent 2's live events, grounding is a hard,
  mechanical check: an event's `link` must exactly match one of the URLs the
  search actually returned, or the event is dropped — not just a prompt
  instruction.
- **Live events: orgs vs. categories as search targets**: `live_events.py`
  searches per org when `profile.orgs` is non-empty (the common case), and falls
  back to per-*category* search when it's empty (Agent 1's fallback-profile case,
  e.g. no token or a failed LLM call) — so a degraded Agent 1 doesn't also zero
  out Agent 2's ability to find anything.
- **Live events: soft vs. hard required fields**: `title`/`date`/`link` are
  hard-grounded — the whole event is dropped if the model can't point to a real
  one of each in the search results. `location`/`price` are softer: when the
  model doesn't have enough to say, `live_events.py` fills in honest placeholder
  text (`"{org}, New York, NY"` / `"See website"`) rather than dropping the event
  or inventing a specific value — judged a reasonable middle ground between the
  handoff's "no defaults" guidance and not discarding an otherwise-real event over
  a minor field.
- **Live events: deterministic IDs**: unlike Agent 1's `uuid4()` org/user ids,
  live event ids are a hash of `(org_id, title, date)` — so the same real event
  surfacing twice (e.g. from two different org searches) collides and gets
  deduped, rather than appearing twice with different random ids.
- **Graceful degradation surface**: any failure in Agent 1 (missing token, search
  failure, network error, malformed JSON from the model, empty search results)
  falls back to a profile with the user's *selected* categories at weight 1.0 and
  an empty `orgs` list. Any failure in Agent 2's live retrieval (missing token,
  nothing to search, no events extracted, a malformed event) falls back to
  `mock_events.json`. Neither ever 500s.
- **Static file mount order**: the frontend's `StaticFiles` mount is registered
  *after* the `/health` and `/agents/*` routes in `main.py` so it can't shadow them.
- **Mixed `price` types** in `mock_events.json` (string `"Free"`, `int 0`, `float`,
  and `"$18"`) intentionally exercise the `Union[str, float]` price field in the
  schema.
- **Agent 3 grounding**: same philosophy as Agents 1/2 — any `event_id` the
  curator model returns that wasn't in the input `RankedEvents` is silently
  dropped rather than trusted, and `best_bets_this_weekend` picks are validated
  against the actual NWS forecast dates before being accepted.
- **Agent 3 fallback**: any failure (no token, HF error, malformed output, or
  the model scoring zero valid events) degrades to a feed built directly from
  `RankedEvents` — `final_score = similarity_score`, a generic reason string,
  and a weekend shortlist computed by date-filtering rather than by the LLM.
- **Event re-embedding for the RL loop**: the locked `Signal` schema only
  carries `event_id`/`action`/`timestamp`, so `rl_loop.py` can't re-embed an
  accepted/skipped event from the signal alone. `curator_ranker.py` writes a
  small metadata cache (`storage/event_cache_<user_id>.json`) every time it
  produces a feed, and the RL loop reads from that cache when a signal for
  that event arrives later — judged simpler than changing the locked schema.
- **Sequential vs. batched RL updates**: `apply_signal_batch` applies each
  signal in a batch one at a time, each nudging the vector a little further,
  rather than averaging all the deltas into a single update. Simpler and
  matches the "lightweight, no RL framework" brief most directly; order-
  sensitive as a tradeoff.
- **Chroma read/write split for preference vectors**: mirroring Agent 2's
  `chroma/` (read-only, committed) vs. `chroma_events/` (writable, gitignored)
  pattern, Agent 3's RL loop never writes to committed `chroma/` — real
  updates go to a new `chroma_user_prefs/` store. On a user's first-ever
  update it seeds from the read-only store if something's there, or embeds
  the profile's seed text if nothing exists anywhere yet.
