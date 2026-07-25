// NYC Event Scout — single-screen, three-section frontend.
// Section 1 (interests) -> Agent 1. Section 2 (orgs) is an editable roster
// built from Agent 1's response. Section 3 (search) chains Agent 2 -> Agent 3
// and renders "Best bets this weekend" + the full ranked grid, each card
// wired to POST /signals on save/skip.

const statusEl = document.getElementById("status");

const categoryTiles = document.querySelectorAll(".tile");
const rawTextEl = document.getElementById("raw-text");
const findOrgsBtn = document.getElementById("find-orgs-btn");

const orgsPanel = document.getElementById("orgs-panel");
const orgChipsEl = document.getElementById("org-chips");
const addOrgForm = document.getElementById("add-org-form");
const addOrgName = document.getElementById("add-org-name");
const addOrgCategory = document.getElementById("add-org-category");

const divider2 = document.getElementById("divider-2");
const searchPanel = document.getElementById("search-panel");
const filterFree = document.getElementById("filter-free");
const filterWeekend = document.getElementById("filter-weekend");
const searchBtn = document.getElementById("search-events-btn");

const bestBetsContainer = document.getElementById("best-bets-container");
const bestBetsList = document.getElementById("best-bets-list");
const eventsContainer = document.getElementById("events-container");
const eventsList = document.getElementById("events-list");

// --- state ---
let profile = null; // PreferenceProfile from Agent 1; profile.orgs is mutated as chips are toggled/added
const orgActive = new Map(); // org_id -> bool (chip on/off state)
let latestFinalFeed = null; // last FinalFeed from Agent 3, re-filtered locally on checkbox change
const sentSignals = new Map(); // event_id -> "accept" | "skip", so re-renders keep the saved/skipped state

// --- small helpers ---

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.hidden = !message;
  statusEl.classList.toggle("error", isError);
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

function escapeAttr(value) {
  return (value ?? "").replace(/"/g, "&quot;");
}

function formatDate(isoString) {
  const parsed = new Date(isoString);
  if (Number.isNaN(parsed.getTime())) return isoString;
  return parsed.toLocaleString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function randomId(prefix) {
  if (window.crypto && crypto.randomUUID) return `${prefix}_${crypto.randomUUID()}`;
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

// --- Section 1: category tiles (toggle, no backend call) ---

categoryTiles.forEach((tile) => {
  tile.addEventListener("click", () => {
    const pressed = tile.getAttribute("aria-pressed") === "true";
    tile.setAttribute("aria-pressed", String(!pressed));
  });
});

function selectedCategories() {
  return Array.from(categoryTiles)
    .filter((t) => t.getAttribute("aria-pressed") === "true")
    .map((t) => t.dataset.category);
}

// --- Section 1 -> Agent 1 ---

findOrgsBtn.addEventListener("click", async () => {
  const rawText = rawTextEl.value.trim();
  const categories = selectedCategories();

  findOrgsBtn.disabled = true;
  orgsPanel.hidden = true;
  divider2.hidden = true;
  searchPanel.hidden = true;
  bestBetsContainer.hidden = true;
  eventsContainer.hidden = true;
  latestFinalFeed = null;
  sentSignals.clear();

  try {
    setStatus("Agent 1 is analyzing your interests and searching NYC orgs...");

    const res = await fetch("/agents/preference-profiler", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_text: rawText, selected_categories: categories }),
    });
    if (!res.ok) throw new Error(`Preference profiler failed (${res.status})`);

    profile = await res.json();
    orgActive.clear();
    for (const org of profile.orgs) orgActive.set(org.org_id, true);
    renderOrgChips();

    orgsPanel.hidden = false;
    divider2.hidden = false;
    searchPanel.hidden = false;
    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(`Something went wrong: ${err.message}`, true);
  } finally {
    findOrgsBtn.disabled = false;
  }
});

// --- Section 2: org chips + inline add form ---

function renderOrgChips() {
  orgChipsEl.innerHTML = "";

  if (profile.orgs.length === 0) {
    const span = document.createElement("span");
    span.className = "chip-empty";
    span.textContent = "No organizations found yet — add your own below.";
    orgChipsEl.appendChild(span);
    return;
  }

  for (const org of profile.orgs) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    const active = orgActive.get(org.org_id) !== false;
    chip.setAttribute("aria-pressed", String(active));
    chip.textContent = org.name;
    chip.title = org.category;
    chip.addEventListener("click", () => {
      const next = !(orgActive.get(org.org_id) !== false);
      orgActive.set(org.org_id, next);
      chip.setAttribute("aria-pressed", String(next));
    });
    orgChipsEl.appendChild(chip);
  }
}

addOrgForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const name = addOrgName.value.trim();
  if (!name || !profile) return;

  const org = {
    org_id: randomId("user"),
    name,
    category: addOrgCategory.value,
    source: "user_added",
  };
  profile.orgs.push(org);
  orgActive.set(org.org_id, true);
  renderOrgChips();
  addOrgName.value = "";
  addOrgName.focus();
});

// --- Section 3 -> Agent 2 -> Agent 3 ---

searchBtn.addEventListener("click", async () => {
  if (!profile) return;

  const activeOrgs = profile.orgs.filter((o) => orgActive.get(o.org_id) !== false);
  // Agent 2 reads profile.orgs directly, so send a copy with only the
  // currently-toggled-on orgs rather than mutating the shared profile object.
  const searchProfile = { ...profile, orgs: activeOrgs };

  searchBtn.disabled = true;
  bestBetsContainer.hidden = true;
  eventsContainer.hidden = true;
  latestFinalFeed = null;
  sentSignals.clear();

  try {
    setStatus("Agent 2 is searching for live events (this can take a minute for several orgs)...");

    const eventsRes = await fetch("/agents/event-retriever", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(searchProfile),
    });
    if (!eventsRes.ok) throw new Error(`Event retriever failed (${eventsRes.status})`);
    const rankedEvents = await eventsRes.json();

    setStatus("Agent 3 is curating your feed and checking the weekend forecast...");

    const feedRes = await fetch("/agents/curator-ranker", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: searchProfile, ranked_events: rankedEvents }),
    });
    if (!feedRes.ok) throw new Error(`Curator/ranker failed (${feedRes.status})`);
    const finalFeed = await feedRes.json();

    latestFinalFeed = finalFeed;
    renderResults();
    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(`Something went wrong: ${err.message}`, true);
  } finally {
    searchBtn.disabled = false;
  }
});

filterFree.addEventListener("change", renderResults);
filterWeekend.addEventListener("change", renderResults);

function isFree(price) {
  if (typeof price === "number") return price === 0;
  return String(price).trim().toLowerCase() === "free";
}

function isThisWeekend(dateStr) {
  const match = String(dateStr).match(/(\d{4}-\d{2}-\d{2})/);
  if (!match) return false;
  const eventDate = new Date(`${match[1]}T00:00:00`);
  if (Number.isNaN(eventDate.getTime())) return false;

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const day = today.getDay(); // 0=Sun ... 6=Sat

  let friday;
  if (day === 5 || day === 6 || day === 0) {
    // Already Fri/Sat/Sun -> this weekend started this week's Friday.
    const daysSinceFriday = day === 5 ? 0 : day === 6 ? 1 : 2;
    friday = new Date(today);
    friday.setDate(today.getDate() - daysSinceFriday);
  } else {
    friday = new Date(today);
    friday.setDate(today.getDate() + ((5 - day + 7) % 7));
  }
  const sunday = new Date(friday);
  sunday.setDate(friday.getDate() + 2);

  return eventDate >= friday && eventDate <= sunday;
}

function applyFilters(items) {
  return items.filter((item) => {
    if (filterFree.checked && !isFree(item.price)) return false;
    if (filterWeekend.checked && !isThisWeekend(item.date)) return false;
    return true;
  });
}

function renderResults() {
  if (!latestFinalFeed) return;

  const bestBetIds = new Set(latestFinalFeed.best_bets_this_weekend || []);
  const bestBets = applyFilters(latestFinalFeed.feed.filter((f) => bestBetIds.has(f.event_id)));
  const rest = applyFilters(latestFinalFeed.feed.filter((f) => !bestBetIds.has(f.event_id)));

  bestBetsList.innerHTML = "";
  if (bestBets.length > 0) {
    for (const item of bestBets) {
      bestBetsList.appendChild(buildEventCard(item, latestFinalFeed.user_id, true));
    }
    bestBetsContainer.hidden = false;
  } else {
    bestBetsContainer.hidden = true;
  }

  eventsList.innerHTML = "";
  if (rest.length === 0 && bestBets.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No events matched your filters.";
    eventsList.appendChild(li);
  } else {
    for (const item of rest) {
      eventsList.appendChild(buildEventCard(item, latestFinalFeed.user_id, false));
    }
  }
  eventsContainer.hidden = false;
}

function buildEventCard(item, userId, highlight) {
  const li = document.createElement("li");
  li.className = "event-card" + (highlight ? " highlight" : "");
  li.dataset.eventId = item.event_id;

  const sent = sentSignals.get(item.event_id);
  if (sent) li.classList.add(sent === "accept" ? "saved" : "skipped");

  li.innerHTML = `
    <a href="${escapeAttr(item.link)}" target="_blank" rel="noopener">${escapeHtml(item.title)}</a>
    <div class="event-meta">${formatDate(item.date)} &middot; ${escapeHtml(item.location)} &middot; ${escapeHtml(String(item.price))}</div>
    <p class="event-reason">${escapeHtml(item.reason)}</p>
    <div class="event-actions">
      <button type="button" class="thumb thumb-up" data-action="accept" ${sent ? "disabled" : ""}>👍 Save</button>
      <button type="button" class="thumb thumb-down" data-action="skip" ${sent ? "disabled" : ""}>👎 Skip</button>
    </div>
    ${sent ? `<div class="signal-note">${sent === "accept" ? "Saved ✓" : "Skipped"}</div>` : ""}
  `;

  li.querySelectorAll(".thumb").forEach((btn) => {
    btn.addEventListener("click", () => sendSignal(userId, item.event_id, btn.dataset.action, li));
  });

  return li;
}

async function sendSignal(userId, eventId, action, cardEl) {
  const buttons = cardEl.querySelectorAll(".thumb");
  buttons.forEach((b) => (b.disabled = true));

  try {
    const res = await fetch("/signals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        signals: [{ event_id: eventId, action, timestamp: new Date().toISOString() }],
      }),
    });
    if (!res.ok) throw new Error(`Signal failed (${res.status})`);

    sentSignals.set(eventId, action);
    cardEl.classList.remove("saved", "skipped");
    cardEl.classList.add(action === "accept" ? "saved" : "skipped");

    let note = cardEl.querySelector(".signal-note");
    if (!note) {
      note = document.createElement("div");
      note.className = "signal-note";
      cardEl.appendChild(note);
    }
    note.textContent = action === "accept" ? "Saved ✓" : "Skipped";
  } catch (err) {
    console.error(err);
    buttons.forEach((b) => (b.disabled = false));
    setStatus(`Couldn't save that — ${err.message}`, true);
  }
}
