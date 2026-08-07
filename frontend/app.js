// NYC Event Scout — five-screen app (Landing / Step 1 Interests / Step 2
// Organizations / Step 3 Filters / Results), single page, client-side
// screen switching only (no new backend routes for navigation).
// Step 1 -> Agent 1. Step 2 is an editable org roster built from Agent 1's
// response. Step 3 -> Agent 2 -> Agent 3, rendered on Results as
// "Best bets this weekend" + the full ranked grid, each card wired to
// POST /signals on save/skip.

const statusEl = document.getElementById("status");

const themeToggle = document.getElementById("theme-toggle");

const landingCta = document.getElementById("landing-cta");

const categoryTiles = document.querySelectorAll(".tile");
const rawTextEl = document.getElementById("raw-text");
const findOrgsBtn = document.getElementById("find-orgs-btn");

const orgChipsEl = document.getElementById("org-chips");
const addOrgForm = document.getElementById("add-org-form");
const addOrgName = document.getElementById("add-org-name");
const addOrgCategory = document.getElementById("add-org-category");
const orgsContinueBtn = document.getElementById("orgs-continue-btn");

const filterFree = document.getElementById("filter-free");
const filterWeekend = document.getElementById("filter-weekend");
const searchBtn = document.getElementById("search-events-btn");

const bestBetsContainer = document.getElementById("best-bets-container");
const bestBetsList = document.getElementById("best-bets-list");
const eventsContainer = document.getElementById("events-container");
const eventsList = document.getElementById("events-list");
const agentStatusList = document.getElementById("agent-status-list");
const startOverBtn = document.getElementById("start-over-btn");

const CATEGORY_LABELS = {
  arts_culture: "Arts & Culture",
  parks_outdoors: "Parks & Outdoors",
  nightlife_bars: "Nightlife & Bars",
  food_restaurants: "Food & Restaurants",
  community_nonprofits: "Community & Nonprofits",
};

// Same glyphs as the step-1 tile icons (viewBox 0 0 20 20), reused on
// results event cards so a category reads the same way everywhere it's
// labeled, per the redesign brief.
const CATEGORY_ICON_SVG = {
  arts_culture:
    '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 16h14M4 16V7M8 16V7M12 16V7M16 16V7M3 7l7-4 7 4" /></svg>',
  parks_outdoors:
    '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 17c-4-1-7-4-7-9 0-2 1-3 2-3 6 0 11 4 11 9 0 1-.5 2-1 2M10 17V7" /></svg>',
  nightlife_bars:
    '<svg viewBox="0 0 20 20" fill="currentColor" stroke="none" aria-hidden="true"><path d="M13 4a7 7 0 100 12 8 8 0 010-12z" /></svg>',
  food_restaurants:
    '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 3v6M10 3v6M13 3v6M7 9c0 1.5 1.3 2 3 2s3-.5 3-2M10 11v6" /></svg>',
  community_nonprofits:
    '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="7" cy="6" r="2.2" fill="currentColor" stroke="none" /><circle cx="13" cy="6" r="2.2" fill="currentColor" stroke="none" /><path d="M2.5 17c0-3 2-5 4.5-5s4.5 2 4.5 5M8.5 17c0-3 2-5 4.5-5s4.5 2 4.5 5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>',
};

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

// --- theme toggle (persisted in localStorage; initial value already set
// on <html data-theme> by the inline head script, before first paint) ---

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeToggle.setAttribute("aria-pressed", String(theme === "dark"));
}

applyTheme(currentTheme());

themeToggle.addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next);
  localStorage.setItem("theme", next);
});

// --- screen navigation ---

const SCREEN_IDS = ["landing", "step1", "step2", "step3", "results"];

function showScreen(id) {
  for (const screenId of SCREEN_IDS) {
    document.getElementById(`screen-${screenId}`).classList.toggle("visible", screenId === id);
  }
  // Move focus to the new screen's heading so keyboard/screen-reader users
  // aren't left focused on a now-hidden element.
  const heading = document.getElementById(`${id}-heading`);
  if (heading) heading.focus();
}

function updateStepNav(activeStep) {
  document.querySelectorAll(".step-nav").forEach((nav) => {
    nav.querySelectorAll(".step-nav-item").forEach((item) => {
      const step = Number(item.dataset.step);
      item.dataset.state = step < activeStep ? "done" : step === activeStep ? "active" : "upcoming";
    });
  });
}

function setAgentStatus(agent, state) {
  const item = agentStatusList.querySelector(`.agent-status-item[data-agent="${agent}"]`);
  if (item) item.dataset.state = state;
}

function resetAgentStatus() {
  agentStatusList.querySelectorAll(".agent-status-item").forEach((item) => {
    item.dataset.state = "pending";
  });
}

landingCta.addEventListener("click", () => {
  updateStepNav(1);
  showScreen("step1");
});

startOverBtn.addEventListener("click", () => {
  profile = null;
  orgActive.clear();
  latestFinalFeed = null;
  sentSignals.clear();
  resetAgentStatus();
  setStatus("");
  updateStepNav(1);
  showScreen("landing");
});

// --- Step 1: category tiles (toggle, no backend call) ---

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

// --- Step 1 -> Agent 1 ---

findOrgsBtn.addEventListener("click", async () => {
  const rawText = rawTextEl.value.trim();
  const categories = selectedCategories();

  findOrgsBtn.disabled = true;
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

    setStatus("");
    updateStepNav(2);
    showScreen("step2");
  } catch (err) {
    console.error(err);
    setStatus(`Something went wrong: ${err.message}`, true);
  } finally {
    findOrgsBtn.disabled = false;
  }
});

// --- Step 2: org chips + inline add form ---

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

orgsContinueBtn.addEventListener("click", () => {
  updateStepNav(3);
  showScreen("step3");
});

// --- Step 3 -> Agent 2 -> Agent 3 ---

searchBtn.addEventListener("click", async () => {
  if (!profile) return;

  const activeOrgs = profile.orgs.filter((o) => orgActive.get(o.org_id) !== false);
  // Agent 2 reads profile.orgs directly, so send a copy with only the
  // currently-toggled-on orgs rather than mutating the shared profile object.
  const searchProfile = { ...profile, orgs: activeOrgs };

  searchBtn.disabled = true;
  latestFinalFeed = null;
  sentSignals.clear();
  bestBetsContainer.hidden = true;
  eventsContainer.hidden = true;

  // Profiler already ran in Step 1 — show Results immediately with live
  // status (not a faked "active" state after the fact) while Agent 2/3 run.
  resetAgentStatus();
  setAgentStatus("profiler", "done");
  setAgentStatus("retriever", "active");
  showScreen("results");

  try {
    setStatus("Agent 2 is searching for live events (this can take a minute for several orgs)...");

    const eventsRes = await fetch("/agents/event-retriever", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(searchProfile),
    });
    if (!eventsRes.ok) throw new Error(`Event retriever failed (${eventsRes.status})`);
    const rankedEvents = await eventsRes.json();

    setAgentStatus("retriever", "done");
    setAgentStatus("curator", "active");
    setStatus("Agent 3 is curating your feed and checking the weekend forecast...");

    const feedRes = await fetch("/agents/curator-ranker", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: searchProfile, ranked_events: rankedEvents }),
    });
    if (!feedRes.ok) throw new Error(`Curator/ranker failed (${feedRes.status})`);
    const finalFeed = await feedRes.json();

    setAgentStatus("curator", "done");
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
  li.dataset.category = item.category;

  const sent = sentSignals.get(item.event_id);
  if (sent) li.classList.add(sent === "accept" ? "saved" : "skipped");

  const icon = CATEGORY_ICON_SVG[item.category] || "";
  const label = CATEGORY_LABELS[item.category] || item.category;

  li.innerHTML = `
    <span class="category-badge">${icon}${escapeHtml(label)}</span>
    <a href="${escapeAttr(item.link)}" target="_blank" rel="noopener">${escapeHtml(item.title)}</a>
    <div class="event-meta"><span class="meta-mono">${formatDate(item.date)}</span> &middot; <span>${escapeHtml(item.location)}</span> &middot; <span class="meta-mono">${escapeHtml(String(item.price))}</span></div>
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
