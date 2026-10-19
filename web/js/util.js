// Small helpers shared by every module.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// Escape text before putting it into HTML (everything from the API is data).
export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Only allow real web links in href attributes.
export function safeUrl(url) {
  return typeof url === "string" && /^https?:\/\//i.test(url) ? esc(url) : "";
}

export const API = window.API_BASE || "";  // same server by default; set window.API_BASE for a separate backend

export function photoUrl(name, width = 640) {
  return name ? `${API}/api/photo?name=${encodeURIComponent(name)}&w=${width}` : "";
}

export const icon = (id, cls = "i") => `<svg class="${cls}"><use href="#i-${id}"/></svg>`;

export function stars(rating, reviews) {
  if (rating == null) return "";
  const count = reviews ? ` (${Number(reviews).toLocaleString()})` : "";
  return `${icon("star")}${Number(rating).toFixed(1)}${count}`;
}

export function price(level) {
  const n = { PRICE_LEVEL_INEXPENSIVE: 1, PRICE_LEVEL_MODERATE: 2, PRICE_LEVEL_EXPENSIVE: 3, PRICE_LEVEL_VERY_EXPENSIVE: 4 }[level];
  return n ? "$".repeat(n) : "";
}

// Friendly names for the pipeline steps (keys match recommender/pipeline.py).
export const STEP_LABELS = {
  warmup: "Warming up",
  orchestrator: "Planning which agents to use",
  preference: "Reading your request",
  memory: "Using what you told me before",
  discovery: "Turning it into map searches",
  weather: "Checking the weather",
  local_places: "Searching Google Maps",
  scoring: "Scoring every place found",
  matching: "Ranking the options",
  place_relevance: "Checking each place really fits",
  mobility: "Getting live drive times",
  venue_rag: "Reading reviews and websites",
  photo_vibe: "Looking at photos",
  synthesizer: "Writing up the picks",
  events: "Live events",
};

export function toast(message, ms = 2600) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), ms);
}

// Swap views with the View Transitions API when the browser has it.
export function transition(update) {
  if (document.startViewTransition && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    document.startViewTransition(update);
  } else {
    update();
  }
}

export const store = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode: memory just won't persist */ }
  },
};
