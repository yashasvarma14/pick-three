// "What I remember": memory lives in this browser only (localStorage).

import { $, esc, store } from "./util.js";

const KEY = "pick-three.memory";
const GROUPS = [
  ["liked_activities", "Liked"], ["disliked_activities", "Not into"], ["liked_places", "Liked places"],
  ["rejected_places", "Skipped places"], ["positive_preferences", "More of"], ["negative_preferences", "Less of"],
];

export const memory = {
  get: () => store.get(KEY, {}),
  set(value) { store.set(KEY, value); render(); },
  useIt: () => $("#use-memory").checked,
};

function isEmpty(m) {
  return !GROUPS.some(([key]) => m[key]?.length) && !m.max_travel_minutes && !m.budget_preference;
}

function render() {
  const m = memory.get();
  $("#memory-dot").hidden = isEmpty(m);
  if (isEmpty(m)) {
    $("#memory-body").innerHTML = `<p class="muted">Nothing yet. After a search, tell me what you thought and it shows up here.</p>`;
    return;
  }
  const groups = GROUPS.filter(([key]) => m[key]?.length).map(([key, label]) => `
    <div class="mem-group"><h4>${label}</h4>
      <div class="mem-items">${m[key].map((v) => `<span class="mem-item">${esc(v)}</span>`).join("")}</div></div>`);
  if (m.max_travel_minutes) groups.push(`<div class="mem-group"><h4>Max drive</h4><div class="mem-items"><span class="mem-item">${esc(m.max_travel_minutes)} min</span></div></div>`);
  if (m.budget_preference) groups.push(`<div class="mem-group"><h4>Budget</h4><div class="mem-items"><span class="mem-item">${esc(m.budget_preference)}</span></div></div>`);
  $("#memory-body").innerHTML = groups.join("");
}

export function setupMemory() {
  const panel = $("#memory-panel");
  const backdrop = $("#sheet-backdrop");
  const open = () => { render(); panel.hidden = false; backdrop.hidden = false; $("#memory-close").focus(); };
  const close = () => { panel.hidden = true; backdrop.hidden = true; $("#memory-btn").focus(); };
  $("#memory-btn").addEventListener("click", open);
  $("#memory-close").addEventListener("click", close);
  backdrop.addEventListener("click", close);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) close(); });
  $("#memory-forget").addEventListener("click", () => memory.set({}));
  render();
}
