// Entry point: wires the page together.

import { getConfig } from "./api.js";
import { setupComposer } from "./composer.js";
import { setupFeatured } from "./featured.js";
import { setupMemory } from "./memory.js";
import { frostedHeader, revealOnScroll, splitWords } from "./motion.js";
import { goHome, runSearch, setupResults } from "./search.js";
import { $, $$ } from "./util.js";

// Entrance: headline words, then the subtitle and the search box.
splitWords($("#hero-title"));
[["#hero-sub", 450], ["#composer", 560]].forEach(([sel, delay]) => {
  $(sel).classList.add("fade-in");
  $(sel).style.setProperty("--d", `${delay}ms`);
});
frostedHeader($("#topbar"));
revealOnScroll();
setupMemory();
setupResults();

const status = $("#status-line");
const composer = setupComposer({
  onSubmit: async (body) => {
    status.textContent = "";
    status.classList.remove("error");
    composer.setBusy(true);
    try {
      await runSearch(body, { onDone: () => composer.setBusy(false) });
    } catch (err) {
      status.textContent = err.message;
      status.classList.add("error");
    }
  },
});

$$("[data-action='home']").forEach((el) => el.addEventListener("click", (e) => { e.preventDefault(); goHome(); }));
$$("[data-action='how']").forEach((el) => el.addEventListener("click", (e) => {
  if (document.body.dataset.view === "results") { e.preventDefault(); goHome(); setTimeout(() => $("#how").scrollIntoView({ behavior: "smooth" }), 350); }
}));

getConfig()
  .then((cfg) => {
    setupFeatured(cfg.categories);
    if (!cfg.ready) { status.textContent = "The site owner still needs to add API keys before searches work."; }
  })
  .catch(() => {
    $("#featured").hidden = true;
    status.textContent = "Can't reach the server right now.";
    status.classList.add("error");
  });
