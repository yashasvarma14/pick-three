// Entry point: wires the page together.
// TODO (next session): search.js - progress checklist, result cards, map, feedback, trace.

import { getConfig } from "./api.js";
import { setupComposer } from "./composer.js";
import { setupFeatured } from "./featured.js";
import { setupMemory } from "./memory.js";
import { frostedHeader, revealOnScroll, splitWords } from "./motion.js";
import { $, toast } from "./util.js";

splitWords($("#hero-title"));
$("#hero-sub").classList.add("fade-in");
$("#hero-sub").style.setProperty("--d", "450ms");
$("#composer").classList.add("fade-in");
$("#composer").style.setProperty("--d", "560ms");
frostedHeader($("#topbar"));
revealOnScroll();
setupMemory();

setupComposer({
  onSubmit: () => toast("Search results view is still being built (see docs/NEXT_STEPS.md)."),
});

getConfig()
  .then((cfg) => setupFeatured(cfg.categories))
  .catch(() => { $("#featured").hidden = true; });
