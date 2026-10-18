"""
All the page's CSS in one place (fonts, colors, the ticket-style result cards).
"""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Atkinson+Hyperlegible:wght@400;700&display=swap');

:root {
  --paper: #F3F5F2;
  --ink: #1E2A2B;
  --teal: #0E7C7B;
  --marigold: #F2B134;
  --muted: #5F6F6E;
  --stub: #FFFFFF;
}

html, body, .stMarkdown, .stTextArea textarea, .stButton button, [data-testid="stSidebar"] {
  font-family: 'Atkinson Hyperlegible', sans-serif;
}
h1, h2, h3, .ticket-name {
  font-family: 'Bricolage Grotesque', sans-serif;
  letter-spacing: -0.01em;
}
.hero-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 700;
  font-size: clamp(2.2rem, 5vw, 3.6rem);
  line-height: 1.02;
  color: var(--ink);
  margin: 0.2rem 0 0.4rem 0;
}
.hero-sub {
  color: var(--muted);
  font-size: 1.05rem;
  max-width: 62ch;
  margin-bottom: 1.2rem;
}

.ticket {
  display: grid;
  grid-template-columns: 76px 1fr;
  background: var(--stub);
  border: 1.5px solid var(--ink);
  border-radius: 14px;
  margin: 0 0 1.1rem 0;
  overflow: hidden;
}
.ticket-rank {
  background: var(--marigold);
  color: var(--ink);
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 700;
  font-size: 2.4rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border-right: 2px dashed var(--ink);
}
.ticket-body { padding: 1rem 1.25rem 1.1rem 1.25rem; }
.ticket-name { font-size: 1.45rem; font-weight: 700; color: var(--ink); margin: 0; }
.ticket-activity { color: var(--teal); font-weight: 700; margin: 0.15rem 0 0.6rem 0; }
.ticket-why { color: var(--ink); line-height: 1.55; max-width: 72ch; margin: 0 0 0.6rem 0; }
.ticket ul { margin: 0 0 0.6rem 1.1rem; padding: 0; color: var(--ink); }
.ticket li { margin: 0.1rem 0; }
.ticket-caveat { color: #8A4B00; font-size: 0.95rem; margin: 0 0 0.6rem 0; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.ticket-photos { display: flex; gap: 0.4rem; margin: 0 0 0.6rem 0; overflow-x: auto; }
.ticket-photos img { height: 96px; width: auto; max-width: 160px; object-fit: cover; border-radius: 6px; flex: 0 0 auto; }
.ticket-vibe { font-size: 0.95rem; margin: 0 0 0.6rem 0; }
.photo-credit { font-size: 0.75rem; opacity: 0.7; margin: -0.3rem 0 0.6rem 0; }
.chip {
  border: 1px solid #C9D2D0;
  border-radius: 999px;
  padding: 0.12rem 0.65rem;
  font-size: 0.88rem;
  color: var(--ink);
  background: var(--paper);
  text-decoration: none;
}
a.chip, a.chip:visited { color: var(--teal) !important; text-decoration: none !important; font-weight: 700; }
a.chip:hover, a.chip:focus-visible { border-color: var(--teal); outline: 2px solid var(--teal); outline-offset: 1px; }

@media (max-width: 640px) {
  .ticket { grid-template-columns: 52px 1fr; }
  .ticket-rank { font-size: 1.7rem; }
}
</style>
"""


def apply_styles():
    st.markdown(CSS, unsafe_allow_html=True)
