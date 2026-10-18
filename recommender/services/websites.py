"""Download the visible text of a venue's own website (for the review RAG agent)."""

import re

import numpy as np
import requests
from bs4 import BeautifulSoup

from ..core.cache import ttl_cache



USER_AGENT = "Mozilla/5.0 (compatible; activity-recommender-student-project/1.0)"


@ttl_cache(seconds=6 * 60 * 60)
def fetch_venue_website_text(url, timeout=4, max_characters=30000):
    """
    Visible text from a venue's public website, or None.
    A 4-second limit: a site slower than that rarely has useful text,
    and it was the single slowest wait in a run. Cached for 6 hours.
    """
    if url is None or (isinstance(url, float) and np.isnan(url)) or not str(url).strip():
        return None
    try:
        response = requests.get(str(url).strip(), headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", "").lower():
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        return text[:max_characters] if len(text) >= 100 else None
    except Exception:
        return None
