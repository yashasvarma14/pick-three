"""Tiny helpers used in more than one place."""

import math
import re



def contains_keyword(text, keyword):
    """
    Whole-word match. Plain substring checks made "bar" match "barbecue"
    and "run" match "brunch" in the first version.
    """
    return re.search(rf"\b{re.escape(keyword)}\b", str(text).lower()) is not None


def distance_km(lat1, lng1, lat2, lng2):
    """Straight-line (haversine) distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))
