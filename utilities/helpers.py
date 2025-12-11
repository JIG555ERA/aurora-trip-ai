"""General utility functions for the trip planner."""

import math
from typing import Optional


def normalize_city_name(city: str) -> str:
    """Convert 'Mumbai, India' → 'mumbai'. Helps with dict lookups."""
    if not city:
        return ""
    main = city.split(",")[0].strip().lower()
    return main


def to_irctc_date(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' to 'DD-MM-YYYY' for IRCTC."""
    parts = date_str.split("-")
    if len(parts) == 3 and len(parts[0]) == 4:
        yyyy, mm, dd = parts
        return f"{dd}-{mm}-{yyyy}"
    return date_str


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points (km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    d = 2 * R * math.asin(math.sqrt(a))
    return round(d, 2)
