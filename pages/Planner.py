import os
import math
import time
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from datetime import date, timedelta

import json
import requests
import streamlit as st
import pandas as pd
import folium
from streamlit.components.v1 import html as components_html
import pandas as pd

# 0. CONFIG & KEYS

# NOTE:
#   - For production, do NOT hardcode keys here.
#   - Instead set environment variables before running:
#       export GROQ_API_KEY="..."
#       export RAPIDAPI_KEY="..."
#       export AMADEUS_API_KEY="..."
#       export AMADEUS_API_SECRET="..."


# Logging setup

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

# Groq Client (LLM) - with graceful fallback

try:
    from groq import Groq  # type: ignore

    if __:
        groq_client = Groq(api_key=__)
        GROQ_AVAILABLE = True
        logging.info("Groq LLM available. Trip chatbot & AI helpers will use Llama 3.")
    else:
        groq_client = None
        GROQ_AVAILABLE = False
        logging.warning("GROQ_API_KEY not set; chatbot will fall back to keyword mode.")
except Exception as e:
    groq_client = None
    GROQ_AVAILABLE = False
    logging.warning("Groq client could not be initialized: %s. Using keyword fallback.", e)

# 1. HTTP HELPERS

def safe_get(
    url: str,
    params: Dict[str, Any] = None,
    headers: Dict[str, str] = None,
    timeout: int = 20,
) -> Optional[requests.Response]:
    headers = headers or {}
    headers.setdefault("User-Agent", "auroratrip-ai/1.0")
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp
        logging.warning("GET %s failed (%s): %s", url, resp.status_code, resp.text[:200])
    except Exception as e:
        logging.warning("GET %s raised %s", url, e)
    return None


def safe_post(
    url: str,
    data: Dict[str, Any],
    headers: Dict[str, str] = None,
    timeout: int = 20,
) -> Optional[requests.Response]:
    headers = headers or {}
    headers.setdefault("User-Agent", "auroratrip-ai/1.0")
    try:
        resp = requests.post(url, data=data, headers=headers, timeout=timeout)
        if resp.status_code in (200, 201):
            return resp
        logging.warning("POST %s failed (%s): %s", url, resp.status_code, resp.text[:200])
    except Exception as e:
        logging.warning("POST %s raised %s", url, e)
    return None

# 2. UTILITIES

def normalize_city_name(city: str) -> str:
    """Take 'Mumbai, India' → 'mumbai'. Helps with dict lookups."""
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
    """Great-circle distance between two points (km)."""
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

# 3. GEOCODING (Nominatim)

@st.cache_data(show_spinner=False)
def geocode_place(place: str) -> Optional[Dict[str, Any]]:
    """Turn a place name into lat/lon using Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "addressdetails": 1, "limit": 1}
    r = safe_get(url, params)
    if not r:
        return None
    data = r.json()
    if not data:
        return None
    d = data[0]
    return {
        "name": d["display_name"],
        "lat": float(d["lat"]),
        "lon": float(d["lon"]),
    }

# 4. WIKIPEDIA HELPERS (summary + images)

@st.cache_data(show_spinner=False)
def wiki_summary_and_image(title: str) -> Dict[str, Optional[str]]:
    """
    Return a dict with:
      - description: short text
      - image_url: thumbnail URL if available
    """
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
    r = safe_get(url)
    if not r:
        return {"description": "No description available yet.", "image_url": None}
    data = r.json()
    desc = data.get("extract") or "No description available yet."
    image_url = None
    thumb = data.get("thumbnail")
    if isinstance(thumb, dict):
        image_url = thumb.get("source")
    return {"description": desc, "image_url": image_url}


@st.cache_data(show_spinner=False)
def wiki_summary(title: str) -> str:
    """Backward-compatible wrapper for old code that only needs text."""
    info = wiki_summary_and_image(title)
    return info["description"]


@st.cache_data(show_spinner=False)
def wiki_coords(title: str) -> Optional[Dict[str, float]]:
    """Get coordinates for a Wikipedia page (if present)."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "coordinates",
        "format": "json",
    }
    r = safe_get(url, params)
    if not r:
        return None
    data = r.json().get("query", {}).get("pages", {})
    for _, page in data.items():
        coords = page.get("coordinates")
        if coords:
            c = coords[0]
            return {"lat": c["lat"], "lon": c["lon"]}
    return None

# 5. REDDIT COMMUNITY HINTS (KEYLESS)

@st.cache_data(show_spinner=False)
def reddit_posts(place: str, limit: int = 10) -> List[str]:
    """Lightweight peek into what travellers say on Reddit."""
    url = "https://www.reddit.com/search.json"
    params = {"q": f"{place} travel tips", "limit": limit, "sort": "relevance"}
    headers = {"User-Agent": "auroratrip-community-bot/1.0"}
    r = safe_get(url, params=params, headers=headers)
    if not r:
        return []
    children = r.json().get("data", {}).get("children", [])
    out = []
    for c in children:
        title = c.get("data", {}).get("title", "").strip()
        if title:
            out.append(title)
    return out


def summarize_sentences(snippets: List[str], max_sentences: int = 4) -> str:
    """Tiny 'summary' of Reddit post titles."""
    if not snippets:
        return "Travellers haven't shared much yet. You might be the one to write the first great story."
    joined = ". ".join(snippets)
    parts = [s.strip() for s in joined.split(".") if s.strip()]
    return ". ".join(parts[:max_sentences]) + "."

# 6. DATA MODELS

@dataclass
class Attraction:
    name: str
    category: str
    description: str
    lat: float
    lon: float
    suggested_hours: float
    approx_cost_inr: float
    image_url: Optional[str] = None   # NEW: thumbnail for UI


@dataclass
class RouteLeg:
    from_place: str
    to_place: str
    distance_km: float
    duration_min: float


@dataclass
class DayPlan:
    day: int
    theme: str
    attractions: List[Attraction]
    route_legs: List[RouteLeg]


@dataclass
class TrainOption:
    origin_station: str
    destination_station: str
    train_name: str
    train_no: str
    departure_time: str
    arrival_time: str
    duration_hours: float
    class_types: str
    available_classes: str
    source: str
    notes: str


@dataclass
class FlightOption:
    origin: str
    destination: str
    airline: str
    flight_no: str
    departure_time: str
    arrival_time: str
    duration_iso: str
    stops: int
    cabin_class: str
    price: float
    currency: str
    nonstop: bool
    source: str


@dataclass
class HotelOption:
    name: str
    lat: float
    lon: float
    approx_category: str
    source: str

# 7. CURATED ATTRACTIONS & FILTERS

CURATED_CITY_ATTRACTIONS: Dict[str, List[Dict[str, Any]]] = {
    "mumbai": [
        {"title": "Gateway of India", "category": "iconic", "hours": 2.0, "cost": 200},
        {"title": "Marine Drive", "category": "seafront", "hours": 2.0, "cost": 0},
        {"title": "Chhatrapati Shivaji Terminus railway station", "category": "heritage", "hours": 1.5, "cost": 0},
        {"title": "Elephanta Caves", "category": "heritage", "hours": 5.0, "cost": 700},
        {"title": "Siddhivinayak Temple", "category": "spiritual", "hours": 2.0, "cost": 200},
        {"title": "Haji Ali Dargah", "category": "spiritual", "hours": 2.0, "cost": 200},
        {"title": "Juhu", "category": "beach", "hours": 3.0, "cost": 200},
        {"title": "Bandra Fort", "category": "viewpoint", "hours": 2.0, "cost": 0},
        {"title": "Hanging Gardens Mumbai", "category": "garden", "hours": 2.0, "cost": 100},
        {"title": "Colaba Causeway", "category": "market", "hours": 2.5, "cost": 500},
    ],
    "delhi": [
        {"title": "India Gate", "category": "iconic", "hours": 1.5, "cost": 0},
        {"title": "Red Fort", "category": "heritage", "hours": 3.0, "cost": 600},
        {"title": "Qutb Minar", "category": "heritage", "hours": 2.0, "cost": 600},
        {"title": "Humayun's Tomb", "category": "heritage", "hours": 2.5, "cost": 600},
        {"title": "Lotus Temple", "category": "spiritual", "hours": 1.5, "cost": 0},
        {"title": "Akshardham", "category": "spiritual", "hours": 3.0, "cost": 300},
        {"title": "Connaught Place", "category": "market", "hours": 2.5, "cost": 600},
        {"title": "Hauz Khas Complex", "category": "heritage", "hours": 2.0, "cost": 300},
    ],
    "goa": [
        {"title": "Baga Beach", "category": "beach", "hours": 3.0, "cost": 300},
        {"title": "Calangute", "category": "beach", "hours": 3.0, "cost": 300},
        {"title": "Candolim", "category": "beach", "hours": 3.0, "cost": 300},
        {"title": "Fort Aguada", "category": "fort", "hours": 2.0, "cost": 200},
        {"title": "Basilica of Bom Jesus", "category": "heritage", "hours": 1.5, "cost": 200},
        {"title": "Dudhsagar Falls", "category": "nature", "hours": 6.0, "cost": 1000},
        {"title": "Palolem Beach", "category": "beach", "hours": 3.0, "cost": 300},
    ],
    "bengaluru": [
        {"title": "Lalbagh", "category": "garden", "hours": 2.0, "cost": 100},
        {"title": "Cubbon Park", "category": "garden", "hours": 2.0, "cost": 0},
        {"title": "Bangalore Palace", "category": "heritage", "hours": 2.0, "cost": 400},
        {"title": "ISKCON Temple Bangalore", "category": "spiritual", "hours": 2.0, "cost": 200},
        {"title": "UB City", "category": "mall", "hours": 2.0, "cost": 500},
        {"title": "Nandi Hills", "category": "viewpoint", "hours": 5.0, "cost": 600},
    ],
}

TOURIST_KEYWORDS = [
    "beach", "fort", "museum", "temple", "garden", "park", "lake",
    "palace", "monument", "heritage", "market", "bazaar",
    "church", "cathedral", "mosque", "viewpoint", "gallery",
    "historic", "cave", "zoo", "aquarium", "gateway", "square",
    "observatory", "tower", "exhibition", "memorial",
    "waterfall", "national park", "sanctuary", "reserve",
    "promenade", "sea face", "seaface", "cliff",
    "valley", "meadow", "hill", "peak",
]

BANNED_KEYWORDS = [
    "assembly constituency", "constituency", "riots", "slum",
    "stock exchange", "exchange", "industrial area", "corporate",
    "bank", "nse", "bse", "monorail station", "railway station",
    "metro station", "station", "college", "university",
    "medical college", "hospital", "court", "police",
    "municipality", "ward", "corporation", "residential area",
    "neighbourhood", "neighborhood", "bus stop", "school",
    "industrial estate",
]


def is_tourist_spot(name: str, desc: str) -> bool:
    """Rough filter to keep 'tourist-ish' spots and drop admin / slum etc."""
    nl = name.lower()
    dl = desc.lower()

    if any(b in nl for b in BANNED_KEYWORDS) or any(b in dl for b in BANNED_KEYWORDS):
        return False
    if any(k in nl for k in TOURIST_KEYWORDS):
        return True
    if any(k in dl for k in TOURIST_KEYWORDS):
        return True
    return False


@st.cache_data(show_spinner=False)
def wiki_geosearch(lat: float, lon: float, radius_m: int = 15_000, limit: int = 60) -> List[Dict[str, Any]]:
    """Nearby Wikipedia pages around a coordinate."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "geosearch",
        "gsradius": radius_m,
        "gscoord": f"{lat}|{lon}",
        "gslimit": limit,
        "format": "json",
    }
    r = safe_get(url, params)
    if not r:
        return []
    out = []
    for p in r.json().get("query", {}).get("geosearch", []):
        out.append(
            {
                "title": p["title"],
                "lat": p["lat"],
                "lon": p["lon"],
                "dist": p["dist"],
            }
        )
    return out

# 8. AI JSON HELPER

def _safe_json_from_text(text: str) -> Any:
    """
    Try to extract JSON from raw LLM output.
    - Strips ``` fences if present
    - Trims whitespace
    - Logs on failure instead of crashing
    """
    text = text.strip()
    # Strip fenced blocks if present
    if text.startswith("```"):
        # Remove first fence
        text = text.split("```", 1)[1]
        # Drop language tag if any
        if "\n" in text:
            text = text.split("\n", 1)[1]
        # Remove trailing fence
        if "```" in text:
            text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except Exception as e:
        logging.warning("Failed to parse LLM JSON: %s\nRaw: %.200s", e, text)
        return None

# 9. AI ATTRACTION GENERATION (Groq)

def ai_generate_attractions(
    city_name: str,
    lat: float,
    lon: float,
    days: int,
    target_count: int,
) -> List[Attraction]:
    """
    Use Groq Llama 3 to propose attractions when:
      - city is not curated, or
      - trip is long and we need more than curated/Wiki provides.

    Returns Attraction objects or [] on failure.
    """
    if not GROQ_AVAILABLE or groq_client is None:
        logging.info("Groq not available; skipping AI attraction generation.")
        return []

    target_count = max(1, min(target_count, 40))

    prompt = f"""
You are an Indian travel expert.

City: "{city_name}"
Approx coordinates: lat={lat:.5f}, lon={lon:.5f}
Trip length: {days} days.

Task:
Return a JSON array of EXACTLY {target_count} attractions that a traveller should visit
during a {days}-day trip to this city or nearby areas (within ~100 km).

Each item MUST be an object with:
- "name": short attraction name (e.g. "Solang Valley")
- "description": 1–3 sentence practical travel description
- "category": one of ["iconic","heritage","nature","beach","viewpoint","seafront",
                     "spiritual","garden","market","mall","museum","adventure","sight"]
- "suggested_hours": float (typical time to spend there, e.g. 1.5, 2.0, 3.0)
- "approx_cost_inr": float (rough local spend: entry + snacks, e.g. 0, 200, 500, 1000)

Rules:
- Output MUST be valid JSON only (no markdown, no extra text).
- Prefer diverse mix: viewpoints, markets, nature, heritage, etc.
- Favour well-known, Googleable attractions where possible.
"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You output only strict JSON (no markdown, no commentary).",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )
        raw_text = resp.choices[0].message.content
        data = _safe_json_from_text(raw_text)
    except Exception as e:
        logging.warning("Groq AI attraction generation failed: %s", e)
        return []

    if not isinstance(data, list):
        logging.warning("Groq attraction JSON was not a list.")
        return []

    attractions: List[Attraction] = []

    for item in data:
        try:
            name = str(item.get("name", "")).strip()
            if not name:
                continue

            desc = str(item.get("description", "")).strip() or "No description available yet."
            category = str(item.get("category", "sight")).lower()
            if category not in [
                "iconic", "heritage", "nature", "beach", "viewpoint", "seafront",
                "spiritual", "garden", "market", "mall", "museum", "adventure", "sight",
            ]:
                category = "sight"

            suggested_hours = float(item.get("suggested_hours", 2.0))
            approx_cost_inr = float(item.get("approx_cost_inr", 300.0))

            # Anchor on map: try Nominatim for the attraction itself
            geo_query = f"{name}, {city_name}, India"
            geo = geocode_place(geo_query)
            if geo:
                lat_a = geo["lat"]
                lon_a = geo["lon"]
            else:
                lat_a, lon_a = lat, lon

            # Try to get an image from Wikipedia, but keep AI description
            wiki_info = wiki_summary_and_image(name)
            image_url = wiki_info.get("image_url")

            attractions.append(
                Attraction(
                    name=name,
                    category=category,
                    description=desc,
                    lat=lat_a,
                    lon=lon_a,
                    suggested_hours=suggested_hours,
                    approx_cost_inr=approx_cost_inr,
                    image_url=image_url,
                )
            )
        except Exception as e:
            logging.warning("Failed to parse AI attraction item: %s", e)
            continue

    logging.info("AI generated %d attractions for %s", len(attractions), city_name)
    return attractions

# 10. ATTRACTIONS FETCHER & ITINERARY

def fetch_attractions_for_city(city_name: str, lat: float, lon: float, days: int) -> List[Attraction]:
    """
    Smart attraction engine:

    - Target attractions = days * 4 (min 4, max 40)
    - For curated cities: start from curated list, then optionally expand with AI
    - For others: start from Wikipedia GeoSearch, then expand with AI if needed
    - Uses Groq LLM when available, gracefully degrades otherwise
    """
    target_count = max(4, min(days * 4, 40))
    norm = normalize_city_name(city_name)
    curated = CURATED_CITY_ATTRACTIONS.get(norm)
    attractions: List[Attraction] = []

    # 1) Curated cities
    if curated:
        logging.info("Using curated attractions for %s", city_name)
        for spec in curated:
            if len(attractions) >= target_count:
                break

            title = spec.get("title")
            category = spec.get("category", "sight")
            hours = float(spec.get("hours", 2.0))
            cost = float(spec.get("cost", 300))

            wiki_info = wiki_summary_and_image(title)
            desc = wiki_info["description"]
            img_url = wiki_info["image_url"]

            coords = wiki_coords(title) or {}
            lat_a = coords.get("lat", lat)
            lon_a = coords.get("lon", lon)

            attractions.append(
                Attraction(
                    name=title,
                    category=category,
                    description=desc,
                    lat=lat_a,
                    lon=lon_a,
                    suggested_hours=hours,
                    approx_cost_inr=cost,
                    image_url=img_url,
                )
            )

    # 2) Wikipedia GeoSearch top-up if needed
    if len(attractions) < target_count:
        logging.info("Topping up with Wikipedia GeoSearch for %s", city_name)
        raw = wiki_geosearch(lat, lon, radius_m=8000, limit=60)
        for p in raw:
            if len(attractions) >= target_count:
                break
            title = p["title"]
            wiki_info = wiki_summary_and_image(title)
            desc = wiki_info["description"]
            img_url = wiki_info["image_url"]
            if not is_tourist_spot(title, desc):
                continue
            attractions.append(
                Attraction(
                    name=title,
                    category="sight",
                    description=desc,
                    lat=p["lat"],
                    lon=p["lon"],
                    suggested_hours=2.0,
                    approx_cost_inr=300.0,
                    image_url=img_url,
                )
            )

    # 3) AI booster if still short
    if len(attractions) < target_count:
        remaining = target_count - len(attractions)
        logging.info(
            "We have %d attractions for %s, need %d more – calling AI.",
            len(attractions),
            city_name,
            remaining,
        )
        extra = ai_generate_attractions(city_name, lat, lon, days, target_count=remaining)
        attractions.extend(extra)

    # Final clamp
    attractions = attractions[:target_count]
    logging.info(
        "Final attraction count for %s (%d days) = %d (target %d)",
        city_name,
        days,
        len(attractions),
        target_count,
    )
    return attractions


def build_itinerary(attractions: List[Attraction], days: int = 3) -> List[DayPlan]:
    """Split attractions into day-wise chunks with simple route legs."""
    days = max(1, days)
    if not attractions:
        return []

    category_priority = {
        "iconic": 1, "heritage": 1,
        "beach": 2, "viewpoint": 2, "seafront": 2,
        "spiritual": 3, "garden": 3, "nature": 3,
        "market": 4,
        "mall": 5,
        "sight": 6,
    }

    def cat_score(at: Attraction) -> int:
        return category_priority.get(at.category, 99)

    attractions_sorted = sorted(attractions, key=cat_score)

    plans: List[DayPlan] = []
    idx = 0
    for d in range(1, days + 1):
        chunk = attractions_sorted[idx:idx + 4]
        idx += 4
        if not chunk:
            break

        legs: List[RouteLeg] = []
        for i in range(len(chunk) - 1):
            start = chunk[i]
            end = chunk[i + 1]
            dist = haversine_km(start.lat, start.lon, end.lat, end.lon)
            duration_min = round((dist / 25.0) * 60.0) if dist > 0 else 15
            legs.append(
                RouteLeg(
                    from_place=start.name,
                    to_place=end.name,
                    distance_km=dist,
                    duration_min=duration_min,
                )
            )

        theme = "Highlights & must-see spots" if d == 1 else "Explore deeper"
        plans.append(DayPlan(day=d, theme=theme, attractions=chunk, route_legs=legs))

    return plans

# 11. TRAIN PLANNER (IRCTC + FALLBACK)

STATION_CODE_MAP = {
    "mumbai": "CSTM",
    "mumbai cst": "CSTM",
    "mumbai central": "BCT",
    "delhi": "NDLS",
    "new delhi": "NDLS",
    "bengaluru": "SBC",
    "bangalore": "SBC",
    "chennai": "MAS",
    "hyderabad": "HYB",
    "pune": "PUNE",
    "goa": "MAO",
    "madgaon": "MAO",
    "kolkata": "HWH",
    "howrah": "HWH",
    "jaipur": "JP",
    "ahmedabad": "ADI",
}


def city_to_station_code(city: str) -> Optional[str]:
    return STATION_CODE_MAP.get(normalize_city_name(city))


def irctc_train_between_stations(from_code: str, to_code: str, date_yyyy_mm_dd: str) -> List[TrainOption]:
    """Call IRCTC RapidAPI; return [] on any failure."""
    if not RAPIDAPI_KEY:
        logging.warning("RAPIDAPI_KEY not set; skipping live IRCTC trains.")
        return []

    query = {
        "fromStationCode": from_code,
        "toStationCode": to_code,
        "dateOfJourney": to_irctc_date(date_yyyy_mm_dd),
    }

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": IRCTC_RAPIDAPI_HOST,
    }

    for version in ("v3", "v1"):
        url = f"https://{IRCTC_RAPIDAPI_HOST}/api/{version}/trainBetweenStations"
        r = safe_get(url, params=query, headers=headers, timeout=20)
        if not r:
            continue
        js = r.json()
        data = js.get("data") or js.get("train") or []
        if not data:
            continue

        out: List[TrainOption] = []
        for t in data:
            try:
                name = t.get("train_name", "") or t.get("trainName", "")
                num = t.get("train_number", "") or t.get("trainNumber", "")
                dep = t.get("from_std", "") or t.get("fromStnTime", "")
                arr = t.get("to_sta", "") or t.get("toStnTime", "")
                dur_str = t.get("duration", "0:00")
                if ":" in dur_str:
                    h, m = dur_str.split(":")
                    dur_hours = int(h) + int(m) / 60
                else:
                    dur_hours = float(dur_str or 0)

                classes = t.get("available_classes") or t.get("classType") or []
                if isinstance(classes, list):
                    classes_joined = ",".join(
                        c.get("class_type", "") if isinstance(c, dict) else str(c)
                        for c in classes
                    )
                else:
                    classes_joined = str(classes)

                out.append(
                    TrainOption(
                        origin_station=from_code,
                        destination_station=to_code,
                        train_name=name,
                        train_no=num,
                        departure_time=dep,
                        arrival_time=arr,
                        duration_hours=dur_hours,
                        class_types=classes_joined,
                        available_classes=classes_joined,
                        source=f"IRCTC RapidAPI {version}",
                        notes="Live data from IRCTC API via RapidAPI",
                    )
                )
            except Exception:
                continue

        if out:
            return out

    logging.warning("IRCTC returned no trains for %s -> %s on %s", from_code, to_code, date_yyyy_mm_dd)
    return []


def sample_train_options(origin_city: str, dest_city: str, date_: str) -> List[TrainOption]:
    """Hardcoded sample trains for demo / fallback."""
    oc = normalize_city_name(origin_city)
    dc = normalize_city_name(dest_city)
    out: List[TrainOption] = []

    if ("mumbai" in oc and "delhi" in dc) or ("delhi" in oc and "mumbai" in dc):
        out.append(
            TrainOption(
                origin_station="CSTM",
                destination_station="NDLS",
                train_name="Rajdhani Express",
                train_no="12951",
                departure_time=f"{to_irctc_date(date_)} 16:35",
                arrival_time=f"{to_irctc_date(date_)} +1 08:30",
                duration_hours=15.9,
                class_types="3A,2A,1A",
                available_classes="3A,2A,1A",
                source="Sample",
                notes="Premium overnight Mumbai–Delhi train.",
            )
        )
        out.append(
            TrainOption(
                origin_station="BCT",
                destination_station="NDLS",
                train_name="Duronto Express",
                train_no="12263",
                departure_time=f"{to_irctc_date(date_)} 21:30",
                arrival_time=f"{to_irctc_date(date_)} +1 09:55",
                duration_hours=12.4,
                class_types="3A,2A",
                available_classes="3A,2A",
                source="Sample",
                notes="Non-stop overnight superfast express.",
            )
        )
    return out


def plan_trains(origin_city: str, dest_city: str, date_: str) -> List[TrainOption]:
    from_code = city_to_station_code(origin_city) or normalize_city_name(origin_city).upper()
    to_code = city_to_station_code(dest_city) or normalize_city_name(dest_city).upper()
    live = irctc_train_between_stations(from_code, to_code, date_)
    if live:
        return live
    return sample_train_options(origin_city, dest_city, date_)

# 12. FLIGHT PLANNER (Amadeus + FALLBACK)

CITY_TO_IATA = {
    "mumbai": "BOM",
    "delhi": "DEL",
    "new delhi": "DEL",
    "bengaluru": "BLR",
    "bangalore": "BLR",
    "goa": "GOI",
    "goa dabolim": "GOI",
    "goa mopa": "GOX",
    "pune": "PNQ",
    "chennai": "MAA",
    "hyderabad": "HYD",
    "kolkata": "CCU",
    "ahmedabad": "AMD",
    "jaipur": "JAI",
    "kochi": "COK",
    "cochin": "COK",
    "lucknow": "LKO",
    "nagpur": "NAG",
    "indore": "IDR",
    "bhopal": "BHO",
}


def city_to_iata(city: str) -> Optional[str]:
    return CITY_TO_IATA.get(normalize_city_name(city))


def amadeus_get_token() -> Optional[str]:
    if not (AMADEUS_API_KEY and AMADEUS_API_SECRET):
        logging.warning("Amadeus keys not set; skipping live flights.")
        return None

    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_API_KEY,
        "client_secret": AMADEUS_API_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = safe_post(url, data=data, headers=headers)
    if not r:
        return None
    return r.json().get("access_token")


def amadeus_flight_search(origin_iata: str, dest_iata: str, date_yyyy_mm_dd: str) -> List[FlightOption]:
    token = amadeus_get_token()
    if not token:
        return []

    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "originLocationCode": origin_iata,
        "destinationLocationCode": dest_iata,
        "departureDate": date_yyyy_mm_dd,
        "adults": 1,
        "currencyCode": "INR",
        "max": 10,
    }

    r = safe_get(url, params=params, headers=headers, timeout=20)
    if not r:
        return []

    data = r.json().get("data", [])
    out: List[FlightOption] = []

    for item in data:
        try:
            price = float(item["price"]["total"])
            currency = item["price"]["currency"]
            it = item["itineraries"][0]
            segments = it["segments"]
            first = segments[0]
            last = segments[-1]
            airline = first["carrierCode"]
            flight_no = first.get("number", "N/A")
            dep = first["departure"]["at"]
            arr = last["arrival"]["at"]
            duration_iso = it.get("duration", "")
            stops = len(segments) - 1

            out.append(
                FlightOption(
                    origin=first["departure"]["iataCode"],
                    destination=last["arrival"]["iataCode"],
                    airline=airline,
                    flight_no=flight_no,
                    departure_time=dep,
                    arrival_time=arr,
                    duration_iso=duration_iso,
                    stops=stops,
                    cabin_class="Economy",
                    price=price,
                    currency=currency,
                    nonstop=(stops == 0),
                    source="Amadeus Flight Offers API (test)",
                )
            )
        except Exception:
            continue
    return out


def sample_flight_options(origin_city: str, dest_city: str, date_: str) -> List[FlightOption]:
    """Minimal fallback flights for demo."""
    oc = normalize_city_name(origin_city)
    dc = normalize_city_name(dest_city)
    out: List[FlightOption] = []

    if ("mumbai" in oc and "delhi" in dc) or ("delhi" in oc and "mumbai" in dc):
        out.append(
            FlightOption(
                origin="BOM",
                destination="DEL",
                airline="IndiGo",
                flight_no="6E123",
                departure_time=f"{date_}T07:10:00",
                arrival_time=f"{date_}T09:15:00",
                duration_iso="PT2H5M",
                stops=0,
                cabin_class="Economy",
                price=4200.0,
                currency="INR",
                nonstop=True,
                source="Sample",
            )
        )
        out.append(
            FlightOption(
                origin="BOM",
                destination="DEL",
                airline="Vistara",
                flight_no="UK955",
                departure_time=f"{date_}T19:20:00",
                arrival_time=f"{date_}T21:30:00",
                duration_iso="PT2H10M",
                stops=0,
                cabin_class="Economy",
                price=5500.0,
                currency="INR",
                nonstop=True,
                source="Sample",
            )
        )
    return out


def plan_flights(origin_city: str, dest_city: str, date_: str) -> List[FlightOption]:
    origin_iata = city_to_iata(origin_city) or normalize_city_name(origin_city).upper()
    dest_iata = city_to_iata(dest_city) or normalize_city_name(dest_city).upper()
    live = amadeus_flight_search(origin_iata, dest_iata, date_)
    if live:
        return live
    return sample_flight_options(origin_city, dest_city, date_)

# 13. HOTELS VIA OSM OVERPASS

@st.cache_data(show_spinner=False)
def osm_hotels_near(lat: float, lon: float, radius_m: int = 5000) -> List[HotelOption]:
    """Simple Overpass query around the destination."""
    url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
      node["tourism"="hotel"](around:{radius_m},{lat},{lon});
      node["tourism"="guest_house"](around:{radius_m},{lat},{lon});
      node["tourism"="hostel"](around:{radius_m},{lat},{lon});
    );
    out body;
    """
    try:
        r = requests.post(url, data={"data": query}, timeout=25)
    except Exception as e:
        logging.warning("Overpass request failed: %s", e)
        return []

    if r.status_code != 200:
        logging.warning("Overpass returned %s: %s", r.status_code, r.text[:200])
        return []

    js = r.json()
    out: List[HotelOption] = []

    for el in js.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue
        lat_h = el.get("lat")
        lon_h = el.get("lon")

        t_type = tags.get("tourism")
        if t_type == "hostel":
            cat = "budget"
        elif t_type == "guest_house":
            cat = "mid"
        else:
            cat = "mid"

        out.append(
            HotelOption(
                name=name,
                lat=lat_h,
                lon=lon_h,
                approx_category=cat,
                source="OSM Overpass",
            )
        )

    return out


def group_hotels(hotels: List[HotelOption]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {"budget": [], "mid": [], "luxury": [], "unknown": []}
    for h in hotels:
        cat = h.approx_category
        if cat not in grouped:
            cat = "unknown"
        grouped[cat].append(asdict(h))
    return grouped

# 14. TRIP CORE BUILDER

def plan_trip_core(destination: str, days: int = 3) -> Dict[str, Any]:
    """One clean object that holds: coordinates, attractions, itinerary, community summary."""
    geo = geocode_place(destination)
    if not geo:
        raise ValueError(f"Could not geocode destination: {destination}")

    lat, lon = geo["lat"], geo["lon"]
    pretty_name = geo["name"]

    logging.info("Planning trip for %s (%.4f, %.4f)", pretty_name, lat, lon)

    attractions = fetch_attractions_for_city(pretty_name, lat, lon, days=days)
    itinerary = build_itinerary(attractions, days=days)
    posts = reddit_posts(pretty_name, limit=12)
    community_summary = summarize_sentences(posts)

    return {
        "destination": pretty_name,
        "coordinates": {"lat": lat, "lon": lon},
        "attractions": [asdict(a) for a in attractions],
        "itinerary": [
            {
                "day": dp.day,
                "theme": dp.theme,
                "attractions": [asdict(a) for a in dp.attractions],
                "route_legs": [asdict(r) for r in dp.route_legs],
            }
            for dp in itinerary
        ],
        "community_summary": community_summary,
    }

# 15. HOLIDAY & WEEKEND ANALYSIS (Hybrid H3)

# Simple core Indian holidays (month, day) - repeat every year
INDIA_FIXED_HOLIDAYS = [
    (1, 26, "Republic Day"),
    (8, 15, "Independence Day"),
    (10, 2, "Gandhi Jayanti"),
    (12, 25, "Christmas"),
]

# Approx festival windows (just month for soft reasoning)
INDIA_FESTIVAL_MONTHS = {
    3: "Holi season (dates vary)",
    10: "Dussehra / Diwali season (dates vary)",
    11: "Diwali season (dates vary)",
}


def analyze_trip_dates(start_date_iso: str, days: int) -> Dict[str, Any]:
    """
    Hybrid date analysis:
      - Detect weekends
      - Detect fixed national holidays (month/day)
      - Detect festival month
      - Suggest alternative 'cheaper / calmer' dates
    """
    try:
        d0 = date.fromisoformat(start_date_iso)
    except Exception:
        return {
            "start_date": start_date_iso,
            "days": days,
            "dates": [],
            "weekend_dates": [],
            "holiday_hits": [],
            "festival_hint": None,
            "is_weekend_heavy": False,
            "suggested_alternatives": {},
        }

    dates = [d0 + timedelta(days=i) for i in range(max(days, 1))]
    weekend_dates = [d for d in dates if d.weekday() >= 5]

    holiday_hits = []
    for d in dates:
        for m, day_, name in INDIA_FIXED_HOLIDAYS:
            if d.month == m and d.day == day_:
                holiday_hits.append({"date": d.isoformat(), "name": name})

    festival_hint = INDIA_FESTIVAL_MONTHS.get(d0.month)

    is_weekend_heavy = len(weekend_dates) / max(len(dates), 1) > 0.4

    # Suggest alternative start dates to avoid crowds:
    #   - If start date is Fri/Sat/Sun -> suggest next Monday
    #   - If start is on holiday -> suggest +2 days
    alt_dates: Dict[str, str] = {}

    # Basic "calmer" suggestion
    calmer = d0
    if d0.weekday() >= 4:  # Fri/Sat/Sun
        days_to_monday = (7 - d0.weekday()) % 7
        if days_to_monday == 0:
            days_to_monday = 7
        calmer = d0 + timedelta(days=days_to_monday)
    elif holiday_hits:
        calmer = d0 + timedelta(days=2)
    alt_dates["calmer_start"] = calmer.isoformat()

    # "Cheaper" suggestion: a weekday (Tue/Wed/Thu) near original
    cheaper = d0
    while cheaper.weekday() in (4, 5, 6):  # Fri-Sun
        cheaper = cheaper + timedelta(days=1)
    alt_dates["cheaper_start"] = cheaper.isoformat()

    return {
        "start_date": d0.isoformat(),
        "days": days,
        "dates": [d.isoformat() for d in dates],
        "weekend_dates": [d.isoformat() for d in weekend_dates],
        "holiday_hits": holiday_hits,
        "festival_hint": festival_hint,
        "is_weekend_heavy": is_weekend_heavy,
        "suggested_alternatives": alt_dates,
    }

# 16. HYBRID BUDGET CALCULATOR (Option C)

BUDGET_LEVEL_CONFIG = {
    "Budget": {
        "hotel_per_night": 1200,
        "food_per_day": 400,
        "local_per_day": 250,
        "buffer_pct": 0.08,
    },
    "Mid": {
        "hotel_per_night": 2500,
        "food_per_day": 650,
        "local_per_day": 350,
        "buffer_pct": 0.10,
    },
    "Luxury": {
        "hotel_per_night": 6000,
        "food_per_day": 1200,
        "local_per_day": 600,
        "buffer_pct": 0.15,
    },
}


def estimate_transport_cost(transport: Dict[str, Any]) -> float:
    """Rough estimate from available flights/trains."""
    trains = transport.get("trains", [])
    flights = transport.get("flights", [])
    estimate = 0.0

    # Prefer flights if we have price
    flight_prices = [f.get("price", 0) for f in flights if f.get("price")]
    if flight_prices:
        estimate = min(flight_prices)

    # If no flights or they are crazy, fallback to simple train estimate
    if not estimate and trains:
        # simple flat-ish estimate
        estimate = 900.0

    return float(estimate)


def estimate_budget(
    guide: Dict[str, Any],
    transport: Dict[str, Any],
    hotels_grouped: Dict[str, List[Dict[str, Any]]],
    trip_days: int,
    budget_level: str,
    date_info: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = BUDGET_LEVEL_CONFIG.get(budget_level, BUDGET_LEVEL_CONFIG["Mid"])

    hotel_per_night = cfg["hotel_per_night"]
    food_per_day = cfg["food_per_day"]
    local_per_day = cfg["local_per_day"]
    buffer_pct = cfg["buffer_pct"]

    attractions = guide.get("attractions", [])
    total_attr_cost = float(sum(a.get("approx_cost_inr", 0.0) for a in attractions))

    hotel_cost = hotel_per_night * max(trip_days, 1)
    food_cost = food_per_day * max(trip_days, 1)
    local_cost = local_per_day * max(trip_days, 1)
    transport_cost = estimate_transport_cost(transport)

    base_total = hotel_cost + food_cost + local_cost + total_attr_cost + transport_cost

    weekend_dates = date_info.get("weekend_dates", [])
    holiday_hits = date_info.get("holiday_hits", [])
    weekend_ratio = len(weekend_dates) / max(trip_days, 1)

    weekend_factor = 1.0 + 0.15 * weekend_ratio   # up to +15% if fully weekend
    holiday_factor = 1.10 if holiday_hits else 1.0

    with_multipliers = base_total * weekend_factor * holiday_factor
    buffer_amount = with_multipliers * buffer_pct

    low_estimate = with_multipliers * 0.92
    high_estimate = with_multipliers * 1.12 + buffer_amount
    recommended = (low_estimate + high_estimate) / 2.0

    breakdown = {
        "hotel_cost": hotel_cost,
        "food_cost": food_cost,
        "local_cost": local_cost,
        "attraction_cost": total_attr_cost,
        "transport_cost": transport_cost,
        "buffer_amount": buffer_amount,
        "weekend_factor": weekend_factor,
        "holiday_factor": holiday_factor,
    }

    return {
        "budget_level": budget_level,
        "trip_days": trip_days,
        "base_total": base_total,
        "low_estimate": low_estimate,
        "high_estimate": high_estimate,
        "recommended": recommended,
        "breakdown": breakdown,
    }

# 17. AI NARRATIVE FOR BUDGET + DATES (H3)

def ai_budget_and_date_advice(
    destination: str,
    budget_info: Dict[str, Any],
    date_info: Dict[str, Any],
) -> str:
    """Use Groq to generate a human explanation for budget and dates."""
    if not GROQ_AVAILABLE or groq_client is None:
        # Simple fallback
        msg = []
        msg.append(f"For **{destination}**, a **{budget_info['budget_level']}** style trip "
                   f"over {budget_info['trip_days']} days looks roughly like this.")
        msg.append(
            f"Recommended total budget: about ₹{int(budget_info['recommended']):,} "
            f"(range ₹{int(budget_info['low_estimate']):,} – ₹{int(budget_info['high_estimate']):,})."
        )

        holidays = date_info.get("holiday_hits", [])
        if holidays:
            names = ", ".join(h["name"] for h in holidays)
            msg.append(f"Your chosen dates include Indian holidays: {names}. Expect higher demand and crowds.")
        if date_info.get("is_weekend_heavy"):
            msg.append("A large part of your trip is on weekends, so prices and crowds may be higher.")

        alt = date_info.get("suggested_alternatives", {})
        if alt:
            msg.append(
                f"If you want a calmer / potentially cheaper experience, "
                f"you could start around **{alt.get('cheaper_start', alt.get('calmer_start'))}** instead."
            )

        return "\n\n".join(msg)

    try:
        breakdown = budget_info["breakdown"]
        holidays = date_info.get("holiday_hits", [])
        weekend_dates = date_info.get("weekend_dates", [])
        alt = date_info.get("suggested_alternatives", {})

        user_context = {
            "destination": destination,
            "budget_level": budget_info["budget_level"],
            "trip_days": budget_info["trip_days"],
            "low_estimate": int(budget_info["low_estimate"]),
            "high_estimate": int(budget_info["high_estimate"]),
            "recommended": int(budget_info["recommended"]),
            "breakdown": {k: int(v) if isinstance(v, (int, float)) else v for k, v in breakdown.items()},
            "holidays": holidays,
            "weekend_dates": weekend_dates,
            "alternatives": alt,
        }

        prompt = (
            "You are AuroraTrip AI, an Indian travel advisor.\n\n"
            "The user has a planned trip and we computed a budget + date analysis for them.\n"
            "You must explain:\n"
            "  - Rough total budget in INR\n"
            "  - What that budget includes (stays, food, local travel, attractions, transport)\n"
            "  - Impact of weekends / holidays on price and crowds\n"
            "  - Any alternative start dates that might be calmer / cheaper\n"
            "  - 2–3 short practical suggestions.\n\n"
            "Keep it conversational, clear, and honest. DO NOT invent exact live prices or exact hotel names.\n"
            "Use bullet points where helpful.\n\n"
            f"Here is the structured context (JSON):\n{json.dumps(user_context, indent=2)}"
        )

        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are AuroraTrip AI, a friendly, practical Indian travel advisor. "
                        "Speak in clear English with short paragraphs and bullet points when needed."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logging.warning("Groq budget/date narrative failed: %s", e)
        return (
            "Here’s a rough read: your budget covers hotels, food, local travel, attractions, and basic transport.\n"
            "Weekends/holidays will push prices and crowds up a bit, so shifting by a day or two towards weekdays "
            "can help both your wallet and peace of mind."
        )

# 18. TRAVEL CHATBOT (Groq + fallback)

class TravelChatbot:
    """
    A small wrapper that:
    - Builds a text context from the planned trip, transport & hotels
    - Answers questions using Groq (if available)
    - Falls back to simple keyword-based answers otherwise
    - Can answer ANY general question, not only travel
    """

    def __init__(
        self,
        guide: Dict[str, Any],
        transport: Dict[str, Any],
        hotels_grouped: Dict[str, List[Dict[str, Any]]],
    ):
        self.guide = guide
        self.transport = transport
        self.hotels = hotels_grouped
        self.history: List[Dict[str, str]] = []
        self.context_text = self._build_context()

    def _build_context(self) -> str:
        g = self.guide
        lines: List[str] = []
        lines.append(f"Destination: {g['destination']}.")
        lines.append(f"Community notes: {g['community_summary']}.")

        lines.append("\nAttractions:")
        for a in g.get("attractions", [])[:50]:
            lines.append(f"- {a['name']}: {a['description']}")

        lines.append("\nDay-wise itinerary:")
        for day in g.get("itinerary", []):
            lines.append(f"Day {day['day']} ({day['theme']}):")
            for a in day["attractions"]:
                lines.append(f"  • {a['name']} ({a['category']}, ~{a['suggested_hours']} hours)")
            for leg in day["route_legs"]:
                lines.append(
                    f"  Travel {leg['from_place']} → {leg['to_place']} "
                    f"~ {leg['distance_km']} km, {leg['duration_min']} minutes."
                )

        if self.transport:
            lines.append("\nTrains:")
            for t in self.transport.get("trains", []):
                lines.append(
                    f"{t['train_name']} ({t['train_no']}) {t['origin_station']} → {t['destination_station']} "
                    f"| ~{t['duration_hours']} hours | classes: {t['available_classes']}"
                )
            lines.append("\nFlights:")
            for f in self.transport.get("flights", []):
                lines.append(
                    f"{f['airline']} {f['flight_no']} {f['origin']} → {f['destination']} "
                    f"| {f['duration_iso']} | ₹{f['price']} {f['currency']} "
                    f"| {'non-stop' if f['nonstop'] else str(f['stops']) + ' stop(s)'}"
                )

        if self.hotels:
            lines.append("\nHotels by category:")
            for cat, hs in self.hotels.items():
                for h in hs[:10]:
                    lines.append(f"{cat.title()} – {h['name']}")

        return "\n".join(lines)

    def ask(self, user_message: str) -> str:
        """Public entrypoint: takes user input, returns answer string."""
        self.history.append({"role": "user", "content": user_message})
        if GROQ_AVAILABLE and groq_client is not None:
            answer = self._answer_with_groq(user_message)
        else:
            answer = self._answer_with_keywords(user_message)
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def _answer_with_groq(self, question: str) -> str:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are AuroraTrip AI, a friendly, intelligent assistant. "
                            "You can answer ANY question the user asks, not just travel-related queries. "
                            "Whenever useful, incorporate the provided trip context to make your answer better. "
                            "If the user's question is unrelated to travel, answer normally like a general AI assistant. "
                            "Be helpful, clear, and conversational."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Here is the travel context for reference:\n{self.context_text}\n\n"
                            f"User message: {question}"
                        ),
                    },
                ],
                temperature=0.7,
                max_tokens=700,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning("Groq failed: %s", e)
            return (
                "AuroraTrip AI is having trouble reaching the cloud right now. "
                "You can still follow the generated itinerary, transport and stays shown in the app, "
                "or try asking again in a moment."
            )

    def _answer_with_keywords(self, question: str) -> str:
        """Simple rule-based assistant used if Groq is unavailable."""
        q = question.lower()

        # Ask about itinerary or day-wise plan
        if "day" in q or "itinerary" in q or "plan" in q:
            lines = ["Here is your planned itinerary:"]
            for day in self.guide.get("itinerary", []):
                lines.append(f"Day {day['day']} – {day['theme']}:")
                for a in day["attractions"]:
                    lines.append(f"  • {a['name']}")
            return "\n".join(lines)

        # Ask about trains
        if "train" in q:
            trains = self.transport.get("trains", [])
            if trains:
                lines = ["Here are some train options:"]
                for t in trains:
                    lines.append(
                        f"{t['train_name']} ({t['train_no']}) {t['origin_station']} → {t['destination_station']} "
                        f"| ~{t['duration_hours']} hours | classes: {t['available_classes']}"
                    )
                return "\n".join(lines)

        # Ask about flights
        if any(word in q for word in ["flight", "air", "plane"]):
            flights = self.transport.get("flights", [])
            if flights:
                lines = ["Here are some flight options:"]
                for f in flights:
                    lines.append(
                        f"{f['airline']} {f['flight_no']} {f['origin']} → {f['destination']} "
                        f"| {f['duration_iso']} | approx ₹{f['price']} {f['currency']} "
                        f"| {'non-stop' if f['nonstop'] else str(f['stops'])+' stop(s)'}"
                    )
                return "\n".join(lines)

        # Ask about stays
        if any(word in q for word in ["hotel", "stay", "accommodation"]):
            if self.hotels:
                lines = ["Here are some stay options grouped by budget:"]
                for cat, hs in self.hotels.items():
                    if not hs:
                        continue
                    lines.append(f"{cat.title()} options:")
                    for h in hs[:5]:
                        lines.append(f"  • {h['name']}")
                return "\n".join(lines)

        # Ask about a specific attraction by name
        for a in self.guide.get("attractions", []):
            if a["name"].lower() in q:
                return f"{a['name']}: {a['description']}"

        # Generic fallback
        dest = self.guide.get("destination", "this place")
        return (
            f"I'm not fully sure about that specific detail, but based on your guide for {dest}, "
            f"you can follow the day-wise plan, choose any of the listed train/flight options that fit your budget, "
            f"and pick a stay from the grouped hotel suggestions. For live availability and last-minute changes, "
            f"please double-check on official booking sites."
        )

# 19. MASTER TRIP BUILDER

def build_full_trip(
    destination: str,
    days: int,
    origin_city_for_transport: Optional[str],
    date_for_transport: Optional[str],
) -> Dict[str, Any]:
    """Glue everything together into one structured object."""
    guide = plan_trip_core(destination, days=days)
    lat = guide["coordinates"]["lat"]
    lon = guide["coordinates"]["lon"]

    # Transport (optional)
    if origin_city_for_transport and date_for_transport:
        trains = plan_trains(origin_city_for_transport, destination, date_for_transport)
        flights = plan_flights(origin_city_for_transport, destination, date_for_transport)
        transport = {
            "origin_city": origin_city_for_transport,
            "destination_city": destination,
            "date": date_for_transport,
            "trains": [asdict(t) for t in trains],
            "flights": [asdict(f) for f in flights],
        }
    else:
        transport = {"trains": [], "flights": []}

    # Stays
    hotels_raw = osm_hotels_near(lat, lon, radius_m=5000)
    hotels_grouped = group_hotels(hotels_raw)

    # Chatbot (doesn't directly depend on budget/date here)
    bot = TravelChatbot(guide, transport, hotels_grouped)

    return {
        "guide": guide,
        "transport": transport,
        "hotels": hotels_grouped,
        "bot": bot,
    }

# 20. STREAMLIT UI

st.set_page_config(
    page_title="AuroraTrip AI",
    page_icon="🛸",
    layout="wide",
)

st.title(":grey[AuroraTrip AI]")
# 3D animated background (Vanta.js GLOBE)
# components_html(
#     """
#     <div id="aurora-bg"></div>
#     <style>
#     #aurora-bg {
#         position: fixed;
#         top: 0;
#         left: 0;
#         width: 100vw;
#         height: 100vh;
#         z-index: -2;
#     }
#     </style>
#     <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.min.js"></script>
#     <script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.globe.min.js"></script>
#     <script>
#     window.addEventListener('load', function() {
#         if (!window.VANTA) return;
#         VANTA.GLOBE({
#           el: "#aurora-bg",
#           mouseControls: true,
#           touchControls: true,
#           minHeight: 200.00,
#           minWidth: 200.00,
#           scale: 1.00,
#           scaleMobile: 1.00,
#           color: 0x22c1c3,
#           color2: 0xfb5e7e,
#           backgroundColor: 0x050510,
#           size: 1.1
#         });
#     });
#     </script>
#     """,
#     height=0,
# )

# Custom CSS for glass UI & subtle animations
# st.markdown(
#     """
#     <style>
#     body, .stApp {
#         background: radial-gradient(circle at top left, #141726, #050511 60%);
#         color: #f5f5f5;
#     }
#     #MainMenu, footer {visibility: hidden;}
#     .main {background: transparent;}

#     .glass-card {
#         background: rgba(15, 23, 42, 0.90);
#         border-radius: 18px;
#         padding: 1.2rem 1.4rem;
#         border: 1px solid rgba(148,163,184,0.4);
#         box-shadow: 0 20px 40px rgba(0,0,0,0.45);
#         backdrop-filter: blur(18px);
#         -webkit-backdrop-filter: blur(18px);
#         position: relative;
#         overflow: hidden;
#     }

#     .glass-card::before {
#         content: "";
#         position: absolute;
#         inset: -40%;
#         background: radial-gradient(circle at top, rgba(56,189,248,0.28), transparent 60%);
#         opacity: 0.75;
#         mix-blend-mode: screen;
#         transform: translate3d(-20px, -40px, 0);
#         pointer-events: none;
#     }

#     .glass-card-lite {
#         background: rgba(15, 23, 42, 0.92);
#         border-radius: 16px;
#         padding: 1rem 1.2rem;
#         border: 1px solid rgba(148,163,184,0.5);
#         box-shadow: 0 14px 30px rgba(0,0,0,0.45);
#         backdrop-filter: blur(12px);
#         -webkit-backdrop-filter: blur(12px);
#     }

#     .hero-title {
#         font-size: 2.7rem;
#         font-weight: 800;
#         background: linear-gradient(120deg, #38bdf8, #a855f7, #f97316);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         letter-spacing: 0.05em;
#     }

#     .hero-subtitle {
#         font-size: 0.95rem;
#         color: #cbd5f5;
#         margin-top: 0.4rem;
#     }

#     .pill-badge {
#         display: inline-flex;
#         align-items: center;
#         gap: 0.4rem;
#         padding: 0.25rem 0.7rem;
#         border-radius: 999px;
#         background: rgba(15, 118, 110, 0.2);
#         color: #5eead4;
#         font-size: 0.75rem;
#         border: 1px solid rgba(45, 212, 191, 0.4);
#     }

#     .metric-label {
#         font-size: 0.8rem;
#         color: #9ca3af;
#         text-transform: uppercase;
#         letter-spacing: 0.08em;
#     }

#     .metric-value {
#         font-size: 1.3rem;
#         font-weight: 700;
#         color: #e5e7eb;
#     }

#     .section-title {
#         font-size: 1.1rem;
#         font-weight: 700;
#         margin-bottom: 0.4rem;
#         color: #e5e7eb;
#     }

#     .section-subtitle {
#         font-size: 0.85rem;
#         color: #9ca3af;
#         margin-bottom: 0.6rem;
#     }

#     .chip {
#         display: inline-flex;
#         align-items: center;
#         padding: 0.15rem 0.6rem;
#         border-radius: 999px;
#         font-size: 0.7rem;
#         border: 1px solid rgba(148,163,184,0.6);
#         color: #e5e7eb;
#         margin-right: 0.3rem;
#         margin-bottom: 0.2rem;
#     }

#     .chat-bubble-user {
#         background: linear-gradient(135deg, #0f766e, #0ea5e9);
#         color: white;
#         padding: 0.55rem 0.8rem;
#         border-radius: 14px;
#         margin-bottom: 0.35rem;
#         margin-left: auto;
#         max-width: 90%;
#     }

#     .chat-bubble-bot {
#         background: rgba(15, 23, 42, 0.9);
#         color: #e5e7eb;
#         padding: 0.55rem 0.8rem;
#         border-radius: 14px;
#         margin-bottom: 0.35rem;
#         margin-right: auto;
#         max-width: 90%;
#         border: 1px solid rgba(148,163,184,0.5);
#     }

#     .chat-container {
#         max-height: 450px;
#         overflow-y: auto;
#         padding-right: 0.5rem;
#     }

#     .plan-button button {
#         width: 100%;
#         border-radius: 999px !important;
#         font-weight: 700 !important;
#         padding: 0.6rem 1rem !important;
#         box-shadow: 0 10px 30px rgba(56, 189, 248, 0.4) !important;
#         background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
#         border: none !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# Session state setup
if "trip" not in st.session_state:
    st.session_state.trip: Dict[str, Any] = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []

# Sidebar Controls

with st.sidebar:
    st.markdown("### 🧭 Trip controls")
    st.write("Configure your AI-powered journey.")

    origin_city = st.text_input("Departure city", value="Delhi")
    destination = st.text_input("Destination", value="Kashmir, India")
    trip_days = st.slider("Trip length (days)", 1, 10, 3)

    trip_date = st.date_input(
        "Outbound travel date",
        value=date(2025, 12, 20),
        help="Used for flights & trains (outbound).",
    )

    # st.markdown("---")
    # st.markdown("#### API status")
    # st.caption(f"Groq key: {'✅ set' if GROQ_API_KEY else '❌ missing'}")
    # st.caption(f"RapidAPI (IRCTC): {'✅ set' if RAPIDAPI_KEY else '❌ missing'}")
    # st.caption(
    #     f"Amadeus: {'✅ set' if (AMADEUS_API_KEY and AMADEUS_API_SECRET) else '❌ missing'}"
    # )

    st.markdown("---")
    st.markdown("<div class='plan-button'>", unsafe_allow_html=True)
    plan_button = st.button("✨ Plan my trip", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    st.info("Tip: try 'Kashmir', 'Goa', 'Leh', 'Munnar', 'Jaipur', etc. – the engine will adapt.")

# Plan Trip

if plan_button:
    with st.spinner("Summoning routes, stays, and experiences..."):
        try:
            trip_obj = build_full_trip(
                destination=destination,
                days=trip_days,
                origin_city_for_transport=origin_city,
                date_for_transport=trip_date.isoformat(),
            )
            st.session_state.trip = trip_obj
            st.session_state.chat_history = []
            st.success("Trip planned successfully!")
        except Exception as e:
            st.error(f"Trip planning failed: {e}")

trip = st.session_state.trip

# Hero Section

# col_hero_left, col_hero_right = st.columns([1.4, 1])

# with col_hero_left:
# st.markdown('<div class="glass-card">', unsafe_allow_html=True)
# st.markdown(
#     """
#     <div class="pill-badge">
#         <span>🧠</span><span>AI Travel Operating System</span>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
st.subheader("AI Travelling system")
st.info("✈️ AuroraTrip AI: Smart Travel Planner Your AI travel companion that crafts thoughtful, realistic, day-wise trip plans based on your destination & length. \nNo generic lists! AuroraTrip AI builds complete, grounded itineraries: \n📅 Day-wise schedule \n🗺️ Realistic routes/distances \n🏨 Nearby stay/transport options (flights, trains)")

if trip:
    guide = trip["guide"]
    t = trip["transport"]
    h = trip["hotels"]

    total_attractions = len(guide.get("attractions", []))
    total_days = len(guide.get("itinerary", []))
    flight_count = len(t.get("flights", []))
    train_count = len(t.get("trains", []))
    hotel_count = sum(len(v) for v in h.values())
else:
    total_attractions = total_days = flight_count = train_count = hotel_count = 0



content_table = pd.DataFrame({
    "Attractions Curated": [total_attractions],
    "Trip Length": [f"{total_days} days"],  
    "Routes & Stays": [f"{train_count} trains · {flight_count} flights · {hotel_count} stays"],
})
st.table(content_table)

# m1, m2, m3 = st.columns(3)
# with m1:
#     st.markdown('<div class="metric-label">ATTRACTIONS CURATED</div>', unsafe_allow_html=True)
#     st.markdown(f'<div class="metric-value">{total_attractions}</div>', unsafe_allow_html=True)
# with m2:
#     st.markdown('<div class="metric-label">TRIP LENGTH</div>', unsafe_allow_html=True)
#     st.markdown(f'<div class="metric-value">{total_days} days</div>', unsafe_allow_html=True)
# with m3:
#     st.markdown('<div class="metric-label">ROUTES & STAYS</div>', unsafe_allow_html=True)
#     st.markdown(
#         f'<div class="metric-value">{train_count} trains · {flight_count} flights · {hotel_count} stays</div>',
#         unsafe_allow_html=True,
#     )

st.markdown("</div>", unsafe_allow_html=True)

# with col_hero_right:
st.markdown('<div class="glass-card-lite">', unsafe_allow_html=True)
# st.markdown(
#     '<div class="section-title">🌍 Interactive Trip Map</div>'
#     '<div class="section-subtitle">Visualize your destination and key hotspots at a glance.</div>',
#     unsafe_allow_html=True,
# )
st.divider()

col1, col2 = st.columns([1, 3])  # 1/4 left, 3/4 right

with col1:
    st.subheader("🗺 Trip Summary")

    if trip:
        guide = trip["guide"]

        # Destination title
        st.markdown(f"### 📍 {guide['destination']}")

        # Coordinates
        st.markdown(
            f"""
            **Latitude:** {guide['coordinates']['lat']}  
            **Longitude:** {guide['coordinates']['lon']}
            """
        )

        # Top attractions
        attractions = guide.get("attractions", [])
        if attractions:
            st.markdown("### ⭐ Attractions")
            for a in attractions:
                st.markdown(
                    f"""
                    - **{a['name']}**  
                      *{a['category'].title()}*
                    """
                )
        else:
            st.info("No attractions found for this destination.")

        # Additional trip metadata
        if "trip_days" in guide:
            st.markdown(f"**Trip Duration:** {guide['trip_days']} days")

        if "description" in guide:
            st.markdown(f"**About the place:** {guide['description']}")

        st.markdown("---")
        st.button("🔄 Recalculate Route")

    else:
        st.info("Plan your trip to see the details here.")

with col2:  # Right 3/4 section for full map
    if trip:
        guide = trip["guide"]
        attractions = guide.get("attractions", [])
        center_lat = guide["coordinates"]["lat"]
        center_lon = guide["coordinates"]["lon"]

        # Collect all location coordinates
        route_points = []

        if attractions:
            route_points = [(a["lat"], a["lon"]) for a in attractions]
        else:
            route_points = [(center_lat, center_lon)]

        # Create Map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles="CartoDB dark_matter",
        )

        # Add markers
        if attractions:
            for a in attractions:
                folium.CircleMarker(
                    location=[a["lat"], a["lon"]],
                    radius=7,
                    popup=f"{a['name']} ({a['category']})",
                    tooltip=a["name"],
                    color="#38bdf8",
                    fill=True,
                    fill_color="#38bdf8",
                    fill_opacity=0.9,
                ).add_to(m)

            # Draw route line between attractions
            if len(route_points) > 1:
                folium.PolyLine(
                    route_points,
                    color="#00e5ff",
                    weight=4,
                    opacity=0.8,
                ).add_to(m)

            # Auto zoom to show all points
            m.fit_bounds(route_points)

        else:
            # Only destination
            folium.Marker(
                [center_lat, center_lon],
                popup=guide["destination"],
                tooltip=guide["destination"],
            ).add_to(m)

        map_html = m.get_root().render()
        components_html(map_html, height=420)

    else:
        st.info("Set origin & destination in the sidebar and click **Plan my trip**.")


st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

if not trip:
    st.info("Configure your trip on the left and click **Plan my trip** to get started.")
    st.stop()

guide = trip["guide"]
transport = trip["transport"]
hotels = trip["hotels"]
bot: TravelChatbot = trip["bot"]

# Date & holiday analysis for the current UI selections
date_info = analyze_trip_dates(trip_date.isoformat(), trip_days)

# Small holiday/weekend banner
if date_info.get("holiday_hits") or date_info.get("is_weekend_heavy"):
    txt = []
    if date_info.get("holiday_hits"):
        names = ", ".join(h["name"] for h in date_info["holiday_hits"])
        txt.append(f"Your selected dates include Indian holidays: **{names}**.")
    if date_info.get("is_weekend_heavy"):
        txt.append("A large part of your trip falls on **weekends**, expect higher demand.")
    st.warning(" ".join(txt))

# Tabs

tab_overview, tab_itinerary, tab_transport, tab_stays, tab_budget, tab_chat, tab_raw = st.tabs(
    ["🌐 Overview", "🗺 Itinerary", "🚄 Transport", "🏨 Stays", "💰 Budget", "🤖 AI Companion", "💾 Raw JSON"]
)

# Overview Tab

with tab_overview:
    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.markdown("#### Destination snapshot")
        st.markdown(
            f"**{guide['destination']}**  \n"
            f"Lat: `{guide['coordinates']['lat']:.4f}`, Lon: `{guide['coordinates']['lon']:.4f}`"
        )

        st.markdown("##### Community vibes")
        st.write(guide.get("community_summary", ""))

        st.markdown("##### Attraction categories")
        df_attr = pd.DataFrame(guide.get("attractions", []))
        if not df_attr.empty:
            cat_counts = df_attr["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            st.bar_chart(cat_counts.set_index("category"))
        else:
            st.info("No attractions loaded yet.")

    with c2:
        st.markdown("#### Quick trip facts")
        total_days = len(guide.get("itinerary", []))
        total_spend = sum(a.get("approx_cost_inr", 0) for a in guide.get("attractions", []))

        st.write("Estimated local experience spend (tickets + snacks):")
        st.markdown(f"**₹{int(total_spend):,}** (excluding hotels & transport)")

        st.write("Top 5 headline attractions:")
        if not df_attr.empty:
            for name_ in df_attr["name"].head(5):
                st.markdown(f"- {name_}")

        st.write("Transport snapshot:")
        st.markdown(
            f"- **Trains**: {len(transport.get('trains', []))} options  \n"
            f"- **Flights**: {len(transport.get('flights', []))} options  \n"
        )

        # Small visual gallery for first 3 attractions with images
        st.write("Photo preview:")
        if not df_attr.empty:
            img_cols = st.columns(3)
            idx = 0
            for _, row in df_attr.head(6).iterrows():
                if not row.get("image_url"):
                    continue
                with img_cols[idx % 3]:
                    st.image(row["image_url"], use_container_width=True, caption=row["name"])
                idx += 1
                if idx >= 3:
                    break

# Itinerary Tab

with tab_itinerary:
    st.markdown("### Day-wise plan")
    for day in guide.get("itinerary", []):
        with st.expander(f"Day {day['day']} – {day['theme']}", expanded=(day["day"] == 1)):
            cols = st.columns([2, 1])

            with cols[0]:
                st.markdown("**Attractions:**")
                for a in day["attractions"]:
                    if a.get("image_url"):
                        st.image(a["image_url"], use_column_width=True, caption=a["name"])

                    chips = f"<span class='chip'>{a['category']}</span>"
                    st.markdown(
                        f"{chips}<br><strong>{a['name']}</strong> · ~{a['suggested_hours']} hours · "
                        f"approx ₹{int(a['approx_cost_inr'])}",
                        unsafe_allow_html=True,
                    )
                    desc = a["description"]
                    st.caption(desc[:280] + ("..." if len(desc) > 280 else ""))

            with cols[1]:
                st.markdown("**Movement between spots:**")
                legs = day["route_legs"]
                if legs:
                    for leg in legs:
                        st.markdown(
                            f"- {leg['from_place']} → {leg['to_place']}  "
                            f"({leg['distance_km']} km · {leg['duration_min']} min)"
                        )
                else:
                    st.caption("Single-base day. Most time in one area.")

# Transport Tab

with tab_transport:
    st.markdown("### Transport options")

    t1, t2 = st.columns(2)
    trains = transport.get("trains", [])
    flights = transport.get("flights", [])

    with t1:
        st.markdown("#### 🚆 Trains")
        if trains:
            df_trains = pd.DataFrame(trains)
            show_cols = [
                "train_name",
                "train_no",
                "origin_station",
                "destination_station",
                "departure_time",
                "arrival_time",
                "duration_hours",
                "available_classes",
                "source",
            ]
            show_cols = [c for c in show_cols if c in df_trains.columns]
            st.dataframe(df_trains[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info(
                "No live trains found (or IRCTC API did not return results). "
                "Sample routes are available only for some pairs like Mumbai ↔ Delhi."
            )

    with t2:
        st.markdown("#### ✈ Flights")
        if flights:
            df_flights = pd.DataFrame(flights)
            show_cols = [
                "airline",
                "flight_no",
                "origin",
                "destination",
                "departure_time",
                "arrival_time",
                "duration_iso",
                "price",
                "currency",
                "nonstop",
                "source",
            ]
            show_cols = [c for c in show_cols if c in df_flights.columns]
            st.dataframe(df_flights[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info(
                "No live flights returned. Amadeus test API can be rate-limited, "
                "so sample routes are used as fallback only for some city pairs."
            )

# Stays Tab

with tab_stays:
    st.markdown("### Nearby stays")

    if not hotels or all(len(v) == 0 for v in hotels.values()):
        st.info(
            "No hotels were found via OSM Overpass. "
            "You can increase the radius in the backend if needed."
        )
    else:
        for cat in ["budget", "mid", "luxury", "unknown"]:
            group = hotels.get(cat, [])
            if not group:
                continue
            st.markdown(f"#### {cat.title()} options")
            cols = st.columns(3)
            for i, h in enumerate(group[:6]):
                col = cols[i % 3]
                with col:
                    st.markdown(
                        f"""
                        <div class="glass-card-lite">
                        <strong>{h['name']}</strong><br/>
                        <span style="font-size:0.8rem;color:#9ca3af;">
                        Lat: {h['lat']:.4f}, Lon: {h['lon']:.4f}<br/>
                        Source: {h['source']}
                        </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Budget Tab

with tab_budget:
    st.markdown("### 💰 Hybrid budget planner")

    budget_level = st.radio(
        "Choose your comfort level",
        ["Budget", "Mid", "Luxury"],
        index=1,
        horizontal=True,
    )

    budget_info = estimate_budget(
        guide=guide,
        transport=transport,
        hotels_grouped=hotels,
        trip_days=trip_days,
        budget_level=budget_level,
        date_info=date_info,
    )

    b = budget_info["breakdown"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Recommended total budget",
            f"₹{int(budget_info['recommended']):,}",
            help="Includes stays, food, local travel, attractions, and basic transport estimate.",
        )
    with c2:
        st.metric("Low estimate", f"₹{int(budget_info['low_estimate']):,}")
    with c3:
        st.metric("High estimate", f"₹{int(budget_info['high_estimate']):,}")

    st.markdown("#### Cost breakdown")
    breakdown_df = pd.DataFrame(
        {
            "component": [
                "Hotels",
                "Food",
                "Local travel",
                "Attractions",
                "Transport",
                "Buffer",
            ],
            "amount": [
                b["hotel_cost"],
                b["food_cost"],
                b["local_cost"],
                b["attraction_cost"],
                b["transport_cost"],
                b["buffer_amount"],
            ],
        }
    )
    st.bar_chart(breakdown_df.set_index("component"))

    st.markdown("#### Date & crowd insights")
    alt = date_info.get("suggested_alternatives", {})
    holiday_hits = date_info.get("holiday_hits", [])
    weekend_dates = date_info.get("weekend_dates", [])

    info_lines = []
    info_lines.append(f"- Trip start: **{date_info.get('start_date', trip_date.isoformat())}**")
    info_lines.append(f"- Trip length: **{trip_days} days**")

    if holiday_hits:
        names = ", ".join(h["name"] for h in holiday_hits)
        info_lines.append(f"- Holidays in your range: **{names}**")
    else:
        info_lines.append("- No fixed Indian national holidays detected in your date range.")

    if weekend_dates:
        info_lines.append(f"- Weekend days in trip: {len(weekend_dates)} day(s).")
    if alt:
        info_lines.append(
            f"- Suggested calmer start: **{alt.get('calmer_start', 'N/A')}**; "
            f"suggested cheaper start: **{alt.get('cheaper_start', 'N/A')}**."
        )

    if date_info.get("festival_hint"):
        info_lines.append(f"- Festival season note: {date_info['festival_hint']}.")

    st.markdown("\n".join(info_lines))

    st.markdown("#### AI perspective on your budget & dates")
    with st.spinner("Asking AuroraTrip AI to interpret your numbers..."):
        advice_text = ai_budget_and_date_advice(
            destination=guide["destination"],
            budget_info=budget_info,
            date_info=date_info,
        )
    st.write(advice_text)

# Chat Tab

with tab_chat:
    st.markdown("### 🤖 AuroraTrip AI – Your travel companion")

    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-bubble-user'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-bubble-bot'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask AuroraTrip AI anything (trip questions, budget, dates, or general topics!)",
            key="chat_input",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking like a travel concierge..."):
            answer = bot.ask(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Raw JSON Tab

with tab_raw:
    st.markdown("### Debug / Raw view")
    raw = {
        "guide": guide,
        "transport": transport,
        "hotels": hotels,
        "date_info": date_info,
        "note": "Budget info is recomputed in the Budget tab based on UI selection.",
    }
    st.json(raw)
