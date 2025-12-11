

import streamlit as st
import folium
import streamlit.components.v1 as components
import pandas as pd
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="AuroraTrip AI — Home", layout="wide", initial_sidebar_state="expanded")

# Helpers

def render_folium_map(center, attractions, tiles="CartoDB positron", zoom_start=6, height=420):
    """Return HTML for a folium map showing attractions and route."""
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)
    route_points = [(a["lat"], a["lon"]) for a in attractions]

    # markers
    for i, a in enumerate(attractions):
        folium.CircleMarker(
            location=[a["lat"], a["lon"]],
            radius=7,
            popup=f"{a['name']} ({a['category'].title()})",
            tooltip=f"{i+1}. {a['name']}",
            color="#1f77b4",
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.9,
        ).add_to(m)

    # route
    if len(route_points) > 1:
        folium.PolyLine(route_points, color="#ff7f0e", weight=5, opacity=0.85).add_to(m)
        m.fit_bounds(route_points)
    else:
        m.location = center

    return m.get_root().render()


def simple_itinerary(destination_name, start_date, length_days, attractions):
    """Create a simple day-wise itinerary distributing attractions over days."""
    itinerary = {}
    idx = 0
    n = max(1, len(attractions))
    for d in range(length_days):
        day = (start_date + timedelta(days=d)).strftime("%A, %d %b %Y")
        # pick up to 4 attractions per day in round-robin
        today = []
        for _ in range(3):
            if attractions:
                today.append(attractions[idx % n])
                idx += 1
        itinerary[day] = today
    return itinerary


# Sample data (replace with real API results)

SAMPLE_GUIDE = {
    "destination": "Goa, India",
    "coordinates": {"lat": 15.2993, "lon": 74.1240},
    "attractions": [
        {"name": "Baga Beach", "lat": 15.5516, "lon": 73.7540, "category": "beach", "rating": 4.5},
        {"name": "Calangute", "lat": 15.5442, "lon": 73.7546, "category": "beach", "rating": 4.2},
        {"name": "Fort Aguada", "lat": 15.4939, "lon": 73.7764, "category": "historic", "rating": 4.4},
        {"name": "Anjuna Market", "lat": 15.6032, "lon": 73.7446, "category": "market", "rating": 4.0},
        {"name": "Dudhsagar Falls", "lat": 15.3142, "lon": 74.3135, "category": "waterfall", "rating": 4.7},
    ],
}

EVENT_GALLERY = [
    # Unsplash image URLs (publicly accessible). Replace or add your own.
    {
        "title": "Sunset at Beach",
        "img": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1200&q=80",
    },
    {
        "title": "Local Market",
        "img": "https://images.unsplash.com/photo-1529025351545-9f6d5a8f6a47?w=1200&q=80",
    },
    {
        "title": "Heritage Fort",
        "img": "https://images.unsplash.com/photo-1505765051205-1c38a3b7f2d1?w=1200&q=80",
    },
    {
        "title": "Mountain Trail",
        "img": "https://images.unsplash.com/photo-1501785888041-af3ef285b470?w=1200&q=80",
    },
]

# Page: Sidebar

st.sidebar.image("assets/aurora_ai_logo.png", caption="AuroraTrip AI")
st.sidebar.info("This page holds sample/demo data. Visit **Planner Page** for full trip planning features.")

destination = st.sidebar.text_input("Destination", value=SAMPLE_GUIDE["destination"], disabled=True)
length_days = st.sidebar.slider("Trip length (days)", min_value=1, max_value=14, value=4)
start_date = st.sidebar.date_input("Start date", value=datetime.now().date())
map_style = st.sidebar.selectbox("Map style", options=["CartoDB positron", "OpenStreetMap", "CartoDB dark_matter"], index=0)
show_route = st.sidebar.checkbox("Show route demo", value=True)

# Secrets (optional)
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else None
RAPIDAPI_KEY = st.secrets.get("RAPIDAPI_KEY") if "RAPIDAPI_KEY" in st.secrets else None

# Header / Hero

col1, col2 = st.columns([2, 1])
with col1:
    st.title("AuroraTrip AI — Your Travel Companion")
    st.markdown(
        """
        **Thoughtful. Realistic. Local.**  
        Tell Aurora where you want to go and for how long — it will plan a complete trip the way a human would: day-wise itineraries, travel routes, stays, transport choices and a built-in AI concierge.  

        No copy-paste. No generic lists. Everything is grounded in real locations, distances and community insight.
        """
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Time to plan", "~30s", "AI-powered")
    c2.metric("Stops considered", "Real places", "+ local insights")
    c3.metric("One-click export", "PDF / Share", "Itinerary + Map")

with col2:
    st.image(
        "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&q=80",
        caption="AuroraTrip — dusk on the shore",
        use_column_width=True,
    )

st.markdown("---")

# Features section

st.header("What AuroraTrip does for you")
st.markdown(
    "- **End-to-end planning** — from where to go each day to how to travel between stops.\n- **Context-aware** — distance, local transport and time-of-day matter.\n- **Multi-modal** — trains, flights, driving options and local taxis.\n- **Concierge** — ask follow-up questions and adjust plans.")

st.subheader("Live demos on this page")
st.markdown("Below are interactive demos. They are simplified but built with real components so you can later plug-in real APIs (Maps, Booking, Pricing).")

# Map Demo & Route

st.subheader("Interactive Map & Route Demo")
col_map, col_info = st.columns([3, 1])
with col_info:
    st.markdown("**Demo controls**")
    st.write(f"Destination: **{destination}**")
    st.write(f"Trip length: **{length_days}** days")
    st.write("Map style: **%s**" % map_style)
    if GROQ_API_KEY or RAPIDAPI_KEY:
        st.success("API keys available in secrets — you can wire real lookups")
    else:
        st.info("No API keys in secrets. This demo uses sample data. Add keys to `.streamlit/secrets.toml` to enable real data.")

with col_map:
    guide = SAMPLE_GUIDE.copy()
    guide["destination"] = destination
    attractions = guide.get("attractions", [])

    if show_route and attractions:
        map_html = render_folium_map((guide["coordinates"]["lat"], guide["coordinates"]["lon"]), attractions, tiles=map_style)
    else:
        map_html = render_folium_map((guide["coordinates"]["lat"], guide["coordinates"]["lon"]), [
            {"name": destination, "lat": guide["coordinates"]["lat"], "lon": guide["coordinates"]["lon"], "category": "destination"}
        ], tiles=map_style)

    components.html(map_html, height=520)

# Itinerary Generator

st.subheader("Day-wise Itinerary (auto-generated)")
it = simple_itinerary(destination, start_date, length_days, attractions)

for day, items in it.items():
    with st.expander(day, expanded=False):
        if items:
            for i, a in enumerate(items):
                st.markdown(f"**{i+1}. {a['name']}** — *{a['category'].title()}*  ")
                st.write(f"Approx. rating: {a.get('rating', '—')}")
                st.write(f"Suggested time: {random.choice(['Morning', 'Afternoon', 'Evening'])}")
        else:
            st.info("No attractions available for this day — the real planner would fetch local suggestions.")

st.markdown("---")

# Nearby stays & transport mockups

st.header("Where to stay & How to move")
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Top nearby stays (mock)")
    stays = [
        {"name": "Seaside Resort", "price": "₹3,499/night", "rating": 4.3},
        {"name": "Boutique Homestay", "price": "₹2,199/night", "rating": 4.6},
        {"name": "Budget Inn", "price": "₹1,099/night", "rating": 4.0},
    ]
    for s in stays:
        st.markdown(f"**{s['name']}** — {s['price']} — Rating: {s['rating']}")
        st.progress(int((s['rating'] / 5) * 100))

with col_b:
    st.subheader("Transport choices (mock)")
    st.markdown("**Flights** — curated quick options with approximate durations and stops.")
    st.table(pd.DataFrame([
        {"route": "DEL → GOA", "duration": "2h 30m", "price": "₹4,500"},
        {"route": "BLR → GOA", "duration": "1h 15m", "price": "₹3,999"},
    ]))
    st.markdown("**Trains** — sleeper and express suggestions (mock).")
    st.table(pd.DataFrame([
        {"route": "CSTM → MAO (Express)", "duration": "8h 40m", "price": "₹600"},
        {"route": "BCT → MAO (Sleeper)", "duration": "10h 10m", "price": "₹400"},
    ]))

st.markdown("---")

# Event gallery

st.header("Local Events & Gallery — Visual Preview")
st.markdown("A curated gallery of local scenes and events — images are example Unsplash URLs.")

cols = st.columns(4)
for i, ev in enumerate(EVENT_GALLERY):
    with cols[i % 4]:
        st.image(ev['img'], caption=ev['title'], use_column_width=True)


# Explain how the page works

st.header("How this homepage is built — Technical notes & integration guide")

st.markdown(
    """
    **Architecture (simple):**  
    - Frontend: Streamlit single-file UI + Folium for map rendering.  
    - Data & AI: Replace the sample data with calls to your backend or external APIs (Sanity/GROQ for POIs, RapidAPI for transport, Amadeus for flight pricing, Railway APIs for trains).  
    - Secrets: Put API keys in `.streamlit/secrets.toml` or Streamlit Cloud secrets. Use `st.secrets` to access them in code.  

    **Where to plug real services next:**  
    1. POI & local guides — call a CMS or Sanity (GROQ) and parse real attractions into `attractions` list.  
    2. Routing — use a route service (OSRM / GraphHopper / Google Directions) to compute real travel times and legs.  
    3. Stays & pricing — call Booking / Airbnb / partner APIs for live inventory.  
    4. Bookable items — expose booking links or a reservation flow.  
    5. AI Concierge — call an LLM (OpenAI / Anthropic / local LLM) with context (itinerary, local info, policies).  

    **Map rendering:** This demo uses Folium to render the map server-side and embeds HTML. For heavy interactivity consider Mapbox GL JS or a React map component served via a small frontend micro-app.

    **Export / Sharing:** You can export the itinerary as PDF using `pdfkit` or by rendering HTML and using a server-side headless browser. We also recommend adding a "shareable link" flow that stores the plan JSON in a backend and returns a short URL.
    """
)

st.caption("AuroraTrip AI — built as a demo. Replace mock blocks with live endpoints to make a production-grade trip planner.")

