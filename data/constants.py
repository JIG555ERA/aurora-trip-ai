from typing import Dict, List, Any

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
