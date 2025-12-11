🌄 Aurora Trip AI

An intelligent trip-planning Streamlit application with authentication, weather insights, and personalized travel recommendations.

📁 Project Structure
aurora-trip-ai/
│
├── .streamlit/
│   └── secrets.toml           # API keys & environment secrets
│
├── data/
│   ├── constants.py           # App-wide constants (API URLs, etc.)
│
├── db/
│   └── aurora_trip_ai.py      # MongoDB initialization & helper functions
│
├── pages/
│   ├── Auth.py                # Login, signup, authentication logic
│   └── Planner.py             # Trip planning UI + weather + guide data
│
├── utilities/
│   ├── helpers.py             # Reusable helper functions
│   └── http_helpers.py        # HTTP-based API communication utilities
│
├── Home.py                    # Main home page
├── requirements.txt           # Dependencies
├── .env                       # Optional environment variables (ignored)
└── README.md                  # You are here!

⚡ App Pages Overview
🏠 1. Home Page (Home.py)

The landing screen of the application.

Features

Project introduction

Quick navigation to:

Login / Signup

Planner

About section

No authentication required

Light & fast UI built using Streamlit

🔐 2. Auth Page (pages/Auth.py)

Handles user authentication using MongoDB.

Features

Signup

Stores user info in MongoDB Atlas

Login

Validates credentials

Sets st.session_state.authenticated = True

Redirection

Automatically navigates authenticated users to Planner

Backend

Uses db/aurora_trip_ai.py to:

Connect to MongoDB

Read / insert user documents

🧭 3. Planner Page (pages/Planner.py)

Core of the application: plan your trip with weather & guide recommendations.

Workflow

User enters a destination

Weather fetched from OpenWeather API

Data displayed using Streamlit metrics

Attractions, guides, and information pulled from your stored schema

If not logged in:

if st.session_state.authenticated == False:
    st.info("Please login to plan your trip and access full features.")
    st.button("Login", on_click=st.switch_page, args=("pages/Auth.py",))

Features

Weather Conditions (Temp, Humidity, Wind, Status)

Dynamic metrics

Backend calling via utilities/http_helpers.py

Guide information rendered in a clean layout

Session-aware UI

🔧 Utilities
utilities/helpers.py

General helper functions for:

Formatting

Data conversions

Common UI utilities

utilities/http_helpers.py

Handles all HTTP requests such as:

Fetching OpenWeather API data

External API handlers

Wrapper to avoid repeating requests.get logic

🚀 Installation & Running
1. Clone the repo
git clone https://github.com/your-username/aurora-trip-ai.git
cd aurora-trip-ai

2. Install dependencies
pip install -r requirements.txt

3. Add API keys
.streamlit/secrets.toml
OPENWEATHER_API_KEY = "your_key"
MONGO_URI = "your_mongo_connection_string"
DB_NAME = "aurora_trip_ai_db"

4. Run app
streamlit run Home.py

🧪 Tech Stack

Streamlit – UI & routing

MongoDB Atlas – Auth database

OpenWeather API – Weather insights

Python – Logic + backend

Requests – API calls

🛠️ Future Enhancements

Multi-day trip predictions

AI itinerary generation

Cost estimation system

Real-time alerts