from pymongo import MongoClient, errors
from bson.objectid import ObjectId
import bcrypt
import streamlit as st

MONGO_URI = st.secrets["API_KEYS"]["MONGO_URI"]

DB_NAME = "aurora_trip_ai_db"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Create index for faster lookup
db["users"].create_index("email", unique=True)


def hash_password(password: str) -> bytes:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def get_user_by_email(email: str):
    """Retrieve user by email"""
    users = db["users"]
    return users.find_one({"email": email})

def verify_password(password: str, hashed: bytes) -> bool:
    """Verify bcrypt password"""
    return bcrypt.checkpw(password.encode("utf-8"), hashed)

def store_user(name: str, email: str, password: str):
    """
    Stores a new user in the database.
    Returns: True if success, False if email exists.
    """
    users = db["users"]

    # check if email already exists
    if users.find_one({"email": email}):
        return False  # email already taken

    hashed_password = hash_password(password)

    user_data = {
        "name": name,
        "email": email,
        "password": hashed_password
    }

    users.insert_one(user_data)
    return True

def validate_user(email: str, password: str) -> bool:
    """Validate user credentials"""
    user = get_user_by_email(email)
    if not user:
        return False
    return verify_password(password, user["password"])

def store_travel_guide(guide_data: dict): 
    guides = db["travel_guides"]
    guides.insert_one(guide_data)