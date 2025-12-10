import streamlit as st 
import random
import smtplib 
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import db.aurora_trip_ai as atai

# ====== Config ======
st.set_page_config(page_title="Auth | A.T.A.I.", layout="centered")

MAIL = st.secrets["auth"]["mail"]
PASS = st.secrets["auth"]["pass"]

# ====== Functions ======

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp(mail, otp):
    msg = MIMEMultipart()
    msg['From'] = MAIL
    msg['To'] = mail
    msg['Subject'] = 'Verification Code'

    body = f"""
    <div style="text-align:center;background-color:black;color:white;
    max-width:400px;font-size:28px;">
        <p>Your OTP Code:</p>
        <p><strong>{otp}</strong></p>
    </div>
    """
    msg.attach(MIMEText(body, 'html'))

    try: 
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(MAIL, PASS)
            server.send_message(msg)
            return True
    except Exception:
        st.error("❌ Failed to send email.")
        return False

def logout():
    st.session_state.update({
        "otp_sent": False,
        "authenticated": False,
        "email": "",
        "otp": None,
        "otp_attempts": 0,
        "otp_timestamp": 0,
        "mode": "login"
    })

# ====== Session Defaults ======
defaults = {
    "otp_sent": False, "otp": None, "name": None, "authenticated": False,
    "email": "", "otp_attempts": 0, "otp_timestamp": 0, "mode": "login"
}
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# ====== UI ======
st.title(":grey[AURORA TRIP AI - AUTHENTICATION]")

# Mode Switch (Login / Signup)
mode = st.radio("Choose Action", ["Login", "Signup"])
st.session_state.mode = mode.lower()

with st.expander(f"**:grey[{mode}]**"):

    # ---------------------- SIGNUP -----------------------
    if mode == "Signup" and not st.session_state.authenticated:
        st.header(":green[Create Account]")

        col1, col2 = st.columns(2)

        with col1:
            name_input = st.text_input("**Name**")
            email_input = st.text_input("**Email**")

            if st.button("Send OTP"):
                if time.time() - st.session_state.otp_timestamp < 60:
                    st.warning("⏳ Wait before resending OTP.")
                else:
                    otp = generate_otp()
                    if send_otp(email_input, otp):
                        st.session_state.otp_sent = True
                        st.session_state.otp = otp
                        st.session_state.email = email_input
                        st.session_state.otp_timestamp = time.time()
                        st.session_state.otp_attempts = 0
                        st.toast("✅ OTP sent!")

        with col2:
            if st.session_state.otp_sent:
                entered_otp = st.text_input("**OTP**")
                password_input = st.text_input("**Create Password**", type="password")

                if st.button("Verify & Signup"):
                    if entered_otp == st.session_state.otp:
                        atai.store_user(
                            name=name_input,
                            email=email_input,
                            password=password_input
                        )
                        st.session_state.authenticated = True
                        st.toast("🎉 Signup Successful!")
                        st.switch_page("pages/Planner.py")
                    else:
                        st.error("❌ Invalid OTP")

    # ---------------------- LOGIN -----------------------
    if mode == "Login" and not st.session_state.authenticated:
        st.header(":blue[Log In]")

        email_input = st.text_input("Email")
        password_input = st.text_input("Password", type="password")

        if st.button("Login"):
            user = atai.get_user_by_email(email_input)
            if not user:
                st.error("❌ User not found. Please Signup.")
            else:
                if atai.validate_user(email_input, password_input):
                    st.session_state.authenticated = True
                    st.session_state.email = email_input
                    st.toast("🎉 Login Successful!")
                    st.switch_page("pages/Planner.py")
                else:
                    st.error("❌ Incorrect Password")

    # --------------------- Already logged in -----------------------
    if st.session_state.authenticated:
        st.success(f"Logged in as: {st.session_state.email}")
        st.button("Logout", on_click=logout)
