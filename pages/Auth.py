import streamlit as st 
import random
import smtplib 
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ====== Config ======
st.set_page_config(page_title="Auth | A.T.A.I.", layout="centered")

# Sender Email Credentials
MAIL = st.secrets["auth"]["mail"]
PASS = st.secrets["auth"]["pass"]

# Allowed Users
# allowed_users = st.secrets["users"]["allowed"]

# ====== Functions ======

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp(mail, otp):
    msg = MIMEMultipart()
    msg['From'] = MAIL
    msg['To'] = mail
    msg['Subject'] = ' Verification Code'

    body = f"""
    <div style="
        text-align: center;
        background-color: black;
        color: white;
        max-width: 400px;
        font-size: 28px;
        ">
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
        "otp_timestamp": 0
    })

# ====== Session Defaults ======
st.session_state.setdefault("otp_sent", False)
st.session_state.setdefault("otp", None)
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("email", "")
st.session_state.setdefault("otp_attempts", 0)
st.session_state.setdefault("otp_timestamp", 0)

# ====== UI ======
st.title(":grey[AURORA TRIP AI - AUTHENTICATION]")

with st.expander('**:grey[Log IN]**'):
    if not st.session_state.authenticated:
        st.header(":blue[Log IN]")
        st.info('**ⓘ You must be an authorized member to access the site.**')
        col1, col2 = st.columns(2)
        with col1:
            email_input = st.text_input("**:grey[MAIL]**", value=st.session_state.email)

            if st.button("Send OTP"):
                # if email_input not in allowed_users:
                #     st.error("🚫 Unauthorized Email")
                if time.time() - st.session_state.otp_timestamp < 60:
                    st.warning("⏳ Please wait before resending OTP.")
                else:
                    otp = generate_otp()
                    if send_otp(email_input, otp):
                        st.session_state.otp_sent = True
                        st.session_state.otp = otp
                        st.session_state.email = email_input
                        st.session_state.otp_timestamp = time.time()
                        st.session_state.otp_attempts = 0
                        st.toast("✅ OTP sent to your email!")

        with col2:
            if st.session_state.otp_sent:
                entered_otp = st.text_input("**:grey[OTP]**")

                if st.button("Verify OTP"):
                    if st.session_state.otp_attempts >= 3:
                        st.error("❌ Too many attempts.")
                        # if st.button("Restart"):
                        #     logout()
                    elif entered_otp == st.session_state.otp:
                        st.session_state.authenticated = True
                        st.toast("🎉 Login successful!")
                        st.switch_page("pages/Planner.py")
                    else:
                        st.session_state.otp_attempts += 1
                        attempts_left = 3 - st.session_state.otp_attempts
                        st.error(f"❌ Invalid OTP. Attempts left: {attempts_left}")

            # Resend OTP
            if st.session_state.otp_sent and time.time() - st.session_state.otp_timestamp >= 60:
                if st.button("🔁 Resend OTP"):
                    otp = generate_otp()
                    if send_otp(st.session_state.email, otp):
                        st.session_state.otp = otp
                        st.session_state.otp_timestamp = time.time()
                        st.session_state.otp_attempts = 0
                        st.toast("✅ OTP resent!")

    else:
        col1, col2, col3 = st.columns([3,1,1])

        with col1:
            st.success(f"✅ Logged in as: {st.session_state.email}")
            st.button("🔓 Logout", on_click=logout)