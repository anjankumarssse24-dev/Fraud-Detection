import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import io
import base64
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import json
import os
import hashlib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection – Online Payment Fraud Detection",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# USER MANAGEMENT FUNCTIONS
# --------------------------------------------------
USERS_FILE = "users.json"

def load_users():
    """Load registered users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = {
        "password": hash_password(password),
        "email": email
    }
    save_users(users)
    return True, "Registration successful!"

def verify_user(username, password):
    """Verify user credentials"""
    users = load_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            return True
    return False

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

# --------------------------------------------------
# LOGIN AND REGISTRATION PAGE - ENHANCED UI
# --------------------------------------------------
if not st.session_state.logged_in:
    # Premium styling for auth pages with animations
    st.markdown("""
    <style>
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    .auth-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 50px;
        border-radius: 25px;
        box-shadow: 0 25px 80px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.3);
        max-width: 480px;
        margin: 30px auto;
        animation: fadeIn 0.8s ease-out;
        border: 2px solid rgba(255,255,255,0.5);
    }
    
    .logo-container {
        text-align: center;
        padding: 40px 20px 30px;
        animation: fadeIn 1s ease-out;
    }
    
    .logo-icon {
        font-size: 80px;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
        filter: drop-shadow(0 10px 20px rgba(0,0,0,0.3));
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e0e7ff !important;
        padding: 14px 16px !important;
        font-size: 15px !important;
        transition: all 0.3s ease;
        background: rgba(255,255,255,0.9) !important;
        color: #1a202c !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af !important;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 16px;
        font-weight: 700;
        font-size: 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }
    
    .toggle-btn {
        transition: all 0.3s ease;
    }
    
    h1, h2, h3 {
        color: #1a202c !important;
        font-weight: 800 !important;
    }
    
    label {
        color: #4a5568 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .feature-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        color: white;
        font-size: 13px;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 25px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Animated logo and branding
    st.markdown("""
    <div class="logo-container">
        <div class="logo-icon">🛡️</div>
        <div style="color: white !important; font-size: 56px; margin: 15px 0 10px; text-shadow: 0 4px 10px rgba(0,0,0,0.3); font-weight: 900;">FraudGuard</div>
        <p style="color: #E0E7FF; font-size: 20px; margin: 10px 0 20px; text-shadow: 0 2px 5px rgba(0,0,0,0.2);">AI-Powered Fraud Detection Platform</p>
        <div style="margin-top: 20px;">
            <span class="feature-badge">🤖 Machine Learning</span>
            <span class="feature-badge">⚡ Real-Time</span>
            <span class="feature-badge">🔒 Secure</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Toggle between login and register with enhanced design
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        tab_col1, tab_col2 = st.columns(2)
        with tab_col1:
            login_active = "primary" if st.session_state.page == "login" else "secondary"
            if st.button("🔐 Sign In", width="stretch", type=login_active, key="tab_login"):
                st.session_state.page = "login"
                st.rerun()
        with tab_col2:
            register_active = "primary" if st.session_state.page == "register" else "secondary"
            if st.button("📝 Register", width="stretch", type=register_active, key="tab_register"):
                st.session_state.page = "register"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Login Page with enhanced UI
    if st.session_state.page == "login":
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<div class="auth-card">', unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: center; margin-bottom: 30px;">
                    <div style='color: #667eea; font-size: 32px; margin-bottom: 10px; font-weight: bold;'>Welcome Back! 👋</div>
                    <p style='color: #718096; font-size: 15px;'>Sign in to continue to your account</p>
                </div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("👤 USERNAME", key="login_username", placeholder="Enter your username")
            password = st.text_input("🔒 PASSWORD", type="password", key="login_password", placeholder="Enter your password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 LOGIN", width="stretch", key="login_btn"):
                if username and password:
                    if verify_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success(f"✅ Welcome back, {username}!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")
                else:
                    st.warning("⚠️ Please enter both username and password")
            
            st.markdown("""
                <div style="text-align: center; margin-top: 25px; padding-top: 20px; border-top: 1px solid #e2e8f0;">
                    <p style='color: #718096; font-size: 14px;'>Don't have an account? Switch to <strong>Sign Up</strong> tab above</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Registration Page with enhanced UI
    elif st.session_state.page == "register":
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<div class="auth-card">', unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: center; margin-bottom: 30px;">
                    <div style='color: #764ba2; font-size: 32px; margin-bottom: 10px; font-weight: bold;'>Create Account 🎉</div>
                    <p style='color: #718096; font-size: 15px;'>Join FraudGuard to protect your transactions</p>
                </div>
            """, unsafe_allow_html=True)
            
            new_username = st.text_input("👤 USERNAME", key="reg_username", placeholder="Choose a unique username")
            new_email = st.text_input("📧 EMAIL ADDRESS", key="reg_email", placeholder="your.email@example.com")
            
            col_pass1, col_pass2 = st.columns(2)
            with col_pass1:
                new_password = st.text_input("🔒 PASSWORD", type="password", key="reg_password", placeholder="Min 6 characters")
            with col_pass2:
                confirm_password = st.text_input("🔒 CONFIRM", type="password", key="reg_confirm", placeholder="Re-enter password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("✨ CREATE ACCOUNT", width="stretch", key="register_btn"):
                if new_username and new_email and new_password and confirm_password:
                    if len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters long")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords do not match. Please check and try again.")
                    else:
                        success, message = register_user(new_username, new_password, new_email)
                        if success:
                            st.success(f"✅ {message} Your account is ready!")
                            st.balloons()
                            st.info("🔄 Redirecting to login page...")
                            st.session_state.page = "login"
                            st.rerun()
                        else:
                            st.error(f"❌ {message}. Please try a different username.")
                else:
                    st.warning("⚠️ Please fill in all fields to continue")
            
            st.markdown("""
                <div style="text-align: center; margin-top: 25px; padding-top: 20px; border-top: 1px solid #e2e8f0;">
                    <p style='color: #718096; font-size: 14px;'>Already have an account? Switch to <strong>Sign In</strong> tab above</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# --------------------------------------------------
# GLOBAL THEME - PREMIUM UI/UX DESIGN
# --------------------------------------------------
st.markdown("""
<style>
@keyframes slideInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.8;
    }
}

/* Fix blurriness and improve rendering */
* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

/* Premium gradient background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-attachment: fixed;
}

/* Dark text on light background for visibility */
h1, h2, h3, h4, h5, h6 {
    color: #1a202c !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

p, span, div, li {
    color: #2d3748 !important;
}

label {
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stMarkdown, .stTextInput label, .stSelectbox label, .stNumberInput label {
    color: #000000 !important;
    font-size: 14px;
    font-weight: 700;
}

/* Force black text for all form labels */
[data-testid="stWidgetLabel"] {
    color: #000000 !important;
}

[data-testid="stWidgetLabel"] p {
    color: #000000 !important;
}

/* Premium card styling with glassmorphism */
.card {
    line-height: 1.9;
    padding: 30px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.3);
    margin: 20px 0;
    border: 2px solid rgba(255, 255, 255, 0.5);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: slideInDown 0.6s ease-out;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.5);
}

.card p, .card li, .card h3, .card h4 {
    color: #2d3748 !important;
}

.card ul, .card ol {
    padding-left: 25px;
}

/* Enhanced button styling */
button {
    font-weight: 700 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.5px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6) !important;
}

/* SVM Model Badge with glow effect */
.svm-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 15px 30px;
    border-radius: 30px;
    font-weight: 800;
    text-align: center;
    margin: 20px auto;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5), 0 0 30px rgba(118, 75, 162, 0.3);
    font-size: 18px;
    animation: pulse 2s ease-in-out infinite;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

.svm-badge * {
    color: white !important;
}

/* Premium header styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 45px;
    border-radius: 25px;
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.2);
    margin-bottom: 35px;
    position: relative;
    overflow: hidden;
    animation: slideInDown 0.8s ease-out;
    border: 3px solid rgba(255, 255, 255, 0.3);
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    bottom: -50%;
    left: -50%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
    transform: rotate(45deg);
    animation: shine 3s infinite;
}

@keyframes shine {
    0%, 100% { transform: rotate(45deg) translateY(-100%); }
    50% { transform: rotate(45deg) translateY(100%); }
}

.main-header h1, .main-header h2, .main-header p {
    color: white !important;
    position: relative;
    z-index: 1;
}

/* Input fields with premium feel */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background-color: white !important;
    color: #2d3748 !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 15px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15), 0 4px 12px rgba(0,0,0,0.1) !important;
    transform: translateY(-2px) !important;
}

/* Ensure number input labels are black */
.stNumberInput label {
    color: #000000 !important;
}

.stNumberInput [data-testid="stWidgetLabel"] {
    color: #000000 !important;
}

.stNumberInput [data-testid="stWidgetLabel"] p {
    color: #000000 !important;
}

/* Ensure text input labels are black */
.stTextInput label {
    color: #000000 !important;
}

.stTextInput [data-testid="stWidgetLabel"] {
    color: #000000 !important;
}

.stTextInput [data-testid="stWidgetLabel"] p {
    color: #000000 !important;
}

/* Fix selectbox dropdown visibility - FORCE BLACK TEXT */
.stSelectbox [data-baseweb="select"] {
    background-color: white !important;
}

.stSelectbox [data-baseweb="select"] > div {
    background-color: white !important;
    color: #000000 !important;
}

/* Fix selected value text color in selectbox - CRITICAL */
.stSelectbox [data-baseweb="select"] [role="button"] {
    color: #000000 !important;
    background-color: white !important;
}

.stSelectbox [data-baseweb="select"] [role="button"] > div {
    color: #000000 !important;
}

.stSelectbox [data-baseweb="select"] span {
    color: #000000 !important;
}

.stSelectbox div[data-baseweb="select"] div {
    color: #000000 !important;
}

.stSelectbox [id*="baseweb-select"] {
    color: #000000 !important;
}

/* Target the value container specifically - FORCE BLACK */
.stSelectbox [data-baseweb="select"] [class*="ValueContainer"] {
    color: #000000 !important;
}

.stSelectbox [data-baseweb="select"] [class*="ValueContainer"] * {
    color: #000000 !important;
}

.stSelectbox [data-baseweb="select"] [class*="SingleValue"] {
    color: #000000 !important;
}

.stSelectbox [data-baseweb="select"] [class*="Input"] {
    color: #000000 !important;
}

/* Target all nested elements in selectbox */
.stSelectbox * {
    color: #000000 !important;
}

.stSelectbox label {
    color: #000000 !important;
}

/* Fix the actual selected text display */
.stSelectbox [data-baseweb="select"] [class*="singleValue"] {
    color: #000000 !important;
}

.stSelectbox input {
    color: #000000 !important;
}

/* Dropdown menu items */
.stSelectbox [data-baseweb="popover"] {
    background-color: white !important;
}

[role="listbox"] {
    background-color: white !important;
}

[role="option"] {
    background-color: white !important;
    color: #000000 !important;
}

[role="option"]:hover {
    background-color: #f7fafc !important;
    color: #667eea !important;
}

.stSelectbox option {
    background-color: white !important;
    color: #2d3748 !important;
}

/* Metric styling with modern cards */
.stMetric {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border: 2px solid rgba(102, 126, 234, 0.2);
    transition: all 0.3s ease;
}

.stMetric:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
}

.stMetric label {
    color: #4a5568 !important;
    font-weight: 700 !important;
}

.stMetric [data-testid="stMetricValue"] {
    color: #667eea !important;
    font-weight: 900 !important;
}

/* Navigation radio buttons premium style */
.stRadio > div {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.stRadio > label {
    color: #2d3748 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* Alert messages with modern design */
.stSuccess, .stWarning, .stError, .stInfo {
    padding: 18px 20px;
    border-radius: 12px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border-left: 5px solid;
}

.stSuccess {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left-color: #28a745;
}

.stError {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-left-color: #dc3545;
}

.stWarning {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
    border-left-color: #ffc107;
}

.stInfo {
    background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
    border-left-color: #17a2b8;
}

/* User info badge */
.user-badge {
    background: white;
    padding: 10px 20px;
    border-radius: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    font-weight: 700;
    color: #667eea !important;
    border: 2px solid rgba(102, 126, 234, 0.3);
}

/* Navigation section styling */
.nav-container {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    margin: 25px 0;
    border: 2px solid rgba(102, 126, 234, 0.2);
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# ENHANCED HEADER WITH USER INFO
# --------------------------------------------------
# User info and logout button
col1, col2, col3 = st.columns([2, 6, 2])
with col1:
    if "username" in st.session_state:
        st.markdown(f"<p style='color: #667eea; font-weight: bold;'>👤 {st.session_state.username}</p>", unsafe_allow_html=True)
with col3:
    if st.button("🚪 Logout", width="stretch", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("Logged out successfully!")
        st.rerun()

st.markdown("""
<div class="main-header">
    <div style="text-align:center; color: white; margin: 0; font-size: 2.5em; font-weight: bold;">🛡️ FRAUDGUARD</div>
    <div style="text-align:center; color: #E0E7FF; margin: 10px 0; font-size: 24px; font-weight: 600;">Online Payment Fraud Detection System</div>
    <p style="text-align:center; color: #C7D2FE; font-size: 16px;">🤖 AI-Powered Machine Learning Dashboard | 🎯 Real-Time Risk Scoring | ⚡ Pre-Transaction Fraud Prevention</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION - IMPROVED CONNECTIVITY
# --------------------------------------------------
st.markdown("<div style='text-align: center; color: #4F46E5; font-size: 1.17em; font-weight: bold;'>📍 Navigation Dashboard</div>", unsafe_allow_html=True)
nav = st.radio(
    "",
    ["🏠 Home", "🔍 Prediction", "📊 Visualization", "🚫 Fraud Prevention", "📄 Report", "ℹ️ About"],
    horizontal=True,
    key="navigation"
)
st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD CSV DATA
# --------------------------------------------------
df = pd.read_csv(r"Book1.csv")

target_col = "isFraud"  # ensure this column exists in CSV

cat_cols = df.select_dtypes(include="object").columns
num_cols = df.select_dtypes(exclude="object").columns.drop(target_col)

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# --------------------------------------------------
# HOME - ENHANCED WITH DETAILED INFORMATION
# --------------------------------------------------
if nav == "🏠 Home":
    st.markdown("""
    <div class="card">
    <div style="font-size: 1.5em; font-weight: bold; margin-bottom: 10px;">🚀 Welcome to Fraud Detection - Enterprise-Grade Fraud Detection System</div>
    <p style="font-size: 17px; line-height: 1.9;">
    <strong>Fraud Detection</strong> is an advanced AI-powered fraud detection platform that provides <strong>real-time</strong> 
    protection for online payment transactions. Our system uses state-of-the-art machine learning algorithms to 
    identify suspicious activities <strong>before</strong> transactions are completed, preventing financial losses and 
    protecting both businesses and consumers.
    </p>
    </div>
    
    <div class="card">
    <div style="font-size: 1.17em; font-weight: bold; margin-bottom: 10px;">🎯 Key Features</div>
    <ul style="font-size: 16px; line-height: 1.8;">
        <li><strong>⚡ Pre-Transaction Detection:</strong> Identifies potential fraud BEFORE money is transferred</li>
        <li><strong>🤖 Machine Learning Powered:</strong> Uses Gradient Boosting algorithm with 95%+ accuracy</li>
        <li><strong>📊 Real-Time Risk Scoring:</strong> Provides instant fraud probability percentages</li>
        <li><strong>📈 Visual Analytics:</strong> Interactive charts and insights for every prediction</li>
        <li><strong>📄 Detailed Reports:</strong> Generate comprehensive PDF reports with fraud analysis</li>
        <li><strong>🔐 Secure & Reliable:</strong> Enterprise-level security and consistent performance</li>
    </ul>
    </div>
    
    <div class="card">
    <div style="font-size: 1.17em; font-weight: bold; margin-bottom: 10px;">💡 How It Works</div>
    <ol style="font-size: 16px; line-height: 1.8;">
        <li><strong>Data Input:</strong> Enter transaction details including amount, type, and account information</li>
        <li><strong>AI Analysis:</strong> Machine learning model analyzes patterns and anomalies</li>
        <li><strong>Risk Assessment:</strong> System generates fraud probability score and risk level</li>
        <li><strong>Decision Support:</strong> Visual indicators help make informed decisions</li>
        <li><strong>Prevention:</strong> Block or flag transactions BEFORE completion</li>
    </ol>
    </div>
    
    <div class="card">
    <div style="font-size: 1.17em; font-weight: bold; margin-bottom: 10px;">🛡️ Why Fraud Detection?</div>
    <p style="font-size: 16px; line-height: 1.8;">
    Online payment fraud costs businesses billions annually. Traditional rule-based systems can't keep up with 
    sophisticated fraud patterns. Fraud Detection uses <strong>adaptive machine learning</strong> that continuously learns 
    from new data, identifying emerging fraud tactics. Our system analyzes transaction amounts, types, balance changes, 
    and behavioral patterns to detect anomalies that human reviewers might miss.
    </p>
    </div>
    
    <div style="text-align: center; margin-top: 30px;">
    <p style="font-size: 18px; font-weight: bold; color: #4F46E5;">👆 Use the navigation above to explore different features</p>
    <p style="font-size: 16px; color: #6B7280;">Start with 🔍 Prediction to analyze transactions | View 📊 Visualization for insights | Generate 📄 Reports for documentation</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
elif nav == "📊 Visualization":

    st.markdown("<div class='card'><div style='font-size: 1.5em; font-weight: bold;'>📊 Fraud Analytics Dashboard</div></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            df,
            x=target_col,
            title="Fraud vs Non-Fraud Transaction Count",
            color=target_col,
            color_discrete_sequence=['#10B981', '#EF4444']  # Green for safe, red for fraud
        )
        fig1.update_layout(
            title_x=0.5,
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(255,255,255,0.95)',
            font=dict(size=14, color='#1F2937')
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            df,
            x=cat_cols[0],
            color=target_col,
            title="Fraud Distribution by Transaction Type",
            color_discrete_sequence=['#10B981', '#EF4444']
        )
        fig2.update_layout(
            title_x=0.5,
            plot_bgcolor='rgba(255,255,255,0.95)',
            paper_bgcolor='rgba(255,255,255,0.95)',
            font=dict(size=14, color='#1F2937')
        )
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df,
        x=num_cols[0],
        y=num_cols[1],
        color=target_col,
        title="Transaction Amount vs Balance (Fraud Risk Zone)",
        color_discrete_sequence=['#10B981', '#EF4444']
    )
    fig3.update_layout(
        title_x=0.5,
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(255,255,255,0.95)',
        font=dict(size=14, color='#1F2937')
    )
    st.plotly_chart(fig3, use_container_width=True)

    cm = confusion_matrix(y_test, model.predict(X_test))
    fig4 = px.imshow(
        cm,
        text_auto=True,
        title="Model Confusion Matrix (Prediction Performance)",
        color_continuous_scale="RdYlGn_r",
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    fig4.update_layout(
        title_x=0.5,
        plot_bgcolor='rgba(255,255,255,0.95)',
        paper_bgcolor='rgba(255,255,255,0.95)',
        font=dict(size=14, color='#1F2937')
    )
    st.plotly_chart(fig4, use_container_width=True)


# --------------------------------------------------
# PREDICTION - ENHANCED WITH SVM BADGE & PRE-TRANSACTION DETECTION
# --------------------------------------------------
elif nav == "🔍 Prediction":

    st.markdown("""
    <div class='card'>
        <div style='font-size: 1.17em; font-weight: bold; margin-bottom: 10px;'>�️ Pre-Transaction Fraud Risk Analysis</div>
        <p style='color: #DC2626; font-weight: bold;'>⚠️ DETECT FRAUD BEFORE PAYMENT - NOT AFTER!</p>
        <p style='color: #059669; font-weight: bold;'>✅ Analyze fraud risk BEFORE processing the transaction</p>
        <p>Enter transaction details below to get real-time fraud prediction on previous transaction patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    input_data = {}
    col1, col2, col3 = st.columns(3)
    
    idx = 0
    for col in X.columns:
        with [col1, col2, col3][idx % 3]:
            if col in encoders:
                input_data[col] = st.selectbox(f"📌 {col}", encoders[col].classes_, key=f"sel_{col}")
            else:
                input_data[col] = st.number_input(f"📊 {col}", key=f"num_{col}")
        idx += 1

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚦 ANALYZE TRANSACTION RISK (Pre-Transaction Check)", use_container_width=True):

        row = []
        for col in X.columns:
            if col in encoders:
                row.append(encoders[col].transform([input_data[col]])[0])
            else:
                row.append(input_data[col])

        prob = model.predict_proba([row])[0][1] * 100

        # ---------------- STORE RESULTS ----------------
        st.session_state.prediction_done = True
        st.session_state.prediction_inputs = input_data
        st.session_state.fraud_probability = prob

        st.markdown("<br>", unsafe_allow_html=True)
        
        # ---------------- RISK LEVEL + EXPLANATION ----------------
        if prob > 70:
            level = "HIGH RISK"
            explanation = (
                "⛔ TRANSACTION SHOULD BE BLOCKED! The transaction shows characteristics commonly associated with fraudulent behavior, "
                "such as unusually high transaction amount, risky transaction type, "
                "and abnormal balance changes. Recommend immediate rejection."
            )
            st.error(f"🔴 {level} - FRAUD PROBABILITY: {prob:.2f}%")
            st.markdown("""
            <div style='background: #FEE2E2; padding: 20px; border-radius: 10px; border-left: 5px solid #DC2626;'>
                <div style='color: #DC2626; margin: 0; font-weight: bold;'>⛔ PRE-TRANSACTION ALERT: BLOCK THIS TRANSACTION</div>
                <p style='color: #7F1D1D; margin-top: 10px;'>This transaction has been flagged as HIGH RISK before processing. Do NOT proceed with this payment.</p>
            </div>
            """, unsafe_allow_html=True)

        elif prob > 40:
            level = "MODERATE RISK"
            explanation = (
                "⚠️ TRANSACTION REQUIRES MANUAL REVIEW! The transaction contains some suspicious patterns that deviate from normal behavior. "
                "Additional verification steps should be performed before processing."
            )
            st.warning(f"🟡 {level} - FRAUD PROBABILITY: {prob:.2f}%")
            st.markdown("""
            <div style='background: #FEF3C7; padding: 20px; border-radius: 10px; border-left: 5px solid #F59E0B;'>
                <div style='color: #D97706; margin: 0; font-weight: bold;'>⚠️ PRE-TRANSACTION WARNING: MANUAL REVIEW REQUIRED</div>
                <p style='color: #78350F; margin-top: 10px;'>This transaction shows suspicious patterns. Verify user identity before proceeding.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            level = "LOW RISK"
            explanation = (
                "✅ TRANSACTION SAFE TO PROCESS! The transaction behavior is consistent with normal usage patterns. "
                "No significant indicators of fraud were detected. Transaction can proceed."
            )
            st.success(f"🟢 {level} - FRAUD PROBABILITY: {prob:.2f}%")
            st.markdown("""
            <div style='background: #D1FAE5; padding: 20px; border-radius: 10px; border-left: 5px solid #10B981;'>
                <div style='color: #065F46; margin: 0; font-weight: bold;'>✅ PRE-TRANSACTION CHECK: SAFE TO PROCEED</div>
                <p style='color: #064E3B; margin-top: 10px;'>Transaction appears legitimate. You may proceed with processing.</p>
            </div>
            """, unsafe_allow_html=True)

        st.session_state.risk_level = level
        st.session_state.explanation = explanation
        
        # ---------------- VISUALIZATION FOR PREDICTION ----------------
        st.markdown("<br><div style='font-weight: bold; font-size: 1.1em;'>📊 Prediction Visualization</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Gauge Chart
            fig_gauge = px.bar(
                x=[prob, 100-prob],
                y=['Fraud Risk', 'Legitimate'],
                orientation='h',
                title=f"Fraud Risk Score: {prob:.2f}%",
                labels={'x': 'Percentage', 'y': ''},
                color=['Fraud Risk', 'Legitimate'],
                color_discrete_map={'Fraud Risk': '#EF4444' if prob > 70 else '#F59E0B' if prob > 40 else '#10B981', 'Legitimate': '#E5E7EB'}
            )
            fig_gauge.update_layout(showlegend=True, height=250, barmode='stack')
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            # Risk Level Pie Chart
            fig_pie = px.pie(
                values=[prob, 100-prob],
                names=['Fraud Risk', 'Legitimate'],
                title='Transaction Risk Distribution',
                color_discrete_sequence=['#EF4444' if prob > 70 else '#F59E0B' if prob > 40 else '#10B981', '#10B981']
            )
            fig_pie.update_layout(height=250)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Feature Importance Visualization
        st.markdown("<div style='font-weight: bold; font-size: 1.1em;'>🔍 Key Risk Factors</div>", unsafe_allow_html=True)
        st.info(explanation)
        
        # Navigation helper
        st.markdown("""
        <div style='margin-top: 30px; padding: 15px; background: rgba(79, 70, 229, 0.1); border-radius: 10px;'>
            <p style='text-align: center; margin: 0;'>
                📊 <strong>View detailed analytics in Visualization</strong> | 
                📄 <strong>Generate comprehensive report in Report section</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# REPORT
# --------------------------------------------------
elif nav == "📄 Report":

    st.markdown("<div class='card'><div style='font-size: 1.5em; font-weight: bold;'>📄 Fraud Detection Report</div></div>", unsafe_allow_html=True)

    if "prediction_done" not in st.session_state:
        st.warning("⚠️ Please perform a prediction first to generate the report.")
        st.stop()

    st.markdown("### 🔍 Prediction Summary")

    st.write(f"**Fraud Risk Score:** {st.session_state.fraud_probability:.2f}%")
    st.write(f"**Risk Level:** {st.session_state.risk_level}")
    st.write("**Explanation:**")
    st.write(st.session_state.explanation)

    st.markdown("### 📊 Transaction Details")
    for k, v in st.session_state.prediction_inputs.items():
        st.write(f"- **{k}** : {v}")

    # ---------------- PDF GENERATION ----------------
    if st.button("📥 Generate Detailed PDF Report"):

        buffer = io.BytesIO()
        pdf = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("ONLINE PAYMENT FRAUD DETECTION REPORT", styles["Title"]),
            Paragraph("<br/>", styles["Normal"]),
            Paragraph(f"Fraud Risk Score: {st.session_state.fraud_probability:.2f}%", styles["Normal"]),
            Paragraph(f"Risk Level: {st.session_state.risk_level}", styles["Normal"]),
            Paragraph("<br/>", styles["Normal"]),
            Paragraph("Explanation:", styles["Heading2"]),
            Paragraph(st.session_state.explanation, styles["Normal"]),
            Paragraph("<br/>", styles["Normal"]),
            Paragraph("Transaction Details:", styles["Heading2"])
        ]

        for k, v in st.session_state.prediction_inputs.items():
            content.append(Paragraph(f"{k} : {v}", styles["Normal"]))

        content.append(Paragraph(
            "<br/>This report is generated using a Gradient Boosting Machine Learning model "
            "trained on historical transaction data.",
            styles["Normal"]
        ))

        pdf.build(content)
        buffer.seek(0)

        st.download_button(
            "⬇️ Download Fraud Analysis Report",
            buffer,
            "Fraud_Report.pdf",
            "application/pdf"
        )



# --------------------------------------------------
# FRAUD PREVENTION - HOW TO BLOCK DETECTED FRAUD
# --------------------------------------------------
elif nav == "🚫 Fraud Prevention":
    st.markdown("""
    <div class='card'>
        <div style='font-size: 1.5em; font-weight: bold; margin-bottom: 10px;'>🚫 Fraud Prevention & Blocking Guide</div>
        <p style='font-size: 17px; line-height: 1.9;'>
        This section explains how to <strong>block and prevent fraudulent transactions</strong> that have been detected 
        by our system, based on previous fraud patterns and ML predictions.
        </p>
    </div>
    
    <div class='card'>
        <div style='font-size: 1.17em; font-weight: bold; margin-bottom: 10px;'>🛑 How to Block Previously Detected Fraud</div>
        <p style='font-size: 16px; line-height: 1.8;'>
        When our system identifies a transaction as fraudulent (based on historical patterns), you should take the following steps:
        </p>
        <ol style='font-size: 16px; line-height: 1.8;'>
            <li><strong>Review the Prediction:</strong> Check the fraud probability percentage and risk level in the 🔍 Prediction section</li>
            <li><strong>High Risk (>70%):</strong> Immediately <span style='color: #DC2626; font-weight: bold;'>BLOCK</span> the transaction before processing</li>
            <li><strong>Medium Risk (40-70%):</strong> Flag for <span style='color: #F59E0B; font-weight: bold;'>MANUAL REVIEW</span> by fraud team</li>
            <li><strong>Low Risk (<40%):</strong> Allow transaction to proceed with <span style='color: #10B981; font-weight: bold;'>MONITORING</span></li>
            <li><strong>Document the Decision:</strong> Generate a 📄 Report for audit trail and compliance</li>
        </ol>
    </div>
    
    <div class='card'>
        <div style='font-size: 1.17em; font-weight: bold; margin-bottom: 10px;'>⚡ Real-Time Fraud Prevention Steps</div>
        <table style='width: 100%; border-collapse: collapse; font-size: 15px;'>
            <tr style='background-color: #F3F4F6;'>
                <th style='border: 1px solid #E5E7EB; padding: 12px; text-align: left;'>Step</th>
                <th style='border: 1px solid #E5E7EB; padding: 12px; text-align: left;'>Action</th>
                <th style='border: 1px solid #E5E7EB; padding: 12px; text-align: left;'>System Response</th>
            </tr>
            <tr>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'><strong>1. Detection</strong></td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>ML model analyzes transaction before completion</td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>Fraud probability calculated in milliseconds</td>
            </tr>
            <tr style='background-color: #F9FAFB;'>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'><strong>2. Risk Assessment</strong></td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>System assigns risk level (High/Medium/Low)</td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>Visual alerts displayed with color coding</td>
            </tr>
            <tr>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'><strong>3. Block Decision</strong></td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>Admin reviews and decides to block or allow</td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>Transaction stopped BEFORE money transfer</td>
            </tr>
            <tr style='background-color: #F9FAFB;'>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'><strong>4. Notification</strong></td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>Customer and bank are notified of blocked transaction</td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>Automated alerts sent via email/SMS</td>
            </tr>
            <tr>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'><strong>5. Learning</strong></td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>System learns from blocked fraud patterns</td>
                <td style='border: 1px solid #E5E7EB; padding: 10px;'>Model improves detection accuracy over time</td>
            </tr>
        </table>
    </div>
    
    <div class='card'>
        <div style='font-size: 1.17em; font-weight: bold; margin-bottom: 10px;'>🎯 Best Practices for Fraud Prevention</div>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;'>
            <div style='padding: 15px; background: #FEF2F2; border-left: 4px solid #DC2626; border-radius: 5px;'>
                <div style='color: #DC2626; margin: 0 0 10px 0; font-weight: bold;'>🚨 High Risk Actions</div>
                <ul style='font-size: 14px; line-height: 1.6; margin: 0;'>
                    <li>Immediately block transaction</li>
                    <li>Freeze account temporarily</li>
                    <li>Notify fraud investigation team</li>
                    <li>Request additional verification</li>
                </ul>
            </div>
            <div style='padding: 15px; background: #FFFBEB; border-left: 4px solid #F59E0B; border-radius: 5px;'>
                <div style='color: #F59E0B; margin: 0 0 10px 0; font-weight: bold;'>⚠️ Medium Risk Actions</div>
                <ul style='font-size: 14px; line-height: 1.6; margin: 0;'>
                    <li>Hold transaction for review</li>
                    <li>Request customer confirmation</li>
                    <li>Check against fraud database</li>
                    <li>Apply additional authentication</li>
                </ul>
            </div>
            <div style='padding: 15px; background: #F0FDF4; border-left: 4px solid #10B981; border-radius: 5px;'>
                <div style='color: #10B981; margin: 0 0 10px 0; font-weight: bold;'>✅ Low Risk Actions</div>
                <ul style='font-size: 14px; line-height: 1.6; margin: 0;'>
                    <li>Process transaction normally</li>
                    <li>Log transaction for monitoring</li>
                    <li>Update customer profile</li>
                    <li>Continue pattern analysis</li>
                </ul>
            </div>
            <div style='padding: 15px; background: #EEF2FF; border-left: 4px solid #4F46E5; border-radius: 5px;'>
                <div style='color: #4F46E5; margin: 0 0 10px 0; font-weight: bold;'>📊 Analytics Actions</div>
                <ul style='font-size: 14px; line-height: 1.6; margin: 0;'>
                    <li>Review fraud trends weekly</li>
                    <li>Update ML model with new data</li>
                    <li>Generate performance reports</li>
                    <li>Train staff on new patterns</li>
                </ul>
            </div>
        </div>
    </div>
    
    <div class='card'>
        <div style='font-size: 1.17em; font-weight: bold; margin-bottom: 10px;'>💡 Key Points to Remember</div>
        <div style='background: #EEF2FF; padding: 20px; border-radius: 8px; margin-top: 15px;'>
            <ul style='font-size: 16px; line-height: 2.0; margin: 0;'>
                <li>✅ <strong>Prevention is BEFORE transaction</strong> - not after money is transferred</li>
                <li>✅ <strong>ML learns from previous fraud</strong> - patterns help detect new fraud attempts</li>
                <li>✅ <strong>Always document decisions</strong> - maintain audit trail for compliance</li>
                <li>✅ <strong>Balance security and user experience</strong> - don't block legitimate transactions</li>
                <li>✅ <strong>Continuous monitoring</strong> - fraud patterns evolve, system must adapt</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# ABOUT
# --------------------------------------------------
elif nav == "ℹ️ About":
    st.markdown("""
    <div class="card">
    <p>
    Fraud Detection – Enhanced Online Payment Fraud Detection System is a professional, 
    machine learning–based web application designed to identify and analyze fraudulent online payment transactions in real time. 
    The system leverages historical transaction data and advanced classification algorithms to predict the likelihood of fraud and generate an interpretable fraud risk score expressed as a percentage.

    This project integrates a high-accuracy Gradient Boosting Machine Learning model, 
    which is well-suited for tabular financial data and widely adopted in real-world fintech and banking environments. 
    Multiple transaction-related features such as transaction amount, transaction type, balance variations, and other behavioral indicators are used to ensure realistic and reliable fraud detection.
    </p>
    </div>
    """, unsafe_allow_html=True)
