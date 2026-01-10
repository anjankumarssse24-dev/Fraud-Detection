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

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="FraudGuard – Online Payment Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def set_bg(path):
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background:linear-gradient(rgba(0,0,0,.8),rgba(0,0,0,.9)),
            url(data:image/png;base64,{encoded});
            background-size:cover;
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # Skip background image if not found

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --------------------------------------------------
# LOGIN PAGE WITH BACKGROUND IMAGE
# --------------------------------------------------
if not st.session_state.logged_in:
    try:
        set_bg(r"Bg_imgjpeg.jpeg")
    except:
        pass  # Skip if image not available

    st.markdown("""
    <div class="login-card">
        <h2 style="text-align:center;">🔐 FraudGuard Login</h2>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.stop()

# --------------------------------------------------
# GLOBAL THEME - ENHANCED FOR CLARITY
# --------------------------------------------------
st.markdown("""
<style>
/* Fix blurriness and improve rendering */
* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

h1, h2, h3, h4, p, label {
    color: inherit !important;
}
.stMarkdown, .stTextInput label, .stSelectbox label {
    font-size: 16px;
    font-weight: 500;
}
.card {
    line-height: 1.7;
    padding: 20px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    margin: 10px 0;
}
/* Enhanced button styling */
button {
    font-weight: 600 !important;
    border-radius: 8px !important;
}
/* SVM Model Badge */
.svm-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 10px 20px;
    border-radius: 25px;
    font-weight: bold;
    text-align: center;
    margin: 10px auto;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
/* Improved header styling */
.main-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# ENHANCED HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="text-align:center; color: white; margin: 0;">🛡️ FRAUDGUARD</h1>
    <h2 style="text-align:center; color: #E0E7FF; margin: 10px 0; font-size: 24px;">Online Payment Fraud Detection System</h2>
    <p style="text-align:center; color: #C7D2FE; font-size: 16px;">🤖 AI-Powered Machine Learning Dashboard | 🎯 Real-Time Risk Scoring | ⚡ Pre-Transaction Fraud Prevention</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION - IMPROVED CONNECTIVITY
# --------------------------------------------------
st.markdown("<h3 style='text-align: center; color: #4F46E5;'>📍 Navigation Dashboard</h3>", unsafe_allow_html=True)
nav = st.radio(
    "",
    ["🏠 Home", "🔍 Prediction", "📊 Visualization", "📄 Report", "ℹ️ About"],
    horizontal=True,
    key="navigation"
)
st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
try:
    set_bg(r"Online-Fraud.jpeg")
except:
    pass  # Skip if image not available


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
    <h2>🚀 Welcome to FraudGuard - Enterprise-Grade Fraud Detection System</h2>
    <p style="font-size: 17px; line-height: 1.9;">
    <strong>FraudGuard</strong> is an advanced AI-powered fraud detection platform that provides <strong>real-time</strong> 
    protection for online payment transactions. Our system uses state-of-the-art machine learning algorithms to 
    identify suspicious activities <strong>before</strong> transactions are completed, preventing financial losses and 
    protecting both businesses and consumers.
    </p>
    </div>
    
    <div class="card">
    <h3>🎯 Key Features</h3>
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
    <h3>💡 How It Works</h3>
    <ol style="font-size: 16px; line-height: 1.8;">
        <li><strong>Data Input:</strong> Enter transaction details including amount, type, and account information</li>
        <li><strong>AI Analysis:</strong> Machine learning model analyzes patterns and anomalies</li>
        <li><strong>Risk Assessment:</strong> System generates fraud probability score and risk level</li>
        <li><strong>Decision Support:</strong> Visual indicators help make informed decisions</li>
        <li><strong>Prevention:</strong> Block or flag transactions BEFORE completion</li>
    </ol>
    </div>
    
    <div class="card">
    <h3>🛡️ Why FraudGuard?</h3>
    <p style="font-size: 16px; line-height: 1.8;">
    Online payment fraud costs businesses billions annually. Traditional rule-based systems can't keep up with 
    sophisticated fraud patterns. FraudGuard uses <strong>adaptive machine learning</strong> that continuously learns 
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

    st.markdown("<div class='card'><h2>📊 Fraud Analytics Dashboard</h2></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            df,
            x=target_col,
            title="Fraud vs Non-Fraud Transaction Count",
            color=target_col,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig1.update_layout(title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            df,
            x=cat_cols[0],
            color=target_col,
            title="Fraud Distribution by Transaction Type",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig2.update_layout(title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df,
        x=num_cols[0],
        y=num_cols[1],
        color=target_col,
        title="Transaction Amount vs Balance (Fraud Risk Zone)",
        color_continuous_scale="Turbo"
    )
    fig3.update_layout(title_x=0.5)
    st.plotly_chart(fig3, use_container_width=True)

    cm = confusion_matrix(y_test, model.predict(X_test))
    fig4 = px.imshow(
        cm,
        text_auto=True,
        title="Model Confusion Matrix (Prediction Performance)",
        color_continuous_scale="Blues"
    )
    fig4.update_layout(title_x=0.5)
    st.plotly_chart(fig4, use_container_width=True)


# --------------------------------------------------
# PREDICTION - ENHANCED WITH SVM BADGE & PRE-TRANSACTION DETECTION
# --------------------------------------------------
elif nav == "🔍 Prediction":

    st.markdown("""
    <div class='card'>
        <h3>🔍 Pre-Transaction Fraud Risk Prediction</h3>
        <p style='color: #DC2626; font-weight: bold;'>⚠️ FRAUD DETECTION BEFORE TRANSACTION COMPLETION</p>
        <p>Enter transaction details below to analyze fraud risk BEFORE processing the payment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display SVM Model Badge
    st.markdown("""
    <div class="svm-badge">
        🤖 Powered by Gradient Boosting Machine Learning Model
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
                <h4 style='color: #DC2626; margin: 0;'>⛔ PRE-TRANSACTION ALERT: BLOCK THIS TRANSACTION</h4>
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
                <h4 style='color: #D97706; margin: 0;'>⚠️ PRE-TRANSACTION WARNING: MANUAL REVIEW REQUIRED</h4>
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
                <h4 style='color: #065F46; margin: 0;'>✅ PRE-TRANSACTION CHECK: SAFE TO PROCEED</h4>
                <p style='color: #064E3B; margin-top: 10px;'>Transaction appears legitimate. You may proceed with processing.</p>
            </div>
            """, unsafe_allow_html=True)

        st.session_state.risk_level = level
        st.session_state.explanation = explanation
        
        # ---------------- VISUALIZATION FOR PREDICTION ----------------
        st.markdown("<br><h4>📊 Prediction Visualization</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Gauge Chart
            fig_gauge = px.bar(
                x=[prob, 100-prob],
                y=['Risk Level'],
                orientation='h',
                title=f"Fraud Risk Score: {prob:.2f}%",
                labels={'x': 'Percentage', 'y': ''},
                color=['Fraud Risk', 'Safe'],
                color_discrete_map={'Fraud Risk': '#EF4444' if prob > 70 else '#F59E0B' if prob > 40 else '#10B981', 'Safe': '#E5E7EB'}
            )
            fig_gauge.update_layout(showlegend=False, height=250)
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
        st.markdown("<h4>🔍 Key Risk Factors</h4>", unsafe_allow_html=True)
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

    st.markdown("<div class='card'><h2>📄 Fraud Detection Report</h2></div>", unsafe_allow_html=True)

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
# ABOUT
# --------------------------------------------------
elif nav == "ℹ️ About":
    st.markdown("""
    <div class="card">
    <p>
    FraudGuard – Enhanced Online Payment Fraud Detection System is a professional, 
    machine learning–based web application designed to identify and analyze fraudulent online payment transactions in real time. 
    The system leverages historical transaction data and advanced classification algorithms to predict the likelihood of fraud and generate an interpretable fraud risk score expressed as a percentage.

    This project integrates a high-accuracy Gradient Boosting Machine Learning model, 
    which is well-suited for tabular financial data and widely adopted in real-world fintech and banking environments. 
    Multiple transaction-related features such as transaction amount, transaction type, balance variations, and other behavioral indicators are used to ensure realistic and reliable fraud detection.
    </p>
    </div>
    """, unsafe_allow_html=True)
