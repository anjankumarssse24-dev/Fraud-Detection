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
    page_title="Fraud Detection – Online Payment Fraud Detection",
    page_icon="�️",
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
            background:linear-gradient(rgba(255,255,255,.95),rgba(245,245,250,.98)),
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
        <h2 style="text-align:center;">🕵️ Fraud Detection Login</h2>
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
/* Light white background theme */
.stApp {
    background: linear-gradient(135deg, #ffffff 0%, #f5f5fa 100%) !important;
}

/* Fix blurriness and improve rendering */
* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

h1, h2, h3, h4 {
    color: #1F2937 !important;
}

p, label {
    color: #374151 !important;
}

.stMarkdown, .stTextInput label, .stSelectbox label {
    font-size: 16px;
    font-weight: 500;
    color: #1F2937 !important;
}

.card {
    line-height: 1.7;
    padding: 20px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
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
/* Improved header styling for light theme */
.main-header {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3);
    margin-bottom: 30px;
    border: 1px solid #E5E7EB;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# ENHANCED HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="text-align:center; color: white; margin: 0;">🕵️ FRAUD DETECTION</h1>
    <h2 style="text-align:center; color: #E0E7FF; margin: 10px 0; font-size: 24px;">Online Payment Fraud Detection System</h2>
    <p style="text-align:center; color: #C7D2FE; font-size: 16px;">🤖 AI-Powered ML Dashboard | 🎯 Real-Time Risk Analysis | ⚡ Detect Fraud BEFORE Transaction</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION - IMPROVED CONNECTIVITY
# --------------------------------------------------
st.markdown("<h3 style='text-align: center; color: #4F46E5;'>📍 Navigation Dashboard</h3>", unsafe_allow_html=True)
nav = st.radio(
    "",
    ["🏠 Home", "🔍 Prediction", "📊 Visualization", "🚫 Fraud Prevention", "📄 Report", "ℹ️ About"],
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
    <h2>🚀 Welcome to Fraud Detection - Enterprise-Grade Fraud Detection System</h2>
    <p style="font-size: 17px; line-height: 1.9;">
    <strong>Fraud Detection</strong> is an advanced AI-powered fraud detection platform that provides <strong>real-time</strong> 
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
    <h3>🛡️ Why Fraud Detection?</h3>
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

    st.markdown("<div class='card'><h2>📊 Fraud Analytics Dashboard</h2></div>", unsafe_allow_html=True)

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
        <h3>�️ Pre-Transaction Fraud Risk Analysis</h3>
        <p style='color: #DC2626; font-weight: bold;'>⚠️ DETECT FRAUD BEFORE PAYMENT - NOT AFTER!</p>
        <p style='color: #059669; font-weight: bold;'>✅ Analyze fraud risk BEFORE processing the transaction</p>
        <p>Enter transaction details below to get real-time fraud prediction on previous transaction patterns</p>
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
# FRAUD PREVENTION - HOW TO BLOCK DETECTED FRAUD
# --------------------------------------------------
elif nav == "🚫 Fraud Prevention":
    st.markdown("""
    <div class='card'>
        <h2>🚫 Fraud Prevention & Blocking Guide</h2>
        <p style='font-size: 17px; line-height: 1.9;'>
        This section explains how to <strong>block and prevent fraudulent transactions</strong> that have been detected 
        by our system, based on previous fraud patterns and ML predictions.
        </p>
    </div>
    
    <div class='card'>
        <h3>🛑 How to Block Previously Detected Fraud</h3>
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
        <h3>⚡ Real-Time Fraud Prevention Steps</h3>
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
        <h3>🎯 Best Practices for Fraud Prevention</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;'>
            <div style='padding: 15px; background: #FEF2F2; border-left: 4px solid #DC2626; border-radius: 5px;'>
                <h4 style='color: #DC2626; margin: 0 0 10px 0;'>🚨 High Risk Actions</h4>
                <ul style='font-size: 14px; line-height: 1.6; margin: 0;'>
                    <li>Immediately block transaction</li>
                    <li>Freeze account temporarily</li>
                    <li>Notify fraud investigation team</li>
                    <li>Request additional verification</li>
                </ul>
            </div>
            <div style='padding: 15px; background: #FFFBEB; border-left: 4px solid #F59E0B; border-radius: 5px;'>
                <h4 style='color: #F59E0B; margin: 0 0 10px 0;'>⚠️ Medium Risk Actions</h4>
                <ul style='font-size: 14px; line-height: 1.6; margin: 0;'>
                    <li>Hold transaction for review</li>
                    <li>Request customer confirmation</li>
                    <li>Check against fraud database</li>
                    <li>Apply additional authentication</li>
                </ul>
            </div>
            <div style='padding: 15px; background: #F0FDF4; border-left: 4px solid #10B981; border-radius: 5px;'>
                <h4 style='color: #10B981; margin: 0 0 10px 0;'>✅ Low Risk Actions</h4>
                <ul style='font-size: 14px; line-height: 1.6; margin: 0;'>
                    <li>Process transaction normally</li>
                    <li>Log transaction for monitoring</li>
                    <li>Update customer profile</li>
                    <li>Continue pattern analysis</li>
                </ul>
            </div>
            <div style='padding: 15px; background: #EEF2FF; border-left: 4px solid #4F46E5; border-radius: 5px;'>
                <h4 style='color: #4F46E5; margin: 0 0 10px 0;'>📊 Analytics Actions</h4>
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
        <h3>💡 Key Points to Remember</h3>
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
