import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset from the provided input data
data = {
    'type': ['PAYMENT', 'PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT', 'PAYMENT', 'PAYMENT', 'PAYMENT', 'PAYMENT', 'DEBIT', 'DEBIT', 'PAYMENT', 'PAYMENT', 'PAYMENT', 'PAYMENT', 'CASH_OUT', 'PAYMENT', 'PAYMENT', 'PAYMENT', 'TRANSFER', 'PAYMENT', 'DEBIT'],
    'amount': [9839.64, 1864.28, 181.0, 181.0, 11668.14, 7817.71, 7107.77, 7861.64, 4024.36, 5337.77, 9644.94, 3099.97, 2560.74, 11633.76, 4098.78, 229133.94, 1563.82, 1157.86, 671.64, 215310.3, 1373.43, 9302.79],
    'nameOrig': ['C1231006815', 'C1666544295', 'C1305486145', 'C840083671', 'C2048537720', 'C90045638', 'C154988899', 'C1912850431', 'C1265012928', 'C712410124', 'C1900366749', 'C249177573', 'C1648232591', 'C1716932897', 'C1026483832', 'C905080434', 'C761750706', 'C1237762639', 'C2033524545', 'C1670993182', 'C20804602', 'C1566511282']
}

df = pd.DataFrame(data)

# Train the classifier (you need to replace this with your own training process)
clf = RandomForestClassifier()
clf.fit(df[['amount']], df['type'])

# Function to predict fraud status for a new transaction ID
def predict_fraud(transaction_id, amount, clf):
    input_data = pd.DataFrame({'amount': [amount]})
    prediction = clf.predict(input_data)[0]
    return prediction

# Function to check if a transaction type is fraudulent
def is_fraud(transaction_type):
    fraudulent_types = ['TRANSFER', 'CASH_OUT']  # List of known fraudulent transaction types
    return transaction_type in fraudulent_types

# Streamlit code
#st.title("Fraud Detection System")

# Sidebar navigation
page = st.sidebar.radio("ONLINE PAYMENT FRAUD DETECTION", ["Home", "Prediction", "About"])
# Sidebar navigation
st.markdown("<h1 style='text-align: center; color: blue;'>ONLINE PAYMENT FRAUD DETECTION</h1>", unsafe_allow_html=True)



# Display content based on selected page
if page == "Home":
    st.header("Home Page")
    st.write("Welcome to the Home Page! Click on the buttons above to navigate.")
    st.image("Online-Fraud.jpeg", use_column_width=True)
elif page == "Prediction":
    st.header("Prediction Page")
    st.write("Welcome to the Prediction Page! Enter the transaction ID and amount below to check for fraud.")

    # Input fields for new transaction ID and amount
    transaction_id = st.text_input("Enter Transaction ID:")
    amount = st.number_input("Enter Amount:")

    # Button to check fraud for the entered transaction
    if st.button("Check Fraud"):
        if clf:
            prediction = predict_fraud(transaction_id, amount, clf)
            if is_fraud(prediction):
                st.write(f"Fraud Status for Transaction ID {transaction_id}: FRAUD (Transaction Type: {prediction})")
            else:
                st.write(f"Fraud Status for Transaction ID {transaction_id}: NOT FRAUD (Transaction Type: {prediction})")
        else:
            st.write("Please train the model first.")

    # Input fields for new transaction ID and amount for prediction
    new_transaction_id = st.text_input("Enter New Transaction ID:")
    new_amount = st.number_input("Enter New Amount:")

    # Button to predict fraud for the new transaction
    if st.button("Predict Fraud for New Transaction"):
        if clf:
            new_prediction = predict_fraud(new_transaction_id, new_amount, clf)
            if is_fraud(new_prediction):
                st.write(f"Fraud Status for New Transaction ID {new_transaction_id}: FRAUD (Transaction Type: {new_prediction})")
            else:
                st.write(f"Fraud Status for New Transaction ID {new_transaction_id}: NOT FRAUD (Transaction Type: {new_prediction})")
        else:
            st.write("Please train the model first.")

elif page == "About":
    st.header("About Page")
    st.write("Online payment fraud is a pervasive threat in the digital age, posing significant risks to businesses and consumers alike. The primary purpose of a fraud detection system is to identify and prevent fraudulent transactions in real-time, thereby safeguarding financial assets and preserving trust in online transactions. Leveraging machine learning algorithms such as logistic regression, decision trees, and neural networks, these systems analyze transaction data in-depth, extracting features like transaction amount, merchant information, and user behavior patterns to detect suspicious activity. Real-time fraud detection is crucial for mitigating losses, albeit challenging due to the need for swift decision-making and continuous monitoring. Evaluation metrics such as precision, recall, and accuracy are utilized to assess the effectiveness of these systems, ensuring optimal performance and refinement. Moreover, stringent adherence to data security and privacy regulations like GDPR and PCI-DSS is imperative to safeguard sensitive customer information. Through case studies and success stories, the tangible impact of these systems in reducing fraud losses and enhancing consumer trust is underscored, urging businesses to prioritize investment in robust fraud detection solutions to mitigate risks and safeguard financial transactions.")
