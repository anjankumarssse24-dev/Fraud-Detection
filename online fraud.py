
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

diabetes_model = pickle.load(open(r'D:\CITY_CLG\29_online payment\online payment\code\code\online.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('online fraud using ml',
                          
                          [ 'Home',
                 
                            'online fraud using ml'
                           
                            ],
                          default_index=0)
# Diabetes Prediction Page
if (selected == 'Home'):
    
    # page title
    st.title('online fraud using ml')

   # st.image("download.jpg")





 
    #st.image('download1.png')
if (selected == 'online fraud using ml'):
     # page title
    st.title('online fraud using ml')
    st.title(' 0 - no fraud && 1 - fraud')


    
  
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        step = st.number_input('step', min_value=0.0, value=0.0)
        
    with col2:
        type = st.number_input('type (encoded)', min_value=0.0, value=0.0, help='0-4: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN')
    
    with col3:
        amount = st.number_input('amount', min_value=0.0, value=0.0)
    
    with col1:
        oldbalanceOrg = st.number_input('oldbalanceOrg', min_value=0.0, value=0.0)
    
    with col2:
        newbalanceOrig = st.number_input('newbalanceOrig', min_value=0.0, value=0.0)
       
    with col3:
        oldbalanceDest = st.number_input('oldbalanceDest', min_value=0.0, value=0.0)

    with col1:
        newbalanceDest = st.number_input('newbalanceDest', min_value=0.0, value=0.0)
        
    with col2:
        errorBalanceOrig = st.number_input('errorBalanceOrig', value=0.0)
    
    with col3:
        errorBalanceDest = st.number_input('errorBalanceDest', value=0.0)   

       


    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Result'):
        try:
            # Ensure all inputs are converted to proper numeric types
            input_data = [[float(step), float(type), float(amount), float(oldbalanceOrg), 
                          float(newbalanceOrig), float(oldbalanceDest), float(newbalanceDest), 
                          float(errorBalanceOrig), float(errorBalanceDest)]]
            
            diab_prediction = diabetes_model.predict(input_data)
            
            if diab_prediction[0] == 0:
                st.success('✅ Result: NO FRAUD - Transaction is legitimate')
            else:
                st.error('🚨 Result: FRAUD DETECTED - High risk transaction!')
        except Exception as e:
            st.error(f'Error in prediction: {str(e)}')
            st.info('Please ensure all fields are filled with valid numeric values.')
        
        

if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")








