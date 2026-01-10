
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

        step	  = st.text_input(' step')
        
    with col2:

        type	 = st.text_input('type')
    

    with col3:

        amount	 = st.text_input('amount')
    
    with col3:
         
         oldbalanceOrg	 = st.text_input(' oldbalanceOrg')
    
    with col1:
       
       newbalanceOrig		 = st.text_input('newbalanceOrig')
       
    with col2:
         
         oldbalanceDest = st.text_input(' oldbalanceDest ')

    with col1:

        newbalanceDest		  = st.text_input('newbalanceDest')
        
    with col2:

        errorBalanceOrig	 = st.text_input('errorBalanceOrig')
    

    with col3:

        errorBalanceDest		 = st.text_input('errorBalanceDest')   

       


    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Result'):
        diab_prediction = diabetes_model.predict([[ step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,errorBalanceOrig,errorBalanceDest]])

        st.success('The output is {}'.format(diab_prediction ))
        
        

if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")








