import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = joblib.load('fraud_detection_model.pkl')

label_encoder = LabelEncoder()

def make_prediction(input_data):
    input_data['merchant'] = label_encoder.fit_transform(input_data['merchant'])
    input_data['category'] = label_encoder.fit_transform(input_data['category'])
    input_data['job'] = label_encoder.fit_transform(input_data['job'])
    input_data['city'] = label_encoder.fit_transform(input_data['city'])  
    input_data['state'] = label_encoder.fit_transform(input_data['state'])  

    input_data = input_data.drop(columns=['dob'])  

    expected_features = 15636
    current_features = input_data.shape[1]

   
    if current_features < expected_features:
        padding = np.zeros((input_data.shape[0], expected_features - current_features))
        input_data_padded = np.hstack([input_data.values, padding])
    else:
        input_data_padded = input_data.values

    
    prediction = model.predict(input_data_padded)
    return prediction[0]

st.title('Real-Time Fraud Detection')


merchant = st.selectbox('Select Merchant', ['Stokes, Christiansen and Sipes', 'Merchant A', 'Merchant B'])
category = st.selectbox('Select Category', ['grocery_net', 'ecommerce', 'retail'])
amount = st.number_input('Enter Amount', min_value=0.0, step=0.01)
city = st.text_input('Enter City', value='Wales')
state = st.text_input('Enter State', value='AK')
latitude = st.number_input('Enter Latitude', min_value=-90.0, max_value=90.0, step=0.01)
longitude = st.number_input('Enter Longitude', min_value=-180.0, max_value=180.0, step=0.01)
city_population = st.number_input('Enter City Population', min_value=0, step=1)
job = st.selectbox('Select Job', ['Administrator', 'Engineer', 'Manager', 'Clerk'])
dob = st.date_input('Enter Date of Birth', value=pd.to_datetime('1939-09-11'))
transaction_number = st.text_input('Enter Transaction Number')
merchant_latitude = st.number_input('Enter Merchant Latitude', min_value=-90.0, max_value=90.0, step=0.01)
merchant_longitude = st.number_input('Enter Merchant Longitude', min_value=-180.0, max_value=180.0, step=0.01)


input_data = {
    'merchant': [merchant],
    'category': [category],
    'amt': [amount],
    'city': [city],
    'state': [state],
    'lat': [latitude],
    'long': [longitude],
    'city_pop': [city_population],
    'job': [job],
    'dob': [dob],
    'trans_num': [transaction_number],
    'merch_lat': [merchant_latitude],
    'merch_long': [merchant_longitude]
}


input_df = pd.DataFrame(input_data)

if st.button('Predict'):
    try:
       
        result = make_prediction(input_df)
        
        if result == 0:
            st.success("Prediction Result: Safe Transaction")
        else:
            st.error("Prediction Result: Fraud Transaction")
    except Exception as e:
        st.error(f"Error: {str(e)}")
