import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Pizza Price Prediction",layout='centered')
st.title("Pizza Price Prediction")
# st.image('health-costs.jpg')
st.text("Fill-in the following values to predict the price of your pizza")

company = st.selectbox('Company',['A','B','C','D','E'])
variant = st.selectbox('Variant',['double_signature', 'american_favorite', 'super_supreme',
       'meat_lovers', 'double_mix', 'classic', 'crunchy', 'new_york',
       'double_decker', 'spicy_tuna', 'BBQ_meat_fiesta', 'BBQ_sausage',
       'extravaganza', 'meat_eater', 'gournet_greek', 'italian_veggie',
       'thai_veggie', 'american_classic', 'neptune_tuna', 'spicy tuna'])
topping = st.selectbox('Topping',['mushrooms', 'black_papper', 'smoked_beef', 'papperoni',
       'mozzarella', 'chicken', 'tuna', 'meat', 'sausage', 'onion',
       'vegetables', 'beef'])
size = st.selectbox('Size',['reguler', 'jumbo', 'small', 'medium', 'large', 'XL'])
diameter = st.slider('Diameter [inch]',8.0,25.0,step=0.5)
extra_cheese = st.checkbox('Extra Cheese')
extra_sauce = st.checkbox('Extra Sauce')
extra_mushroom = st.checkbox('Extra Mushroom')
btn = st.button("Submit")

if btn:
    


    scaler = joblib.load('scaler3.pkl')
    target_scaler = joblib.load('target_scaler3.pkl')
    model = joblib.load('model.pkl')
    # st.text(diameter)
    company_mapping = {'A':0,'B':1,'C':2,'D':3,'E':4}
    variant_mapping = {'double_signature': 8,
 'american_favorite': 3,
 'super_supreme': 18,
 'meat_lovers': 13,
 'double_mix': 7,
 'classic': 4,
 'crunchy': 5,
 'new_york': 15,
 'double_decker': 6,
 'spicy_tuna': 17,
 'BBQ_meat_fiesta': 0,
 'BBQ_sausage': 1,
 'extravaganza': 9,
 'meat_eater': 12,
 'gournet_greek': 10,
 'italian_veggie': 11,
 'thai_veggie': 19,
 'american_classic': 2,
 'neptune_tuna': 14,
 'spicy tuna': 16}
    topping_mapping = {'mushrooms': 5,
 'black_papper': 1,
 'smoked_beef': 9,
 'papperoni': 7,
 'mozzarella': 4,
 'chicken': 2,
 'tuna': 10,
 'meat': 3,
 'sausage': 8,
 'onion': 6,
 'vegetables': 11,
 'beef': 0}
    size_mapping = {'reguler': 4, 'jumbo': 1, 'small': 5, 'medium': 3, 'large': 2, 'XL': 0}
    # extra_mapping = {'True':1,'False':0}

    company_encoded = company_mapping[company]
    variant_encoded = variant_mapping[variant]
    topping_encoded = topping_mapping[topping]
    size_encoded = size_mapping[size]
    # extra_mushroom_encoded = extra_mapping[extra_mushroom]
    # extra_sauce_encoded = extra_mapping[extra_sauce]
    # extra_cheese_encoded = extra_mapping[extra_cheese]
    # st.text(int(extra_cheese))



    input_data = np.array([[int(diameter),int(extra_sauce),int(extra_cheese),int(extra_mushroom),size_encoded,variant_encoded,topping_encoded,company_encoded]])
    # Index(['diameter', 'extra_sauce_encoded', 'extra_cheese_encoded',
    #    'extra_mushrooms_encoded', 'size_encoded', 'variant_encoded',
    #    'topping_encoded', 'company_encoded'],
    #   dtype='object')
    input_data_scaled = scaler.transform(input_data)

    prediction_scaled = model.predict(input_data_scaled)
    # st.text(prediction_scaled.shape)
    prediction_original = target_scaler.inverse_transform(prediction_scaled.reshape(-1,1))
    
    # st.text(target_scaler.scale_.shape)
    st.success(f'Your Pizza Price is Rp {round(prediction_original[0,0],2)}')
