import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import pickle
import streamlit as st
import requests
import os
from io import BytesIO


def page_one_bk():
    
        
    st.markdown(
         f"""
         <style>
         .stApp {{
            
             background: url("https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/image1.jpeg");            
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def page_two_bk():
    
        
    st.markdown(
         f"""
         <style>
         .stApp {{
            
             background: url(" https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/image2.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Function to load the model using pickle from GitHub
def load_model(url= r"https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/lr_mod.pkl"):
    #r"https://github.com/JoshuaSamuel07/Washington-Home-Price/blob/main/RandomFor_predict.pkl"
    response = requests.get(url)
    if response.status_code == 200:
        model = pickle.load(BytesIO(response.content))

        #model = pickle.load(BytesIO(response.content))
        return model
    else:
        st.error(f"Failed. Status code: {response.status_code}")
        return None
    
# Function to load the model using pickle
def load_model1(filename="lr_mod.pkl"):

    filename = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/lr_mod.pkl'
    #'C:\Users\joshu\Desktop\GCU\DSC-580\project\RandomFor_predict.pkl'
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model

def random_predict_model(city, bedrooms, bathrooms, sqft_living, floors):
    # current_directory = os.path.dirname(__file__)
    # file_path = os.path.join(current_directory, "Home_price_final.csv")
    # Home_price_final = pd.read_csv(file_path)
    url = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/Home_price_final.csv'
    X = pd.read_csv(url)
    model = load_model()
    loc_index = np.where(X.columns == city)[0][0]
    x = np.zeros(len(X.columns))
    x[1] = bedrooms
    x[2] = bathrooms
    x[3] = sqft_living
    x[4] = floors
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]
 
# Streamlit app
def menu_page():
    #background image for the menu_page page 
    page_one_bk()
    
    st.markdown(
    '<h1 style="color: black; white-space: nowrap;">Washington Home Price Prediction Application</h1>',
    unsafe_allow_html=True
)
    st.markdown(
    '<p style="color: black;">This project aims at helping its users select certain parameters that will help them get a potential prediction from a model that will predict a price of a home in Washington</p>',
    unsafe_allow_html=True
)
    st.markdown(
    '<p style="color: white;">Created By: Joshua Samuel</p>',
    unsafe_allow_html=True
)
    #st.title("Washington Home Price Prediction Application", unsafe_allow_html=True, style={"color": "white"})
    #st.write("This project aims at helping it's users select certain parameters that will help them get a potential prediction from a model that will predict a price of a home in Washington", unsafe_allow_html=True, style={"color": "white"})

def predict_model():
    #background image for the predict_model page
    page_two_bk()

    st.markdown('<h1 style="color: black; white-space: nowrap;">Washington Home Price Prediction Application</h1>',unsafe_allow_html=True)
    #st.title("Washington Home Price Prediction Application")

    #making the buttons center

    row_input = st.columns((2, 2, 2, 1))

    #city input at column 1
    with row_input[0]:
        #city input with adjusted width
        param1 = st.text_input('City', value="Bothell", placeholder= 'Enter city')    
        param2 = st.text_input("Bedroom Count", value="5.0", placeholder= 'Enter Bedroom')
        param3 = st.text_input("Bathroom Count", value="3.5", placeholder= 'Enter Bathroom')
        param4 = st.text_input("Square Feet Living", value="4500.0", placeholder= 'Enter Square Feet')
        param5 = st.text_input("Floor Count", value="2.0", placeholder= 'Enter Floor')
    
    # Convert input values to float
    param1 = param1
    param2 = float(param2)
    param3 = float(param3)
    param4 = float(param4)
    param5 = float(param5)

    

    # Button to trigger regression model
    button_html = (
    f'<style>'
    f'.custom-button {{ background-color: green; color: black; }}'
    f'</style>'
    f'<button class="custom-button">Predict Price</button>'
)
    if st.button("Predict Price"):
        # Perform regression and get the result
        result = random_predict_model(param1, param2, param3, param4, param5)

        # Display the result
        # st.success(f"Predicted value of the home: {np.round(result)}")
        formatted_result = '{:,.0f}'.format(np.round(result))
        st.markdown(
            f'<p style="color: white;">Predicted value of the home: ${formatted_result}</p>',
            unsafe_allow_html=True
        )

def main():
    page = st.sidebar.selectbox("Content", ["Main Menu", "Price Prediction"])

    if page == "Main Menu":
        menu_page()
    elif page == "Price Prediction":
        predict_model()

if __name__ == "__main__":
    main()