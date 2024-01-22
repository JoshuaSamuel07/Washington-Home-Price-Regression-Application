import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import requests
import os
from io import BytesIO
import plotly.express as px


st.set_page_config(layout="centered")


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


def help_info():
    st.title("Help")
    st.write(
        """
        *   Welcome to the Regression Application Deployment Guide! \n\n

        *   Follow the steps below to successfully deploy and run the application:\n\n

        *   The application is intended to provide a user-friendly experience in predicting the home price using linear regression.
        """
    )

# Define a function for session state management
def get_session_state():
    if "username" not in st.session_state:
        st.session_state.username = None
    return st.session_state
    
# Sign-in page
def sign_in_page():
    st.title("Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        # Replace with actual authentication logic
        if username == "user1" and password == "user1":
            get_session_state()["username"] = username
            st.success("Sign in successful!")
            st.experimental_rerun()  # Rerun the app to display the introduction page
        elif username == "user2" and password == "user2":
            get_session_state()["username"] = username
            st.success("Sign in successful!")
            st.experimental_rerun()  # Rerun the app to display the introduction page
        elif username == "user3" and password == "user3":
            get_session_state()["username"] = username
            st.success("Sign in successful!")
            st.experimental_rerun()  # Rerun the app to display the introduction page
        else:
            st.error("Invalid username or password")
    show_help = st.session_state.get("show_help", False)    

    if st.button("Help"):
        st.sidebar.info(
            "• Welcome to the Regression Application Deployment Guide!\n\n"
            "• Follow the steps below to successfully deploy and run the application.\n\n"
            "• The application is intended to provide a user-friendly experience in predicting the home price using linear regression."
        )

def bedroom_trends_page():
    url = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/Home_Price_data.csv'
    df = pd.read_csv(url)
    data = df['bedrooms'].value_counts().reset_index()
    data.columns = ['bedrooms', 'Frequency']
    fig = px.bar(data, x='bedrooms', y='Frequency', title='Bedrooms', labels={'Frequency': 'Frequency'}, text='Frequency')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        xaxis=dict(tickangle=45),
        xaxis_title='Bedrooms',
        yaxis_title='Frequency',
        autosize=False,
        width=1000,
        height=500
    )
    st.plotly_chart(fig)  # Use st.plotly_chart() instead of st.pyplot()
    # Provide a download link for the data
    csv_data = data.to_csv(index=False).encode()
    st.download_button(
        label="Download Bedroom Data",
        data=csv_data,
        file_name="bedroom_data.csv",
        key="download_bedroom_data"
    )

def main():
    if not get_session_state()["username"]:
        sign_in_page()
        return    
    page = st.sidebar.radio('Select what you want to display:', ['Main Menu', 'Price Prediction', 'Help','Bedroom Trends'])

    if page == "Main Menu":
        menu_page()
        

    elif page == "Price Prediction":
        predict_model()
    elif page == "Bedroom Trends":  # Handle the new page
        bedroom_trends_page()
    elif page == "Help":  # Handle the new page
        help_info()
if __name__ == "__main__":
    main()
