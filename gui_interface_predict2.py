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
            
             background: url("https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/image2.jpg");
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

def lr_predict_model(city, bedrooms, bathrooms, sqft_living, floors):

    url = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/Home_price_final.csv'
    df = pd.read_csv(url)
    model = load_model()
    loc_index = np.where(df.columns == city)[0][0]
    x = np.zeros(len(df.columns))
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
    #st.markdown(
    #'<p style="color: black;">This project aims at helping its users select certain parameters that will help them get a potential prediction from a model that will predict a price of a home in Washington</p>',
   # unsafe_allow_html=True
#)
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
        city_add = st.text_input('City', value="Bellevue", placeholder= 'Enter city')    
        bed_add = st.text_input("Bedroom Count", value="3.0", placeholder= 'Enter Bedroom')
        bath_add = st.text_input("Bathroom Count", value="2.5", placeholder= 'Enter Bathroom')
        sqft_add = st.text_input("Square Feet Living", value="2500.0", placeholder= 'Enter Square Feet')
        floor_add = st.text_input("Floor Count", value="2.0", placeholder= 'Enter Floor')
    
    # Convert input values to float
    city_add = city_add
    bed_add = float(bed_add)
    bath_add = float(bath_add)
    sqft_add = float(sqft_add)
    floor_add = float(floor_add)

    

    # Button to trigger regression model
    button_html = (
    f'<style>'
    f'.custom-button {{ background-color: green; color: black; }}'
    f'</style>'
    f'<button class="custom-button">Predict Price</button>'
)
    if st.button("Predict Price"):
        # execution of the model
        result = lr_predict_model(city_add, bed_add, bath_add, sqft_add, floor_add)

        # Display the result
        # st.success(f"Predicted value of the home: {np.round(result)}")
        formatted_result = '{:,.0f}'.format(np.round(result))
        st.markdown(
            f'<p style="color: white;">Predicted Value of the Home: ${formatted_result}</p>',
            unsafe_allow_html=True
        )


def help_info():
    st.title("Help and Navigation")
    st.write(
        """
        This section of the application is intended to guide the user in terms of troubleshooting \n\n

        *   Cities available: Bellevue,
        , Renton, Seattle, Redmond, Issaquah, Kirkland, Kent, Auburn, Sammamish, Federal Way, Shoreline, Woodinville, Maple Valley, Mercer Island, Burien, Snoqualmie, Kenmore, Des Moines, North Bend, Covington, Duvall,
        Lake Forest Park, Bothell, Newcastle, SeaTac, Tukwila, Vashon, Enumclaw, Carnation, Normandy Park, Clyde Hill, Medina, Fall City, Black Diamond, Ravensdale, Pacific, Yarrow Point, Skykomish, Preston, Milton, Inglewood-Finn Hill, Renton, Snoqualmie Pass, Beaux Arts Village\n\n

        *  Bedroom choices: 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00

        *  Bathroom choices: 0.00, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.25, 6.50, 6.75, 8.00

        *  Floor choices: 1.00, 1.50, 2.00, 2.50, 3.00, 3.50
        """
    )

# Define a function for session state management
def session_create():
    if "username" not in st.session_state:
        st.session_state.username = None
    return st.session_state
    
# Sign-in page
def sign_in_page():
    st.title("Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        
        if username == "person_1" and password == "gcu123":
            session_create()["username"] = username
            st.success("Signed In")
            st.experimental_rerun()  #used to try the log in again
        elif username == "person_2" and password == "gcu456":
            session_create()["username"] = username
            st.success("Signed In")
            st.experimental_rerun()
        elif username == "person_3" and password == "gcu789":
            session_create()["username"] = username
            st.success("Signed In")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password, try again")
    #show_help = st.session_state.get("show_help", False)    

    if st.button("Help"):
        st.sidebar.info(
            "Help:\n\n"
            "• Please check entered Username and Password carefully\n\n"
            
        )

def bedroom_trends_page():
    url = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/Home_Price_data.csv'
    df = pd.read_csv(url)
    

    data = df['bedrooms'].value_counts().reset_index()
    data.columns = ['bedrooms', 'Frequency']
    fig = px.line(data, x='bedrooms', y='Frequency', title='Bedrooms',
              labels={'Frequency': 'Frequency'}, line_shape='linear',
              line_dash_sequence=['solid'], markers=True, color='bedrooms')
    fig.update_traces(texttemplate='%{text}', textposition='top center')

    fig.update_layout(
            xaxis=dict(tickangle=45),
            xaxis_title='Bedrooms',
            yaxis_title='Frequency',
            autosize=False,
            width=1000,
            height=500
    )
    st.plotly_chart(fig)

def main():
    if not session_create()["username"]:
        sign_in_page()
        return    
    page = st.sidebar.selectbox('Select what you want to display:', ['Main Menu', 'Price Prediction','Bedroom Trends', 'Help'])

    if page == "Main Menu":
        menu_page()
        

    elif page == "Price Prediction":
        predict_model()
    elif page == "Bedroom Trends":
        bedroom_trends_page()
    elif page == "Help":
        help_info()
if __name__ == "__main__":
    main()
