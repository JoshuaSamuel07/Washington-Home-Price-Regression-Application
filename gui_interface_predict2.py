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
import seaborn as sns

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

# Function to connect to GitHub and the pickle file
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
    
# loading the model using pickle
def load_model1(filename="lr_mod.pkl"):

    filename = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/lr_mod.pkl'
    
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model

def lr_predict_model(city, bedrooms, bathrooms, sqft_living, floors):  #adding the parameters in importance of the final dataset

    url = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/Home_price_final.csv'   #gathering information from final cleaned dataset
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

def generate_report(city, bedrooms, bathrooms, sqft_living, floors, predicted_value):
    st.title("Prediction Report")
    st.write(f"**City:** {city}", unsafe_allow_html=True)
    st.write(f"**Bedrooms:** {bedrooms}", unsafe_allow_html=True)
    st.write(f"**Bathrooms:** {bathrooms}", unsafe_allow_html=True)
    st.write(f"**Square Feet Living:** {sqft_living}", unsafe_allow_html=True)
    st.write(f"**Floors:** {floors}", unsafe_allow_html=True)
    st.write(f"**Predicted Home Value:** ${predicted_value:,.0f}", unsafe_allow_html=True)


 
# Streamlit code
def menu_page():
    
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
def validate_nos(p2, p3,  p5 ):
    if  (p2>0) & (p2<8) & (p3>0) & (p3<8) & (p5>0) & (p5<8)  :
        # st.error("Please Enter Values Between 1-8")
        return False
    return True
    
    #st.title("Washington Home Price Prediction Application", unsafe_allow_html=True, style={"color": "white"})
    #st.write("This project aims at helping it's users select certain parameters that will help them get a potential prediction from a model that will predict a price of a home in Washington", unsafe_allow_html=True, style={"color": "white"})

def predict_model():
    
    page_two_bk()

    st.markdown('<h1 style="color: black; white-space: nowrap;">Washington Home Price Prediction Application</h1>',unsafe_allow_html=True)

    row_input = st.columns((2, 2, 2, 1))

    with row_input[0]:
        city_add = st.text_input('City', value="Bellevue", placeholder='Enter city')
        bed_add = st.text_input("Bedroom Count", value="3.0", placeholder='Enter Bedroom')
        bath_add = st.text_input("Bathroom Count", value="2.5", placeholder='Enter Bathroom')
        sqft_add = st.text_input("Square Feet Living", value="2500.0", placeholder='Enter Square Feet')
        floor_add = st.text_input("Floor Count", value="2.0", placeholder='Enter Floor')

    city_add = city_add
    bed_add = float(bed_add)
    bath_add = float(bath_add)
    sqft_add = float(sqft_add)
    floor_add = float(floor_add)

    if st.button("Predict Price"):
        result = lr_predict_model(city_add, bed_add, bath_add, sqft_add, floor_add)
        formatted_result = '{:,.0f}'.format(np.round(result))
        st.markdown(
            f'<p style="color: white;">Predicted Value of the Home: ${formatted_result}</p>',
            unsafe_allow_html=True
        )
        
        # Generate and display the report
        generate_report(city_add, bed_add, bath_add, sqft_add, floor_add, result)


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

# creating a session for each user after log in
def active_session():
    if "username" not in st.session_state:  #each session runs the api again
        st.session_state.username = None
    return st.session_state
    
# Sign-in page
def sign_in_page():
    st.title("Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        
        if username == "person_1" and password == "gcu123":
            active_session()["username"] = username
            st.success("Signed In")
            st.experimental_rerun()  #used to try the log in again
        elif username == "person_2" and password == "gcu456":
            active_session()["username"] = username
            st.success("Signed In")
            st.experimental_rerun()
        elif username == "person_3" and password == "gcu789":
            active_session()["username"] = username
            st.success("Signed In")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password, try again")
        

    if st.button("Help"):  #help button for the sign in page
        st.sidebar.info(
            "Help:\n\n"
            "Please check entered Username and Password carefully\n\n"
            
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
            xaxis=dict(tickangle=0),
            xaxis_title='Bedrooms',
            yaxis_title='Frequency',
            autosize=False,
            width=1000,
            height=500
    )
    st.plotly_chart(fig)
    csv_data = data.to_csv(index=False).encode()
    st.download_button(
        label="Download Bedroom Report",
        data=csv_data,
        file_name="bedroom_report.csv",
        key="download_bedroom_report"
    )



#def city_trends_page():
#    url = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/Home_Price_data.csv'
#    
#    df = pd.read_csv(url)
#    print(df.head())

#    data = df['city'].value_counts().reset_index()
#    data.columns = ['city', 'Frequency']
#    fig = px.bar(data, x='city', y='Frequency', title='City', labels={'Frequency': 'Frequency'}, text='Frequency')
#    fig.update_traces(texttemplate='%{text}', textposition='outside')
#    fig.update_layout(
#        xaxis=dict(tickangle=45),
#        xaxis_title='City',
#        yaxis_title='Frequency',
#        autosize=False,
#        width=1000,
#        height=500
#    )
#    st.plotly_chart(fig)
        # Provide a download link for the data
#    csv_data = data.to_csv(index=False).encode()
#    st.download_button(
#        label="Download City Report",
#        data=csv_data,
#        file_name="city_report.csv",
#        key="download_city_report"
#    )


def bathroom_trends_updated(from_year, to_year):
    url = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/Home_Price_data.csv'
    df = pd.read_csv(url)
    # added in a filter for selecting datewise
    date_fil_df = df[(df['yr_built'] >= from_year) & (df['yr_built'] <= to_year)]    
    fig_house, ax_house = plt.subplots(figsize=(24, 9))
    sns.boxenplot(x=date_fil_df['bathrooms'], y=date_fil_df['price'])
    ax_house.set_xlabel(str(date_fil_df['bathrooms']))
    ax_house.set_ylabel('Price')
    ax_house.set_title('From ' + str(from_year) + ' To ' + str(to_year) )
    ax_house.tick_params(axis='x', rotation=90)
    st.pyplot(fig_house)
    csv_data = date_fil_df.to_csv(index=False).encode()
    st.download_button(
        label="Download Bathroom Data",
        data=csv_data,
        file_name="bathroom.csv",
        key="bathroom_data"
    )


def city_trends_updated(from_year, to_year):
    url = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/Home_Price_data.csv'
    df = pd.read_csv(url)
    date_fil_df = df[(df['yr_built'] >= from_year) & (df['yr_built'] <= to_year)]
    data = date_fil_df['city'].value_counts().reset_index()       
    data.columns = ['city', 'Frequency']
    fig = px.bar(data, x='city', y='Frequency', labels={'Frequency': 'Frequency'}, text='Frequency')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        xaxis=dict(tickangle=50),
        xaxis_title='City Name',
        yaxis_title='Total Count',
        autosize=False,
        width=1000,
        height=500
    )
    st.plotly_chart(fig)
    csv_data = date_fil_df.to_csv(index=False).encode()
    st.download_button(
        label="Download City Trends Data",
        data=csv_data,
        file_name="city_data.csv",
        key="download_city_data"
    )

def house_trends_page(from_year, to_year):
    url = 'https://raw.githubusercontent.com/JoshuaSamuel07/Washington-Home-Price/main/Home_Price_data.csv'
    df = pd.read_csv(url)
    df_extract = df[(df['yr_built'] >= from_year) & (df['yr_built'] <= to_year)]
    fig_house, ax_house = plt.subplots(figsize=(24, 9))
    sns.countplot(x='yr_built', data=df_extract, palette='icefire', ax=ax_house) #using seaborn so user can zoom in to see the graphs clearly
    ax_house.set_xlabel('Year Built')
    ax_house.set_ylabel('Homes Built')
    ax_house.set_title('From ' + str(from_year) + ' To ' + str(to_year) )
    ax_house.tick_params(axis='x', rotation=75)
    st.pyplot(fig_house)
    csv_data = df_extract.to_csv(index=False).encode()
    st.download_button(
        label="Download House Trends Data",
        data=csv_data,
        file_name="house_trends.csv",
        key="download_house_data"
    )


def validate_year(year):  #ensuring the user can enter only 4 digit values for years
    if len(year) != 4:
        st.error("Year must be four digits.")
        return False
    return True
    
def main():
    if not active_session()["username"]:
        sign_in_page()
        return    
    page = st.sidebar.selectbox('Select Menu', ['Main Menu', 'Price Prediction','Bedroom Trends', 'City Trends','Bathroom Trends','House Trends','Help'])

    if page == "Main Menu":
        menu_page()
        

    elif page == "Price Prediction":
        predict_model()
    elif page == "Bedroom Trends":
        st.title("Range of Bedrooms available")
        bedroom_trends_page()
    elif page == "City Trends":
        st.title("Cities Count")
        from_year = st.selectbox("From", options=range(1900, 2024), index=0)
        to_year = st.selectbox("To", options=range(1900, 2024), index=len(range(1900, 2024))-1)
        city_trends_updated(from_year, to_year)
    elif page == "Bathroom Trends":
        st.title("Bathroom Count")
        from_year = st.selectbox("From", options=range(1900, 2014), index=0)
        to_year = st.selectbox("To", options=range(1900, 2014), index=len(range(1900, 2014))-1)
        if validate_year(str(from_year)) or  validate_year(str(to_year)):
            if from_year >= to_year:
                st.error("From Year must be less than To Year") #making sure the sure user cannot enter invalid dates
            else:
                bathroom_trends_updated(from_year, to_year)
    elif page == "House Trends":
        st.title("Home purchasing trends")
        from_year = st.selectbox("From", options=range(1950, 2014), index=0)
        to_year = st.selectbox("To", options=range(1950, 2014), index=len(range(1950, 2014))-1)
        if validate_year(str(from_year)) or  validate_year(str(to_year)):
            if from_year >= to_year:
                st.error("From Year must be less than To Year")
            else:
                house_trends_page(from_year, to_year)
        
  
    elif page == "Help":
        help_info()
if __name__ == "__main__":
    main()
