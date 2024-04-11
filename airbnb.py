import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import io
import pandas as pd
import pymongo
import plotly.express as px
import torch
import seaborn as s
#from flask import Flask
#import requests
from matplotlib import pyplot as plt
#from tableau_tools import*
import numpy as np
import os
#from urllib.request import urlopen # Website connections
#import codecs
import csv
import sys
st.set_page_config(layout="wide")


#home page --contains about airbnb and technologies 
def home():
    col1, col2 = st.columns([4, 4])
    with col1:
        st.title("AIRBNB ANALYSIS")
        icon=Image.open(r"C:\Users\navit\OneDrive\Pictures\IMAGES\airbnb.png")
        st.image(icon,use_column_width=True)
        st.write('''Airbnb: A Global Community
    Airbnb is a platform that connects hosts who have space to share with guests who need a place to stay,

    It started in 2008 when two designers rented out air mattresses in their apartment to three travelers in San Francisco
    Now, it offers millions of unique accommodations and experiences around the world, from cozy cabins to luxury villas

    Airbnb makes money by charging a service fee to both hosts and guests for each booking, as well as offering other services like adventures and online experiences

    Airbnb verifies the identity and background of its users and provides a secure payment system and a review system to ensure trust and quality

    Airbnb is more than just a lodging service, it is a community that enables people to explore new places, cultures, and passions''')
    with col2:
        st.title("TECHNOLOGIES")
        st.write("Python scripting")
        st.write("Visualization")
        st.write("EDA")
        st.write("Streamlit")
        st.write("MongoDB")
        st.write("Tableau")

# CREATING CONNECTION WITH MONGODB ATLAS AND RETRIEVING THE DATA
client = pymongo.MongoClient("Enter your connection string")
db = client.sample_airbnb
col = db.listingsAndReviews

# READING THE CLEANED DATAFRAME
data = pd.read_csv('AIRBNB1.csv')
df=pd.DataFrame(data)


#UNIVARIATE ANALYSIS

def univariate(df):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    #CATEGORICAL ANALYSIS OF ROOM TYPES
    plt.title("CATEGORICAL VARIABLE ANALYSIS")
    plt.figure(figsize=(10,8))
    s.countplot(data=df,x=df.Room_type.values,order=df.Room_type.value_counts().index[:10])
    plt.title("Room types")
    st.pyplot() 
   
    #CATEGORICAL ANALYSIS OF PROPERTY TYPES
    plt.figure(figsize=(15,8))
    s.countplot(data=df,x=df.Property_type.values,order=df.Property_type.value_counts().index[:10])
    plt.title("Top 10 Property Types available") 
    st.pyplot()  
    
    
    #CATEGORICAL ANALYSIS OF HOST WITH LISTING
    plt.figure(figsize=(10,8))
    s.countplot(data=df,x=df.Host_name,order=df.Host_name.value_counts().index[:10])
    plt.title("Top 10 Hosts with Highest number of Listings")
    st.pyplot() 
    
  
    #NUMERICAL ANALALYSIS OF AVAILABILITY 
    plt.title("NUMERICAL VARIABLE ANALYSIS")
    plt.figure(figsize=(10,8))
    s.boxplot(data=df,x=df.Availability_365)
    plt.title("Availability_365")
    st.pyplot() 
    
    #SEPERATION OF CATEGORICAL AND NUMERICAL COLUMN
    categorical_cols=df.select_dtypes(include=['object']).columns
    numumerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    #ALL NUMERICAL COLUMNS ANALYSIS
    for Price in numumerical_cols:
            plt.title("Numerical variable analysis")

            print('Skew :', round(df[Price].skew(), 2))
            plt.figure(figsize = (15, 4))
            plt.subplot(1, 2, 1)
            df[Price].hist(grid=False)
            plt.ylabel('count')
            plt.subplot(1, 2, 2)
            s.boxplot(x=df[Price])
            st.pyplot()




#BIVARIATE ANALYSIS
def bivariate(df):
    plt.title("BIVARIATE ANALYSIS")
    
    # Convert 'Price' column to numeric data type
    #ANALYSIS OF PRICE AND COUNTRY
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    plt.figure(figsize=(16, 8))
    s.scatterplot(data=df, y="Country", x="Price")
    plt.title('Price vs Country')
    st.pyplot()
    
    
    #ANALYSIS OF AVAILABILITY AND COUNTRY_CODE
    plt.figure(figsize=(16,8))
    s.scatterplot(data=df, y="Country_code", x="Availability_365")
    plt.title('Availability_365')
    st.pyplot()
     
     
    #ANALYSIS OF REVIEWSCORE AND COUNTRY_CODE
    plt.figure(figsize=(16,8))
    s.scatterplot(data=df, y="Country_code", x="Review_scores")
    plt.title('REVIEWSCORE OF COUNTRY')
    st.pyplot()
    
    
    #BIVARIATE ANALYSIS OF NUMERICAL VARIABLES
    g = s.PairGrid(df,vars=['Price','Min_nights','Max_nights','Availability_365','No_of_reviews'])
    g.map_upper(s.scatterplot, color='crimson')
    g.map_lower(s.scatterplot, color='limegreen')
    g.map_diag(plt.hist, color='orange')
    plt.title('BIVARIATE ANALYSIS OF NUMERICAL VARIABLES')
    st.pyplot()
    
    
    #categorical bivariate analysis
    plt.title("country Vs Price")
    df.groupby('Country')['Price'].mean().nlargest(10).plot.bar()
    st.pyplot()
  
    plt.title("property_type Vs Price", fontsize=18)
    df.groupby('Property_type')['Price'].mean().nlargest(10).plot.bar()
    st.pyplot()
    plt.title("Host name listing", fontsize=18)
    df.groupby('Host_name')['Price'].mean().nlargest(10).plot.bar()
    


#MULTIVARIATE ANALYSIS
def correlation_plot(df):
    # Identify and handle problematic columns
    non_numeric_cols_to_drop = ['Id','Country','Is_location_exact','Host_id', 'Longitude', 'Latitude']
    df_numeric = df.drop(non_numeric_cols_to_drop, axis=1)
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce').dropna()

    # Plot correlation heatmap
    plt.figure(figsize=(8, 8))
    s.heatmap(df_numeric.corr(), annot=True,cmap='RdPu')
    st.pyplot()


#TABLEAU ANALYSIS

def ana():
    st.title("Embedding Tableau Dashboard in Streamlit")
    st.markdown("## Tableau Dashboard")
    icon=Image.open(r"C:\Users\navit\Downloads\final dashboard 1.png")
    st.image(icon,use_column_width=True)
    
    icon=Image.open(r"C:\Users\navit\Downloads\final dashboard 2.png")
    st.image(icon,use_column_width=True)


#MAIN PAGE
def main():
    page = option_menu("SELECT",["ABOUT", "EDA PROCESS","DATA EXPLORE","DASHBORADS"],orientation='horizontal')
  

    if page == "ABOUT":
        home()
    elif page == "EDA PROCESS":
        #page1 =  st.sidebar.selectbox("SELECT",["UNIVARIATE_ANALYSIS", "BIVARIATE_ANALYSIS","MULTIVARAIATE_ANALYSIS"])
       
        page1 =st.radio("SELECT",["UNIVARIATE_ANALYSIS", "BIVARIATE_ANALYSIS","MULTIVARAIATE_ANALYSIS"])
        if page1=="UNIVARIATE_ANALYSIS":
            st.title("UNIVARIATE ANALYSIS")
            univariate(df)
        elif page1=="BIVARIATE_ANALYSIS":
             st.title("BIVARIATE ANALYSIS")
             bivariate(df)
        elif page1=="MULTIVARAIATE_ANALYSIS":
             st.title("MULTIVARIATE ANALYSIS")
             correlation_plot(df)
    elif page == "DASHBORADS":
        ana()
    elif page == "DATA EXPLORE":
        st.markdown("## Explore more about the Airbnb data")
        
        # GETTING USER INPUTS
        country = st.multiselect('Select a Country',sorted(df.Country.unique()),sorted(df.Country.unique()))
        property_type = st.multiselect('Select Property_type',sorted(df.Property_type.unique()),sorted(df.Property_type.unique()))
        room_type = st.multiselect('Select Room_type',sorted(df.Room_type.unique()),sorted(df.Room_type.unique()))
        price = st.slider('Select Price',df.Price.min(),df.Price.max(),(df.Price.min(),df.Price.max()))
        
        # CONVERTING THE USER INPUT INTO QUERY
        query = f'Country in {country} & Room_type in {room_type} & Property_type in {property_type} & Price >= {price[0]} & Price <= {price[1]}'
        
        # HEADING 1
        st.markdown("## Price Analysis")
        
        # CREATING COLUMNS
        col1,col2 = st.columns(2,gap='medium')
        
        with col1:
            
            # AVG PRICE BY ROOM TYPE BARCHART
            pr_df = df.query(query).groupby('Room_type',as_index=False)['Price'].mean().sort_values(by='Price')
            fig = px.bar(data_frame=pr_df,
                        x='Room_type',
                        y='Price',
                        color='Price',
                        title='Avg Price in each Room type'
                        )
            st.plotly_chart(fig,use_container_width=True)
            
            # HEADING 2
            st.markdown("## Availability Analysis")
            
            # AVAILABILITY BY ROOM TYPE BOX PLOT
            fig = px.box(data_frame=df.query(query),
                        x='Room_type',
                        y='Availability_365',
                        color='Room_type',
                        title='Availability by Room_type'
                        )
            st.plotly_chart(fig,use_container_width=True)
            
        with col2:
            
            # AVG PRICE IN COUNTRIES SCATTERGEO
            country_df = df.query(query).groupby('Country',as_index=False)['Price'].mean()
            fig = px.scatter_geo(data_frame=country_df,
                                        locations='Country',
                                        color= 'Price', 
                                        hover_data=['Price'],
                                        locationmode='country names',
                                        size='Price',
                                        title= 'Avg Price in each Country',
                                        color_continuous_scale='agsunset'
                                )
            col2.plotly_chart(fig,use_container_width=True)
            
            # BLANK SPACE
            st.markdown("#   ")
            st.markdown("#   ")
            
            # AVG AVAILABILITY IN COUNTRIES SCATTERGEO
            country_df = df.query(query).groupby('Country',as_index=False)['Availability_365'].mean()
            country_df.Availability_365 = country_df.Availability_365.astype(int)
            fig = px.scatter_geo(data_frame=country_df,
                                        locations='Country',
                                        color= 'Availability_365', 
                                        hover_data=['Availability_365'],
                                        locationmode='country names',
                                        size='Availability_365',
                                        title= 'Avg Availability in each Country',
                                        color_continuous_scale='agsunset'
                                )
            st.plotly_chart(fig,use_container_width=True)
            
            
if __name__ == "__main__":
    main()
