# Project-4-AIRBNB
AIRBNB ANALYSIS
**MongoDB Connection and Data Retrieval:**
   Established a connection to the MongoDB Atlas cluter database and retrieved the Airbnb dataset and dataset converted into DataFrame
**Data Cleaning Process:**
1.Handled missing values
2.Removed duplicates values
3.Type conversion(some datatype converted to necessary)
**EDA Process:**
1.Univariate Analysis: 
*Categorical analysis of room types, property types, and hosts with listings using count plots.
*Numerical analysis of availability using box plots and Numerical analysis of all columns using histograms and box plots
.
2.Bivariate Analysis: 

*numerical bivariate analysis:Scatter plots of price vs. country, availability vs. country code, review score vs. country code, and pairs plots for numerical variables.
*categorical bivariate analysis :Bar plots for categorical bivariate analysis such as country vs. price, property type vs. price, and host name vs. price.


3.Multivariate Analysis: 
The correlation_plot(df) function generates a correlation heatmap for multivariate analysis, identifying correlations between numerical variables in the DataFrame.

**Data Exploreing:**
 To explore the Airbnb data by selecting countries, property types, room types, and price ranges. It dynamically generates visualizations based on user input, including bar charts and scatter plots using Plotly Express.

    
    


















