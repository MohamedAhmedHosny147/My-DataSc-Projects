import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import warnings
warnings.filterwarnings('ignore')

with open('best_rf.pkl', 'rb') as file:
    pipeline = joblib.load(file)

# Set the title and layout of the Streamlit app
st.set_page_config(page_title="Real Estate Analysis", layout="wide")

# Define the pages in the app
PAGES = {
    "Personal Information": "personal_info",
    "Visualizations": "visualizations",
    "Price Prediction": "price_prediction",
}

# Load and preprocess the dataset
df = pd.read_csv(r'C:\Users\OSAMA AHMED\Desktop\The Final Project For Data Science ( USA Real Estate)\realtor-data.zip.csv')

# Create frequency mappings for 'city' and 'state'
city_frequency = df['city'].value_counts().to_dict()
state_frequency = df['state'].value_counts().to_dict()

# Create a selectbox for page navigation
selected_page = st.sidebar.selectbox("Select Page", list(PAGES.keys()))

# Define the page content
if selected_page == "Personal Information":
    st.title("Personal Information")
    st.subheader("Name Information")
    st.write("First Name: Mohamed")
    st.write("Last Name: Hosny")
    
    st.subheader("Contact Information")
    st.write("Email: big1moody@gmail.com")
    
    st.subheader("Brief Description")
    st.write("""I am a data scientist with a strong background in machine learning, data analysis, and model deployment.
                 I have worked on several projects, including real estate price prediction and credit score classification.""")
    
    st.subheader("My Image")
    st.image("my-photo.jpg", use_column_width=True)

elif selected_page == "Visualizations":
    st.title("Data Visualizations")
    df.fillna({
        'price': 0, 'brokered_by': 0, 'bed': 0, 'bath': 0, 'acre_lot': 0, 
        'street': 0, 'zip_code': 0, 'house_size': 0, 'state': 'Unknown', 
        'prev_sold_date': 'Unknown', 'city': 'Unknown', 'status': 'Unknown'
    }, inplace=True)
    
    df = df[['status', 'price', 'city', 'state', 'zip_code', 'house_size', 'prev_sold_date']]
    df['price'] = df['price'].astype(int)
    # df['zip_code'] = df['zip_code'].astype(int)
    df['house_size'] = df['house_size'].astype(int)
    df.sort_values(by='house_size', inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Word Cloud
    st.subheader('Word Cloud for States')
    wordcloud = WordCloud(background_color='white').generate(' '.join(df['state'].astype(str)))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # Visualizations
    st.subheader('Univariate Analysis')
    # Set the figure size for better visualization
    plt.figure(figsize=(35,10))
    sns.countplot(x=df['status']).set(title='Attacks and subtypes')
    st.pyplot(plt)


    # plt.figure(figsize=(15, 7))
    # # A hisogram for price column
    # px.histogram(df, x = 'price', marginal = 'box', color = 'status', title = 'price distribution')
    # st.pyplot(plt)

    plt.figure(figsize=(15, 7))
    # A prcentage pie chart for status column and put prcentage on it
    df["status"].value_counts().plot.pie(autopct = '%1.1f%%' , colors = ['hotpink', 'skyblue'], title = 'sold vs for sale')
    st.pyplot(plt)

    plt.figure(figsize=(15, 7))
    # A bar plot for city column
    df["city"].value_counts().head(30).plot.barh(color = 'mediumvioletred', title = 'Top 30 cities with the most houses for sale')
    st.pyplot(plt)

    plt.figure(figsize=(15, 7))
    # This is the bar plot for which state have most expensive houses 
    df["state"].value_counts().head(20).plot.bar(color = 'lightseagreen', title = 'state distribution')
    st.pyplot(plt)

    st.subheader('Bivariate Analysis')
    plt.figure(figsize=(15, 7))
    # this is the lineplot for status column with price column
    sns.lineplot(data = df, x = 'status', y = 'price')
    st.pyplot(plt)

    plt.figure(figsize=(15, 7))
    # this is the lineplot for city column with price column
    sns.lineplot(data = df[:30], x = 'city', y = 'price')
    st.pyplot(plt)

    plt.figure(figsize=(15, 7))
    # the countplot for status column with state column
    sns.countplot(data= df , y= 'state', hue= 'status')
    st.pyplot(plt)

    plt.figure(figsize=(15, 7))
    # the pairplot for status column with other columns
    sns.pairplot(data= df[:1000] , hue='status')
    st.pyplot(plt)

    st.subheader('Correlation Heatmap')
   # Filter numeric columns only
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Drop columns that are all NaN or that have constant values
    numeric_df = numeric_df.dropna(axis=1, how='all')  # Drop columns with all NaN
    numeric_df = numeric_df.loc[:, (numeric_df != numeric_df.iloc[0]).any()]  # Drop constant columns

    # Calculate correlation
    heatmap = numeric_df.corr()

    # Plot heatmap if the correlation matrix is not empty
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
    st.pyplot(plt)

elif selected_page == "Price Prediction":
    st.title('House Price Prediction')

    # Input fields for prediction
    city = st.selectbox('City', df['city'].unique())
    brokered_by = st.number_input('Brokered By', min_value=0.0, value=0.0)
    house_size = st.number_input('House Size (sq ft)', min_value=0.0, value=0.0)
    bath = st.number_input('Bathrooms', min_value=0.0, value=0.0)
    state = st.selectbox('State', df['state'].unique())

    # Button to make prediction
    if st.button('Predict House Price'):
        # Encode the city and state inputs using frequency mapping
        city_encoded = city_frequency[city]
        state_encoded = state_frequency[state]

        # Prepare input data with encoded features
        input_data = pd.DataFrame({
            'city': [city_encoded],
            'brokered_by': [brokered_by],
            'house_size': [house_size],
            'bath': [bath],
            'state': [state_encoded],
        })

        # Predict the price
        prediction = pipeline.predict(input_data)

        # Show the prediction
        st.write(prediction)


