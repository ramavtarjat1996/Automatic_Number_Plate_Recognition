# zomato_recommendation_system.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv('data/zomato.csv')  # Replace 'zomato.csv' with the path to your dataset

# Preprocessing function to clean the text
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords and non-alphabetical characters
    text = ' '.join([word for word in text.split() if word.isalpha() and word not in stop_words])
    return text

# Apply preprocessing to the 'Reviews' column
df['cleaned_reviews'] = df['Reviews'].apply(preprocess_text)

# Create a TF-IDF Vectorizer to convert reviews to numerical data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the cleaned reviews
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_reviews'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a DataFrame with restaurant names as indices and columns
cosine_sim_df = pd.DataFrame(cosine_sim, index=df['Restaurant Name'], columns=df['Restaurant Name'])

def recommend_restaurants(restaurant_name, cosine_sim_df, df, top_n=5):
    # Check if the restaurant exists in the dataset
    if restaurant_name not in cosine_sim_df.index:
        return f"Sorry, '{restaurant_name}' not found in the dataset."

    # Get similar restaurants based on cosine similarity
    sim_scores = cosine_sim_df[restaurant_name]
    
    # Sort the scores in descending order and exclude the input restaurant
    sim_scores = sim_scores.sort_values(ascending=False)
    sim_scores = sim_scores[sim_scores.index != restaurant_name]
    
    # Get the top N similar restaurants
    top_restaurants = sim_scores.head(top_n).index
    
    # Get the ratings of the top restaurants
    top_ratings = df[df['Restaurant Name'].isin(top_restaurants)][['Restaurant Name', 'Rating']]
    
    return top_ratings

# Example usage
restaurant_name = "The Biryani Place"  # Replace with any restaurant name
recommended = recommend_restaurants(restaurant_name, cosine_sim_df, df)
print(recommended)
