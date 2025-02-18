# Zomato Restaurant Recommendation System

## Overview

This is a restaurant recommendation system built using **Natural Language Processing (NLP)** and **Cosine Similarity** to recommend similar restaurants based on user reviews. The system processes textual reviews from a dataset and recommends restaurants that are similar to the given input restaurant.

The system uses:
- **TF-IDF Vectorization** to convert text data (reviews) into numerical vectors.
- **Cosine Similarity** to compute the similarity between restaurant reviews.
- **Pandas** for data manipulation and cleaning.

## Features

- **Restaurant Recommendations**: Given a restaurant name, the system recommends the top N similar restaurants based on review content.
- **Text Preprocessing**: Reviews are preprocessed by converting text to lowercase, removing stopwords, and eliminating non-alphabetical characters.
- **Cosine Similarity**: Similarity between restaurants is computed using the cosine similarity metric.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk

You can install the required libraries using `pip`:

```bash
pip install -r requirements.txt
#   A u t o m a t i c _ N u m b e r _ P l a t e _ R e c o g n i t i o n  
 