# Book Recommendation System

A simple **Book Recommendation System** built using Python, NLP, TF-IDF vectorization, and cosine similarity. The system recommends books that are similar to a selected book based on information such as the book description, author, and genres.

## Dataset

Dataset: **Best Books 10K - Multi Genre Data**

Source: Kaggle
https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data

The dataset contains information about books from different genres and is used to build the recommendation system.

---

## Live Demo

Render URL: https://book-recommendation-system-sxna.onrender.com/

GitHub Repository

https://github.com/BinuShaw/Book-Recommendation-System



## Project Objectives

The main objectives of this project are:

* Load and understand the book dataset.
* Preprocess text data using NLP techniques.
* Convert book information into numerical vectors using TF-IDF.
* Calculate similarity between books using cosine similarity.
* Build a recommendation function.
* Create a simple Streamlit web application for book recommendations.

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Joblib
* Pickle
* Streamlit

---

## Project Structure

```text
GenAi_Task20_Binu_Shaw/
│
├── app.py
├── Main.ipynb
├── requirements.txt
├── README.md
├── books.pickle
├── similarities.joblib
└── style.css
```

---

# PART 1 - Data Preprocessing

## Task 1: Load & Understand Dataset

The dataset is loaded using Pandas.

The following details are checked:

* Dataset shape
* Column names
* First 5 rows
* Dataset information
* Missing values
* Data types

The text columns used for recommendations are identified from the dataset.

For this project, book-related information such as:

* Book name
* Author
* Description
* Genres

Combined to create meaningful text for recommendation.

---

## Task 2: Text Preprocessing

The text data is cleaned before applying vectorization.

The following preprocessing steps are performed:

1. Convert text to lowercase.
2. Remove punctuation and special characters.
3. Remove stopwords.
4. Handle missing values by replacing them with an empty string.
5. Store the processed text in a new column named `clean_text`.


NLTK stopwords and regular expressions are used during preprocessing.

---

# PART 2 - Text Vectorization

## Task 3: TF-IDF Vectorization

The cleaned text is converted into numerical vectors using `TfidfVectorizer` from Scikit-learn.

TF-IDF helps identify words that are important to a particular book while reducing the importance of commonly occurring words.

The vectorizer is configured using parameters such as:

* `max_features`
* `ngram_range`

Example:

```python
TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2)
)
```

The shape of the resulting TF-IDF matrix is displayed to understand the number of books and features.

---

## Task 4: Cosine Similarity

Cosine similarity is calculated between all books using the TF-IDF vectors.

```python
cosine_similarity(tfidf_matrix)
```

The resulting similarity matrix is stored for use by the recommendation function.

### Why Cosine Similarity?

Cosine similarity measures how similar two text vectors are based on the angle between them.

It is commonly used for text-based recommendation systems because:

* It works well with TF-IDF vectors.
* It focuses on the direction of vectors rather than their magnitude.
* It is useful for comparing documents with different lengths.
* A higher cosine similarity means the books have more similar content.

The similarity score ranges from:

```text
0 → Not similar
1 → Highly similar
```

---

# PART 3 - Recommendation Logic

## Task 5: Recommendation Function

A recommendation function is created to return the most similar books for a selected book.

The function follows these steps:

1. Find the index of the selected book.
2. Retrieve its similarity scores.
3. Sort the books based on similarity.
4. Remove the selected book itself.
5. Return the top `N` recommendations.

Example:

```python
def recommend(item_name, top_n=5):
    # Find selected book
    # Calculate similarity scores
    # Sort similarity scores
    # Return top N books
```

The recommendation function is tested using at least three different books.

---

# PART 4 - Streamlit Application

## Task 6: User Interface

A simple web interface is created using **Streamlit**.

The application provides:

* A dropdown menu containing available books.
* A button to generate recommendations.
* A list of recommended books.
* Similarity scores for the recommended books.

### Application Workflow

```text
User selects a book
        ↓
Click "Recommend"
        ↓
Find selected book
        ↓
Retrieve similarity scores
        ↓
Sort similar books
        ↓
Display top recommendations
```

---

## Running the Application

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project directory

```bash
cd Book-Recommendation-System
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser.

---

# Project Files

### `app.py`

Contains the Streamlit user interface and recommendation functionality used by the application.

### `recommendation.ipynb`

Contains the data preprocessing, TF-IDF vectorization, cosine similarity calculation, recommendation logic, and testing.

### `books.pickle`

Stores the processed book data used by the application.

### `similarity.pkl`

Stores the precomputed cosine similarity matrix.

### `tfidf_vectorizer.pkl`

Stores the trained TF-IDF vectorizer.

### `requirements.txt`

Contains the Python libraries required to run the project.

---

# Libraries Used

```python
import pandas as pd
import re
import nltk

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import joblib
import pickle
```

The Streamlit application uses:

```python
import streamlit as st
import nltk
import pandas as pd
import sklearn

from pathlib import Path

import joblib
import pickle
```

---

# Future Improvements

The recommendation system can be improved by:

* Using more advanced NLP techniques.
* Adding book ratings and popularity.
* Using semantic embeddings instead of only TF-IDF.
* Adding collaborative filtering.
* Adding book cover images and additional book information.
* Deploying the Streamlit application online.

---

# Conclusion

This project demonstrates how NLP and machine learning techniques can be used to build a simple content-based book recommendation system.

The system preprocesses book-related text, converts it into TF-IDF vectors, calculates cosine similarity, and recommends books with similar content through a Streamlit interface.

---

# Author

**Binu Shaw**

**Course:** Data Science / NLP

**Assignment:** Assignment 20 - Book Recommendation System