import streamlit as st
import nltk
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path

import joblib
import pickle

# ------ Config ------

st.set_page_config(
    page_title= "Book Recommendation System",
    page_icon="📚",
    layout="centered")

# ------ Css Load ------

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "style.css", 'r') as f:
    css = f.read()
    
st.markdown(f"<style>{css}<style>", unsafe_allow_html=True)

# ------ Header ------

st.markdown(
    '<div class="title">📚 Book Recommendation System </div>',
    unsafe_allow_html=True
    )

st.markdown(
    '<div class="subtitle">Discover books similar to your favorite reads</div>',
    unsafe_allow_html=True
    )

# ------ Main ------ 

with open("books.pickle", "rb") as b:
    books = pickle.load(b)
    
similarities = joblib.load("similarities.joblib", "rb")
tf_idf_matrix = joblib.load("tf_idf_matrix.joblib", "rb")

book_name = books['Book'].values

def recommend(item_name, top_n=5):
    
    # 1. Find index of the selected item
    book_index = books[books['Book'] == item_name].index[0]
    
    # 2. Compute similarity scores
    similarity_score = cosine_similarity(
        tf_idf_matrix[book_index], 
        tf_idf_matrix
    ).flatten()
    
    # 3. Sort
    book_list = sorted(enumerate(similarity_score), reverse=True, key= lambda x : x[1])[1:6]
    
    result = []
    for i,score in book_list:
        result.append((books.iloc[i]['Book'], score))

    # return top recommendations
    return result

# ------ Buttons ------ 

st.markdown(
    '<div class="Selection Button">📖 Select a Book</div>',
    unsafe_allow_html=True
    )


book = st.selectbox(
    "Enter A Book Name",
    book_name,
    format_func=lambda x: x.title(),
    label_visibility="collapsed"
)

recommend_button = st.button(
    "🔍 Recommend Books",
    use_container_width=True
)

# ------ Recommendation code ------

if recommend_button:
    
    recommended_book = recommend(book)
    
    st.markdown(
        '<div class="section-title">✨ Recommended For You</div>',
        unsafe_allow_html=True
    )
    
    for i,(book, score) in enumerate(recommended_book):
        percentage = score * 100
        
        if i == 0:
            rank = "🥇"
        elif i == 1:
            rank = "🥈"
        elif i == 2:
            rank = "🥉"
        else:
            rank = f"#{i + 1}"
            
        st.html(
            f"""
            <div class="book-card">
            
                <div class="book-title">
                    {rank}&nbsp;&nbsp; {book.title()}
                </div>
                
                <div class="similarity">
                    similarity: <b>{percentage:.2f}%</b>
                </div>
            
        </div> 
        """
            
        )
        st.progress(min(score,1.0))
        
# ------ Footer ------

st.markdown(
    '<div class="footer", Built By Binu Shaw • Streamlit • Machine Learning</div>',
    unsafe_allow_html=True
)