import streamlit as st
from movie_recommender import new_df, similarity
import requests


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(new_df.iloc[i[0]].title)

    return recommended_movies


movies_list = new_df['title'].values

st.title("Movie Recommender Project")

selected_movie_name = st.selectbox(
    "List of Movies",
    movies_list
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)

