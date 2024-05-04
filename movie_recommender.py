import numpy as np
import pandas as pd
import ast

movies = pd.read_csv('movie_recommender_system/tmdb_5000_movies.csv')
credits = pd.read_csv('movie_recommender_system/tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.dropna(inplace=True)  # Removing missing data


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
