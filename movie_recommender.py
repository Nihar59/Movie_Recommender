import numpy as np
import pandas as pd

movies = pd.read_csv('movie_recommender_system/tmdb_5000_movies.csv')
credits = pd.read_csv('movie_recommender_system/tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]