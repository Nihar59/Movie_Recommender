import numpy as np
import pandas as pd
import ast
import nltk

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


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convert3)


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())  # Converting Overview from string to list

# Removing Spaces from between the words which should be a single tag. For eg: "Sam Worthington" -> "SamWorthington" & "Sam Mendes" -> "SamMendes"
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# New Column combining overview, genres, keywords, cast and crew
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# New data frame with only 3 data columns namely -> movie_id, title, tags
new_df = movies[['movie_id', 'title', 'tags']]

# Converting list of words in tags column to string format
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Transforming all the data in tags column to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    txt = []
    for i in text.split():
        txt.append(ps.stem(i))

    return " ".join(txt)

new_df['tags'] = new_df['tags'].apply(stem)

