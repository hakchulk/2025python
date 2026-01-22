import pandas as pd
from ast import literal_eval
import numpy as np

movies_metadata = pd.read_csv('./data/movies_metadata.csv')
links_small = pd.read_csv('./data/links_small.csv')
movies_keywords = pd.read_csv('./data/keywords.csv')

links_small = links_small['tmdbId'].dropna().astype(int)
target_ids = links_small.astype(str)
movies_metadata = movies_metadata[movies_metadata['id'].isin(target_ids)]
movies = movies_metadata[['id', 'title', 'genres', 'popularity', 'release_date']]
movies_keywords['id'] = movies_keywords['id'].astype('str')
movies = movies.merge(movies_keywords,on='id')

movies_genres = movies['genres'].apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
movies_genres = movies_genres.apply(lambda x: sorted(x))
movies['genres'] = movies_genres

movies_k = movies['keywords'].apply(literal_eval).apply(lambda x: sorted(i['name'] for i in x) if isinstance(x, list) else [])
movies['keywords'] = movies_k

movies['str_genres_keyword'] = movies['genres'] + movies['keywords']
movies['str_genres_keyword'] = movies['str_genres_keyword'] \
    .apply(lambda x: sorted(x)) \
    .apply(lambda x: ' '.join(x) if len(x) > 0 else None)

movies['release_date'] = pd.to_datetime(movies['release_date'])
movies['year'] = movies['release_date'].dt.year

movies['popularity'] = movies['popularity'].astype('float')
movies['popularity_log'] = np.log( movies['popularity'])

movies = movies.dropna().reset_index(drop=True)
print( movies.info() )
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_mat = tfidf_vectorizer.fit_transform(movies['str_genres_keyword'])
arr_tfidf = tfidf_mat.toarray()

from sklearn.metrics.pairwise import euclidean_distances
similarity_of_euclidean = euclidean_distances(arr_tfidf, arr_tfidf)
sorted_similarity_of_euclidean = np.argsort(similarity_of_euclidean, axis=1) 

def reacomm_of_euclidean(movie_name, top_n=5):
    movie_of_title = movies[movies['title'] == movie_name]
    movie_index = movie_of_title.index.values[0]
    # print(movie_index)
    similarity_indexes = sorted_similarity_of_euclidean[movie_index, 1:top_n * 2]
    similarity_indexes = similarity_indexes.reshape(-1) # 한줄로 바꿈
    # print(similarity_indexes)
    return movies.iloc[similarity_indexes].sort_values(['popularity_log', 'year'], ascending=False)[:top_n]

recomm_movies = reacomm_of_euclidean('Robin Hood')
print(recomm_movies[['title','popularity_log','year']])
# movies.info()