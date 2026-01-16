import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
tqdm.pandas()

print("Loading data...")
movies_metadata = pd.read_csv("data/movies_metadata.csv")
links_small_raw = pd.read_csv("data/links_small.csv")
print("Data loaded.")
print('-' * 20)
links_small = links_small_raw.loc[links_small_raw['tmdbId'].notnull(), 'tmdbId'].astype('int') 
movies_metadata_small = movies_metadata[ movies_metadata['id'].isin(links_small.astype('str')) ]
movies = movies_metadata_small[['title','popularity','genres','release_date']].copy()

str_genres = movies['genres'].fillna('[]') \
    .progress_apply(literal_eval) \
    .progress_apply(lambda x: sorted(i['name'] for i in x) if isinstance(x, list) else [])
movies['str_genres'] = str_genres \
    .progress_apply(lambda x: " ".join(x) if len(x) > 0 else False)

movies['release_date'] = pd.to_datetime(movies['release_date'])
movies['year'] = movies['release_date'].dt.year
# print(movies.info())
# print(movies.head())
# print(movies.isnull().sum())
# movies1 = movies.dropna()

# movies = movies1.reset_index(drop=True)

# 1. 불리언 타입인 행의 인덱스 추출
bool_idx = movies[movies['str_genres'].apply(lambda x: type(x) == bool)].index
# 2. 해당 인덱스 삭제
movies = movies.drop(bool_idx)

# print(movies.info())

bow_vector = CountVectorizer()
genre_mat = bow_vector.fit_transform(movies['str_genres'])
print("유사도 계산 중...")
similarity_of_genre = cosine_similarity(genre_mat, genre_mat) # data:genre_mat
print("유사도 계산 완료")
print('-' * 20)

sorted_similarity_of_genre = similarity_of_genre.argsort()
sorted_similarity_of_genre =  sorted_similarity_of_genre[:,::-1]

def recommend(title_name, top_k=5):
    movies_of_title = movies[movies['title'] == title_name]
    if movies_of_title.empty:
            print(f"'{title_name}'을(를) 찾을 수 없습니다.")
            return None    
    print(f'{title_name} 의 장르 {movies_of_title["str_genres"].values[0]}')

    movies_of_title_idx = movies_of_title.index.values[0]
    # sorted_similarity_of_genre: 유사도 내림차순으로 정렬된 인덱스 행렬(numpy.ndarray)
    similar_indexes = sorted_similarity_of_genre[movies_of_title_idx,1:top_k*2]
    similar_indexes = similar_indexes.reshape(-1)
    print(f'{title_name} 의 인덱스 {similar_indexes}')

    return movies.iloc[similar_indexes].sort_values(by=['year'],ascending=False)[:top_k]
    # # title_name이 포함된 행의 인덱스 추출
    # idx = .index[0]
    # # 해당 영화와 유사한 영화들의 인덱스 추출
    # sim_scores = list(enumerate(sorted_similarity_of_genre[idx]))
    # # 유사도가 높은 순으로 정렬
    # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # # 가장 유사한 영화들 추출 (top_k개)
    # sim_scores = sim_scores[1:top_k+1]
    # # 인덱스만 추출
    # movie_indices = [i[0] for i in sim_scores]
    # return movies['title'].iloc[movie_indices]

res_movies = recommend("Toy Story", top_k=5)
print(res_movies[['title','year','str_genres','popularity']])

res_movies = recommend("Jumanji", top_k=5)
print(res_movies[['title','year','str_genres','popularity']])