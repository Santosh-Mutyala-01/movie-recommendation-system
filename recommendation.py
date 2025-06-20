import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')  
ratings = pd.read_csv('ratings.csv')

movie_ratings = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')

user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

user_movie_ratings.fillna(0, inplace=True)

cosine_sim = cosine_similarity(user_movie_ratings.T)

cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)


def get_recommendations(movie_title):
    if movie_title not in cosine_sim_df:
        return []

    sim_scores = cosine_sim_df[movie_title]

    sorted_sim_scores = sim_scores.sort_values(ascending=False)

    recommendations = sorted_sim_scores.iloc[1:11]

    return list(zip(recommendations.index, recommendations.values))
