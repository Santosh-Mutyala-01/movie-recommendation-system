import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the movies data (assuming you have the movie data in 'movies.csv' and 'ratings.csv')
movies = pd.read_csv('movies.csv')  # Assuming you saved the 'movies.csv' earlier
ratings = pd.read_csv('ratings.csv')  # Assuming you saved the 'ratings.csv' earlier

# Merge the dataframes to include movie titles in the ratings data
movie_ratings = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')

# Create a pivot table to get a user-item matrix
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values with 0
user_movie_ratings.fillna(0, inplace=True)

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(user_movie_ratings.T)

# Create a dataframe for cosine similarity with movie titles as indices
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)


# Function to get movie recommendations based on a movie title
def get_recommendations(movie_title):
    if movie_title not in cosine_sim_df:
        return []

    # Get similarity scores for the input movie
    sim_scores = cosine_sim_df[movie_title]

    # Sort the movies based on similarity score (descending)
    sorted_sim_scores = sim_scores.sort_values(ascending=False)

    # Return the top 10 recommended movies
    recommendations = sorted_sim_scores.iloc[1:11]  # Exclude the movie itself (at index 0)

    return list(zip(recommendations.index, recommendations.values))
