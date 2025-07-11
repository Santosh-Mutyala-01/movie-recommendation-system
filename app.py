from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')


movie_titles = movies['title'].tolist()


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []

    if request.method == "POST":
        selected_movie_title = request.form["movie_title"]

        movie_id = movies[movies['title'] == selected_movie_title]['movieId'].values[0]

        movie_index = movies[movies['movieId'] == movie_id].index[0]

        similarity_matrix = cosine_similarity(
            ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0))

        similarity_scores = list(enumerate(similarity_matrix[movie_index]))

        sorted_similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]

        recommendations = [(movies.iloc[i[0]]['title'], i[1]) for i in sorted_similar_movies]

    return render_template("index.html", movie_titles=movie_titles, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
