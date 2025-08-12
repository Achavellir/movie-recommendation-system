import argparse
import pickle
import pandas as pd
from surprise import Dataset, Reader


def load_model(model_path: str):
    """Load a trained Surprise model from a pickle file."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_top_n_recommendations(model, ratings_df: pd.DataFrame, user_id: int, n: int = 10):
    """Generate top-N movie recommendations for a given user using a trained model.

    :param model: Trained Surprise model
    :param ratings_df: DataFrame with columns userId, movieId, rating
    :param user_id: ID of the user for whom to generate recommendations
    :param n: Number of recommendations to return
    :return: List of (movieId, predicted_rating) tuples
    """
    # Movies the user has already rated
    user_rated = ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
    # All unique movies
    all_movies = ratings_df["movieId"].unique()
    # Candidates are movies not yet rated by the user
    candidates = [m for m in all_movies if m not in user_rated]
    predictions = []
    for movie_id in candidates:
        est = model.predict(user_id, movie_id).est
        predictions.append((movie_id, est))
    # Sort by estimated rating descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


def main():
    parser = argparse.ArgumentParser(description="Generate movie recommendations using a trained SVD model")
    parser.add_argument("--ratings", required=True, help="Path to ratings CSV (userId, movieId, rating)")
    parser.add_argument("--model", required=True, help="Path to trained model pickle file")
    parser.add_argument("--user-id", type=int, required=True, help="User ID for which to generate recommendations")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to return")
    args = parser.parse_args()
    # Load ratings
    df = pd.read_csv(args.ratings)
    # Load model
    model = load_model(args.model)
    # Generate recommendations
    top_recs = get_top_n_recommendations(model, df, args.user_id, args.top_n)
    print(f"Top {args.top_n} recommendations for user {args.user_id}:")
    for movie_id, score in top_recs:
        print(f"Movie ID {movie_id}: predicted rating {score:.2f}")


if __name__ == "__main__":
    main()
