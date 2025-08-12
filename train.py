import argparse
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle


def load_ratings(ratings_path: str):
    """Load ratings CSV into a Surprise Dataset."""
    df = pd.read_csv(ratings_path)
    # Expect columns: userId, movieId, rating
    reader = Reader(rating_scale=(df['rating'].min(), df['rating'].max()))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    return data


def train_svd(data):
    """Train an SVD model on the dataset."""
    trainset, _ = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    return algo


def main():
    parser = argparse.ArgumentParser(description="Train a movie recommendation model using SVD.")
    parser.add_argument("--ratings", required=True, help="Path to MovieLens ratings CSV (with columns userId,movieId,rating)")
    parser.add_argument("--model_path", default="model.pkl", help="Where to save the trained model (pickle file)")
    args = parser.parse_args()

    print("Loading ratings...")
    data = load_ratings(args.ratings)
    print("Training model...")
    model = train_svd(data)
    print(f"Saving model to {args.model_path}...")
    with open(args.model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Done.")


if __name__ == "__main__":
    main()
