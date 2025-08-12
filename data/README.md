# Data Instructions

This project does not include the raw MovieLens ratings dataset due to size and licensing constraints. To train and evaluate the recommendation model you will need to download the MovieLens dataset yourself.

1. Visit the MovieLens website at https://grouplens.org/datasets/movielens/ and download the **MovieLens 10M** dataset (or a smaller variant such as MovieLens 100K for quick experimentation).
2. Extract the archive and locate the `ratings.csv` file. Rename or place the file at a known location.
3. Use the `--ratings` argument when running `train.py` or `recommend.py` to specify the path to your ratings CSV.

The ratings CSV must contain columns `userId`, `movieId`, and `rating`. Additional columns (e.g., timestamps) are ignored by the scripts.
