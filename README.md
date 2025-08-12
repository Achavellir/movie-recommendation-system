# Movie Recommendation System

This project implements a simple movie recommendation engine using the [MovieLens](https://grouplens.org/datasets/movielens/) ratings dataset. It uses a collaborative filtering approach based on matrix factorization (SVD) to learn latent user and item factors.

## Features

* **Data Loading**: Reads the MovieLens ratings CSV and splits it into train/validation sets.
* **Model**: Trains a matrix factorization model using the Surprise library's SVD algorithm.
* **Persistence**: Saves the trained model to a binary file using pickle.
* **Inference**: Provides a CLI to generate top–N movie recommendations for a given user based on the trained model.

## Requirements

See `requirements.txt` for the list of Python dependencies.  You can install them via:

```bash
pip install -r requirements.txt
```

## Usage

1. **Download the Dataset**: Download the MovieLens ratings data (e.g., `ratings.csv` from the 10M or latest dataset) and place it in a `data/` directory.

2. **Train the Model**:

   ```bash
   python train.py --ratings data/ratings.csv --model_path model.pkl
   ```

   This will load the ratings, train an SVD model and save it to `model.pkl`.

3. **Generate Recommendations**:

   ```bash
   python recommend.py --model_path model.pkl --ratings data/ratings.csv --user_id 1 --top_n 10
   ```

   This will print the top 10 recommended movies for user with ID 1.

## Files

* `train.py` – Script to load data and train the SVD model.
* `recommend.py` – Script to load the trained model and generate recommendations for a user.
* `requirements.txt` – Python dependencies.
* `data/README.md` – Instructions on obtaining the MovieLens dataset.
