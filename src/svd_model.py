import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import joblib
from pathlib import Path

class SVDRecommender:
    """
    Matrix Factorization via truncated SVD using scipy.
    Decomposes the mean-centered user-item matrix into latent factors.
    """

    def __init__(self, n_factors=100):
        self.n_factors = n_factors
        self.predicted_matrix = None
        self.user_index  = None
        self.item_index  = None
        self.user_ids    = None
        self.item_ids    = None
        self.user_mean   = None
        self.global_mean = None

    def fit(self, train_df):
        self.global_mean = train_df['rating'].mean()

        self.user_ids = sorted(train_df['user_id'].unique())
        self.item_ids = sorted(train_df['item_id'].unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {m: i for i, m in enumerate(self.item_ids)}

        # Mean-center per user
        user_means = train_df.groupby('user_id')['rating'].mean()
        self.user_mean = user_means

        df = train_df.copy()
        df['rating_centered'] = df['rating'] - df['user_id'].map(user_means)

        # Build sparse matrix
        rows = df['user_id'].map(self.user_index)
        cols = df['item_id'].map(self.item_index)
        n_users, n_items = len(self.user_ids), len(self.item_ids)
        matrix = csr_matrix(
            (df['rating_centered'].values, (rows, cols)),
            shape=(n_users, n_items))

        # Truncated SVD
        k = min(self.n_factors, min(n_users, n_items) - 1)
        U, sigma, Vt = svds(matrix, k=k)
        idx = np.argsort(sigma)[::-1]
        U, sigma, Vt = U[:, idx], sigma[idx], Vt[idx, :]

        # Reconstruct full matrix and add back user means
        self.predicted_matrix = np.clip(
            U @ np.diag(sigma) @ Vt
            + user_means.reindex(self.user_ids).values.reshape(-1, 1),
            1, 5)
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user_index or item_id not in self.item_index:
            return float(self.user_mean.get(user_id, self.global_mean))
        u = self.user_index[user_id]
        i = self.item_index[item_id]
        return float(self.predicted_matrix[u, i])

    def predict_batch(self, df):
        return df.apply(
            lambda row: self.predict(row['user_id'], row['item_id']), axis=1)

    def recommend(self, user_id, all_movie_ids, seen_movie_ids=None, n=10):
        seen = seen_movie_ids or set()
        candidates = [m for m in all_movie_ids if m not in seen]
        preds = [(m, self.predict(user_id, m)) for m in candidates]
        return sorted(preds, key=lambda x: x[1], reverse=True)[:n]

    def save(self, path='models/svd_model.pkl'):
        Path(path).parent.mkdir(exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path='models/svd_model.pkl'):
        return joblib.load(path)