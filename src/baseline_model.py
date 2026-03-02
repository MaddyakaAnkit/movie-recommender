import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path

class UserBasedCF:
    """User-Based Collaborative Filtering with mean-adjusted predictions."""

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.user_movie_matrix = None
        self.similarity_matrix = None
        self.global_mean = None
        self.user_means = None

    def fit(self, train_df):
        self.user_movie_matrix = train_df.pivot_table(
            index='user_id', columns='item_id', values='rating')
        self.global_mean = train_df['rating'].mean()
        self.user_means  = self.user_movie_matrix.mean(axis=1)
        matrix_filled = self.user_movie_matrix.T.fillna(self.user_means).T
        self.similarity_matrix = pd.DataFrame(
            cosine_similarity(matrix_filled),
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index)
        return self

    def predict(self, user_id, item_id):
        if user_id not in self.user_movie_matrix.index:
            return self.global_mean
        if item_id not in self.user_movie_matrix.columns:
            return self.global_mean
        item_col = self.user_movie_matrix[item_id].dropna()
        item_col = item_col[item_col.index != user_id]
        if len(item_col) == 0:
            return float(self.user_means.get(user_id, self.global_mean))
        sims     = self.similarity_matrix.loc[user_id, item_col.index]
        top_sims = sims.nlargest(self.n_neighbors)
        if top_sims.sum() == 0:
            return float(self.user_means.get(user_id, self.global_mean))
        user_mean        = self.user_means.get(user_id, self.global_mean)
        neighbor_means   = self.user_means[top_sims.index]
        neighbor_ratings = item_col[top_sims.index]
        numerator   = (top_sims * (neighbor_ratings - neighbor_means)).sum()
        denominator = top_sims.abs().sum()
        return float(np.clip(user_mean + numerator / denominator, 1, 5))

    def predict_batch(self, df):
        return df.apply(
            lambda row: self.predict(row['user_id'], row['item_id']), axis=1)

    def recommend(self, user_id, all_movie_ids, seen_movie_ids=None, n=10):
        if user_id not in self.user_movie_matrix.index:
            return []
        seen = seen_movie_ids or set()
        candidates = [m for m in all_movie_ids if m not in seen]
        preds = [(m, self.predict(user_id, m)) for m in candidates]
        return sorted(preds, key=lambda x: x[1], reverse=True)[:n]

    def save(self, path='models/user_cf.pkl'):
        Path(path).parent.mkdir(exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path='models/user_cf.pkl'):
        return joblib.load(path)