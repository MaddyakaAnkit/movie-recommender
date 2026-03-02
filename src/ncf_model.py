import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import joblib


class NeuMF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=32, dropout=0.3):
        super().__init__()
        self.gmf_user  = nn.Embedding(num_users+1, embedding_size)
        self.gmf_movie = nn.Embedding(num_movies+1, embedding_size)
        self.mlp_user  = nn.Embedding(num_users+1, embedding_size)
        self.mlp_movie = nn.Embedding(num_movies+1, embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size*2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),              nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,  32),              nn.ReLU()
        )
        self.output = nn.Linear(embedding_size + 32, 1)

    def forward(self, user, movie):
        gmf = self.gmf_user(user) * self.gmf_movie(movie)
        mlp = self.mlp(torch.cat([self.mlp_user(user),
                                   self.mlp_movie(movie)], dim=1))
        return self.output(torch.cat([gmf, mlp], dim=1)).squeeze()


class NCFRecommender:
    def __init__(self, num_users=943, num_movies=1682,
                 embedding_size=32, dropout_rate=0.3):
        self.num_users  = num_users
        self.num_movies = num_movies
        self.device = torch.device('cpu')
        self.model  = NeuMF(num_users, num_movies,
                            embedding_size, dropout_rate).to(self.device)
        self.history = {'loss': [], 'val_loss': []}

    def fit(self, train_df, val_df, epochs=20, batch_size=256):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        def make_loader(df):
            u = torch.tensor(df['user_id'].values, dtype=torch.long)
            m = torch.tensor(df['item_id'].values, dtype=torch.long)
            r = torch.tensor(df['rating'].values,  dtype=torch.float)
            return DataLoader(TensorDataset(u, m, r),
                              batch_size=batch_size, shuffle=True)

        train_loader = make_loader(train_df)
        val_loader   = make_loader(val_df)
        best_val, patience, wait = float('inf'), 5, 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for u, m, r in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(u, m), r)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for u, m, r in val_loader:
                    val_loss += criterion(self.model(u, m), r).item()
            val_loss /= len(val_loader)

            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} — "
                  f"loss: {train_loss:.4f} — val_loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                torch.save(self.model.state_dict(), 'models/ncf_best.pt')
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping!")
                    break

        self.model.load_state_dict(torch.load('models/ncf_best.pt',
                                               weights_only=True))
        return self

    def predict(self, user_id, item_id):
        self.model.eval()
        with torch.no_grad():
            u = torch.tensor([user_id], dtype=torch.long)
            m = torch.tensor([item_id], dtype=torch.long)
        return float(np.clip(self.model(u, m).item(), 1, 5))

    def predict_batch(self, df):
        self.model.eval()
        with torch.no_grad():
            u = torch.tensor(df['user_id'].values, dtype=torch.long)
            m = torch.tensor(df['item_id'].values, dtype=torch.long)
            preds = self.model(u, m).numpy()
        return np.clip(preds, 1, 5)

    def recommend(self, user_id, all_movie_ids, seen_movie_ids=None, n=10):
        seen = seen_movie_ids or set()
        candidates = [m for m in all_movie_ids if m not in seen]
        self.model.eval()
        with torch.no_grad():
            u = torch.tensor([user_id]*len(candidates), dtype=torch.long)
            m = torch.tensor(candidates, dtype=torch.long)
            preds = self.model(u, m).numpy()
        top_idx = np.argsort(preds)[::-1][:n]
        return [(candidates[i], float(preds[i])) for i in top_idx]

    def save(self, path='models/ncf_model'):
        Path(path).parent.mkdir(exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path='models/ncf_model'):
        return joblib.load(path)