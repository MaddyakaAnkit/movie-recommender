import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

GENRE_COLS = [
    'unknown','Action','Adventure','Animation','Children','Comedy',
    'Crime','Documentary','Drama','Fantasy','Film-Noir','Horror',
    'Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'
]

def load_ratings(path='data/raw/ml-100k/u.data'):
    return pd.read_csv(path, sep='\t',
        names=['user_id','item_id','rating','timestamp'])

def load_movies(path='data/raw/ml-100k/u.item'):
    cols = ['movie_id','title','release_date','video_release','imdb_url'] + GENRE_COLS
    df = pd.read_csv(path, sep='|', encoding='latin-1', names=cols)
    df['genres'] = df[GENRE_COLS].apply(
        lambda row: [g for g, v in zip(GENRE_COLS, row) if v == 1], axis=1)
    return df[['movie_id','title','release_date','genres']]

def load_users(path='data/raw/ml-100k/u.user'):
    return pd.read_csv(path, sep='|',
        names=['user_id','age','gender','occupation','zip_code'])

def split_data(ratings, train_size=0.7, val_size=0.1, random_state=42):
    train, temp = train_test_split(ratings, test_size=1-train_size,
                                   random_state=random_state)
    val_ratio = val_size / (1 - train_size)
    val, test = train_test_split(temp, test_size=1-val_ratio,
                                 random_state=random_state)
    Path('data/processed').mkdir(exist_ok=True)
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

def build_user_movie_matrix(ratings):
    return ratings.pivot_table(
        index='user_id', columns='item_id', values='rating')

if __name__ == '__main__':
    ratings = load_ratings()
    movies  = load_movies()
    users   = load_users()
    train, val, test = split_data(ratings)
    print(f"\nUsers:    {ratings['user_id'].nunique()}")
    print(f"Movies:   {ratings['item_id'].nunique()}")
    print(f"Ratings:  {len(ratings)}")
    print(f"Sparsity: {1 - len(ratings)/(943*1682):.3f}")