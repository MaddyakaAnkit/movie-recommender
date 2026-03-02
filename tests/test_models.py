import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_ratings():
    return pd.DataFrame({
        'user_id':   [1,1,1,2,2,3,3,3],
        'item_id':   [1,2,3,1,4,2,3,5],
        'rating':    [5,3,4,2,5,4,3,1],
        'timestamp': [0]*8
    })

def test_precision_at_k():
    from src.evaluate import precision_at_k
    recs = [(1,5.0),(2,4.5),(3,4.0),(4,3.5)]
    assert precision_at_k(recs, {1,3}, k=4) == 0.5

def test_recall_at_k():
    from src.evaluate import recall_at_k
    recs = [(1,5.0),(2,4.5)]
    assert recall_at_k(recs, {1,3}, k=2) == 0.5

def test_ndcg_bounded():
    from src.evaluate import ndcg_at_k
    recs = [(1,5.0),(2,4.5),(3,4.0)]
    assert 0 <= ndcg_at_k(recs, {1,3}, k=3) <= 1

def test_rmse_perfect():
    from src.evaluate import rmse
    a = np.array([4,3,5])
    assert rmse(a, a) == 0.0

def test_user_cf_predict_in_range(sample_ratings):
    from src.baseline_model import UserBasedCF
    model = UserBasedCF(n_neighbors=2).fit(sample_ratings)
    pred  = model.predict(1, 2)
    assert 1.0 <= pred <= 5.0

def test_user_cf_cold_start(sample_ratings):
    from src.baseline_model import UserBasedCF
    model = UserBasedCF().fit(sample_ratings)
    pred  = model.predict(999, 1)
    assert 1.0 <= pred <= 5.0

def test_user_cf_excludes_seen(sample_ratings):
    from src.baseline_model import UserBasedCF
    model = UserBasedCF(n_neighbors=2).fit(sample_ratings)
    recs  = model.recommend(1, [1,2,3,4,5], seen_movie_ids={1,2,3})
    returned_ids = [mid for mid, _ in recs]
    assert 1 not in returned_ids
    assert 2 not in returned_ids

def test_svd_predict_in_range(sample_ratings):
    from src.svd_model import SVDRecommender
    model = SVDRecommender(n_factors=2).fit(sample_ratings)
    pred  = model.predict(1, 2)
    assert 1.0 <= pred <= 5.0

def test_svd_cold_start(sample_ratings):
    from src.svd_model import SVDRecommender
    model = SVDRecommender(n_factors=2).fit(sample_ratings)
    pred  = model.predict(999, 1)
    assert 1.0 <= pred <= 5.0