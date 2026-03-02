import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def mae(actual, predicted):
    return mean_absolute_error(actual, predicted)

def precision_at_k(recommendations, relevant_items, k=10):
    top_k = [item for item, _ in recommendations[:k]]
    return len(set(top_k) & set(relevant_items)) / k

def recall_at_k(recommendations, relevant_items, k=10):
    if not relevant_items:
        return 0.0
    top_k = [item for item, _ in recommendations[:k]]
    return len(set(top_k) & set(relevant_items)) / len(relevant_items)

def ndcg_at_k(recommendations, relevant_items, k=10):
    top_k = [item for item, _ in recommendations[:k]]
    dcg   = sum(1/np.log2(i+2)
                for i, item in enumerate(top_k) if item in relevant_items)
    idcg  = sum(1/np.log2(i+2) for i in range(min(len(relevant_items), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_rating_prediction(model, test_df, model_name="Model"):
    predictions = model.predict_batch(test_df)
    r = rmse(test_df['rating'].values, predictions)
    m = mae(test_df['rating'].values, predictions)
    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"  RMSE: {r:.4f}  |  MAE: {m:.4f}")
    print(f"{'='*40}")
    return {'rmse': r, 'mae': m}

def evaluate_ranking(model, test_df, train_df, all_movie_ids,
                     k=10, threshold=4.0, n_users=100, model_name="Model"):
    test_users = test_df['user_id'].unique()[:n_users]
    p_scores, r_scores, ndcg_scores = [], [], []

    for user_id in test_users:
        relevant = set(test_df[
            (test_df['user_id']==user_id) &
            (test_df['rating']>=threshold)
        ]['item_id'])
        if not relevant:
            continue
        seen = set(train_df[train_df['user_id']==user_id]['item_id'])
        recs = model.recommend(user_id, all_movie_ids,
                               seen_movie_ids=seen, n=k)
        p_scores.append(precision_at_k(recs, relevant, k))
        r_scores.append(recall_at_k(recs, relevant, k))
        ndcg_scores.append(ndcg_at_k(recs, relevant, k))

    results = {
        f'precision@{k}': np.mean(p_scores),
        f'recall@{k}':    np.mean(r_scores),
        f'ndcg@{k}':      np.mean(ndcg_scores)
    }
    print(f"\n  {model_name} Ranking@{k}")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")
    return results