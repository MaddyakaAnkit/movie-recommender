"""
Run with:  python -m src.train_all
MLflow UI: mlflow ui --port 5002
"""
import time
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader    import load_ratings, split_data
from src.baseline_model import UserBasedCF
from src.svd_model      import SVDRecommender
from src.ncf_model      import NCFRecommender
from src.evaluate       import evaluate_rating_prediction, evaluate_ranking

mlflow.set_experiment("movie-recommender")
Path("models").mkdir(exist_ok=True)

def run_experiment(name, model, train, val, test, all_movies, params):
    print(f"\n{'='*50}")
    print(f"  Training: {name}")
    print(f"{'='*50}")

    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        t0 = time.time()

        if name == "NCF":
            model.fit(train, val)
            for ep, (tl, vl) in enumerate(zip(
                model.history['loss'],
                model.history['val_loss'])):
                mlflow.log_metric("train_loss", tl, step=ep)
                mlflow.log_metric("val_loss",   vl, step=ep)
        else:
            model.fit(train)

        train_time = time.time() - t0

        rating  = evaluate_rating_prediction(model, test, name)
        ranking = evaluate_ranking(
            model, test, train, all_movies,
            model_name=name, n_users=100)

        mlflow.log_metric("rmse", rating['rmse'])
        mlflow.log_metric("mae",  rating['mae'])
        mlflow.log_metric("training_time_sec", train_time)
        for k, v in ranking.items():
            mlflow.log_metric(k.replace('@', '_at_'), v)

    return rating, ranking

def plot_comparison(results):
    models = list(results.keys())
    colors = ['#4C72B0', '#55A868', '#C44E52']
    metrics = [
        ([results[m]['rating']['rmse'] for m in models], 'RMSE (lower is better)'),
        ([results[m]['rating']['mae']  for m in models], 'MAE (lower is better)'),
        ([results[m]['ranking'].get('precision@10', 0) for m in models],
         'Precision@10 (higher is better)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (vals, title) in zip(axes, metrics):
        bars = ax.bar(models, vals, color=colors, edgecolor='white', width=0.5)
        ax.set_title(title, fontsize=12)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('models/comparison_chart.png', dpi=150)
    print("\n📊 Chart saved to models/comparison_chart.png")

if __name__ == '__main__':
    ratings    = load_ratings()
    train, val, test = split_data(ratings)
    all_movies = ratings['item_id'].unique().tolist()

    results = {}

    r, rk = run_experiment(
        "UserCF", UserBasedCF(n_neighbors=20),
        train, val, test, all_movies, {"n_neighbors": 20})
    results["UserCF"] = {"rating": r, "ranking": rk}

    r, rk = run_experiment(
        "SVD", SVDRecommender(n_factors=100),
        train, val, test, all_movies, {"n_factors": 100})
    results["SVD"] = {"rating": r, "ranking": rk}

    r, rk = run_experiment(
        "NCF", NCFRecommender(),
        train, val, test, all_movies,
        {"embedding_size": 32, "epochs": 30})
    results["NCF"] = {"rating": r, "ranking": rk}

    plot_comparison(results)
    print("\n✅ All done! Run: mlflow ui --port 5002")