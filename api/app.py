from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and data at startup
SVD_MODEL  = joblib.load('models/svd_model.pkl')
RATINGS    = pd.read_csv('data/processed/train.csv')
ALL_MOVIES = RATINGS['item_id'].unique().tolist()
MOVIE_META = pd.read_csv(
    'data/raw/ml-100k/u.item', sep='|', encoding='latin-1',
    header=None, usecols=[0, 1]
).set_index(0)[1].to_dict()


def get_seen(user_id):
    return set(RATINGS[RATINGS['user_id'] == user_id]['item_id'].tolist())


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/recommend/user/<int:user_id>')
def recommend(user_id):
    n    = min(request.args.get('n', 10, type=int), 50)
    seen = get_seen(user_id)
    recs = SVD_MODEL.recommend(user_id, ALL_MOVIES,
                               seen_movie_ids=seen, n=n)
    return jsonify({
        'user_id': user_id,
        'recommendations': [
            {'movie_id': mid,
             'predicted_rating': round(score, 3),
             'title': MOVIE_META.get(mid, 'Unknown')}
            for mid, score in recs
        ]
    })


@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    uid  = body.get('user_id')
    mid  = body.get('movie_id')
    if not uid or not mid:
        return jsonify({'error': 'user_id and movie_id required'}), 400
    return jsonify({
        'user_id':          uid,
        'movie_id':         mid,
        'title':            MOVIE_META.get(mid, 'Unknown'),
        'predicted_rating': round(SVD_MODEL.predict(uid, mid), 3)
    })


@app.route('/movies/popular')
def popular():
    n = request.args.get('n', 10, type=int)
    top = (
        RATINGS.groupby('item_id')
        .agg(avg_rating=('rating','mean'), count=('rating','count'))
        .query('count >= 50')
        .sort_values('avg_rating', ascending=False)
        .head(n).reset_index()
    )
    return jsonify({'popular_movies': [
        {'movie_id':    int(r['item_id']),
         'title':       MOVIE_META.get(int(r['item_id']), 'Unknown'),
         'avg_rating':  round(r['avg_rating'], 2),
         'num_ratings': int(r['count'])}
        for _, r in top.iterrows()
    ]})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)