import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

TRAIN_FILE = 'train_ratings.csv'
TEST_FILE  = 'test_ratings.csv'
META_FILE  = 'clean_meta.csv'

TOP_K = 10
POPULARITY_ALPHA = 0.3
MMR_LAMBDA = 0.7
MMR_POOL = 100
RELEVANCE_THRESHOLD = 3.5

DEMO_USER_ID = 22744

print("1. Loading data...")
try:
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    meta_df = pd.read_csv(META_FILE)
except FileNotFoundError as e:
    print(f"Error: File not found. ({e})")
    exit()

df_full = pd.concat([df_train, df_test], ignore_index=True)
meta_dict = dict(zip(meta_df['show_id'], meta_df['title_netflix']))

print(f"   Train: {len(df_train):,} rows | {df_train['userId'].nunique():,} users | {df_train['show_id'].nunique():,} movies")

def get_popular_recommendations(df, meta_dict, top_n=TOP_K):
    stats = df.groupby('show_id').agg(
        rating_mean=('rating', 'mean'),
        rating_count=('rating', 'count')
    )

    global_mean = stats['rating_mean'].mean()
    min_votes = stats['rating_count'].quantile(0.7)

    scores = []
    for show_id, row in stats.iterrows():
        votes = row['rating_count']
        rating = row['rating_mean']
        bayesian_score = (votes / (votes + min_votes)) * rating + (min_votes / (min_votes + votes)) * global_mean
        scores.append((show_id, bayesian_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [(meta_dict.get(show_id, str(show_id)), score) for show_id, score in scores[:top_n]]


def mmr_rerank(candidates, lmbda=MMR_LAMBDA, top_n=TOP_K):
    if lmbda >= 1.0 or len(candidates) <= top_n:
        return [(title, score) for _, title, score, _ in candidates[:top_n]]

    raw_scores = np.array([score for _, _, score, _ in candidates])
    min_score, max_score = raw_scores.min(), raw_scores.max()

    selected = []
    selected_vectors = []
    remaining = list(candidates)

    while len(selected) < top_n and remaining:
        best_idx, best_mmr = None, -np.inf

        for idx, (_, _, orig_score, vector_q) in enumerate(remaining):
            norm_score = (orig_score - min_score) / (max_score - min_score + 1e-9)

            max_sim = 0.0
            if selected_vectors:
                max_sim = max([np.dot(vector_q, sel_vec) for sel_vec in selected_vectors])

            mmr = lmbda * norm_score - (1 - lmbda) * max_sim

            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = idx

        chosen = remaining.pop(best_idx)
        selected.append((chosen[1], chosen[2]))
        selected_vectors.append(chosen[3])

    return selected


def get_cf_recommendations(algo, df, meta_dict, user_id, top_n=TOP_K):
    trainset = algo.trainset
    rated_items = set(df[df['userId'] == user_id]['show_id'])

    try:
        inner_uid = trainset.to_inner_uid(user_id)
        p_u, b_u = algo.pu[inner_uid], algo.bu[inner_uid]
    except ValueError:
        return get_popular_recommendations(df, meta_dict, top_n)

    candidates = []
    for sid in df['show_id'].unique():
        if sid in rated_items:
            continue

        try:
            inner_iid = trainset.to_inner_iid(sid)
            q_i = algo.qi[inner_iid]
            score = trainset.global_mean + b_u + POPULARITY_ALPHA * algo.bi[inner_iid] + np.dot(q_i, p_u)

            q_norm = np.linalg.norm(q_i)
            q_normed = q_i / q_norm if q_norm > 0 else q_i
            candidates.append((sid, meta_dict.get(sid, str(sid)), score, q_normed))
        except ValueError:
            continue

    candidates.sort(key=lambda x: x[2], reverse=True)
    return mmr_rerank(candidates[:MMR_POOL], lmbda=MMR_LAMBDA, top_n=top_n)


def calculate_ranking_metrics(predictions, k=TOP_K, threshold=RELEVANCE_THRESHOLD):
    user_est_true = defaultdict(list)
    for uid, title, true_r, est_r in predictions:
        user_est_true[uid].append((est_r, true_r))

    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        hits = sum((true_r >= threshold) for (_, true_r) in top_k)
        relevant_total = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        if relevant_total == 0:
            continue

        precisions.append(hits / k)
        recalls.append(hits / relevant_total)

    return np.mean(precisions), np.mean(recalls)

print("\n2. Training SVD model...")
reader = Reader(rating_scale=(0.5, 5.0))
train_data = Dataset.load_from_df(df_train[['userId', 'show_id', 'rating']], reader)
trainset = train_data.build_full_trainset()

algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42, verbose=False)

print("   Running Cross-Validation...")
cv_results = cross_validate(algo, train_data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
print(f"   RMSE (CV): {np.mean(cv_results['test_rmse']):.4f}")
print(f"   MAE  (CV): {np.mean(cv_results['test_mae']):.4f}")

algo.fit(trainset)

print("\n" + "=" * 60)
print(" PART 1: METRICS EVALUATION ON TEST SET")
print("=" * 60)

predictions_list = []
actuals = []
preds = []

print("Predicting on Test Set...")
for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    uid = row['userId']
    sid = row['show_id']
    true_r = row['rating']
    title = meta_dict.get(sid)

    if title:
        est_r = algo.predict(uid, sid).est
        predictions_list.append((uid, title, true_r, est_r))
        actuals.append(true_r)
        preds.append(est_r)

rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
mae = float(mean_absolute_error(actuals, preds))
precision, recall = calculate_ranking_metrics(predictions_list, k=TOP_K, threshold=RELEVANCE_THRESHOLD)

print("\n--- FINAL METRICS ---")
print(f"RMSE        : {rmse:.4f}")
print(f"MAE         : {mae:.4f}")
print("-" * 24)
print(f"Precision@{TOP_K}: {precision:.4f}")
print(f"Recall@{TOP_K}   : {recall:.4f}")
print("\n" + "=" * 60)
print("RECOMMENDATION DEMONSTRATION")
print("=" * 60)
print("\n[COLD START]")
print("New user with no rating history")

for i, (title, score) in enumerate(get_popular_recommendations(df_full, meta_dict, top_n=TOP_K), 1):
    print(f" {i:2d}. {title:<40} | Bayesian score: {score:.3f}")

print(f"\n[WARM START]")
print(f"Profile of an active user (ID={DEMO_USER_ID})")

user_history = df_train[df_train['userId'] == DEMO_USER_ID].sort_values('rating', ascending=False)
print("Favorite movies:")
top_history = user_history[user_history['rating'] >= 4.5]

for _, row in top_history.head(5).iterrows():
    print(f"  ✓ {meta_dict.get(row['show_id'], 'Unknown'):<35} [Rating: {row['rating']}]")

print(f"\nCollaborative model recommends (Top-{TOP_K}):")
recs = get_cf_recommendations(algo, df_train, meta_dict, DEMO_USER_ID, top_n=TOP_K)

if not recs:
    print(" - No recommendations found.")
else:
    for i, (title, score) in enumerate(recs, 1):
        print(f" {i:2d}. {title:<40} | Prediction: {score:.3f}")

print("\n" + "=" * 60)