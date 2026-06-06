import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict

TRAIN_FILE = './train_ratings.csv'
TEST_FILE = './test_ratings.csv'
META_FILE = './clean_meta.csv'
TOP_K = 10
RELEVANCE_THRESHOLD = 3.5
DEMO_USER_ID = 22744
COLD_START_MOVIE = 'Inception'
MIN_RATINGS_PER_ITEM = 50
MIN_RATINGS_PER_USER = 10

print("Loading data...")
df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)
meta_df  = pd.read_csv(META_FILE)

meta_dict = dict(zip(meta_df['show_id'], meta_df['title_netflix']))

print(f"   Train: {len(df_train):,} ratings | Test: {len(df_test):,} ratings")

print("Building item-item cosine similarity matrix...")

df_filtered = df_train.copy()
df_filtered = df_filtered[
    df_filtered.groupby('title_netflix')['title_netflix'].transform('count') >= MIN_RATINGS_PER_ITEM
]
df_filtered = df_filtered[
    df_filtered.groupby('userId')['userId'].transform('count') >= MIN_RATINGS_PER_USER
]
print(f"   After filtering: {len(df_filtered):,} ratings | "
      f"{df_filtered['title_netflix'].nunique()} movies | "
      f"{df_filtered['userId'].nunique()} users")

users = df_filtered['userId'].astype('category')
items = df_filtered['title_netflix'].astype('category')

rating_matrix = csr_matrix(
    (df_filtered['rating'], (items.cat.codes, users.cat.codes)),
    shape=(len(items.cat.categories), len(users.cat.categories))
)

print("   Computing cosine similarity...")
cosine_sim   = cosine_similarity(rating_matrix)
movie_titles = items.cat.categories.tolist()
title_to_idx = {title: idx for idx, title in enumerate(movie_titles)}

def get_similar_movies(title, top_n=TOP_K):
    if title not in title_to_idx:
        return []
    idx         = title_to_idx[title]
    sim_indices = np.argsort(cosine_sim[idx])[-(top_n + 1):-1][::-1]
    return [(movie_titles[i], cosine_sim[idx][i]) for i in sim_indices]


def predict_for_user(user_profile, top_n=TOP_K):
    scores = np.zeros(len(movie_titles))
    for title, rating in user_profile.items():
        if title in title_to_idx:
            idx     = title_to_idx[title]
            scores += cosine_sim[idx] * rating

    recommended_indices = np.argsort(scores)[::-1]
    recommendations = []
    for idx in recommended_indices:
        title = movie_titles[idx]
        if title not in user_profile:
            recommendations.append((title, scores[idx]))
        if len(recommendations) == top_n:
            break
    return recommendations


def predict_rating_for_metric(target_title, user_profile):
    if target_title not in title_to_idx:
        if user_profile:
            return float(np.mean(list(user_profile.values())))
        return global_mean

    if not user_profile:
        return global_mean

    target_idx = title_to_idx[target_title]
    numerator = 0.0
    denominator = 0.0

    for watched_title, actual_rating in user_profile.items():
        if watched_title in title_to_idx:
            watched_idx = title_to_idx[watched_title]
            sim_score = cosine_sim[target_idx, watched_idx]
            numerator += sim_score * actual_rating
            denominator += np.abs(sim_score)

    if denominator == 0:
        return float(np.mean(list(user_profile.values())))

    return float(np.clip(numerator / denominator, 1.0, 5.0))

print("Building user profiles from train set...")
user_profiles = defaultdict(dict)
for _, row in df_train.iterrows():
    title = row['title_netflix']
    user_profiles[row['userId']][title] = row['rating']

global_mean = df_train['rating'].mean()

def calculate_ranking_metrics(predictions, k=10, threshold=3.5):

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


print("Predicting on test set (this may take a while)...")
predictions, actuals, preds = [], [], []
total = len(df_test)

for i, (_, row) in enumerate(df_test.iterrows()):
    if i % 50000 == 0:
        print(f"   Progress: {i}/{total}")
    uid    = row['userId']
    title  = row['title_netflix']
    true_r = row['rating']

    est_r = predict_rating_for_metric(title, user_profiles.get(uid, {}))
    predictions.append((uid, title, true_r, est_r))
    actuals.append(true_r)
    preds.append(est_r)

rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
mae = float(mean_absolute_error(actuals, preds))
precision, recall = calculate_ranking_metrics(
    predictions, k=TOP_K, threshold=RELEVANCE_THRESHOLD
)

print(" SIMPLE COLLABORATIVE MODEL: FINAL METRICS")
print(f"RMSE        : {rmse:.4f}")
print(f"MAE         : {mae:.4f}")
print("-" * 42)
print(f"Precision@{TOP_K}: {precision:.4f}")
print(f"Recall@{TOP_K}   : {recall:.4f}")
print("==========================================")

print("\n" + "=" * 62)
print(" RECOMMENDATIONS: SIMPLE COLLABORATIVE MODEL")
print("=" * 62)

print(f"\n[COLD START]")
similar = get_similar_movies(COLD_START_MOVIE, top_n=TOP_K)
if similar:
    print(f"User visited the '{COLD_START_MOVIE}' movie page.")
    print(f"The system recommends similar movies:")
    for i, (title, score) in enumerate(similar, 1):
        print(f"  {i:2d}. {title:<40} | Similarity: {score:.3f}")
else:
    print(f"  '{COLD_START_MOVIE}' not found in filtered dataset.")

print(f"\n[WARM START]")
user_history = df_train[df_train['userId'] == DEMO_USER_ID].sort_values('rating', ascending=False)
print(f"Analyzing the active user profile (ID={DEMO_USER_ID})")
print("Their favorite movies:")

count = 0
for _, row in user_history[user_history['rating'] >= 4.5].iterrows():
    title = row['title_netflix']
    if title in title_to_idx:
        print(f"   {title:<40} | Rating: {row['rating']}")
        count += 1
        if count == 5:
            break

user_profile = user_profiles.get(DEMO_USER_ID, {})
recs = predict_for_user(user_profile, top_n=TOP_K)

print(f"\nThe baseline collaborative model predicts scores for (Top {TOP_K}):")
for i, (title, score) in enumerate(recs, 1):
    print(f"  {i:2d}. {title:<40} | Score: {score:.2f}")
 
print("\n" + "=" * 62)