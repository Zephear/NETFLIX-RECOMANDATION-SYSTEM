import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict

CONTENT_FILE = './netflix_cleaned.csv'
TRAIN_FILE = './train_ratings.csv'
TEST_FILE = './test_ratings.csv'

TOP_K = 10
RELEVANCE_THRESHOLD = 3.5
ALPHA = 0.6
DEMO_USER_ID = 22744
COLD_START_MOVIE = 'Inception'

print("Loading data...")
df_cb = pd.read_csv(CONTENT_FILE)
df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

df_cb['combined_features'] = df_cb['combined_features'].fillna('')
print(f"   Content: {len(df_cb)} movies | Train: {len(df_train):,} | Test: {len(df_test):,}")

print("Building content-based module...")
known_titles = set(df_train['title_netflix'].unique()) | set(df_test['title_netflix'].unique())
df_cb_known  = df_cb[df_cb['title'].isin(known_titles)].reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_cb_known['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cb_indices = pd.Series(df_cb_known.index, index=df_cb_known['title']).drop_duplicates()

def cb_get_scores(title, top_n=50):
    if title not in cb_indices:
        return {}
    idx = cb_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    return {df_cb_known['title'].iloc[i]: score for i, score in sim_scores}

def cb_predict_rating(target_movie, user_profile):
    if target_movie not in cb_indices:
        return np.nan
    target_idx = cb_indices[target_movie]
    numerator = 0.0
    denominator = 0.0
    for watched_movie, actual_rating in user_profile.items():
        if watched_movie in cb_indices:
            watched_idx  = cb_indices[watched_movie]
            sim          = cosine_sim[target_idx][watched_idx]
            numerator   += sim * actual_rating
            denominator += abs(sim)
    if denominator == 0:
        return float(np.mean(list(user_profile.values())))
    return numerator / denominator

print("Building CF surrogate...")
item_stats = df_train.groupby('title_netflix').agg(item_mean=('rating', 'mean'))
max_mean  = item_stats['item_mean'].max()
cf_score_lookup = (item_stats['item_mean'] / max_mean).to_dict()

def cf_get_score(title):
    return cf_score_lookup.get(title, None)

def cf_predict_rating(target_movie, user_profile):
    cf_score = cf_get_score(target_movie)
    if cf_score is None:
        return np.nan
    user_mean = np.mean(list(user_profile.values()))
    return user_mean * cf_score

def hybrid_recommend(target_movie, user_profile=None, alpha=ALPHA, top_n=TOP_K):
    has_user_history = bool(user_profile)
    if not has_user_history:
        alpha = 1.0  

    cb_scores = cb_get_scores(target_movie, top_n=top_n * 5)
    scored = []
    
    for title, cb_score in cb_scores.items():
        cf_score_raw = cf_get_score(title)
        if cf_score_raw is None:
            effective_alpha = 1.0
            cf_score_norm = 0.0
        else:
            effective_alpha = alpha
            cf_score_norm = cf_score_raw

        hybrid = effective_alpha * cb_score + (1 - effective_alpha) * cf_score_norm
        scored.append((title, hybrid, cb_score, cf_score_norm))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

def hybrid_predict_rating(target_movie, user_profile, alpha=ALPHA):
    cb_pred = cb_predict_rating(target_movie, user_profile)
    cf_pred = cf_predict_rating(target_movie, user_profile)

    if np.isnan(cb_pred) and np.isnan(cf_pred):
        return np.nan
    if np.isnan(cb_pred):
        return cf_pred
    if np.isnan(cf_pred):
        return cb_pred
    return alpha * cb_pred + (1 - alpha) * cf_pred

print("Building user profiles from train set...")
user_profiles = defaultdict(dict)
for _, row in df_train.iterrows():
    user_profiles[row['userId']][row['title_netflix']] = row['rating']

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
    uid = row['userId']
    title = row['title_netflix']
    true_r = row['rating']

    profile = user_profiles.get(uid, {})
    est_r = hybrid_predict_rating(title, profile, alpha=ALPHA)

    if not np.isnan(est_r):
        predictions.append((uid, title, true_r, est_r))
        actuals.append(true_r)
        preds.append(est_r)

rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
mae  = float(mean_absolute_error(actuals, preds))
precision, recall = calculate_ranking_metrics(
    predictions, k=TOP_K, threshold=RELEVANCE_THRESHOLD
)

print(" BASELINE HYBRID MODEL: FINAL METRICS")
print(f"RMSE        : {rmse:.4f}")
print(f"MAE         : {mae:.4f}")
print("-" * 42)
print(f"Precision@{TOP_K}: {precision:.4f}")
print(f"Recall@{TOP_K}   : {recall:.4f}")
print("==========================================")
print("\n" + "=" * 62)
print(" RECOMMENDATIONS: BASELINE HYBRID MODEL")
print("=" * 62)

print(f"\n[COLD START]")
recs_cold = hybrid_recommend(COLD_START_MOVIE, user_profile=None, alpha=ALPHA, top_n=TOP_K)
if recs_cold:
    print(f"New user searches for '{COLD_START_MOVIE}'. The system (Hybrid) recommends:")
    for i, (title, hybrid, cb, cf) in enumerate(recs_cold, 1):
        genres_row = df_cb[df_cb['title'] == title]
        genres = genres_row['genres'].iloc[0] if not genres_row.empty else "Unknown"
        print(f"  {i:2d}. {title:<40} | Hybrid Score: {hybrid:.2f} | Genres: {genres}")
else:
    print(f"  '{COLD_START_MOVIE}' not found in dataset.")

print(f"\n[WARM START]")
user_history = df_train[df_train['userId'] == DEMO_USER_ID].sort_values('rating', ascending=False)
print(f"Analyzing the active user profile (ID={DEMO_USER_ID})")
print("Their favorite movies:")

count = 0
for _, row in user_history[user_history['rating'] >= 4.5].iterrows():
    title = row['title_netflix']
    if title in cb_indices:
        genres_row = df_cb[df_cb['title'] == title]
        genres = genres_row['genres'].iloc[0] if not genres_row.empty else "Unknown"
        print(f"   {title:<40} | Genres: {genres}")
        count += 1
        if count == 5:
            break

user_profile     = user_profiles.get(DEMO_USER_ID, {})
candidate_titles = [t for t in cb_indices.index if t not in user_profile]

print(f"\nThe baseline hybrid model predicts ratings for (Top {TOP_K}):")
user_predictions = []
for title in candidate_titles:
    est_r = hybrid_predict_rating(title, user_profile, alpha=ALPHA)
    if not np.isnan(est_r):
        user_predictions.append((title, est_r))

user_predictions.sort(key=lambda x: x[1], reverse=True)
for i, (title, est_rating) in enumerate(user_predictions[:TOP_K], 1):
    genres_row = df_cb[df_cb['title'] == title]
    genres     = genres_row['genres'].iloc[0] if not genres_row.empty else "Unknown"
    print(f"  {i:2d}. {title:<40} | Prediction: {est_rating:.2f} | Genres: {genres}")

print("\n" + "=" * 62)