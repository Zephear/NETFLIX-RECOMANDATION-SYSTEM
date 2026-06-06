import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================


CONTENT_FILE = 'netflix_cleaned.csv'
TRAIN_FILE   = 'train_ratings.csv'
TEST_FILE    = 'test_ratings.csv'

TOP_K              = 10
RELEVANCE_THRESHOLD = 3.5
DEMO_USER_ID       = 22744          # Same user as in final-results.txt
COLD_START_MOVIE   = 'Inception'    # Same movie as in final-results.txt

# ==========================================
# 1. LOADING DATA
# ==========================================
print("1. Loading data...")
df = pd.read_csv(CONTENT_FILE)
df_train = pd.read_csv(TRAIN_FILE)
df_test  = pd.read_csv(TEST_FILE)

# ==========================================
# 2. BUILDING TF-IDF MATRIX
# ==========================================
print("2. Building TF-IDF matrix...")
tfidf = TfidfVectorizer(stop_words='english')
df['combined_features'] = df['combined_features'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

print("   Computing global cosine similarity matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# ==========================================
# 3. BUILDING USER PROFILES FROM TRAIN
# ==========================================
print("3. Building user profiles from train set...")
user_profiles = defaultdict(dict)
for _, row in df_train.iterrows():
    title = row['title_netflix']
    if title in indices:
        user_profiles[row['userId']][title] = row['rating']

global_mean = df_train['rating'].mean()

# ==========================================
# 4. PREDICTION FUNCTIONS
# ==========================================
def predict_rating(target_movie, user_profile):
    if target_movie not in indices or not user_profile:
        return global_mean

    target_idx = indices[target_movie]
    user_mean  = np.mean(list(user_profile.values()))

    numerator, denominator = 0.0, 0.0
    for watched_movie, actual_rating in user_profile.items():
        if watched_movie in indices:
            watched_idx = indices[watched_movie]
            sim_score   = cosine_sim[target_idx, watched_idx]
            numerator   += sim_score * actual_rating
            denominator += np.abs(sim_score)

    if denominator == 0:
        return user_mean

    return float(np.clip(numerator / denominator, 1.0, 5.0))


def calculate_ranking_metrics(predictions, k=10, threshold=3.5):
    """Same function as in the advanced model for fair comparison."""
    user_est_true = defaultdict(list)
    for uid, title, true_r, est_r in predictions:
        user_est_true[uid].append((est_r, true_r))

    precisions, recalls, ndcgs = [], [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        hits           = sum((true_r >= threshold) for (_, true_r) in top_k)
        relevant_total = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        if relevant_total == 0:
            continue

        precisions.append(hits / k)
        recalls.append(hits / relevant_total)

        dcg  = sum((1 / np.log2(i + 2)) for i, (_, true_r) in enumerate(top_k) if true_r >= threshold)
        idcg = sum((1 / np.log2(i + 2)) for i in range(min(relevant_total, k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

# ==========================================
# 5. METRICS ON TEST SET
# ==========================================
print("4. Predicting on test set (this may take a while)...")
predictions, actuals, preds = [], [], []

total = len(df_test)
for i, (_, row) in enumerate(df_test.iterrows()):
    if i % 50000 == 0:
        print(f"   Progress: {i}/{total}")
    uid    = row['userId']
    title  = row['title_netflix']
    true_r = row['rating']

    if title in indices:
        est_r = predict_rating(title, user_profiles.get(uid, {}))
        predictions.append((uid, title, true_r, est_r))
        actuals.append(true_r)
        preds.append(est_r)

rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
mae  = float(mean_absolute_error(actuals, preds))
precision, recall, ndcg = calculate_ranking_metrics(
    predictions, k=TOP_K, threshold=RELEVANCE_THRESHOLD
)

print("\n==========================================")
print(" SIMPLE CONTENT-BASED MODEL: FINAL METRICS")
print("==========================================")
print(f"RMSE        : {rmse:.4f}")
print(f"MAE         : {mae:.4f}")
print("-" * 42)
print(f"Precision@{TOP_K}: {precision:.4f}")
print(f"Recall@{TOP_K}   : {recall:.4f}")
print(f"NDCG@{TOP_K}     : {ndcg:.4f}")
print("==========================================")

# ==========================================
# 6. DEMONSTRATION (same format as final-results.txt)
# ==========================================
print("\n" + "=" * 62)
print(" RECOMMENDATIONS: SIMPLE CONTENT-BASED MODEL")
print("=" * 62)

# --- COLD START ---
print(f"\n[COLD START]")
if COLD_START_MOVIE in indices:
    cold_genres = df[df['title'] == COLD_START_MOVIE]['genres'].iloc[0]
    print(f"User visited the '{COLD_START_MOVIE}' movie page.")
    print(f"Original Genres: {cold_genres}")
    print(f"The system recommends similar movies:")

    cold_idx    = indices[COLD_START_MOVIE]
    sim_scores  = cosine_sim[cold_idx]
    top_indices = sim_scores.argsort()[-(TOP_K + 1):][::-1][1:]

    for i, sim_idx in enumerate(top_indices, 1):
        rec_title = df['title'].iloc[sim_idx]
        genres    = df['genres'].iloc[sim_idx]
        score     = sim_scores[sim_idx]
        print(f"  {i:2d}. {rec_title:<40} | Similarity: {score:.3f} | Genres: {genres}")
else:
    print(f"  '{COLD_START_MOVIE}' not found in dataset.")

# --- WARM START ---
print(f"\n[WARM START]")
user_history = df_train[df_train['userId'] == DEMO_USER_ID].sort_values('rating', ascending=False)
print(f"Analyzing the active user profile (ID={DEMO_USER_ID})")
print("Their favorite movies (Rating 4.5+):")

top_history = user_history[user_history['rating'] >= 4.5]
count = 0
for _, row in top_history.iterrows():
    title = row['title_netflix']
    if title in indices:
        genres = df[df['title'] == title]['genres'].iloc[0]
        print(f"   {title:<40} | Genres: {genres}")
        count += 1
        if count == 5:
            break

user_profile  = user_profiles.get(DEMO_USER_ID, {})
candidates    = [m for m in indices.keys() if m not in user_profile]

print(f"\nThe simple content-based model predicts the highest ratings for (Top {TOP_K}):")
user_predictions = []
for movie in candidates:
    est_r = predict_rating(movie, user_profile)
    user_predictions.append((movie, est_r))

user_predictions.sort(key=lambda x: x[1], reverse=True)

for i, (rec_title, est_rating) in enumerate(user_predictions[:TOP_K], 1):
    genres = df[df['title'] == rec_title]['genres'].iloc[0]
    print(f"  {i:2d}. {rec_title:<40} | Prediction: {est_rating:.2f} | Genres: {genres}")

print("\n" + "=" * 62)