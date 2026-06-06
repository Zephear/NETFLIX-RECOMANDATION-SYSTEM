import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import hstack
from tqdm import tqdm

CONTENT_FILE = 'netflix_cleaned.csv'
TRAIN_FILE = 'train_ratings.csv'
TEST_FILE = 'test_ratings.csv'
META_FILE = 'clean_meta.csv'
TOP_K = 10
RELEVANCE_THRESHOLD = 3.5

print("1. Loading data...")
try:
    df_content = pd.read_csv(CONTENT_FILE)
    df_train = pd.read_csv(TRAIN_FILE)
    df_test = pd.read_csv(TEST_FILE)
    meta_df = pd.read_csv(META_FILE)
except FileNotFoundError as e:
    print(f"Error: File not found. ({e})")
    exit()

meta_dict = dict(zip(meta_df['show_id'], meta_df['title_netflix']))

# FEATURE PROCESSING
print("2. Processing content and building vectors...")
df_content['genres'] = df_content['genres'].fillna('')
df_content['description'] = df_content['description'].fillna('')
df_content['cast'] = df_content['cast'].fillna('')
df_content['director'] = df_content['director'].fillna('')

def clean_data(x):
    return str.lower(x.replace(" ", "").replace(",", " "))

df_content['genres_clean'] = df_content['genres'].apply(clean_data)
df_content['cast_clean'] = df_content['cast'].apply(clean_data)
df_content['director_clean'] = df_content['director'].apply(clean_data)

count = CountVectorizer(stop_words='english')
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)

mat_genres = count.fit_transform(df_content['genres_clean'])
mat_cast = count.fit_transform(df_content['cast_clean'])
mat_dir = count.fit_transform(df_content['director_clean'])
mat_desc = tfidf.fit_transform(df_content['description'])

WEIGHT_GENRES = 0.5
WEIGHT_DESC = 0.2
WEIGHT_CAST = 0.2
WEIGHT_DIR = 0.1

combined_matrix = hstack([
    mat_genres * np.sqrt(WEIGHT_GENRES),
    mat_desc * np.sqrt(WEIGHT_DESC),
    mat_cast * np.sqrt(WEIGHT_CAST),
    mat_dir * np.sqrt(WEIGHT_DIR),
])

indices = pd.Series(df_content.index, index=df_content['title']).drop_duplicates()

print("   Calculating global similarity matrix...")
sim_matrix = cosine_similarity(combined_matrix, combined_matrix)

# BUILDING USER PROFILES
print("3. Building user profiles...")
user_profiles = defaultdict(dict)
global_mean = df_train['rating'].mean()

for _, row in df_train.iterrows():
    title = meta_dict.get(row['show_id'])
    if title and title in indices:
        user_profiles[row['userId']][title] = row['rating']

# FUNCTIONS
def predict_rating_fast(target_movie, user_profile):
    if target_movie not in indices or not user_profile:
        return global_mean

    target_idx = indices[target_movie]
    user_mean = np.mean(list(user_profile.values()))

    numerator, denominator = 0.0, 0.0

    for watched_movie, actual_rating in user_profile.items():
        if watched_movie in indices:
            watched_idx = indices[watched_movie]
            sim_score = sim_matrix[target_idx, watched_idx]

            numerator += sim_score * (actual_rating - user_mean)
            denominator += np.abs(sim_score)

    if denominator == 0:
        return user_mean

    return float(np.clip(user_mean + (numerator / denominator), 1.0, 5.0))

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

# METRICS EVALUATION
print("\n" + "=" * 60)
print("METRICS EVALUATION ON TEST SET")
print("=" * 60)

predictions_list = []
actuals = []
preds = []

print("Predicting on Test Set...")
for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    uid = row['userId']
    title = meta_dict.get(row['show_id'])
    true_r = row['rating']

    if title:
        est_r = predict_rating_fast(title, user_profiles.get(uid, {}))
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

print("\n[COLD START (ITEM-BASED)]")
test_titles = ['Inception', 'Interstellar', 'The Dark Knight', 'Spider-Man', 'The Avengers']
target_movie = next((t for t in test_titles if t in indices), None)

if not target_movie:
    action_movies = df_content[df_content['genres'].str.contains('Action', case=False)]
    target_movie = action_movies.iloc[0]['title']

print(f"User opened the movie page for '{target_movie}'.")
print(f"Original genres: {df_content[df_content['title'] == target_movie]['genres'].iloc[0]}")
print("System recommends similar movies:")

idx = indices[target_movie]
sim_scores = sim_matrix[idx]
similar_indices = sim_scores.argsort()[-(TOP_K + 1):][::-1][1:]

for i, sim_idx in enumerate(similar_indices, 1):
    rec_title = df_content['title'].iloc[sim_idx]
    score = sim_scores[sim_idx]
    genres = df_content['genres'].iloc[sim_idx]
    print(f" {i:2d}. {rec_title:<35} | Similarity: {score:.3f} | Genres: {genres}")

print(f"\n[WARM START]")
active_users = df_train['userId'].value_counts()
DEMO_USER_ID = active_users.index[0]
user_history = df_train[df_train['userId'] == DEMO_USER_ID].sort_values('rating', ascending=False)

print(f"Analyzing profile of an active user (ID={DEMO_USER_ID})")
print("Favorite movies:")

top_history = user_history[user_history['rating'] >= 4.5]
count = 0
for _, row in top_history.iterrows():
    title = meta_dict.get(row['show_id'])
    if title in indices:
        print(f"  ✓ {title:<35} [Rating: {row['rating']}]")
        count += 1
        if count == 5:
            break

print(f"\nContent-based model predicts ratings for (Top-{TOP_K}):")
user_profile = user_profiles.get(DEMO_USER_ID, {})
candidate_movies = [m for m in indices.keys() if m not in user_profile]

user_predictions = []
for movie in candidate_movies:
    est_r = predict_rating_fast(movie, user_profile)
    user_predictions.append((movie, est_r))

user_predictions.sort(key=lambda x: x[1], reverse=True)

for i, (rec_title, est_rating) in enumerate(user_predictions[:TOP_K], 1):
    genres = df_content[df_content['title'] == rec_title]['genres'].iloc[0]
    print(f" {i:2d}. {rec_title:<35} | Prediction: {est_rating:.2f} | Genres: {genres}")

print("\n" + "=" * 60)