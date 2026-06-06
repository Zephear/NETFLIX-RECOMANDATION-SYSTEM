import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from surprise import SVD, Dataset, Reader
import xgboost as xgb
from tqdm import tqdm 

# CONFIGURATION
CONTENT_FILE = 'netflix_cleaned.csv'
TRAIN_FILE = 'train_ratings.csv'
TEST_FILE = 'test_ratings.csv'
META_FILE = 'clean_meta.csv'
TOP_K = 10
RELEVANCE_THRESHOLD = 3.5
FEATURES = ['content_score_norm', 'svd_score_norm', 'item_count_norm', 'item_mean_norm']

COLD_CONTENT_WEIGHT = 0.7
COLD_CANDIDATE_POOL = 200
DEMO_USER_ID = 22744

# CONTENT MODEL
def build_content_model(content_file):
    print("[Content] Building content model...")
    df = pd.read_csv(content_file)
    for col in ['genres', 'description', 'cast', 'director']:
        df[col] = df[col].fillna('')

    def clean(x):
        return x.lower().replace(",", " ").strip()

    count_genres = CountVectorizer(stop_words='english')
    count_cast = CountVectorizer(stop_words='english')
    count_dir = CountVectorizer(stop_words='english')
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)

    mat_genres = count_genres.fit_transform(df['genres'].apply(clean))
    mat_cast = count_cast.fit_transform(df['cast'].apply(clean))
    mat_dir = count_dir.fit_transform(df['director'].apply(clean))
    mat_desc = tfidf.fit_transform(df['description'])

    combined = hstack([
        mat_genres * np.sqrt(0.5),
        mat_desc * np.sqrt(0.2),
        mat_cast * np.sqrt(0.2),
        mat_dir * np.sqrt(0.1),
    ])

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    print(f"[Content] Model built successfully: {len(df)} movies processed.")
    return df, combined, indices

def get_content_score(target_title, ref_titles, combined_matrix, indices):
    if target_title not in indices or not ref_titles:
        return 0.0
    target_vec = combined_matrix[indices[target_title]]
    scores = [
        float(cosine_similarity(target_vec, combined_matrix[indices[ref]])[0, 0])
        for ref in ref_titles if ref in indices
    ]
    return float(np.mean(scores)) if scores else 0.0

#LOAD DATA
def load_pre_split_data(train_file, test_file, meta_file):
    print("[Data] Loading pre-split data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    meta_df = pd.read_csv(meta_file)
    
    meta_dict = dict(zip(meta_df['show_id'], meta_df['title_netflix']))
    
    print(f"[Data] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df, meta_dict

#SVD MODEL
def train_svd(train_df):
    print("[SVD] Training baseline SVD model...")
    reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
    data = Dataset.load_from_df(train_df[['userId', 'show_id', 'rating']], reader)
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    algo.fit(data.build_full_trainset())
    print("[SVD] Training complete.")
    return algo

#FEATURE ENGINEERING FOR XGBOOST
def build_features(df_subset, algo, meta_dict, combined_matrix, content_indices,
                   df_train, scaler=None, fit_scaler=False, n_samples=None):
    item_stats = df_train.groupby('show_id').agg(
        item_count=('rating', 'count'),
        item_mean=('rating', 'mean')
    )
    user_tops = (df_train[df_train['rating'] >= 4.0]
        .groupby('userId')['title_netflix']
        .apply(lambda s: s.head(3).tolist())
        .to_dict()
    )

    rows = []
    if n_samples and n_samples < len(df_subset):
        sample = df_subset.sample(n_samples, random_state=42)
    else:
        sample = df_subset

    print(f"[Features] Building features for {len(sample):,} rows...")
    for _, row in tqdm(sample.iterrows(), total=len(sample), leave=False):
        uid = int(row['userId'])
        sid = int(row['show_id'])
        title = meta_dict.get(sid, '')

        svd_pred = algo.predict(uid, sid).est
        ref_titles = user_tops.get(uid, [])
        c_score = get_content_score(title, ref_titles, combined_matrix, content_indices)
        i_count = item_stats.loc[sid, 'item_count'] if sid in item_stats.index else 0
        i_mean = item_stats.loc[sid, 'item_mean']  if sid in item_stats.index else 3.5

        rows.append({
            'userId':        uid, 
            'content_score': c_score,
            'svd_score':     svd_pred,
            'item_count':    np.log1p(i_count),
            'item_mean':     i_mean,
            'rating':        row['rating'],
        })

    df_feat = pd.DataFrame(rows)
    raw_cols = ['content_score', 'svd_score', 'item_count', 'item_mean']
    norm_cols = ['content_score_norm', 'svd_score_norm', 'item_count_norm', 'item_mean_norm']

    if fit_scaler:
        scaler = MinMaxScaler()
        df_feat[norm_cols] = scaler.fit_transform(df_feat[raw_cols])
    else:
        df_feat[norm_cols] = scaler.transform(df_feat[raw_cols])
    return df_feat, scaler, item_stats, user_tops

#XGBOOST MODEL
def train_xgboost(df_features):
    from sklearn.model_selection import train_test_split 
    print("[XGBoost] Training XGBoost model...")
    X = df_features[FEATURES]
    y = df_features['rating']
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    imp = dict(zip(FEATURES, model.feature_importances_.round(3)))
    print(f"[XGBoost] Feature importances: {imp}")
    return model

#EVALUATION METRICS
def calculate_ranking_metrics(df_test_feat, k=10, threshold=3.5):
    user_est_true = defaultdict(list)

    for _, row in df_test_feat.iterrows():
        user_est_true[row['userId']].append((row['hybrid_pred'], row['rating']))
    
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

# MAIN EXECUTION
if __name__ == "__main__":
    print("=" * 60)
    print("  HYBRID RECOMMENDER — METRICS & DEMONSTRATION")
    print("=" * 60)

    content_df, combined_matrix, content_indices = build_content_model(CONTENT_FILE)
    train_df, test_df, meta_dict = load_pre_split_data(TRAIN_FILE, TEST_FILE, META_FILE)
    
    algo = train_svd(train_df)

    print("\n[Pipeline] Preparing Train dataset features...")
    df_train_feat, scaler, item_stats, user_tops = build_features(
        train_df, algo, meta_dict, combined_matrix, content_indices,
        train_df, fit_scaler=True, n_samples=None
    )
    xgb_model = train_xgboost(df_train_feat)

    print("\n" + "=" * 60)
    print("  EVALUATING ON TEST SET")
    print("=" * 60)
    
    df_test_feat, _, _, _ = build_features(
        test_df, algo, meta_dict, combined_matrix, content_indices,
        train_df, scaler=scaler, fit_scaler=False, n_samples=None
    )
   
    preds = xgb_model.predict(df_test_feat[FEATURES])
    df_test_feat['hybrid_pred'] = preds
    y_test = df_test_feat['rating']

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    precision, recall = calculate_ranking_metrics(df_test_feat, k=TOP_K, threshold=RELEVANCE_THRESHOLD)

    print("\n==========================================")
    print(" HYBRID MODEL: FINAL METRICS")
    print("==========================================")
    print(f"RMSE        : {rmse:.4f}")
    print(f"MAE         : {mae:.4f}")
    print("-" * 42)
    print(f"Precision@{TOP_K}: {precision:.4f}")
    print(f"Recall@{TOP_K}   : {recall:.4f}")
    print("==========================================")

    print("\n" + "=" * 60)
    print(" DEMONSTRATION: HYBRID MODEL IN ACTION")
    print("=" * 60)

    print(f"\n[SCENARIO A: COLD START]")
    print("A new user searches for 'Inception'. The system (Content + Quality) recommends:")
    ref_titles = ['Inception']

    valid_refs = [t for t in ref_titles if t in content_indices]
    if valid_refs:
        profile_vec = sum(combined_matrix[content_indices[t]] for t in valid_refs)
        sims = cosine_similarity(profile_vec, combined_matrix).flatten()
        rating_titles = set(meta_dict.values())
        candidates = [(content_indices.index[i], float(sims[i])) for i in np.argsort(sims)[::-1]
                      if content_indices.index[i] not in set(ref_titles) and content_indices.index[i] in rating_titles]
        candidate_titles = [t for t, _ in candidates[:COLD_CANDIDATE_POOL]]
    else:
        candidate_titles = []

    title_to_sid = {v: k for k, v in meta_dict.items()}
    cand_rows = []
    for title in candidate_titles:
        sid = title_to_sid.get(title)
        c_sim = get_content_score(title, ref_titles, combined_matrix, content_indices)
        i_m = item_stats.loc[sid, 'item_mean'] if sid and sid in item_stats.index else 3.5
        cand_rows.append({'title': title, 'content_score': c_sim, 'item_mean': i_m})

    if cand_rows:
        df_cold = pd.DataFrame(cand_rows)
        for col in ['content_score', 'item_mean']:
            lo, hi = df_cold[col].min(), df_cold[col].max()
            if hi > lo:
                df_cold[col + '_norm'] = (df_cold[col] - lo) / (hi - lo + 1e-9)
            else:
                df_cold[col + '_norm'] = 0.0

        df_cold['hybrid_score'] = (COLD_CONTENT_WEIGHT * df_cold['content_score_norm'] + (1 - COLD_CONTENT_WEIGHT) * df_cold['item_mean_norm'])
        res_cold = df_cold.sort_values('hybrid_score', ascending=False).head(TOP_K)

        for i, row in enumerate(res_cold.itertuples(), 1):
            genres_match = content_df[content_df['title'] == row.title]
            genres = genres_match['genres'].iloc[0] if not genres_match.empty else "Unknown"
            print(f" {i:2d}. {row.title:<35} | Hybrid score: {row.hybrid_score:.2f} | Genres: {genres}")

    print(f"\n [WARM START]")
    print(f"Analyzing the profile of an active user (ID={DEMO_USER_ID})")
    user_history = train_df[train_df['userId'] == DEMO_USER_ID]
    top_history = user_history[user_history['rating'] >= 4.5]
    
    print("Favorite movies:")
    for _, row in top_history.head(5).iterrows():
        print(f"  ✓ {meta_dict.get(row['show_id'], 'Unknown'):<35} [Rating: {row['rating']}]")

    print(f"\nHybrid model (XGBoost) predicts (Top-{TOP_K}):")
    rated = set(user_history['show_id'])
    candidate_sids = set(meta_dict.keys()) - rated
    ref_tops = user_tops.get(DEMO_USER_ID, [])

    cand_rows_warm = []
    for sid in list(candidate_sids)[:2000]:
        title = meta_dict.get(sid, '')
        c_sim = get_content_score(title, ref_tops, combined_matrix, content_indices)
        svd_s = algo.predict(DEMO_USER_ID, sid).est
        i_c = item_stats.loc[sid, 'item_count'] if sid in item_stats.index else 0
        i_m = item_stats.loc[sid, 'item_mean'] if sid in item_stats.index else 3.5
        
        cand_rows_warm.append({
            'title': title, 
            'content_score': c_sim, 
            'svd_score': svd_s, 
            'item_count': np.log1p(i_c), 
            'item_mean': i_m
        })
    df_cand = pd.DataFrame(cand_rows_warm)
    raw_cols = ['content_score', 'svd_score', 'item_count', 'item_mean']
    df_cand[FEATURES] = scaler.transform(df_cand[raw_cols])
    df_cand['xgb_score'] = xgb_model.predict(df_cand[FEATURES])

    res_warm = df_cand.sort_values('xgb_score', ascending=False).head(TOP_K)
    for i, row in enumerate(res_warm.itertuples(), 1):
        genres_match = content_df[content_df['title'] == row.title]
        genres = genres_match['genres'].iloc[0] if not genres_match.empty else "Unknown"
        print(f" {i:2d}. {row.title:<35} | XGB prediction: {row.xgb_score:.2f} | Genres: {genres}")

    print("\n" + "=" * 60)