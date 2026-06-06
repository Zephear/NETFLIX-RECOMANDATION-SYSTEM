import pandas as pd
import re
from thefuzz import fuzz, process

NETFLIX_FILE = r'C:\Users\dovhu\OneDrive\Рабочий стол\bc.Netflix\netflix_movies_detailed_up_to_2025.csv'
MOVIELENS_FILE = r'C:\Users\dovhu\OneDrive\Рабочий стол\bc.Netflix\movies.csv'
NETFLIX_ID_COL = 'show_id'
NETFLIX_TITLE_COL = 'title'
NETFLIX_YEAR_COL = 'release_year'
FUZZY_THRESHOLD = 85
OUTPUT_FILENAME = 'netflix_to_movielens_mapping.csv'

def clean_movie_title(title):
    if pd.isna(title):
        return ""
    
    title = str(title).lower()
    title = re.sub(r'^(.*),\s*(the|a|an)$', r'\2 \1', title)
    title = re.sub(r'[^\w\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title

def main():
    print("Loading data...")
    netflix_df = pd.read_csv(NETFLIX_FILE)
    movielens_df = pd.read_csv(MOVIELENS_FILE)

    # 2. Prepare & Clean Text
    print("Preparing and cleaning text...")
    
    movielens_df['release_year'] = movielens_df['title'].str.extract(r'\((\d{4})\)').astype(float)
    netflix_df['clean_title'] = netflix_df[NETFLIX_TITLE_COL].apply(clean_movie_title)
    movielens_df['clean_title'] = movielens_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).apply(clean_movie_title)
    netflix_df = netflix_df.dropna(subset=[NETFLIX_YEAR_COL])
    movielens_df = movielens_df.dropna(subset=['release_year'])

    # 3. Exact Matching
    print("Searching for exact matches...")
    exact_matches = pd.merge(
        netflix_df, 
        movielens_df, 
        left_on=['clean_title', NETFLIX_YEAR_COL],
        right_on=['clean_title', 'release_year'],
        how='inner',
        suffixes=('_netflix', '_ml') 
    )

    # Format exact match results
    exact_results = exact_matches[[NETFLIX_ID_COL, 'title_netflix', 'movieId', 'title_ml']].copy()
    exact_results.columns = ['show_id', 'title_netflix', 'movieId', 'title_movielens']
    exact_results['match_type'] = 'Exact'
    exact_results['match_score'] = 100

    print(f"Exact matches found: {len(exact_results)}")

    # 4. Fuzzy Matching
    print("Running Fuzzy Matching for the remaining movies...")
    matched_netflix_ids = exact_results['show_id'].tolist()
    unmatched_netflix = netflix_df[~netflix_df[NETFLIX_ID_COL].isin(matched_netflix_ids)].copy()

    fuzzy_results = []

    for index, row in unmatched_netflix.iterrows():
        n_title = row['clean_title']
        n_year = row[NETFLIX_YEAR_COL]
        
        candidates = movielens_df[
            (movielens_df['release_year'] >= n_year - 1) & 
            (movielens_df['release_year'] <= n_year + 1)
        ]
        
        if candidates.empty:
            continue
            
        candidate_dict = dict(zip(candidates['movieId'], candidates['clean_title']))
        best_match = process.extractOne(n_title, candidate_dict, scorer=fuzz.token_sort_ratio)
        
        if best_match and best_match[1] >= FUZZY_THRESHOLD:
            ml_id = best_match[2]
            ml_title_original = candidates[candidates['movieId'] == ml_id]['title'].values[0]
            
            fuzzy_results.append({
                'show_id': row[NETFLIX_ID_COL],
                'title_netflix': row[NETFLIX_TITLE_COL],
                'movieId': ml_id,
                'title_movielens': ml_title_original,
                'match_type': 'Fuzzy',
                'match_score': best_match[1]
            })

    # Combine Results
    fuzzy_df = pd.DataFrame(fuzzy_results)
    
    if not fuzzy_df.empty:
        print(f"Fuzzy matches found: {len(fuzzy_df)}")
    else:
        print("No fuzzy matches found.")

    print("Generating final file...")
    if not fuzzy_df.empty:
        final_mapping = pd.concat([exact_results, fuzzy_df], ignore_index=True)
    else:
        final_mapping = exact_results

    final_mapping.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Saved {len(final_mapping)} matches to file '{OUTPUT_FILENAME}'.")


if __name__ == "__main__":
    main()