import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load encoded data and cleaned data
encoded_data = pd.read_csv("data/encoded_data.csv")
cleaned_data = pd.read_csv("data/cleaned_data.csv")

# Load OneHotEncoder for 'city'
with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -------- Recommendation Function --------
def recommend_restaurants(user_city, user_cuisines, min_rating=3.0, max_cost=500, top_n=5):
    # --- Step 1: Filter cleaned_data by rating & cost ---
    filtered = cleaned_data[
        (cleaned_data['rating'] >= min_rating) &
        (cleaned_data['cost'] <= max_cost)
    ]

    # If no matches after filtering
    if filtered.empty:
        return pd.DataFrame(columns=cleaned_data.columns)

    # --- Step 2: Get indices of filtered restaurants ---
    filtered_indices = filtered.index

    # --- Step 3: Extract corresponding encoded rows ---
    encoded_filtered = encoded_data.iloc[filtered_indices]

    # --- Step 4: Encode city input ---
    # --- Step 4: Encode city input ---
    city_encoded = encoder.transform(pd.DataFrame({'city': [user_city]}))


    # --- Step 5: One-hot encode cuisines manually ---
    # --- Step 5: One-hot encode cuisines manually ---
    all_cuisines = encoded_filtered.columns[len(city_encoded[0]):]  # cuisine columns start after city one-hot columns
    cuisine_vector = [1 if cuisine in user_cuisines else 0 for cuisine in all_cuisines]

# --- Step 6: Combine city + cuisine + dummy rating/cost for similarity ---
# We'll only use city & cuisine similarity for now
    user_vector = list(city_encoded[0]) + cuisine_vector
    user_vector = [float(i) for i in user_vector]  # ensure numeric

# --- Step 7: Compute cosine similarity ---
    similarities = cosine_similarity([user_vector], encoded_filtered)

    # --- Step 8: Sort & pick top N recommendations ---
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    recommended = cleaned_data.iloc[filtered_indices[top_indices]]

    return recommended[['name', 'city', 'rating', 'cost', 'cuisine', 'link', 'address']]

# Test the recommender
recs = recommend_restaurants(
    user_city="Abohar",
    user_cuisines=["Fast Food", "Indian"],
    min_rating=3.5,
    max_cost=300,
    top_n=5
)

print(recs)
