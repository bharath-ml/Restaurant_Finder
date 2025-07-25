import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Swiggy Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E8B57;
        margin-bottom: 1rem;
    }
    .restaurant-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the cleaned and encoded datasets"""
    try:
        cleaned_data = pd.read_csv("data/cleaned_data.csv")
        encoded_data = pd.read_csv("data/encoded_data.csv")
        return cleaned_data, encoded_data
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.stop()

@st.cache_data
def load_encoder():
    """Load the OneHotEncoder"""
    try:
        with open("model/encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        return encoder
    except FileNotFoundError:
        st.error("Encoder file not found. Please run the data preprocessing script first.")
        st.stop()

# ...existing code...

def recommend_restaurants(cleaned_data, encoded_data, encoder, user_cities, user_cuisines, min_rating=3.0, max_cost=500, top_n=5):
    """
    Recommend restaurants based on user preferences using cosine similarity
    """
    # Step 1: Filter cleaned_data by selected cities, rating & cost
    filtered = cleaned_data[
        (cleaned_data['city'].isin(user_cities)) &
        (cleaned_data['rating'] >= min_rating) &
        (cleaned_data['cost'] <= max_cost)
    ]
    
    # If no matches after filtering
    if filtered.empty:
        return pd.DataFrame(columns=cleaned_data.columns)
    
    # Step 2: Get indices of filtered restaurants
    filtered_indices = filtered.index
    
    # Step 3: Extract corresponding encoded rows
    encoded_filtered = encoded_data.iloc[filtered_indices]
    
    # Step 4: Encode city input (combine one-hot vectors for all selected cities)
    try:
        city_encoded = encoder.transform(pd.DataFrame({'city': user_cities}))
        # Combine city encodings by OR-ing (if any city matches, set to 1)
        city_encoded_combined = (city_encoded.sum(axis=0) > 0).astype(float)
    except:
        st.warning(f"Some selected cities not found in training data. Using default encoding.")
        city_encoded_combined = np.zeros(len(encoder.get_feature_names_out(['city'])))
    
    # Step 5: One-hot encode cuisines manually
    all_cuisines = [col for col in encoded_filtered.columns if col not in ['rating', 'rating_count', 'cost'] and not col.startswith('city_')]
    cuisine_vector = [1 if cuisine in user_cuisines else 0 for cuisine in all_cuisines]
    
    # Step 6: Combine city + cuisine for similarity calculation
    user_vector = list(city_encoded_combined) + cuisine_vector
    user_vector = [float(i) for i in user_vector]
    
    # Extract only the categorical features from encoded data for similarity
    categorical_features = encoded_filtered.drop(['rating', 'rating_count', 'cost'], axis=1)
    
    # Step 7: Compute cosine similarity
    if len(user_vector) == len(categorical_features.columns):
        similarities = cosine_similarity([user_vector], categorical_features)
        
        # Step 8: Sort & pick top N recommendations
        top_indices = similarities[0].argsort()[-top_n:][::-1]
        recommended = cleaned_data.iloc[filtered_indices.to_numpy()[top_indices]]
        
        return recommended[['name', 'city', 'rating', 'cost', 'cuisine', 'link', 'address']]
    else:
        st.error("Feature dimension mismatch. Please check your data preprocessing.")
        return pd.DataFrame()

def main():
    # Load data and encoder
    cleaned_data, encoded_data = load_data()
    encoder = load_encoder()
    
    # Main header
    st.markdown('<h1 class="main-header">🍽️ Swiggy Restaurant Recommender</h1>', unsafe_allow_html=True)
    st.markdown("Find the perfect restaurant based on your preferences!")
    
    # Sidebar for user inputs
    st.sidebar.markdown('<h2 class="sub-header">🔍 Your Preferences</h2>', unsafe_allow_html=True)
    
    # Get unique cities and cuisines for dropdown options
    unique_cities = sorted(cleaned_data['city'].unique())
    
    # Extract all unique cuisines (handling comma-separated values)
    all_cuisines = set()
    for cuisine_list in cleaned_data['cuisine'].dropna():
        cuisines = [c.strip() for c in str(cuisine_list).split(',')]
        all_cuisines.update(cuisines)
    unique_cuisines = sorted(list(all_cuisines))
    
    # User input widgets
    user_cities = st.sidebar.multiselect(
        "🏙️ Select City/Cities:",
        options=unique_cities,
        default=[unique_cities[0]],
        help="Choose one or more preferred cities"
    )
    
    user_cuisines = st.sidebar.multiselect(
        "🍜 Select Cuisines:",
        options=unique_cuisines,
        default=["Indian", "Fast Food"],
        help="Choose one or more cuisine types"
    )
    
    min_rating = st.sidebar.slider(
        "⭐ Minimum Rating:",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Filter restaurants by minimum rating"
    )
    
    max_cost = st.sidebar.slider(
        "💰 Maximum Cost (₹):",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Set your budget limit"
    )
    
    top_n = st.sidebar.selectbox(
        "📋 Number of Recommendations:",
        options=[5, 10, 15, 20],
        index=0,
        help="How many restaurants to recommend"
    )
    
    # Recommendation button
    if st.sidebar.button("🔍 Get Recommendations", type="primary"):
        if not user_cuisines:
            st.warning("Please select at least one cuisine type.")
        elif not user_cities:
            st.warning("Please select at least one city.")
        else:
            with st.spinner("Finding the best restaurants for you..."):
                recommendations = recommend_restaurants(
                    cleaned_data, encoded_data, encoder,
                    user_cities, user_cuisines, min_rating, max_cost, top_n
                )
                # ...existing code...
# ...existing code...
                
                if recommendations.empty:
                    st.error("No restaurants found matching your criteria. Try adjusting your filters.")
                else:
                    # Display recommendations
                    st.markdown('<h2 class="sub-header">🎯 Recommended Restaurants</h2>', unsafe_allow_html=True)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Found", len(recommendations))
                    with col2:
                        st.metric("Avg Rating", f"{recommendations['rating'].mean():.1f}")
                    with col3:
                        st.metric("Avg Cost", f"₹{recommendations['cost'].mean():.0f}")
                    with col4:
                        st.metric("City", ", ".join(user_cities))
                    
                    st.markdown("---")
                    
                    # Display each restaurant
                    for idx, (_, restaurant) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="restaurant-card">
                                <h3>🍽️ {idx}. {restaurant['name']}</h3>
                                <div class="metric-container">
                                    <strong>📍 Location:</strong> {restaurant['city']}<br>
                                    <strong>⭐ Rating:</strong> {restaurant['rating']}/5<br>
                                    <strong>💰 Cost:</strong> ₹{restaurant['cost']}<br>
                                    <strong>🍜 Cuisine:</strong> {restaurant['cuisine']}<br>
                                    <strong>📧 Address:</strong> {restaurant['address']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if pd.notna(restaurant['link']) and restaurant['link'].strip():
                                st.markdown(f"🔗 [Visit Restaurant Page]({restaurant['link']})")
                            
                            st.markdown("---")
    
    # Dataset info in main area
    if 'recommendations' not in locals() or st.sidebar.button("📊 Show Dataset Info"):
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏙️ Cities Available")
            city_counts = cleaned_data['city'].value_counts().head(10)
            st.bar_chart(city_counts)
            
        with col2:
            st.markdown("### ⭐ Rating Distribution") 
            rating_dist = cleaned_data['rating'].value_counts().sort_index()
            st.bar_chart(rating_dist)
        
        # Additional statistics
        st.markdown("### 📈 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Restaurants", len(cleaned_data))
        with col2:
            st.metric("Cities", len(unique_cities))
        with col3:
            st.metric("Cuisine Types", len(unique_cuisines))
        with col4:
            st.metric("Avg Rating", f"{cleaned_data['rating'].mean():.2f}")
        
        # Show sample data
        st.markdown("### 🔍 Sample Restaurant Data")
        st.dataframe(cleaned_data.head(10), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🍽️ Built with Streamlit | Restaurant Recommendation System<br>
        <small>Powered by Machine Learning & Cosine Similarity</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

