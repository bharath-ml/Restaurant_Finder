# encoding_data.py

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# --------- SPLIT MULTI-LABEL CUISINE ---------
# Convert 'Fast Food,Indian' to multiple entries
df['cuisine'] = df['cuisine'].str.split(',')

# Convert multi-label 'cuisine' to binary features
cuisine_dummies = df['cuisine'].explode().str.strip().str.get_dummies().groupby(level=0).sum()

# One-hot encode 'city'
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
city_encoded = encoder.fit_transform(df[['city']])
city_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(['city']))

# Numerical features
numerical_df = df[['rating', 'rating_count', 'cost']].reset_index(drop=True)

# Concatenate all encoded features
encoded_df = pd.concat([numerical_df, city_df, cuisine_dummies], axis=1)

# Save encoded dataset
encoded_df.to_csv("data/encoded_data.csv", index=False)

# Save encoder
import os

# Create 'model' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Now save the encoder
with open("model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Encoded data saved to 'data/encoded_data.csv'")
print("✅ OneHotEncoder saved as 'model/encoder.pkl'")
