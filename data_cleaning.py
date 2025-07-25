# data_cleaning.py

import pandas as pd
import numpy as np
import re

# Load raw dataset
df = pd.read_csv(r"data/swiggy.csv")

# Remove duplicates
df.drop_duplicates(subset="id", inplace=True)

# Clean 'rating' column
df['rating'] = df['rating'].replace('--', np.nan)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Clean 'rating_count'
df['rating_count'] = df['rating_count'].apply(lambda x: re.sub(r'\D', '', str(x)))
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

# Clean 'cost'
df['cost'] = df['cost'].astype(str).str.replace("â‚¹", "").str.replace("₹", "").str.strip()
df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

# Drop rows with missing critical values
df.dropna(subset=['name', 'city', 'rating', 'rating_count', 'cost', 'cuisine'], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned dataset
df.to_csv("data/cleaned_data.csv", index=False)

print("✅ Cleaned data saved to 'data/cleaned_data.csv'")
