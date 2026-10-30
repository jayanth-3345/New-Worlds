
import pandas as pd
from pathlib import Path

# Load original NASA dataset
df = pd.read_csv("data/raw/exoplanets.csv")

print("Original dataset:", df.shape)

# Select useful, mostly complete features
selected_columns = [
    "pl_name",
    "pl_rade",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_mass",
    "st_rad"
]

clean_df = df[selected_columns].copy()

# Fill missing stellar values with their column medians
stellar_columns = ["st_teff", "st_mass", "st_rad"]

for column in stellar_columns:
    clean_df[column] = clean_df[column].fillna(
        clean_df[column].median()
    )

# Check for missing values
print("\nMissing values after cleaning:")
print(clean_df.isnull().sum())

# Save processed dataset
Path("data/processed").mkdir(parents=True, exist_ok=True)

clean_df.to_csv(
    "data/processed/exoplanets_clean.csv",
    index=False
)

print("\nCleaned dataset shape:", clean_df.shape)
print("\nSaved: data/processed/exoplanets_clean.csv")
print("\nDataset preparation completed!")
