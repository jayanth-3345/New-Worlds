
import pandas as pd

# Load both datasets
nasa = pd.read_csv("data/raw/exoplanets.csv")
phl = pd.read_csv("data/raw/hwc.csv")

# Select NASA features
nasa_features = [
    "pl_name",
    "pl_rade",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_mass",
    "st_rad"
]

nasa = nasa[nasa_features].copy()

# Select PHL planet names and habitability labels
phl_labels = phl[["P_NAME", "S_NAME", "P_HABITABLE"]].copy()

# Normalize planet names for matching
nasa["planet_key"] = nasa["pl_name"].str.strip().str.lower()
phl_labels["planet_key"] = phl_labels["P_NAME"].str.strip().str.lower()

# Check duplicate keys before merging
print("NASA duplicate names:", nasa["planet_key"].duplicated().sum())
print("PHL duplicate names:", phl_labels["planet_key"].duplicated().sum())

# Merge datasets
merged = pd.merge(
    nasa,
    phl_labels[["planet_key", "S_NAME", "P_HABITABLE"]],
    on="planet_key",
    how="inner",
    validate="one_to_one"
)

# Remove matching helper column
merged = merged.drop(columns=["planet_key"])

# Rename columns
merged = merged.rename(columns={
    "pl_name": "planet_name",
    "P_HABITABLE": "habitability_label",
    "S_NAME": "host_star",
})

# Save labelled dataset
merged.to_csv(
    "data/processed/exoplanets_labelled.csv",
    index=False
)

# Display results
print("\n========== LABELLED DATASET ==========")
print("Total matched planets:", len(merged))

print("\nLabel distribution:")
print(merged["habitability_label"].value_counts().sort_index())

print("\nMissing values:")
print(merged.isnull().sum())

print("\nFirst five records:")
print(merged.head())

print("\nSaved: data/processed/exoplanets_labelled.csv")
