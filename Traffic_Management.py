import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import BallTree
from catboost import CatBoostClassifier

# ==========================================
# 1. DATA LOADING & BASIC CLEANING
# ==========================================
print("Loading Cleaned_Data_V2.csv...")
df = pd.read_csv("Cleaned_Data_V2.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Filter for rows with valid targets
df = df[df['validation_status'].str.upper().isin(['APPROVED', 'REJECTED'])].copy()

# Map target to binary labels
y = df['validation_status'].apply(lambda x: 1 if str(x).upper() == 'APPROVED' else 0)

# Drop missing coordinates so spatial lookups don't crash
df = df.dropna(subset=['latitude', 'longitude']).copy()
y = y.loc[df.index]  # Keep target perfectly aligned with remaining rows

"""
print("Running DBSCAN spatial clustering...")
coords = np.radians(df[['latitude', 'longitude']].values)
kms_per_radian = 6371.0088
eps_radians = (200 / 1000.0) / kms_per_radian

# CONTRADICTION : DBSCAN flags outliers as "Cluster_-1". 
# It cannot natively predict clusters for new incoming points.
db = DBSCAN(eps=eps_radians, min_samples=3, metric='haversine').fit(coords)
df['cluster_id'] = 'Cluster_' + db.labels_.astype(str)

# CONTRADICTION : Forcing a BallTree to map new points to a DBSCAN cluster 
# completely bypasses DBSCAN's density logic, forcing noise data into clusters anyway.
tree = BallTree(coords, metric='haversine')
with open("spatial_tree.pkl", "wb") as f:
    pickle.dump({'tree': tree, 'cluster_labels': df['cluster_id'].values}, f)
print("Saved spatial_tree.pkl")
"""

print("Building BallTree and calculating spatial features...")

# Convert coordinates to radians for the Haversine metric
coords = np.radians(df[['latitude', 'longitude']].values)

# Build the BallTree directly on the coordinates
tree = BallTree(coords, metric='haversine')

# Query the 11 nearest neighbors (k=11 because index 0 is always the point itself)
distances, indices = tree.query(coords, k=11)
neighbor_indices = indices[:, 1:]  # Drop the first column (self-reference)

# Calculate the local historical approval rate among neighbors to use as a feature
local_approval_rates = []
for idx_list in neighbor_indices:
    # Grab the historical binary target (1 or 0) of the nearest points and average them
    rate = y.iloc[idx_list].mean()
    local_approval_rates.append(rate)

df['local_approval_rate'] = local_approval_rates

# Save the BallTree AND the corresponding targets so you can recreate this feature during real-time inference
with open("spatial_tree.pkl", "wb") as f:
    pickle.dump({'tree': tree, 'training_targets': y.values}, f)
print("Saved spatial_tree.pkl")


# ==========================================
# 4. PREPARING FEATURES FOR CATBOOST
# ==========================================
# Clean up missing values for remaining categorical variables
# Note: 'cluster_id' is removed since we abandoned the DBSCAN approach
categorical_cols = ['police_station', 'junction_name', 'vehicle_type']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

# Define data leakage columns and useless strings to drop
leaky_columns = [
    'validation_status', 'data_sent_to_scita', 'data_sent_to_scita_timestamp', 
    'updated_vehicle_number', 'updated_vehicle_type', 'is_validated',
    'val_is_weekend', 'val_time_sin', 'val_time_cos', 
    'val_day_sin', 'val_day_cos', 'val_month_sin', 'val_month_cos'
]
useless_text_columns = ['id', 'location', 'vehicle_number']

#  CONTRADICTION 3 : Previously, 'latitude' and 'longitude' were left in X, 
# meaning CatBoost was getting redundant raw data alongside the spatial feature.
# We now add them to the drop list to avoid spatial feature redundancy.
redundant_spatial_columns = ['latitude', 'longitude']

X = df.drop(columns=leaky_columns + useless_text_columns + redundant_spatial_columns, errors='ignore')


# ==========================================
# 5. CATBOOST TRAINING
# ==========================================
print("Initiating CatBoost Training...")
cb_model = CatBoostClassifier(
    cat_features=categorical_cols,
    iterations=500,
    depth=6,
    learning_rate=0.1,
    verbose=50
)

# Fit the model using the engineered BallTree feature instead of the broken clusters
cb_model.fit(X, y)
cb_model.save_model("traffic_management_model.cbm")
print("✅ Model trained successfully with pure BallTree features and saved!")