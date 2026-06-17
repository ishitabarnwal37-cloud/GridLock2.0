import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from catboost import CatBoostClassifier

# ==========================================
# 1. LOAD DATA & FIX HEADERS
# ==========================================
print("Loading Cleaned_Data_V2.csv...")
df = pd.read_csv("Cleaned_Data_V2.csv")

# Strip invisible trailing spaces from headers (This kills the KeyError)
df.columns = df.columns.str.strip()

# ==========================================
# 2. FILTER USING YOUR COLUMN
# ==========================================
df = df[df['validation_status'].str.upper().isin(['APPROVED', 'REJECTED'])].copy()

# Create Target Variable (1 = Approved, 0 = Rejected)
y = df['validation_status'].apply(lambda x: 1 if str(x).upper() == 'APPROVED' else 0)

# ==========================================
# 3. DBSCAN SPATIAL CLUSTERING
# ==========================================
print("Running DBSCAN spatial clustering...")
df = df.dropna(subset=['latitude', 'longitude']).copy()
y = y.loc[df.index]  # Keep target aligned with dropped rows

coords = np.radians(df[['latitude', 'longitude']].values)
kms_per_radian = 6371.0088
eps_radians = (200 / 1000.0) / kms_per_radian

db = DBSCAN(eps=eps_radians, min_samples=3, metric='haversine').fit(coords)
df['cluster_id'] = 'Cluster_' + db.labels_.astype(str)

# Save spatial tree for the web app
tree = BallTree(coords, metric='haversine')
with open("spatial_tree.pkl", "wb") as f:
    pickle.dump({'tree': tree, 'cluster_labels': df['cluster_id'].values}, f)
print("Saved spatial_tree.pkl")

# ==========================================
# 4. PREVENT LEAKAGE & PREP FEATURES
# ==========================================
categorical_cols = ['police_station', 'junction_name', 'vehicle_type', 'cluster_id']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN").astype(str)

# Drop the leaky future columns and useless text strings
leaky_columns = [
    'validation_status', 'data_sent_to_scita', 'data_sent_to_scita_timestamp', 
    'updated_vehicle_number', 'updated_vehicle_type', 'is_validated',
    'val_is_weekend', 'val_time_sin', 'val_time_cos', 
    'val_day_sin', 'val_day_cos', 'val_month_sin', 'val_month_cos'
]
useless_text_columns = ['id', 'location', 'vehicle_number']

X = df.drop(columns=leaky_columns + useless_text_columns, errors='ignore')

# ==========================================
# 5. TRAIN CATBOOST
# ==========================================
print("Initiating CatBoost Training...")
cb_model = CatBoostClassifier(
    cat_features=categorical_cols,
    iterations=500,
    depth=6,
    learning_rate=0.1,
    verbose=50
)

cb_model.fit(X, y)
cb_model.save_model("traffic_management_model_Rudra.cbm")
print("✅ Model trained successfully on Cleaned_Data_V2.csv and saved!")