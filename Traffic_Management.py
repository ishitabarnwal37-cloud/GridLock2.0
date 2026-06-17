import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import ast
from catboost import CatBoostClassifier

df = pd.read_csv('Cleaned_Data_V2.csv')
if 'validation_status' in df.columns:
    df = pd.get_dummies(df, columns=['validation_status'], prefix='status', dtype=int)
df= df[(df["status_APPROVED"] == 1) | (df["status_REJECTED"] == 1)]
Y = df["status_APPROVED"].values
X = df.drop(columns=['status_APPROVED','status_REJECTED','status_PENDING','status_PROCESSING','status_DUPLICATE','location','data_sent_to_scita_timestamp'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

cat_features = ['vehicle_number','vehicle_type','police_station','data_sent_to_scita','junction_name','updated_vehicle_number','updated_vehicle_type']

model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.3,
    depth=8,
    verbose=10 # Prints log every 10 iterations
)

model.fit(
    X_train, 
    y_train, 
    cat_features=cat_features, 
    eval_set=(X_test, y_test)
)

y_pred = model.predict(X_test)
predictions = (y_pred >= 0.5).astype(int)
print("--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=['Rejected', 'Approved']))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))

model.save_model("traffic_management_model.cbm")
print("model saved!")