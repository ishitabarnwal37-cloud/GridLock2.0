import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import folium as fl
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight,compute_sample_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from xgboost import XGBClassifier

df = pd.read_csv("Dataset.csv")
columns_to_drop = ['closed_datetime', 'description', 'action_taken_timestamp']
df = df.drop(columns=columns_to_drop, errors='ignore')
df = df.dropna(subset=['latitude', 'longitude'])

def clean_violation_string(val):
    try:
        val_list = ast.literal_eval(val)
        if isinstance(val_list, list) and len(val_list) > 0:
            return val_list[0].split(' [')[0].strip()
    except (ValueError, SyntaxError):
        return val
    return val

if 'violation_type' in df.columns:
    df['clean_violation'] = df['violation_type'].apply(clean_violation_string)
    df = df.drop(columns=['violation_type'])
    df = pd.get_dummies(df, columns=['clean_violation'], prefix='viol', dtype=int)


if 'validation_timestamp' in df.columns:
    df['is_validated'] = df['validation_timestamp'].notna().astype(int)



def extract_cyclical_features(df, col_name, prefix):
    """Safely extracts dates, times, and cyclical math from any datetime column."""
    if col_name in df.columns:
        
        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
        
        # UI Features (Standard Strings)
        df[f'{prefix}_date'] = df[col_name].dt.date
        df[f'{prefix}_time'] = df[col_name].dt.time
        
       
        hour = df[col_name].dt.hour
        minute = df[col_name].dt.minute
        day_of_week = df[col_name].dt.dayofweek
        month = df[col_name].dt.month
        
       
        df[f'{prefix}_is_weekend'] = (day_of_week >= 5).astype('Int64')

        
        time_in_hours = hour + (minute / 60.0)
        df[f'{prefix}_time_sin'] = np.sin(2 * np.pi * time_in_hours / 24.0)
        df[f'{prefix}_time_cos'] = np.cos(2 * np.pi * time_in_hours / 24.0)
        
        df[f'{prefix}_day_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
        df[f'{prefix}_day_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
        
        df[f'{prefix}_month_sin'] = np.sin(2 * np.pi * month / 12.0)
        df[f'{prefix}_month_cos'] = np.cos(2 * np.pi * month / 12.0)
        
    return df


df = df.dropna(subset=['created_datetime'])

df = extract_cyclical_features(df, 'created_datetime', 'created')
df = extract_cyclical_features(df, 'modified_datetime', 'modified')
df = extract_cyclical_features(df, 'validation_timestamp', 'val')


if 'modified_datetime' in df.columns and 'created_datetime' in df.columns:
    df['processing_delay_hours'] = ((df['modified_datetime'] - df['created_datetime']).dt.total_seconds() / 3600.0).round(2)


final_drop_columns = ['created_datetime', 'modified_datetime', 'validation_timestamp','val_date','val_time','created_time','created_date','modified_date','modified_time']
df = df.drop(columns=final_drop_columns, errors='ignore')

val_cyclical_columns = [
    'val_time_sin', 'val_time_cos', 
    'val_day_sin', 'val_day_cos', 
    'val_month_sin', 'val_month_cos'
]

for col in val_cyclical_columns:
    if col in df.columns:
        df[col] = df[col].fillna(-2.0)
        

if 'val_is_weekend' in df.columns:
    df['val_is_weekend'] = df['val_is_weekend'].fillna(-1)


if 'updated_vehicle_number' in df.columns:
    df['updated_vehicle_number'] = df['updated_vehicle_number'].fillna('NOT_UPDATED')

if 'updated_vehicle_type' in df.columns:
    df['updated_vehicle_type'] = df['updated_vehicle_type'].str.upper().str.strip()
    df['updated_vehicle_type'] = df['updated_vehicle_type'].fillna('UNKNOWN')
    
    df = pd.get_dummies(df, columns=['updated_vehicle_type'], prefix='veh', dtype=int)

if 'validation_status' in df.columns:
    df['validation_status'] = df['validation_status'].str.upper().str.strip()
    df['validation_status'] = df['validation_status'].fillna('PENDING')
    df = pd.get_dummies(df, columns=['validation_status'], prefix='status', dtype=int)
    
#output_filename = 'Cleaned_Data.csv'
#df.to_csv(output_filename, index=False)


#-------------------------------------------------------------------------------------------------------------------------
#using folium because geocoding api is only for a month, I'm not paying for that shit
df_cleaned = pd.read_csv("Cleaned_Data.csv")
df_cleaned= df_cleaned[(df_cleaned['status_APPROVED'] == 1) | (df_cleaned['status_REJECTED'] == 1)]

#dropping the created and modified time columns as it doesn't make chronological sense to the lore timeline 
# creation -> modification by human -> validation time 
#prediction happening in the first step itself 

df_cleaned = df_cleaned[(df_cleaned["status_APPROVED"] == 1) | (df_cleaned["status_REJECTED"] == 1)]
Y = df_cleaned["status_APPROVED"].values

experimental_features = [
    "modified_is_weekend", "modified_time_sin", "modified_time_cos",
    "modified_day_sin", "modified_day_cos", "modified_month_sin", "modified_month_cos",
    "val_is_weekend", "val_time_sin", "val_time_cos",
    "val_day_sin", "val_day_cos", "val_month_sin", "val_month_cos",
    "processing_delay_hours"
]

features = [
    "latitude", "longitude", "center_code",
    "created_is_weekend", "created_time_sin", "created_time_cos",
    "created_day_sin", "created_day_cos", "created_month_sin", "created_month_cos",
    "viol_DEFECTIVE NUMBER PLATE", "viol_DEMANDING EXCESS FARE", "viol_DOUBLE PARKING",
    "viol_FAIL TO USE SAFETY BELTS", "viol_H T V PROHIBITED", "viol_NO PARKING",
    "viol_OBSTRUCTING DRIVER", "viol_PARKING IN A MAIN ROAD", "viol_PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC",
    "viol_PARKING NEAR ROAD CROSSING", "viol_PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS", "viol_PARKING ON FOOTPATH",
    "viol_PARKING OPPOSITE TO ANOTHER PARKED VEHICLE", "viol_PARKING OTHER THAN BUS STOP",
    "viol_REFUSE TO GO FOR HIRE", "viol_WITHOUT SIDE MIRROR", "viol_WRONG PARKING",
    "veh_BUS (BMTC/KSRTC)", "veh_CAR", "veh_FACTORY BUS", "veh_GOODS AUTO", "veh_HGV", 
    "veh_JEEP", "veh_LGV", "veh_LORRY/GOODS VEHICLE", "veh_MAXI-CAB", "veh_MINI LORRY", 
    "veh_MOPED", "veh_MOTOR CYCLE", "veh_OTHERS", "veh_PASSENGER AUTO", "veh_PRIVATE BUS", 
    "veh_SCHOOL VEHICLE", "veh_SCOOTER", "veh_TANKER", "veh_TEMPO", "veh_TOURIST BUS", 
    "veh_TRACTOR", "veh_UNKNOWN", "veh_VAN"
]

X = df_cleaned[features].values
X = np.nan_to_num(X, nan=0.0)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler1 = StandardScaler()
x_train = scaler1.fit_transform(x_train)
x_test = scaler1.transform(x_test)

#this is for experimenting with what to include 
all_features = experimental_features + features
X_exp = df_cleaned[all_features].values
X_exp = np.nan_to_num(X_exp, nan=0.0)
x_train_exp, x_test_exp, y_train_exp, y_test_exp = train_test_split(X_exp, Y, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
x_train_exp = scaler2.fit_transform(x_train_exp)
x_test_exp = scaler2.transform(x_test_exp)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))
print("Computed Class Weights:", class_weight_dict)

model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') 
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

epochs = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128, 
    validation_split=0.2,
    class_weight=class_weight_dict,
    verbose=1     
)

probabilities = model.predict(x_test)
predictions = (probabilities >= 0.5).astype(int)

print("--- Classification Report ---")
print(classification_report(y_test, predictions, target_names=['Rejected', 'Approved']))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, predictions))

''' without weights : 
-- Classification Report ---
              precision    recall  f1-score   support

    Rejected       0.71      0.23      0.35      9732
    Approved       0.75      0.96      0.84     23299

    accuracy                           0.75     33031
   macro avg       0.73      0.59      0.59     33031
weighted avg       0.74      0.75      0.70     33031

--- Confusion Matrix ---
[[ 2214  7518]
 [  883 22416]]
'''

'''with weights : 
--- Classification Report ---
              precision    recall  f1-score   support

    Rejected       0.46      0.55      0.50      9732
    Approved       0.79      0.73      0.76     23299

    accuracy                           0.68     33031
   macro avg       0.63      0.64      0.63     33031
weighted avg       0.70      0.68      0.68     33031

--- Confusion Matrix ---
[[ 5339  4393]
 [ 6290 17009]]
'''
#accuracy went down but recall increased , so i guess thats good (shrugs)
#confusion matrix order : TN    FP
#                         FN    TP

#moving on from this model 
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=2.4, 
    random_state=42
)
class_weights = compute_sample_weight(class_weight='balanced', y=y_train)
xgb_model.fit(x_train, y_train,sample_weight=class_weights)
xgb_preds = xgb_model.predict(x_test)

print("\nXGBBOOST CLASSIFIER REPORT : ")
print(classification_report(y_test, xgb_preds))
print(confusion_matrix(y_test,xgb_preds))

'''
XGBBOOST CLASSIFIER REPORT : 
              precision    recall  f1-score   support

           0       0.81      0.25      0.38      9732
           1       0.76      0.98      0.85     23299

    accuracy                           0.76     33031
   macro avg       0.78      0.61      0.62     33031
weighted avg       0.77      0.76      0.71     33031

[[ 2400  7332]
 [  550 22749]]
 comments : none , this is shit 
 I should probably apply random subsampling,idk at this point. 
 don't try mapping the points , the browser crashes for me  
'''
