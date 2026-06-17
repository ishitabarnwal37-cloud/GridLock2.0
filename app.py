import streamlit as strl
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from datetime import datetime

# Set page configurations
strl.set_page_config(page_title="GridLock2.0 - Traffic Ticket Validator", layout="wide")

strl.title("🚦 GridLock 2.0: AI Traffic Management & Ticket Validator")
strl.markdown("Predict the verification outcome of traffic violations using optimized machine learning layers.")

# 1. Load the trained CatBoost weights
@strl.cache_resource
def load_prediction_engine():
    model = CatBoostClassifier()
    model.load_model("traffic_management_model.cbm")
    return model

try:
    cb_model = load_prediction_engine()
    strl.success("CatBoost Production Engine Loaded Successfully!")
except Exception as e:
    strl.error(f"Could not load model file 'traffic_processed_model.cbm'.Error: {e}")
    strl.stop()

strl.divider()

# Create columns for clean UI segmentation
left_column, right_column = strl.columns([1, 1])

with left_column:
    strl.subheader("Geospatial & Metadata Inputs")
    
    # Coordinates for Bengaluru bounds as defaults
    latitude = strl.number_input("Latitude", value=12.9255567, format="%.7f")
    longitude = strl.number_input("Longitude", value=77.618665, format="%.7f")
    
    # Categorical selectors matching your training dictionary
    police_station = strl.selectbox("Police Station Jurisdiction", 
        ["Madiwala", "Bellandur", "Byatarayanapura", "Upparpet", "Shivajinagar", "Pulikeshinagar(F.Town)", "UNKNOWN"])
    
    junction_name = strl.selectbox("Junction Name", 
        ["No Junction", "BTP044 - Sagar Theatre Junction", "BTP051 - Safina Plaza Junction", "UNKNOWN"])
    
    vehicle_type = strl.selectbox("Vehicle Classification Type", 
        ["CAR", "SCOOTER", "MAXI-CAB", "TANKER", "MOTOR CYCLE", "UNKNOWN"])

    strl.subheader("⏱️ Operational Chronology")
    ticket_time = strl.time_input("Violation Timestamp Creation Time", datetime.now().time())
    ticket_date = strl.date_input("Violation Registration Date", datetime.now().date())
    processing_delay = strl.number_input("System Backlog Processing Delay (Hours)", min_value=0.0, value=2.5, step=0.5)

with right_column:
    strl.subheader("⚠️ Active Traffic Violations")
    strl.caption("Check all applicable automated camera violation flags:")
    
    # Manual multi-select mapping to individual binary vector features
    viol_wrong_parking = strl.checkbox("WRONG PARKING")
    viol_no_parking = strl.checkbox("NO PARKING")
    viol_double_parking = strl.checkbox("DOUBLE PARKING")
    viol_obstruct_driver = strl.checkbox("OBSTRUCTING DRIVER")
    viol_main_road = strl.checkbox("PARKING IN A MAIN ROAD")
    viol_footpath = strl.checkbox("PARKING ON FOOTPATH")
    viol_safety_belt = strl.checkbox("FAIL TO USE SAFETY BELTS")
    viol_defective_plate = strl.checkbox("DEFECTIVE NUMBER PLATE")

# --- DATA PIPELINE & FEATURE ENGINEERING ---
if strl.button("⚡ Validate Ticket Status", type="primary", use_container_width=True):
    
    hour = ticket_time.hour
    day_of_week = ticket_date.weekday()
    month = ticket_date.month
    is_weekend = 1 if day_of_week in [5, 6] else 0
    
    input_payload = {
        # Categorical
        "police_station": [str(police_station)],
        "junction_name": [str(junction_name)],
        "final_vehicle_type": [str(vehicle_type)],
        
        # Binary Violation Flags
        "viol_DEFECTIVE NUMBER PLATE": [int(viol_defective_plate)],
        "viol_DEMANDING EXCESS FARE": [0],
        "viol_DOUBLE PARKING": [int(viol_double_parking)],
        "viol_FAIL TO USE SAFETY BELTS": [int(viol_safety_belt)],
        "viol_H T V PROHIBITED": [0],
        "viol_NO PARKING": [int(viol_no_parking)],
        "viol_OBSTRUCTING DRIVER": [int(viol_obstruct_driver)],
        "viol_PARKING IN A MAIN ROAD": [int(viol_main_road)],
        "viol_PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": [0],
        "viol_PARKING NEAR ROAD CROSSING": [0],
        "viol_PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": [0],
        "viol_PARKING ON FOOTPATH": [int(viol_footpath)],
        "viol_PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": [0],
        "viol_PARKING OTHER THAN BUS STOP": [0],
        "viol_REFUSE TO GO FOR HIRE": [0],
        "viol_WITHOUT SIDE MIRROR": [0],
        "viol_WRONG PARKING": [int(viol_wrong_parking)],
        
        # Spatial Coordinates & Operational metrics
        "latitude": [float(latitude)],
        "longitude": [float(longitude)],
        "created_is_weekend": [is_weekend],
        "created_time_sin": [np.sin(2 * np.pi * hour / 24.0)],
        "created_time_cos": [np.cos(2 * np.pi * hour / 24.0)],
        "created_day_sin": [np.sin(2 * np.pi * day_of_week / 7.0)],
        "created_day_cos": [np.cos(2 * np.pi * day_of_week / 7.0)],
        "created_month_sin": [np.sin(2 * np.pi * month / 12.0)],
        "created_month_cos": [np.cos(2 * np.pi * month / 12.0)],
        
        # Dummy values matching modification/validation timestamps defaults
        "modified_is_weekend": [is_weekend],
        "modified_time_sin": [np.sin(2 * np.pi * hour / 24.0)],
        "modified_time_cos": [np.cos(2 * np.pi * hour / 24.0)],
        "modified_day_sin": [np.sin(2 * np.pi * day_of_week / 7.0)],
        "modified_day_cos": [np.cos(2 * np.pi * day_of_week / 7.0)],
        "modified_month_sin": [np.sin(2 * np.pi * month / 12.0)],
        "modified_month_cos": [np.cos(2 * np.pi * month / 12.0)],
        "processing_delay_hours": [float(processing_delay)]
    }

    inference_df = pd.DataFrame(input_payload)
    
    prediction = cb_model.predict(inference_df)[0]
    probabilities = cb_model.predict_proba(inference_df)[0]
    
    strl.subheader("Model Verification Output")
    
    if prediction == 1:
        strl.markdown(f"### Status: ✅ **APPROVED**")
        strl.metric(label="Approval Confidence Score", value=f"{probabilities[1]*100:.2f}%")
        strl.info("Ticket properties match expected baseline regulations. Safely forwarded to payment clearing channels.")
    else:
        strl.markdown(f"### Status: ❌ **REJECTED**")
        strl.metric(label="Rejection Confidence Score", value=f"{probabilities[0]*100:.2f}%")
        strl.warning("High anomaly risk or invalid criteria sequence detected. Routed directly to manual human audit queue.")