import streamlit as st
import pandas as pd
import numpy as np
import folium
from catboost import CatBoostClassifier
from datetime import datetime
import streamlit.components.v1 as components



st.set_page_config(page_title="GridLock 2.0 - Traffic AI", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
if 'geocoded_data' not in st.session_state:
    st.session_state.geocoded_data = {'lat': None, 'lon': None, 'address': None}

@st.cache_resource(show_spinner=False)
def load_prediction_engine():
    """Loads the compiled CatBoost model."""
    model = CatBoostClassifier()
    try:
        model.load_model("traffic_management_model.cbm")
        return model
    except Exception as e:
        st.error(f"Critical System Failure: Model weights not found. Error: {e}")
        st.stop()

cb_model = load_prediction_engine()



@st.cache_data
def load_categorical_options():
    try:
        df = pd.read_csv("Cleaned_Data_V2.csv") 
        stations = df['police_station'].dropna().unique().tolist()
        stations.sort()
        if "UNKNOWN" not in stations: stations.append("UNKNOWN")
        junctions = df['junction_name'].dropna().unique().tolist()
        junctions.sort()
        if "UNKNOWN" not in junctions: junctions.append("UNKNOWN")
        return stations, junctions
    except:
        return ["UNKNOWN"], ["UNKNOWN"]

POLICE_STATIONS, JUNCTIONS = load_categorical_options()

@st.cache_data
def load_logistics_data():
    try:
        df = pd.read_csv("Cleaned_Data_V2.csv")
        if 'created_time_sin' in df.columns and 'created_hour' not in df.columns:
            angle_h = np.arctan2(df['created_time_sin'], df['created_time_cos'])
            angle_h = np.where(angle_h < 0, angle_h + 2 * np.pi, angle_h)
            df['created_hour'] = np.round((angle_h / (2 * np.pi)) * 24.0) % 24
            
            angle_d = np.arctan2(df['created_day_sin'], df['created_day_cos'])
            angle_d = np.where(angle_d < 0, angle_d + 2 * np.pi, angle_d)
            df['created_day_of_week'] = np.round((angle_d / (2 * np.pi)) * 7.0) % 7
            
            angle_m = np.arctan2(df['created_month_sin'], df['created_month_cos'])
            angle_m = np.where(angle_m < 0, angle_m + 2 * np.pi, angle_m)
            m_val = np.round((angle_m / (2 * np.pi)) * 12.0)
            df['created_month'] = np.where(m_val == 0, 12, m_val)
        return df
    except:
        return pd.DataFrame()

logistics_df = load_logistics_data()



def geocode_address(street, area, city, pincode):
    from geopy.geocoders import ArcGIS
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    geolocator = ArcGIS(user_agent="gridlock_validator_v4")
    
    street = street.strip() if street else ""
    area = area.strip() if area else ""
    city = city.strip() if city else ""
    pincode = pincode.strip() if pincode else ""

    queries_to_try = [
        f"{street}, {area}, {city}, {pincode}", 
        f"{area}, {city}, {pincode}",           
        f"{city}, {pincode}"                    
    ]
    
    for query in queries_to_try:
        clean_query = ", ".join([part.strip() for part in query.split(",") if part.strip()])
        try:
            with st.spinner(f"Querying ArcGIS for: {clean_query}..."):
                location = geolocator.geocode(clean_query, timeout=5)
                if location:
                    st.session_state.geocoded_data = {
                        'lat': location.latitude,
                        'lon': location.longitude,
                        'address': location.address
                    }
                    return True
        except:
            continue 
    return False




st.sidebar.image("Logo.png", use_container_width=True)
st.sidebar.markdown("### AI Traffic Intelligence Platform")
st.sidebar.divider()

page = st.sidebar.radio("Navigate System:", 
    ["🏠 Executive Home", "🚨 Police Command Center", "📦 Flipkart Logistics API"]
)

st.sidebar.divider()




if page == "🏠 Executive Home":
    st.title("🏙️ Welcome to RISE AI ")
    st.markdown("### The dual-engine AI platform solving urban traffic congestion and optimizing last-mile delivery.")
    
    st.divider()
    
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickets Processed (YTD)", "148,431", "+12%")
    col2.metric("AI Auto-Triage Rate", "84.2%", "Model Efficiency")
    col3.metric("Human Hours Saved", "4,200 hrs", "Cost Reduction")
    col4.metric("Delivery SLAs Saved", "18,000+", "Logistics Impact")
    
    st.divider()
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.info("#### 🚨 Module 1: Police Command Center")
        st.write("An automated verification pipeline for municipal traffic enforcement.")
        st.write("- **Machine Learning:** CatBoost Gradient Boosting")
        st.write("- **Function:** Instantly verifies or rejects camera-flagged parking tickets.")
        st.write("- **Value:** Eliminates the human bottleneck in ticket processing, allowing for rapid enforcement.")


    with col_b:
        st.success("#### 📦 Module 2: Flipkart Logistics API")
        st.write("A predictive routing intelligence dashboard for supply chain optimization.")
        st.write("- **Spatial AI:** DBSCAN Density Clustering")
        st.write("- **Function:** Maps historical congestion chokepoints by specific hours and days.")
        st.write("- **Value:** Allows Flipkart routing algorithms to dynamically avoid gridlock and switch to EV 2-wheelers in dense zones.")
        



elif page == "🚨 Police Command Center":
    st.title("🚨 GridLock: Police Dispatch & Triage")
    st.markdown("Intelligent pre-processing and verification pipeline for municipal traffic violations.")
    st.divider()

    col_data, col_geo, col_violation = st.columns([1.2, 1, 1.2])

    with col_data:
        st.subheader("1. Incident Metadata")
        vehicle_number = st.text_input("Vehicle Registration (License Plate)", placeholder="e.g., KA-01-AB-1234")
        vehicle_type = st.selectbox("Vehicle Classification", ["CAR", "SCOOTER", "MAXI-CAB", "TANKER", "MOTOR CYCLE", "UNKNOWN"])
        police_station = st.selectbox("Jurisdictional Police Station", POLICE_STATIONS)
        junction_name = st.selectbox("Junction Name", JUNCTIONS) 
        st.markdown("##### Temporal Data")
        
        if "default_time" not in st.session_state:
            default_date = datetime.now().date()
            default_time = datetime.now().time()
        ticket_date = st.date_input("Violation Date", value = default_date)
        ticket_time = st.time_input("Violation Time", value = default_time)

    with col_geo:
        st.subheader("2. Geospatial Locator")
        street = st.text_input("Street / Road Name", placeholder="18th Main Road")
        area = st.text_input("Locality / Area", placeholder="Koramangala")
        city = st.text_input("City", value="Bengaluru")
        pincode = st.text_input("Pincode", placeholder="560068")
        
        if st.button("🌐 Fetch Coordinates", use_container_width=True):
            if not all([street, area, city]):
                st.warning("Please provide at least Street, Area, and City.")
            else:
                success = geocode_address(street, area, city, pincode)
                if not success:
                    st.warning("Could not resolve address. Try simplifying the street name.")
        
        if st.session_state.geocoded_data['lat']:
            st.success("Location Resolved!")
            st.metric(label="Latitude", value=f"{st.session_state.geocoded_data['lat']:.6f}")
            st.metric(label="Longitude", value=f"{st.session_state.geocoded_data['lon']:.6f}")
            st.caption(f"ArcGIS Standardized: {st.session_state.geocoded_data['address']}")

    with col_violation:
        st.subheader("3. Camera Violation Flags")
        with st.container(border=True):
            viol_wrong_parking = st.checkbox("WRONG PARKING")
            viol_no_parking = st.checkbox("NO PARKING")
            viol_double_parking = st.checkbox("DOUBLE PARKING")
            viol_obstruct_driver = st.checkbox("OBSTRUCTING DRIVER")
            viol_main_road = st.checkbox("PARKING IN A MAIN ROAD")
            viol_footpath = st.checkbox("PARKING ON FOOTPATH")
            viol_safety_belt = st.checkbox("FAIL TO USE SAFETY BELTS")
            viol_defective_plate = st.checkbox("DEFECTIVE NUMBER PLATE")

    st.divider()

    if st.button("⚡ Execute ML Validation Pipeline", type="primary", use_container_width=True):
        if not vehicle_number:
            st.error("Pipeline Halted: Vehicle Number is required.")
            st.stop()
        if not st.session_state.geocoded_data['lat']:
            st.error("Pipeline Halted: Coordinates missing. Please 'Fetch Coordinates' first.")
            st.stop()
            
        hour = ticket_time.hour
        day_of_week = ticket_date.weekday()
        month = ticket_date.month
        is_weekend = 1 if day_of_week in [5, 6] else 0
        
        import pickle
        try:
            with open("spatial_tree.pkl", "rb") as f:
                spatial_data = pickle.load(f)
                spatial_tree = spatial_data['tree']
                cluster_labels = spatial_data['cluster_labels']
            new_coord_radians = np.radians([[float(st.session_state.geocoded_data['lat']), float(st.session_state.geocoded_data['lon'])]])
            dist, ind = spatial_tree.query(new_coord_radians, k=1)
            eps_radians = (200 / 1000.0) / 6371.0088
            assigned_cluster = cluster_labels[ind[0][0]] if dist[0][0] <= eps_radians else "Cluster_-1"
        except Exception as e:
            st.error(f"Failed to load spatial mapping: {e}")
            st.stop()
            
        input_payload = {
            "police_station": [str(police_station)], "junction_name": [str(junction_name)],
            "vehicle_type": [str(vehicle_type)], "cluster_id": [str(assigned_cluster)], 
            "latitude": [float(st.session_state.geocoded_data['lat'])], "longitude": [float(st.session_state.geocoded_data['lon'])],
            "viol_DEFECTIVE NUMBER PLATE": [int(viol_defective_plate)], "viol_DEMANDING EXCESS FARE": [0],
            "viol_DOUBLE PARKING": [int(viol_double_parking)], "viol_FAIL TO USE SAFETY BELTS": [int(viol_safety_belt)],
            "viol_H T V PROHIBITED": [0], "viol_NO PARKING": [int(viol_no_parking)],
            "viol_OBSTRUCTING DRIVER": [int(viol_obstruct_driver)], "viol_PARKING IN A MAIN ROAD": [int(viol_main_road)],
            "viol_PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": [0], "viol_PARKING NEAR ROAD CROSSING": [0],
            "viol_PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": [0], "viol_PARKING ON FOOTPATH": [int(viol_footpath)],
            "viol_PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": [0], "viol_PARKING OTHER THAN BUS STOP": [0],
            "viol_REFUSE TO GO FOR HIRE": [0], "viol_WITHOUT SIDE MIRROR": [0], "viol_WRONG PARKING": [int(viol_wrong_parking)],
            "created_is_weekend": [is_weekend], "created_time_sin": [np.sin(2 * np.pi * hour / 24.0)],
            "created_time_cos": [np.cos(2 * np.pi * hour / 24.0)], "created_day_sin": [np.sin(2 * np.pi * day_of_week / 7.0)],
            "created_day_cos": [np.cos(2 * np.pi * day_of_week / 7.0)], "created_month_sin": [np.sin(2 * np.pi * month / 12.0)],
            "created_month_cos": [np.cos(2 * np.pi * month / 12.0)], "modified_is_weekend": [is_weekend],
            "modified_time_sin": [np.sin(2 * np.pi * hour / 24.0)], "modified_time_cos": [np.cos(2 * np.pi * hour / 24.0)],
            "modified_day_sin": [np.sin(2 * np.pi * day_of_week / 7.0)], "modified_day_cos": [np.cos(2 * np.pi * day_of_week / 7.0)],
            "modified_month_sin": [np.sin(2 * np.pi * month / 12.0)], "modified_month_cos": [np.cos(2 * np.pi * month / 12.0)],
            "processing_delay_hours": [2.5]
        }

        inference_df = pd.DataFrame(input_payload)
        
        try:
            expected_features = cb_model.feature_names_
            for col in expected_features:
                if col not in inference_df.columns:
                    inference_df[col] = "UNKNOWN" if any(x in col for x in ['vehicle','station','junction']) else 0.0
            inference_df = inference_df[expected_features] 
        except:
            st.error("Critical Feature Alignment Error")
            st.stop()
            
        try:
            probabilities = cb_model.predict_proba(inference_df)[0]
            prob_approved = probabilities[1]
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.stop()
        
        st.divider()
        st.subheader(f"Verification Dashboard: {vehicle_number}")
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            st.markdown("#### AI Prediction")
            if prob_approved > 0.40: 
                st.success(f"### Status:  **APPROVED**")
                st.metric("Approval Confidence", f"{prob_approved*100:.1f}%")
            else:
                st.error(f"### Status:  **REJECTED**")
                st.metric("Rejection Confidence", f"{probabilities[0]*100:.1f}%")
            st.markdown("#### Geographic Logic")
            st.write(f"**Assigned Region:** `{assigned_cluster}`")
                
        with res_col2:
            st.markdown("#### Spatial Cluster Mapping")
            input_lat = float(st.session_state.geocoded_data['lat'])
            input_lon = float(st.session_state.geocoded_data['lon'])
            m = folium.Map(location=[input_lat, input_lon], zoom_start=16)
            folium.Circle(location=[input_lat, input_lon], radius=200, color='gray', fill=True, fillOpacity=0.2).add_to(m)
            folium.Marker([input_lat, input_lon], icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

            if assigned_cluster != "Cluster_-1":
                try:
                    tree_coords_rad = np.asarray(spatial_tree.data)
                    tree_coords_deg = np.degrees(tree_coords_rad)
                    cluster_indices = np.where(cluster_labels == assigned_cluster)[0]
                    for idx in cluster_indices[:100]: 
                        pt_lat, pt_lon = tree_coords_deg[idx]
                        folium.CircleMarker(location=[pt_lat, pt_lon], radius=4, color='blue', fill=True, fillOpacity=0.7).add_to(m)
                except:
                    pass
            components.html(m._repr_html_(), height=400)




elif page == "📦 Flipkart Logistics API":
    st.title("📦 Last-Mile Routing Intelligence")
    st.markdown("Query historical spatial density models to predict delivery chokepoints and optimize fleet dispatch.")
    st.divider()
    
    if logistics_df.empty:
        st.warning("Data Matrix unavailable. Ensure 'Cleaned_Data_V2.csv' is in the directory.")
    else:
        st.subheader("Filter Historical Congestion Window")
        col_m, col_d, col_h = st.columns(3)
        with col_m:
            selected_month = st.selectbox("Month", options=[11, 12, 1, 2, 3], 
                format_func=lambda x: {11:'November', 12:'December', 1:'January', 2:'February', 3:'March'}.get(x, str(x)))
        with col_d:
            selected_day = st.selectbox("Day of the Week", options=[0, 1, 2, 3, 4, 5, 6], 
                format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        with col_h:
            selected_hour = st.slider("Time of Day (24-Hour)", min_value=0, max_value=23, value=18)

        filtered_df = logistics_df[
            (logistics_df['created_month'] == selected_month) & 
            (logistics_df['created_day_of_week'] == selected_day) & 
            (logistics_df['created_hour'] == selected_hour)
        ]
        
        st.divider()

        st.subheader("📍 Delivery Route Validation")
    
        warehouse_coords = [13.0645, 77.4585]
        destination = st.selectbox("Select Delivery Destination Zone:", options=POLICE_STATIONS)
    
        if st.button("Calculate Route Safety"):
            st.write(f"Analyzing historical congestion between Warehouse and {destination}...")
            area_df = filtered_df[filtered_df['police_station'] == destination]
            area_violations = len(area_df) 
        
            if area_violations < 50:
                st.success(f"Route to {destination} is Clear. Dispatch Authorized.")
                # 
            else:
                st.error(f"High Congestion Alert! {area_violations} violations detected in {destination}.")
                st.warning("Recommendation: Reroute via Outer Ring Road or delay dispatch by 60 minutes.") #fix this , why only outer ring road 
                # 

        st.divider()
        st.subheader("Operational Impact Dashboard")
        
        kpi1, kpi2, kpi3 = st.columns(3)
        active_chokepoints = len(filtered_df)
        
        kpi1.metric("Active Urban Chokepoints", active_chokepoints)
        kpi2.metric("Est. Fleet Delay Avoided", f"{active_chokepoints * 8} mins") 
        
        if active_chokepoints > 150:
            kpi3.metric("Fleet Routing Protocol", "2-Wheeler EVs", delta="Severe Congestion", delta_color="inverse")
        elif active_chokepoints > 50:
            kpi3.metric("Fleet Routing Protocol", "Light Vans (LMVs)", delta="Moderate Congestion", delta_color="off")
        else:
            kpi3.metric("Fleet Routing Protocol", "Heavy Trucks (HGVs)", delta="Clear Routes", delta_color="normal")

        st.markdown(f"#### Active Blockages at {selected_hour}:00")
        
        if active_chokepoints > 0:
            bengaluru_center = [12.9716, 77.5946]
            m_logistics = folium.Map(location=bengaluru_center, zoom_start=11, tiles='CartoDB dark_matter')
            display_df = filtered_df.sample(n=min(500, active_chokepoints))
            
            for idx, row in display_df.iterrows():
                rad = 8 if row.get('veh_LORRY/GOODS VEHICLE', 0) == 1 else 3
                color = 'red' if rad == 8 else 'orange'
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=rad, color=color, fill=True, fill_opacity=0.6,
                ).add_to(m_logistics)
                
            components.html(m_logistics._repr_html_(), height=550)
        else:
            st.success("No significant blockages detected for this temporal window. Standard routing protocol engaged.")