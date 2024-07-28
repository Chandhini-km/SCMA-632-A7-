import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
from datetime import timedelta

# Load model and encoders

model_path = 'model/rf_model.pkl'
encoders_path = 'model/label_encoders.pkl'
model = joblib.load(model_path)
label_encoders = joblib.load(encoders_path)

# Load the dataframe
file_path = 'data/Flight_Fare.xlsx'
df = pd.read_excel(file_path)

# Function to filter dropdown values
def filter_options(main_df, column, filters, time_only=False):
    filtered_df = main_df
    for col, val in filters.items():
        if val is not None and col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col] == val]
    options = filtered_df[column].unique()
    
    if time_only:
        options = [pd.to_datetime(opt).strftime('%H:%M') for opt in options]
    
    return sorted(options)

# Function to safely transform labels
def safe_transform(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        # Assign a new label for unseen values
        le.classes_ = np.append(le.classes_, value)
        return le.transform([value])[0]

# Function to convert time to minutes since midnight
def time_to_minutes(time_str):
    time_obj = pd.to_datetime(time_str, format='%H:%M')
    return time_obj.hour * 60 + time_obj.minute

# Function to calculate duration
def calculate_duration(dep_time, arr_time):
    dep_time_minutes = time_to_minutes(dep_time)
    arr_time_minutes = time_to_minutes(arr_time)
    duration_minutes = arr_time_minutes - dep_time_minutes
    if duration_minutes < 0:
        duration_minutes += 24 * 60  # Handle cases where the arrival time is on the next day
    duration = timedelta(minutes=duration_minutes)
    return duration

# Layout for the title
st.markdown("<h1 style='text-align: center;'>AirFare Prediction - By Chandhini Kerachan Muraleedharan</h1>", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    # Airline selection
    selected_airline = st.selectbox("Airline", filter_options(df, 'Airline', {}))
    selected_airline_index = safe_transform(label_encoders['Airline'], selected_airline)
    
    # Source selection based on airline
    filtered_sources = filter_options(df, 'Source', {'Airline': selected_airline})
    selected_source = st.selectbox("Source", filtered_sources)
    selected_source_index = safe_transform(label_encoders['Source'], selected_source)

    # Destination selection based on airline and source
    filtered_destinations = filter_options(df, 'Destination', {'Airline': selected_airline, 'Source': selected_source})
    selected_destination = st.selectbox("Destination", filtered_destinations)
    selected_destination_index = safe_transform(label_encoders['Destination'], selected_destination)

with col2:
    # Route selection based on airline, source, and destination
    filtered_routes = filter_options(df, 'Route', {'Airline': selected_airline, 'Source': selected_source, 'Destination': selected_destination})
    selected_route = st.selectbox("Route", filtered_routes)
    selected_route_index = safe_transform(label_encoders['Route'], selected_route)
    
    # Departure Time input
    filtered_dep_times = filter_options(df, 'Dep_Time', {'Airline': selected_airline, 'Source': selected_source, 'Destination': selected_destination, 'Route': selected_route})
    selected_dep_time = st.selectbox("Departure Time", filtered_dep_times)
    
    # Arrival Time input
    filtered_arrival_times = filter_options(df, 'Arrival_Time', {'Airline': selected_airline, 'Source': selected_source, 'Destination': selected_destination, 'Route': selected_route, 'Dep_Time': selected_dep_time}, time_only=True)
    selected_arrival_time = st.selectbox("Arrival Time", filtered_arrival_times)
    
    # Total Stops selection based on previous filters
    filtered_stops = filter_options(df, 'Total_Stops', {'Airline': selected_airline, 'Source': selected_source, 'Destination': selected_destination, 'Route': selected_route, 'Dep_Time': selected_dep_time, 'Arrival_Time': selected_arrival_time})
    selected_stops = st.selectbox("Total Stops", filtered_stops)
    selected_stops_index = safe_transform(label_encoders['Total_Stops'], selected_stops)
    
    # Date of journey
    date_of_journey = st.date_input("Date of Journey")
    journey_day = date_of_journey.day
    journey_month = date_of_journey.month
    
    # Calculate duration dynamically
    duration = calculate_duration(selected_dep_time, selected_arrival_time)
    st.write(f"Duration: {duration}")
    duration_in_minutes = duration.total_seconds() // 60

# Additional Info selection based on previous filters (moved out of columns for simplicity)
filtered_info = filter_options(df, 'Additional_Info', {'Airline': selected_airline, 'Source': selected_source, 'Destination': selected_destination, 'Route': selected_route, 'Dep_Time': selected_dep_time, 'Arrival_Time': selected_arrival_time, 'Total_Stops': selected_stops})
selected_info = st.selectbox("Additional Info", filtered_info)
selected_info_index = safe_transform(label_encoders['Additional_Info'], selected_info)

# Read feature names directly from the model to ensure consistency
feature_columns = model.feature_names_in_

# Match the order of features exactly as the model expects
data_dict = {
    'Airline': selected_airline_index,
    'Source': selected_source_index,
    'Destination': selected_destination_index,
    'Route': selected_route_index,
    'Dep_Time': time_to_minutes(selected_dep_time),
    'Arrival_Time': time_to_minutes(selected_arrival_time),
    'Total_Stops': selected_stops_index,
    'Additional_Info': selected_info_index,
    'Duration': duration_in_minutes,
    'Journey_day': journey_day,
    'Journey_month': journey_month,
}

# Create data record for prediction in the correct order
data = [data_dict[feature] for feature in feature_columns]

# Convert to DataFrame for prediction ensuring column order matches
input_df = pd.DataFrame([data], columns=feature_columns)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_rounded = round(prediction, 2)
    st.write(f"The predicted flight fare is: {prediction_rounded}")

# To run the Streamlit app:
# Save this script and run in terminal:
# streamlit run <script_name>.py