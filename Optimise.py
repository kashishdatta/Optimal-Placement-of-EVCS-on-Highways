import math
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import folium

# Load and preprocess data
cars_data = pd.read_csv('EV_Cars.csv')
battery_data = pd.read_csv('car_battery_dataset_with_random_departure_time.csv')
traffic_data = pd.read_csv('highway_traffic_density_random.csv')
petrol_pump_data = pd.read_csv('density_pp.csv')

chargers_per_station = 4
total_distance = 990
working_minutes_per_day = 1440

def calculate_charging_time(car_type):
    charging_time = cars_data.loc[cars_data['Name of some of the long-range electric cars in India'] == car_type, 'Charging time (10% to 80%)'].values[0]
    return charging_time

def calculate_waiting_time():
    waiting_time = random.randint(5, 15)
    return waiting_time

def generate_training_data(num_samples):
    X = []
    y = []

    for _ in range(num_samples):
        num_cars = random.randint(100, 5000)
        total_charging_time = 0
        total_cars_supported = 0

        while total_charging_time < working_minutes_per_day:
            car_type = random.choice(cars_data['Name of some of the long-range electric cars in India'].tolist())
            charging_time = calculate_charging_time(car_type)
            waiting_time = calculate_waiting_time()

            total_time = charging_time + waiting_time

            if total_charging_time + total_time <= working_minutes_per_day:
                total_cars_supported += 1
                total_charging_time += total_time
            else:
                break

        total_chargers = math.ceil(num_cars / total_cars_supported)
        total_stations = math.ceil(total_chargers / chargers_per_station)

        X.append([num_cars])
        y.append(total_stations)

    return np.array(X), np.array(y)

# Generate training data
num_samples = 1000
X_train, y_train = generate_training_data(num_samples)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the number of EV cars
num_cars = 9000  # Number of cars to support
num_clusters = model.predict([[num_cars]])[0]
num_clusters = math.ceil(num_clusters)

# Perform K-means clustering on petrol pump locations
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(petrol_pump_data[[' latitude', ' longitude']])
petrol_pump_data['Cluster'] = kmeans.labels_

# Select one representative petrol pump from each cluster
representative_pumps = petrol_pump_data.groupby('Cluster').first().reset_index()

# Calculate the average distance between consecutive representative petrol pumps
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

distances = []
for i in range(len(representative_pumps) - 1):
    lat1, lon1 = representative_pumps.iloc[i][[' latitude', ' longitude']]
    lat2, lon2 = representative_pumps.iloc[i+1][[' latitude', ' longitude']]
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    distances.append(distance)

avg_distance = sum(distances) / len(distances)

# Create a map centered on the mean latitude and longitude of representative pumps
map_center = representative_pumps[[' latitude', ' longitude']].mean().values.tolist()
map_obj = folium.Map(location=map_center, zoom_start=6)

# Add markers for each representative petrol pump with address popup
for _, row in representative_pumps.iterrows():
    popup_text = f"Address: {row[' Address']}"
    folium.Marker(location=[row[' latitude'], row[' longitude']], popup=popup_text).add_to(map_obj)

# Print the number of clusters and average distance
print(f"Number of clusters (predicted petrol pumps): {num_clusters}")
print(f"Average distance between consecutive representative petrol pumps: {avg_distance:.2f} km")

# Save the map as an HTML file
map_obj.save('petrol_pump_map.html')