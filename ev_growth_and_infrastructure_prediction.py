import math
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load and preprocess data
cars_data = pd.read_csv('EV_Cars.csv')
battery_data = pd.read_csv('car_battery_dataset_with_random_departure_time.csv')
traffic_data = pd.read_csv('highway_traffic_density_random.csv')
petrol_pump_data = pd.read_csv('ev_charger_traffic_with_distance.csv')

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

# Evaluate the model
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Predict the number of EV charging stations required from 2020 to 2030
predicted_evs = [3927, 14603, 54200, 199839, 719310, 2386787, 6331517, 11391532, 14507588, 15658777, 16000000]
years = list(range(2020, 2031))
required_stations = []

for year, num_cars in zip(years, predicted_evs):
    stations = model.predict([[num_cars]])[0]
    required_stations.append(math.ceil(stations))
    print(f"Year: {year}, Predicted EVs: {num_cars}, Required EV Stations: {math.ceil(stations)}")

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(years, required_stations, marker='o')
plt.xlabel('Year')
plt.ylabel('Required EV Stations')
plt.title('Predicted Number of EV Charging Stations Required (2020-2030)')
plt.xticks(years, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()