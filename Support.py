import pandas as pd
import numpy as np
import math
import random

# Load and preprocess data
cars_data = pd.read_csv('EV_Cars.csv')
battery_data = pd.read_csv('car_battery_dataset_with_random_departure_time.csv')
traffic_data = pd.read_csv('highway_traffic_density_random.csv')
petrol_pump_data = pd.read_csv('ev_charger_traffic_with_distance.csv')

total_stations = 228
chargers_per_station = 4
total_chargers = total_stations * chargers_per_station
total_distance = 990
working_minutes_per_day = 1440

def calculate_charging_time(car_type):
    charging_time = cars_data.loc[cars_data['Name of some of the long-range electric cars in India'] == car_type, 'Charging time (10% to 80%)'].values[0]
    return charging_time

def calculate_waiting_time():
    waiting_time = random.randint(5, 30)
    return waiting_time

def calculate_total_cars_supported():
    total_cars = 0
    total_charging_time = 0

    while total_charging_time < working_minutes_per_day:
        car_type = random.choice(cars_data['Name of some of the long-range electric cars in India'].tolist())
        charging_time = calculate_charging_time(car_type)
        waiting_time = calculate_waiting_time()

        total_time = charging_time + waiting_time

        if total_charging_time + total_time <= working_minutes_per_day:
            total_cars += 1
            total_charging_time += total_time
        else:
            break

    return total_cars

total_cars_supported = calculate_total_cars_supported() * total_chargers
print(f"Total cars supported by {total_stations} EV stations: {total_cars_supported}")