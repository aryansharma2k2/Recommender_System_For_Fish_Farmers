from flask import Flask, render_template, request, session, redirect, url_for
import math
from math import radians, cos, sin, sqrt, atan2
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'aryan'
market_data = pd.read_csv('C:\Capstone\MarketPriceData.csv')
def get_pond_dimensions(num_ponds, pond_details):
    number_of_ponds = num_ponds
    total_volume = 0
    total_surface_area = 0
    surface_areas = []  # List to store surface area of each pond
    pond_volumes = []   # List to store volume of each pond

    for length, breadth, depth in pond_details:
        volume = length * breadth * depth
        surface_area = length * breadth

        total_volume += volume
        total_surface_area += surface_area
        surface_areas.append(surface_area)  # Add surface area of this pond to the list
        pond_volumes.append(volume)         # Add volume of this pond to the list

    return number_of_ponds, total_volume, total_surface_area, surface_areas, pond_volumes

# Function to convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees
def dms_to_dd(d, m, s):
    return d + float(m)/60 + float(s)/3600

# Haversine formula to calculate distance between two points on the earth
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat/2) * sin(dLat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2) * sin(dLon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

def get_farmer_location(user_lat, user_lon):
    # Locations of markets (converted to decimal degrees if necessary)
    # Example: KFDC: (13°02'13.8"N, 77°34'22.5"E)
    kfdc_lat = dms_to_dd(13, 2, 13.8)
    kfdc_lon = dms_to_dd(77, 34, 22.5)
    # Madikeri and Padubidri are already in decimal degrees
    madikeri_lat, madikeri_lon = 12.437043254998047, 75.72585748
    padubidri_lat = dms_to_dd(13, 8, 43.0)
    padubidri_lon = dms_to_dd(74, 46, 18.1)

    # Calculate distances
    distance_to_kfdc = haversine(user_lat, user_lon, kfdc_lat, kfdc_lon)
    distance_to_madikeri = haversine(user_lat, user_lon, madikeri_lat, madikeri_lon)
    distance_to_padubidri = haversine(user_lat, user_lon, padubidri_lat, padubidri_lon)

    # Assuming cost per km is given or can be calculated
    cost_per_km = 30  # 30 rupees per km
    cost_to_kfdc = distance_to_kfdc * cost_per_km
    cost_to_madikeri = distance_to_madikeri * cost_per_km
    cost_to_padubidri = distance_to_padubidri * cost_per_km

    return cost_to_kfdc, cost_to_madikeri, cost_to_padubidri

# Forecasting function
def forecast_price(series, months_ahead, start_month=0):
    adjusted_series = series.iloc[start_month:]
    model = ARIMA(adjusted_series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=months_ahead)
    return forecast.iloc[-1]

# Forecast for each market
def forecast_for_market(columns, months_ahead, start_month):
    return {col: forecast_price(market_data[col], months_ahead, start_month) for col in columns}

# Remove market suffixes
def remove_market_suffixes(forecast_dict, suffix):
    return {key.replace(suffix, ''): value for key, value in forecast_dict.items()}

# Market caps data
species_market_caps = {

    'Catla': {
        1: 600, 2: 600, 3: 600, 4: 550, 5: 400, 6: 400, 7: 400, 8: 400, 9: 700, 10: 1000, 11: 1000, 12: 1000
    },
    'Rohu': {
        1: 600, 2: 600, 3: 600, 4: 550, 5: 400, 6: 400, 7: 400, 8: 400, 9: 700, 10: 1000, 11: 1000, 12: 1000
    },
    'Mrigal': {
        1: 600, 2: 600, 3: 600, 4: 550, 5: 400, 6: 400, 7: 400, 8: 400, 9: 700, 10: 1000, 11: 1000, 12: 1000
    },
    'Tilapia': {
        1: 550, 2: 550, 3: 550, 4: 500, 5: 425, 6: 425, 7: 425, 8: 425, 9: 675, 10: 900, 11: 900, 12: 900
    },
    'AsianSeabass': {
        1: 500, 2: 500, 3: 500, 4: 425, 5: 350, 6: 350, 7: 350, 8: 350, 9: 575, 10: 800, 11: 800, 12: 800
    },
    'GiantFreshwaterPrawn': {
        1: 500, 2: 500, 3: 500, 4: 425, 5: 350, 6: 350, 7: 350, 8: 350, 9: 575, 10: 800, 11: 800, 12: 800
    }
}


# Complete forecasting process
def complete_forecasting_process(market_data):
    market_suffixes = {'kfdc': '_kfdc', 'madikeri': '_madikeri', 'padubidri': '_padubidri'}
    forecasts = {}

    for market, suffix in market_suffixes.items():
        market_columns = [col for col in market_data.columns if col.endswith(suffix)]
        for month in range(7, 10):
            forecast_key = f'forecast_{market}_{month}'
            forecasts[forecast_key] = forecast_for_market(market_columns, 7, month - 7)
            forecasts[forecast_key] = remove_market_suffixes(forecasts[forecast_key], suffix)

    combined_forecasts = []
    for key, forecast_dict in forecasts.items():
        market, month = key.split('_')[1], int(key.split('_')[2])
        for species, price in forecast_dict.items():
            combined_forecasts.append([species, market, month, price])

    sorted_forecasts = sorted(combined_forecasts, key=lambda x: x[3], reverse=True)

    # Standardize species names
    species_name_mapping = {
        'asianSeabass': 'AsianSeabass',
        'tilapia': 'Tilapia',
        'catla' : 'Catla',
        'rohu' : 'Rohu',
        'mrigal' : 'Mrigal',
        'giantFreshWaterPrawn' : 'GiantFreshWaterPrawn'
        # Add more mappings if needed
    }

    for forecast in sorted_forecasts:
        forecast[0] = species_name_mapping.get(forecast[0], forecast[0])

    # Calculate future month based on current date
    current_month = datetime.now().month
    def get_future_month(month, months_ahead):
        return ((month + months_ahead - 1) % 12) + 1

    # Append market cap to each forecast
    for forecast in sorted_forecasts:
        species, market, months_ahead, price = forecast
        future_month = get_future_month(current_month, months_ahead)
        market_cap = species_market_caps.get(species, {}).get(future_month, 0)
        forecast.append(market_cap)

    return sorted_forecasts


def calculate_fish_weights_and_logistics(user_lat, user_lon, truck_capacity, num_ponds, fishes_per_pond, ponddata):
    pond_weights = []
    total_weight_kg = 0

    growth_phases = {
        "Catla": [
            {"phase": "fry", "duration_months": 1, "weight": 0.0005},
            {"phase": "fingerling", "duration_months": 1.5, "weight": 0.01},
            {"phase": "juvenile", "duration_months": 1.5, "weight": 0.1},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.5},
            {"phase": "adult", "duration_months": 2, "weight": 1}
        ],
        "Rohu": [
            {"phase": "fry", "duration_months": 1, "weight": 0.0005},
            {"phase": "fingerling", "duration_months": 1.5, "weight": 0.01},
            {"phase": "juvenile", "duration_months": 1.5, "weight": 0.1},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.5},
            {"phase": "adult", "duration_months": 2, "weight": 1}
        ],
        "Mrigal": [
            {"phase": "fry", "duration_months": 1, "weight": 0.0005},
            {"phase": "fingerling", "duration_months": 1.5, "weight": 0.01},
            {"phase": "juvenile", "duration_months": 1.5, "weight": 0.1},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.5},
            {"phase": "adult", "duration_months": 2, "weight": 1}
        ],
        "Tilapia": [
            {"phase": "fry", "duration_months": 1, "weight": 0.0005},
            {"phase": "fingerling", "duration_months": 1.5, "weight": 0.01},
            {"phase": "juvenile", "duration_months": 1.5, "weight": 0.1},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.5},
            {"phase": "adult", "duration_months": 2, "weight": 1}
        ],
        "GiantFreshwaterPrawn": [
            {"phase": "larval", "duration_months": 1, "weight": 0.001},
            {"phase": "post_larval", "duration_months": 1, "weight": 0.005},
            {"phase": "juvenile", "duration_months": 2, "weight": 0.02},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.05},
            {"phase": "adult", "duration_months": 2, "weight": 0.2}
        ],
        "AsianSeabass": [
            {"phase": "larval", "duration_months": 1, "weight": 0.001},
            {"phase": "fry", "duration_months": 1, "weight": 0.005},
            {"phase": "fingerling", "duration_months": 1, "weight": 0.05},
            {"phase": "juvenile", "duration_months": 1, "weight": 0.5},
            {"phase": "sub_adult", "duration_months": 1.5, "weight": 1},
            {"phase": "adult", "duration_months": 1.5, "weight": 2}
        ]
    }

    # Calculate weight in each pond
    for i in range(num_ponds):
        species = ponddata[i][0]
        phases = growth_phases[species]
        weight_per_fish = sum([phase['weight'] for phase in phases])
        total_weight_per_pond = weight_per_fish * fishes_per_pond[i]
        pond_weights.append(total_weight_per_pond)
        total_weight_kg += total_weight_per_pond

    # Convert total weight to tonnes
    total_weight_tonnes = total_weight_kg / 1000


    num_trucks = math.ceil(total_weight_tonnes / truck_capacity)

    # Calculate total logistics costs
    kfdc_lat, kfdc_lon = dms_to_dd(13, 2, 13.8), dms_to_dd(77, 34, 22.5)
    madikeri_lat, madikeri_lon = 12.437043254998047, 75.72585748
    padubidri_lat, padubidri_lon = dms_to_dd(13, 8, 43.0), dms_to_dd(74, 46, 18.1)

    distance_to_kfdc = haversine(user_lat, user_lon, kfdc_lat, kfdc_lon)
    distance_to_madikeri = haversine(user_lat, user_lon, madikeri_lat, madikeri_lon)
    distance_to_padubidri = haversine(user_lat, user_lon, padubidri_lat, padubidri_lon)

    # Assuming cost per km
    cost_per_km = 30  # Example cost per km
    cost_to_kfdc = distance_to_kfdc * cost_per_km * 2 * num_trucks
    cost_to_madikeri = distance_to_madikeri * cost_per_km * 2 * num_trucks
    cost_to_padubidri = distance_to_padubidri * cost_per_km * 2 * num_trucks

    # Total logistics costs
    total_logistics_costs = {
        "KFDC": cost_to_kfdc,
        "Madikeri": cost_to_madikeri,
        "Padubidri": cost_to_padubidri
    }

    return pond_weights, total_weight_tonnes, num_trucks, total_logistics_costs

def revenue_recommendation_calculation(sorted_ponds, sorted_forecasts):
    revenue = []
    ponddata = []
    fishes_per_pond = [0] * len(sorted_ponds)  # Initialize with zeros for each pond
    number_of_ponds = len(sorted_ponds)

    for i in range(number_of_ponds):
        potential_revenues = []
        for j in range(len(sorted_forecasts)):
            species, market, months_ahead, price, market_cap = sorted_forecasts[j]
            m = min(sorted_ponds[i], market_cap)
            
            potential_revenue = price * m 
            potential_revenues.append(potential_revenue)

        # Find the index of the maximum potential revenue
        max_revenue_index = potential_revenues.index(max(potential_revenues))
        revenue.append(potential_revenues[max_revenue_index])
        ponddata.append(sorted_forecasts[max_revenue_index])

        # Calculate number of fish for this pond and update fishes_per_pond
        number_of_fish = min(sorted_ponds[i], sorted_forecasts[max_revenue_index][4])
        fishes_per_pond[i] += number_of_fish

        # Update the market capacity in sorted_forecasts
        if sorted_ponds[i] >= sorted_forecasts[max_revenue_index][4]:
            sorted_forecasts[max_revenue_index][4] = 0
        else:
            sorted_forecasts[max_revenue_index][4] -= number_of_fish

    return revenue, ponddata, fishes_per_pond



def calculate_feeding_costs_per_pond(ponddata,fishes_per_pond):
    # Growth phases and weights for each species
    growth_phases = {
        "Catla": [
            {"phase": "fry", "duration_months": 1, "weight": 0.0005},
            {"phase": "fingerling", "duration_months": 1.5, "weight": 0.01},
            {"phase": "juvenile", "duration_months": 1.5, "weight": 0.1},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.5},
            {"phase": "adult", "duration_months": 2, "weight": 1}
        ],
        "Rohu": [
            {"phase": "fry", "duration_months": 1, "weight": 0.0005},
            {"phase": "fingerling", "duration_months": 1.5, "weight": 0.01},
            {"phase": "juvenile", "duration_months": 1.5, "weight": 0.1},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.5},
            {"phase": "adult", "duration_months": 2, "weight": 1}
        ],
        "Mrigal": [
            {"phase": "fry", "duration_months": 1, "weight": 0.0005},
            {"phase": "fingerling", "duration_months": 1.5, "weight": 0.01},
            {"phase": "juvenile", "duration_months": 1.5, "weight": 0.1},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.5},
            {"phase": "adult", "duration_months": 2, "weight": 1}
        ],
        "Tilapia": [
            {"phase": "fry", "duration_months": 1, "weight": 0.0005},
            {"phase": "fingerling", "duration_months": 1.5, "weight": 0.01},
            {"phase": "juvenile", "duration_months": 1.5, "weight": 0.1},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.5},
            {"phase": "adult", "duration_months": 2, "weight": 1}
        ],
        "GiantFreshwaterPrawn": [
            {"phase": "larval", "duration_months": 1, "weight": 0.001},
            {"phase": "post_larval", "duration_months": 1, "weight": 0.005},
            {"phase": "juvenile", "duration_months": 2, "weight": 0.02},
            {"phase": "sub_adult", "duration_months": 1, "weight": 0.05},
            {"phase": "adult", "duration_months": 2, "weight": 0.2}
        ],
        "AsianSeabass": [
            {"phase": "larval", "duration_months": 1, "weight": 0.001},
            {"phase": "fry", "duration_months": 1, "weight": 0.005},
            {"phase": "fingerling", "duration_months": 1, "weight": 0.05},
            {"phase": "juvenile", "duration_months": 1, "weight": 0.5},
            {"phase": "sub_adult", "duration_months": 1.5, "weight": 1},
            {"phase": "adult", "duration_months": 1.5, "weight": 2}
        ]
    }

    # Cost of food per kg for each species
    cost_rates = {
    "Catla": 20,
    "Rohu": 20,
    "Mrigal": 20,
    "Tilapia": 25,
    "GiantFreshwaterPrawn": 25,
    "AsianSeabass": 25
    }   

    feeding_details_per_pond = []

    for entry, number_of_fish in zip(ponddata, fishes_per_pond):
        species = entry[0]
        months_ahead = entry[2]  # Assuming the 3rd element is months ahead
        phases = growth_phases[species]
        pond_feeding_details = []
        pond_total_cost = 0

        for phase in phases:
            feedings_per_day = 4 if (species.lower() == "giantfreshwaterprawn" and phase["phase"] in ["larval", "post_larval"]) \
                or (species.lower() == "asianseabass" and phase["phase"] in ["larval", "fry", "fingerling"]) \
                else 2

            daily_feed_per_fish = phase["weight"] * 0.05  # 5% of body weight
            monthly_feed_per_fish = daily_feed_per_fish * feedings_per_day * 30  # Total monthly feed per fish in kg
            phase_duration = min(phase["duration_months"], months_ahead)  # Adjust for the planning period
            monthly_cost = monthly_feed_per_fish * cost_rates[species] * number_of_fish  # Monthly cost for this phase
            pond_total_cost += monthly_cost * phase_duration

            pond_feeding_details.append({
                "Phase": phase["phase"],
                "Duration (Months)": phase_duration,
                "Feed per Day (kg)": daily_feed_per_fish * number_of_fish * feedings_per_day,
                "Feedings per Day": feedings_per_day,
                "Total Cost": monthly_cost
            })

        feeding_details_per_pond.append({
            "Pond": entry,
            "Feeding Details": pond_feeding_details,
            "Total Cost": pond_total_cost
        })

    return feeding_details_per_pond


def generate_chemical_readings(temperature, ph_level, salinity, dissolved_oxygen, start_date, num_ponds, year_duration=1):
    x = timedelta(days=365)
    end_date = start_date + x
    date_range = pd.date_range(start=start_date, end=end_date, freq='2D')

    # Ask user for initial readings
    initial_pH = ph_level
    initial_temperature = temperature
    initial_salinity = salinity
    initial_do = dissolved_oxygen

    for pond_number in range(1, num_ponds + 1):
        data = {
            'Date': date_range,
            'pH': np.random.normal(loc=initial_pH, scale=0.1, size=len(date_range)),
            'Temperature': np.random.normal(loc=initial_temperature, scale=1, size=len(date_range)),
            'Salinity': np.random.normal(loc=initial_salinity, scale=0.1, size=len(date_range)),
            'Dissolved_Oxygen': np.random.normal(loc=initial_do, scale=0.1, size=len(date_range))
        }

        # Introduce specific conditions every 4 weeks
        for i in range(0, len(date_range), 28):
            data['pH'][i] = 5.3  # Example specific condition
            data['Salinity'][i] = 6  # Example specific condition
            data['Dissolved_Oxygen'][i] = 3  # Example specific condition

        df = pd.DataFrame(data)
        df.to_csv(f'pond_{pond_number}_chemical_readings.csv', index=False)

def analyze_and_calculate_chemical_costs(num_ponds, pond_volumes, start_date, year_duration=1):
    weeks_in_7_months = 7 * 4
    lime_cost_per_tonne = 6000
    lime_amount_per_m3 = 1  # 2 kg per cubic meter when pH dips below 6
    ammonia_remover_cost_per_kg = 300
    fish_medicine_cost_per_bottle = 500
    fish_medicine_bottles_per_week = 3

    total_lime_cost = 0
    total_ammonia_remover_cost = weeks_in_7_months * num_ponds * ammonia_remover_cost_per_kg
    total_fish_medicine_cost = weeks_in_7_months * num_ponds * fish_medicine_bottles_per_week * fish_medicine_cost_per_bottle

    chemical_costs_per_pond = []
    y = timedelta(days=365)
    end_date = start_date + y
    date_range = pd.date_range(start=start_date, end=end_date, freq='2D')

    for pond_number in range(1, num_ponds + 1):
        readings_file = f'pond_{pond_number}_chemical_readings.csv'
        df = pd.read_csv(readings_file)

        lime_needed_for_pond = 0
        for index, row in df.iterrows():
            if row['pH'] < 6:
                lime_needed_for_pond += lime_amount_per_m3  # Add lime for each instance pH drops below 6

        lime_cost_for_pond = (lime_needed_for_pond * pond_volumes[pond_number - 1] / 1000) * lime_cost_per_tonne
        total_lime_cost += lime_cost_for_pond

        chemical_cost_for_pond = lime_cost_for_pond + (ammonia_remover_cost_per_kg * weeks_in_7_months) + (fish_medicine_cost_per_bottle * fish_medicine_bottles_per_week * weeks_in_7_months) 
        chemical_costs_per_pond.append(chemical_cost_for_pond)

    total_cost = total_lime_cost  + total_ammonia_remover_cost + total_fish_medicine_cost

    total_chemical_cost = sum(chemical_costs_per_pond)

    total_chemical_cost = total_chemical_cost 

    return total_chemical_cost, chemical_costs_per_pond
    


def get_harvest_cost(num_ponds, ponddata, fishes_per_pond):
    harvest_cost = []
    seed_prices = {
        'AsianSeabass': 2,
        'GiantFreshwaterPrawn': 0.2,
        'Catla': 0.5,
        'Mrigal': 0.5,
        'Rohu': 0.5,
        'Tilapia': 0.5
    }

    for i in range(num_ponds):
        species = ponddata[i][0]  # Fetch the species name
        print(species)
        cost_per_fish = seed_prices.get(species, 0)  # Get the cost per fish, default to 0 if not found
        total_cost = cost_per_fish * fishes_per_pond[i]  # Calculate total cost for this pond
        harvest_cost.append(total_cost)

    return harvest_cost

def additional_expenses(surface_areas):
    m2_to_sqft = 10.7639  # 1 square meter = 10.7639 square feet
    net_cost_per_sqft = 100

    # Convert surface area to sqft and calculate net costs
    total_net_area_sqft = sum(area * m2_to_sqft for area in surface_areas)
    net_cost = total_net_area_sqft * net_cost_per_sqft

    # Ask for labor salary budget
    labor_salary = 10,000

    # Display and return total costs
    total_costs = net_cost + labor_salary 
    print(f"Net Costs: {net_cost}")
    print(f"Labor Salary: {labor_salary}")

    return total_costs

def final_recommendations(revenue, ponddata, fishes_per_pond, feeding_costs, logistics_costs, chemical_costs_per_pond, harvest_costs):

    profit = 0
    total_finalcost = 0
    per_pond_profits = []
    per_pond_final_costs = []
    total_chemical_cost = sum(chemical_costs_per_pond)
    total_chemical_cost = total_chemical_cost - 300000

    for i in range(len(ponddata)):
        # Calculate total final cost for each pond
        total_cost_per_pond = feeding_costs[i]['Total Cost'] + logistics_costs[i] + total_chemical_cost + harvest_costs[i]
        per_pond_final_costs.append(total_cost_per_pond)

        # Calculate profit for each pond
        profit_per_pond = revenue[i] - total_cost_per_pond
        per_pond_profits.append(profit_per_pond)

        # Summing up for total costs and profits
        total_finalcost += total_cost_per_pond
        profit += profit_per_pond

        # Printing out recommendations
        print(f"Pond {i+1} Recommendations:")
        print(f"  Revenue: {revenue[i]}")
        print(f"  Total Cost: {total_finalcost}")
        print(f"  Profit: {profit}")

        # Instruction based on month number
        month_number = ponddata[i][2]
        if month_number == 7:
            print("  Start operations today.")
        elif month_number == 8:
            print("  Start operations next month.")
        elif month_number == 9:
            print("  Start operations in two months.")
        print("\n")
    return profit, total_finalcost, revenue


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    global stock_allocation, forecasted_prices
    try:
        num_ponds = int(request.form.get('num_ponds', 0))
        ph=float(request.form.get('ph',0))
        dissolvedoxygen=float(request.form.get('dissolvedoxygen',0))
        salinity=float(request.form.get('salinity'))
        temperature= int(request.form.get(f'temperature', 0))
        truck_capacity= int(request.form.get(f'truck_capacity', 0))


    except ValueError:
        num_ponds = 0

    pond_details = []
    total_surface_area = 0

    for i in range(num_ponds):
        try:
            length = float(request.form.get(f'length_{i+1}', 0))
            breadth = float(request.form.get(f'breadth_{i+1}', 0))
            depth = float(request.form.get(f'depth_{i+1}', 0))
        except ValueError:
            length, breadth, depth = 0, 0, 0

        pond_details.append((length, breadth, depth))
    
    num_ponds, total_vol, total_sa, sa_per_pond, volumes_per_pond = get_pond_dimensions(num_ponds, pond_details)
    #session['total_surface_area'] = total_surface_area

    try:
        user_lat = float(request.form.get('farmer_lat', 0))
        user_lon = float(request.form.get('farmer_lon', 0))
    except ValueError:
        farmer_lat, farmer_lon = 0, 0
    truck_capacity= int(request.form.get(f'truck_capacity', 0))
    ph= float(request.form.get(f'truck_capacity', 0))
    salinity=int(request.form.get(f'truck_capacity', 0))
    temperature = float(request.form.get(f'temperature', 0))
    dissolved_oxygen=int(request.form.get(f'truck_capacity', 0))
    session['ponds'] = num_ponds
    session['pond_length'] = length
    session['pond_width'] = breadth
    session['pond_depth']=depth
    ponds = [area * 3 for area in sa_per_pond]
    sorted_ponds = sorted(ponds, reverse=True)
    sorted_forecasts = complete_forecasting_process(market_data)
    revenue, ponddata, fishes_per_pond = revenue_recommendation_calculation(sorted_ponds, sorted_forecasts)
    pond_weights, total_weight_tonnes, num_trucks, logistics_costs = calculate_fish_weights_and_logistics(user_lat, user_lon, truck_capacity, num_ponds, fishes_per_pond, ponddata)
    feeding_costs = calculate_feeding_costs_per_pond(ponddata, fishes_per_pond)


    start_date = datetime(year = 2023, month = 1, day=30)
    generate_chemical_readings(temperature, ph, salinity, dissolved_oxygen, start_date, num_ponds, year_duration=1)
    total_chemical_cost, chemical_costs_per_pond = analyze_and_calculate_chemical_costs(num_ponds, volumes_per_pond, start_date)
    harvest_costs = get_harvest_cost(num_ponds, ponddata, fishes_per_pond)
    profit, total_finalcost, revenue = final_recommendations(revenue, ponddata, fishes_per_pond, feeding_costs, logistics_costs, chemical_costs_per_pond, harvest_costs)

    
    return render_template('result.html', num_ponds=num_ponds, ponds=ponds,
                               user_lat=user_lat, user_lon=user_lon,
                               total_surface_area=total_surface_area,
                               revenue=revenue, ponddata = ponddata, fishes_per_pond = fishes_per_pond,pond_weights=pond_weights, total_weight_tonnes=total_weight_tonnes, num_trucks=num_trucks, logistics_costs=logistics_costs,
                               market_data=market_data, sorted_forecasts=sorted_forecasts,
                               feeding_costs=feeding_costs, start_date=start_date,total_chemical_cost=total_chemical_cost,harvest_costs=harvest_costs,
                               profit=profit, total_finalcost =  total_finalcost)










if __name__ == '__main__':
    app.run(debug=True)
