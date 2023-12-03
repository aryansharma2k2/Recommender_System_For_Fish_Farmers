import pandas as pd
import math
from scipy.spatial.distance import cdist
from statsmodels.tsa.arima.model import ARIMA

# Assuming you have the farmer inputs stored in variables lat, lon, num_ponds, pond_dimensions
lat = input("Enter latitude: ")
lon = input("Enter longitude: ")
num_ponds = int(input("Enter the number of ponds: "))

ponds = []
for i in range(num_ponds):
    pond_length = float(input(f"Enter length of pond {i+1}: "))
    pond_width = float(input(f"Enter width of pond {i+1}: "))
    ponds.append((pond_length, pond_width))

# Load weather data from Excel sheet
weather_data = pd.read_excel('/content/MERGED_KARNATAKA_FINAL (1).xlsx')  # Replace 'weather_data.xlsx' with your file name

# Assuming your weather data contains columns: 'Latitude', 'Longitude', 'Precipitation', 'Temperature', etc.

# Combine latitude and longitude from weather data into a single array
locations = weather_data[['LAT', 'LON']]

# Create an array from farmer input (lat, lon)
farmer_location = [(float(lat), float(lon))]

# Find the nearest location
nearest_idx = cdist(farmer_location, locations).argmin()

# Extract data for the nearest location
nearest_weather = weather_data.iloc[nearest_idx]

def calculate_pet(temperature):
    # Constants for Hargreaves method
    lat_rad = math.radians(nearest_weather['LAT'])  # Assuming you have 'nearest_weather' from previous code
    avg_temp = temperature.mean()  # Assuming 'temperature' is a pandas Series or array of daily temperatures
    temp_range = temperature.max() - temperature.min()

    # Calculate PET using the Hargreaves method
    const = 0.0023  # Constant value used in Hargreaves method
    pet = const * (temp_range ** 0.5) * (avg_temp + 17.8) * (math.sin(lat_rad) ** 0.5)

    return pet

weather_data['PET'] = calculate_pet(weather_data['TEMP'])

# Calculate evaporation loss rate
weather_data['Evaporation_Loss'] = weather_data['PET'] - weather_data['PRECIPITATION (MM/DAY)']

# Assuming surface area is constant for all ponds
surface_area = pond_length * pond_width  # Assuming you have pond_length and pond_width values

# Calculate net water loss for each row
weather_data['Net_Water_Loss'] = weather_data['Evaporation_Loss'] * surface_area

# Assuming 'Net_Water_Loss_Rate' is the column containing net water loss rate
# Fit ARIMA model
model = ARIMA(weather_data['Net_Water_Loss'], order=(1, 1, 1))  # Example order, adjust as needed
model_fit = model.fit()

# Forecast for 7 months (28 weeks)
forecast = model_fit.forecast(steps=28)

# Calculate weekly net water loss for 7 months for each pond
for i, pond in enumerate(ponds):
    surface_area = pond[0] * pond[1]
    weekly_net_water_loss = forecast * surface_area
    weather_data[f'Weekly_Net_Water_Loss_Pond_{i+1}'] = weekly_net_water_loss


for week_num, net_loss in enumerate(forecast, start=1):
    print(f'Week {week_num} - Net Water Loss:')
    for i, pond in enumerate(ponds, start=1):
        surface_area = pond[0] * pond[1]
        weekly_net_water_loss = abs(net_loss * surface_area)
        print(f'Pond {i}: {weekly_net_water_loss}')
    print('\n')

