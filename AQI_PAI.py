import requests
import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

def get_cordinates(query):

    url = "https://us1.locationiq.com/v1/search"
    params = {
        'key': 'pk.f35ace94576aac6b9b1ce7dca1a390a4',
        'q': f'{query}',
        'format': 'json',
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            lat = data[0]['lat']
            lon = data[0]['lon']
            print(f"Latitude: {lat}, Longitude: {lon}")
            return lat, lon
        else:
            print("No results found.")
            return None, None
        # print(data)
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)


def get_AQI(query):

    latitude, longitude=get_cordinates(query)
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude":longitude,
        "hourly": ["pm10", "pm2_5", "nitrogen_dioxide", "sulphur_dioxide"],
        "past_days": 1,
        "forecast_days": 1
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
    hourly_nitrogen_dioxide = hourly.Variables(2).ValuesAsNumpy()
    hourly_sulphur_dioxide = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["pm10"] = hourly_pm10
    hourly_data["pm2_5"] = hourly_pm2_5
    hourly_data["nitrogen_dioxide"] = hourly_nitrogen_dioxide
    hourly_data["sulphur_dioxide"] = hourly_sulphur_dioxide

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    # Convert the 'date' column to just the date part for grouping
    hourly_dataframe['date'] = hourly_dataframe['date'].dt.date

    # Group by date and calculate the daily mean for each pollutant
    daily_df = hourly_dataframe.groupby('date')[['pm10', 'pm2_5', 'nitrogen_dioxide', 'sulphur_dioxide']].mean().reset_index()
    # print(daily_df)
    return daily_df
    # print(hourly_dataframe)
    return hourly_dataframe


# get_AQI('manipal')