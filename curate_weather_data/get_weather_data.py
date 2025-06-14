import urllib
import json
from get_location_coordinates import get_coordinates

def get_uk_weather_data(lat, lon, year=2023):
    """
    Get weather data from multiple UK sources
    """
    weather_data = {}
    
    # Source 1: PVGIS (European Commission)
    print("Fetching data from PVGIS...")
    pvgis_url = "https://re.jrc.ec.europa.eu/api/tmy"
    pvgis_params = {
        'lat': lat,
        'lon': lon,
        'outputformat': 'json'
    }
    
    try:
        query_string = urllib.parse.urlencode(pvgis_params)
        full_url = f"{pvgis_url}?{query_string}"
        
        with urllib.request.urlopen(full_url) as response:
            pvgis_data = json.loads(response.read().decode())
        weather_data['pvgis'] = pvgis_data
        print("✅ PVGIS data obtained")
    except Exception as e:
        print(f"❌ PVGIS data failed: {e}")
    
    # Source 2: Open-Meteo (Historical weather)
    print("Fetching data from Open-Meteo...")
    meteo_url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Get full year of hourly data
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    meteo_params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ','.join([
            'temperature_2m',
            'relative_humidity_2m',
            'dewpoint_2m',
            'pressure_msl',
            'surface_pressure',
            'cloudcover',
            'windspeed_10m',
            'winddirection_10m',
            'shortwave_radiation',
            'direct_radiation',
            'diffuse_radiation'
        ]),
        'timezone': 'Europe/London'
    }
    
    try:
        query_string = urllib.parse.urlencode(meteo_params)
        full_url = f"{meteo_url}?{query_string}"
        
        with urllib.request.urlopen(full_url) as response:
            meteo_data = json.loads(response.read().decode())
        weather_data['meteo'] = meteo_data
        print("✅ Open-Meteo data obtained")
    except Exception as e:
        print(f"❌ Open-Meteo data failed: {e}")
    
    return weather_data

if __name__ == "__main__":
    # Download weather data for a particular location
    address = "Charter Way, Macclesfield SK10 2NA, United Kingdom"
    lat, lon = get_coordinates(address)
    weather_data = get_uk_weather_data(lat, lon, year=2023)
    print("The weather data is a dictionary with the following keys:")
    for key in weather_data.keys():
        print(key)