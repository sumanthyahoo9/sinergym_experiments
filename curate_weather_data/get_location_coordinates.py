import urllib.request as requests
import urllib.parse as parse
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

def get_coordinates(address):
    """Get lat/lon for the address"""
    # Using OpenStreetMap Nominatim (free)
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'limit': '1'
    }
    
    # Encode parameters
    query_string = parse.urlencode(params)
    url = f"{base_url}?{query_string}"
    
    # Make request
    with requests.urlopen(url) as response:
        data = json.loads(response.read().decode())
    
    if data:
        lat = float(data[0]['lat'])
        lon = float(data[0]['lon'])
        print(f"Location: {data[0]['display_name']}")
        print(f"Coordinates: {lat:.4f}, {lon:.4f}")
        return lat, lon
    else:
        raise ValueError("Address not found")

if __name__ == "__main__":
    # Get coordinates for Macclesfield
    address = "Charter Way, Macclesfield SK10 2NA, United Kingdom"
    lat, lon = get_coordinates(address)