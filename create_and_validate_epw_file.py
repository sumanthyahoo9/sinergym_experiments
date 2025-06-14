import pandas as pd
from get_location_coordinates import get_coordinates
from get_weather_data import get_uk_weather_data

def create_epw_file(weather_data, lat, lon, output_filename="Macclesfield_UK_2023.epw"):
    """
    Convert weather data to EPW format
    """
    
    # Use Open-Meteo data as primary source (most complete)
    if 'meteo' not in weather_data:
        raise ValueError("No suitable weather data available")
    
    meteo = weather_data['meteo']
    hourly = meteo['hourly']
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly['time']),
        'dry_bulb_temp': hourly['temperature_2m'],  # °C
        'relative_humidity': hourly['relative_humidity_2m'],  # %
        'pressure': hourly['pressure_msl'],  # hPa
        'wind_speed': hourly['windspeed_10m'],  # m/s
        'wind_direction': hourly['winddirection_10m'],  # degrees
        'global_horizontal_radiation': hourly['shortwave_radiation'],  # W/m²
        'direct_normal_radiation': hourly['direct_radiation'],  # W/m²
        'diffuse_horizontal_radiation': hourly['diffuse_radiation'],  # W/m²
        'dewpoint_temp': hourly['dewpoint_2m']  # °C
    })
    
    print(f"Processing {len(df)} hours of weather data...")
    
    # Calculate additional EPW variables
    df['atmospheric_pressure'] = df['pressure'] * 100  # Convert hPa to Pa
    df['horizontal_infrared_radiation'] = 400  # Typical value W/m²
    df['aerosol_optical_depth'] = 0.15  # Typical UK value
    
    # EPW requires specific format
    epw_header = create_epw_header(lat, lon)
    epw_data = create_epw_data_section(df)
    
    # Write EPW file
    with open(output_filename, 'w') as f:
        # Write header lines
        for line in epw_header:
            f.write(line + '\n')
        
        # Write data
        for line in epw_data:
            f.write(line + '\n')
    
    print(f"✅ EPW file created: {output_filename}")
    return output_filename

def create_epw_header(lat, lon):
    """Create EPW header lines"""
    location_name = "Macclesfield"
    country = "GBR"
    time_zone = 0  # UTC
    elevation = 100  # meters (approximate for Macclesfield)
    
    header_lines = [
        f"LOCATION,{location_name},,{country},TMY3,,,{lat:.2f},{lon:.2f},{time_zone:.1f},{elevation:.1f}",
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,Generated from Open-Meteo data for Macclesfield UK",
        "COMMENTS 2,Created for Sinergym RL training",
        "DATA PERIODS,1,1,Data,Sunday,1/1,12/31"
    ]
    
    return header_lines

def create_epw_data_section(df):
    """Convert DataFrame to EPW data format"""
    epw_lines = []
    
    for idx, row in df.iterrows():
        dt = row['datetime']
        
        # EPW format: Year,Month,Day,Hour,Minute,Data Source and Uncertainty Flags,
        # Dry Bulb Temperature,Dew Point Temperature,Relative Humidity,Atmospheric Station Pressure,
        # Extraterrestrial Horizontal Radiation,Extraterrestrial Direct Normal Radiation,
        # Horizontal Infrared Radiation Intensity,Global Horizontal Radiation,
        # Direct Normal Radiation,Diffuse Horizontal Radiation,Global Horizontal Illuminance,
        # Direct Normal Illuminance,Diffuse Horizontal Illuminance,Zenith Luminance,
        # Wind Direction,Wind Speed,Total Sky Cover,Opaque Sky Cover,Visibility,
        # Ceiling Height,Present Weather Observation,Present Weather Codes,
        # Precipitable Water,Aerosol Optical Depth,Snow Depth,Days Since Last Snowfall,
        # Albedo,Liquid Precipitation Depth,Liquid Precipitation Quantity
        
        line = (
            f"{dt.year},{dt.month},{dt.day},{dt.hour},0,"
            f"?9?9?9?9E0?9?9?9?9?9?9?9?9?9?9?9?9?9?9?9*9*9?9?9?9,"
            f"{row['dry_bulb_temp']:.1f},"  # Dry bulb temp
            f"{row['dewpoint_temp']:.1f},"  # Dew point temp
            f"{row['relative_humidity']:.0f},"  # Relative humidity
            f"{row['atmospheric_pressure']:.0f},"  # Pressure
            f"0,0,"  # Extraterrestrial radiation (calculated if needed)
            f"{row['horizontal_infrared_radiation']:.0f},"  # IR radiation
            f"{row['global_horizontal_radiation']:.0f},"  # Global horizontal
            f"{row['direct_normal_radiation']:.0f},"  # Direct normal
            f"{row['diffuse_horizontal_radiation']:.0f},"  # Diffuse horizontal
            f"0,0,0,0,"  # Illuminance values (set to 0)
            f"{row['wind_direction']:.0f},"  # Wind direction
            f"{row['wind_speed']:.1f},"  # Wind speed
            f"0,0,99999,99999,9,999999999,0,"  # Sky cover, visibility, weather
            f"{row['aerosol_optical_depth']:.3f},0,88,0.2,0,0"  # Additional parameters
        )
        
        epw_lines.append(line)
    
    return epw_lines

def validate_epw_file(filename):
    """Basic validation of EPW file"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        print(f"📊 EPW File Validation: {filename}")
        print(f"Total lines: {len(lines)}")
        
        # Check header (first 8 lines)
        header_lines = lines[:8]
        print(f"Header lines: {len(header_lines)}")
        
        # Check data lines (should be 8760 for full year)
        data_lines = lines[8:]
        print(f"Data lines: {len(data_lines)}")
        
        if len(data_lines) == 8760:
            print("✅ Complete year of hourly data")
        else:
            print(f"⚠️  Expected 8760 hours, got {len(data_lines)}")
        
        # Show sample data
        print("\n📋 Sample EPW data:")
        for i, line in enumerate(data_lines[:3]):
            print(f"Hour {i+1}: {line.strip()[:100]}...")
        
        print("\n✅ EPW file validation complete!")
        return True
        
    except Exception as e:
        print(f"❌ EPW validation failed: {e}")
        return False


# Create and validte EPW file
if __name__ == "__main__":
    # Download weather data for a particular location
    address = "Charter Way, Macclesfield SK10 2NA, United Kingdom"
    lat, lon = get_coordinates(address)
    weather_data = get_uk_weather_data(lat, lon, year=2023)
    epw_filename = create_epw_file(weather_data, lat, lon)
    validate_epw_file(epw_filename)
    
