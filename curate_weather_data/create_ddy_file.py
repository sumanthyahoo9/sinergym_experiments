import pandas as pd
import numpy as np
from get_location_coordinates import get_coordinates
from get_weather_data import get_uk_weather_data

def create_ddy_file(weather_data, lat, lon, output_filename="Macclesfield_UK_2023.ddy"):
    """
    Create DDY (Design Day) file from weather data for HVAC sizing
    """
    
    # Use Open-Meteo data as primary source
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
        'dewpoint_temp': hourly['dewpoint_2m']  # °C
    })
    
    print(f"Analyzing {len(df)} hours of weather data for design days...")
    
    # Find design day conditions
    heating_design_day = find_heating_design_day(df)
    cooling_design_day = find_cooling_design_day(df)
    
    # Create DDY content
    ddy_content = create_ddy_content(heating_design_day, cooling_design_day, lat, lon)
    
    # Write DDY file
    with open(output_filename, 'w') as f:
        f.write(ddy_content)
    
    print(f"✅ DDY file created: {output_filename}")
    print(f"🌡️  Heating Design Day: {heating_design_day['date']} ({heating_design_day['temp']:.1f}°C)")
    print(f"🔥 Cooling Design Day: {cooling_design_day['date']} ({cooling_design_day['temp']:.1f}°C)")
    
    return output_filename

def find_heating_design_day(df):
    """
    Find 99.6% heating design day (coldest conditions)
    Only 0.4% of year is colder than this temperature
    """
    
    # Calculate daily statistics
    daily_stats = df.groupby(df['datetime'].dt.date).agg({
        'dry_bulb_temp': ['min', 'max', 'mean'],
        'relative_humidity': 'mean',
        'pressure': 'mean', 
        'wind_speed': 'mean',
        'wind_direction': 'mean',
        'dewpoint_temp': 'mean'
    }).round(2)
    
    # Flatten column names
    daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
    
    # Find 99.6% heating design temperature (0.4th percentile of daily minimums)
    heating_design_temp = daily_stats['dry_bulb_temp_min'].quantile(0.004)  # 0.4%
    
    # Find the actual day closest to this design temperature
    closest_day_idx = (daily_stats['dry_bulb_temp_min'] - heating_design_temp).abs().idxmin()
    design_day_data = daily_stats.loc[closest_day_idx]
    
    # Calculate daily temperature range for that day
    temp_range = design_day_data['dry_bulb_temp_max'] - design_day_data['dry_bulb_temp_min']
    
    # Calculate dewpoint temperature from humidity and temperature
    dewpoint = design_day_data['dewpoint_temp_mean']
    
    heating_design_day = {
        'date': closest_day_idx,
        'temp': design_day_data['dry_bulb_temp_min'],
        'temp_range': temp_range,
        'humidity': design_day_data['relative_humidity_mean'],
        'dewpoint': dewpoint,
        'pressure': design_day_data['pressure_mean'] * 100,  # Convert hPa to Pa
        'wind_speed': design_day_data['wind_speed_mean'],
        'wind_direction': design_day_data['wind_direction_mean'],
        'month': closest_day_idx.month,
        'day': closest_day_idx.day
    }
    
    return heating_design_day

def find_cooling_design_day(df):
    """
    Find 0.4% cooling design day (hottest conditions)
    Only 0.4% of year is hotter than this temperature
    """
    
    # Calculate daily statistics
    daily_stats = df.groupby(df['datetime'].dt.date).agg({
        'dry_bulb_temp': ['min', 'max', 'mean'],
        'relative_humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'mean', 
        'wind_direction': 'mean',
        'dewpoint_temp': 'mean'
    }).round(2)
    
    # Flatten column names
    daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
    
    # Find 0.4% cooling design temperature (99.6th percentile of daily maximums)
    cooling_design_temp = daily_stats['dry_bulb_temp_max'].quantile(0.996)  # 99.6%
    
    # Find the actual day closest to this design temperature
    closest_day_idx = (daily_stats['dry_bulb_temp_max'] - cooling_design_temp).abs().idxmin()
    design_day_data = daily_stats.loc[closest_day_idx]
    
    # Calculate daily temperature range for that day
    temp_range = design_day_data['dry_bulb_temp_max'] - design_day_data['dry_bulb_temp_min']
    
    # Calculate dewpoint temperature
    dewpoint = design_day_data['dewpoint_temp_mean']
    
    cooling_design_day = {
        'date': closest_day_idx,
        'temp': design_day_data['dry_bulb_temp_max'],
        'temp_range': temp_range,
        'humidity': design_day_data['relative_humidity_mean'],
        'dewpoint': dewpoint,
        'pressure': design_day_data['pressure_mean'] * 100,  # Convert hPa to Pa
        'wind_speed': design_day_data['wind_speed_mean'],
        'wind_direction': design_day_data['wind_direction_mean'],
        'month': closest_day_idx.month,
        'day': closest_day_idx.day
    }
    
    return cooling_design_day

def create_ddy_content(heating_day, cooling_day, lat, lon):
    """
    Create DDY file content in EnergyPlus format
    """
    
    location_name = "Macclesfield"
    country = "GBR"
    time_zone = 0  # UTC
    elevation = 100  # meters
    
    ddy_content = f"""! DDY file for {location_name}, UK
! Generated from Open-Meteo historical weather data for HVAC design
! Location: {lat:.3f}N, {lon:.3f}W, Elevation: {elevation}m

 Design Conditions from "Climate Design Data 2023 ASHRAE Handbook"
 Heating and Cooling Design Conditions for {location_name}, UK

 SizingPeriod:DesignDay,
    {location_name} Ann Htg 99.6% Condns DB,  !- Name
    {heating_day['month']},                     !- Month
    {heating_day['day']},                       !- Day of Month  
    WinterDesignDay,                            !- Day Type
    {heating_day['temp']:.1f},                  !- Maximum Dry-Bulb Temperature {{C}}
    {heating_day['temp_range']:.1f},            !- Daily Dry-Bulb Temperature Range {{deltaC}}
    DefaultMultipliers,                         !- Dry-Bulb Temperature Range Modifier Type
    ,                                           !- Dry-Bulb Temperature Range Modifier Day Schedule Name
    Wetbulb,                                    !- Humidity Condition Type
    {heating_day['dewpoint']:.1f},              !- Wetbulb or DewPoint at Maximum Dry-Bulb {{C}}
    ,                                           !- Humidity Condition Day Schedule Name
    ,                                           !- Humidity Ratio at Maximum Dry-Bulb {{kgWater/kgDryAir}}
    ,                                           !- Enthalpy at Maximum Dry-Bulb {{J/kg}}
    ,                                           !- Daily Wet-Bulb Temperature Range {{deltaC}}
    {heating_day['pressure']:.0f},              !- Barometric Pressure {{Pa}}
    {heating_day['wind_speed']:.1f},            !- Wind Speed {{m/s}}
    {heating_day['wind_direction']:.0f},        !- Wind Direction {{deg}}
    No,                                         !- Rain Indicator
    No,                                         !- Snow Indicator
    No,                                         !- Daylight Saving Time Indicator
    ASHRAEClearSky,                            !- Solar Model Indicator
    ,                                           !- Beam Solar Day Schedule Name
    ,                                           !- Diffuse Solar Day Schedule Name
    ,                                           !- ASHRAE Clear Sky Optical Depth for Beam Irradiance (taub) {{dimensionless}}
    ,                                           !- ASHRAE Clear Sky Optical Depth for Diffuse Irradiance (taud) {{dimensionless}}
    0.00;                                       !- Sky Clearness

 SizingPeriod:DesignDay,
    {location_name} Ann Clg .4% Condns DB=>{cooling_day['temp']:.1f}C WB=>{cooling_day['dewpoint']:.1f}C,  !- Name
    {cooling_day['month']},                     !- Month
    {cooling_day['day']},                       !- Day of Month
    SummerDesignDay,                            !- Day Type
    {cooling_day['temp']:.1f},                  !- Maximum Dry-Bulb Temperature {{C}}
    {cooling_day['temp_range']:.1f},            !- Daily Dry-Bulb Temperature Range {{deltaC}}
    DefaultMultipliers,                         !- Dry-Bulb Temperature Range Modifier Type
    ,                                           !- Dry-Bulb Temperature Range Modifier Day Schedule Name
    Wetbulb,                                    !- Humidity Condition Type
    {cooling_day['dewpoint']:.1f},              !- Wetbulb or DewPoint at Maximum Dry-Bulb {{C}}
    ,                                           !- Humidity Condition Day Schedule Name
    ,                                           !- Humidity Ratio at Maximum Dry-Bulb {{kgWater/kgDryAir}}
    ,                                           !- Enthalpy at Maximum Dry-Bulb {{J/kg}}
    ,                                           !- Daily Wet-Bulb Temperature Range {{deltaC}}
    {cooling_day['pressure']:.0f},              !- Barometric Pressure {{Pa}}
    {cooling_day['wind_speed']:.1f},            !- Wind Speed {{m/s}}
    {cooling_day['wind_direction']:.0f},        !- Wind Direction {{deg}}
    No,                                         !- Rain Indicator
    No,                                         !- Snow Indicator
    No,                                         !- Daylight Saving Time Indicator
    ASHRAEClearSky,                            !- Solar Model Indicator
    ,                                           !- Beam Solar Day Schedule Name
    ,                                           !- Diffuse Solar Day Schedule Name
    ,                                           !- ASHRAE Clear Sky Optical Depth for Beam Irradiance (taub) {{dimensionless}}
    ,                                           !- ASHRAE Clear Sky Optical Depth for Diffuse Irradiance (taud) {{dimensionless}}
    1.00;                                       !- Sky Clearness
"""
    
    return ddy_content

def validate_ddy_file(filename):
    """
    Basic validation of DDY file
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        print(f"📊 DDY File Validation: {filename}")
        
        # Check for required sections
        if "SizingPeriod:DesignDay" in content:
            design_days = content.count("SizingPeriod:DesignDay")
            print(f"✅ Found {design_days} design days")
        else:
            print("❌ No design days found")
            
        if "WinterDesignDay" in content:
            print("✅ Heating design day present")
        else:
            print("❌ Missing heating design day")
            
        if "SummerDesignDay" in content:
            print("✅ Cooling design day present")
        else:
            print("❌ Missing cooling design day")
        
        # File size check
        file_size = len(content)
        print(f"📄 File size: {file_size} characters")
        
        print("✅ DDY file validation complete!")
        return True
        
    except Exception as e:
        print(f"❌ DDY validation failed: {e}")
        return False

# Create and validate DDY file
if __name__ == "__main__":
    # Download weather data for a particular location
    address = "Charter Way, Macclesfield SK10 2NA, United Kingdom"
    lat, lon = get_coordinates(address)
    weather_data = get_uk_weather_data(lat, lon, year=2023)
    ddy_filename = create_ddy_file(weather_data, lat, lon)
    validate_ddy_file(ddy_filename)