"""
CCMEWS Event Forecasting Module
Predicts when the next rainfall and heat wave events will occur
Integrates with Open-Meteo forecast API for 7-day predictions
"""
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CCMEWS-Forecast')

# Open-Meteo Forecast API (free, no key required)
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Thresholds for Ghana/Volta Region
THRESHOLDS = {
    "rainfall": {
        "light": 2.5,        # mm - light rain
        "moderate": 10,      # mm - moderate rain
        "heavy": 25,         # mm - heavy rain
        "very_heavy": 50,    # mm - very heavy rain
        "extreme": 80,       # mm - extreme rainfall
    },
    "temperature": {
        "warm": 33,          # °C - warm
        "hot": 35,           # °C - hot
        "very_hot": 37,      # °C - very hot (heat advisory)
        "extreme": 40,       # °C - extreme heat (heat warning)
        "dangerous": 42,     # °C - dangerous heat (emergency)
    },
    "heat_index": {
        "caution": 33,       # °C feels-like
        "extreme_caution": 40,
        "danger": 45,
        "extreme_danger": 52,
    }
}


@dataclass
class RainfallEvent:
    """Predicted rainfall event"""
    start_date: str
    start_time: str
    end_date: str
    end_time: str
    duration_hours: int
    total_precipitation_mm: float
    max_hourly_mm: float
    intensity: str  # light, moderate, heavy, very_heavy, extreme
    probability: float
    location: str
    days_until: int
    
    def to_dict(self):
        return asdict(self)
    
    def __str__(self):
        return (f"🌧️ {self.intensity.upper()} RAIN expected at {self.location}\n"
                f"   Starting: {self.start_date} at {self.start_time}\n"
                f"   Duration: ~{self.duration_hours} hours\n"
                f"   Total: {self.total_precipitation_mm:.1f}mm\n"
                f"   Probability: {self.probability:.0%}")


@dataclass  
class HeatEvent:
    """Predicted heat wave event"""
    start_date: str
    end_date: str
    duration_days: int
    max_temperature: float
    avg_temperature: float
    max_heat_index: float
    severity: str  # warm, hot, very_hot, extreme, dangerous
    location: str
    days_until: int
    consecutive_hot_days: int
    
    def to_dict(self):
        return asdict(self)
    
    def __str__(self):
        return (f"🔥 {self.severity.upper()} HEAT expected at {self.location}\n"
                f"   Period: {self.start_date} to {self.end_date}\n"
                f"   Duration: {self.duration_days} day(s)\n"
                f"   Max Temperature: {self.max_temperature:.1f}°C\n"
                f"   Feels Like: {self.max_heat_index:.1f}°C")


class EventForecaster:
    """Forecasts rainfall and heat events using Open-Meteo API"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def fetch_forecast(self, lat: float, lon: float) -> Optional[Dict]:
        """
        Fetch 7-day hourly forecast from Open-Meteo
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with hourly forecast data
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "precipitation_probability",
                "rain",
                "weather_code",
                "wind_speed_10m",
                "apparent_temperature",
            ],
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
                "sunrise",
                "sunset",
            ],
            "timezone": "Africa/Accra",
            "forecast_days": 7
        }
        
        try:
            response = self.session.get(FORECAST_API, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch forecast: {e}")
            return None
    
    def predict_next_rainfall(self, lat: float, lon: float, 
                              location_name: str = "Location") -> Optional[RainfallEvent]:
        """
        Predict the next significant rainfall event
        
        Args:
            lat: Latitude
            lon: Longitude
            location_name: Name of location for display
            
        Returns:
            RainfallEvent if rain predicted, None otherwise
        """
        forecast = self.fetch_forecast(lat, lon)
        if not forecast:
            return None
        
        hourly = forecast.get("hourly", {})
        times = hourly.get("time", [])
        precip = hourly.get("precipitation", [])
        precip_prob = hourly.get("precipitation_probability", [])
        
        if not times or not precip:
            return None
        
        # Find first hour with significant precipitation (>= 0.5mm)
        rain_start_idx = None
        for i, (p, prob) in enumerate(zip(precip, precip_prob)):
            if p and p >= 0.5 and (prob is None or prob >= 30):
                rain_start_idx = i
                break
        
        if rain_start_idx is None:
            return None  # No rain in forecast
        
        # Find duration of rain event (continuous precipitation)
        rain_end_idx = rain_start_idx
        total_precip = 0
        max_hourly = 0
        
        for i in range(rain_start_idx, len(precip)):
            p = precip[i] or 0
            if p >= 0.1:  # Still raining
                rain_end_idx = i
                total_precip += p
                max_hourly = max(max_hourly, p)
            elif i > rain_start_idx and p < 0.1:
                # Check if rain continues after a short break (< 3 hours)
                future_rain = any(precip[j] >= 0.5 for j in range(i, min(i+3, len(precip))))
                if not future_rain:
                    break
        
        # Parse times
        start_dt = datetime.fromisoformat(times[rain_start_idx])
        end_dt = datetime.fromisoformat(times[rain_end_idx])
        now = datetime.now()
        
        # Calculate duration
        duration_hours = (rain_end_idx - rain_start_idx) + 1
        
        # Determine intensity
        if total_precip >= THRESHOLDS["rainfall"]["extreme"]:
            intensity = "extreme"
        elif total_precip >= THRESHOLDS["rainfall"]["very_heavy"]:
            intensity = "very_heavy"
        elif total_precip >= THRESHOLDS["rainfall"]["heavy"]:
            intensity = "heavy"
        elif total_precip >= THRESHOLDS["rainfall"]["moderate"]:
            intensity = "moderate"
        else:
            intensity = "light"
        
        # Get probability
        avg_prob = sum(precip_prob[rain_start_idx:rain_end_idx+1]) / (rain_end_idx - rain_start_idx + 1)
        
        # Days until event
        days_until = (start_dt.date() - now.date()).days
        
        return RainfallEvent(
            start_date=start_dt.strftime("%Y-%m-%d"),
            start_time=start_dt.strftime("%H:%M"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            end_time=end_dt.strftime("%H:%M"),
            duration_hours=duration_hours,
            total_precipitation_mm=round(total_precip, 1),
            max_hourly_mm=round(max_hourly, 1),
            intensity=intensity,
            probability=round(avg_prob / 100, 2) if avg_prob else 0.7,
            location=location_name,
            days_until=max(0, days_until)
        )
    
    def predict_next_heat_event(self, lat: float, lon: float,
                                location_name: str = "Location") -> Optional[HeatEvent]:
        """
        Predict the next heat wave or high temperature event
        
        Args:
            lat: Latitude
            lon: Longitude
            location_name: Name of location for display
            
        Returns:
            HeatEvent if heat event predicted, None otherwise
        """
        forecast = self.fetch_forecast(lat, lon)
        if not forecast:
            return None
        
        daily = forecast.get("daily", {})
        hourly = forecast.get("hourly", {})
        
        dates = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        
        hourly_times = hourly.get("time", [])
        hourly_temp = hourly.get("temperature_2m", [])
        hourly_humidity = hourly.get("relative_humidity_2m", [])
        apparent_temp = hourly.get("apparent_temperature", [])
        
        if not dates or not temp_max:
            return None
        
        # Find first day with temperature >= hot threshold (35°C)
        heat_start_idx = None
        for i, t in enumerate(temp_max):
            if t and t >= THRESHOLDS["temperature"]["hot"]:
                heat_start_idx = i
                break
        
        if heat_start_idx is None:
            return None  # No heat event in forecast
        
        # Find consecutive hot days
        heat_end_idx = heat_start_idx
        for i in range(heat_start_idx + 1, len(temp_max)):
            if temp_max[i] and temp_max[i] >= THRESHOLDS["temperature"]["warm"]:
                heat_end_idx = i
            else:
                break
        
        # Calculate statistics
        heat_temps = [t for t in temp_max[heat_start_idx:heat_end_idx+1] if t]
        max_temp = max(heat_temps)
        avg_temp = sum(heat_temps) / len(heat_temps)
        duration = heat_end_idx - heat_start_idx + 1
        
        # Calculate max heat index from hourly data
        max_heat_index = max_temp  # Default to temp
        
        if hourly_temp and hourly_humidity:
            # Find hours within the heat event period
            start_date = dates[heat_start_idx]
            end_date = dates[heat_end_idx]
            
            for i, (t, h, at) in enumerate(zip(hourly_temp, hourly_humidity, apparent_temp or hourly_temp)):
                hour_date = hourly_times[i][:10]
                if start_date <= hour_date <= end_date:
                    if at and at > max_heat_index:
                        max_heat_index = at
                    elif t and h:
                        hi = self._calculate_heat_index(t, h)
                        if hi > max_heat_index:
                            max_heat_index = hi
        
        # Determine severity
        if max_temp >= THRESHOLDS["temperature"]["dangerous"]:
            severity = "dangerous"
        elif max_temp >= THRESHOLDS["temperature"]["extreme"]:
            severity = "extreme"
        elif max_temp >= THRESHOLDS["temperature"]["very_hot"]:
            severity = "very_hot"
        elif max_temp >= THRESHOLDS["temperature"]["hot"]:
            severity = "hot"
        else:
            severity = "warm"
        
        # Days until event
        now = datetime.now()
        start_dt = datetime.strptime(dates[heat_start_idx], "%Y-%m-%d")
        days_until = (start_dt.date() - now.date()).days
        
        return HeatEvent(
            start_date=dates[heat_start_idx],
            end_date=dates[heat_end_idx],
            duration_days=duration,
            max_temperature=round(max_temp, 1),
            avg_temperature=round(avg_temp, 1),
            max_heat_index=round(max_heat_index, 1),
            severity=severity,
            location=location_name,
            days_until=max(0, days_until),
            consecutive_hot_days=duration
        )
    
    def _calculate_heat_index(self, temp_c: float, humidity: float) -> float:
        """Calculate heat index (feels-like temperature)"""
        if temp_c < 27:
            return temp_c
        
        T = temp_c
        R = humidity
        
        hi = (-8.78469475556 + 
              1.61139411 * T + 
              2.33854883889 * R +
              -0.14611605 * T * R +
              -0.012308094 * T**2 +
              -0.0164248277778 * R**2 +
              0.002211732 * T**2 * R +
              0.00072546 * T * R**2 +
              -0.000003582 * T**2 * R**2)
        
        return max(T, hi)
    
    def get_full_forecast(self, lat: float, lon: float, 
                          location_name: str = "Location") -> Dict:
        """
        Get complete event forecast for a location
        
        Returns:
            Dictionary with rainfall and heat event predictions
        """
        rainfall = self.predict_next_rainfall(lat, lon, location_name)
        heat = self.predict_next_heat_event(lat, lon, location_name)
        
        result = {
            "location": location_name,
            "latitude": lat,
            "longitude": lon,
            "generated_at": datetime.now().isoformat(),
            "next_rainfall": rainfall.to_dict() if rainfall else None,
            "next_heat_event": heat.to_dict() if heat else None,
        }
        
        # Generate summary
        summaries = []
        if rainfall:
            if rainfall.days_until == 0:
                summaries.append(f"🌧️ {rainfall.intensity.upper()} rain expected TODAY at {rainfall.start_time}")
            elif rainfall.days_until == 1:
                summaries.append(f"🌧️ {rainfall.intensity.upper()} rain expected TOMORROW")
            else:
                summaries.append(f"🌧️ {rainfall.intensity.upper()} rain in {rainfall.days_until} days")
        else:
            summaries.append("☀️ No significant rainfall in 7-day forecast")
        
        if heat:
            if heat.days_until == 0:
                summaries.append(f"🔥 {heat.severity.upper()} heat TODAY ({heat.max_temperature}°C)")
            elif heat.days_until == 1:
                summaries.append(f"🔥 {heat.severity.upper()} heat TOMORROW ({heat.max_temperature}°C)")
            else:
                summaries.append(f"🔥 {heat.severity.upper()} heat in {heat.days_until} days")
        else:
            summaries.append("🌡️ No extreme heat in 7-day forecast")
        
        result["summary"] = summaries
        
        return result


def get_forecast_for_locations(locations: List[Tuple[str, float, float]]) -> List[Dict]:
    """
    Get forecasts for multiple locations
    
    Args:
        locations: List of (name, lat, lon) tuples
        
    Returns:
        List of forecast dictionaries
    """
    forecaster = EventForecaster()
    results = []
    
    for name, lat, lon in locations:
        logger.info(f"Getting forecast for {name}...")
        forecast = forecaster.get_full_forecast(lat, lon, name)
        results.append(forecast)
    
    return results


# Example usage
if __name__ == "__main__":
    # North Tongu locations
    locations = [
        ("Battor", 6.0833, 0.4200),
        ("Adidome", 6.1200, 0.3150),
        ("Sogakope", 6.0100, 0.4200),
    ]
    
    forecaster = EventForecaster()
    
    for name, lat, lon in locations:
        print(f"\n{'='*60}")
        print(f"FORECAST FOR {name.upper()}")
        print('='*60)
        
        forecast = forecaster.get_full_forecast(lat, lon, name)
        
        print("\n📋 Summary:")
        for s in forecast["summary"]:
            print(f"   {s}")
        
        if forecast["next_rainfall"]:
            print(f"\n{RainfallEvent(**forecast['next_rainfall'])}")
        
        if forecast["next_heat_event"]:
            print(f"\n{HeatEvent(**forecast['next_heat_event'])}")
