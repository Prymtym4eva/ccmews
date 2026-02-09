"""
CCMEWS Data Service - Climate Data Ingestion & Storage
Fetches real-time climate data from Open-Meteo API every 5 hours
Stores historical data for AI model training
"""
# Try httpx first, fall back to requests
try:
    import httpx
    HTTP_CLIENT = "httpx"
except ImportError:
    try:
        import requests
        HTTP_CLIENT = "requests"
    except ImportError:
        raise ImportError("Either 'httpx' or 'requests' must be installed")

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CCMEWS-DataService')

# Database path
DB_PATH = Path(__file__).parent / "ccmews_climate.db"

# North Tongu monitoring points (subset for API calls - grid coverage)
MONITORING_GRID = [
    # Format: (name, lat, lon)
    ("Battor", 6.0833, 0.4200),
    ("Adidome", 6.1200, 0.3150),
    ("Aveyime", 6.1600, 0.4550),
    ("Mepe", 6.0500, 0.3850),
    ("Mafi Kumase", 6.0900, 0.3500),
    ("Sogakope", 6.0100, 0.4200),
    ("Mafi Asiekpe", 6.2800, 0.2100),
    ("Mafi Anfoe", 6.3100, 0.2800),
    ("Togorme", 6.2650, 0.3500),
    ("Mafi Adonkia", 6.2000, 0.1760),
    ("Dorfor", 6.2000, 0.3150),
    ("Fievie", 6.2000, 0.3850),
    ("Volo", 6.1200, 0.3850),
    ("Torgome", 6.1600, 0.2800),
    ("Bakpa", 5.9700, 0.3500),
    ("Station_NW", 6.2800, 0.1760),
    ("Station_NE", 6.2400, 0.4550),
    ("Station_S", 6.0100, 0.4900),
]

# Open-Meteo API configuration
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"

class ClimateDataService:
    """Service for fetching and storing climate data"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Climate observations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS climate_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                observation_time TIMESTAMP NOT NULL,
                temperature_2m REAL,
                temperature_max REAL,
                temperature_min REAL,
                relative_humidity REAL,
                precipitation REAL,
                precipitation_probability REAL,
                soil_moisture REAL,
                evapotranspiration REAL,
                wind_speed REAL,
                wind_direction REAL,
                pressure REAL,
                cloud_cover REAL,
                uv_index REAL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(location_name, observation_time)
            )
        """)
        
        # Weather forecasts table (7-day predictions from API)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                forecast_time TIMESTAMP NOT NULL,
                forecast_horizon_hours INTEGER,
                temperature_2m REAL,
                temperature_max REAL,
                temperature_min REAL,
                relative_humidity REAL,
                precipitation REAL,
                precipitation_probability REAL,
                soil_moisture REAL,
                wind_speed REAL,
                cloud_cover REAL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(location_name, forecast_time, fetched_at)
            )
        """)
        
        # AI hazard predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hazard_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                prediction_date DATE NOT NULL,
                prediction_horizon_days INTEGER,
                flood_risk REAL,
                heat_risk REAL,
                drought_risk REAL,
                composite_risk REAL,
                confidence_score REAL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(location_name, prediction_date, prediction_horizon_days)
            )
        """)
        
        # Data fetch log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fetch_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fetch_type TEXT NOT NULL,
                status TEXT NOT NULL,
                locations_count INTEGER,
                records_inserted INTEGER,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_obs_location_time ON climate_observations(location_name, observation_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_forecast_location_time ON weather_forecasts(location_name, forecast_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hazard_location_date ON hazard_predictions(location_name, prediction_date)")
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def fetch_current_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch current weather from Open-Meteo"""
        try:
            url = f"{OPEN_METEO_BASE}/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,pressure_msl,cloud_cover",
                "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,soil_moisture_0_to_1cm,evapotranspiration",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
                "timezone": "Africa/Accra",
                "past_days": 7,
                "forecast_days": 7
            }
            
            if HTTP_CLIENT == "httpx":
                with httpx.Client(timeout=30) as client:
                    response = client.get(url, params=params)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.error(f"API error: {response.status_code}")
                        return None
            else:
                # Use requests
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"API error: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to fetch weather for ({lat}, {lon}): {e}")
            return None
    
    def fetch_all_locations(self) -> Tuple[int, int]:
        """Fetch weather data for all monitoring locations"""
        start_time = datetime.now()
        logger.info("Starting data fetch for all locations...")
        
        records_inserted = 0
        locations_processed = 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for name, lat, lon in MONITORING_GRID:
            data = self.fetch_current_weather(lat, lon)
            if data:
                try:
                    # Process current weather
                    current = data.get("current", {})
                    current_time = current.get("time", datetime.now().isoformat())
                    
                    # Insert current observation
                    cursor.execute("""
                        INSERT OR REPLACE INTO climate_observations
                        (location_name, latitude, longitude, observation_time,
                         temperature_2m, relative_humidity, precipitation,
                         wind_speed, wind_direction, pressure, cloud_cover, uv_index)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        name, lat, lon, current_time,
                        current.get("temperature_2m"),
                        current.get("relative_humidity_2m"),
                        current.get("precipitation"),
                        current.get("wind_speed_10m"),
                        current.get("wind_direction_10m"),
                        current.get("pressure_msl"),
                        current.get("cloud_cover"),
                        current.get("uv_index")
                    ))
                    records_inserted += 1
                    
                    # Process hourly data (past + forecast)
                    hourly = data.get("hourly", {})
                    hourly_times = hourly.get("time", [])
                    
                    for i, time_str in enumerate(hourly_times):
                        # Determine if this is past (observation) or future (forecast)
                        obs_time = datetime.fromisoformat(time_str)
                        is_forecast = obs_time > datetime.now()
                        
                        if is_forecast:
                            # Store as forecast
                            hours_ahead = int((obs_time - datetime.now()).total_seconds() / 3600)
                            cursor.execute("""
                                INSERT OR REPLACE INTO weather_forecasts
                                (location_name, latitude, longitude, forecast_time, forecast_horizon_hours,
                                 temperature_2m, relative_humidity, precipitation, precipitation_probability, soil_moisture)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                name, lat, lon, time_str, hours_ahead,
                                hourly.get("temperature_2m", [None] * len(hourly_times))[i],
                                hourly.get("relative_humidity_2m", [None] * len(hourly_times))[i],
                                hourly.get("precipitation", [None] * len(hourly_times))[i],
                                hourly.get("precipitation_probability", [None] * len(hourly_times))[i],
                                hourly.get("soil_moisture_0_to_1cm", [None] * len(hourly_times))[i]
                            ))
                        else:
                            # Store as historical observation
                            cursor.execute("""
                                INSERT OR REPLACE INTO climate_observations
                                (location_name, latitude, longitude, observation_time,
                                 temperature_2m, relative_humidity, precipitation,
                                 soil_moisture, evapotranspiration)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                name, lat, lon, time_str,
                                hourly.get("temperature_2m", [None] * len(hourly_times))[i],
                                hourly.get("relative_humidity_2m", [None] * len(hourly_times))[i],
                                hourly.get("precipitation", [None] * len(hourly_times))[i],
                                hourly.get("soil_moisture_0_to_1cm", [None] * len(hourly_times))[i],
                                hourly.get("evapotranspiration", [None] * len(hourly_times))[i]
                            ))
                        records_inserted += 1
                    
                    locations_processed += 1
                    logger.info(f"Processed {name}: {len(hourly_times)} hourly records")
                    
                except Exception as e:
                    logger.error(f"Error processing data for {name}: {e}")
        
        # Log the fetch
        cursor.execute("""
            INSERT INTO fetch_log (fetch_type, status, locations_count, records_inserted, started_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("weather_data", "success", locations_processed, records_inserted, start_time))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Data fetch complete: {locations_processed} locations, {records_inserted} records")
        return locations_processed, records_inserted
    
    def get_latest_observations(self) -> List[Dict]:
        """Get most recent observations for all locations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM climate_observations
            WHERE observation_time = (
                SELECT MAX(observation_time) FROM climate_observations AS co2
                WHERE co2.location_name = climate_observations.location_name
            )
            ORDER BY location_name
        """)
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_historical_data(self, location: str, days: int = 30) -> List[Dict]:
        """Get historical observations for a location"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT * FROM climate_observations
            WHERE location_name = ? AND observation_time >= ?
            ORDER BY observation_time
        """, (location, start_date))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_forecasts(self, location: str = None) -> List[Dict]:
        """Get weather forecasts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if location:
            cursor.execute("""
                SELECT * FROM weather_forecasts
                WHERE location_name = ? AND forecast_time > datetime('now')
                ORDER BY forecast_time
            """, (location,))
        else:
            cursor.execute("""
                SELECT * FROM weather_forecasts
                WHERE forecast_time > datetime('now')
                ORDER BY location_name, forecast_time
            """)
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_fetch_status(self) -> Dict:
        """Get last fetch status"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM fetch_log ORDER BY completed_at DESC LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return {"status": "no_data", "message": "No data fetches recorded"}


# Standalone functions for CLI/scheduler use
def update_climate_data():
    """Main function to update all climate data"""
    service = ClimateDataService()
    locations, records = service.fetch_all_locations()
    return {"locations": locations, "records": records, "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    # Run a data fetch when executed directly
    print("=" * 60)
    print("CCMEWS Climate Data Service")
    print("=" * 60)
    
    service = ClimateDataService()
    print(f"\nDatabase: {service.db_path}")
    print(f"Monitoring points: {len(MONITORING_GRID)}")
    print("\nFetching data from Open-Meteo API...")
    
    locations, records = service.fetch_all_locations()
    
    print(f"\n✅ Fetch complete!")
    print(f"   Locations processed: {locations}")
    print(f"   Records inserted: {records}")
    
    # Show sample of latest data
    print("\n📊 Latest observations:")
    latest = service.get_latest_observations()
    for obs in latest[:5]:
        print(f"   {obs['location_name']}: {obs['temperature_2m']}°C, "
              f"Humidity: {obs['relative_humidity']}%, "
              f"Precip: {obs['precipitation']}mm")
