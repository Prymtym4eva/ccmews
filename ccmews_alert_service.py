"""
CCMEWS Integrated Alert Service
Combines event forecasting with SMS alerting for automated hazard notifications

This service:
1. Fetches weather forecasts for all monitoring locations
2. Identifies upcoming rainfall and heat events
3. Checks hazard predictions from AI engine
4. Sends SMS alerts when thresholds are exceeded

Run as a scheduled service (e.g., every 6 hours via cron or scheduler)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import CCMEWS modules
try:
    from ccmews_event_forecast import EventForecaster, RainfallEvent, HeatEvent
    from ccmews_sms_alerts import AlertSystem, SMSConfig, AlertMessage
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure ccmews_event_forecast.py and ccmews_sms_alerts.py are in the same directory")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CCMEWS-AlertService')

# Paths
BASE_DIR = Path(__file__).parent
SERVICE_LOG = BASE_DIR / "alert_service_log.json"

# North Tongu monitoring locations
MONITORING_LOCATIONS = [
    ("Battor", 6.0833, 0.4200),
    ("Adidome", 6.1200, 0.3150),
    ("Sogakope", 6.0100, 0.4200),
    ("Mepe", 6.0500, 0.3850),
    ("Mafi Kumase", 6.0900, 0.3500),
    ("Aveyime", 6.1600, 0.4550),
]

# Alert thresholds
ALERT_THRESHOLDS = {
    # Rainfall alerts
    "rainfall": {
        "heavy": 25,      # mm - send alert
        "very_heavy": 50, # mm - urgent alert
        "extreme": 80,    # mm - emergency alert
    },
    # Temperature alerts
    "temperature": {
        "hot": 37,        # °C - send advisory
        "very_hot": 40,   # °C - send warning
        "dangerous": 42,  # °C - send emergency
    },
    # Days ahead to alert
    "advance_warning": {
        "rainfall": 2,    # Alert if rain expected within 2 days
        "heat": 1,        # Alert if heat expected within 1 day
    }
}


class AlertService:
    """Integrated alert service for CCMEWS"""
    
    def __init__(self):
        self.forecaster = EventForecaster()
        self.alert_system = AlertSystem()
        self.run_log = self._load_run_log()
    
    def _load_run_log(self) -> Dict:
        """Load the service run log"""
        if SERVICE_LOG.exists():
            with open(SERVICE_LOG, 'r') as f:
                return json.load(f)
        return {"runs": [], "alerts_sent": []}
    
    def _save_run_log(self):
        """Save the service run log"""
        # Keep only last 100 runs
        self.run_log["runs"] = self.run_log["runs"][-100:]
        self.run_log["alerts_sent"] = self.run_log["alerts_sent"][-500:]
        
        with open(SERVICE_LOG, 'w') as f:
            json.dump(self.run_log, f, indent=2, default=str)
    
    def _should_send_alert(self, alert_key: str, cooldown_hours: int = 12) -> bool:
        """
        Check if we should send an alert (avoid duplicates)
        
        Args:
            alert_key: Unique key for this alert (e.g., "flood_battor_2024-01-15")
            cooldown_hours: Minimum hours between duplicate alerts
            
        Returns:
            True if alert should be sent
        """
        recent_alerts = self.run_log.get("alerts_sent", [])
        cutoff = datetime.now().timestamp() - (cooldown_hours * 3600)
        
        for alert in recent_alerts:
            if alert.get("key") == alert_key and alert.get("timestamp", 0) > cutoff:
                return False
        return True
    
    def _record_alert(self, alert_key: str):
        """Record that an alert was sent"""
        self.run_log["alerts_sent"].append({
            "key": alert_key,
            "timestamp": datetime.now().timestamp(),
            "datetime": datetime.now().isoformat()
        })
    
    def check_and_alert_rainfall(self, location: str, lat: float, lon: float) -> Optional[Dict]:
        """
        Check for upcoming rainfall and send alerts if needed
        
        Returns:
            Alert result if alert was sent, None otherwise
        """
        rainfall = self.forecaster.predict_next_rainfall(lat, lon, location)
        
        if not rainfall:
            logger.info(f"  {location}: No significant rainfall in forecast")
            return None
        
        # Check if rainfall meets alert thresholds
        if rainfall.total_precipitation_mm < ALERT_THRESHOLDS["rainfall"]["heavy"]:
            logger.info(f"  {location}: Light rain ({rainfall.total_precipitation_mm}mm) - no alert needed")
            return None
        
        # Check if within advance warning period
        if rainfall.days_until > ALERT_THRESHOLDS["advance_warning"]["rainfall"]:
            logger.info(f"  {location}: Rain in {rainfall.days_until} days - too far ahead")
            return None
        
        # Create alert key for deduplication
        alert_key = f"rain_{location}_{rainfall.start_date}"
        
        if not self._should_send_alert(alert_key):
            logger.info(f"  {location}: Rainfall alert already sent recently")
            return None
        
        # Determine urgency
        if rainfall.total_precipitation_mm >= ALERT_THRESHOLDS["rainfall"]["extreme"]:
            intensity = "EXTREME"
        elif rainfall.total_precipitation_mm >= ALERT_THRESHOLDS["rainfall"]["very_heavy"]:
            intensity = "VERY HEAVY"
        else:
            intensity = "HEAVY"
        
        # Send alert
        logger.info(f"  {location}: 🌧️ Sending {intensity} rainfall alert!")
        
        result = self.alert_system.send_rainfall_forecast(
            location=f"{location}, North Tongu",
            rainfall_mm=rainfall.total_precipitation_mm,
            intensity=intensity,
            start_time=f"{rainfall.start_date} {rainfall.start_time}",
            duration_hours=rainfall.duration_hours,
            probability=rainfall.probability
        )
        
        self._record_alert(alert_key)
        
        return {
            "type": "rainfall",
            "location": location,
            "event": rainfall.to_dict(),
            "result": result
        }
    
    def check_and_alert_heat(self, location: str, lat: float, lon: float) -> Optional[Dict]:
        """
        Check for upcoming heat events and send alerts if needed
        
        Returns:
            Alert result if alert was sent, None otherwise
        """
        heat = self.forecaster.predict_next_heat_event(lat, lon, location)
        
        if not heat:
            logger.info(f"  {location}: No significant heat event in forecast")
            return None
        
        # Check if temperature meets alert thresholds
        if heat.max_temperature < ALERT_THRESHOLDS["temperature"]["hot"]:
            logger.info(f"  {location}: Warm temps ({heat.max_temperature}°C) - no alert needed")
            return None
        
        # Check if within advance warning period
        if heat.days_until > ALERT_THRESHOLDS["advance_warning"]["heat"]:
            logger.info(f"  {location}: Heat in {heat.days_until} days - too far ahead")
            return None
        
        # Create alert key for deduplication
        alert_key = f"heat_{location}_{heat.start_date}"
        
        if not self._should_send_alert(alert_key):
            logger.info(f"  {location}: Heat alert already sent recently")
            return None
        
        # Send alert
        logger.info(f"  {location}: 🔥 Sending {heat.severity.upper()} heat alert!")
        
        result = self.alert_system.send_heat_alert(
            location=f"{location}, North Tongu",
            max_temp=heat.max_temperature,
            heat_index=heat.max_heat_index,
            duration=f"{heat.duration_days} day(s)"
        )
        
        self._record_alert(alert_key)
        
        return {
            "type": "heat",
            "location": location,
            "event": heat.to_dict(),
            "result": result
        }
    
    def run_alert_check(self, locations: List[Tuple[str, float, float]] = None) -> Dict:
        """
        Run complete alert check for all locations
        
        Args:
            locations: List of (name, lat, lon) tuples. Uses defaults if not provided.
            
        Returns:
            Summary of alerts sent
        """
        if locations is None:
            locations = MONITORING_LOCATIONS
        
        run_start = datetime.now()
        
        logger.info("=" * 60)
        logger.info("CCMEWS ALERT SERVICE - Running Check")
        logger.info(f"Time: {run_start.isoformat()}")
        logger.info(f"Locations: {len(locations)}")
        logger.info("=" * 60)
        
        results = {
            "start_time": run_start.isoformat(),
            "locations_checked": len(locations),
            "alerts_sent": [],
            "errors": []
        }
        
        for name, lat, lon in locations:
            logger.info(f"\nChecking {name}...")
            
            try:
                # Check rainfall
                rainfall_alert = self.check_and_alert_rainfall(name, lat, lon)
                if rainfall_alert:
                    results["alerts_sent"].append(rainfall_alert)
                
                # Check heat
                heat_alert = self.check_and_alert_heat(name, lat, lon)
                if heat_alert:
                    results["alerts_sent"].append(heat_alert)
                    
            except Exception as e:
                logger.error(f"Error checking {name}: {e}")
                results["errors"].append({"location": name, "error": str(e)})
        
        # Calculate summary
        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = (datetime.now() - run_start).total_seconds()
        results["total_alerts"] = len(results["alerts_sent"])
        results["total_errors"] = len(results["errors"])
        
        # Log run
        self.run_log["runs"].append({
            "timestamp": run_start.isoformat(),
            "locations": len(locations),
            "alerts": len(results["alerts_sent"]),
            "errors": len(results["errors"])
        })
        self._save_run_log()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("RUN COMPLETE")
        logger.info(f"Alerts sent: {results['total_alerts']}")
        logger.info(f"Errors: {results['total_errors']}")
        logger.info(f"Duration: {results['duration_seconds']:.1f}s")
        logger.info("=" * 60)
        
        return results
    
    def get_current_forecasts(self, locations: List[Tuple[str, float, float]] = None) -> List[Dict]:
        """
        Get current forecasts for all locations (for dashboard display)
        
        Returns:
            List of forecast dictionaries
        """
        if locations is None:
            locations = MONITORING_LOCATIONS
        
        forecasts = []
        
        for name, lat, lon in locations:
            try:
                forecast = self.forecaster.get_full_forecast(lat, lon, name)
                forecasts.append(forecast)
            except Exception as e:
                logger.error(f"Error getting forecast for {name}: {e}")
                forecasts.append({
                    "location": name,
                    "error": str(e)
                })
        
        return forecasts
    
    def get_district_summary(self) -> Dict:
        """
        Get summary of forecasts and alerts for the entire district
        
        Returns:
            Dictionary with district-wide summary
        """
        forecasts = self.get_current_forecasts()
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "district": "North Tongu",
            "locations_monitored": len(forecasts),
            "next_rainfall": None,
            "next_heat_event": None,
            "alerts_today": 0,
            "forecasts": forecasts
        }
        
        # Find earliest rainfall and heat events
        earliest_rain = None
        earliest_heat = None
        
        for f in forecasts:
            rain = f.get("next_rainfall")
            if rain:
                if earliest_rain is None or rain["days_until"] < earliest_rain["days_until"]:
                    earliest_rain = rain
            
            heat = f.get("next_heat_event")
            if heat:
                if earliest_heat is None or heat["days_until"] < earliest_heat["days_until"]:
                    earliest_heat = heat
        
        summary["next_rainfall"] = earliest_rain
        summary["next_heat_event"] = earliest_heat
        
        # Count today's alerts
        today = datetime.now().date().isoformat()
        for alert in self.run_log.get("alerts_sent", []):
            if alert.get("datetime", "")[:10] == today:
                summary["alerts_today"] += 1
        
        return summary


def main():
    """Main entry point for alert service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CCMEWS Alert Service")
    parser.add_argument("--check", action="store_true", help="Run alert check")
    parser.add_argument("--forecast", action="store_true", help="Show current forecasts")
    parser.add_argument("--summary", action="store_true", help="Show district summary")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    service = AlertService()
    
    if args.check:
        results = service.run_alert_check()
        if args.json:
            print(json.dumps(results, indent=2))
    
    elif args.forecast:
        forecasts = service.get_current_forecasts()
        if args.json:
            print(json.dumps(forecasts, indent=2))
        else:
            for f in forecasts:
                print(f"\n{'='*50}")
                print(f"📍 {f['location']}")
                print('='*50)
                for s in f.get("summary", []):
                    print(f"  {s}")
    
    elif args.summary:
        summary = service.get_district_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("\n" + "="*60)
            print("NORTH TONGU DISTRICT - WEATHER SUMMARY")
            print("="*60)
            print(f"Generated: {summary['generated_at']}")
            print(f"Locations Monitored: {summary['locations_monitored']}")
            print(f"Alerts Today: {summary['alerts_today']}")
            
            rain = summary["next_rainfall"]
            if rain:
                print(f"\n🌧️ Next Rainfall:")
                print(f"   Location: {rain['location']}")
                print(f"   When: {rain['start_date']} at {rain['start_time']}")
                print(f"   Amount: {rain['total_precipitation_mm']}mm ({rain['intensity']})")
            else:
                print("\n☀️ No significant rainfall in forecast")
            
            heat = summary["next_heat_event"]
            if heat:
                print(f"\n🔥 Next Heat Event:")
                print(f"   Location: {heat['location']}")
                print(f"   When: {heat['start_date']}")
                print(f"   Max Temp: {heat['max_temperature']}°C ({heat['severity']})")
            else:
                print("\n🌡️ No extreme heat in forecast")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
