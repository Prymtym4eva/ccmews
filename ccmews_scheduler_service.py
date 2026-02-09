"""
CCMEWS Automated Scheduler with SMS Alerting
Runs climate data ingestion, AI predictions, and AUTOMATIC SMS ALERTS
Sends alerts when hazard thresholds are reached or predicted to be reached
"""
import time
import threading
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

# Ensure modules are importable
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "ccmews_scheduler.log")
    ]
)
logger = logging.getLogger('CCMEWS-Scheduler')

# Configuration
UPDATE_INTERVAL_HOURS = 5
STATUS_FILE = Path(__file__).parent / "ccmews_status.json"
ALERT_COOLDOWN_HOURS = 12  # Minimum hours between duplicate alerts


class AutomaticAlertChecker:
    """Automatically checks predictions and sends SMS alerts when thresholds exceeded"""
    
    def __init__(self):
        self.alert_history_file = Path(__file__).parent / "sent_alerts.json"
        self.alert_history = self._load_alert_history()
        
        # Default thresholds (can be overridden from sms_config.json)
        self.thresholds = {
            "flood_risk": 0.45,
            "heat_risk": 0.45,
            "drought_risk": 0.45,
            "composite_risk": 0.50,
            "rainfall_mm": 30,
            "temperature_c": 37,
        }
        
        # Load thresholds from SMS config if available
        self._load_thresholds_from_config()
    
    def _load_thresholds_from_config(self):
        """Load alert thresholds from SMS configuration"""
        try:
            config_path = Path(__file__).parent / "sms_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "alert_thresholds" in config:
                        self.thresholds.update(config["alert_thresholds"])
                        logger.info(f"Loaded alert thresholds from config: {self.thresholds}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from config: {e}")
    
    def _load_alert_history(self) -> dict:
        """Load history of sent alerts to avoid duplicates"""
        try:
            if self.alert_history_file.exists():
                with open(self.alert_history_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {"sent_alerts": []}
    
    def _save_alert_history(self):
        """Save alert history"""
        # Keep only last 500 alerts
        self.alert_history["sent_alerts"] = self.alert_history["sent_alerts"][-500:]
        with open(self.alert_history_file, 'w') as f:
            json.dump(self.alert_history, f, indent=2)
    
    def _should_send_alert(self, alert_key: str) -> bool:
        """Check if alert should be sent (not recently sent)"""
        cutoff = (datetime.now() - timedelta(hours=ALERT_COOLDOWN_HOURS)).isoformat()
        
        for alert in self.alert_history.get("sent_alerts", []):
            if alert.get("key") == alert_key and alert.get("timestamp", "") > cutoff:
                return False
        return True
    
    def _record_alert_sent(self, alert_key: str, alert_type: str, location: str, details: dict):
        """Record that an alert was sent"""
        self.alert_history["sent_alerts"].append({
            "key": alert_key,
            "type": alert_type,
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "details": details
        })
        self._save_alert_history()
    
    def check_and_alert(self, predictions: list) -> dict:
        """
        Check predictions against thresholds and send SMS alerts automatically
        
        Args:
            predictions: List of prediction dictionaries from AI engine
            
        Returns:
            Dictionary with alert results
        """
        results = {
            "checked": 0,
            "alerts_triggered": 0,
            "alerts_sent": 0,
            "alerts_skipped": 0,
            "errors": [],
            "details": []
        }
        
        # Try to import alert system
        try:
            from ccmews_sms_alerts import AlertSystem, SMSConfig
            config = SMSConfig()
            
            # Check if SMS is enabled
            if not config.config.get("enabled", False):
                logger.info("SMS alerts are disabled in config - skipping automatic alerts")
                results["status"] = "disabled"
                return results
            
            alert_system = AlertSystem()
            
        except ImportError as e:
            logger.warning(f"SMS alert system not available: {e}")
            results["status"] = "unavailable"
            return results
        except Exception as e:
            logger.error(f"Failed to initialize alert system: {e}")
            results["status"] = "error"
            results["errors"].append(str(e))
            return results
        
        logger.info(f"📲 Checking {len(predictions)} predictions against thresholds...")
        
        for pred in predictions:
            results["checked"] += 1
            location = pred.get("location", "Unknown")
            
            try:
                # Check FLOOD risk
                flood_risk = pred.get("flood_risk", 0)
                if flood_risk >= self.thresholds["flood_risk"]:
                    alert_key = f"flood_{location}_{datetime.now().strftime('%Y-%m-%d')}"
                    
                    if self._should_send_alert(alert_key):
                        results["alerts_triggered"] += 1
                        logger.warning(f"🌊 FLOOD THRESHOLD EXCEEDED at {location}: {flood_risk:.0%}")
                        
                        # Get rainfall amount from prediction
                        rainfall = pred.get("drivers", {}).get("flood", {}).get("precip_forecast_mm", 0)
                        if not rainfall:
                            rainfall = pred.get("precip_forecast", 30)
                        
                        send_result = alert_system.send_flood_alert(
                            location=f"{location}, North Tongu",
                            risk=flood_risk,
                            rainfall_mm=rainfall or 30,
                            when=pred.get("prediction_date", "Next 24 hours")
                        )
                        
                        if send_result.get("sent", 0) > 0:
                            results["alerts_sent"] += 1
                            self._record_alert_sent(alert_key, "flood", location, {
                                "risk": flood_risk, "rainfall_mm": rainfall
                            })
                            results["details"].append({
                                "type": "flood", "location": location, 
                                "risk": flood_risk, "status": "sent"
                            })
                        else:
                            results["details"].append({
                                "type": "flood", "location": location,
                                "risk": flood_risk, "status": "failed",
                                "reason": send_result
                            })
                    else:
                        results["alerts_skipped"] += 1
                        logger.info(f"  Flood alert for {location} already sent recently - skipping")
                
                # Check HEAT risk
                heat_risk = pred.get("heat_risk", 0)
                max_temp = pred.get("drivers", {}).get("heat", {}).get("max_temp", 35)
                heat_index = pred.get("drivers", {}).get("heat", {}).get("heat_index", max_temp)
                
                if heat_risk >= self.thresholds["heat_risk"] or max_temp >= self.thresholds["temperature_c"]:
                    alert_key = f"heat_{location}_{datetime.now().strftime('%Y-%m-%d')}"
                    
                    if self._should_send_alert(alert_key):
                        results["alerts_triggered"] += 1
                        logger.warning(f"🔥 HEAT THRESHOLD EXCEEDED at {location}: {max_temp}°C, risk={heat_risk:.0%}")
                        
                        send_result = alert_system.send_heat_alert(
                            location=f"{location}, North Tongu",
                            max_temp=max_temp or 37,
                            heat_index=heat_index or max_temp,
                            duration=pred.get("heat_duration", "1-2 days")
                        )
                        
                        if send_result.get("sent", 0) > 0:
                            results["alerts_sent"] += 1
                            self._record_alert_sent(alert_key, "heat", location, {
                                "risk": heat_risk, "max_temp": max_temp
                            })
                            results["details"].append({
                                "type": "heat", "location": location,
                                "temp": max_temp, "status": "sent"
                            })
                        else:
                            results["details"].append({
                                "type": "heat", "location": location,
                                "temp": max_temp, "status": "failed"
                            })
                    else:
                        results["alerts_skipped"] += 1
                        logger.info(f"  Heat alert for {location} already sent recently - skipping")
                
                # Check DROUGHT risk
                drought_risk = pred.get("drought_risk", 0)
                if drought_risk >= self.thresholds["drought_risk"]:
                    alert_key = f"drought_{location}_{datetime.now().strftime('%Y-%m-%d')}"
                    
                    if self._should_send_alert(alert_key):
                        results["alerts_triggered"] += 1
                        logger.warning(f"🏜️ DROUGHT THRESHOLD EXCEEDED at {location}: {drought_risk:.0%}")
                        
                        # Drought alerts are less urgent - just log for now
                        # Can add drought-specific alert if needed
                        self._record_alert_sent(alert_key, "drought", location, {
                            "risk": drought_risk
                        })
                        results["details"].append({
                            "type": "drought", "location": location,
                            "risk": drought_risk, "status": "logged"
                        })
                    else:
                        results["alerts_skipped"] += 1
                        
            except Exception as e:
                logger.error(f"Error checking alerts for {location}: {e}")
                results["errors"].append(f"{location}: {str(e)}")
        
        results["status"] = "success"
        
        logger.info(f"📊 Alert check complete:")
        logger.info(f"   Predictions checked: {results['checked']}")
        logger.info(f"   Alerts triggered: {results['alerts_triggered']}")
        logger.info(f"   SMS sent: {results['alerts_sent']}")
        logger.info(f"   Skipped (cooldown): {results['alerts_skipped']}")
        
        return results
    
    def check_event_forecasts(self) -> dict:
        """
        Check event forecasts (rainfall/heat) and send alerts
        
        Returns:
            Dictionary with forecast alert results
        """
        results = {
            "locations_checked": 0,
            "rainfall_alerts": 0,
            "heat_alerts": 0,
            "errors": []
        }
        
        try:
            from ccmews_event_forecast import EventForecaster
            from ccmews_sms_alerts import AlertSystem, SMSConfig
            
            config = SMSConfig()
            if not config.config.get("enabled", False):
                results["status"] = "disabled"
                return results
            
            forecaster = EventForecaster()
            alert_system = AlertSystem()
            
            # Key locations to check
            locations = [
                ("Battor", 6.0833, 0.4200),
                ("Adidome", 6.1200, 0.3150),
                ("Sogakope", 6.0100, 0.4200),
                ("Mepe", 6.0500, 0.3850),
            ]
            
            for name, lat, lon in locations:
                results["locations_checked"] += 1
                
                try:
                    # Check rainfall forecast
                    rainfall = forecaster.predict_next_rainfall(lat, lon, name)
                    if rainfall and rainfall.total_precipitation_mm >= self.thresholds["rainfall_mm"]:
                        if rainfall.days_until <= 2:  # Alert if rain within 2 days
                            alert_key = f"rain_forecast_{name}_{rainfall.start_date}"
                            
                            if self._should_send_alert(alert_key):
                                logger.warning(f"🌧️ HEAVY RAIN FORECAST for {name}: {rainfall.total_precipitation_mm}mm")
                                
                                send_result = alert_system.send_rainfall_forecast(
                                    location=f"{name}, North Tongu",
                                    rainfall_mm=rainfall.total_precipitation_mm,
                                    intensity=rainfall.intensity.upper(),
                                    start_time=f"{rainfall.start_date} {rainfall.start_time}",
                                    duration_hours=rainfall.duration_hours,
                                    probability=rainfall.probability
                                )
                                
                                if send_result.get("sent", 0) > 0:
                                    results["rainfall_alerts"] += 1
                                    self._record_alert_sent(alert_key, "rainfall_forecast", name, {
                                        "rainfall_mm": rainfall.total_precipitation_mm,
                                        "start_date": rainfall.start_date
                                    })
                    
                    # Check heat forecast
                    heat = forecaster.predict_next_heat_event(lat, lon, name)
                    if heat and heat.max_temperature >= self.thresholds["temperature_c"]:
                        if heat.days_until <= 1:  # Alert if heat within 1 day
                            alert_key = f"heat_forecast_{name}_{heat.start_date}"
                            
                            if self._should_send_alert(alert_key):
                                logger.warning(f"🔥 EXTREME HEAT FORECAST for {name}: {heat.max_temperature}°C")
                                
                                send_result = alert_system.send_heat_alert(
                                    location=f"{name}, North Tongu",
                                    max_temp=heat.max_temperature,
                                    heat_index=heat.max_heat_index,
                                    duration=f"{heat.duration_days} day(s)"
                                )
                                
                                if send_result.get("sent", 0) > 0:
                                    results["heat_alerts"] += 1
                                    self._record_alert_sent(alert_key, "heat_forecast", name, {
                                        "max_temp": heat.max_temperature,
                                        "start_date": heat.start_date
                                    })
                                    
                except Exception as e:
                    results["errors"].append(f"{name}: {str(e)}")
            
            results["status"] = "success"
            
        except ImportError as e:
            logger.warning(f"Event forecasting not available: {e}")
            results["status"] = "unavailable"
        except Exception as e:
            logger.error(f"Error checking event forecasts: {e}")
            results["status"] = "error"
            results["errors"].append(str(e))
        
        return results


class CCMEWSScheduler:
    """Automated scheduler for climate data updates, predictions, and SMS alerts"""
    
    def __init__(self, interval_hours: int = 5):
        self.interval_hours = interval_hours
        self.running = False
        self.thread = None
        self.last_run = None
        self.next_run = None
        self.alert_checker = AutomaticAlertChecker()
        self.status = {
            "last_update": None,
            "next_update": None,
            "total_updates": 0,
            "last_status": "idle",
            "last_error": None,
            "alerts_sent_today": 0
        }
    
    def run_update_cycle(self) -> dict:
        """Execute complete data update, prediction, and alert cycle"""
        start_time = datetime.now()
        logger.info("=" * 70)
        logger.info(f"🚀 CCMEWS UPDATE CYCLE STARTED - {start_time.isoformat()}")
        logger.info("=" * 70)
        
        result = {
            "started_at": start_time.isoformat(),
            "data_ingestion": None,
            "predictions": None,
            "automatic_alerts": None,
            "forecast_alerts": None,
            "alerts": [],
            "status": "running"
        }
        
        try:
            # Step 1: Fetch climate data
            logger.info("\n📡 STEP 1: Climate Data Ingestion")
            logger.info("-" * 50)
            
            from ccmews_data_service import ClimateDataService, MONITORING_GRID
            
            data_service = ClimateDataService()
            locations, records = data_service.fetch_all_locations()
            
            result["data_ingestion"] = {
                "locations_processed": locations,
                "records_inserted": records,
                "status": "success" if locations > 0 else "partial"
            }
            logger.info(f"✅ Data ingestion complete: {locations} locations, {records} records")
            
            # Step 2: Run AI predictions
            logger.info("\n🧠 STEP 2: AI Hazard Predictions")
            logger.info("-" * 50)
            
            from ccmews_ai_engine import run_prediction_cycle
            
            prediction_result = run_prediction_cycle(MONITORING_GRID)
            result["predictions"] = prediction_result
            
            # Extract critical alerts
            if prediction_result.get("critical_alerts"):
                result["alerts"] = prediction_result["critical_alerts"]
                logger.warning(f"⚠️ {len(result['alerts'])} critical alerts generated!")
            
            # Step 3: AUTOMATIC SMS ALERTS based on predictions
            logger.info("\n📲 STEP 3: Automatic SMS Alerts")
            logger.info("-" * 50)
            
            # Get predictions list for alert checking
            predictions_list = prediction_result.get("predictions", [])
            if predictions_list:
                alert_result = self.alert_checker.check_and_alert(predictions_list)
                result["automatic_alerts"] = alert_result
                
                if alert_result.get("alerts_sent", 0) > 0:
                    self.status["alerts_sent_today"] += alert_result["alerts_sent"]
                    logger.info(f"📤 Sent {alert_result['alerts_sent']} automatic SMS alerts")
            else:
                logger.info("No predictions to check for alerts")
            
            # Step 4: Check event forecasts for additional alerts
            logger.info("\n🌤️ STEP 4: Event Forecast Alerts")
            logger.info("-" * 50)
            
            forecast_alert_result = self.alert_checker.check_event_forecasts()
            result["forecast_alerts"] = forecast_alert_result
            
            total_forecast_alerts = (forecast_alert_result.get("rainfall_alerts", 0) + 
                                    forecast_alert_result.get("heat_alerts", 0))
            if total_forecast_alerts > 0:
                self.status["alerts_sent_today"] += total_forecast_alerts
                logger.info(f"📤 Sent {total_forecast_alerts} forecast-based alerts")
            
            result["status"] = "success"
            self.status["last_status"] = "success"
            self.status["last_error"] = None
            
        except Exception as e:
            logger.error(f"❌ Update cycle failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            self.status["last_status"] = "error"
            self.status["last_error"] = str(e)
        
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result["completed_at"] = end_time.isoformat()
            result["duration_seconds"] = duration
            
            # Update scheduler status
            self.status["last_update"] = end_time.isoformat()
            self.status["total_updates"] += 1
            self.last_run = end_time
            self._save_status()
            
            logger.info("=" * 70)
            logger.info(f"✨ UPDATE CYCLE COMPLETED in {duration:.1f}s")
            logger.info(f"   Status: {result['status']}")
            if result.get("data_ingestion"):
                logger.info(f"   Data: {result['data_ingestion']['records_inserted']} records")
            if result.get("predictions"):
                logger.info(f"   Predictions: {result['predictions'].get('predictions_generated', 0)}")
            if result.get("automatic_alerts"):
                logger.info(f"   SMS Alerts: {result['automatic_alerts'].get('alerts_sent', 0)} sent")
            logger.info("=" * 70)
        
        return result
    
    def _scheduler_loop(self):
        """Background scheduler loop"""
        interval_seconds = self.interval_hours * 3600
        
        while self.running:
            # Calculate next run time
            self.next_run = datetime.now() + timedelta(hours=self.interval_hours)
            self.status["next_update"] = self.next_run.isoformat()
            self._save_status()
            
            # Wait for next update (check every minute if still running)
            wait_until = time.time() + interval_seconds
            while self.running and time.time() < wait_until:
                time.sleep(60)  # Check every minute
            
            # Run update if still running
            if self.running:
                self.run_update_cycle()
    
    def _save_status(self):
        """Save scheduler status to file"""
        try:
            with open(STATUS_FILE, 'w') as f:
                json.dump(self.status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save status: {e}")
    
    def _load_status(self):
        """Load scheduler status from file"""
        try:
            if STATUS_FILE.exists():
                with open(STATUS_FILE) as f:
                    self.status = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load status: {e}")
    
    def start(self, run_immediately: bool = True):
        """Start the scheduler"""
        logger.info(f"🚀 Starting CCMEWS Scheduler")
        logger.info(f"   Update interval: {self.interval_hours} hours")
        
        self._load_status()
        
        if run_immediately:
            logger.info("   Running initial update cycle...")
            self.run_update_cycle()
        
        self.running = True
        self.thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.thread.start()
        
        self.next_run = datetime.now() + timedelta(hours=self.interval_hours)
        self.status["next_update"] = self.next_run.isoformat()
        self._save_status()
        
        logger.info(f"✅ Scheduler started. Next update: {self.next_run}")
    
    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping scheduler...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def get_status(self) -> dict:
        """Get current scheduler status"""
        self._load_status()
        return {
            **self.status,
            "running": self.running,
            "interval_hours": self.interval_hours,
            "next_run": self.next_run.isoformat() if self.next_run else None
        }
    
    def force_update(self) -> dict:
        """Force an immediate update"""
        logger.info("Manual update triggered")
        return self.run_update_cycle()


# Global scheduler instance
_scheduler = None


def get_scheduler():
    """Get or create scheduler instance"""
    global _scheduler
    if _scheduler is None:
        _scheduler = CCMEWSScheduler(UPDATE_INTERVAL_HOURS)
    return _scheduler


def start_scheduler(run_immediately: bool = True):
    """Start the background scheduler"""
    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start(run_immediately=run_immediately)
    return scheduler


def get_system_status() -> dict:
    """Get complete system status for dashboard"""
    scheduler = get_scheduler()
    scheduler_status = scheduler.get_status()
    
    # Get data status
    try:
        from ccmews_data_service import ClimateDataService
        data_service = ClimateDataService()
        fetch_status = data_service.get_fetch_status()
        latest_obs = data_service.get_latest_observations()
        data_status = {
            "last_fetch": fetch_status,
            "observation_count": len(latest_obs),
            "latest_time": latest_obs[0]['observation_time'] if latest_obs else None
        }
    except Exception as e:
        data_status = {"error": str(e)}
    
    # Get prediction status
    try:
        from ccmews_ai_engine import HazardPredictionEngine
        engine = HazardPredictionEngine()
        predictions = engine.get_predictions()
        
        # Count by risk level
        risk_counts = {"critical": 0, "high": 0, "moderate": 0, "low": 0}
        for pred in predictions:
            level = "critical" if pred['composite_risk'] >= 0.65 else \
                    "high" if pred['composite_risk'] >= 0.45 else \
                    "moderate" if pred['composite_risk'] >= 0.25 else "low"
            risk_counts[level] += 1
        
        prediction_status = {
            "total_predictions": len(predictions),
            "risk_counts": risk_counts
        }
    except Exception as e:
        prediction_status = {"error": str(e)}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "scheduler": scheduler_status,
        "data": data_status,
        "predictions": prediction_status,
        "update_interval_hours": UPDATE_INTERVAL_HOURS
    }


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CCMEWS Scheduler Service")
    parser.add_argument("--run-once", action="store_true", help="Run single update and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--interval", type=int, default=5, help="Update interval (hours)")
    
    args = parser.parse_args()
    
    if args.status:
        status = get_system_status()
        print(json.dumps(status, indent=2, default=str))
        return
    
    if args.run_once:
        scheduler = CCMEWSScheduler(args.interval)
        result = scheduler.run_update_cycle()
        print(json.dumps(result, indent=2, default=str))
        return
    
    if args.daemon:
        scheduler = CCMEWSScheduler(args.interval)
        scheduler.start()
        
        print(f"CCMEWS Scheduler running (Ctrl+C to stop)")
        print(f"Update interval: {args.interval} hours")
        
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\nShutting down...")
            scheduler.stop()
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
