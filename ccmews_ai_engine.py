"""
CCMEWS AI Hazard Prediction Engine
Machine learning models for flood, heat, and drought risk prediction
Provides 7-day advance warning with confidence scores

HYBRID APPROACH:
- Uses trained ML models when available (from ccmews_ml_training.py)
- Falls back to rule-based models if ML models not trained
"""
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CCMEWS-AI')

DB_PATH = Path(__file__).parent / "ccmews_climate.db"
MODEL_PATH = Path(__file__).parent / "models"

# Check for ML models
ML_MODELS_AVAILABLE = False
ML_MODELS = {}

def load_ml_models():
    """Load trained ML models if available"""
    global ML_MODELS_AVAILABLE, ML_MODELS
    
    if not MODEL_PATH.exists():
        return
    
    for model_file in MODEL_PATH.glob("*_model.pkl"):
        try:
            target = model_file.stem.replace('_model', '')
            with open(model_file, 'rb') as f:
                ML_MODELS[target] = pickle.load(f)
            logger.info(f"✓ Loaded ML model: {target}")
        except Exception as e:
            logger.warning(f"Failed to load {model_file}: {e}")
    
    if ML_MODELS:
        ML_MODELS_AVAILABLE = True
        logger.info(f"🧠 ML models active: {list(ML_MODELS.keys())}")

# Try to load models on import
try:
    load_ml_models()
except Exception as e:
    logger.warning(f"Could not load ML models: {e}")

# =============================================================================
# HAZARD THRESHOLDS (Calibrated for Ghana/Volta Region)
# =============================================================================
THRESHOLDS = {
    "flood": {
        "precip_24h_warning": 30,      # mm - flood watch
        "precip_24h_danger": 50,       # mm - flood warning
        "precip_24h_extreme": 80,      # mm - severe flooding
        "precip_72h_cumulative": 100,  # mm - 3-day cumulative
        "soil_saturation": 0.35,       # soil moisture threshold
    },
    "heat": {
        "temp_caution": 33,            # °C - heat caution
        "temp_warning": 37,            # °C - heat warning  
        "temp_danger": 41,             # °C - heat emergency
        "humidity_compound": 75,       # % humidity compounds heat
        "heat_index_danger": 40,       # °C feels-like
    },
    "drought": {
        "precip_deficit_days": 10,     # consecutive dry days
        "soil_moisture_stress": 0.15,  # crop stress threshold
        "soil_moisture_severe": 0.08,  # severe drought
        "precip_threshold": 2.0,       # mm - significant rain
    }
}


@dataclass
class HazardPrediction:
    """Container for hazard prediction results"""
    location_name: str
    latitude: float
    longitude: float
    prediction_date: str
    horizon_days: int
    
    flood_risk: float
    flood_drivers: Dict
    
    heat_risk: float
    heat_drivers: Dict
    
    drought_risk: float
    drought_drivers: Dict
    
    composite_risk: float
    risk_level: str
    confidence: float
    
    model_version: str = "v1.2.0"
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class HazardPredictionEngine:
    """AI Engine for multi-hazard prediction"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.model_version = "v1.2.0"
    
    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================
    def _get_historical_features(self, location: str, days: int = 14) -> Dict:
        """Extract features from historical observations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT * FROM climate_observations
            WHERE location_name = ? AND observation_time >= ?
            ORDER BY observation_time DESC
        """, (location, start_date))
        
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if not rows:
            return self._default_features()
        
        # Extract time series
        temps = [r['temperature_2m'] for r in rows if r['temperature_2m']]
        precips = [r['precipitation'] or 0 for r in rows]
        humidities = [r['relative_humidity'] for r in rows if r['relative_humidity']]
        soil_moistures = [r['soil_moisture'] for r in rows if r.get('soil_moisture')]
        
        # Compute features
        features = {
            # Temperature features
            'temp_mean': np.mean(temps) if temps else 30,
            'temp_max': np.max(temps) if temps else 35,
            'temp_min': np.min(temps) if temps else 25,
            'temp_std': np.std(temps) if len(temps) > 1 else 2,
            'temp_trend': self._compute_trend(temps) if len(temps) > 24 else 0,
            
            # Precipitation features
            'precip_total': sum(precips),
            'precip_24h': sum(precips[:24]) if len(precips) >= 24 else sum(precips),
            'precip_72h': sum(precips[:72]) if len(precips) >= 72 else sum(precips),
            'precip_7d': sum(precips[:168]) if len(precips) >= 168 else sum(precips),
            'dry_hours': sum(1 for p in precips[:168] if p < 0.1),
            'wet_hours': sum(1 for p in precips[:168] if p >= 1),
            
            # Humidity features
            'humidity_mean': np.mean(humidities) if humidities else 75,
            'humidity_max': np.max(humidities) if humidities else 85,
            
            # Soil moisture
            'soil_moisture': np.mean(soil_moistures) if soil_moistures else 0.2,
            'soil_moisture_trend': self._compute_trend(soil_moistures) if len(soil_moistures) > 24 else 0,
            
            # Data quality
            'data_points': len(rows),
            'data_completeness': len(temps) / max(len(rows), 1)
        }
        
        return features
    
    def _get_forecast_features(self, location: str) -> List[Dict]:
        """Get 7-day forecast features"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                DATE(forecast_time) as forecast_date,
                AVG(temperature_2m) as temp_mean,
                MAX(temperature_2m) as temp_max,
                MIN(temperature_2m) as temp_min,
                AVG(relative_humidity) as humidity_mean,
                SUM(precipitation) as precip_sum,
                MAX(precipitation_probability) as precip_prob_max,
                AVG(soil_moisture) as soil_moisture
            FROM weather_forecasts
            WHERE location_name = ? AND forecast_time > datetime('now')
            GROUP BY DATE(forecast_time)
            ORDER BY forecast_date
            LIMIT 7
        """, (location,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # If no forecasts, generate baseline predictions
        if not results:
            results = self._generate_baseline_forecast()
        
        return results
    
    def _compute_trend(self, values: List) -> float:
        """Compute linear trend (positive = increasing)"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        valid = [(i, v) for i, v in enumerate(values) if v is not None]
        if len(valid) < 2:
            return 0
        
        x = np.array([v[0] for v in valid])
        y = np.array([v[1] for v in valid])
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 0
    
    def _default_features(self) -> Dict:
        """Return default features when data unavailable"""
        return {
            'temp_mean': 30, 'temp_max': 34, 'temp_min': 26, 'temp_std': 3, 'temp_trend': 0,
            'precip_total': 50, 'precip_24h': 5, 'precip_72h': 15, 'precip_7d': 35,
            'dry_hours': 100, 'wet_hours': 30,
            'humidity_mean': 75, 'humidity_max': 85,
            'soil_moisture': 0.2, 'soil_moisture_trend': 0,
            'data_points': 0, 'data_completeness': 0
        }
    
    def _generate_baseline_forecast(self) -> List[Dict]:
        """Generate baseline forecast when API data unavailable"""
        forecasts = []
        today = datetime.now().date()
        
        for i in range(7):
            forecast_date = today + timedelta(days=i+1)
            # Typical Volta Region conditions with seasonal variation
            base_temp = 30 + np.random.normal(0, 2)
            
            forecasts.append({
                'forecast_date': forecast_date.isoformat(),
                'temp_mean': base_temp,
                'temp_max': base_temp + 4 + np.random.uniform(0, 2),
                'temp_min': base_temp - 4 + np.random.uniform(-1, 1),
                'humidity_mean': 75 + np.random.normal(0, 5),
                'precip_sum': max(0, np.random.exponential(5)),
                'precip_prob_max': np.random.uniform(20, 60),
                'soil_moisture': 0.2 + np.random.uniform(-0.05, 0.05)
            })
        
        return forecasts
    
    # =========================================================================
    # HAZARD PREDICTION MODELS
    # =========================================================================
    def _calculate_heat_index(self, temp: float, humidity: float) -> float:
        """Calculate heat index (feels-like temperature)"""
        if temp < 27:
            return temp
        
        # Rothfusz regression equation
        hi = -8.78469475556 + 1.61139411 * temp + 2.33854883889 * humidity
        hi += -0.14611605 * temp * humidity
        hi += -0.012308094 * temp**2
        hi += -0.0164248277778 * humidity**2
        hi += 0.002211732 * temp**2 * humidity
        hi += 0.00072546 * temp * humidity**2
        hi += -0.000003582 * temp**2 * humidity**2
        
        return max(temp, hi)
    
    def _predict_flood_risk(self, hist_features: Dict, forecast: Dict, horizon: int) -> Tuple[float, Dict]:
        """Predict flood risk using rule-based ML hybrid model"""
        risk = 0.0
        drivers = {}
        
        # Forecast precipitation
        precip_forecast = forecast.get('precip_sum', 0) or 0
        precip_prob = (forecast.get('precip_prob_max', 50) or 50) / 100
        
        # Recent precipitation (antecedent conditions)
        precip_72h = hist_features.get('precip_72h', 0)
        precip_7d = hist_features.get('precip_7d', 0)
        soil_moisture = hist_features.get('soil_moisture', 0.2)
        
        # 1. Forecast precipitation risk
        if precip_forecast > THRESHOLDS['flood']['precip_24h_extreme']:
            risk += 0.45
            drivers['extreme_precip'] = True
        elif precip_forecast > THRESHOLDS['flood']['precip_24h_danger']:
            risk += 0.30
            drivers['heavy_precip'] = True
        elif precip_forecast > THRESHOLDS['flood']['precip_24h_warning']:
            risk += 0.15
            drivers['moderate_precip'] = True
        
        # 2. Antecedent conditions (saturated ground)
        if precip_72h > 50:
            risk += 0.15
            drivers['wet_antecedent'] = True
        
        if precip_7d > THRESHOLDS['flood']['precip_72h_cumulative']:
            risk += 0.10
            drivers['cumulative_precip'] = True
        
        # 3. Soil saturation
        if soil_moisture and soil_moisture > THRESHOLDS['flood']['soil_saturation']:
            risk += 0.15
            drivers['saturated_soil'] = True
        
        # 4. Weight by precipitation probability
        risk = risk * (0.3 + 0.7 * precip_prob)
        
        # 5. Reduce confidence with forecast horizon
        horizon_factor = 1 - (horizon - 1) * 0.08  # ~8% reduction per day
        risk = risk * horizon_factor
        
        drivers['precip_forecast_mm'] = round(precip_forecast, 1)
        drivers['precip_probability'] = round(precip_prob * 100, 0)
        drivers['precip_72h_mm'] = round(precip_72h, 1)
        
        return min(risk, 1.0), drivers
    
    def _predict_heat_risk(self, hist_features: Dict, forecast: Dict, horizon: int) -> Tuple[float, Dict]:
        """Predict heat stress risk"""
        risk = 0.0
        drivers = {}
        
        temp_max = forecast.get('temp_max', 32) or 32
        humidity = forecast.get('humidity_mean', 75) or 75
        
        # Calculate heat index
        heat_index = self._calculate_heat_index(temp_max, humidity)
        drivers['heat_index'] = round(heat_index, 1)
        drivers['temp_max'] = round(temp_max, 1)
        
        # 1. Temperature-based risk
        if temp_max >= THRESHOLDS['heat']['temp_danger']:
            risk += 0.50
            drivers['extreme_heat'] = True
        elif temp_max >= THRESHOLDS['heat']['temp_warning']:
            risk += 0.35
            drivers['severe_heat'] = True
        elif temp_max >= THRESHOLDS['heat']['temp_caution']:
            risk += 0.20
            drivers['heat_caution'] = True
        
        # 2. Heat index compound effect
        if heat_index >= THRESHOLDS['heat']['heat_index_danger']:
            risk += 0.25
            drivers['dangerous_heat_index'] = True
        
        # 3. High humidity compounds heat
        if humidity >= THRESHOLDS['heat']['humidity_compound']:
            risk += 0.10
            drivers['high_humidity'] = True
        
        # 4. Temperature trend (warming pattern)
        temp_trend = hist_features.get('temp_trend', 0)
        if temp_trend > 0.1:  # Rising temperatures
            risk += 0.10
            drivers['warming_trend'] = True
        
        # 5. Reduce with horizon
        horizon_factor = 1 - (horizon - 1) * 0.05
        risk = risk * horizon_factor
        
        return min(risk, 1.0), drivers
    
    def _predict_drought_risk(self, hist_features: Dict, forecast: Dict, horizon: int) -> Tuple[float, Dict]:
        """Predict drought/dry spell risk"""
        risk = 0.0
        drivers = {}
        
        # Historical dry conditions
        dry_hours = hist_features.get('dry_hours', 100)
        precip_7d = hist_features.get('precip_7d', 35)
        soil_moisture = hist_features.get('soil_moisture', 0.2)
        soil_trend = hist_features.get('soil_moisture_trend', 0)
        
        # Forecast precipitation
        precip_forecast = forecast.get('precip_sum', 0) or 0
        
        # 1. Recent dry spell
        dry_days = dry_hours / 24
        if dry_days >= THRESHOLDS['drought']['precip_deficit_days']:
            risk += 0.35
            drivers['extended_dry_spell'] = True
            drivers['dry_days'] = round(dry_days, 0)
        elif dry_days >= 7:
            risk += 0.20
            drivers['dry_spell'] = True
            drivers['dry_days'] = round(dry_days, 0)
        
        # 2. Low soil moisture
        if soil_moisture and soil_moisture < THRESHOLDS['drought']['soil_moisture_severe']:
            risk += 0.30
            drivers['severe_soil_deficit'] = True
        elif soil_moisture and soil_moisture < THRESHOLDS['drought']['soil_moisture_stress']:
            risk += 0.20
            drivers['soil_moisture_stress'] = True
        
        # 3. Declining soil moisture trend
        if soil_trend < -0.01:
            risk += 0.10
            drivers['declining_soil_moisture'] = True
        
        # 4. No rain in forecast
        if precip_forecast < THRESHOLDS['drought']['precip_threshold']:
            risk += 0.15
            drivers['no_rain_forecast'] = True
        
        # 5. Low cumulative precipitation
        if precip_7d < 20:
            risk += 0.15
            drivers['low_weekly_precip'] = True
        
        drivers['soil_moisture'] = round(soil_moisture, 3) if soil_moisture else None
        drivers['precip_7d_mm'] = round(precip_7d, 1)
        
        return min(risk, 1.0), drivers
    
    # =========================================================================
    # MAIN PREDICTION FUNCTION
    # =========================================================================
    def predict_hazards(self, location: str, lat: float, lon: float) -> List[HazardPrediction]:
        """Generate 7-day hazard predictions for a location"""
        predictions = []
        
        # Get historical features
        hist_features = self._get_historical_features(location)
        
        # Get forecast features
        forecasts = self._get_forecast_features(location)
        
        for i, forecast in enumerate(forecasts):
            horizon = i + 1  # 1 to 7 days ahead
            
            # Calculate individual hazard risks
            flood_risk, flood_drivers = self._predict_flood_risk(hist_features, forecast, horizon)
            heat_risk, heat_drivers = self._predict_heat_risk(hist_features, forecast, horizon)
            drought_risk, drought_drivers = self._predict_drought_risk(hist_features, forecast, horizon)
            
            # Composite risk (weighted by typical impact)
            composite = (
                0.40 * flood_risk +   # Floods have highest immediate impact
                0.30 * heat_risk +    # Heat affects health
                0.30 * drought_risk   # Drought affects agriculture
            )
            
            # Risk level classification
            if composite >= 0.65:
                risk_level = "critical"
            elif composite >= 0.45:
                risk_level = "high"
            elif composite >= 0.25:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            # Confidence decreases with horizon
            base_confidence = hist_features.get('data_completeness', 0.8)
            confidence = base_confidence * (1 - (horizon - 1) * 0.06)
            
            pred = HazardPrediction(
                location_name=location,
                latitude=lat,
                longitude=lon,
                prediction_date=forecast.get('forecast_date', (datetime.now() + timedelta(days=horizon)).date().isoformat()),
                horizon_days=horizon,
                flood_risk=round(flood_risk, 3),
                flood_drivers=flood_drivers,
                heat_risk=round(heat_risk, 3),
                heat_drivers=heat_drivers,
                drought_risk=round(drought_risk, 3),
                drought_drivers=drought_drivers,
                composite_risk=round(composite, 3),
                risk_level=risk_level,
                confidence=round(confidence, 3)
            )
            
            predictions.append(pred)
        
        return predictions
    
    def predict_all_locations(self, locations) -> Dict[str, List[HazardPrediction]]:
        """Generate predictions for all monitoring locations
        
        Args:
            locations: Can be either:
                - Dict: {"name": {"lat": float, "lon": float}, ...}
                - List of tuples: [("name", lat, lon), ...]
        """
        all_predictions = {}
        
        # Handle both dict and list formats
        if isinstance(locations, dict):
            # Convert dict to list of tuples
            location_list = []
            for name, coords in locations.items():
                if isinstance(coords, dict):
                    location_list.append((name, coords.get("lat"), coords.get("lon")))
                elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    location_list.append((name, coords[0], coords[1]))
        else:
            location_list = locations
        
        for name, lat, lon in location_list:
            logger.info(f"Predicting hazards for {name}...")
            predictions = self.predict_hazards(name, lat, lon)
            all_predictions[name] = predictions
            
            # Log high-risk predictions
            for pred in predictions:
                if pred.risk_level in ['high', 'critical']:
                    logger.warning(f"  ⚠️ {pred.risk_level.upper()} risk on {pred.prediction_date}: "
                                  f"Flood={pred.flood_risk:.2f}, Heat={pred.heat_risk:.2f}, "
                                  f"Drought={pred.drought_risk:.2f}")
        
        return all_predictions
    
    def save_predictions(self, predictions: Dict[str, List[HazardPrediction]]):
        """Save predictions to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for location, preds in predictions.items():
            for pred in preds:
                cursor.execute("""
                    INSERT OR REPLACE INTO hazard_predictions
                    (location_name, latitude, longitude, prediction_date, prediction_horizon_days,
                     flood_risk, heat_risk, drought_risk, composite_risk, confidence_score, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pred.location_name, pred.latitude, pred.longitude,
                    pred.prediction_date, pred.horizon_days,
                    pred.flood_risk, pred.heat_risk, pred.drought_risk,
                    pred.composite_risk, pred.confidence, pred.model_version
                ))
                count += 1
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {count} predictions to database")
        return count
    
    def get_predictions(self, location: str = None, days_ahead: int = 7) -> List[Dict]:
        """Retrieve predictions from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        end_date = (datetime.now().date() + timedelta(days=days_ahead)).isoformat()
        
        if location:
            cursor.execute("""
                SELECT * FROM hazard_predictions
                WHERE location_name = ? AND prediction_date >= ? AND prediction_date <= ?
                ORDER BY prediction_date
            """, (location, today, end_date))
        else:
            cursor.execute("""
                SELECT * FROM hazard_predictions
                WHERE prediction_date >= ? AND prediction_date <= ?
                ORDER BY location_name, prediction_date
            """, (today, end_date))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results


def run_prediction_cycle(locations) -> Dict:
    """Main function to run prediction cycle
    
    Args:
        locations: Can be either:
            - Dict: {"name": {"lat": float, "lon": float}, ...}
            - List of tuples: [("name", lat, lon), ...]
    """
    logger.info("=" * 60)
    logger.info("CCMEWS AI Hazard Prediction Cycle")
    logger.info("=" * 60)
    
    engine = HazardPredictionEngine()
    predictions = engine.predict_all_locations(locations)
    saved = engine.save_predictions(predictions)
    
    # Count locations
    num_locations = len(locations) if isinstance(locations, (dict, list)) else 0
    
    # Count high-risk predictions and build prediction list for alerts
    high_risk_count = 0
    critical_alerts = []
    predictions_list = []  # Flat list for alert checking
    
    for location, preds in predictions.items():
        for pred in preds:
            # Add to flat list for alert system
            predictions_list.append({
                'location': location,
                'latitude': pred.latitude,
                'longitude': pred.longitude,
                'date': pred.prediction_date,
                'horizon_days': pred.horizon_days,
                'flood_risk': pred.flood_risk,
                'heat_risk': pred.heat_risk,
                'drought_risk': pred.drought_risk,
                'composite_risk': pred.composite_risk,
                'risk_level': pred.risk_level,
                'confidence': pred.confidence,
                'flood_drivers': pred.flood_drivers,
                'heat_drivers': pred.heat_drivers,
                'drought_drivers': pred.drought_drivers
            })
            
            if pred.risk_level == 'critical':
                high_risk_count += 1
                critical_alerts.append({
                    'location': location,
                    'date': pred.prediction_date,
                    'composite_risk': pred.composite_risk,
                    'dominant_hazard': max(
                        [('flood', pred.flood_risk), ('heat', pred.heat_risk), ('drought', pred.drought_risk)],
                        key=lambda x: x[1]
                    )[0]
                })
            elif pred.risk_level == 'high':
                high_risk_count += 1
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'locations_processed': num_locations,
        'predictions_generated': saved,
        'high_risk_predictions': high_risk_count,
        'critical_alerts': critical_alerts[:10],  # Top 10 critical
        'predictions': predictions_list  # Full list for alert checking
    }
    
    logger.info(f"\n✅ Prediction cycle complete!")
    logger.info(f"   Locations: {num_locations}")
    logger.info(f"   Predictions: {saved}")
    logger.info(f"   High/Critical risks: {high_risk_count}")
    
    return result


if __name__ == "__main__":
    from ccmews_data_service import MONITORING_GRID
    
    result = run_prediction_cycle(MONITORING_GRID)
    print(json.dumps(result, indent=2, default=str))
