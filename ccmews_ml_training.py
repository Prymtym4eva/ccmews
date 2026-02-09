"""
CCMEWS Machine Learning Training Module
Fetches historical data from Open-Meteo Archive API and trains real ML models
for flood, heat, and drought prediction.

Open-Meteo provides historical data back to 1940 for some variables!
"""
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Run: pip install scikit-learn")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CCMEWS-ML')

# Paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ccmews_climate.db"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Open-Meteo Historical Archive API
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"

# North Tongu monitoring locations (subset for training)
TRAINING_LOCATIONS = [
    ("Battor", 6.0833, 0.4200),
    ("Adidome", 6.1200, 0.3150),
    ("Sogakope", 6.0100, 0.4200),
    ("Mepe", 6.0500, 0.3850),
    ("Mafi Kumase", 6.0900, 0.3500),
]

# =============================================================================
# HAZARD THRESHOLDS FOR LABEL GENERATION
# =============================================================================
THRESHOLDS = {
    "flood": {
        "precip_daily_high": 50,      # mm/day - high flood risk
        "precip_daily_moderate": 30,   # mm/day - moderate flood risk
        "precip_3day_high": 100,       # mm/3days cumulative
    },
    "heat": {
        "temp_extreme": 40,            # °C - extreme heat
        "temp_high": 37,               # °C - high heat risk
        "temp_moderate": 35,           # °C - moderate heat risk
    },
    "drought": {
        "dry_days_severe": 14,         # consecutive days < 1mm
        "dry_days_moderate": 7,
        "precip_weekly_low": 5,        # mm/week - drought indicator
    }
}


class HistoricalDataFetcher:
    """Fetches historical weather data from Open-Meteo Archive API"""
    
    def __init__(self):
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
    
    def fetch_historical_data(
        self, 
        lat: float, 
        lon: float, 
        start_date: str, 
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical daily weather data from Open-Meteo Archive
        
        Args:
            lat: Latitude
            lon: Longitude  
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with daily weather data
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available")
            return None
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "precipitation_hours",
                "rain_sum",
                "relative_humidity_2m_mean",
                "relative_humidity_2m_max",
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "et0_fao_evapotranspiration",
                "shortwave_radiation_sum"
            ],
            "timezone": "Africa/Accra"
        }
        
        try:
            response = self.session.get(ARCHIVE_API, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                daily = data.get("daily", {})
                
                if not daily.get("time"):
                    return None
                
                df = pd.DataFrame({
                    "date": pd.to_datetime(daily["time"]),
                    "temp_max": daily.get("temperature_2m_max"),
                    "temp_min": daily.get("temperature_2m_min"),
                    "temp_mean": daily.get("temperature_2m_mean"),
                    "precip": daily.get("precipitation_sum"),
                    "precip_hours": daily.get("precipitation_hours"),
                    "rain": daily.get("rain_sum"),
                    "humidity_mean": daily.get("relative_humidity_2m_mean"),
                    "humidity_max": daily.get("relative_humidity_2m_max"),
                    "wind_max": daily.get("wind_speed_10m_max"),
                    "wind_gusts": daily.get("wind_gusts_10m_max"),
                    "et0": daily.get("et0_fao_evapotranspiration"),
                    "radiation": daily.get("shortwave_radiation_sum")
                })
                
                df["lat"] = lat
                df["lon"] = lon
                
                return df
            else:
                logger.error(f"API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return None
    
    def fetch_multi_location(
        self,
        locations: List[Tuple[str, float, float]],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch historical data for multiple locations"""
        all_data = []
        
        for name, lat, lon in locations:
            logger.info(f"Fetching historical data for {name}...")
            df = self.fetch_historical_data(lat, lon, start_date, end_date)
            
            if df is not None:
                df["location"] = name
                all_data.append(df)
                logger.info(f"  ✓ {len(df)} days of data")
            else:
                logger.warning(f"  ✗ Failed to fetch data for {name}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


class FeatureEngineer:
    """Creates ML features from weather data"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for ML training
        
        Features include:
        - Rolling statistics (7, 14, 30 days)
        - Lag features
        - Trend indicators
        - Seasonal components
        - Derived indices
        """
        df = df.copy()
        df = df.sort_values(["location", "date"]).reset_index(drop=True)
        
        features = pd.DataFrame()
        
        for location in df["location"].unique():
            loc_df = df[df["location"] == location].copy()
            loc_df = loc_df.sort_values("date").reset_index(drop=True)
            
            # === BASIC FEATURES ===
            loc_df["temp_range"] = loc_df["temp_max"] - loc_df["temp_min"]
            loc_df["precip_intensity"] = loc_df["precip"] / (loc_df["precip_hours"] + 0.1)
            
            # === ROLLING FEATURES ===
            for window in [3, 7, 14, 30]:
                # Temperature
                loc_df[f"temp_max_roll{window}"] = loc_df["temp_max"].rolling(window).mean()
                loc_df[f"temp_max_std{window}"] = loc_df["temp_max"].rolling(window).std()
                loc_df[f"temp_min_roll{window}"] = loc_df["temp_min"].rolling(window).min()
                
                # Precipitation
                loc_df[f"precip_sum{window}"] = loc_df["precip"].rolling(window).sum()
                loc_df[f"precip_max{window}"] = loc_df["precip"].rolling(window).max()
                loc_df[f"precip_days{window}"] = (loc_df["precip"] >= 1).rolling(window).sum()
                
                # Humidity
                loc_df[f"humidity_roll{window}"] = loc_df["humidity_mean"].rolling(window).mean()
                
                # ET0 (evapotranspiration)
                if "et0" in loc_df.columns:
                    loc_df[f"et0_sum{window}"] = loc_df["et0"].rolling(window).sum()
            
            # === DRY SPELL TRACKING ===
            loc_df["is_dry_day"] = (loc_df["precip"] < 1).astype(int)
            
            # Count consecutive dry days
            dry_spell = []
            count = 0
            for is_dry in loc_df["is_dry_day"]:
                if is_dry:
                    count += 1
                else:
                    count = 0
                dry_spell.append(count)
            loc_df["dry_spell_days"] = dry_spell
            
            # === LAG FEATURES ===
            for lag in [1, 2, 3, 7]:
                loc_df[f"temp_max_lag{lag}"] = loc_df["temp_max"].shift(lag)
                loc_df[f"precip_lag{lag}"] = loc_df["precip"].shift(lag)
                loc_df[f"humidity_lag{lag}"] = loc_df["humidity_mean"].shift(lag)
            
            # === TREND FEATURES ===
            loc_df["temp_trend_7d"] = loc_df["temp_max"] - loc_df["temp_max"].shift(7)
            loc_df["precip_trend_7d"] = loc_df["precip_sum7"] - loc_df["precip_sum7"].shift(7)
            
            # === SEASONAL FEATURES ===
            loc_df["day_of_year"] = loc_df["date"].dt.dayofyear
            loc_df["month"] = loc_df["date"].dt.month
            loc_df["is_rainy_season"] = loc_df["month"].isin([4, 5, 6, 7, 8, 9, 10]).astype(int)
            
            # Cyclical encoding
            loc_df["day_sin"] = np.sin(2 * np.pi * loc_df["day_of_year"] / 365)
            loc_df["day_cos"] = np.cos(2 * np.pi * loc_df["day_of_year"] / 365)
            
            # === HEAT INDEX ===
            loc_df["heat_index"] = FeatureEngineer._calculate_heat_index(
                loc_df["temp_max"], loc_df["humidity_max"]
            )
            
            # === DROUGHT INDEX (simplified) ===
            # Ratio of ET0 to precipitation
            loc_df["aridity_index"] = loc_df["et0_sum7"] / (loc_df["precip_sum7"] + 1)
            
            features = pd.concat([features, loc_df], ignore_index=True)
        
        return features
    
    @staticmethod
    def _calculate_heat_index(temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index (feels-like temperature)"""
        # Simplified Rothfusz regression
        hi = temp.copy()
        
        mask = temp >= 27
        T = temp[mask]
        R = humidity[mask]
        
        hi_calc = (-8.78469475556 + 
                   1.61139411 * T + 
                   2.33854883889 * R +
                   -0.14611605 * T * R +
                   -0.012308094 * T**2 +
                   -0.0164248277778 * R**2 +
                   0.002211732 * T**2 * R +
                   0.00072546 * T * R**2 +
                   -0.000003582 * T**2 * R**2)
        
        hi[mask] = hi_calc
        return hi


class LabelGenerator:
    """Generates training labels from historical weather data"""
    
    @staticmethod
    def create_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary labels for each hazard type
        
        Labels are based on actual weather conditions:
        - Flood: Heavy precipitation events
        - Heat: Extreme temperature events  
        - Drought: Extended dry periods
        """
        df = df.copy()
        
        # === FLOOD LABELS ===
        # High risk if: heavy daily precip OR high cumulative precip
        df["flood_risk_binary"] = (
            (df["precip"] >= THRESHOLDS["flood"]["precip_daily_moderate"]) |
            (df["precip_sum3"] >= THRESHOLDS["flood"]["precip_3day_high"])
        ).astype(int)
        
        # Multi-class flood risk
        df["flood_risk_level"] = 0  # Low
        df.loc[df["precip"] >= THRESHOLDS["flood"]["precip_daily_moderate"], "flood_risk_level"] = 1  # Moderate
        df.loc[df["precip"] >= THRESHOLDS["flood"]["precip_daily_high"], "flood_risk_level"] = 2  # High
        
        # === HEAT LABELS ===
        df["heat_risk_binary"] = (
            df["temp_max"] >= THRESHOLDS["heat"]["temp_moderate"]
        ).astype(int)
        
        # Multi-class heat risk
        df["heat_risk_level"] = 0  # Low
        df.loc[df["temp_max"] >= THRESHOLDS["heat"]["temp_moderate"], "heat_risk_level"] = 1  # Moderate
        df.loc[df["temp_max"] >= THRESHOLDS["heat"]["temp_high"], "heat_risk_level"] = 2  # High
        df.loc[df["temp_max"] >= THRESHOLDS["heat"]["temp_extreme"], "heat_risk_level"] = 3  # Extreme
        
        # === DROUGHT LABELS ===
        df["drought_risk_binary"] = (
            (df["dry_spell_days"] >= THRESHOLDS["drought"]["dry_days_moderate"]) |
            (df["precip_sum7"] <= THRESHOLDS["drought"]["precip_weekly_low"])
        ).astype(int)
        
        # Multi-class drought risk
        df["drought_risk_level"] = 0  # Low
        df.loc[df["dry_spell_days"] >= THRESHOLDS["drought"]["dry_days_moderate"], "drought_risk_level"] = 1
        df.loc[df["dry_spell_days"] >= THRESHOLDS["drought"]["dry_days_severe"], "drought_risk_level"] = 2
        
        return df


class HazardMLModels:
    """Train and manage ML models for hazard prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.model_info = {}
    
    def get_feature_columns(self) -> List[str]:
        """Define feature columns for ML training"""
        return [
            # Temperature features
            "temp_max", "temp_min", "temp_mean", "temp_range",
            "temp_max_roll3", "temp_max_roll7", "temp_max_roll14",
            "temp_max_std7", "temp_min_roll7",
            "temp_max_lag1", "temp_max_lag3", "temp_max_lag7",
            "temp_trend_7d",
            
            # Precipitation features
            "precip", "precip_hours", "precip_intensity",
            "precip_sum3", "precip_sum7", "precip_sum14", "precip_sum30",
            "precip_max7", "precip_days7", "precip_days14",
            "precip_lag1", "precip_lag3", "precip_lag7",
            "precip_trend_7d",
            
            # Humidity features
            "humidity_mean", "humidity_max",
            "humidity_roll7", "humidity_roll14",
            "humidity_lag1", "humidity_lag7",
            
            # Derived features
            "heat_index", "dry_spell_days", "aridity_index",
            
            # ET0 features
            "et0", "et0_sum7", "et0_sum14",
            
            # Seasonal features
            "day_sin", "day_cos", "is_rainy_season", "month"
        ]
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        target_col: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for training"""
        feature_cols = self.get_feature_columns()
        
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        
        # Drop rows with missing values
        clean_df = df[available_cols + [target_col]].dropna()
        
        X = clean_df[available_cols].values
        y = clean_df[target_col].values
        
        return X, y, available_cols
    
    def train_flood_model(self, df: pd.DataFrame) -> Dict:
        """Train flood risk prediction model"""
        logger.info("Training Flood Risk Model...")
        
        X, y, feature_cols = self.prepare_training_data(df, "flood_risk_binary")
        self.feature_columns = feature_cols
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["flood"] = scaler
        
        # Split with time-aware validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False  # No shuffle for time series
        )
        
        # Train ensemble
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=20,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": (y_pred == y_test).mean(),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_rate": y.mean()
        }
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='f1')
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()
        
        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
        metrics["top_features"] = top_features
        
        self.models["flood"] = model
        self.model_info["flood"] = metrics
        
        logger.info(f"  ✓ Flood Model - F1: {metrics['f1_score']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        
        return metrics
    
    def train_heat_model(self, df: pd.DataFrame) -> Dict:
        """Train heat risk prediction model"""
        logger.info("Training Heat Risk Model...")
        
        X, y, feature_cols = self.prepare_training_data(df, "heat_risk_binary")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["heat"] = scaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": (y_pred == y_test).mean(),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_rate": y.mean()
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='f1')
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()
        
        importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
        metrics["top_features"] = top_features
        
        self.models["heat"] = model
        self.model_info["heat"] = metrics
        
        logger.info(f"  ✓ Heat Model - F1: {metrics['f1_score']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        
        return metrics
    
    def train_drought_model(self, df: pd.DataFrame) -> Dict:
        """Train drought risk prediction model"""
        logger.info("Training Drought Risk Model...")
        
        X, y, feature_cols = self.prepare_training_data(df, "drought_risk_binary")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["drought"] = scaler
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False
        )
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=15,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": (y_pred == y_test).mean(),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_rate": y.mean()
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='f1')
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()
        
        importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
        metrics["top_features"] = top_features
        
        self.models["drought"] = model
        self.model_info["drought"] = metrics
        
        logger.info(f"  ✓ Drought Model - F1: {metrics['f1_score']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        
        return metrics
    
    def train_all(self, df: pd.DataFrame) -> Dict:
        """Train all hazard models"""
        results = {}
        results["flood"] = self.train_flood_model(df)
        results["heat"] = self.train_heat_model(df)
        results["drought"] = self.train_drought_model(df)
        return results
    
    def save_models(self, path: Path = MODELS_DIR):
        """Save trained models to disk"""
        for hazard_type, model in self.models.items():
            model_path = path / f"{hazard_type}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {hazard_type} model to {model_path}")
        
        # Save scalers
        scaler_path = path / "scalers.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save feature columns
        features_path = path / "feature_columns.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
        
        # Save model info
        info_path = path / "model_info.json"
        # Convert numpy types for JSON serialization
        info_serializable = {}
        for k, v in self.model_info.items():
            info_serializable[k] = {
                key: (float(val) if isinstance(val, (np.floating, np.integer)) else val)
                for key, val in v.items()
                if key != "top_features"
            }
            if "top_features" in v:
                info_serializable[k]["top_features"] = [
                    (feat, float(imp)) for feat, imp in v["top_features"]
                ]
        
        with open(info_path, 'w') as f:
            json.dump(info_serializable, f, indent=2)
        
        logger.info(f"✓ All models saved to {path}")
    
    def load_models(self, path: Path = MODELS_DIR):
        """Load trained models from disk"""
        for hazard_type in ["flood", "heat", "drought"]:
            model_path = path / f"{hazard_type}_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[hazard_type] = pickle.load(f)
                logger.info(f"Loaded {hazard_type} model")
        
        scaler_path = path / "scalers.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
        
        features_path = path / "feature_columns.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
        
        info_path = path / "model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        """
        Make predictions using trained models
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            Dictionary with predictions for each hazard
        """
        if not self.models:
            raise ValueError("No models loaded. Train or load models first.")
        
        predictions = {}
        
        # Get feature values
        available_cols = [c for c in self.feature_columns if c in features_df.columns]
        X = features_df[available_cols].fillna(0).values
        
        for hazard_type, model in self.models.items():
            scaler = self.scalers.get(hazard_type)
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            prob = model.predict_proba(X_scaled)[:, 1]
            pred_class = model.predict(X_scaled)
            
            predictions[hazard_type] = {
                "probability": float(prob[-1]) if len(prob) > 0 else 0,
                "class": int(pred_class[-1]) if len(pred_class) > 0 else 0,
                "probabilities": prob.tolist(),
            }
        
        # Composite risk
        if all(h in predictions for h in ["flood", "heat", "drought"]):
            predictions["composite"] = (
                0.4 * predictions["flood"]["probability"] +
                0.3 * predictions["heat"]["probability"] +
                0.3 * predictions["drought"]["probability"]
            )
        
        return predictions


def generate_synthetic_training_data(
    locations: List[Tuple[str, float, float]],
    years: int = 5
) -> pd.DataFrame:
    """
    Generate realistic synthetic weather data for ML training
    Based on typical West African (Ghana Volta Region) climate patterns
    
    This is used when Open-Meteo Archive API is not accessible.
    Replace with real data by running locally.
    """
    logger.info("Generating synthetic training data based on Ghana climate patterns...")
    
    np.random.seed(42)
    all_data = []
    
    # Ghana climate characteristics:
    # - Rainy season: April-October (bimodal: May-June, September-October peaks)
    # - Dry season: November-March (Harmattan winds)
    # - Temps: 25-35°C typical, can reach 40°C in dry season
    
    for name, lat, lon in locations:
        start_date = datetime.now() - timedelta(days=years * 365)
        dates = pd.date_range(start=start_date, periods=years * 365, freq='D')
        
        n_days = len(dates)
        
        # Day of year for seasonal patterns
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        
        # === TEMPERATURE ===
        # Base temp around 30°C with seasonal variation
        temp_seasonal = 30 + 3 * np.sin(2 * np.pi * (day_of_year - 60) / 365)  # Higher in dry season
        temp_mean = temp_seasonal + np.random.normal(0, 1.5, n_days)
        temp_max = temp_mean + np.random.uniform(4, 8, n_days)
        temp_min = temp_mean - np.random.uniform(3, 6, n_days)
        
        # Occasional heat waves (random spikes)
        heat_wave_prob = 0.05 * (1 + np.sin(2 * np.pi * (day_of_year - 60) / 365))  # More in dry season
        heat_wave = np.random.random(n_days) < heat_wave_prob
        temp_max[heat_wave] += np.random.uniform(3, 7, heat_wave.sum())
        
        # === PRECIPITATION ===
        # Bimodal rainfall pattern
        rain_prob_base = 0.15  # Base probability
        # Peak 1: May-June (day 120-180)
        rain_prob = rain_prob_base + 0.35 * np.exp(-((day_of_year - 150) ** 2) / (2 * 30 ** 2))
        # Peak 2: September-October (day 250-300)
        rain_prob += 0.30 * np.exp(-((day_of_year - 275) ** 2) / (2 * 25 ** 2))
        # Very low in dry season
        dry_season = (day_of_year < 90) | (day_of_year > 320)
        rain_prob[dry_season] *= 0.2
        
        # Generate rain events
        has_rain = np.random.random(n_days) < rain_prob
        precip = np.zeros(n_days)
        precip[has_rain] = np.random.exponential(15, has_rain.sum())  # Exponential distribution
        
        # Occasional heavy rainfall events
        heavy_rain_prob = 0.03 * rain_prob
        heavy_rain = np.random.random(n_days) < heavy_rain_prob
        precip[heavy_rain] += np.random.uniform(30, 80, heavy_rain.sum())
        
        precip = np.clip(precip, 0, 150)  # Cap at 150mm
        
        # Precipitation hours (roughly correlated with amount)
        precip_hours = np.zeros(n_days)
        precip_hours[precip > 0] = np.clip(precip[precip > 0] / 5 + np.random.normal(0, 2, (precip > 0).sum()), 1, 24)
        
        # === HUMIDITY ===
        # Higher in rainy season
        humidity_base = 70 + 15 * np.sin(2 * np.pi * (day_of_year - 180) / 365)
        humidity_mean = humidity_base + np.random.normal(0, 8, n_days)
        humidity_mean = np.clip(humidity_mean, 30, 95)
        humidity_max = np.clip(humidity_mean + np.random.uniform(5, 15, n_days), 40, 100)
        
        # === WIND ===
        # Higher during Harmattan (December-February)
        wind_base = 8 + 4 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
        wind_max = wind_base + np.random.exponential(3, n_days)
        wind_gusts = wind_max + np.random.exponential(5, n_days)
        
        # === EVAPOTRANSPIRATION (ET0) ===
        # Higher in dry season due to more radiation and wind
        et0 = 4 + 2 * np.sin(2 * np.pi * (day_of_year - 60) / 365) + np.random.normal(0, 0.5, n_days)
        et0 = np.clip(et0, 1, 8)
        
        # === RADIATION ===
        radiation = 18 + 6 * np.sin(2 * np.pi * (day_of_year - 172) / 365) + np.random.normal(0, 3, n_days)
        radiation = np.clip(radiation, 5, 30)
        
        df = pd.DataFrame({
            "date": dates,
            "location": name,
            "lat": lat,
            "lon": lon,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "temp_mean": temp_mean,
            "precip": precip,
            "precip_hours": precip_hours,
            "rain": precip,  # Assuming all precip is rain in this region
            "humidity_mean": humidity_mean,
            "humidity_max": humidity_max,
            "wind_max": wind_max,
            "wind_gusts": wind_gusts,
            "et0": et0,
            "radiation": radiation
        })
        
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"  ✓ Generated {len(combined)} days of synthetic data for {len(locations)} locations")
    
    return combined


def run_training_pipeline(
    years_of_history: int = 5,
    locations: List[Tuple[str, float, float]] = None
) -> Dict:
    """
    Complete ML training pipeline
    
    Args:
        years_of_history: Number of years of historical data to fetch
        locations: List of (name, lat, lon) tuples
    
    Returns:
        Dictionary with training results
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not installed. Run: pip install scikit-learn"}
    
    if not REQUESTS_AVAILABLE:
        return {"error": "requests not installed. Run: pip install requests"}
    
    if locations is None:
        locations = TRAINING_LOCATIONS
    
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=years_of_history * 365)
    
    logger.info("=" * 60)
    logger.info("CCMEWS ML TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Historical data: {start_date} to {end_date}")
    logger.info(f"Locations: {len(locations)}")
    logger.info(f"Years of history: {years_of_history}")
    
    # Step 1: Fetch historical data
    logger.info("\n📡 Step 1: Fetching historical data from Open-Meteo...")
    fetcher = HistoricalDataFetcher()
    raw_data = fetcher.fetch_multi_location(
        locations,
        start_date.isoformat(),
        end_date.isoformat()
    )
    
    if raw_data.empty:
        logger.warning("Could not fetch from Open-Meteo API. Using synthetic data...")
        logger.warning("(Run this script locally for real historical data)")
        raw_data = generate_synthetic_training_data(locations, years_of_history)
    
    logger.info(f"  ✓ Fetched {len(raw_data)} days of data across {raw_data['location'].nunique()} locations")
    
    # Step 2: Feature engineering
    logger.info("\n🔧 Step 2: Engineering features...")
    featured_data = FeatureEngineer.create_features(raw_data)
    logger.info(f"  ✓ Created {len(featured_data.columns)} features")
    
    # Step 3: Generate labels
    logger.info("\n🏷️ Step 3: Generating labels...")
    labeled_data = LabelGenerator.create_labels(featured_data)
    
    # Log label distribution
    logger.info(f"  Flood events: {labeled_data['flood_risk_binary'].sum()} ({labeled_data['flood_risk_binary'].mean():.1%})")
    logger.info(f"  Heat events: {labeled_data['heat_risk_binary'].sum()} ({labeled_data['heat_risk_binary'].mean():.1%})")
    logger.info(f"  Drought events: {labeled_data['drought_risk_binary'].sum()} ({labeled_data['drought_risk_binary'].mean():.1%})")
    
    # Step 4: Train models
    logger.info("\n🧠 Step 4: Training ML models...")
    ml_models = HazardMLModels()
    training_results = ml_models.train_all(labeled_data)
    
    # Step 5: Save models
    logger.info("\n💾 Step 5: Saving models...")
    ml_models.save_models()
    
    # Save training data summary
    data_summary_path = MODELS_DIR / "training_data_summary.csv"
    labeled_data[["date", "location", "temp_max", "precip", "dry_spell_days",
                  "flood_risk_binary", "heat_risk_binary", "drought_risk_binary"]].to_csv(
        data_summary_path, index=False
    )
    logger.info(f"  Saved training data summary to {data_summary_path}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    
    for hazard, metrics in training_results.items():
        logger.info(f"\n{hazard.upper()} MODEL:")
        logger.info(f"  F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        logger.info(f"  CV F1 (mean±std): {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
        logger.info(f"  Top 5 features: {[f[0] for f in metrics['top_features'][:5]]}")
    
    return {
        "status": "success",
        "data_points": len(labeled_data),
        "locations": len(locations),
        "years": years_of_history,
        "results": {
            hazard: {
                "f1_score": float(m["f1_score"]),
                "roc_auc": float(m["roc_auc"]),
                "cv_f1_mean": float(m["cv_f1_mean"])
            }
            for hazard, m in training_results.items()
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CCMEWS ML Training")
    parser.add_argument("--years", type=int, default=5, help="Years of historical data (default: 5)")
    parser.add_argument("--quick", action="store_true", help="Quick test with 1 year of data")
    
    args = parser.parse_args()
    
    years = 1 if args.quick else args.years
    
    print(f"\n🚀 Starting ML training with {years} year(s) of historical data...\n")
    
    results = run_training_pipeline(years_of_history=years)
    
    print("\n" + json.dumps(results, indent=2, default=str))
