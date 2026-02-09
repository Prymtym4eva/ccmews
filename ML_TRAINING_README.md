# CCMEWS Machine Learning Training Module

## Overview

This module trains **real machine learning models** for climate hazard prediction using historical weather data from the **Open-Meteo Archive API** (free, no API key required).

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML TRAINING PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DATA FETCHING          2. FEATURE ENGINEERING      3. MODEL TRAINING    │
│  ─────────────────         ───────────────────────     ─────────────────    │
│                                                                             │
│  ┌─────────────────┐       ┌─────────────────────┐     ┌─────────────────┐  │
│  │ Open-Meteo      │       │ Rolling Statistics  │     │ Gradient        │  │
│  │ Archive API     │──────▶│ Lag Features        │────▶│ Boosting        │  │
│  │ (5+ years)      │       │ Trend Indicators    │     │ Random Forest   │  │
│  └─────────────────┘       │ Seasonal Encoding   │     └─────────────────┘  │
│                            │ Heat Index          │              │           │
│                            │ Aridity Index       │              ▼           │
│                            └─────────────────────┘     ┌─────────────────┐  │
│                                                        │ Trained Models  │  │
│                                                        │ (.pkl files)    │  │
│                                                        └─────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install scikit-learn pandas numpy requests
```

### 2. Run Training (5 years of data)

```bash
python ccmews_ml_training.py --years 5
```

### 3. Quick Test (1 year of data)

```bash
python ccmews_ml_training.py --quick
```

## Output

After training, you'll have:

```
models/
├── flood_model.pkl       # Gradient Boosting model for flood risk
├── heat_model.pkl        # Random Forest model for heat risk
├── drought_model.pkl     # Gradient Boosting model for drought risk
├── scalers.pkl           # Feature scalers for normalization
├── feature_columns.json  # List of features used by models
├── model_info.json       # Performance metrics for each model
└── training_data_summary.csv  # Summary of training data
```

## Model Details

### Flood Risk Model
- **Algorithm**: Gradient Boosting Classifier
- **Key Features**: `precip`, `precip_sum3`, `precip_sum7`
- **Thresholds**: 
  - Moderate risk: >30mm/day
  - High risk: >50mm/day
  - Very high: >100mm in 3 days

### Heat Risk Model
- **Algorithm**: Random Forest Classifier
- **Key Features**: `temp_max`, `temp_mean`, `heat_index`
- **Thresholds**:
  - Moderate risk: >35°C
  - High risk: >37°C
  - Extreme: >40°C

### Drought Risk Model
- **Algorithm**: Gradient Boosting Classifier
- **Key Features**: `precip_sum7`, `dry_spell_days`, `aridity_index`
- **Thresholds**:
  - Moderate: >7 consecutive dry days
  - Severe: >14 consecutive dry days
  - Low weekly precip: <5mm in 7 days

## Feature Engineering

The module creates **40+ features** from raw weather data:

### Temperature Features
- Daily max, min, mean temperature
- Rolling averages (3, 7, 14, 30 days)
- Temperature trends
- Lag features (1, 3, 7 days)

### Precipitation Features
- Daily precipitation and hours
- Cumulative sums (3, 7, 14, 30 days)
- Days with rain
- Precipitation intensity
- Dry spell tracking

### Derived Features
- **Heat Index**: Feels-like temperature (Rothfusz equation)
- **Aridity Index**: ET0 / Precipitation ratio
- **Seasonal encoding**: Cyclical day-of-year

## Using Trained Models

```python
from ccmews_ml_training import HazardMLModels, FeatureEngineer

# Load trained models
ml = HazardMLModels()
ml.load_models()

# Prepare your data (must have same columns as training data)
features_df = FeatureEngineer.create_features(your_data)

# Make predictions
predictions = ml.predict(features_df)

print(predictions)
# {
#   'flood': {'probability': 0.35, 'class': 0},
#   'heat': {'probability': 0.72, 'class': 1},
#   'drought': {'probability': 0.15, 'class': 0},
#   'composite': 0.41
# }
```

## Integration with AI Engine

The trained models are automatically loaded by `ccmews_ai_engine.py` if they exist:

```python
from ccmews_ai_engine import HazardPredictionEngine

engine = HazardPredictionEngine()
# If models exist in ./models/, they'll be used for prediction
# Otherwise, falls back to rule-based models
```

## Data Sources

### Open-Meteo Archive API
- **URL**: `https://archive-api.open-meteo.com/v1/archive`
- **Cost**: Free (no API key required)
- **Historical range**: 1940 - present (varies by variable)
- **Resolution**: Daily

### Variables Used
| Variable | Description |
|----------|-------------|
| `temperature_2m_max` | Daily maximum temperature |
| `temperature_2m_min` | Daily minimum temperature |
| `temperature_2m_mean` | Daily mean temperature |
| `precipitation_sum` | Total daily precipitation |
| `precipitation_hours` | Hours with precipitation |
| `relative_humidity_2m_mean` | Mean relative humidity |
| `relative_humidity_2m_max` | Maximum relative humidity |
| `wind_speed_10m_max` | Maximum wind speed |
| `wind_gusts_10m_max` | Maximum wind gusts |
| `et0_fao_evapotranspiration` | Reference evapotranspiration |
| `shortwave_radiation_sum` | Solar radiation |

## Performance Metrics

After training, check `models/model_info.json` for:
- **F1 Score**: Balance of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Cross-validation scores**: 5-fold time-series CV
- **Feature importance**: Top contributing features

## Customization

### Add More Locations

Edit `TRAINING_LOCATIONS` in `ccmews_ml_training.py`:

```python
TRAINING_LOCATIONS = [
    ("Battor", 6.0833, 0.4200),
    ("Adidome", 6.1200, 0.3150),
    # Add more locations...
]
```

### Adjust Thresholds

Edit `THRESHOLDS` dictionary:

```python
THRESHOLDS = {
    "flood": {
        "precip_daily_high": 50,      # Adjust for local conditions
        "precip_daily_moderate": 30,
        ...
    },
    ...
}
```

### Use Different Models

Modify the training methods in `HazardMLModels` class to use different algorithms:

```python
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
# or
model = SVC(probability=True, kernel='rbf')
```

## Troubleshooting

### "No data fetched" Error
- Check internet connection
- Open-Meteo API may be temporarily unavailable
- The script will use synthetic data as fallback

### Low Performance Scores
- Increase `years_of_history` for more training data
- Adjust thresholds for your specific region
- Check for class imbalance

### Memory Issues
- Reduce number of locations
- Reduce years of history
- Use `--quick` flag for testing

## Next Steps

1. **Train with real data**: Run locally with internet access to Open-Meteo
2. **Validate predictions**: Compare against historical disaster records
3. **Fine-tune thresholds**: Calibrate for North Tongu specific conditions
4. **Set up retraining schedule**: Periodically retrain with new data
