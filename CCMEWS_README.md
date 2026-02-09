# CCMEWS - Climate Change Monitoring & Early Warning System

## 🌍 Overview

CCMEWS is a real-time climate monitoring and AI-powered hazard prediction system focused on North Tongu District, Ghana. It provides **7-day advance warnings** for flood, heat, and drought risks.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CCMEWS Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Open-Meteo  │───▶│ Data Service │───▶│   SQLite DB  │      │
│  │     API      │    │  (5h update) │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                              │                   │              │
│                              ▼                   ▼              │
│                      ┌──────────────┐    ┌──────────────┐      │
│                      │  AI Engine   │───▶│  Predictions │      │
│                      │  (ML Models) │    │   (7 days)   │      │
│                      └──────────────┘    └──────────────┘      │
│                              │                   │              │
│                              ▼                   ▼              │
│                      ┌──────────────────────────────────┐      │
│                      │     Streamlit Dashboard          │      │
│                      │   (Maps, Charts, Alerts)         │      │
│                      └──────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Files

| File | Description |
|------|-------------|
| `ccmews_north_tongu.py` | Main Streamlit dashboard |
| `ccmews_data_service.py` | Climate data ingestion from Open-Meteo API |
| `ccmews_ai_engine.py` | AI hazard prediction models |
| `ccmews_scheduler_service.py` | Automated 5-hour update scheduler |
| `north_tongu.geojson` | District boundary file |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit pandas numpy plotly scipy folium streamlit-folium requests
```

### 2. Place Files Together

Put all files in the same directory:
```
ccmews/
├── ccmews_north_tongu.py
├── ccmews_data_service.py
├── ccmews_ai_engine.py
├── ccmews_scheduler_service.py
└── north_tongu.geojson
```

### 3. Fetch Initial Data

```bash
cd ccmews
python ccmews_scheduler_service.py --run-once
```

This will:
- 📡 Fetch climate data from Open-Meteo API (free, no API key)
- 🧠 Generate AI predictions for 18 monitoring locations
- 💾 Store data in SQLite database

### 4. Run Dashboard

```bash
streamlit run ccmews_north_tongu.py
```

## ⏰ Automated Updates (Every 5 Hours)

### Option A: Run as Daemon

```bash
python ccmews_scheduler_service.py --daemon
```

### Option B: Use Cron (Linux/Mac)

```bash
# Add to crontab (crontab -e)
0 */5 * * * cd /path/to/ccmews && python ccmews_scheduler_service.py --run-once
```

### Option C: Windows Task Scheduler

Create a scheduled task to run:
```
python C:\path\to\ccmews\ccmews_scheduler_service.py --run-once
```

## 🧠 AI Prediction Models

### Hazard Types

| Hazard | Key Indicators | Threshold |
|--------|---------------|-----------|
| 🌊 **Flood** | Precipitation, soil moisture, cumulative rainfall | >50mm/24h |
| 🔥 **Heat** | Temperature, humidity, heat index | >37°C |
| 🏜️ **Drought** | Dry days, soil moisture deficit, precip deficit | >10 dry days |

### Prediction Horizon

- **Day 1-2**: High confidence (85%+)
- **Day 3-4**: Good confidence (75%+)
- **Day 5-7**: Moderate confidence (60%+)

### Risk Levels

| Level | Composite Score | Action |
|-------|----------------|--------|
| 🟢 Low | < 0.25 | Normal operations |
| 🟡 Moderate | 0.25 - 0.45 | Monitor conditions |
| 🟠 High | 0.45 - 0.65 | Prepare response |
| 🔴 Critical | > 0.65 | Immediate action |

## 📊 Data Sources

### Open-Meteo API (Primary)
- **URL**: https://api.open-meteo.com/v1/forecast
- **Cost**: Free, no API key required
- **Data**: Temperature, precipitation, humidity, soil moisture, forecasts
- **Resolution**: Hourly observations, 7-day forecasts

### Data Coverage

18 monitoring points across North Tongu District:
- 1 District capital (Battor)
- 5 Major towns
- 9 Villages
- 3 Automated monitoring stations

## 🗺️ Dashboard Pages

1. **🗺️ Hazard Maps** - Interpolated risk surfaces
2. **🔮 AI Predictions** - 7-day forecasts with confidence scores
3. **🌡️ Climate Maps** - Current weather conditions
4. **📊 Dashboard** - Overview and summary statistics
5. **📈 Time Series** - Historical trends
6. **⚠️ Alerts** - Active warnings
7. **🏘️ Communities** - Location-specific data

## 🛠️ Customization

### Add New Monitoring Points

Edit `MONITORING_GRID` in `ccmews_data_service.py`:

```python
MONITORING_GRID = [
    ("Location Name", latitude, longitude),
    # Add more points...
]
```

### Adjust Thresholds

Edit `THRESHOLDS` in `ccmews_ai_engine.py`:

```python
THRESHOLDS = {
    "flood": {
        "precip_24h_danger": 50,  # mm
        # ...
    }
}
```

### Change Update Interval

Edit `UPDATE_INTERVAL_HOURS` in `ccmews_scheduler_service.py`:

```python
UPDATE_INTERVAL_HOURS = 5  # Change to desired hours
```

## 📱 Notifications (Future)

The system includes placeholder for notifications. To enable:

1. SMS via Twilio/Africa's Talking
2. Email via SMTP
3. WhatsApp Business API
4. Push notifications

Edit `send_notifications()` in `ccmews_scheduler_service.py`.

## 🔧 Troubleshooting

### No predictions showing?
```bash
python ccmews_scheduler_service.py --run-once
```

### Check system status:
```bash
python ccmews_scheduler_service.py --status
```

### View logs:
```bash
cat ccmews_scheduler.log
```

### Reset database:
```bash
rm ccmews_climate.db
python ccmews_scheduler_service.py --run-once
```

## 📄 License

MIT License - Free for educational and research use.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional ML models (LSTM, XGBoost)
- More data sources (NASA POWER, ERA5)
- Mobile app integration
- SMS/WhatsApp alerts
- Multi-district support
