# 🌍 CCMEWS - Climate Change Monitoring & Early Warning System

**AI-Powered Dashboard Prototype for Ghana**

A comprehensive Streamlit-based dashboard for climate monitoring, land cover change detection, hazard risk assessment, and early warning alerts.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Features

### 1. **Overview Dashboard**
- National risk overview map with interactive district markers
- Active alerts panel with severity levels
- Key climate metrics and trends
- District risk distribution charts

### 2. **Climate Monitoring**
- Real-time climate variable tracking (temperature, rainfall, humidity, soil moisture, NDVI)
- Time-series analysis with rolling averages
- Anomaly detection and visualization
- Customizable date ranges and district selection

### 3. **Land Cover Change Detection**
- Multi-year land cover composition analysis
- Change detection between comparison years
- Deforestation and urbanization tracking
- Environmental degradation insights

### 4. **Hazard Risk Assessment**
- Composite risk indices (flood, heat, drought)
- AI-powered risk scoring and ranking
- 7-day risk forecasting with confidence intervals
- Risk component breakdown by district

### 5. **Alert Management**
- Active alert monitoring with severity levels
- Alert configuration interface
- Alert history and analytics
- Export functionality

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone or download the project files**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Application

Start the Streamlit server:

```bash
s and
```

The application will open in your default browser at `http://localhost:8501`

### Command-line options:

```bash
# Run on a specific port
streamlit run app.py --server.port 8080

# Run with specific theme
streamlit run app.py --theme.base dark

# Allow external access
streamlit run app.py --server.address 0.0.0.0
```

---

## 🗂️ Project Structure

```
ccmews/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

---

## 🔧 Configuration

### Data Sources (Simulated in Prototype)
The prototype uses simulated data. For production, you would connect to:
- **Satellite imagery**: Sentinel, Landsat, MODIS (via APIs)
- **Climate data**: ERA5 reanalysis, GFS forecasts
- **Administrative boundaries**: Ghana Statistical Service
- **Community reports**: Custom APIs or DHIS2 integration

### Customization
- Modify district data in `get_ghana_districts()`
- Adjust alert thresholds in `generate_alerts()`
- Customize hazard indices in `generate_hazard_indices()`
- Update land cover categories in `generate_land_cover_data()`

---

## 🏗️ Architecture for Production

For a full production deployment, consider this architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                      │
│                    (Streamlit Dashboard)                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      APPLICATION LAYER                       │
│              (FastAPI / Flask REST APIs)                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      ANALYTICS LAYER                         │
│        (ML Models, Anomaly Detection, Risk Indices)          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                        DATA LAYER                            │
│           (PostGIS, GeoServer, Time-series DB)               │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      INGESTION LAYER                         │
│        (ETL Pipelines, Scheduled Jobs, Data APIs)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Sample Screenshots

The application includes:
- 🗺️ Interactive maps with Folium
- 📈 Time-series charts with Plotly
- 📊 Risk distribution visualizations
- 🚨 Alert management interface

---

## 🔐 Security Considerations

For production deployment:
- Implement proper authentication (e.g., `streamlit-authenticator`)
- Use HTTPS for all communications
- Secure API endpoints
- Implement role-based access control
- Encrypt sensitive data at rest and in transit

---

## 🤝 Contributing

This is a proof-of-concept developed by HISP Ghana. For contributions:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📞 Contact

**HISP Ghana**  
Technical Implementation Partner

For questions about this prototype, please contact the CCMEWS development team.

---

*Built with ❤️ for climate resilience in Ghana*
