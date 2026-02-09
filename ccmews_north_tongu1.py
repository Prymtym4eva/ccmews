"""
CCMEWS - Climate Change Monitoring & Early Warning System
NORTH TONGU DISTRICT FOCUS
Localized Dashboard with Community-Level Raster Maps
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import httpx
from scipy.interpolate import Rbf, griddata
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import json
import os

# Configuration
API_BASE = "http://localhost:8000"
DISTRICT_ID = 1  # North Tongu

# Load GeoJSON boundary
GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "north_tongu.geojson")

@st.cache_data
def load_district_boundary():
    """Load North Tongu district boundary from GeoJSON file"""
    try:
        with open(GEOJSON_PATH, 'r') as f:
            geojson_data = json.load(f)
        
        # Extract coordinates from MultiPolygon
        coords = geojson_data['features'][0]['geometry']['coordinates'][0][0]
        
        # Convert to list of dicts for consistency
        boundary = [{"lon": p[0], "lat": p[1]} for p in coords]
        
        # Calculate actual bounds
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        bounds = {
            "lon_min": min(lons),
            "lon_max": max(lons),
            "lat_min": min(lats),
            "lat_max": max(lats)
        }
        center = {
            "lon": (bounds["lon_min"] + bounds["lon_max"]) / 2,
            "lat": (bounds["lat_min"] + bounds["lat_max"]) / 2
        }
        
        return boundary, bounds, center, geojson_data
    except FileNotFoundError:
        st.warning(f"GeoJSON file not found at {GEOJSON_PATH}. Using default boundary.")
        return None, None, None, None

# Load boundary data
DISTRICT_BOUNDARY, DISTRICT_BOUNDS, DISTRICT_CENTER, GEOJSON_DATA = load_district_boundary()

st.set_page_config(
    page_title="CCMEWS - North Tongu District",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# NORTH TONGU DISTRICT DATA
# ============================================================================
NORTH_TONGU = {
    "name": "North Tongu",
    "id": 1,
    "capital": "Battor",
    "region": "Volta Region",
    "area_km2": 1843,
    "population": 89000,
    "center": DISTRICT_CENTER or {"lat": 6.1439, "lon": 0.2979},
    "bounds": DISTRICT_BOUNDS or {
        "lat_min": 5.954,
        "lat_max": 6.334,
        "lon_min": 0.091,
        "lon_max": 0.505
    }
}

# Communities/Towns in North Tongu District with monitoring data
# Comprehensive coverage across the entire district based on actual boundary
# All coordinates verified to be INSIDE the district boundary from GeoJSON
COMMUNITIES = {
    # === DISTRICT CAPITAL ===
    "Battor": {
        "lat": 6.0833, "lon": 0.4200, "type": "capital", "population": 8500,
        "temp": 29.8, "precip": 3.1, "humidity": 76,
        "flood": 0.55, "heat": 0.42, "drought": 0.35
    },
    
    # === MAJOR TOWNS (5) ===
    "Adidome": {
        "lat": 6.1200, "lon": 0.3150, "type": "town", "population": 6200,
        "temp": 30.3, "precip": 2.5, "humidity": 72,
        "flood": 0.45, "heat": 0.48, "drought": 0.42
    },
    "Aveyime": {
        "lat": 6.1600, "lon": 0.4550, "type": "town", "population": 5200,
        "temp": 29.5, "precip": 3.4, "humidity": 78,
        "flood": 0.58, "heat": 0.38, "drought": 0.32
    },
    "Mepe": {
        "lat": 6.0500, "lon": 0.3850, "type": "town", "population": 4800,
        "temp": 29.2, "precip": 3.8, "humidity": 80,
        "flood": 0.62, "heat": 0.35, "drought": 0.28
    },
    "Mafi Kumase": {
        "lat": 6.0900, "lon": 0.3500, "type": "town", "population": 3500,
        "temp": 30.1, "precip": 2.8, "humidity": 74,
        "flood": 0.48, "heat": 0.45, "drought": 0.40
    },
    "Sogakope": {
        "lat": 6.0100, "lon": 0.4200, "type": "town", "population": 3800,
        "temp": 29.6, "precip": 3.2, "humidity": 77,
        "flood": 0.52, "heat": 0.40, "drought": 0.35
    },
    
    # === NORTHERN SECTOR (lat > 6.24) ===
    "Mafi Asiekpe": {
        "lat": 6.2800, "lon": 0.2100, "type": "village", "population": 800,
        "temp": 29.8, "precip": 3.0, "humidity": 75,
        "flood": 0.50, "heat": 0.42, "drought": 0.38
    },
    "Mafi Anfoe": {
        "lat": 6.3100, "lon": 0.2800, "type": "village", "population": 750,
        "temp": 29.2, "precip": 3.5, "humidity": 78,
        "flood": 0.55, "heat": 0.38, "drought": 0.32
    },
    "Togorme": {
        "lat": 6.2650, "lon": 0.3500, "type": "village", "population": 850,
        "temp": 29.5, "precip": 3.2, "humidity": 76,
        "flood": 0.52, "heat": 0.40, "drought": 0.36
    },
    "Agave Afedome": {
        "lat": 6.2400, "lon": 0.4200, "type": "village", "population": 920,
        "temp": 28.9, "precip": 3.8, "humidity": 80,
        "flood": 0.60, "heat": 0.35, "drought": 0.28
    },
    
    # === NORTH-CENTRAL SECTOR (6.20 - 6.24) ===
    "Mafi Adonkia": {
        "lat": 6.2000, "lon": 0.1760, "type": "village", "population": 1200,
        "temp": 30.4, "precip": 2.2, "humidity": 69,
        "flood": 0.36, "heat": 0.52, "drought": 0.50
    },
    "Seva": {
        "lat": 6.2150, "lon": 0.2450, "type": "village", "population": 1100,
        "temp": 30.1, "precip": 2.5, "humidity": 71,
        "flood": 0.40, "heat": 0.48, "drought": 0.46
    },
    "Dorfor": {
        "lat": 6.2000, "lon": 0.3150, "type": "village", "population": 2100,
        "temp": 29.8, "precip": 2.8, "humidity": 73,
        "flood": 0.44, "heat": 0.44, "drought": 0.42
    },
    "Fievie": {
        "lat": 6.2000, "lon": 0.3850, "type": "village", "population": 1800,
        "temp": 29.4, "precip": 3.2, "humidity": 76,
        "flood": 0.50, "heat": 0.40, "drought": 0.36
    },
    "Agave": {
        "lat": 6.2000, "lon": 0.4550, "type": "village", "population": 900,
        "temp": 28.8, "precip": 3.9, "humidity": 81,
        "flood": 0.65, "heat": 0.33, "drought": 0.26
    },
    
    # === CENTRAL-WEST SECTOR (6.12 - 6.20) ===
    "Mafi Dadoboe": {
        "lat": 6.1600, "lon": 0.1410, "type": "village", "population": 950,
        "temp": 30.6, "precip": 2.0, "humidity": 67,
        "flood": 0.32, "heat": 0.54, "drought": 0.54
    },
    "Mafi Zongo": {
        "lat": 6.1200, "lon": 0.2100, "type": "village", "population": 1350,
        "temp": 30.3, "precip": 2.3, "humidity": 70,
        "flood": 0.38, "heat": 0.50, "drought": 0.48
    },
    "Torgome": {
        "lat": 6.1600, "lon": 0.2800, "type": "village", "population": 1500,
        "temp": 30.0, "precip": 2.7, "humidity": 73,
        "flood": 0.44, "heat": 0.46, "drought": 0.44
    },
    
    # === CENTRAL-EAST SECTOR (6.12 - 6.20) ===
    "Volo": {
        "lat": 6.1200, "lon": 0.3850, "type": "village", "population": 2500,
        "temp": 29.6, "precip": 3.3, "humidity": 77,
        "flood": 0.56, "heat": 0.40, "drought": 0.33
    },
    "Kpotame": {
        "lat": 6.1200, "lon": 0.4550, "type": "village", "population": 1100,
        "temp": 29.2, "precip": 3.6, "humidity": 79,
        "flood": 0.62, "heat": 0.36, "drought": 0.28
    },
    "Tornu": {
        "lat": 6.1600, "lon": 0.3500, "type": "village", "population": 1400,
        "temp": 29.7, "precip": 3.0, "humidity": 75,
        "flood": 0.50, "heat": 0.42, "drought": 0.38
    },
    
    # === SOUTH-CENTRAL SECTOR (6.05 - 6.12) ===
    "Sasekpe": {
        "lat": 6.0900, "lon": 0.2800, "type": "village", "population": 1250,
        "temp": 30.2, "precip": 2.5, "humidity": 72,
        "flood": 0.42, "heat": 0.48, "drought": 0.44
    },
    "Anfoe": {
        "lat": 6.0500, "lon": 0.4200, "type": "village", "population": 1600,
        "temp": 29.4, "precip": 3.5, "humidity": 78,
        "flood": 0.58, "heat": 0.38, "drought": 0.32
    },
    "Mafi Dove": {
        "lat": 6.0900, "lon": 0.4200, "type": "village", "population": 1400,
        "temp": 29.5, "precip": 3.4, "humidity": 77,
        "flood": 0.55, "heat": 0.40, "drought": 0.34
    },
    
    # === SOUTHERN SECTOR (lat < 6.05) ===
    "Tefle": {
        "lat": 6.0100, "lon": 0.3500, "type": "village", "population": 1300,
        "temp": 29.8, "precip": 3.0, "humidity": 75,
        "flood": 0.48, "heat": 0.43, "drought": 0.38
    },
    "Bakpa": {
        "lat": 5.9700, "lon": 0.3500, "type": "village", "population": 2000,
        "temp": 30.0, "precip": 2.8, "humidity": 74,
        "flood": 0.46, "heat": 0.45, "drought": 0.40
    },
    "Bakpa Avenya": {
        "lat": 5.9700, "lon": 0.3850, "type": "village", "population": 1150,
        "temp": 29.8, "precip": 3.0, "humidity": 75,
        "flood": 0.50, "heat": 0.43, "drought": 0.38
    },
    "Kpedzeglo": {
        "lat": 6.0100, "lon": 0.4550, "type": "village", "population": 980,
        "temp": 29.3, "precip": 3.5, "humidity": 78,
        "flood": 0.58, "heat": 0.38, "drought": 0.32
    },
    "Dabala": {
        "lat": 6.0500, "lon": 0.4550, "type": "village", "population": 1080,
        "temp": 29.2, "precip": 3.6, "humidity": 79,
        "flood": 0.60, "heat": 0.36, "drought": 0.30
    },
    
    # === AUTOMATED MONITORING STATIONS (for interpolation coverage at edges) ===
    "Station NW": {
        "lat": 6.2800, "lon": 0.1760, "type": "station", "population": 0,
        "temp": 30.5, "precip": 2.0, "humidity": 66,
        "flood": 0.30, "heat": 0.54, "drought": 0.56
    },
    "Station N": {
        "lat": 6.3100, "lon": 0.2450, "type": "station", "population": 0,
        "temp": 30.0, "precip": 2.4, "humidity": 70,
        "flood": 0.38, "heat": 0.48, "drought": 0.48
    },
    "Station NE": {
        "lat": 6.2400, "lon": 0.4550, "type": "station", "population": 0,
        "temp": 28.7, "precip": 4.0, "humidity": 82,
        "flood": 0.68, "heat": 0.32, "drought": 0.22
    },
    "Station W": {
        "lat": 6.1600, "lon": 0.1060, "type": "station", "population": 0,
        "temp": 30.8, "precip": 1.8, "humidity": 64,
        "flood": 0.28, "heat": 0.56, "drought": 0.58
    },
    "Station E": {
        "lat": 6.1600, "lon": 0.4900, "type": "station", "population": 0,
        "temp": 28.6, "precip": 4.2, "humidity": 83,
        "flood": 0.72, "heat": 0.30, "drought": 0.20
    },
    "Station S": {
        "lat": 6.0100, "lon": 0.4900, "type": "station", "population": 0,
        "temp": 29.0, "precip": 3.8, "humidity": 80,
        "flood": 0.64, "heat": 0.34, "drought": 0.26
    },
    "Station Central": {
        "lat": 6.1200, "lon": 0.3150, "type": "station", "population": 0,
        "temp": 29.9, "precip": 2.9, "humidity": 74,
        "flood": 0.46, "heat": 0.44, "drought": 0.40
    },
}

# Volta River points (for flood risk visualization)
# Eastern edge of district along the Volta River
VOLTA_RIVER_POINTS = [
    {"lat": 6.24, "lon": 0.48},
    {"lat": 6.20, "lon": 0.47},
    {"lat": 6.16, "lon": 0.48},
    {"lat": 6.12, "lon": 0.46},
    {"lat": 6.08, "lon": 0.44},
    {"lat": 6.04, "lon": 0.46},
    {"lat": 6.00, "lon": 0.48},
    {"lat": 5.98, "lon": 0.46},
]

# Note: DISTRICT_BOUNDARY is loaded from GeoJSON file at the top of the file

# ============================================================================
# API FUNCTIONS
# ============================================================================
@st.cache_data(ttl=60)
def fetch_api(endpoint: str):
    """Fetch data from backend API"""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{API_BASE}{endpoint}")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        pass
    return None

def check_api_health():
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{API_BASE}/")
            return resp.status_code == 200
    except:
        return False

@st.cache_data(ttl=120)
def get_north_tongu_climate(days: int = 30):
    """Get climate data specifically for North Tongu"""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    data = fetch_api(
        f"/api/v1/climate/districts/{DISTRICT_ID}/observations"
        f"?start_date={start_date}&end_date={end_date}"
    )
    
    if data:
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("observations") or data.get("data") or []
    return []

# ============================================================================
# BUILD DATAFRAMES
# ============================================================================
def build_communities_df():
    """Build communities dataframe"""
    data = []
    for name, info in COMMUNITIES.items():
        composite = (info["flood"] + info["heat"] + info["drought"]) / 3
        level = "critical" if composite >= 0.6 else "high" if composite >= 0.45 else "moderate" if composite >= 0.3 else "low"
        dominant = "flood" if info["flood"] >= max(info["heat"], info["drought"]) else \
                   "heat" if info["heat"] >= info["drought"] else "drought"
        
        data.append({
            "name": name,
            "lat": info["lat"],
            "lon": info["lon"],
            "type": info["type"],
            "population": info["population"],
            "temp": info["temp"],
            "precip": info["precip"],
            "humidity": info["humidity"],
            "flood": info["flood"],
            "heat": info["heat"],
            "drought": info["drought"],
            "composite": composite,
            "level": level,
            "dominant": dominant
        })
    return pd.DataFrame(data)

# ============================================================================
# INTERPOLATION FUNCTIONS
# ============================================================================
def create_interpolated_grid(df, value_col, resolution=60):
    """Create interpolated grid for North Tongu"""
    bounds = NORTH_TONGU["bounds"]
    
    lats = df['lat'].values
    lons = df['lon'].values
    values = df[value_col].values
    
    grid_lon = np.linspace(bounds['lon_min'], bounds['lon_max'], resolution)
    grid_lat = np.linspace(bounds['lat_min'], bounds['lat_max'], resolution)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    try:
        rbf = Rbf(lons, lats, values, function='multiquadric', smooth=0.3)
        grid_values = rbf(grid_lon_mesh, grid_lat_mesh)
    except:
        points = np.column_stack((lons, lats))
        grid_values = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='cubic')
        if np.any(np.isnan(grid_values)):
            grid_values_nearest = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='nearest')
            grid_values = np.where(np.isnan(grid_values), grid_values_nearest, grid_values)
    
    # Clip risk values
    if value_col in ['flood', 'heat', 'drought', 'composite']:
        grid_values = np.clip(grid_values, 0, 1)
    
    return grid_lon, grid_lat, grid_values

def create_raster_map(df, value_col, title, colorscale, show_river=False, resolution=60, marker_df=None):
    """Create raster heatmap for North Tongu
    
    Args:
        df: DataFrame with all points for interpolation
        value_col: Column to interpolate
        title: Map title
        colorscale: Plotly colorscale
        show_river: Whether to show Volta River
        resolution: Grid resolution
        marker_df: Optional separate DataFrame for markers (if None, uses df)
    """
    bounds = NORTH_TONGU["bounds"]
    grid_lon, grid_lat, grid_values = create_interpolated_grid(df, value_col, resolution)
    
    # Use marker_df if provided, otherwise use df
    display_df = marker_df if marker_df is not None else df
    
    fig = go.Figure()
    
    # Heatmap layer
    fig.add_trace(go.Heatmap(
        x=grid_lon,
        y=grid_lat,
        z=grid_values,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=dict(text=title, side='right')),
        hovertemplate='Lon: %{x:.3f}<br>Lat: %{y:.3f}<br>Value: %{z:.2f}<extra></extra>',
        opacity=0.85
    ))
    
    # Contour lines
    fig.add_trace(go.Contour(
        x=grid_lon,
        y=grid_lat,
        z=grid_values,
        colorscale=colorscale,
        showscale=False,
        contours=dict(showlines=True, coloring='none', showlabels=True,
                     labelfont=dict(size=9, color='black')),
        line=dict(width=1, color='rgba(0,0,0,0.4)')
    ))
    
    # Volta River (if showing flood risk)
    if show_river:
        river_lats = [p["lat"] for p in VOLTA_RIVER_POINTS]
        river_lons = [p["lon"] for p in VOLTA_RIVER_POINTS]
        fig.add_trace(go.Scatter(
            x=river_lons, y=river_lats,
            mode='lines',
            line=dict(color='blue', width=4),
            name='Volta River',
            hoverinfo='name'
        ))
    
    # District boundary outline (use simplified version for performance)
    if DISTRICT_BOUNDARY:
        # Simplify boundary for plotting (take every Nth point)
        step = max(1, len(DISTRICT_BOUNDARY) // 200)  # Keep ~200 points max
        simplified_boundary = DISTRICT_BOUNDARY[::step]
        # Make sure we close the polygon
        if simplified_boundary[0] != simplified_boundary[-1]:
            simplified_boundary.append(simplified_boundary[0])
        
        boundary_lats = [p["lat"] for p in simplified_boundary]
        boundary_lons = [p["lon"] for p in simplified_boundary]
        fig.add_trace(go.Scatter(
            x=boundary_lons, y=boundary_lats,
            mode='lines',
            line=dict(color='black', width=2.5),
            name='District Boundary',
            hoverinfo='name',
            fill='none'
        ))
    
    # Community markers (grouped by type for cleaner legend)
    for _, row in display_df.iterrows():
        if row['type'] == 'capital':
            marker_size, marker_symbol, marker_color = 18, 'star', 'gold'
        elif row['type'] == 'town':
            marker_size, marker_symbol, marker_color = 14, 'circle', 'white'
        elif row['type'] == 'station':
            marker_size, marker_symbol, marker_color = 8, 'diamond', 'cyan'
        else:  # village
            marker_size, marker_symbol, marker_color = 10, 'circle', 'white'
        
        # Only show text labels for capital, towns, and villages (not stations)
        show_text = row['type'] != 'station'
        text_label = row['name'] if show_text else ''
        
        fig.add_trace(go.Scatter(
            x=[row['lon']], y=[row['lat']],
            mode='markers+text' if show_text else 'markers',
            marker=dict(size=marker_size, color=marker_color, symbol=marker_symbol,
                       line=dict(width=2, color='black')),
            text=[text_label] if show_text else None,
            textposition='top center',
            textfont=dict(size=8, color='black'),
            hovertemplate=f"<b>{row['name']}</b><br>Type: {row['type'].title()}<br>{value_col}: {row[value_col]:.2f}<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sub>North Tongu District</sub>", x=0.5),
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=550,
        xaxis=dict(range=[bounds['lon_min'], bounds['lon_max']], dtick=0.1),
        yaxis=dict(range=[bounds['lat_min'], bounds['lat_max']], dtick=0.1, scaleanchor="x"),
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

def create_folium_map(df, value_col, title, gradient=None, marker_df=None):
    """Create Folium interactive map for North Tongu
    
    Args:
        df: DataFrame with all points for heatmap
        value_col: Column to display
        title: Map title
        gradient: Color gradient for heatmap
        marker_df: Optional separate DataFrame for markers (if None, uses df)
    """
    center = NORTH_TONGU["center"]
    bounds = NORTH_TONGU["bounds"]
    
    # Use marker_df if provided, otherwise use df
    display_df = marker_df if marker_df is not None else df
    
    m = folium.Map(
        location=[center['lat'], center['lon']],
        zoom_start=11,
        tiles='cartodbpositron'
    )
    
    # Default gradient
    if gradient is None:
        gradient = {0.0: 'green', 0.3: 'yellow', 0.5: 'orange', 0.7: 'red', 1.0: 'darkred'}
    
    # Heatmap data
    heat_data = []
    for _, row in df.iterrows():
        heat_data.append([row['lat'], row['lon'], row[value_col]])
        # Add surrounding points for smoother gradient
        for d in [0.02, 0.04]:
            for dlat, dlon in [(d,0), (-d,0), (0,d), (0,-d)]:
                heat_data.append([row['lat']+dlat, row['lon']+dlon, row[value_col]*0.8])
    
    HeatMap(heat_data, min_opacity=0.4, radius=25, blur=20, gradient=gradient).add_to(m)
    
    # Volta River
    if value_col == 'flood':
        river_points = [[p['lat'], p['lon']] for p in VOLTA_RIVER_POINTS]
        folium.PolyLine(river_points, color='blue', weight=5, opacity=0.8, 
                       popup='Volta River').add_to(m)
    
    # Community markers
    for _, row in display_df.iterrows():
        value = row[value_col]
        
        # Color based on risk value
        risk_color = 'green' if value < 0.3 else 'orange' if value < 0.5 else 'red' if value < 0.7 else 'darkred'
        
        if row['type'] == 'station':
            # Monitoring stations - use CircleMarker for smaller display
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color='blue',
                fill=True,
                fillColor='cyan',
                fillOpacity=0.7,
                popup=f"<b>{row['name']}</b><br>Monitoring Station<br>{value_col}: {value:.2f}",
                tooltip=row['name']
            ).add_to(m)
        else:
            # Communities - use regular markers
            icon = 'star' if row['type'] == 'capital' else 'home' if row['type'] == 'town' else 'circle'
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"<b>{row['name']}</b><br>{row['type'].title()}<br>{value_col}: {value:.2f}<br>Pop: {row['population']:,}",
                tooltip=row['name'],
                icon=folium.Icon(color=risk_color, icon=icon, prefix='fa')
            ).add_to(m)
    
    # District boundary from GeoJSON
    if GEOJSON_DATA:
        folium.GeoJson(
            GEOJSON_DATA,
            name='District Boundary',
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'black',
                'weight': 3,
                'fillOpacity': 0
            },
            tooltip='North Tongu District'
        ).add_to(m)
    elif DISTRICT_BOUNDARY:
        # Fallback to polygon if GeoJSON not available
        boundary_coords = [[p['lat'], p['lon']] for p in DISTRICT_BOUNDARY]
        folium.Polygon(
            locations=boundary_coords,
            color='black',
            weight=3,
            fill=False,
            popup='North Tongu District Boundary',
            tooltip='North Tongu District'
        ).add_to(m)
    
    return m

def create_3d_surface(df, value_col, title, colorscale):
    """Create 3D surface for North Tongu"""
    grid_lon, grid_lat, grid_values = create_interpolated_grid(df, value_col, 40)
    
    fig = go.Figure()
    
    # Add surface
    fig.add_trace(go.Surface(
        x=grid_lon, y=grid_lat, z=grid_values,
        colorscale=colorscale,
        colorbar=dict(title=value_col.title()),
        name='Risk Surface'
    ))
    
    # Add district boundary at base (z=0)
    if DISTRICT_BOUNDARY:
        # Simplify boundary for 3D plotting
        step = max(1, len(DISTRICT_BOUNDARY) // 150)
        simplified_boundary = DISTRICT_BOUNDARY[::step]
        if simplified_boundary[0] != simplified_boundary[-1]:
            simplified_boundary.append(simplified_boundary[0])
        
        boundary_lons = [p["lon"] for p in simplified_boundary]
        boundary_lats = [p["lat"] for p in simplified_boundary]
        boundary_z = [0] * len(simplified_boundary)
        
        fig.add_trace(go.Scatter3d(
            x=boundary_lons, y=boundary_lats, z=boundary_z,
            mode='lines',
            line=dict(color='black', width=5),
            name='District Boundary'
        ))
    
    fig.update_layout(
        title=f"3D {title} - North Tongu",
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude', 
            zaxis_title=value_col.title(),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=550
    )
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # District header
    st.markdown("""
    <div style="text-align: center; padding: 10px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px; margin-bottom: 15px;">
        <h2 style="color: white; margin: 0;">🌍 CCMEWS</h2>
        <p style="color: #ddd; margin: 5px 0; font-size: 14px;">North Tongu District</p>
    </div>
    """, unsafe_allow_html=True)
    
    # District info
    st.info(f"📍 **Capital:** {NORTH_TONGU['capital']}\n\n"
            f"🗺️ **Region:** {NORTH_TONGU['region']}\n\n"
            f"📐 **Area:** {NORTH_TONGU['area_km2']:,} km²\n\n"
            f"👥 **Population:** ~{NORTH_TONGU['population']:,}")
    
    st.divider()
    
    # API Status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.warning("⚠️ API Offline - Using cached data")
    
    st.divider()
    
    # Navigation
    page = st.radio("Navigation", [
        "🗺️ Hazard Maps",
        "🌡️ Climate Maps",
        "📊 Dashboard",
        "📈 Time Series",
        "⚠️ Alerts",
        "🏘️ Communities"
    ])
    
    st.divider()
    
    # Map settings
    st.subheader("⚙️ Map Settings")
    resolution = st.slider("Resolution", 30, 100, 60, help="Higher = more detail")
    show_river = st.checkbox("Show Volta River", value=True)
    show_stations = st.checkbox("Show Monitoring Stations", value=True, help="Toggle visibility of automated monitoring stations")
    
    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Build data
communities_df_all = build_communities_df()  # All data points (for interpolation)
climate_data = get_north_tongu_climate(30)

# Filter for display based on toggle (but keep all for interpolation)
if show_stations:
    communities_df = communities_df_all
else:
    communities_df = communities_df_all[communities_df_all['type'] != 'station'].copy()

# Separate dataframes for stats
communities_only = communities_df_all[communities_df_all['type'] != 'station']
stations_only = communities_df_all[communities_df_all['type'] == 'station']

# ============================================================================
# HAZARD MAPS PAGE
# ============================================================================
if page == "🗺️ Hazard Maps":
    st.title("🗺️ North Tongu Hazard Risk Maps")
    st.caption("Interpolated hazard surfaces across the district")
    
    # Current risk summary
    col1, col2, col3, col4 = st.columns(4)
    avg_flood = communities_df['flood'].mean()
    avg_heat = communities_df['heat'].mean()
    avg_drought = communities_df['drought'].mean()
    avg_composite = communities_df['composite'].mean()
    
    with col1:
        st.metric("🌊 Flood Risk", f"{avg_flood:.2f}", 
                 delta="High" if avg_flood > 0.5 else "Moderate")
    with col2:
        st.metric("🔥 Heat Risk", f"{avg_heat:.2f}")
    with col3:
        st.metric("🏜️ Drought Risk", f"{avg_drought:.2f}")
    with col4:
        st.metric("📊 Composite", f"{avg_composite:.2f}")
    
    st.divider()
    
    # Map controls
    col1, col2 = st.columns(2)
    with col1:
        hazard_type = st.selectbox(
            "Hazard Layer",
            ["composite", "flood", "heat", "drought"],
            format_func=lambda x: {"composite": "📊 Composite Risk", "flood": "🌊 Flood Risk",
                                  "heat": "🔥 Heat Risk", "drought": "🏜️ Drought Risk"}[x]
        )
    with col2:
        map_style = st.selectbox("Map Style", ["Raster Heatmap", "Interactive Map", "3D Surface"])
    
    # Color scales
    colorscales = {
        "composite": "RdYlGn_r",
        "flood": [[0, '#fff7fb'], [0.25, '#d0d1e6'], [0.5, '#74a9cf'], [0.75, '#0570b0'], [1, '#023858']],
        "heat": "YlOrRd",
        "drought": "YlOrBr"
    }
    
    gradients = {
        "composite": {0: 'green', 0.3: 'yellow', 0.5: 'orange', 0.7: 'red', 1: 'darkred'},
        "flood": {0: '#f7fbff', 0.3: '#c6dbef', 0.5: '#6baed6', 0.7: '#2171b5', 1: '#084594'},
        "heat": {0: '#ffffb2', 0.3: '#fecc5c', 0.5: '#fd8d3c', 0.7: '#f03b20', 1: '#bd0026'},
        "drought": {0: '#ffffd4', 0.3: '#fed98e', 0.5: '#fe9929', 0.7: '#d95f0e', 1: '#993404'}
    }
    
    st.divider()
    
    if map_style == "Raster Heatmap":
        fig = create_raster_map(
            communities_df_all, hazard_type,
            f"{hazard_type.title()} Risk Index",
            colorscales[hazard_type],
            show_river=(show_river and hazard_type == 'flood'),
            resolution=resolution,
            marker_df=communities_df
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif map_style == "Interactive Map":
        folium_map = create_folium_map(
            communities_df_all, hazard_type,
            f"{hazard_type.title()} Risk",
            gradients[hazard_type],
            marker_df=communities_df
        )
        st_folium(folium_map, width=None, height=550)
        
    else:  # 3D Surface
        fig = create_3d_surface(communities_df_all, hazard_type, 
                               f"{hazard_type.title()} Risk", colorscales[hazard_type])
        st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    st.markdown("""
    **Risk Levels:** 🟢 Low (<0.3) | 🟡 Moderate (0.3-0.5) | 🟠 High (0.5-0.7) | 🔴 Critical (>0.7)
    """)
    
    st.divider()
    
    # All hazards comparison
    st.subheader("📊 All Hazard Layers")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_raster_map(communities_df_all, "flood", "Flood Risk", colorscales["flood"], 
                               show_river=show_river, resolution=40, marker_df=communities_df)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = create_raster_map(communities_df_all, "drought", "Drought Risk", colorscales["drought"],
                               resolution=40, marker_df=communities_df)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_raster_map(communities_df_all, "heat", "Heat Risk", colorscales["heat"],
                               resolution=40, marker_df=communities_df)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = create_raster_map(communities_df_all, "composite", "Composite Risk", colorscales["composite"],
                               resolution=40, marker_df=communities_df)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CLIMATE MAPS PAGE
# ============================================================================
elif page == "🌡️ Climate Maps":
    st.title("🌡️ North Tongu Climate Maps")
    st.caption("Current climate conditions across the district")
    
    # Climate summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡️ Avg Temperature", f"{communities_df['temp'].mean():.1f}°C",
                 delta=f"Max: {communities_df['temp'].max():.1f}°C")
    with col2:
        st.metric("🌧️ Avg Precipitation", f"{communities_df['precip'].mean():.1f} mm/day",
                 delta=f"Total: {communities_df['precip'].sum():.1f} mm")
    with col3:
        st.metric("💧 Avg Humidity", f"{communities_df['humidity'].mean():.0f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        climate_var = st.selectbox(
            "Climate Variable",
            ["temp", "precip", "humidity"],
            format_func=lambda x: {"temp": "🌡️ Temperature", "precip": "🌧️ Precipitation",
                                  "humidity": "💧 Humidity"}[x]
        )
    with col2:
        map_style = st.selectbox("Style", ["Raster", "Interactive", "3D"], key="climate_style")
    
    colorscales_climate = {
        "temp": "RdBu_r",
        "precip": "Blues",
        "humidity": "Greens"
    }
    
    gradients_climate = {
        "temp": {0: '#313695', 0.25: '#74add1', 0.5: '#ffffbf', 0.75: '#f46d43', 1: '#a50026'},
        "precip": {0: '#f7fbff', 0.25: '#c6dbef', 0.5: '#6baed6', 0.75: '#2171b5', 1: '#084594'},
        "humidity": {0: '#f7fcf5', 0.25: '#c7e9c0', 0.5: '#74c476', 0.75: '#238b45', 1: '#00441b'}
    }
    
    titles = {"temp": "Temperature (°C)", "precip": "Precipitation (mm/day)", "humidity": "Humidity (%)"}
    
    st.divider()
    
    if map_style == "Raster":
        fig = create_raster_map(communities_df_all, climate_var, titles[climate_var],
                               colorscales_climate[climate_var], resolution=resolution,
                               marker_df=communities_df)
        st.plotly_chart(fig, use_container_width=True)
    elif map_style == "Interactive":
        folium_map = create_folium_map(communities_df_all, climate_var, titles[climate_var],
                                       gradients_climate[climate_var], marker_df=communities_df)
        st_folium(folium_map, width=None, height=550)
    else:
        fig = create_3d_surface(communities_df_all, climate_var, titles[climate_var],
                               colorscales_climate[climate_var])
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # All climate variables
    st.subheader("📊 Climate Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = create_raster_map(communities_df_all, "temp", "Temperature (°C)", "RdBu_r", 
                               resolution=35, marker_df=communities_df)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_raster_map(communities_df_all, "precip", "Precipitation (mm)", "Blues", 
                               resolution=35, marker_df=communities_df)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_raster_map(communities_df_all, "humidity", "Humidity (%)", "Greens", 
                               resolution=35, marker_df=communities_df)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
elif page == "📊 Dashboard":
    st.title("📊 North Tongu District Dashboard")
    
    # Alert check (only for actual communities, not stations)
    high_risk_communities = communities_only[communities_only['composite'] >= 0.5]
    if len(high_risk_communities) > 0:
        st.warning(f"⚠️ {len(high_risk_communities)} communities at HIGH RISK")
    
    # Key metrics (use communities_only for population stats)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🏘️ Communities", len(communities_only))
    with col2:
        st.metric("🔴 High Risk", len(high_risk_communities))
    with col3:
        st.metric("🌡️ Avg Temp", f"{communities_only['temp'].mean():.1f}°C")
    with col4:
        st.metric("🌊 Flood Risk", f"{communities_only['flood'].mean():.2f}")
    with col5:
        st.metric("👥 Population", f"{communities_only['population'].sum():,}")
    
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Composite Risk Map")
        fig = create_raster_map(communities_df_all, "composite", "Composite Risk", "RdYlGn_r",
                               show_river=show_river, resolution=50, marker_df=communities_df)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Risk Summary")
        
        # Risk level breakdown (communities only, not stations)
        for level in ['critical', 'high', 'moderate', 'low']:
            count = len(communities_only[communities_only['level'] == level])
            emoji = {'critical': '🔴', 'high': '🟠', 'moderate': '🟡', 'low': '🟢'}[level]
            st.write(f"{emoji} **{level.title()}:** {count} communities")
        
        st.divider()
        
        st.subheader("🏘️ High Risk Areas")
        for _, row in high_risk_communities.sort_values('composite', ascending=False).iterrows():
            st.markdown(f"**{row['name']}** - {row['composite']:.2f}")
            st.caption(f"   Dominant: {row['dominant'].title()}")

# ============================================================================
# TIME SERIES PAGE
# ============================================================================
elif page == "📈 Time Series":
    st.title("📈 North Tongu Climate Time Series")
    
    if climate_data and len(climate_data) > 0:
        df = pd.DataFrame(climate_data)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'temperature_mean' in df.columns:
                st.metric("Avg Temperature", f"{df['temperature_mean'].mean():.1f}°C")
        with col2:
            if 'precipitation' in df.columns:
                st.metric("Total Precipitation", f"{df['precipitation'].sum():.1f}mm")
        with col3:
            st.metric("Data Points", len(df))
        
        st.divider()
        
        # Temperature
        if 'temperature_mean' in df.columns:
            st.subheader("🌡️ Temperature Trend")
            fig = go.Figure()
            if 'temperature_max' in df.columns:
                fig.add_trace(go.Scatter(x=df['observation_date'], y=df['temperature_max'],
                                        name='Max', line=dict(color='red', dash='dot')))
            fig.add_trace(go.Scatter(x=df['observation_date'], y=df['temperature_mean'],
                                    name='Mean', line=dict(color='orange', width=3)))
            if 'temperature_min' in df.columns:
                fig.add_trace(go.Scatter(x=df['observation_date'], y=df['temperature_min'],
                                        name='Min', line=dict(color='blue', dash='dot')))
            fig.update_layout(height=350, yaxis_title="°C", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        
        # Precipitation
        if 'precipitation' in df.columns:
            st.subheader("🌧️ Precipitation")
            fig = px.bar(df, x='observation_date', y='precipitation')
            fig.update_traces(marker_color='steelblue')
            fig.update_layout(height=300, yaxis_title="mm", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No climate time series data available. Check API connection.")
        st.info("Using community-level static data for maps.")

# ============================================================================
# ALERTS PAGE
# ============================================================================
elif page == "⚠️ Alerts":
    st.title("⚠️ North Tongu Alert Management")
    
    # Only show alerts for actual communities (not stations)
    high_risk = communities_only[communities_only['composite'] >= 0.5].sort_values('composite', ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Alert Statistics")
        st.metric("Total Communities", len(communities_only))
        st.metric("🔴 High/Critical Risk", len(high_risk))
        st.metric("Affected Population", f"{high_risk['population'].sum():,}")
        
        # Pie chart
        level_counts = communities_only['level'].value_counts()
        fig = px.pie(values=level_counts.values, names=level_counts.index,
                    color=level_counts.index,
                    color_discrete_map={'critical': '#dc3545', 'high': '#fd7e14',
                                       'moderate': '#ffc107', 'low': '#28a745'})
        fig.update_layout(height=250, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚨 Active Risk Alerts")
        
        if len(high_risk) > 0:
            for _, row in high_risk.iterrows():
                color = "#dc3545" if row['level'] == 'critical' else "#fd7e14"
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 15px; 
                           border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0;">⚠️ {row['dominant'].title()} Risk - {row['name']}</h4>
                    <p style="margin: 5px 0;">Risk Index: {row['composite']:.2f} | Level: {row['level'].title()}</p>
                    <small>🌊 Flood: {row['flood']:.2f} | 🔥 Heat: {row['heat']:.2f} | 
                           🏜️ Drought: {row['drought']:.2f}</small><br>
                    <small>👥 Population at risk: {row['population']:,}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No high-risk alerts")

# ============================================================================
# COMMUNITIES PAGE
# ============================================================================
elif page == "🏘️ Communities":
    st.title("🏘️ North Tongu Communities")
    
    # Summary (communities only, not stations)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Communities", len(communities_only))
    with col2:
        st.metric("Total Population", f"{communities_only['population'].sum():,}")
    with col3:
        capitals = len(communities_only[communities_only['type'] == 'capital'])
        towns = len(communities_only[communities_only['type'] == 'town'])
        st.metric("Towns", f"{capitals + towns}")
    
    st.divider()
    
    # Interactive selection (only actual communities)
    selected = st.selectbox("Select Community", communities_only['name'].tolist())
    
    if selected:
        row = communities_only[communities_only['name'] == selected].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📍 {row['name']}")
            st.write(f"**Type:** {row['type'].title()}")
            st.write(f"**Population:** {row['population']:,}")
            st.write(f"**Coordinates:** {row['lat']:.4f}°N, {row['lon']:.4f}°E")
            
            st.divider()
            
            st.subheader("🌡️ Climate")
            st.write(f"**Temperature:** {row['temp']}°C")
            st.write(f"**Precipitation:** {row['precip']} mm/day")
            st.write(f"**Humidity:** {row['humidity']}%")
        
        with col2:
            st.subheader("⚠️ Risk Assessment")
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=row['composite'],
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "#28a745"},
                        {'range': [0.3, 0.5], 'color': "#ffc107"},
                        {'range': [0.5, 0.7], 'color': "#fd7e14"},
                        {'range': [0.7, 1], 'color': "#dc3545"}
                    ]
                },
                title={'text': "Composite Risk"}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🌊 Flood", f"{row['flood']:.2f}")
            with col_b:
                st.metric("🔥 Heat", f"{row['heat']:.2f}")
            with col_c:
                st.metric("🏜️ Drought", f"{row['drought']:.2f}")
    
    st.divider()
    
    # Full table
    st.subheader("📋 All Communities")
    
    display_df = communities_df[['name', 'type', 'population', 'composite', 'flood', 'heat', 'drought', 'dominant']].copy()
    display_df.columns = ['Community', 'Type', 'Population', 'Composite', 'Flood', 'Heat', 'Drought', 'Dominant']
    
    st.dataframe(
        display_df.sort_values('Composite', ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 10px;">
    🌍 <strong>CCMEWS</strong> - North Tongu District Climate Monitoring<br>
    <small>Volta Region, Ghana | {len(communities_df)} monitoring points | API: {API_BASE}</small>
</div>
""", unsafe_allow_html=True)