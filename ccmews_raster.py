"""
CCMEWS - Climate Change Monitoring & Early Warning System
Interactive Dashboard with Raster Maps, Interpolated Surfaces, and Heatmaps
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import httpx
from scipy.interpolate import griddata, Rbf
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Configuration
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="CCMEWS - Ghana Climate Early Warning",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ghana bounds
GHANA_BOUNDS = {
    "lat_min": 4.5,
    "lat_max": 11.5,
    "lon_min": -3.5,
    "lon_max": 1.5,
    "center_lat": 7.9465,
    "center_lon": -1.0232
}

# District coordinates with hazard data
DISTRICTS_DATA = {
    "North Tongu": {"lat": 6.1000, "lon": 0.4667, "region": "Volta", "id": 1,
                   "flood": 0.53, "heat": 0.39, "drought": 0.37, "composite": 0.43},
    "Keta": {"lat": 5.9167, "lon": 0.9833, "region": "Volta", "id": 2,
             "flood": 0.30, "heat": 0.25, "drought": 0.57, "composite": 0.37},
    "Ho": {"lat": 6.6000, "lon": 0.4667, "region": "Volta", "id": 3,
           "flood": 0.39, "heat": 0.40, "drought": 0.35, "composite": 0.38},
    "Kpando": {"lat": 6.9833, "lon": 0.2833, "region": "Volta", "id": 4,
               "flood": 0.50, "heat": 0.19, "drought": 0.57, "composite": 0.42},
    "Accra Metropolitan": {"lat": 5.5560, "lon": -0.1969, "region": "Greater Accra", "id": 5,
                          "flood": 0.39, "heat": 0.21, "drought": 0.39, "composite": 0.33},
    "Tema Metropolitan": {"lat": 5.6698, "lon": -0.0166, "region": "Greater Accra", "id": 6,
                         "flood": 0.36, "heat": 0.34, "drought": 0.53, "composite": 0.41},
    "Tamale Metropolitan": {"lat": 9.4008, "lon": -0.8393, "region": "Northern", "id": 7,
                           "flood": 0.19, "heat": 0.65, "drought": 0.90, "composite": 0.58},
    "Sagnarigu": {"lat": 9.4500, "lon": -0.8700, "region": "Northern", "id": 8,
                  "flood": 0.34, "heat": 0.66, "drought": 0.88, "composite": 0.63},
    "Kumasi Metropolitan": {"lat": 6.6885, "lon": -1.6244, "region": "Ashanti", "id": 9,
                           "flood": 0.94, "heat": 0.15, "drought": 0.00, "composite": 0.36},
    "Obuasi": {"lat": 6.2004, "lon": -1.6634, "region": "Ashanti", "id": 10,
               "flood": 0.97, "heat": 0.12, "drought": 0.00, "composite": 0.36},
    "Cape Coast Metropolitan": {"lat": 5.1315, "lon": -1.2795, "region": "Central", "id": 11,
                               "flood": 0.97, "heat": 0.00, "drought": 0.00, "composite": 0.32},
    "Bolgatanga": {"lat": 10.7856, "lon": -0.8514, "region": "Upper East", "id": 12,
                   "flood": 0.00, "heat": 0.68, "drought": 1.00, "composite": 0.56},
    "Wa": {"lat": 10.0601, "lon": -2.5099, "region": "Upper West", "id": 13,
           "flood": 0.28, "heat": 0.61, "drought": 0.86, "composite": 0.58},
    "Sekondi-Takoradi": {"lat": 4.9340, "lon": -1.7137, "region": "Western", "id": 14,
                        "flood": 0.58, "heat": 0.11, "drought": 0.08, "composite": 0.26},
}

# Simulated climate data (would come from API in production)
CLIMATE_DATA = {
    "North Tongu": {"temp": 29.3, "precip": 3.2, "humidity": 78},
    "Keta": {"temp": 28.9, "precip": 2.8, "humidity": 82},
    "Ho": {"temp": 28.6, "precip": 4.1, "humidity": 79},
    "Kpando": {"temp": 29.5, "precip": 2.9, "humidity": 74},
    "Accra Metropolitan": {"temp": 28.7, "precip": 2.1, "humidity": 80},
    "Tema Metropolitan": {"temp": 29.2, "precip": 1.8, "humidity": 77},
    "Tamale Metropolitan": {"temp": 32.1, "precip": 0.5, "humidity": 45},
    "Sagnarigu": {"temp": 32.5, "precip": 0.4, "humidity": 42},
    "Kumasi Metropolitan": {"temp": 27.6, "precip": 5.2, "humidity": 80},
    "Obuasi": {"temp": 27.2, "precip": 5.8, "humidity": 82},
    "Cape Coast Metropolitan": {"temp": 27.8, "precip": 6.1, "humidity": 86},
    "Bolgatanga": {"temp": 33.5, "precip": 0.1, "humidity": 35},
    "Wa": {"temp": 31.8, "precip": 0.8, "humidity": 40},
    "Sekondi-Takoradi": {"temp": 27.5, "precip": 5.5, "humidity": 86},
}

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
    except:
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
def get_climate_data(district_id: int, days: int = 30):
    """Get climate data for a district"""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    data = fetch_api(f"/api/v1/climate/districts/{district_id}/observations?start_date={start_date}&end_date={end_date}")
    if data:
        return data if isinstance(data, list) else data.get("observations", [])
    return []

# ============================================================================
# RASTER/INTERPOLATION FUNCTIONS
# ============================================================================
def create_interpolated_grid(points_df, value_column, resolution=50):
    """Create interpolated grid from point data using RBF interpolation"""
    
    # Extract coordinates and values
    lats = points_df['lat'].values
    lons = points_df['lon'].values
    values = points_df[value_column].values
    
    # Create grid
    grid_lon = np.linspace(GHANA_BOUNDS['lon_min'], GHANA_BOUNDS['lon_max'], resolution)
    grid_lat = np.linspace(GHANA_BOUNDS['lat_min'], GHANA_BOUNDS['lat_max'], resolution)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate using RBF (Radial Basis Function) for smooth surfaces
    try:
        rbf = Rbf(lons, lats, values, function='multiquadric', smooth=0.5)
        grid_values = rbf(grid_lon_mesh, grid_lat_mesh)
    except:
        # Fallback to linear interpolation
        points = np.column_stack((lons, lats))
        grid_values = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='cubic')
        # Fill NaN with nearest neighbor
        if np.any(np.isnan(grid_values)):
            grid_values_nearest = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='nearest')
            grid_values = np.where(np.isnan(grid_values), grid_values_nearest, grid_values)
    
    # Clip values to valid range
    grid_values = np.clip(grid_values, 0, 1) if value_column in ['flood', 'heat', 'drought', 'composite'] else grid_values
    
    return grid_lon, grid_lat, grid_values

def create_raster_heatmap(points_df, value_column, title, colorscale='RdYlGn_r', resolution=50):
    """Create a plotly heatmap/contour map"""
    
    grid_lon, grid_lat, grid_values = create_interpolated_grid(points_df, value_column, resolution)
    
    fig = go.Figure()
    
    # Add interpolated surface as heatmap
    fig.add_trace(go.Heatmap(
        x=grid_lon,
        y=grid_lat,
        z=grid_values,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=title),
        hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>'
    ))
    
    # Add contour lines
    fig.add_trace(go.Contour(
        x=grid_lon,
        y=grid_lat,
        z=grid_values,
        colorscale=colorscale,
        showscale=False,
        contours=dict(
            showlines=True,
            coloring='none',
            showlabels=True,
            labelfont=dict(size=10, color='black')
        ),
        line=dict(width=1, color='rgba(0,0,0,0.3)')
    ))
    
    # Add district markers
    fig.add_trace(go.Scatter(
        x=points_df['lon'],
        y=points_df['lat'],
        mode='markers+text',
        marker=dict(size=10, color='black', symbol='circle'),
        text=points_df['name'].apply(lambda x: x.split()[0]),  # First word only
        textposition='top center',
        textfont=dict(size=8),
        hovertemplate='%{customdata[0]}<br>Value: %{customdata[1]:.2f}<extra></extra>',
        customdata=np.column_stack((points_df['name'], points_df[value_column]))
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600,
        xaxis=dict(range=[GHANA_BOUNDS['lon_min'], GHANA_BOUNDS['lon_max']]),
        yaxis=dict(range=[GHANA_BOUNDS['lat_min'], GHANA_BOUNDS['lat_max']]),
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1
    )
    
    return fig

def create_contour_map(points_df, value_column, title, colorscale='RdYlGn_r', resolution=50):
    """Create filled contour map"""
    
    grid_lon, grid_lat, grid_values = create_interpolated_grid(points_df, value_column, resolution)
    
    fig = go.Figure()
    
    # Filled contours
    fig.add_trace(go.Contour(
        x=grid_lon,
        y=grid_lat,
        z=grid_values,
        colorscale=colorscale,
        contours=dict(
            coloring='heatmap',
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        colorbar=dict(title=title),
        hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>'
    ))
    
    # District markers
    fig.add_trace(go.Scatter(
        x=points_df['lon'],
        y=points_df['lat'],
        mode='markers',
        marker=dict(size=12, color='white', symbol='circle', line=dict(width=2, color='black')),
        hovertemplate='%{customdata[0]}<br>%{customdata[1]}: %{customdata[2]:.2f}<extra></extra>',
        customdata=np.column_stack((points_df['name'], [value_column]*len(points_df), points_df[value_column]))
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600,
        xaxis=dict(range=[GHANA_BOUNDS['lon_min'], GHANA_BOUNDS['lon_max']]),
        yaxis=dict(range=[GHANA_BOUNDS['lat_min'], GHANA_BOUNDS['lat_max']]),
        yaxis_scaleanchor="x"
    )
    
    return fig

def create_folium_heatmap(points_df, value_column, title):
    """Create Folium heatmap with smooth gradient"""
    
    # Create base map
    m = folium.Map(
        location=[GHANA_BOUNDS['center_lat'], GHANA_BOUNDS['center_lon']],
        zoom_start=7,
        tiles='cartodbpositron'
    )
    
    # Prepare heatmap data: [lat, lon, intensity]
    heat_data = []
    for _, row in points_df.iterrows():
        # Add multiple points around each location for smoother interpolation
        base_val = row[value_column]
        heat_data.append([row['lat'], row['lon'], base_val])
        # Add surrounding points with slightly lower values for gradient effect
        for d in [0.2, 0.4]:
            for dlat, dlon in [(d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, -d), (d, -d), (-d, d)]:
                heat_data.append([row['lat'] + dlat, row['lon'] + dlon, base_val * 0.7])
    
    # Add heatmap layer
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_val=1.0,
        radius=40,
        blur=30,
        gradient={
            0.0: 'green',
            0.25: 'yellow',
            0.5: 'orange',
            0.75: 'red',
            1.0: 'darkred'
        }
    ).add_to(m)
    
    # Add district markers
    for _, row in points_df.iterrows():
        value = row[value_column]
        color = 'green' if value < 0.25 else 'orange' if value < 0.5 else 'red' if value < 0.75 else 'darkred'
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            color='black',
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            popup=f"<b>{row['name']}</b><br>{value_column}: {value:.2f}",
            tooltip=row['name']
        ).add_to(m)
    
    return m

def create_3d_surface(points_df, value_column, title, colorscale='RdYlGn_r'):
    """Create 3D surface plot"""
    
    grid_lon, grid_lat, grid_values = create_interpolated_grid(points_df, value_column, resolution=30)
    
    fig = go.Figure(data=[go.Surface(
        x=grid_lon,
        y=grid_lat,
        z=grid_values,
        colorscale=colorscale,
        colorbar=dict(title=value_column.title()),
        hovertemplate='Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>Value: %{z:.2f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title=value_column.title(),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600
    )
    
    return fig

# ============================================================================
# BUILD DATAFRAME
# ============================================================================
def build_districts_df():
    """Build districts dataframe with all data"""
    data = []
    for name, info in DISTRICTS_DATA.items():
        climate = CLIMATE_DATA.get(name, {})
        data.append({
            "name": name,
            "id": info["id"],
            "lat": info["lat"],
            "lon": info["lon"],
            "region": info["region"],
            "flood": info["flood"],
            "heat": info["heat"],
            "drought": info["drought"],
            "composite": info["composite"],
            "temp": climate.get("temp", 28),
            "precip": climate.get("precip", 3),
            "humidity": climate.get("humidity", 70),
            "level": "critical" if info["composite"] >= 0.75 else "high" if info["composite"] >= 0.5 else "moderate" if info["composite"] >= 0.25 else "low"
        })
    return pd.DataFrame(data)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("🌍 CCMEWS")
    st.caption("Climate Monitoring & Early Warning")
    st.divider()
    
    # API Status
    api_healthy = check_api_health()
    st.success("✅ API Connected") if api_healthy else st.error("❌ API Offline")
    
    st.divider()
    
    # Navigation
    page = st.radio("Navigation", [
        "🗺️ Raster Maps",
        "🌡️ Climate Maps",
        "📊 3D Visualization",
        "🏠 Dashboard",
        "⚠️ Alerts"
    ])
    
    st.divider()
    
    # Filters
    st.subheader("🔍 Filters")
    districts_df = build_districts_df()
    region_names = ["All Regions"] + sorted(districts_df["region"].unique().tolist())
    selected_region = st.selectbox("Region", region_names)
    
    st.divider()
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# Load data
districts_df = build_districts_df()
summary = fetch_api("/api/v1/hazards/national-summary") or {}

# Filter by region
if selected_region != "All Regions":
    filtered_df = districts_df[districts_df["region"] == selected_region]
else:
    filtered_df = districts_df

# ============================================================================
# RASTER MAPS PAGE
# ============================================================================
if page == "🗺️ Raster Maps":
    st.title("🗺️ Interpolated Hazard Raster Maps")
    st.caption("Continuous surface visualization of climate hazards across Ghana")
    
    # Map controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hazard_type = st.selectbox(
            "Hazard Type",
            ["composite", "flood", "heat", "drought"],
            format_func=lambda x: x.title() + " Risk"
        )
    
    with col2:
        map_style = st.selectbox(
            "Visualization Style",
            ["Heatmap", "Contour", "Folium Heatmap"],
        )
    
    with col3:
        resolution = st.slider("Resolution", 20, 100, 50)
    
    st.divider()
    
    # Color scales for hazards (red = bad)
    hazard_colorscales = {
        "composite": "RdYlGn_r",
        "flood": "Blues",
        "heat": "YlOrRd",
        "drought": "YlOrBr"
    }
    
    # Generate map
    if map_style == "Heatmap":
        fig = create_raster_heatmap(
            filtered_df, 
            hazard_type, 
            f"{hazard_type.title()} Risk Index - Ghana",
            hazard_colorscales.get(hazard_type, "RdYlGn_r"),
            resolution
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif map_style == "Contour":
        fig = create_contour_map(
            filtered_df,
            hazard_type,
            f"{hazard_type.title()} Risk Contours - Ghana",
            hazard_colorscales.get(hazard_type, "RdYlGn_r"),
            resolution
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif map_style == "Folium Heatmap":
        st.subheader(f"{hazard_type.title()} Risk Heatmap")
        folium_map = create_folium_heatmap(filtered_df, hazard_type, f"{hazard_type.title()} Risk")
        st_folium(folium_map, width=None, height=600)
    
    # Legend
    st.markdown("""
    **Risk Levels:** 🟢 Low (0-0.25) | 🟡 Moderate (0.25-0.5) | 🟠 High (0.5-0.75) | 🔴 Critical (0.75-1.0)
    """)
    
    st.divider()
    
    # Show all hazard types side by side
    st.subheader("📊 All Hazard Types Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_contour_map(filtered_df, "flood", "Flood Risk", "Blues", 30)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = create_contour_map(filtered_df, "drought", "Drought Risk", "YlOrBr", 30)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_contour_map(filtered_df, "heat", "Heat Risk", "YlOrRd", 30)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        fig = create_contour_map(filtered_df, "composite", "Composite Risk", "RdYlGn_r", 30)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CLIMATE MAPS PAGE
# ============================================================================
elif page == "🌡️ Climate Maps":
    st.title("🌡️ Climate Variable Raster Maps")
    st.caption("Interpolated climate conditions across Ghana")
    
    col1, col2 = st.columns(2)
    
    with col1:
        climate_var = st.selectbox(
            "Climate Variable",
            ["temp", "precip", "humidity"],
            format_func=lambda x: {"temp": "Temperature (°C)", "precip": "Precipitation (mm)", "humidity": "Humidity (%)"}[x]
        )
    
    with col2:
        map_style = st.selectbox("Style", ["Heatmap", "Contour", "Folium"])
    
    # Color scales for climate
    climate_colorscales = {
        "temp": "RdBu_r",      # Red = hot
        "precip": "Blues",     # Blue = wet
        "humidity": "Greens"   # Green = humid
    }
    
    titles = {
        "temp": "Temperature (°C)",
        "precip": "Precipitation (mm/day)",
        "humidity": "Relative Humidity (%)"
    }
    
    st.divider()
    
    if map_style == "Heatmap":
        fig = create_raster_heatmap(
            filtered_df,
            climate_var,
            titles[climate_var],
            climate_colorscales[climate_var],
            50
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif map_style == "Contour":
        fig = create_contour_map(
            filtered_df,
            climate_var,
            titles[climate_var],
            climate_colorscales[climate_var],
            50
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        folium_map = create_folium_heatmap(filtered_df, climate_var, titles[climate_var])
        st_folium(folium_map, width=None, height=600)
    
    st.divider()
    
    # All climate variables
    st.subheader("📊 Climate Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = create_contour_map(filtered_df, "temp", "Temperature (°C)", "RdBu_r", 30)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_contour_map(filtered_df, "precip", "Precipitation (mm)", "Blues", 30)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_contour_map(filtered_df, "humidity", "Humidity (%)", "Greens", 30)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 3D VISUALIZATION PAGE
# ============================================================================
elif page == "📊 3D Visualization":
    st.title("📊 3D Surface Visualization")
    st.caption("Interactive 3D surface maps of hazards and climate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        var_type = st.selectbox(
            "Variable",
            ["composite", "flood", "heat", "drought", "temp", "precip", "humidity"],
            format_func=lambda x: {
                "composite": "Composite Risk",
                "flood": "Flood Risk",
                "heat": "Heat Risk",
                "drought": "Drought Risk",
                "temp": "Temperature",
                "precip": "Precipitation",
                "humidity": "Humidity"
            }[x]
        )
    
    with col2:
        colorscale = st.selectbox(
            "Color Scale",
            ["RdYlGn_r", "Viridis", "Plasma", "Blues", "RdBu_r", "YlOrRd"]
        )
    
    st.divider()
    
    fig = create_3d_surface(filtered_df, var_type, f"3D Surface - {var_type.title()}", colorscale)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Drag to rotate, scroll to zoom, right-click to pan")

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
elif page == "🏠 Dashboard":
    st.title("🏠 National Dashboard")
    
    # Alert banner
    warning_count = summary.get("warning_alerts", 0)
    critical_count = summary.get("critical_alerts", 0)
    
    if critical_count > 0:
        st.error(f"🚨 {critical_count} CRITICAL ALERT(S)")
    elif warning_count > 0:
        st.warning(f"⚠️ {warning_count} WARNING ALERT(S)")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📍 Districts", len(filtered_df))
    with col2:
        st.metric("🔴 Critical", len(filtered_df[filtered_df["level"] == "critical"]))
    with col3:
        st.metric("🟠 High", len(filtered_df[filtered_df["level"] == "high"]))
    with col4:
        st.metric("🟡 Moderate", len(filtered_df[filtered_df["level"] == "moderate"]))
    with col5:
        st.metric("⚠️ Alerts", summary.get("active_alerts", 4))
    
    st.divider()
    
    # Mini raster map
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_contour_map(filtered_df, "composite", "Composite Risk", "RdYlGn_r", 40)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ High Risk Areas")
        high_risk = filtered_df[filtered_df["composite"] >= 0.5].sort_values("composite", ascending=False)
        for _, row in high_risk.iterrows():
            emoji = "🔴" if row["composite"] >= 0.75 else "🟠"
            st.markdown(f"{emoji} **{row['name']}** ({row['region']})")
            st.caption(f"   Risk: {row['composite']:.2f}")

# ============================================================================
# ALERTS PAGE
# ============================================================================
elif page == "⚠️ Alerts":
    st.title("⚠️ Alert Management")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Statistics")
        high_risk = filtered_df[filtered_df["composite"] >= 0.5]
        critical = filtered_df[filtered_df["composite"] >= 0.75]
        
        st.metric("Total Active", len(high_risk))
        st.metric("🔴 Critical", len(critical))
        st.metric("🟠 Warning", len(high_risk) - len(critical))
    
    with col2:
        st.subheader("Active Alerts")
        for _, row in high_risk.sort_values("composite", ascending=False).iterrows():
            color = "#dc3545" if row["composite"] >= 0.75 else "#ffc107"
            text_color = "white" if row["composite"] >= 0.75 else "black"
            st.markdown(f"""
            <div style="background-color: {color}; color: {text_color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h4 style="margin: 0;">⚠️ High Risk Alert</h4>
                <p><strong>{row['name']}</strong> - {row['region']}</p>
                <small>Flood: {row['flood']:.2f} | Heat: {row['heat']:.2f} | Drought: {row['drought']:.2f}</small>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption(f"🌍 CCMEWS | API: {API_BASE} | Updated: {datetime.now().strftime('%H:%M')}")
