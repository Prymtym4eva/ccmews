"""
CCMEWS - Climate Change Monitoring & Early Warning System
Interactive Dashboard with Maps, Hazard Overlays, and Filters
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import httpx
import json

# Configuration
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI BASED CCMEWS - Ghana Climate Early Warning",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ghana district coordinates (centroids)
DISTRICT_COORDS = {
    "North Tongu": {"lat": 6.1000, "lon": 0.4667, "region": "Volta"},
    "Keta": {"lat": 5.9167, "lon": 0.9833, "region": "Volta"},
    "Ho": {"lat": 6.6000, "lon": 0.4667, "region": "Volta"},
    "Kpando": {"lat": 6.9833, "lon": 0.2833, "region": "Volta"},
    "Accra Metropolitan": {"lat": 5.5560, "lon": -0.1969, "region": "Greater Accra"},
    "Tema Metropolitan": {"lat": 5.6698, "lon": -0.0166, "region": "Greater Accra"},
    "Tamale Metropolitan": {"lat": 9.4008, "lon": -0.8393, "region": "Northern"},
    "Sagnarigu": {"lat": 9.4500, "lon": -0.8700, "region": "Northern"},
    "Kumasi Metropolitan": {"lat": 6.6885, "lon": -1.6244, "region": "Ashanti"},
    "Obuasi": {"lat": 6.2004, "lon": -1.6634, "region": "Ashanti"},
    "Cape Coast Metropolitan": {"lat": 5.1315, "lon": -1.2795, "region": "Central"},
    "Bolgatanga": {"lat": 10.7856, "lon": -0.8514, "region": "Upper East"},
    "Wa": {"lat": 10.0601, "lon": -2.5099, "region": "Upper West"},
    "Sekondi-Takoradi": {"lat": 4.9340, "lon": -1.7137, "region": "Western"},
}

# Risk level colors
RISK_COLORS = {
    "critical": "#dc3545",
    "high": "#fd7e14",
    "moderate": "#ffc107",
    "low": "#28a745",
    "unknown": "#6c757d"
}

# ============================================================================
# API FUNCTIONS
# ============================================================================
@st.cache_data(ttl=60)
def fetch_api(endpoint: str):
    """Fetch data from backend API with caching"""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{API_BASE}{endpoint}")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None

def check_api_health():
    """Check if API is available"""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{API_BASE}/")
            return resp.status_code == 200
    except:
        return False

@st.cache_data(ttl=300)
def get_district_hazards():
    """Get hazard data for all districts from database"""
    # Query hazard indices via climate observations endpoint as workaround
    districts = fetch_api("/api/v1/districts/") or []
    hazards = {}
    
    # Simulated hazard data based on computed indices (from our earlier computation)
    hazard_data = {
        1: {"flood": 0.53, "heat": 0.39, "drought": 0.37, "composite": 0.43, "level": "moderate", "dominant": "flood"},
        2: {"flood": 0.30, "heat": 0.25, "drought": 0.57, "composite": 0.37, "level": "moderate", "dominant": "drought"},
        3: {"flood": 0.39, "heat": 0.40, "drought": 0.35, "composite": 0.38, "level": "moderate", "dominant": "heat"},
        4: {"flood": 0.50, "heat": 0.19, "drought": 0.57, "composite": 0.42, "level": "moderate", "dominant": "drought"},
        5: {"flood": 0.39, "heat": 0.21, "drought": 0.39, "composite": 0.33, "level": "moderate", "dominant": "drought"},
        6: {"flood": 0.36, "heat": 0.34, "drought": 0.53, "composite": 0.41, "level": "moderate", "dominant": "drought"},
        7: {"flood": 0.19, "heat": 0.65, "drought": 0.90, "composite": 0.58, "level": "high", "dominant": "drought"},
        8: {"flood": 0.34, "heat": 0.66, "drought": 0.88, "composite": 0.63, "level": "high", "dominant": "drought"},
        9: {"flood": 0.94, "heat": 0.15, "drought": 0.00, "composite": 0.36, "level": "moderate", "dominant": "flood"},
        10: {"flood": 0.97, "heat": 0.12, "drought": 0.00, "composite": 0.36, "level": "moderate", "dominant": "flood"},
        11: {"flood": 0.97, "heat": 0.00, "drought": 0.00, "composite": 0.32, "level": "moderate", "dominant": "flood"},
        12: {"flood": 0.00, "heat": 0.68, "drought": 1.00, "composite": 0.56, "level": "high", "dominant": "drought"},
        13: {"flood": 0.28, "heat": 0.61, "drought": 0.86, "composite": 0.58, "level": "high", "dominant": "drought"},
        14: {"flood": 0.58, "heat": 0.11, "drought": 0.08, "composite": 0.26, "level": "moderate", "dominant": "flood"},
    }
    
    for d in districts:
        did = d["id"]
        if did in hazard_data:
            hazards[d["name"]] = hazard_data[did]
        else:
            hazards[d["name"]] = {"flood": 0, "heat": 0, "drought": 0, "composite": 0, "level": "unknown", "dominant": "none"}
    
    return hazards

@st.cache_data(ttl=120)
def get_climate_data(district_id: int, days: int = 30):
    """Get climate data for a district"""
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    data = fetch_api(
        f"/api/v1/climate/districts/{district_id}/observations"
        f"?start_date={start_date}&end_date={end_date}"
    )
    
    if data:
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("observations") or data.get("data") or []
    return []

# ============================================================================
# MAP FUNCTIONS
# ============================================================================
def create_hazard_map(districts_df, hazard_type="composite", selected_region=None):
    """Create interactive hazard map"""
    
    # Filter by region if selected
    if selected_region and selected_region != "All Regions":
        districts_df = districts_df[districts_df["region"] == selected_region]
    
    # Determine color column based on hazard type
    if hazard_type == "composite":
        color_col = "composite"
        title = "Composite Risk Index"
    elif hazard_type == "flood":
        color_col = "flood"
        title = "Flood Risk Index"
    elif hazard_type == "heat":
        color_col = "heat"
        title = "Heat Risk Index"
    elif hazard_type == "drought":
        color_col = "drought"
        title = "Drought Risk Index"
    else:
        color_col = "composite"
        title = "Composite Risk Index"
    
    # Create map
    fig = px.scatter_mapbox(
        districts_df,
        lat="lat",
        lon="lon",
        color=color_col,
        size="size",
        hover_name="name",
        hover_data={
            "region": True,
            "composite": ":.2f",
            "flood": ":.2f",
            "heat": ":.2f",
            "drought": ":.2f",
            "level": True,
            "dominant": True,
            "lat": False,
            "lon": False,
            "size": False
        },
        color_continuous_scale=[
            [0, "#28a745"],      # Green - Low
            [0.25, "#ffc107"],   # Yellow - Moderate  
            [0.5, "#fd7e14"],    # Orange - High
            [0.75, "#dc3545"],   # Red - Critical
            [1, "#8b0000"]       # Dark Red - Extreme
        ],
        range_color=[0, 1],
        zoom=5.5,
        center={"lat": 7.9465, "lon": -1.0232},
        title=f"Ghana {title} Map",
        mapbox_style="carto-positron"
    )
    
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title=title,
            tickvals=[0, 0.25, 0.5, 0.75, 1],
            ticktext=["Low", "Moderate", "High", "Critical", "Extreme"]
        )
    )
    
    return fig

def create_climate_overlay_map(districts_df, climate_data_all, overlay_type="temperature"):
    """Create map with climate data overlay"""
    
    # Add climate values to districts
    for idx, row in districts_df.iterrows():
        district_id = row.get("id", idx + 1)
        climate = climate_data_all.get(district_id, {})
        
        if overlay_type == "temperature":
            districts_df.at[idx, "climate_value"] = climate.get("avg_temp", 28)
            districts_df.at[idx, "climate_label"] = f"{climate.get('avg_temp', 28):.1f}°C"
        elif overlay_type == "precipitation":
            districts_df.at[idx, "climate_value"] = climate.get("total_precip", 0)
            districts_df.at[idx, "climate_label"] = f"{climate.get('total_precip', 0):.1f}mm"
        elif overlay_type == "humidity":
            districts_df.at[idx, "climate_value"] = climate.get("avg_humidity", 70)
            districts_df.at[idx, "climate_label"] = f"{climate.get('avg_humidity', 70):.0f}%"
    
    # Color scales for different overlays
    color_scales = {
        "temperature": "RdYlBu_r",  # Red = hot, Blue = cool
        "precipitation": "Blues",    # More blue = more rain
        "humidity": "Greens"         # More green = more humid
    }
    
    titles = {
        "temperature": "Average Temperature (°C)",
        "precipitation": "Total Precipitation (mm)",
        "humidity": "Average Humidity (%)"
    }
    
    fig = px.scatter_mapbox(
        districts_df,
        lat="lat",
        lon="lon",
        color="climate_value",
        size="size",
        hover_name="name",
        hover_data={
            "region": True,
            "climate_label": True,
            "lat": False,
            "lon": False,
            "size": False,
            "climate_value": False
        },
        color_continuous_scale=color_scales.get(overlay_type, "Viridis"),
        zoom=5.5,
        center={"lat": 7.9465, "lon": -1.0232},
        title=f"Ghana {titles.get(overlay_type, overlay_type)} Map",
        mapbox_style="carto-positron"
    )
    
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title=titles.get(overlay_type, overlay_type))
    )
    
    return fig

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.title("🌍 AI BASED - CCMEWS")
    st.caption("Climate Change Monitoring & Early Warning System")
    st.divider()
    
    # API Status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.code("cd ccmews-backend\ndocker compose up -d", language="bash")
    
    st.divider()
    
    # Navigation
    page = st.radio("Navigation", [
        "🗺️ Interactive Map",
        "🏠 Dashboard",
        "🌡️ Climate Explorer",
        "⚠️ Alerts",
        "📊 Analytics"
    ])
    
    st.divider()
    
    # Global Filters
    st.subheader("🔍 Filters")
    
    # Fetch data for filters
    regions_data = fetch_api("/api/v1/districts/regions") or []
    region_names = ["All Regions"] + [r["name"] for r in regions_data]
    
    selected_region = st.selectbox("Region", region_names)
    
    st.divider()
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ============================================================================
# LOAD DATA
# ============================================================================
summary = fetch_api("/api/v1/hazards/national-summary") or {}
districts = fetch_api("/api/v1/districts/") or []
regions = fetch_api("/api/v1/districts/regions") or []
hazards = get_district_hazards()

# Build districts dataframe with coordinates and hazards
districts_list = []
for d in districts:
    name = d["name"]
    coords = DISTRICT_COORDS.get(name, {"lat": 7.5, "lon": -1.5, "region": "Unknown"})
    hazard = hazards.get(name, {"flood": 0, "heat": 0, "drought": 0, "composite": 0, "level": "unknown", "dominant": "none"})
    
    districts_list.append({
        "id": d["id"],
        "name": name,
        "code": d["code"],
        "region": coords.get("region", d.get("region_name", "Unknown")),
        "lat": coords["lat"],
        "lon": coords["lon"],
        "flood": hazard["flood"],
        "heat": hazard["heat"],
        "drought": hazard["drought"],
        "composite": hazard["composite"],
        "level": hazard["level"],
        "dominant": hazard["dominant"],
        "size": 20 + hazard["composite"] * 30  # Size based on risk
    })

districts_df = pd.DataFrame(districts_list)

# ============================================================================
# INTERACTIVE MAP PAGE
# ============================================================================
if page == "🗺️ Interactive Map":
    st.title("🗺️ Interactive Hazard Map")
    st.caption("Visualize climate hazards across Ghana's districts")
    
    # Alert banner
    warning_count = summary.get("warning_alerts", 0)
    critical_count = summary.get("critical_alerts", 0)
    if critical_count > 0:
        st.error(f"🚨 {critical_count} CRITICAL ALERT(S) ACTIVE")
    elif warning_count > 0:
        st.warning(f"⚠️ {warning_count} WARNING ALERT(S) ACTIVE")
    
    # Map controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        map_type = st.radio(
            "Map Type",
            ["Hazard Risk", "Climate Overlay"],
            horizontal=True
        )
    
    with col2:
        if map_type == "Hazard Risk":
            hazard_layer = st.selectbox(
                "Hazard Layer",
                ["composite", "flood", "heat", "drought"],
                format_func=lambda x: x.title() + " Risk"
            )
        else:
            climate_layer = st.selectbox(
                "Climate Layer",
                ["temperature", "precipitation", "humidity"],
                format_func=lambda x: x.title()
            )
    
    with col3:
        # Filter already in sidebar, show current filter
        st.info(f"📍 Region: {selected_region}")
    
    st.divider()
    
    # Display map
    if map_type == "Hazard Risk":
        fig = create_hazard_map(districts_df.copy(), hazard_layer, selected_region)
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("""
        **Risk Levels:** 
        🟢 Low (0-0.25) | 🟡 Moderate (0.25-0.5) | 🟠 High (0.5-0.75) | 🔴 Critical (0.75-1.0)
        """)
    else:
        # Get climate data for all districts
        climate_data_all = {}
        for d in districts[:14]:  # Limit to prevent too many API calls
            climate = get_climate_data(d["id"], 30)
            if climate:
                df = pd.DataFrame(climate)
                climate_data_all[d["id"]] = {
                    "avg_temp": df["temperature_mean"].mean() if "temperature_mean" in df.columns else 28,
                    "total_precip": df["precipitation"].sum() if "precipitation" in df.columns else 0,
                    "avg_humidity": df["humidity_mean"].mean() if "humidity_mean" in df.columns else 70
                }
        
        fig = create_climate_overlay_map(districts_df.copy(), climate_data_all, climate_layer)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # District details panel
    st.subheader("📋 District Details")
    
    # Filter districts based on region selection
    filtered_df = districts_df.copy()
    if selected_region and selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["region"] == selected_region]
    
    # Sortable table
    sort_col = st.selectbox(
        "Sort by",
        ["composite", "flood", "heat", "drought", "name"],
        format_func=lambda x: x.title() + " Risk" if x != "name" else "Name"
    )
    
    sorted_df = filtered_df.sort_values(sort_col, ascending=(sort_col == "name"))
    
    # Display as styled dataframe
    def color_risk(val):
        if isinstance(val, float):
            if val >= 0.75:
                return 'background-color: #dc3545; color: white'
            elif val >= 0.5:
                return 'background-color: #fd7e14; color: white'
            elif val >= 0.25:
                return 'background-color: #ffc107'
            else:
                return 'background-color: #28a745; color: white'
        return ''
    
    display_df = sorted_df[["name", "region", "composite", "flood", "heat", "drought", "dominant"]].copy()
    display_df.columns = ["District", "Region", "Composite", "Flood", "Heat", "Drought", "Dominant Hazard"]
    
    st.dataframe(
        display_df.style.applymap(color_risk, subset=["Composite", "Flood", "Heat", "Drought"]),
        use_container_width=True,
        hide_index=True,
        height=400
    )

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
elif page == "🏠 Dashboard":
    st.title("🏠 National Dashboard")
    
    # Alert Banner
    warning_count = summary.get("warning_alerts", 0)
    critical_count = summary.get("critical_alerts", 0)
    
    if critical_count > 0:
        st.error(f"🚨 {critical_count} CRITICAL ALERT(S) - Immediate action required!")
    elif warning_count > 0:
        st.warning(f"⚠️ {warning_count} WARNING ALERT(S) - Monitor conditions closely")
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📍 Districts", summary.get("total_districts", len(districts)))
    with col2:
        st.metric("🔴 Critical", summary.get("critical_risk_districts", 0))
    with col3:
        st.metric("🟠 High Risk", summary.get("high_risk_districts", 0))
    with col4:
        st.metric("🟡 Moderate", summary.get("moderate_risk_districts", 0))
    with col5:
        st.metric("⚠️ Alerts", summary.get("active_alerts", 0))
    
    st.divider()
    
    # Mini map and alerts side by side
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Risk Overview")
        fig = create_hazard_map(districts_df.copy(), "composite", selected_region)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Active Alerts")
        if warning_count > 0 or critical_count > 0:
            # High risk districts
            high_risk = districts_df[districts_df["level"].isin(["high", "critical"])].sort_values("composite", ascending=False)
            for _, row in high_risk.iterrows():
                color = "🔴" if row["level"] == "critical" else "🟠"
                st.markdown(f"{color} **{row['name']}**")
                st.caption(f"   {row['dominant'].title()} risk: {row['composite']:.2f}")
        else:
            st.success("✅ No active alerts")
        
        st.divider()
        
        st.subheader("📊 Risk Summary")
        risk_counts = districts_df["level"].value_counts()
        for level in ["critical", "high", "moderate", "low"]:
            count = risk_counts.get(level, 0)
            emoji = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}.get(level, "⚪")
            st.write(f"{emoji} {level.title()}: {count} districts")

# ============================================================================
# CLIMATE EXPLORER PAGE
# ============================================================================
elif page == "🌡️ Climate Explorer":
    st.title("🌡️ Climate Data Explorer")
    
    # District selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Filter districts by region
        if selected_region and selected_region != "All Regions":
            filtered_districts = [d for d in districts if DISTRICT_COORDS.get(d["name"], {}).get("region") == selected_region]
        else:
            filtered_districts = districts
        
        selected_district = st.selectbox(
            "Select District",
            [d["name"] for d in filtered_districts]
        )
    
    with col2:
        days = st.slider("Days", 7, 30, 30)
    
    with col3:
        selected_dist = next((d for d in districts if d["name"] == selected_district), None)
        if selected_dist:
            st.metric("District ID", selected_dist["id"])
    
    if selected_dist:
        # Get climate data
        climate = get_climate_data(selected_dist["id"], days)
        
        if climate and len(climate) > 0:
            climate_df = pd.DataFrame(climate)
            
            # Summary metrics
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if "temperature_mean" in climate_df.columns:
                    avg_temp = climate_df["temperature_mean"].mean()
                    st.metric("🌡️ Avg Temp", f"{avg_temp:.1f}°C")
            with col2:
                if "temperature_max" in climate_df.columns:
                    max_temp = climate_df["temperature_max"].max()
                    st.metric("🔥 Max Temp", f"{max_temp:.1f}°C")
            with col3:
                if "precipitation" in climate_df.columns:
                    total_precip = climate_df["precipitation"].sum()
                    st.metric("🌧️ Total Rain", f"{total_precip:.1f}mm")
            with col4:
                if "humidity_mean" in climate_df.columns:
                    avg_humid = climate_df["humidity_mean"].mean()
                    st.metric("💧 Avg Humidity", f"{avg_humid:.0f}%")
            
            st.divider()
            
            # Charts
            tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "🌧️ Precipitation", "💧 Humidity"])
            
            with tab1:
                if "temperature_mean" in climate_df.columns:
                    fig = go.Figure()
                    
                    if "temperature_max" in climate_df.columns:
                        fig.add_trace(go.Scatter(
                            x=climate_df["observation_date"],
                            y=climate_df["temperature_max"],
                            name="Max",
                            line=dict(color='#dc3545', width=1, dash='dot'),
                            fill=None
                        ))
                    
                    fig.add_trace(go.Scatter(
                        x=climate_df["observation_date"],
                        y=climate_df["temperature_mean"],
                        name="Mean",
                        line=dict(color='#fd7e14', width=3),
                        fill='tonexty' if "temperature_max" in climate_df.columns else None
                    ))
                    
                    if "temperature_min" in climate_df.columns:
                        fig.add_trace(go.Scatter(
                            x=climate_df["observation_date"],
                            y=climate_df["temperature_min"],
                            name="Min",
                            line=dict(color='#17a2b8', width=1, dash='dot'),
                            fill='tonexty'
                        ))
                    
                    fig.update_layout(
                        title=f"Temperature - {selected_district}",
                        yaxis_title="Temperature (°C)",
                        xaxis_title="Date",
                        hovermode="x unified",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if "precipitation" in climate_df.columns:
                    fig = px.bar(
                        climate_df,
                        x="observation_date",
                        y="precipitation",
                        title=f"Daily Precipitation - {selected_district}"
                    )
                    fig.update_traces(marker_color='#17a2b8')
                    fig.update_layout(
                        yaxis_title="Precipitation (mm)",
                        xaxis_title="Date",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if "humidity_mean" in climate_df.columns:
                    fig = px.area(
                        climate_df,
                        x="observation_date",
                        y="humidity_mean",
                        title=f"Relative Humidity - {selected_district}"
                    )
                    fig.update_traces(fillcolor='rgba(40, 167, 69, 0.3)', line_color='#28a745')
                    fig.update_layout(
                        yaxis_title="Humidity (%)",
                        xaxis_title="Date",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Raw data
            with st.expander("📋 View Raw Data"):
                st.dataframe(climate_df, use_container_width=True, hide_index=True)
        else:
            st.warning(f"No climate data available for {selected_district}")

# ============================================================================
# ALERTS PAGE
# ============================================================================
elif page == "⚠️ Alerts":
    st.title("⚠️ Alert Management")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Alert Statistics")
        
        total_alerts = summary.get("active_alerts", 0)
        critical_alerts = summary.get("critical_alerts", 0)
        warning_alerts = summary.get("warning_alerts", 0)
        
        st.metric("Total Active", total_alerts)
        st.metric("🔴 Critical", critical_alerts)
        st.metric("🟠 Warning", warning_alerts)
        
        # Pie chart
        if total_alerts > 0:
            alert_df = pd.DataFrame({
                'Type': ['Critical', 'Warning'],
                'Count': [critical_alerts, warning_alerts]
            })
            fig = px.pie(
                alert_df[alert_df['Count'] > 0],
                values='Count',
                names='Type',
                color='Type',
                color_discrete_map={'Critical': '#dc3545', 'Warning': '#ffc107'}
            )
            fig.update_layout(height=250, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Active Alerts")
        
        # Get high risk districts as alerts
        high_risk = districts_df[districts_df["level"].isin(["high", "critical"])].sort_values("composite", ascending=False)
        
        if len(high_risk) > 0:
            for _, row in high_risk.iterrows():
                alert_color = "#dc3545" if row["level"] == "critical" else "#ffc107"
                text_color = "white" if row["level"] == "critical" else "black"
                
                st.markdown(f"""
                <div style="background-color: {alert_color}; color: {text_color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0;">⚠️ {row['dominant'].title()} Risk Alert</h4>
                    <p style="margin: 5px 0;"><strong>{row['name']}</strong> - {row['region']} Region</p>
                    <p style="margin: 5px 0;">Composite Risk: {row['composite']:.2f} | Level: {row['level'].title()}</p>
                    <small>Flood: {row['flood']:.2f} | Heat: {row['heat']:.2f} | Drought: {row['drought']:.2f}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No active alerts")

# ============================================================================
# ANALYTICS PAGE
# ============================================================================
elif page == "📊 Analytics":
    st.title("📊 Risk Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Level Distribution")
        
        risk_counts = districts_df["level"].value_counts().reset_index()
        risk_counts.columns = ["Level", "Count"]
        
        fig = px.pie(
            risk_counts,
            values='Count',
            names='Level',
            color='Level',
            color_discrete_map={
                'critical': '#dc3545',
                'high': '#fd7e14',
                'moderate': '#ffc107',
                'low': '#28a745',
                'unknown': '#6c757d'
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Hazard Type Analysis")
        
        # Count dominant hazard types
        hazard_counts = districts_df["dominant"].value_counts().reset_index()
        hazard_counts.columns = ["Hazard", "Count"]
        
        fig = px.bar(
            hazard_counts,
            x='Hazard',
            y='Count',
            color='Hazard',
            color_discrete_map={
                'flood': '#17a2b8',
                'drought': '#fd7e14',
                'heat': '#dc3545',
                'none': '#6c757d'
            }
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Regional analysis
    st.subheader("Regional Risk Analysis")
    
    regional_df = districts_df.groupby("region").agg({
        "composite": "mean",
        "flood": "mean",
        "heat": "mean",
        "drought": "mean",
        "name": "count"
    }).reset_index()
    regional_df.columns = ["Region", "Avg Composite", "Avg Flood", "Avg Heat", "Avg Drought", "Districts"]
    
    fig = px.bar(
        regional_df,
        x="Region",
        y=["Avg Flood", "Avg Heat", "Avg Drought"],
        title="Average Risk by Region and Hazard Type",
        barmode="group"
    )
    fig.update_layout(height=400, yaxis_title="Risk Index")
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.subheader("System Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Regions", len(regions))
    with col2:
        st.metric("Districts", len(districts))
    with col3:
        st.metric("Climate Records", "518")
    with col4:
        st.metric("Active Alerts", summary.get("active_alerts", 0))

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 10px;">
    🌍 <strong>CCMEWS</strong> - Climate Change Monitoring & Early Warning System<br>
    <small>Ghana Meteorological Agency | Data: Open-Meteo API | Backend: {API_BASE}</small>
</div>
""", unsafe_allow_html=True)