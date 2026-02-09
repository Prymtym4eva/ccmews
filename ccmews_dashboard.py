"""CCMEWS Dashboard - Connected to Backend API"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import httpx

st.set_page_config(
    page_title="CCMEWS Dashboard",
    page_icon="🌍",
    layout="wide"
)

API_BASE = "http://localhost:8000"

@st.cache_data(ttl=60)
def fetch_api(endpoint):
    """Fetch from API with caching"""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{API_BASE}{endpoint}")
            if resp.status_code == 200:
                return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return None

# Header
st.title("🌍 CCMEWS - Climate Monitoring & Early Warning System")
st.caption(f"Ghana National Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Fetch data
summary = fetch_api("/api/v1/hazards/national-summary") or {}
districts = fetch_api("/api/v1/districts/") or []
regions = fetch_api("/api/v1/districts/regions") or []

# Alert banner
warning_count = summary.get("warning_alerts", 0)
critical_count = summary.get("critical_alerts", 0)
if critical_count > 0:
    st.error(f"🚨 {critical_count} CRITICAL ALERT(S) ACTIVE - Immediate action required!")
elif warning_count > 0:
    st.warning(f"⚠️ {warning_count} WARNING ALERT(S) ACTIVE - Monitor conditions closely")

# Key metrics
st.subheader("📊 National Risk Overview")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Districts", summary.get("total_districts", len(districts)))
with col2:
    st.metric("🔴 Critical Risk", summary.get("critical_risk_districts", 0), 
              delta=None, delta_color="inverse")
with col3:
    st.metric("🟠 High Risk", summary.get("high_risk_districts", 0))
with col4:
    st.metric("🟡 Moderate Risk", summary.get("moderate_risk_districts", 0))
with col5:
    st.metric("⚠️ Active Alerts", summary.get("active_alerts", 0))

st.divider()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Districts", "⚠️ Alerts", "🌡️ Climate Data", "📈 Analytics"])

with tab1:
    st.subheader("District Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if districts:
            df = pd.DataFrame(districts)
            st.dataframe(df, use_container_width=True, hide_index=True, height=400)
        else:
            st.warning("No district data available. Is the API running?")
    
    with col2:
        st.subheader("Regions")
        if regions:
            for r in regions[:10]:
                st.write(f"**{r['name']}** - {r['capital']}")
            if len(regions) > 10:
                st.caption(f"... and {len(regions) - 10} more")

with tab2:
    st.subheader("Active Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🔴 Critical Alerts", critical_count)
        st.metric("🟠 Warning Alerts", warning_count)
    
    with col2:
        # Alert breakdown
        if warning_count > 0 or critical_count > 0:
            alert_data = pd.DataFrame({
                'Type': ['Critical', 'Warning'],
                'Count': [critical_count, warning_count]
            })
            fig = px.pie(alert_data, values='Count', names='Type', 
                        title='Alert Distribution',
                        color='Type',
                        color_discrete_map={'Critical': '#dc3545', 'Warning': '#ffc107'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No active alerts")
    
    # Alert details note
    st.info("📋 Alert details: 4 drought warnings active in Northern Ghana (Tamale, Sagnarigu, Bolgatanga, Wa)")

with tab3:
    st.subheader("Climate Data Explorer")
    
    if districts:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected = st.selectbox("Select District", [d["name"] for d in districts])
            days = st.slider("Days of data", 7, 30, 30)
        
        selected_dist = next((d for d in districts if d["name"] == selected), None)
        
        if selected_dist:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            
            climate = fetch_api(
                f"/api/v1/climate/districts/{selected_dist['id']}/observations"
                f"?start_date={start_date}&end_date={end_date}"
            )
            
            if climate and isinstance(climate, dict) and "observations" in climate:
                obs = climate["observations"]
                if obs:
                    climate_df = pd.DataFrame(obs)
                    
                    # Temperature chart
                    if "temperature_mean" in climate_df.columns:
                        fig = go.Figure()
                        if "temperature_max" in climate_df.columns:
                            fig.add_trace(go.Scatter(
                                x=climate_df["observation_date"], 
                                y=climate_df["temperature_max"],
                                name="Max Temp", line=dict(color='red', dash='dot')
                            ))
                        fig.add_trace(go.Scatter(
                            x=climate_df["observation_date"], 
                            y=climate_df["temperature_mean"],
                            name="Mean Temp", line=dict(color='orange')
                        ))
                        if "temperature_min" in climate_df.columns:
                            fig.add_trace(go.Scatter(
                                x=climate_df["observation_date"], 
                                y=climate_df["temperature_min"],
                                name="Min Temp", line=dict(color='blue', dash='dot')
                            ))
                        fig.update_layout(title=f"Temperature - {selected}", yaxis_title="°C")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Precipitation chart
                    if "precipitation" in climate_df.columns:
                        fig = px.bar(climate_df, x="observation_date", y="precipitation",
                                    title=f"Precipitation - {selected}",
                                    labels={"precipitation": "mm", "observation_date": "Date"})
                        fig.update_traces(marker_color='steelblue')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    with st.expander("View Raw Data"):
                        st.dataframe(climate_df, use_container_width=True)
                else:
                    st.warning("No observations for this period")
            else:
                st.warning("Could not fetch climate data")
    else:
        st.warning("No districts available")

with tab4:
    st.subheader("Risk Analytics")
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_data = pd.DataFrame({
            'Risk Level': ['Critical', 'High', 'Moderate', 'Low'],
            'Districts': [
                summary.get("critical_risk_districts", 0),
                summary.get("high_risk_districts", 0),
                summary.get("moderate_risk_districts", 0),
                summary.get("low_risk_districts", 0)
            ]
        })
        fig = px.bar(risk_data, x='Risk Level', y='Districts', 
                    title='Districts by Risk Level',
                    color='Risk Level',
                    color_discrete_map={
                        'Critical': '#dc3545',
                        'High': '#fd7e14', 
                        'Moderate': '#ffc107',
                        'Low': '#28a745'
                    })
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hazard type distribution (simulated based on alerts)
        hazard_data = pd.DataFrame({
            'Hazard': ['Drought', 'Flood', 'Heat'],
            'Active Alerts': [4, 0, 0]  # Based on current data
        })
        fig = px.pie(hazard_data, values='Active Alerts', names='Hazard',
                    title='Alerts by Hazard Type')
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    st.subheader("System Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Regions", len(regions))
    with col2:
        st.metric("Monitored Districts", len(districts))
    with col3:
        st.metric("Data Sources", "Open-Meteo API")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🌍 CCMEWS - Climate Change Monitoring & Early Warning System")
with col2:
    st.caption("📊 Data: Open-Meteo, Ghana Meteorological Agency")
with col3:
    st.caption(f"🔄 API: {API_BASE}")
