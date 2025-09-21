import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Digital Twin | Solar Irradiance",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded" # Make sidebar visible by default
)

# --- Custom CSS for a Polished Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main app background */
    .stApp {
        background-color: #0B1120;
        color: #EAECEF;
    }

    /* --- FIX FOR HEADER --- */
    /* Change header to match the app background */
    [data-testid="stHeader"] {
        background-color: #0B1120 !important;
    }
    /* --- End of Header Fix --- */


    /* --- FINAL, ROBUST FIX FOR SIDEBAR --- */

    /* 1. Target the sidebar container itself and hide any overflow */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        overflow: hidden !important;
    }

    /* 2. Prevent the sidebar navigation from being used to collapse it */
    div[data-testid="stSidebarNav"] {
        pointer-events: none !important;
    }
    
    /* 3. Aggressively hide the collapse button and its container */
    button[data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarContent"] > div:first-child[class*="st-emotion-cache-"] {
        display: none !important;
        visibility: hidden !important;
        width: 0px !important;
        height: 0px !important;
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !importan;
        z-index: -9999 !important;
    }

    /* --- End of Sidebar Fix --- */
    
    /* --- UPDATED GRAPHIC CONTAINER --- */
    /* This container no longer has a background or border */
    .graphic-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 220px; /* Ensures vertical alignment */
    }

    /* --- NEW: Navigation Tab Styling --- */
    .nav-link {
        display: block;
        padding: 0.5rem 0;
        color: #9CA3AF; /* Dimmed color for inactive tabs */
        text-decoration: none;
        font-weight: 600;
        text-align: center;
        border-bottom: 3px solid transparent; /* Placeholder for active state */
        transition: all 0.2s ease-in-out;
    }
    .nav-link:hover {
        color: #EAECEF; /* Brighten on hover */
    }
    .nav-link.active {
        color: #34D399; /* Active color */
        border-bottom-color: #34D399; /* Underline for active tab */
    }

</style>
""", unsafe_allow_html=True)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(filepath, panel_area, initial_efficiency, loss_per_year):
    """
    Loads and preprocesses data.
    - Calculates predicted and actual power production based on PV system parameters.
    - Calculates dynamic uncertainty quantification.
    """
    if not os.path.exists(filepath):
        st.sidebar.error(f"Dataset not found at '{filepath}'. Please ensure 'gru_predictions_full.csv' is in the same folder.")
        return None
    
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    # FIX: Corrected the column name from 'Actual__Diffuse' to 'Actual_Diffuse'
    df['Actual_Global'] = (df['Actual_Direct'] + df['Actual_Diffuse']).clip(lower=0)
    df['Predicted_Global'] = (df['Predicted_Direct'] + df['Predicted_Diffuse']).clip(lower=0)
    df['Error'] = df['Actual_Global'] - df['Predicted_Global']
    
    # Persistence model calculation
    df['Persistence_Global'] = df['Actual_Global'].shift(1)
    
    # DYNAMIC UNCERTAINTY QUANTIFICATION
    rolling_std = df['Error'].rolling(window=24, min_periods=1).std().bfill()
    df['Predicted_Upper'] = (df['Predicted_Global'] + 1.96 * rolling_std).clip(lower=0)
    df['Predicted_Lower'] = (df['Predicted_Global'] - 1.96 * rolling_std).clip(lower=0)
    
    # POWER PRODUCTION CALCULATION
    base_year = df['Datetime'].dt.year.min() 
    df['Year'] = df['Datetime'].dt.year
    df['Efficiency'] = initial_efficiency - ((df['Year'] - base_year) * loss_per_year)
    
    df['Predicted_Power_kWh'] = (df['Predicted_Global'] * panel_area * df['Efficiency']) / 1000
    df['Actual_Power_kWh'] = (df['Actual_Global'] * panel_area * df['Efficiency']) / 1000

    return df

# --- XAI and Plotting Functions ---

def plot_simplified_xai_bar_chart(selected_point_df, full_data_context):
    """Creates a simple, intuitive bar chart to explain a single prediction."""
    if selected_point_df.empty:
        return go.Figure()

    point = selected_point_df.iloc[0]
    features = ['Temp (C)', 'Cloudcover (%)', 'Wind Speed (m/s)', 'Hour']
    
    avg_values = full_data_context[features].mean()
    deviations = {feat: point[feat] - avg_values[feat] for feat in features}
    
    impacts = {k: v * 0.5 for k, v in deviations.items()}
    
    impact_df = pd.DataFrame(list(impacts.items()), columns=['Feature', 'Impact'])
    impact_df['Color'] = np.where(impact_df['Impact'] > 0, '#F87171', '#60A5FA')
    impact_df['AbsoluteImpact'] = impact_df['Impact'].abs()
    impact_df = impact_df.sort_values(by='AbsoluteImpact', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=impact_df['Impact'],
        y=impact_df['Feature'],
        orientation='h',
        marker_color=impact_df['Color'],
        text=impact_df['Impact'].apply(lambda x: f'{x:+.2f}'),
        textposition='outside',
    ))

    final_prediction = point['Predicted_Global']
    fig.update_layout(
        title=f"Feature Impact on Forecast for {point['Datetime'].strftime('%Y-%m-%d %H:%M')} (Prediction: {final_prediction:.2f} W/m²)",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        xaxis_title="Impact on Prediction (Red = Increase, Blue = Decrease)",
        yaxis_title="Feature",
        showlegend=False,
        margin=dict(l=120)
    )
    return fig

def plot_main_irradiance(df, show_persistence=False):
    """Plots the main time-series chart with confidence intervals and optional persistence model."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([df['Datetime'], df['Datetime'][::-1]]),
        y=np.concatenate([df['Predicted_Upper'], df['Predicted_Lower'][::-1]]),
        fill='toself', fillcolor='rgba(56, 189, 248, 0.2)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="none", name='95% CI (Dynamic)'
    ))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Actual_Global'], name='Actual GHI', mode='lines', line=dict(color='#34D399', width=2.5)))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Predicted_Global'], name='GRU Model', mode='lines', line=dict(color='#FACC15', width=2.5, dash='dash')))
    
    if show_persistence:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Persistence_Global'], name='Persistence Model', mode='lines', line=dict(color='#F87171', width=2, dash='dot')))

    fig.update_layout(
        title="Solar Irradiance: Forecast vs. Actual", template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30, 41, 59, 0.5)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title='Irradiance (W/m²)', xaxis_range=[df['Datetime'].min(), df['Datetime'].max()]
    )
    return fig

def plot_power_production(df):
    """Plots the predicted power production."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Predicted_Power_kWh'], name='Predicted Power', mode='lines', line=dict(color='#FACC15', width=2.5)))
    fig.update_layout(
        title="Predicted Power Output Over Time", template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30, 41, 59, 0.5)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title='Power Output (kWh)', xaxis_range=[df['Datetime'].min(), df['Datetime'].max()]
    )
    return fig
    
def plot_hourly_earnings(df, price_per_kwh):
    """Plots the predicted earnings for each hour as a bar chart."""
    df_plot = df.copy()
    df_plot['Hourly_Earnings'] = df_plot['Predicted_Power_kWh'] * price_per_kwh
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot['Datetime'], 
        y=df_plot['Hourly_Earnings'], 
        name='Hourly Earnings',
        marker_color='#FACC15'
    ))
    fig.update_layout(
        title="Predicted Hourly Earnings",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        yaxis_title='Earnings (Rs)',
        xaxis_range=[df['Datetime'].min(), df['Datetime'].max()]
    )
    return fig

def plot_efficiency_degradation(initial_efficiency, loss_per_year, years):
    """Creates a bar chart to visualize the annual panel efficiency degradation."""
    base_year = min(years)
    efficiencies = [initial_efficiency - ((year - base_year) * loss_per_year) for year in years]
    efficiencies_percent = [eff * 100 for eff in efficiencies]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years,
        y=efficiencies_percent,
        text=[f'{eff:.2f}%' for eff in efficiencies_percent],
        textposition='auto',
        marker_color='#60A5FA'
    ))
    
    fig.update_layout(
        title="Annual Panel Efficiency Degradation",
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        xaxis_title='Year',
        yaxis_title='Efficiency (%)',
        yaxis_range=[0, (initial_efficiency * 100) + 5]
    )
    return fig

# --- Analysis Plotting Functions ---
def plot_residual_error_analysis(df):
    """Plots a line and filled area chart of the model's residual errors."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Error'], mode='lines', line=dict(color='#EAECEF', width=2), name='Error'))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Error'].clip(lower=0), fill='tozeroy', mode='none', fillcolor='rgba(52, 211, 153, 0.5)', name='Under-prediction'))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Error'].clip(upper=0), fill='tozeroy', mode='none', fillcolor='rgba(248, 113, 113, 0.5)', name='Over-prediction'))
    fig.add_hline(y=0, line_width=1.5, line_color="#9CA3AF")
    fig.update_layout(
        title_text="Irradiance Residual Error Deviation Over Time", template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)', yaxis_title="Error (W/m²)", xaxis_title="Time",
        xaxis_range=[df['Datetime'].min(), df['Datetime'].max()], showlegend=False
    )
    return fig

def plot_model_inputs(df):
    """Creates time-series charts for the main model input features."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Temperature (°C)', 'Cloud Cover (%)', 'Wind Speed (m/s)'))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Temp (C)'], mode='lines', name='Temperature', line=dict(color='#F87171')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Cloudcover (%)'], mode='lines', name='Cloud Cover', line=dict(color='#9CA3AF')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Wind Speed (m/s)'], mode='lines', name='Wind Speed', line=dict(color='#60A5FA')), row=3, col=1)
    fig.update_layout(
        title_text="Model Input Feature Analysis Over Time", template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)', showlegend=False, height=600
    )
    fig.update_yaxes(title_text="°C", row=1, col=1); fig.update_yaxes(title_text="%", row=2, col=1); fig.update_yaxes(title_text="m/s", row=3, col=1)
    return fig

def plot_error_heatmap(df):
    """Creates a heatmap of average error by hour and month."""
    df['Month'] = df['Datetime'].dt.strftime('%B')
    heatmap_data = df.pivot_table(index='Hour', columns='Month', values='Error', aggfunc='mean')
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    heatmap_data = heatmap_data.reindex(columns=month_order)
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index,
        colorscale='RdBu_r', zmid=0,
        hovertemplate='Month: %{x}<br>Hour: %{y}<br>Avg Error: %{z:.2f} W/m²<extra></extra>'
    ))
    fig.update_layout(title='Diurnal and Seasonal Error Heatmap', xaxis_title='Month', yaxis_title='Hour of Day', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30, 41, 59, 0.5)')
    return fig

def plot_combined_error_driver_scatter(df):
    """Creates a single scatter plot to visualize the combined impact of weather features on error."""
    if df.empty or df['Error'].abs().sum() == 0:
        return go.Figure().update_layout(title="Not enough data to display combined error analysis.")
    
    df['Absolute_Error'] = df['Error'].abs()
    
    fig = px.scatter(
        df,
        x='Temp (C)',
        y='Wind Speed (m/s)',
        size='Absolute_Error',
        color='Cloudcover (%)',
        color_continuous_scale='Viridis',
        hover_name=df.index,
        hover_data={'Error': ':.2f', 'Cloudcover (%)': True, 'Temp (C)': True, 'Wind Speed (m/s)': True, 'Absolute_Error': False},
        title="Combined Error Driver Analysis"
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30, 41, 59, 0.5)',
        xaxis_title="Temperature (°C)",
        yaxis_title="Wind Speed (m/s)",
        coloraxis_colorbar_title_text='Cloud Cover (%)'
    )
    fig.update_traces(marker=dict(sizemin=4))
    return fig


# --- SVG & Animated Metric Functions ---
def create_earnings_graphic(earnings, period_text):
    """Creates an animated SVG graphic for predicted earnings."""
    formatted_earnings = f"₹ {earnings:,.0f}"
    return f"""
    <div class="graphic-container">
        <svg width="100" height="100" viewBox="0 0 100 100">
            <style>
                @keyframes spin {{ 
                    0% {{ transform: rotateY(0deg); }} 
                    100% {{ transform: rotateY(360deg); }} 
                }}
                .coin {{ 
                    animation: spin 3s linear infinite; 
                    transform-origin: center;
                }}
            </style>
            <g class="coin">
                <circle cx="50" cy="50" r="45" fill="#FACC15" />
                <circle cx="50" cy="50" r="40" fill="none" stroke="#EAB308" stroke-width="4"/>
                <text x="50" y="60" font-family="Inter, sans-serif" font-size="30" fill="#374151" text-anchor="middle" font-weight="bold">₹</text>
            </g>
        </svg>
        <p style="font-size: 1.5rem; font-weight: 600; color: #EAECEF; margin-top: 10px;">{formatted_earnings}</p>
        <p style="font-size: 1rem; color: #9CA3AF; margin-top: -10px; text-align: center;">Predicted Earnings {period_text}</p>
    </div>
    """

def create_power_graphic(power_kwh, period_text):
    """Creates an animated SVG graphic for predicted power."""
    formatted_power = f"{power_kwh:,.0f} kWh"
    return f"""
    <div class="graphic-container">
        <svg width="100" height="100" viewBox="0 0 100 100">
            <style>
                @keyframes pulse {{
                    0%, 100% {{ transform: scale(1); opacity: 1; }}
                    50% {{ transform: scale(1.1); opacity: 0.8; }}
                }}
                .bolt {{
                    animation: pulse 2s ease-in-out infinite;
                    transform-origin: center;
                }}
            </style>
            <g class="bolt">
                <path d="M50 10 L40 45 L55 45 L45 80 L60 45 L45 45 L50 10 Z" fill="#34D399"/>
            </g>
        </svg>
        <p style="font-size: 1.5rem; font-weight: 600; color: #EAECEF; margin-top: 10px;">{formatted_power}</p>
        <p style="font-size: 1rem; color: #9CA3AF; margin-top: -10px; text-align: center;">Predicted Power {period_text}</p>
    </div>
    """

def create_temperature_graphic(temperature):
    temp_min, temp_max = 0, 50; temp_norm = max(0, min(1, (temperature - temp_min) / (temp_max - temp_min))); fill_height = 80 * temp_norm; fill_y = 100 - fill_height
    if temperature < 20: color = "#60A5FA"
    elif temperature < 35: color = "#FACC15"
    else: color = "#F87171"
    return f"""<div class="graphic-container"><svg viewBox="0 0 80 140" width="80" height="140"><style>@keyframes shimmer {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.8; }} }} .liquid {{ animation: shimmer 2s ease-in-out infinite; }}</style><rect x="30" y="10" width="20" height="90" rx="10" fill="#374151" /><rect class="liquid" x="30" y="{fill_y}" width="20" height="{fill_height}" rx="10" fill="{color}" style="transition: y 0.5s ease, height 0.5s ease;" /><circle cx="40" cy="110" r="20" fill="#374151" /><circle class="liquid" cx="40" cy="110" r="16" fill="{color}" /></svg><p style="font-size: 1.5rem; font-weight: 600; color: #EAECEF; margin-top: 10px;">{temperature:.1f} °C</p><p style="font-size: 1rem; color: #9CA3AF; margin-top: -10px;">Live Temperature</p></div>"""

def create_wind_graphic(wind_speed):
    duration = max(0.5, 5 - (wind_speed / 4))
    return f"""<div class="graphic-container"><svg viewBox="0 0 150 120" width="150" height="120"><style>.wind-path {{ stroke: #9CA3AF; stroke-width: 2; fill: none; animation: dash {duration:.2f}s linear infinite; stroke-dasharray: 20, 40; }} @keyframes dash {{ to {{ stroke-dashoffset: -60; }} }}</style><path d="M 0 30 Q 75 10, 150 30" class="wind-path" style="animation-delay: 0s;" /><path d="M 0 60 Q 75 40, 150 60" class="wind-path" style="animation-delay: {duration*0.3:.2f}s;" /><path d="M 0 90 Q 75 70, 150 90" class="wind-path" style="animation-delay: {duration*0.6:.2f}s;" /></svg><p style="font-size: 1.5rem; font-weight: 600; color: #EAECEF; margin-top: 10px;">{wind_speed:.1f} m/s</p><p style="font-size: 1rem; color: #9CA3AF; margin-top: -10px;">Live Wind Speed</p></div>"""

def create_cloud_cover_graphic(cloud_cover):
    """Creates an animated and correctly scaled SVG for cloud cover."""
    x_translation = 90 - (48 * (cloud_cover / 100))
    return f"""
    <div class="graphic-container">
        <svg viewBox="0 0 120 120" width="100" height="100">
            <style>
                @keyframes bob {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-5px); }} }} 
                @keyframes glow {{ 0%, 100% {{ filter: brightness(1); }} 50% {{ filter: brightness(1.2); }} }} 
                .cloud {{ animation: bob 4s ease-in-out infinite; }} 
                .sun {{ animation: glow 4s ease-in-out infinite; }}
            </style>
            <circle class="sun" cx="60" cy="60" r="30" fill="#FACC15"/>
            <g class="cloud" transform="translate({x_translation}, 30) scale(0.9)" style="transition: transform 0.5s ease;">
                <path d="M 0 30 C -20 30, -20 10, 0 10 C 10 0, 30 0, 40 10 C 60 10, 60 30, 40 30 Z" fill="#D1D5DB"/>
            </g>
        </svg>
        <p style="font-size: 1.5rem; font-weight: 600; color: #EAECEF; margin-top: 10px;">{cloud_cover:.0f} %</p>
        <p style="font-size: 1rem; color: #9CA3AF; margin-top: -10px;">Live Cloud Cover</p>
    </div>
    """
    
# --- Main App Logic ---
st.sidebar.subheader("PV System Parameters")
panel_area = st.sidebar.number_input("Panel Area (m²)", min_value=100, value=5556, step=100)
initial_efficiency = st.sidebar.slider("Initial Efficiency (%)", min_value=10.0, max_value=25.0, value=18.0, step=0.1) / 100.0
loss_per_year = st.sidebar.slider("Annual Degradation (%)", min_value=0.0, max_value=2.0, value=1.0, step=0.05) / 100.0

st.sidebar.subheader("Financial Parameters")
price_per_kwh = st.sidebar.number_input("Price per kWh (Rs)", min_value=1.0, value=9.0, step=0.5)

# --- Load Data First to Make Options Dynamic ---
data = load_data("gru_predictions_full.csv", panel_area, initial_efficiency, loss_per_year)

if data is not None:
    min_date = data['Datetime'].min().date()
    max_date = data['Datetime'].max().date()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Period Selection")
    view_type = st.sidebar.selectbox("Select View Type", ['Daily', 'Weekly', 'Monthly', 'Yearly', 'Custom Range'], index=0, key="view_type_selector")

    # --- DYNAMICALLY CREATE SELECTORS BASED ON DATA RANGE ---
    if view_type == 'Daily': 
        selected_option = st.sidebar.date_input("Select Day", value=max_date, min_value=min_date, max_value=max_date)
    elif view_type == 'Weekly': 
        data['week_year'] = data['Datetime'].dt.strftime('%Y-W%U')
        week_options = sorted(data['week_year'].unique(), reverse=True)
        selected_option = st.sidebar.selectbox("Select Week", options=week_options)
    elif view_type == 'Monthly': 
        month_keys = sorted(data['Datetime'].dt.strftime('%Y-%m').unique(), reverse=True)
        month_options = [pd.to_datetime(key).strftime('%Y - %B') for key in month_keys]
        selected_option = st.sidebar.selectbox("Select Month", options=month_options)
    elif view_type == 'Yearly': 
        year_options = sorted(data['Datetime'].dt.year.unique(), reverse=True)
        selected_option = st.sidebar.selectbox("Select Year", options=year_options)
    else: # Custom Range
        selected_option = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # Filter data based on selection
    try:
        if view_type == 'Daily': 
            start_date = pd.to_datetime(selected_option); end_date = start_date + pd.Timedelta(days=1)
            filtered_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] < end_date)]; display_text = f"for {start_date.strftime('%B %d, %Y')}"
        elif view_type == 'Weekly': 
            filtered_data = data[data['week_year'] == selected_option]; display_text = f"for Week: {selected_option}"
        elif view_type == 'Monthly': 
            month_str = selected_option.split(' - ')[1]; year_str = selected_option.split(' - ')[0]
            month_num = pd.to_datetime(month_str, format='%B').month; selected_month_key = f"{year_str}-{month_num:02d}"
            data['month_key'] = data['Datetime'].dt.strftime('%Y-%m'); filtered_data = data[data['month_key'] == selected_month_key]; display_text = f"for {selected_option}"
        elif view_type == 'Yearly': 
            filtered_data = data[data['Datetime'].dt.year == selected_option]; display_text = f"for Year: {selected_option}"
        else: # Custom Range
            start_date = pd.to_datetime(selected_option[0]); end_date = pd.to_datetime(selected_option[1]) + pd.Timedelta(days=1)
            filtered_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] < end_date)]; display_text = f"from {start_date.strftime('%Y-%m-%d')} to {pd.to_datetime(selected_option[1]).strftime('%Y-%m-%d')}"
    except (IndexError, TypeError, StopIteration, KeyError, ValueError): 
        st.warning("Please select a valid time period."); st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis & Export")
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    csv_data = convert_df_to_csv(filtered_data)
    st.sidebar.download_button(label="Download Data as CSV", data=csv_data, file_name=f"filtered_data_{view_type}.csv", mime="text/csv")
    
    st.title("Advanced Solar Farm Digital Twin")
    st.text(f"Monitoring Facility in Tiruchirappalli, Tamil Nadu | Displaying data {display_text}")

    if not filtered_data.empty:
        # --- CALCULATIONS ---
        kpi_data = filtered_data.iloc[-1:]
        current_temp = kpi_data['Temp (C)'].values[0]
        current_wind = kpi_data['Wind Speed (m/s)'].values[0]; current_cloud = kpi_data['Cloudcover (%)'].values[0]
        
        total_predicted_kwh = filtered_data['Predicted_Power_kWh'].sum()
        predicted_earnings = total_predicted_kwh * price_per_kwh

        # --- KPI LAYOUT ---
        st.markdown("##### High-Level Summary")
        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            st.markdown(create_earnings_graphic(predicted_earnings, display_text), unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(create_power_graphic(total_predicted_kwh, display_text), unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(create_temperature_graphic(current_temp), unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(create_wind_graphic(current_wind), unsafe_allow_html=True)
        with kpi_cols[4]:
            st.markdown(create_cloud_cover_graphic(current_cloud), unsafe_allow_html=True)

        st.markdown("---")
    
        # --- VIEW NAVIGATION ---
        if 'view' not in st.session_state:
            st.session_state.view = st.query_params.get("view", "Forecasts")

        if 'view' in st.query_params and st.session_state.view != st.query_params['view']:
            st.session_state.view = st.query_params['view']

        # Centered navigation tabs
        nav_cols = st.columns(4)
        views = {
            "Forecasts": "📈 Forecasts",
            "Performance": "🛠️ System & Performance",
            "Error Analysis": "🔍 Error Analysis",
            "Explainability": "🧠 Model Explainability"
        }

        for i, (view_name, view_label) in enumerate(views.items()):
            with nav_cols[i]:
                active = "active" if st.session_state.view == view_name else ""
                st.markdown(f'<a href="?view={view_name}" target="_self" class="nav-link {active}">{view_label}</a>', unsafe_allow_html=True)


        st.markdown("---")

        # --- CONDITIONAL VIEW DISPLAY ---
        if st.session_state.view == "Forecasts":
            show_persistence = st.checkbox("Compare with Persistence Model")
            st.plotly_chart(plot_main_irradiance(filtered_data, show_persistence), use_container_width=True)
            st.plotly_chart(plot_power_production(filtered_data), use_container_width=True)
            st.plotly_chart(plot_hourly_earnings(filtered_data, price_per_kwh), use_container_width=True)

        elif st.session_state.view == "Performance":
            st.header("PV System Performance Analysis")
            unique_years_in_data = sorted(data['Year'].unique())
            efficiency_fig = plot_efficiency_degradation(initial_efficiency, loss_per_year, unique_years_in_data)
            st.plotly_chart(efficiency_fig, use_container_width=True)
            st.header("Model Performance Analysis")
            st.plotly_chart(plot_residual_error_analysis(filtered_data), use_container_width=True)

        elif st.session_state.view == "Error Analysis":
            st.header("Error Driver Analysis")
            st.markdown("""
            This chart helps identify the root cause of large forecast errors. Look for the **largest points**, as their size represents the magnitude of the error. 
            Their color (Cloud Cover) and position (Temperature and Wind Speed) will tell you the exact weather conditions that are most challenging for the model to predict.
            """)
            st.plotly_chart(plot_combined_error_driver_scatter(filtered_data), use_container_width=True)
            
            if filtered_data['Datetime'].dt.day.nunique() > 1: 
                st.plotly_chart(plot_error_heatmap(filtered_data), use_container_width=True)
        
        elif st.session_state.view == "Explainability":
            st.header("Model Input Analysis")
            if view_type == 'Daily': 
                input_data_for_plot = filtered_data
            else: 
                numeric_cols = ['Temp (C)', 'Cloudcover (%)', 'Wind Speed (m/s)']
                input_data_for_plot = filtered_data.set_index('Datetime')[numeric_cols].resample('D').mean().reset_index()
            st.plotly_chart(plot_model_inputs(input_data_for_plot), use_container_width=True)
            
            st.header("Model Explainability (XAI)")
            st.markdown("This chart shows the most important factors for the selected forecast. Longer bars mean a bigger impact. **Red bars increased** the prediction, and **blue bars decreased** it.")
            time_options = filtered_data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            if time_options:
                selected_time_str = st.selectbox("Select a Timestamp to Explain:", options=time_options, index=len(time_options) - 1)
                if selected_time_str:
                    selected_timestamp = pd.to_datetime(selected_time_str)
                    point_to_explain = filtered_data[filtered_data['Datetime'] == selected_timestamp]
                    xai_fig = plot_simplified_xai_bar_chart(point_to_explain, data)
                    st.plotly_chart(xai_fig, use_container_width=True)

    else:
        st.warning("No data available for the selected time period.")
else:
    st.warning("Please ensure the `gru_predictions_full.csv` file is present in the same directory.")

