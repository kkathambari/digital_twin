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

    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stMetric"] > div > div > div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #9CA3AF;
    }
    
    .graphic-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }

</style>
""", unsafe_allow_html=True)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(filepath, panel_area, initial_efficiency, loss_per_year):
    """
    Loads and preprocesses data.
    - Implements dynamic uncertainty quantification.
    - NEW: Calculates predicted and actual power production based on PV system parameters.
    """
    if not os.path.exists(filepath):
        st.sidebar.error(f"Dataset not found at '{filepath}'. Please ensure 'gru_predictions_full.csv' is in the same folder.")
        return None
    
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Actual_Global'] = (df['Actual_Direct'] + df['Actual_Diffuse']).clip(lower=0)
    df['Predicted_Global'] = (df['Predicted_Direct'] + df['Predicted_Diffuse']).clip(lower=0)
    df['Error'] = df['Actual_Global'] - df['Predicted_Global']
    
    df['Persistence_Global'] = df['Actual_Global'].shift(1)
    
    # DYNAMIC UNCERTAINTY QUANTIFICATION
    rolling_std = df['Error'].rolling(window=24, min_periods=1).std().bfill()
    df['Predicted_Upper'] = (df['Predicted_Global'] + 1.96 * rolling_std).clip(lower=0)
    df['Predicted_Lower'] = (df['Predicted_Global'] - 1.96 * rolling_std).clip(lower=0)
    
    # --- NEW: POWER PRODUCTION CALCULATION ---
    base_year = df['Datetime'].dt.year.min() # Dynamically set base year
    df['Year'] = df['Datetime'].dt.year
    df['Efficiency'] = initial_efficiency - ((df['Year'] - base_year) * loss_per_year)
    
    # Calculate power in kWh (W/m² * m² * eff) / 1000
    df['Predicted_Power_kWh'] = (df['Predicted_Global'] * panel_area * df['Efficiency']) / 1000
    df['Actual_Power_kWh'] = (df['Actual_Global'] * panel_area * df['Efficiency']) / 1000

    df['Error_Percent'] = (df['Error'] / df['Actual_Global']).replace([np.inf, -np.inf], np.nan) * 100

    return df

# --- Advanced Metrics and XAI Functions ---
def calculate_advanced_metrics(df):
    """Calculates advanced statistical metrics for model evaluation."""
    metrics = {}
    if df.empty: return { 'RMSE': 0, 'MBE': 0, 'FSS': 0 }

    y_true = df['Actual_Global']; y_pred_gru = df['Predicted_Global']
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred_gru))
    metrics['MBE'] = np.mean(y_pred_gru - y_true)

    persistence_df = df.dropna(subset=['Persistence_Global'])
    if not persistence_df.empty:
        y_true_pers = persistence_df['Actual_Global']; y_pred_pers = persistence_df['Persistence_Global']
        mse_gru = mean_squared_error(y_true_pers, persistence_df['Predicted_Global'])
        mse_pers = mean_squared_error(y_true_pers, y_pred_pers)
        metrics['FSS'] = 1 - (mse_gru / mse_pers) if mse_pers > 0 else -np.inf
    else:
        metrics['FSS'] = np.nan
    return metrics

def plot_simplified_xai_bar_chart(selected_point_df, full_data_context):
    """Creates a simple, intuitive bar chart to explain a single prediction."""
    if selected_point_df.empty:
        return go.Figure()

    point = selected_point_df.iloc[0]
    features = ['Temp (C)', 'Cloudcover (%)', 'Wind Speed (m/s)', 'Hour']
    
    # Calculate the deviation from the mean for each feature
    avg_values = full_data_context[features].mean()
    deviations = {feat: point[feat] - avg_values[feat] for feat in features}
    
    # Simplified impact calculation (you can use more complex logic if needed)
    # The sign of deviation determines the direction of impact. 
    # The absolute value can be scaled to represent magnitude.
    impacts = {k: v * 0.5 for k, v in deviations.items()} # Simple scaling
    
    impact_df = pd.DataFrame(list(impacts.items()), columns=['Feature', 'Impact'])
    impact_df['Color'] = np.where(impact_df['Impact'] > 0, '#F87171', '#60A5FA') # Red for positive, Blue for negative
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
        margin=dict(l=120) # Add left margin to prevent feature names from being cut off
    )
    return fig


# --- Plotting & Graphics Functions ---
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
        title="Generation: Digital Twin Forecast vs. Actual", template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30, 41, 59, 0.5)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title='Irradiance (W/m²)', xaxis_range=[df['Datetime'].min(), df['Datetime'].max()]
    )
    return fig

def plot_power_production(df):
    """Plots the predicted vs. actual power production."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Actual_Power_kWh'], name='Actual Power', mode='lines', line=dict(color='#34D399', width=2.5)))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['Predicted_Power_kWh'], name='Predicted Power', mode='lines', line=dict(color='#FACC15', width=2.5, dash='dash')))
    fig.update_layout(
        title="Power Production: Predicted vs. Actual Output", template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30, 41, 59, 0.5)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title='Power Output (kWh)', xaxis_range=[df['Datetime'].min(), df['Datetime'].max()]
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

def plot_actual_vs_predicted_scatter(df):
    """Creates a scatter plot of actual vs. predicted values."""
    fig = px.scatter(df, x='Actual_Global', y='Predicted_Global', 
                        title="Model Accuracy: Actual vs. Predicted Irradiance",
                        labels={'Actual_Global': 'Actual Global Irradiance (W/m²)', 'Predicted_Global': 'Predicted Global Irradiance (W/m²)'},
                        opacity=0.5)
    fig.add_shape(type='line', x0=0, y0=0, x1=df['Actual_Global'].max(), y1=df['Actual_Global'].max(), 
                    line=dict(color='#F87171', width=2, dash='dash'))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30, 41, 59, 0.5)')
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

def plot_error_density_heatmap(df, factor):
    """Creates a 2D density heatmap of error vs. a given factor."""
    if factor == 'Cloudcover (%)': title_text = 'Error Density vs. Cloud Cover'; x_axis_title = 'Cloud Cover (%)'; hover_text = 'Cloud Cover: %{x}%'
    elif factor == 'Temp (C)': title_text = 'Error Density vs. Temperature'; x_axis_title = 'Temperature (°C)'; hover_text = 'Temperature: %{x}°C'
    else: title_text = 'Error Density vs. Wind Speed'; x_axis_title = 'Wind Speed (m/s)'; hover_text = 'Wind Speed: %{x} m/s'
    fig = go.Figure(go.Histogram2d(
        x = df[factor], y = df['Error'], colorscale='Viridis', nbinsx=20, nbinsy=20,
        hovertemplate=hover_text + '<br>Error: %{y:.2f} W/m²<br>Count: %{z}<extra></extra>'
    ))
    fig.add_hline(y=0, line_width=2, line_dash="dash", line_color="#EAECEF")
    fig.update_layout(title=title_text, xaxis_title=x_axis_title, yaxis_title='Prediction Error (W/m²)', template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30, 41, 59, 0.5)')
    return fig

# --- SVG Graphics Functions ---
def create_temperature_graphic(temperature):
    temp_min, temp_max = 0, 50; temp_norm = max(0, min(1, (temperature - temp_min) / (temp_max - temp_min))); fill_height = 100 * temp_norm; fill_y = 110 - fill_height
    if temperature < 20: color = "#60A5FA"
    elif temperature < 35: color = "#FACC15"
    else: color = "#F87171"
    return f"""<div class="graphic-container"><svg viewBox="0 0 80 160" width="80" height="160"><defs><linearGradient id="grad1" x1="0%" y1="0%" x2="0%" y2="100%"><stop offset="0%" style="stop-color:{color};stop-opacity:1" /><stop offset="100%" style="stop-color:{color};stop-opacity:0.8" /></linearGradient></defs><rect x="30" y="10" width="20" height="110" rx="10" fill="#374151" /><rect x="30" y="{fill_y}" width="20" height="{fill_height}" rx="10" fill="url(#grad1)" /><circle cx="40" cy="130" r="20" fill="#374151" /><circle cx="40" cy="130" r="{10 + 10 * temp_norm}" fill="{color}" /></svg><p style="font-size: 1.5rem; font-weight: 600; color: #EAECEF; margin-top: 10px;">{temperature:.1f} °C</p><p style="font-size: 1rem; color: #9CA3AF; margin-top: -10px;">Temperature</p></div>"""
def create_wind_graphic(wind_speed):
    duration = max(0.5, 5 - (wind_speed / 4))
    return f"""<div class="graphic-container"><svg viewBox="0 0 150 120" width="150" height="120"><style>.wind-path {{ stroke: #9CA3AF; stroke-width: 2; fill: none; animation: dash {duration:.2f}s linear infinite; stroke-dasharray: 20, 40; }} @keyframes dash {{ to {{ stroke-dashoffset: -60; }} }}</style><path d="M 0 30 Q 75 10, 150 30" class="wind-path" style="animation-delay: 0s;" /><path d="M 0 60 Q 75 40, 150 60" class="wind-path" style="animation-delay: {duration*0.3:.2f}s;" /><path d="M 0 90 Q 75 70, 150 90" class="wind-path" style="animation-delay: {duration*0.6:.2f}s;" /></svg><p style="font-size: 1.5rem; font-weight: 600; color: #EAECEF; margin-top: 10px;">{wind_speed:.1f} m/s</p><p style="font-size: 1rem; color: #9CA3AF; margin-top: -10px;">Wind Speed</p></div>"""
def create_cloud_cover_graphic(cloud_cover):
    cloud_offset = 80 * (cloud_cover / 100)
    return f"""<div class="graphic-container"><svg viewBox="0 0 120 120" width="120" height="120"><defs><filter id="glow"><feGaussianBlur stdDeviation="3.5" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs><circle cx="60" cy="60" r="30" fill="#FACC15" filter="url(#glow)"/><g transform="translate({100 - cloud_offset}, 30)" style="transition: transform 0.5s ease;"><path d="M 0 30 C -20 30, -20 10, 0 10 C 10 0, 30 0, 40 10 C 60 10, 60 30, 40 30 Z" fill="#D1D5DB"/></g></svg><p style="font-size: 1.5rem; font-weight: 600; color: #EAECEF; margin-top: 10px;">{cloud_cover:.0f} %</p><p style="font-size: 1rem; color: #9CA3AF; margin-top: -10px;">Cloud Cover</p></div>"""
    
# --- Main App Logic ---
st.sidebar.title("🛰️ Control Panel")

# --- PV System Parameters (Defined before loading data) ---
st.sidebar.markdown("---")
st.sidebar.subheader("PV System Parameters")
panel_area = st.sidebar.number_input("Panel Area (m²)", min_value=100, value=5556, step=100)
initial_efficiency = st.sidebar.slider("Initial Efficiency (%)", min_value=10.0, max_value=25.0, value=18.0, step=0.1) / 100.0
loss_per_year = st.sidebar.slider("Annual Degradation (%)", min_value=0.0, max_value=2.0, value=1.0, step=0.05) / 100.0

# --- Load Data First to Make Options Dynamic ---
data = load_data("gru_predictions_full.csv", panel_area, initial_efficiency, loss_per_year)

if data is not None:
    min_date = data['Datetime'].min().date()
    max_date = data['Datetime'].max().date()

    st.sidebar.markdown("---")
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
            start_date = pd.to_datetime(selected_option)
            end_date = start_date + pd.Timedelta(days=1)
            filtered_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] < end_date)]
            display_text = f"Displaying data for {start_date.strftime('%B %d, %Y')}"
        
        elif view_type == 'Weekly': 
            filtered_data = data[data['week_year'] == selected_option]
            display_text = f"Displaying data for Week: {selected_option}"

        elif view_type == 'Monthly': 
            month_str = selected_option.split(' - ')[1]
            year_str = selected_option.split(' - ')[0]
            month_num = pd.to_datetime(month_str, format='%B').month
            selected_month_key = f"{year_str}-{month_num:02d}"
            data['month_key'] = data['Datetime'].dt.strftime('%Y-%m')
            filtered_data = data[data['month_key'] == selected_month_key]
            display_text = f"Displaying data for {selected_option}"
        
        elif view_type == 'Yearly': 
            filtered_data = data[data['Datetime'].dt.year == selected_option]
            display_text = f"Displaying data for Year: {selected_option}"
        
        else: # Custom Range
            start_date = pd.to_datetime(selected_option[0])
            end_date = pd.to_datetime(selected_option[1]) + pd.Timedelta(days=1)
            filtered_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] < end_date)]
            display_text = f"Data from {start_date.strftime('%Y-%m-%d')} to {pd.to_datetime(selected_option[1]).strftime('%Y-%m-%d')}"
    
    except (IndexError, TypeError, StopIteration, KeyError, ValueError): 
        st.warning("Please select a valid time period.")
        st.stop()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis & Export Options")
    show_persistence = st.sidebar.checkbox("Compare with Persistence Model")
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(filtered_data)
    st.sidebar.download_button(label="Download Data as CSV", data=csv_data, file_name=f"filtered_data_{view_type}.csv", mime="text/csv")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("About the Model")
    st.sidebar.markdown("""- **Architecture:** 2 GRU Layers (128 & 64 units) with Dropout (0.2). - **Optimizer:** Adam - **Loss Function:** MSE - **Training Data:** 10 years of hourly historical weather data for Tiruchirappalli (2015-2024).""")
    st.sidebar.markdown("""<div style="font-size: 0.8rem; color: #9CA3AF; text-align: center; margin-top: 20px;">To save a chart, hover over it and click the camera icon that appears in the top right.</div>""", unsafe_allow_html=True)

    st.title("Advanced Solar Farm Digital Twin")
    st.text(f"Monitoring Facility in Tiruchirappalli, Tamil Nadu | {display_text}")

    if not filtered_data.empty:
        kpi_data = filtered_data.iloc[-1:]
        current_ghi = kpi_data['Actual_Global'].values[0]
        current_prediction = kpi_data['Predicted_Global'].values[0]
        current_temp = kpi_data['Temp (C)'].values[0]
        current_wind = kpi_data['Wind Speed (m/s)'].values[0]
        current_cloud = kpi_data['Cloudcover (%)'].values[0]
        current_predicted_power = kpi_data['Predicted_Power_kWh'].values[0]
        
        mae_gru = mean_absolute_error(filtered_data['Actual_Global'], filtered_data['Predicted_Global'])
        r2_gru = r2_score(filtered_data['Actual_Global'], filtered_data['Predicted_Global'])
        advanced_metrics = calculate_advanced_metrics(filtered_data)
        
        col1, col2 = st.columns([2, 1]) 

        with col1:
            st.markdown("##### Performance Metrics")
            
            row1_cols = st.columns(2)
            with row1_cols[0]:
                st.metric(label="Live Predicted Power (kWh)", value=f"{current_predicted_power:.2f}", help="Estimated power for the latest hour based on predicted irradiance and PV parameters.")
            with row1_cols[1]:
                st.metric(label="Live GHI (W/m²)", value=f"{current_ghi:.2f}", delta=f"Pred: {current_prediction:.2f} W/m²")

            row2_cols = st.columns(2)
            with row2_cols[0]:
                st.metric(label="GRU Model R²", value=f"{r2_gru:.3f}")
            with row2_cols[1]:
                st.metric(label="GRU Model MAE", value=f"{mae_gru:.2f} W/m²", delta_color="inverse")

            row3_cols = st.columns(2)
            with row3_cols[0]:
                st.metric(label="GRU Model RMSE", value=f"{advanced_metrics['RMSE']:.2f} W/m²", delta_color="inverse")
            with row3_cols[1]:
                st.metric(label="GRU Model MBE", value=f"{advanced_metrics['MBE']:.2f} W/m²", help="Mean Bias Error. Negative values indicate under-prediction on average.")

            row4_cols = st.columns(2)
            with row4_cols[0]:
                st.metric(label="Forecast Skill (vs Persistence)", value=f"{advanced_metrics['FSS']:.3f}", help="1 - (MSE_GRU / MSE_Persistence). Positive values indicate the model is better than persistence.")

        with col2:
            st.markdown("##### Live Environmental Conditions")
            st.markdown(create_temperature_graphic(current_temp), unsafe_allow_html=True)
            st.markdown(create_wind_graphic(current_wind), unsafe_allow_html=True)
            st.markdown(create_cloud_cover_graphic(current_cloud), unsafe_allow_html=True)
        st.markdown("---")
    
        main_fig = plot_main_irradiance(filtered_data, show_persistence)
        st.plotly_chart(main_fig, use_container_width=True)

        st.markdown("---")
        st.header("Predicted Power Production")
        st.markdown("This section translates the irradiance forecast into an estimated power output (in kWh) based on the PV system parameters set in the sidebar.")
        st.plotly_chart(plot_power_production(filtered_data), use_container_width=True)

        st.header("PV System Performance Analysis")
        st.markdown("Visualizing the calculated year-on-year degradation of the solar panel efficiency based on the parameters set in the sidebar.")
        unique_years_in_data = sorted(data['Year'].unique())
        efficiency_fig = plot_efficiency_degradation(initial_efficiency, loss_per_year, unique_years_in_data)
        st.plotly_chart(efficiency_fig, use_container_width=True)


        st.plotly_chart(plot_residual_error_analysis(filtered_data), use_container_width=True)
        st.plotly_chart(plot_actual_vs_predicted_scatter(filtered_data), use_container_width=True)
        st.markdown("---")
        st.header("Model Input Analysis")
        st.markdown("Visualizing the key environmental factors that drive the forecast.")
        if view_type == 'Daily': 
            input_data_for_plot = filtered_data
        else: 
            numeric_cols = ['Temp (C)', 'Cloudcover (%)', 'Wind Speed (m/s)']
            input_data_for_plot = filtered_data.set_index('Datetime')[numeric_cols].resample('D').mean().reset_index()
        st.plotly_chart(plot_model_inputs(input_data_for_plot), use_container_width=True)
        st.markdown("---")
        st.header("Error Driver Analysis")
        st.markdown("Investigating *why* the model makes errors under different conditions using advanced visualizations.")
        if filtered_data['Datetime'].dt.day.nunique() > 1: 
            st.plotly_chart(plot_error_heatmap(filtered_data), use_container_width=True)
        st.markdown("##### Error Density Analysis by Environmental Factor")
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        with diag_col1: 
            st.plotly_chart(plot_error_density_heatmap(filtered_data, 'Cloudcover (%)'), use_container_width=True)
        with diag_col2: 
            st.plotly_chart(plot_error_density_heatmap(filtered_data, 'Temp (C)'), use_container_width=True)
        with diag_col3: 
            st.plotly_chart(plot_error_density_heatmap(filtered_data, 'Wind Speed (m/s)'), use_container_width=True)
        st.markdown("---")
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

