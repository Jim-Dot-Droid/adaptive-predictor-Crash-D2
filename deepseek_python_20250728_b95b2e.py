import streamlit as st
import requests
import pandas as pd
import numpy as np

# Import matplotlib with Agg backend for Streamlit Cloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime

# Set page config
st.set_page_config(
    page_title="DegenCrash Predictor",
    page_icon="🎰",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        color: #FF4B4B;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .prediction-card {
        background-color: #0E1117;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 1em;
        color: #9EA6B7;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        color: white;
    }
    .tabs {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "https://api.degencoinflip.com/history?game=crash&limit=1000"

@st.cache_data(ttl=60, show_spinner="Fetching latest crash data...")
def fetch_crash_data():
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None

def process_data(data):
    if not data:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=False)
    
    # Extract relevant data
    df = df[['timestamp', 'multiplier']]
    df['log_multiplier'] = np.log(df['multiplier'])
    
    return df

def calculate_metrics(df, window=50):
    if df is None or len(df) < window:
        return None
    
    # Calculate metrics
    metrics = {
        "last_multiplier": df['multiplier'].iloc[0],
        "avg_last_10": df['multiplier'].head(10).mean(),
        "avg_last_50": df['multiplier'].head(50).mean(),
        "median_last_50": df['multiplier'].head(50).median(),
        "min_last_100": df['multiplier'].head(100).min(),
        "max_last_100": df['multiplier'].head(100).max(),
        "std_last_50": df['multiplier'].head(50).std(),
        "next_prediction": df['multiplier'].head(window).mean(),
        "probability_bust_2x": (df['multiplier'].head(window) < 2.0).mean(),
        "probability_10x": (df['multiplier'].head(window) > 10.0).mean(),
        "probability_50x": (df['multiplier'].head(window) > 50.0).mean(),
    }
    
    return metrics

# App header
st.markdown('<div class="header">🎰 DegenCrash Predictor</div>', unsafe_allow_html=True)
st.caption("Predicting crash multipliers for degencoinflip.com")

# Fetch and process data
data = fetch_crash_data()
df = process_data(data)

if df is None:
    st.warning("Failed to load crash data. Please try again later.")
    st.stop()

# Calculate prediction metrics
metrics = calculate_metrics(df)

# Display prediction metrics
if metrics:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">NEXT PREDICTED MULTIPLIER</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["next_prediction"]:.2f}x</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">LAST MULTIPLIER</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["last_multiplier"]:.2f}x</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">MEDIAN (LAST 50)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["median_last_50"]:.2f}x</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Multiplier History", "📊 Statistics", "🔍 Probabilities", "ℹ️ About"])

with tab1:
    st.subheader("Crash Multiplier History")
    
    # Slider for selecting number of games to show
    num_games = st.slider("Number of games to display", 10, 1000, 200, key="num_games")
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['timestamp'].head(num_games), 
            df['multiplier'].head(num_games), 
            marker='o', 
            markersize=3,
            linewidth=1,
            color='#FF4B4B')
    
    ax.set_title(f"Last {num_games} Crash Multipliers", fontsize=16)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Multiplier", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_yscale('log')
    
    st.pyplot(fig)

with tab2:
    st.subheader("Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Recent Multiplier Statistics**")
        stats_df = pd.DataFrame({
            "Period": ["Last 10", "Last 50", "Last 100"],
            "Average": [
                metrics["avg_last_10"],
                metrics["avg_last_50"],
                df['multiplier'].head(100).mean()
            ],
            "Minimum": [
                df['multiplier'].head(10).min(),
                df['multiplier'].head(50).min(),
                metrics["min_last_100"]
            ],
            "Maximum": [
                df['multiplier'].head(10).max(),
                df['multiplier'].head(50).max(),
                metrics["max_last_100"]
            ]
        })
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    with col2:
        st.markdown("**Multiplier Distribution (Last 100 Games)**")
        fig, ax = plt.subplots(figsize=(8, 6))
        bins = [1, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000]
        ax.hist(df['multiplier'].head(100), bins=bins, color='#FF4B4B', edgecolor='black')
        ax.set_xscale('log')
        ax.set_title("Multiplier Frequency Distribution")
        ax.set_xlabel("Multiplier (log scale)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

with tab3:
    st.subheader("Probability Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">PROBABILITY OF BUSTING BEFORE 2x</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["probability_bust_2x"]*100:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">PROBABILITY OF >10x MULTIPLIER</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["probability_10x"]*100:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">PROBABILITY OF >50x MULTIPLIER</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{metrics["probability_50x"]*100:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Survival probability chart
    st.markdown("**Multiplier Survival Probability**")
    multipliers = np.geomspace(1.1, 1000, 50)
    survival_probs = []
    
    for m in multipliers:
        survival_probs.append((df['multiplier'] > m).mean())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(multipliers, survival_probs, marker='o', color='#FF4B4B')
    ax.set_xscale('log')
    ax.set_title("Probability of Multiplier Exceeding Value")
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Probability")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

with tab4:
    st.subheader("About This Predictor")
    st.markdown("""
    **How This Predictor Works:**
    
    This tool analyzes historical crash data from degencoinflip.com to predict probabilities for future games:
    
    1. **Data Collection:** Fetches the last 1,000 crash results from degencoinflip's API
    2. **Statistical Analysis:** Calculates moving averages, probabilities, and distribution statistics
    3. **Prediction Model:** Uses historical averages to estimate future multiplier values
    4. **Probability Calculation:** Computes likelihood of different outcomes based on recent data
    
    **Key Metrics Explained:**
    
    - **Next Predicted Multiplier:** Average of last 50 multipliers
    - **Probability Calculations:** Based on frequency of events in historical data
    - **Survival Probability:** Chance that the multiplier will exceed a given value
    
    **Important Notes:**
    
    - Crash outcomes are statistically independent events
    - Predictions are based on historical patterns, not future guarantees
    - The house always has an edge in crash games
    - Use this tool for entertainment purposes only
    
    *Always gamble responsibly and never risk more than you can afford to lose.*
    """)
    st.markdown("---")
    st.markdown("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.markdown(f"**Games Analyzed:** {len(df)}")

# Disclaimer
st.markdown("---")
st.warning("""
**Disclaimer:** This tool provides statistical predictions based on historical data. 
Crash games are games of chance with known house edges. Predictions are not guarantees of future results. 
Gambling involves significant financial risk and may be addictive. Only play with funds you can afford to lose.
""")