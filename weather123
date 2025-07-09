import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
import feedparser

# ----------------------
# Configuration
# ----------------------
# Placeholder endpoints — replace with actual API/RSS URLs if needed
ENDPOINTS = {
    "vedur": "https://api.vedur.is/weather/forecasts",            # Replace with real veður.is API
    "belgingur": "https://belgingur.is/api/v2/weather",          # Replace with real belgingur.is API
    "yr": "https://api.met.no/weatherapi/locationforecast/2.0/compact",
    "blika": "https://blika.is/rss/forecast.xml",               # Replace with real blika.is RSS
}

# Ideal comfort temperature (°C)
IDEAL_TEMP = 15.0
MAX_WIND = 10.0       # m/s
MAX_PRECIP = 10.0     # mm/day

# List of camping spots with approximate coordinates
CAMPING_SPOTS = [
    {"name": "Þingvellir",   "lat": 64.255, "lon": -21.13},
    {"name": "Vík í Mýrdal", "lat": 63.417, "lon": -19.006},
    {"name": "Akureyri",     "lat": 65.683, "lon": -18.100},
    {"name": "Húsavík",      "lat": 66.041, "lon": -17.338},
    {"name": "Skaftafell",   "lat": 64.016, "lon": -16.966},
    {"name": "Egilsstaðir",  "lat": 65.267, "lon": -14.400},
    {"name": "Kirkjubæjarklaustur", "lat": 63.813, "lon": -18.048},
    {"name": "Stykkishólmur","lat": 65.073, "lon": -22.725},
    {"name": "Reykjanesbær","lat": 63.998, "lon": -22.555},
    {"name": "Blönduós",      "lat": 65.683, "lon": -20.333},
]

# ----------------------
# Data Fetching & Caching
# ----------------------
@st.cache_data(ttl=1800)
def fetch_json(url, params=None, headers=None):
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Error fetching data from {url}: {e}")
        return None

@st.cache_data(ttl=1800)
def fetch_rss(url):
    try:
        feed = feedparser.parse(url)
        return feed.entries
    except Exception as e:
        st.error(f"Error parsing RSS from {url}: {e}")
        return []

# Example fetch functions — adapt parsing logic per API
@lru_cache(maxsize=None)
def fetch_vedur_data(lat, lon):
    # TODO: Update endpoint & params per veður.is API spec
    url = ENDPOINTS['vedur']
    data = fetch_json(url, params={'lat': lat, 'lon': lon})
    # Parse data into DataFrame with required fields
    # ... implement parsing ...
    return pd.DataFrame()  # stub

@lru_cache(maxsize=None)
def fetch_belgingur_data(lat, lon):
    url = ENDPOINTS['belgingur']
    data = fetch_json(url, params={'lat': lat, 'lon': lon})
    # Parse data into DataFrame
    return pd.DataFrame()  # stub

@lru_cache(maxsize=None)
def fetch_yr_data(lat, lon):
    url = ENDPOINTS['yr']
    headers = {'User-Agent': 'StreamlitWeatherApp/1.0 your_email@example.com'}
    data = fetch_json(url, params={'lat': lat, 'lon': lon}, headers=headers)
    # Parse JSON into DataFrame
    return pd.DataFrame()  # stub

@lru_cache(maxsize=None)
def fetch_blika_data(lat, lon):
    url = ENDPOINTS['blika']
    entries = fetch_rss(url)
    # Parse RSS entries into DataFrame
    return pd.DataFrame()  # stub

# ----------------------
# Data Normalization & Aggregation
# ----------------------
def normalize_df(df, source):
    # Ensure columns: ['datetime', 'temp_c', 'wind_m_s', 'precip_mm', 'cloud_pct']
    df = df.copy()
    df['source'] = source
    return df

@st.cache_data(ttl=1800)
def load_all_data(lat, lon, start_date, end_date):
    # Fetch from each source
    dfs = []
    for fn, fetcher in [('vedur', fetch_vedur_data),
                        ('belgingur', fetch_belgingur_data),
                        ('yr', fetch_yr_data),
                        ('blika', fetch_blika_data)]:
        df = fetcher(lat, lon)
        if df is not None and not df.empty:
            dfs.append(normalize_df(df, fn))
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    # Filter by date range
    combined = combined[(combined['datetime'] >= start_date) & (combined['datetime'] <= end_date)]
    return combined

# ----------------------
# Scoring Logic
# ----------------------
def compute_quality_score(df):
    # Group by date & source then average metrics if needed
    daily = df.groupby(df['datetime'].dt.date).mean().reset_index()
    scores = []
    for _, row in daily.iterrows():
        # Temperature comfort score
        temp_score = max(0, 1 - abs(row['temp_c'] - IDEAL_TEMP) / IDEAL_TEMP)
        # Precipitation score (assuming daily mm)
        rain_score = max(0, 1 - row['precip_mm'] / MAX_PRECIP)
        # Wind score
        wind_score = max(0, 1 - row['wind_m_s'] / MAX_WIND)
        # Cloud cover score
        cloud_score = max(0, 1 - row['cloud_pct'] / 100)
        # Weighted
        score = (0.4 * temp_score + 0.3 * rain_score + 0.2 * wind_score + 0.1 * cloud_score)
        scores.append({'date': row['datetime'], 'score': score})
    return pd.DataFrame(scores)

def rank_camping_spots(scores, lat, lon):
    # For simplicity, apply same score to all spots; in reality, fetch each spot
    spots = pd.DataFrame(CAMPING_SPOTS)
    spots['score'] = scores['score'].mean()
    top5 = spots.nlargest(5, 'score')
    return top5

# ----------------------
# Streamlit App
# ----------------------
def main():
    st.title("Iceland Weather Aggregator")

    # Sidebar controls
    st.sidebar.header("Settings")
    today = datetime.utcnow().date()
    start_date, end_date = st.sidebar.date_input("Date range",
                                               [today, today + timedelta(days=7)])
    lat = st.sidebar.number_input("Latitude", min_value=63.0, max_value=67.0, value=64.0)
    lon = st.sidebar.number_input("Longitude", min_value=-25.0, max_value=-13.0, value=-19.0)

    # Load & process data
    with st.spinner("Fetching data..."):
        raw = load_all_data(lat, lon, pd.to_datetime(start_date), pd.to_datetime(end_date))
    if raw.empty:
        st.warning("No data available for this location/time range.")
        return

    st.subheader("Combined Forecast Data")
    st.dataframe(raw)

    # Compute scores
    scores = compute_quality_score(raw)
    st.subheader("Daily Weather Quality Scores")
    st.line_chart(scores.set_index('date')['score'])

    # Top camping spots
    top5 = rank_camping_spots(scores, lat, lon)
    st.subheader("Top 5 Recommended Camping Spots")
    st.map(top5.rename(columns={'lat':'latitude','lon':'longitude'}))
    st.dataframe(top5)

if __name__ == '__main__':
    main()
