# Iceland Weather Aggregator

A Streamlit web application that aggregates short-term and long-term forecasts from four Icelandic sources and visualizes the best weather regions.

## Features
- Fetches data from veður.is, belgingur.is, yr.no, and blika.is
- Normalizes temperature (°C), wind (m/s), precipitation (mm), and cloud cover (%)
- Computes a daily weather quality score per location
- Recommends top 5 camping spots based on the score
- Interactive map, tables, and charts

## Installation

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>

streamlit
pandas
requests
feedparser

pip install streamlit pandas requests
