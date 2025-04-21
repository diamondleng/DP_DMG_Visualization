# streamlit_app.py
import streamlit as st
import geopandas as gpd
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from streamlit_folium import st_folium

# Streamlit setup
st.set_page_config(layout='wide', page_title="Pressure Visualization Dashboard")
st.title("Pressure Map with Induced Seismicity")

# Sidebar selection
pressure_type = st.sidebar.radio("Select Pressure Type:", ["None", "Pressure Difference", "Pressure Gradient"])
formation_dict = {
    1: "Bell Canyon",
    2: "Bell Canyon",
    3: "Bell Canyon",
    4: "Bell Canyon",
    7: "Cherry Canyon",
    8: "Cherry Canyon",
    9: "Cherry Canyon",
    10: "Brushy Canyon",
    11: "Brushy Canyon",
    12: "Brushy Canyon"
}
layer_selection = st.sidebar.selectbox(
    "Select Layer to Visualize:",
    options=[1, 2, 3, 4, 7, 8, 9, 10, 11, 12],
    format_func=lambda x: f"Layer {x} ({formation_dict[x]})"
)

@st.cache_data
def load_data():
    AOI_SHAPEFILE = "GreenAOI_Polygon.shp"
    COUNTY_SHAPEFILE = "County.shp"
    SHMAX_SHAPEFILE = "NA_stress_SHmax_orientations.shp"
    EARTHQUAKE_CSV = "texnet_events.csv"
    DP_NPY = "DP.npy"
    PG_NPY = "New_PG_4.19.npy"
    MD_NPY = "New_MD_4.19.npy"

    gdf = gpd.read_file(AOI_SHAPEFILE).to_crs(epsg=4326)
    county_gdf = gpd.read_file(COUNTY_SHAPEFILE).to_crs(epsg=4326)
    shmax_gdf = gpd.read_file(SHMAX_SHAPEFILE).to_crs(epsg=4326)
    earthquake_df = pd.read_csv(EARTHQUAKE_CSV)
    dp_data = np.load(DP_NPY)
    dp_data = np.where(dp_data > 1000, 1000, dp_data)
    pg_data = np.load(PG_NPY)
    md_data = np.load(MD_NPY)

    # Replace MD filter with outlier exclusion for PG
    pg_valid = pg_data[(pg_data > 0) & ~np.isnan(pg_data)]
    q_low, q_high = np.percentile(pg_valid, [1, 99])
    pg_data = np.where((pg_data >= q_low) & (pg_data <= q_high), pg_data, np.nan)

    return gdf, county_gdf, shmax_gdf, earthquake_df, dp_data, pg_data

# [rest of the code remains unchanged from canvas]
