# streamlit_app.py
import streamlit as st
import geopandas as gpd
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
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

# Load the data before proceeding
gdf, county_gdf, shmax_gdf, earthquake_df, dp_data, pg_data = load_data()

# Extend AOI bounds by 50%
minx, miny, maxx, maxy = gdf.total_bounds
width = maxx - minx
height = maxy - miny
ext_minx, ext_maxx = minx - 0.5 * width, maxx + 0.5 * width
ext_miny, ext_maxy = miny - 0.5 * height, maxy + 0.5 * height
x_coords = np.linspace(minx, maxx, dp_data.shape[2])
y_coords = np.linspace(miny, maxy, dp_data.shape[1])

def plot_static(data_array, label, unit, norm_top, use_log):
    layer_data = data_array[layer_selection - 1, :, :]
    if use_log:
        threshold_min, threshold_max = 10, 1000
        data_normalized = np.log1p(layer_data / norm_top) / np.log1p(1)
    else:
        threshold_min, threshold_max = 0.4, 1.0
        data_normalized = np.clip(layer_data, threshold_min, threshold_max)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot PG data
    cax = ax.imshow(
        data_normalized,
        cmap='jet',
        vmin=threshold_min, vmax=threshold_max,
        extent=[minx, maxx, miny, maxy],
        origin='lower'
    )

    # Add colorbar
    fig.colorbar(cax, ax=ax, label=f"{label} ({unit})")

    # Plot county boundaries
    county_gdf.boundary.plot(ax=ax, edgecolor='grey', linewidth=1)

    # Annotate county names
    for _, row in county_gdf.iterrows():
        if row['geometry'].centroid.is_valid:
            ax.text(row['geometry'].centroid.x, row['geometry'].centroid.y,
                    s=row['CNTY_NM'], fontsize=8, color='black', ha='center')

    # Draw AOI boundary in green
    gdf.boundary.plot(ax=ax, edgecolor='green', linewidth=2)

    # Set extended limits
    ax.set_xlim(ext_minx, ext_maxx)
    ax.set_ylim(ext_miny, ext_maxy)

    # Label axes
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{label} - Layer {layer_selection}")
    st.pyplot(fig)

if pressure_type == "Pressure Difference":
    tab1, tab2 = st.tabs(["Static Plot", "Dynamic Map"])
    with tab1:
        plot_static(dp_data, "Pressure Difference", "psi", norm_top=1000, use_log=True)
    with tab2:
        st.write("Dynamic plot is under development.")

elif pressure_type == "Pressure Gradient":
    tab1, tab2 = st.tabs(["Static Plot", "Dynamic Map"])
    with tab1:
        plot_static(pg_data, "Pressure Gradient", "psi/ft", norm_top=0.5, use_log=False)
    with tab2:
        st.write("Dynamic plot is under development.")
