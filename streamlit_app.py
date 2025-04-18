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

    # Filter PG data using MD > 500
    pg_data = np.where(md_data > 500, pg_data, np.nan)

    return gdf, county_gdf, shmax_gdf, earthquake_df, dp_data, pg_data

if pressure_type == "None":
    st.info("Select a pressure type from the left panel to begin visualization.")
else:
    gdf, county_gdf, shmax_gdf, earthquake_df, dp_data, pg_data = load_data()

    minx, miny, maxx, maxy = gdf.total_bounds
    x_coords = np.linspace(minx, maxx, dp_data.shape[2])
    y_coords = np.linspace(miny, maxy, dp_data.shape[1])
    aoi_polygon = gdf.unary_union

    def create_map(data_array, label, unit, norm_top, use_log):
        m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=9, dragging=False, zoom_control=False)

        folium.GeoJson(gdf, style_function=lambda x: {'color': 'green', 'weight': 2, 'dashArray': '5,5'}).add_to(m)
        folium.GeoJson(county_gdf, style_function=lambda x: {'color': 'black', 'weight': 1}).add_to(m)
        for _, row in county_gdf.iterrows():
            centroid = row.geometry.centroid
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color:black;">{row["CNTY_NM"]}</div>')
            ).add_to(m)
        for _, row in earthquake_df.iterrows():
            lat, lon, mag = row['Latitude (WGS84)'], row['Longitude (WGS84)'], row['Local Magnitude']
            if aoi_polygon.contains(Point(lon, lat)) and mag >= 3.0:
                color = 'red' if mag >= 3.5 else 'grey'
                folium.CircleMarker(location=[lat, lon], radius=mag**2, color='black', fill=True, fill_color=color, fill_opacity=0.6).add_to(m)

        layer_data = data_array[layer_selection - 1, :, :]
        if use_log:
            data_normalized = np.log1p(layer_data / norm_top) / np.log1p(1)
            threshold_min, threshold_max = 10, 1000
        else:
            threshold_min, threshold_max = 0.43, 0.5
            data_normalized = (layer_data - threshold_min) / (threshold_max - threshold_min)

        ny, nx = layer_data.shape
        for i in range(ny - 1):
            for j in range(nx - 1):
                val = layer_data[i, j]
                if threshold_min < val <= threshold_max:
                    center = Point((x_coords[j] + x_coords[j+1]) / 2, (y_coords[i] + y_coords[i+1]) / 2)
                    if aoi_polygon.contains(center):
                        color = plt.cm.jet(data_normalized[i, j])
                        folium.Rectangle(
                            bounds=[[y_coords[i], x_coords[j]], [y_coords[i+1], x_coords[j+1]]],
                            fill=True,
                            fill_color=f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}",
                            fill_opacity=0.8,
                            stroke=False
                        ).add_to(m)

        legend_html = f"""
        <div style='width: 260px; background-color: white; padding: 10px; 
             border:2px solid grey; font-size:14px; color: black;'>
        <b>Legend</b><br>
        {label} ({unit}):<br>
        <div style='background: linear-gradient(to right, blue, cyan, green, yellow, orange, red); 
             height: 15px; width: 100%; margin-bottom: 5px;'></div>
        <div style='display: flex; justify-content: space-between; font-size: 12px;'>
          <span>{threshold_min:.2f}</span>
          <span>{(threshold_min + (threshold_max - threshold_min) * 0.25):.2f}</span>
          <span>{(threshold_min + (threshold_max - threshold_min) * 0.5):.2f}</span>
          <span>{(threshold_min + (threshold_max - threshold_min) * 0.75):.2f}</span>
          <span>{threshold_max:.2f}</span>
        </div>
        <br>
        <i style='color:grey;'>●</i> Earthquake Magnitude 3.0 - 3.5<br>
        <i style='color:red;'>●</i> Earthquake Magnitude > 3.5<br>
        <span style='color:grey;'>━</span> SH_Max Orientation
        </div>
        """
        folium.Marker(
            location=[miny + 0.15, minx - 0.6],
            icon=folium.DivIcon(html=legend_html)
        ).add_to(m)
        return m

    def plot_static(data_array, label, unit, norm_top, use_log):
        layer_data = data_array[layer_selection - 1, :, :]
        if use_log:
            threshold_min, threshold_max = 10, 1000
            data_normalized = np.log1p(layer_data / norm_top) / np.log1p(1)
        else:
            threshold_min, threshold_max = 0.43, 0.5
            data_normalized = np.clip(layer_data, threshold_min, threshold_max)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(data_normalized, cmap='jet', vmin=threshold_min, vmax=threshold_max)
        fig.colorbar(cax, ax=ax, label=f"{label} ({unit})")
        ax.set_title(f"{label} - Layer {layer_selection}")
        ax.axis('off')
        st.pyplot(fig)

    if pressure_type == "Pressure Difference":
        tab1, tab2 = st.tabs(["Static Plot", "Dynamic Map"])
        with tab1:
            plot_static(dp_data, "Pressure Difference", "psi", norm_top=1000, use_log=True)
        with tab2:
            norm_top = 1000
            dp_map = create_map(dp_data, "Pressure Difference", "psi", norm_top=norm_top, use_log=True)
            st_folium(dp_map, width=1500, height=750)

    elif pressure_type == "Pressure Gradient":
        tab1, tab2 = st.tabs(["Static Plot", "Dynamic Map"])
        with tab1:
            plot_static(pg_data, "Pressure Gradient", "psi/ft", norm_top=0.5, use_log=False)
        with tab2:
            norm_top = 0.5
            pg_map = create_map(pg_data, "Pressure Gradient", "psi/ft", norm_top=norm_top, use_log=False)
            st_folium(pg_map, width=1500, height=750)
