# streamlit_app.py
import streamlit as st
import geopandas as gpd
import folium
from folium.plugins import Fullscreen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from streamlit_folium import st_folium

# Streamlit setup
st.set_page_config(layout='wide', page_title="Pressure Difference Visualization")
st.title("Pressure Difference and Induced Seismicity Visualization")

# Formation mapping
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

# Sidebar selection
layer_selection = st.sidebar.selectbox(
    "Select Layer to Visualize:",
    options=[1, 2, 3, 4, 7, 8, 9, 10, 11, 12],
    format_func=lambda x: f"Layer {x} ({formation_dict[x]})"
)

# File paths
AOI_SHAPEFILE = "GreenAOI_Polygon.shp"
COUNTY_SHAPEFILE = "County.shp"
SHMAX_SHAPEFILE = "NA_stress_SHmax_orientations.shp"
EARTHQUAKE_CSV = "texnet_events.csv"
PRESSURE_NPY = "DP.npy"

# Load spatial data
gdf = gpd.read_file(AOI_SHAPEFILE).to_crs(epsg=4326)
county_gdf = gpd.read_file(COUNTY_SHAPEFILE).to_crs(epsg=4326)
shmax_gdf = gpd.read_file(SHMAX_SHAPEFILE).to_crs(epsg=4326)
earthquake_df = pd.read_csv(EARTHQUAKE_CSV)
dp_data = np.load(PRESSURE_NPY)
dp_data = np.where(dp_data > 1000, 1000, dp_data)

# Initialize Folium map
minx, miny, maxx, maxy = gdf.total_bounds
m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=10)
Fullscreen().add_to(m)

# AOI and County boundaries
folium.GeoJson(gdf, style_function=lambda x: {'color': 'green', 'weight': 2, 'dashArray': '5,5', 'fillOpacity': 0}).add_to(m)
folium.GeoJson(county_gdf, style_function=lambda x: {'color': 'black', 'weight': 1, 'fillOpacity': 0}).add_to(m)

# County labels
for _, row in county_gdf.iterrows():
    centroid = row.geometry.centroid
    folium.Marker(
        location=[centroid.y, centroid.x],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt;">{row["CNTY_NM"]}</div>')
    ).add_to(m)

# Earthquake visualization
aoi_polygon = gdf.unary_union
for _, row in earthquake_df.iterrows():
    lat, lon, mag = row['Latitude (WGS84)'], row['Longitude (WGS84)'], row['Local Magnitude']
    if aoi_polygon.contains(Point(lon, lat)) and mag >= 3.0:
        color = 'red' if mag >= 3.5 else 'grey'
        folium.CircleMarker(
            location=[lat, lon],
            radius=mag**2,
            color='black',
            fill=True,
            fill_color=color,
            fill_opacity=0.6
        ).add_to(m)

# Pressure visualization
layer_idx = layer_selection - 1
data = dp_data[layer_idx, :, :]
ny, nx = data.shape
x_coords = np.linspace(minx, maxx, nx)
y_coords = np.linspace(miny, maxy, ny)
data_log_norm = np.log1p(data / 1000) / np.log1p(1)

for i in range(ny - 1):
    for j in range(nx - 1):
        value = data[i, j]
        if value > 10:
            center_x, center_y = (x_coords[j] + x_coords[j+1]) / 2, (y_coords[i] + y_coords[i+1]) / 2
            if aoi_polygon.contains(Point(center_x, center_y)):
                color = plt.cm.jet(data_log_norm[i, j])
                folium.Rectangle(
                    bounds=[[y_coords[i], x_coords[j]], [y_coords[i+1], x_coords[j+1]]],
                    fill=True,
                    fill_color=f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}",
                    fill_opacity=data_log_norm[i, j],
                    stroke=False
                ).add_to(m)

# SH_Max orientations
for _, row in shmax_gdf.iterrows():
    if aoi_polygon.contains(row.geometry):
        angle = row['SHmax_or1_']
        distance = 15 * 1609.34
        end_lon = row.geometry.x + (distance / 111320) * np.sin(np.radians(angle))
        end_lat = row.geometry.y + (distance / 111320) * np.cos(np.radians(angle))
        folium.PolyLine(
            [(row.geometry.y, row.geometry.x), (end_lat, end_lon)],
            color='grey', weight=2
        ).add_to(m)

# Legend
legend_html = '''
<div style="position: fixed; bottom: 20px; left: 20px; background-color: white; padding: 10px; border:2px solid grey;">
<b>Legend</b><br>
<i style="color:grey;">●</i> Earthquake Magnitude 3.0 - 3.5<br>
<i style="color:red;">●</i> Earthquake Magnitude > 3.5<br>
<span style="color:grey;">━</span> SH_Max Orientation
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Display the map
st_folium(m, width=1200, height=800)
