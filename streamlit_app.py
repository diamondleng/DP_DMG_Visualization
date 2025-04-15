# streamlit_app.py
import streamlit as st
import geopandas as gpd
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from streamlit_folium import st_folium
from branca.element import Template, MacroElement

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
    PG_NPY = "PG.npy"

    gdf = gpd.read_file(AOI_SHAPEFILE).to_crs(epsg=4326)
    county_gdf = gpd.read_file(COUNTY_SHAPEFILE).to_crs(epsg=4326)
    shmax_gdf = gpd.read_file(SHMAX_SHAPEFILE).to_crs(epsg=4326)
    earthquake_df = pd.read_csv(EARTHQUAKE_CSV)
    dp_data = np.load(DP_NPY)
    pg_data = np.load(PG_NPY)

    return gdf, county_gdf, shmax_gdf, earthquake_df, dp_data, pg_data

if pressure_type == "None":
    st.info("Select a pressure type from the left panel to begin visualization.")
else:
    gdf, county_gdf, shmax_gdf, earthquake_df, dp_data, pg_data = load_data()

    minx, miny, maxx, maxy = gdf.total_bounds
    x_coords = np.linspace(minx, maxx, dp_data.shape[2])
    y_coords = np.linspace(miny, maxy, dp_data.shape[1])
    aoi_polygon = gdf.unary_union

    def create_map(data_array, label, unit, norm_top, color_max):
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
        data_log_normalized = log1p(layer_data / norm_top)/log1p(1)
        ny, nx = layer_data.shape

        for i in range(ny - 1):
            for j in range(nx - 1):
                val = layer_data[i, j]
                if val > 10:
                    center = Point((x_coords[j] + x_coords[j+1]) / 2, (y_coords[i] + y_coords[i+1]) / 2)
                    if aoi_polygon.contains(center):
                        color = plt.cm.jet(data_log_normalized[i, j])
                        folium.Rectangle(
                            bounds=[[y_coords[i], x_coords[j]], [y_coords[i+1], x_coords[j+1]]],
                            fill=True,
                            fill_color=f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}",
                            fill_opacity=0.8,
                            stroke=False
                        ).add_to(m)

        legend_html = f'''
        <div style="position: fixed; bottom: 20px; left: 20px; width: 250px; background-color: white; padding: 10px; border:2px solid grey; z-index:9999;">
        <b>Legend</b><br>
        {label} ({unit}):<br>
        <div style="background: linear-gradient(to right, blue, cyan, green, yellow, orange, red); height: 15px; width: 100%;"></div>
        0 <span style="float:right;">{color_max}</span><br>
        <i style="color:grey;">●</i> Earthquake Magnitude 3.0 - 3.5<br>
        <i style="color:red;">●</i> Earthquake Magnitude > 3.5<br>
        <span style="color:grey;">━</span> SH_Max Orientation
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        return m

    if pressure_type == "Pressure Difference":
        st.subheader("Pressure Difference (psi)")
        dp_max = np.nanmax(dp_data[layer_selection - 1])
        dp_map = create_map(dp_data, "Pressure Difference", "psi", norm_top=1000, color_max=1000)
        st_folium(dp_map, width=1000, height=750)
    elif pressure_type == "Pressure Gradient":
        st.subheader("Pressure Gradient (psi/ft)")
        pg_max = np.nanmax(pg_data[layer_selection - 1])
        pg_map = create_map(pg_data, "Pressure Gradient", "psi/ft", norm_top=0.5, color_max=0.5)
        st_folium(pg_map, width=1000, height=750)
