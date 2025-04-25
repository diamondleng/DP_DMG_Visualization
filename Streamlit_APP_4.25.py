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
pressure_type = st.sidebar.radio("Select Pressure Type:", ["None", "Pressure Difference", "Pressure Gradient", "PG (Constant Compressibility Test)"])

compressibility_files = {
    "0.75e-6": np.load("PG_0.75.npy"),
    "1.0e-6": np.load("PG_1.npy"),
    "2.5e-6": np.load("PG_2.5.npy"),
    "5.0e-6": np.load("PG_5.npy"),
    "7.5e-6": np.load("PG_7.5.npy")
}
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

    # Use MD > 500 as a mask before applying outlier exclusion
    pg_data = np.where(md_data > 1300, pg_data, np.nan)
    # Use MD > 500 as a mask before applying outlier exclusion
    pg_valid = pg_data[(pg_data > 0) & ~np.isnan(pg_data)].astype(float)
    q_low, q_high = np.percentile(pg_valid, [0, 100])
    pg_data = np.where((pg_data >= q_low) & (pg_data <= q_high), pg_data, 0.43)

    return gdf, county_gdf, shmax_gdf, earthquake_df, dp_data, pg_data, md_data

# Load the data before proceeding
gdf, county_gdf, shmax_gdf, earthquake_df, dp_data, pg_data, md_data = load_data()

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
        data_normalized = np.log1p(layer_data / norm_top) / np.log1p(1)
        threshold_min = np.log1p(10 / norm_top) / np.log1p(1)
        threshold_max = np.log1p(1000 / norm_top) / np.log1p(1)
    else:
        threshold_min, threshold_max = 0.4, 1.0
        data_normalized = np.clip(layer_data, threshold_min, threshold_max)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Mask out values outside AOI geometry
    ny, nx = layer_data.shape
    masked_data = np.full_like(layer_data, np.nan, dtype=float)
    for i in range(ny):
        for j in range(nx):
            lon = x_coords[j]
            lat = y_coords[i]
            point = Point(lon, lat)
            if gdf.unary_union.contains(point) and (pressure_type != "Pressure Gradient" or md_data[layer_selection - 1, i, j] >= 1300):
                masked_data[i, j] = data_normalized[i, j]

    # Plot masked PG data
    cax = ax.imshow(
        masked_data,
        cmap='jet',
        vmin=threshold_min, vmax=threshold_max,
        extent=[minx, maxx, miny, maxy],
        origin='lower'
    )

    # Add colorbar with correct tick labels
    if use_log:
        cbar = fig.colorbar(cax, ax=ax)
        tick_vals = [np.log1p(v / norm_top) / np.log1p(1) for v in [0, 250, 500, 750, 1000]]
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(['0', '250', '500', '750', '1000'])
        cbar.set_label(f"{label} ({unit})")
    else:
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_ticks([0.4, 0.55, 0.7, 0.85, 1.0])
        cbar.set_label(f"{label} ({unit})")

    # Plot county boundaries
    county_gdf[county_gdf.intersects(gdf.unary_union.buffer(width * 0.5))].boundary.plot(ax=ax, edgecolor='grey', linewidth=1)

    # Annotate county names
    for _, row in county_gdf[county_gdf.intersects(gdf.unary_union.buffer(width * 0.5))].iterrows():
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
    tab1, tab2 = st.tabs(["Static Plot", "Dynamic Map (Slow Response)"])
    with tab1:
        plot_static(dp_data, "Pressure Difference", "psi", norm_top=1000, use_log=True)
    with tab2:
        m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=9)

        folium.GeoJson(gdf, style_function=lambda x: {'color': 'green', 'weight': 2, 'fillOpacity': 0}).add_to(m)
        folium.GeoJson(county_gdf, style_function=lambda x: {'color': 'gray', 'weight': 1, 'fillOpacity': 0}).add_to(m)

        layer_data = dp_data[layer_selection - 1, :, :]
        norm_top = 1000
        data_normalized = np.log1p(layer_data / norm_top) / np.log1p(1)
        cmap_range = (np.log1p(10 / norm_top) / np.log1p(1), np.log1p(1000 / norm_top) / np.log1p(1))

        ny, nx = layer_data.shape
        for i in range(ny - 1):
            for j in range(nx - 1):
                val = layer_data[i, j]
                if not np.isnan(val):
                    lon1, lon2 = x_coords[j], x_coords[j + 1]
                    lat1, lat2 = y_coords[i], y_coords[i + 1]
                    point = Point((lon1 + lon2) / 2, (lat1 + lat2) / 2)
                    if gdf.unary_union.contains(point):
                        norm_val = np.log1p(val / norm_top) / np.log1p(1)
                        color = plt.cm.jet((norm_val - cmap_range[0]) / (cmap_range[1] - cmap_range[0]))
                        color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                        folium.Rectangle(
                            bounds=[[lat1, lon1], [lat2, lon2]],
                            fill=True,
                            fill_color=color_hex,
                            fill_opacity=0.7,
                            stroke=False
                        ).add_to(m)

        for _, row in earthquake_df.iterrows():
            lon, lat, mag = row['Longitude (WGS84)'], row['Latitude (WGS84)'], row['Local Magnitude']
            if gdf.unary_union.contains(Point(lon, lat)) and mag >= 3.0:
                color = 'grey' if mag < 3.5 else 'red'
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=mag**2,
                    color='black',
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.5
                ).add_to(m)

        for _, row in shmax_gdf.iterrows():
            if gdf.unary_union.contains(row.geometry):
                start_point = row.geometry
                angle = row['SHmax_or1_']
                dist = 15 * 1609.34
                end_lon = start_point.x + (dist / 111320) * np.sin(np.deg2rad(angle))
                end_lat = start_point.y + (dist / 111320) * np.cos(np.deg2rad(angle))
                folium.PolyLine(
                    locations=[(start_point.y, start_point.x), (end_lat, end_lon)],
                    color='grey', weight=2
                ).add_to(m)

        label = 'Pressure Difference (psi)'
        scale = 'background: linear-gradient(to right, blue, cyan, green, yellow, orange, red);'
        ticks = ['0', '250', '500', '750', '1000']

        cols = st.columns([4, 1])
        with cols[0]:
            st_folium(m, width=1000, height=750)
        with cols[1]:
            st.markdow
            st.markdown("<div style='display: flex; justify-content: space-between;'>" + ''.join([f"<span>{t}</span>" for t in ticks]) + "</div>", unsafe_allow_html=True)
            st.markdown("<br><span style='color:grey;'>● Earthquake Magnitude 3.0 - 3.5</span>", unsafe_allow_html=True)
            st.markdown("<span style='color:red;'>● Earthquake Magnitude > 3.5</span>", unsafe_allow_html=True)
            st.markdown("<span style='color:grey;'>━ SH_Max Orientation</span>", unsafe_allow_html=True)
            

elif pressure_type == "PG (Constant Compressibility Test)":
    comp_choice = st.sidebar.select_slider(
        "Select Compressibility (1/psi):",
        options=["0.75e-6", "1.0e-6", "2.5e-6", "5.0e-6", "7.5e-6"]
    )
    pg_data = compressibility_files[comp_choice]
    st.markdown(f"**Note:** Showing results for Compressibility = {comp_choice} 1/psi")
    tab1, tab2 = st.tabs(["Static Plot", "Dynamic Map (Slow Response)"])
    with tab1:
        plot_static(pg_data, "Pressure Gradient", "psi/ft", norm_top=0.5, use_log=False)
    with tab2:
        m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=9)
        folium.GeoJson(gdf, style_function=lambda x: {'color': 'green', 'weight': 2, 'fillOpacity': 0}).add_to(m)
        folium.GeoJson(county_gdf, style_function=lambda x: {'color': 'gray', 'weight': 1, 'fillOpacity': 0}).add_to(m)

        layer_data = pg_data[layer_selection - 1, :, :]
        norm_top = 1.0
        data_normalized = np.clip(layer_data, 0.4, norm_top)
        cmap_range = (0.4, 1.0)

        ny, nx = layer_data.shape
        for i in range(ny - 1):
            for j in range(nx - 1):
                val = layer_data[i, j]
                if not np.isnan(val):
                    lon1, lon2 = x_coords[j], x_coords[j + 1]
                    lat1, lat2 = y_coords[i], y_coords[i + 1]
                    point = Point((lon1 + lon2) / 2, (lat1 + lat2) / 2)
                    if gdf.unary_union.contains(point):
                        norm_val = data_normalized[i, j]
                        color = plt.cm.jet((norm_val - cmap_range[0]) / (cmap_range[1] - cmap_range[0]))
                        color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                        folium.Rectangle(
                            bounds=[[lat1, lon1], [lat2, lon2]],
                            fill=True,
                            fill_color=color_hex,
                            fill_opacity=0.7,
                            stroke=False
                        ).add_to(m)

        for _, row in earthquake_df.iterrows():
            lon, lat, mag = row['Longitude (WGS84)'], row['Latitude (WGS84)'], row['Local Magnitude']
            if gdf.unary_union.contains(Point(lon, lat)) and mag >= 3.0:
                color = 'grey' if mag < 3.5 else 'red'
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=mag**2,
                    color='black',
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.5
                ).add_to(m)

        for _, row in shmax_gdf.iterrows():
            if gdf.unary_union.contains(row.geometry):
                start_point = row.geometry
                angle = row['SHmax_or1_']
                dist = 15 * 1609.34
                end_lon = start_point.x + (dist / 111320) * np.sin(np.deg2rad(angle))
                end_lat = start_point.y + (dist / 111320) * np.cos(np.deg2rad(angle))
                folium.PolyLine(
                    locations=[(start_point.y, start_point.x), (end_lat, end_lon)],
                    color='grey', weight=2
                ).add_to(m)

        label = 'Pressure Gradient (psi/ft)'
        scale = 'background: linear-gradient(to right, blue, cyan, green, yellow, orange, red);'
        ticks = ['0.4', '0.55', '0.7', '0.85', '1.0']

        cols = st.columns([4, 1])
        with cols[0]:
            st_folium(m, width=1000, height=750)
        with cols[1]:
            st.markdown("### Legend")
            st.markdown(f"**{label}**")
            st.markdown(f"<div style='{scale} height: 15px; width: 100%; margin-bottom: 5px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='display: flex; justify-content: space-between;'>" + ''.join([f"<span>{t}</span>" for t in ticks]) + "</div>", unsafe_allow_html=True)
            st.markdown("<br><span style='color:grey;'>● Earthquake Magnitude 3.0 - 3.5</span>", unsafe_allow_html=True)
            st.markdown("<span style='color:red;'>● Earthquake Magnitude > 3.5</span>", unsafe_allow_html=True)
            st.markdown("<span style='color:grey;'>━ SH_Max Orientation</span>", unsafe_allow_html=True)
            st.markdown("**Note:** This plot excludes the outcrop area in the west of the Delaware Mountain Group (DMG).")
            st.markdown(f"**{label}**")
            st.markdown(f"<div style='{scale} height: 15px; width: 100%; margin-bottom: 5px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='display: flex; justify-content: space-between;'>" + ''.join([f"<span>{t}</span>" for t in ticks]) + "</div>", unsafe_allow_html=True)
            st.markdown("<br><span style='color:grey;'>● Earthquake Magnitude 3.0 - 3.5</span>", unsafe_allow_html=True)
            st.markdown("<span style='color:red;'>● Earthquake Magnitude > 3.5</span>", unsafe_allow_html=True)
            st.markdown("<span style='color:grey;'>━ SH_Max Orientation</span>", unsafe_allow_html=True)

elif pressure_type == "Pressure Gradient":
    st.markdown("**Note:** This plot excludes the outcrop area in the west of the Delaware Mountain Group (DMG).")
    tab1, tab2 = st.tabs(["Static Plot", "Dynamic Map (Slow Response)"])
    with tab1:
        plot_static(pg_data, "Pressure Gradient", "psi/ft", norm_top=0.5, use_log=False)
    with tab2:
        m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=9)
        folium.GeoJson(gdf, style_function=lambda x: {'color': 'green', 'weight': 2, 'fillOpacity': 0}).add_to(m)
        folium.GeoJson(county_gdf, style_function=lambda x: {'color': 'gray', 'weight': 1, 'fillOpacity': 0}).add_to(m)

        layer_data = pg_data[layer_selection - 1, :, :]
        norm_top = 1.0
        data_normalized = np.clip(layer_data, 0.4, norm_top)
        cmap_range = (0.4, 1.0)

        ny, nx = layer_data.shape
        for i in range(ny - 1):
            for j in range(nx - 1):
                val = layer_data[i, j]
                if not np.isnan(val) and md_data[layer_selection - 1, i, j] >= 1000:
                    lon1, lon2 = x_coords[j], x_coords[j + 1]
                    lat1, lat2 = y_coords[i], y_coords[i + 1]
                    point = Point((lon1 + lon2) / 2, (lat1 + lat2) / 2)
                    if gdf.unary_union.contains(point):
                        norm_val = data_normalized[i, j]
                        color = plt.cm.jet((norm_val - cmap_range[0]) / (cmap_range[1] - cmap_range[0]))
                        color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
                        folium.Rectangle(
                            bounds=[[lat1, lon1], [lat2, lon2]],
                            fill=True,
                            fill_color=color_hex,
                            fill_opacity=0.7,
                            stroke=False
                        ).add_to(m)

        for _, row in earthquake_df.iterrows():
            lon, lat, mag = row['Longitude (WGS84)'], row['Latitude (WGS84)'], row['Local Magnitude']
            if gdf.unary_union.contains(Point(lon, lat)) and mag >= 3.0:
                color = 'grey' if mag < 3.5 else 'red'
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=mag**2,
                    color='black',
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.5
                ).add_to(m)

        label = 'Pressure Gradient (psi/ft)'
        scale = 'background: linear-gradient(to right, blue, cyan, green, yellow, orange, red);'
        ticks = ['0.4', '0.55', '0.7', '0.85', '1.0']
        for _, row in shmax_gdf.iterrows():
            if gdf.unary_union.contains(row.geometry):
                start_point = row.geometry
                angle = row['SHmax_or1_']
                dist = 15 * 1609.34
                end_lon = start_point.x + (dist / 111320) * np.sin(np.deg2rad(angle))
                end_lat = start_point.y + (dist / 111320) * np.cos(np.deg2rad(angle))
                folium.PolyLine(
                    locations=[(start_point.y, start_point.x), (end_lat, end_lon)],
                    color='grey', weight=2
                ).add_to(m)

        legend_html = f"""
        <div style='position: absolute; left: 20px; bottom: 20px; width: 270px;
                     background-color: white; padding: 10px; border:2px solid grey; z-index:9999;'>
        <b>Legend</b><br>
        {label}:<br>
        <div style='{scale} height: 15px; width: 100%; margin-bottom: 5px;'></div>
        <div style='display: flex; justify-content: space-between;'>
        {''.join([f'<span>{t}</span>' for t in ticks])}
        </div><br>
        <i style='color:grey;'>●</i> Earthquake Magnitude 3.0 - 3.5<br>
        <i style='color:red;'>●</i> Earthquake Magnitude > 3.5<br>
        <span style='color:grey;'>━</span> SH_Max Orientation
        </div>
        """
        from branca.element import Template, MacroElement
        legend = MacroElement()
        legend._template = Template(legend_html)
        m.get_root().add_child(legend)

        cols = st.columns([4, 1])
        with cols[0]:
            st_folium(m, width=1000, height=750)
        with cols[1]:
            st.markdown("### Legend")
            st.markdown(f"**{label}**")
            st.markdown(f"<div style='{scale} height: 15px; width: 100%; margin-bottom: 5px;'></div>", unsafe_allow_html=True)
            st.markdown("<div style='display: flex; justify-content: space-between;'>" + ''.join([f"<span>{t}</span>" for t in ticks]) + "</div>", unsafe_allow_html=True)
            st.markdown("<br><span style='color:grey;'>● Earthquake Magnitude 3.0 - 3.5</span>", unsafe_allow_html=True)
            st.markdown("<span style='color:red;'>● Earthquake Magnitude > 3.5</span>", unsafe_allow_html=True)
            st.markdown("<span style='color:grey;'>━ SH_Max Orientation</span>", unsafe_allow_html=True)
