import pandas as pd
from azure.storage.blob import BlobServiceClient
import io
import os
import streamlit as st

import plotly #TODO clean up all plotly imports
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import h3
import numpy as np

st.set_page_config(layout="wide")

mapbox_token='pk.eyJ1IjoicmZxZWQiLCJhIjoiY2t4MHBxZjE4MHU3NzJ2bnl3cmV6bzZodCJ9.qwxACnMntkPpdmBIa1zzug'
px.set_mapbox_access_token(mapbox_token)

connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

def filter_data_quarry_distance(data: pd.DataFrame, base_lat, base_lng, max_distance) -> pd.DataFrame:
    def check_distance(latlng: np.ndarray) -> bool:
        distance = h3.great_circle_distance((base_lat, base_lng), latlng, unit="km")
        return distance <= max_distance

    mask = data[["latitude", "longitude"]].apply(axis=1, raw=True, func=check_distance)

    return data[mask]


@st.cache_data(show_spinner="Fetching data from Azure...")
def get_data():
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client("globaldataset")
    blob_client = container_client.get_blob_client("cropland_only_global_data.csv")
    #blob_client = container_client.get_blob_client("regridded_data_v5.csv")
    csv_content = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(csv_content))
    #change precip to int
    df = df.astype({'precipitation':'int'})
    df = df.drop(columns=['is_soil'])
    return df

df = get_data()


with st.form(key='starting'):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        point_lat = st.number_input('Point (lat)', value=55.8852, format="%.6f")
    with col2:
        point_lon = st.number_input('Point (lon)', value=-3.8819, format="%.6f")
    with col3:
        chosen_radius = st.slider("Pick radius from the chosen point (km)", 0, 1000, 80)
    with col4:
        chosen_per_crop = st.slider("Pick a % cropland to use", 15, 100, 20)
    with col5:
        zoom_level = st.slider("Plot zoom level", 0.0, 10.0, 6.5)
    with col6:
        submitted = st.form_submit_button('Run')
        weighted_avg = st.checkbox('Calculated weighted average using cropland % weight')
        

# reduce dataframe to just around the point
df = df[df['latitude'] < point_lat + 20]
df = df[df['latitude'] > point_lat - 20]

df = df[df['longitude'] < point_lon + 20]
df = df[df['longitude'] > point_lon - 20]

# cut on cropland %
df = df[df['is_crop'] > chosen_per_crop]

#filter by distance
df = filter_data_quarry_distance(df, point_lat, point_lon, chosen_radius)

chosen_point_data = [[point_lat, point_lon, 100]]
df_chosen_point = pd.DataFrame(chosen_point_data, columns=['lat', 'lon', 'size'])

df['soil temp 0-7cm'] = df['soil_temperature_0_to_7cm']
df['soil temp 7-28cm'] = df['soil_temperature_7_to_28cm']

df['water_filled_porosity'] = ((df['soil_moisture_0_to_7cm'] + df['soil_moisture_7_to_28cm'])/2) * df['bulk_den']
df['water_filled_porosity'] = np.round(df['water_filled_porosity'],decimals = 3)
#,longitude,latitude,precipitation,temperature_2m,soil_temperature_0_to_7cm,soil_temperature_7_to_28cm,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,ph,cec,bulk_den,is_soil,is_crop

col1, col2 = st.columns(2)

with col1:
    st.write("Topsoil pH (in H2O), -log(H+)")
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='ph', zoom=zoom_level, center={"lat":point_lat, "lon":point_lon}, color_continuous_scale="inferno")
    fig2 = px.scatter_mapbox(df_chosen_point, lat='lat', lon='lon', size='size', opacity=0.9, zoom=8, center={"lat":point_lat, "lon":point_lon})
    fig.add_trace(fig2.data[0])
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0},
                       mapbox = { 'style': "mapbox://styles/rfqed/ckx0prtk02gmq15mty3tlmhpu"},
                       showlegend = False,
                       coloraxis_colorbar_title_text = 'pH')
    st.plotly_chart(fig, use_container_width=True)

    st.write("Soil Temperature 0-7cm, dC")
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='soil_temperature_0_to_7cm', zoom=zoom_level, center={"lat":point_lat, "lon":point_lon}, color_continuous_scale="inferno")
    fig2 = px.scatter_mapbox(df_chosen_point, lat='lat', lon='lon', size='size', opacity=0.9, zoom=8, center={"lat":point_lat, "lon":point_lon})
    fig.add_trace(fig2.data[0])
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0},
                       mapbox = { 'style': "mapbox://styles/rfqed/ckx0prtk02gmq15mty3tlmhpu"},
                       showlegend = False,
                       coloraxis_colorbar_title_text = '°C')
    
    st.plotly_chart(fig, use_container_width=True)

    st.write("% Cropland")
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='is_crop', zoom=zoom_level, center={"lat":point_lat, "lon":point_lon}, color_continuous_scale="inferno")
    fig2 = px.scatter_mapbox(df_chosen_point, lat='lat', lon='lon', size='size', opacity=0.9, zoom=8, center={"lat":point_lat, "lon":point_lon})
    fig.add_trace(fig2.data[0])
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0},
                       mapbox = { 'style': "mapbox://styles/rfqed/ckx0prtk02gmq15mty3tlmhpu"},
                       showlegend = False,
                       coloraxis_colorbar_title_text = '% Cropland')
    
    st.plotly_chart(fig, use_container_width=True)

    
with col2:
    st.write("Avg Yearly Precipitation (mm/yr), (Over 10yrs)")
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='precipitation', zoom=zoom_level, center={"lat":point_lat, "lon":point_lon}, color_continuous_scale="inferno")
    fig2 = px.scatter_mapbox(df_chosen_point, lat='lat', lon='lon', size='size', opacity=0.9, zoom=8, center={"lat":point_lat, "lon":point_lon})
    fig.add_trace(fig2.data[0])
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0},
                       mapbox = { 'style': "mapbox://styles/rfqed/ckx0prtk02gmq15mty3tlmhpu"},
                       showlegend = False,
                       coloraxis_colorbar_title_text = 'mm/yr')
    
    st.plotly_chart(fig, use_container_width=True)

    st.write("Soil Moisture 0-7cm, dC")
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='soil_moisture_0_to_7cm', zoom=zoom_level, center={"lat":point_lat, "lon":point_lon}, color_continuous_scale="inferno")
    fig2 = px.scatter_mapbox(df_chosen_point, lat='lat', lon='lon', size='size', opacity=0.9, zoom=8, center={"lat":point_lat, "lon":point_lon})
    fig.add_trace(fig2.data[0])
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0},
                       mapbox = { 'style': "mapbox://styles/rfqed/ckx0prtk02gmq15mty3tlmhpu"},
                       showlegend = False,
                       coloraxis_colorbar_title_text = '%')
    
    st.plotly_chart(fig, use_container_width=True)


    st.write("Water Filled Porosity")
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='water_filled_porosity', zoom=zoom_level, center={"lat":point_lat, "lon":point_lon}, color_continuous_scale="inferno")
    fig2 = px.scatter_mapbox(df_chosen_point, lat='lat', lon='lon', size='size', opacity=0.9, zoom=8, center={"lat":point_lat, "lon":point_lon})
    fig.add_trace(fig2.data[0])
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0},
                       mapbox = { 'style': "mapbox://styles/rfqed/ckx0prtk02gmq15mty3tlmhpu"},
                       showlegend = False,
                       coloraxis_colorbar_title_text = 'L_pw / L_soil')
    
    st.plotly_chart(fig, use_container_width=True)



df['CEC_eqL'] = (df['cec'] / 100000) * (df['bulk_den'] / df['water_filled_porosity']) * 1000

st.header("Final values from dataset")

vars_to_plot = ['ph', 'bulk_den', 'precipitation', 'soil_temperature_0_to_7cm' , 'soil_temperature_7_to_28cm', 'soil_moisture_0_to_7cm' , 'soil_moisture_7_to_28cm', 'cec', 'is_crop', 'water_filled_porosity'] 
    
fig = make_subplots(rows=1, cols=len(vars_to_plot))
for i, var in enumerate(vars_to_plot):
    fig.add_trace(go.Box(y=df[var], name=var, boxmean=True),row=1, col=i+1)

fig.update_traces(boxpoints='all', jitter=.3)
fig.update_layout(height=750)
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

with col1:
  st.metric("Mean ph", round(df['ph'].mean(),2), "", delta_color="off")

with col2:
  st.metric("Mean bulk_den", round(df['bulk_den'].mean(),2), "kg dm-3", delta_color="off")

with col3:
  st.metric("Mean Precip", round(df['precipitation'].mean(),0), "mm/yr", delta_color="off")

with col4:
  st.metric("Mean soil_temperature_0_to_7cm", round(df['soil_temperature_0_to_7cm'].mean(),2), "°C", delta_color="off")

with col5:
  st.metric("Mean soil_temperature_7_to_28cm", round(df['soil_temperature_7_to_28cm'].mean(),2), "°C", delta_color="off")
  
with col6:
  st.metric("Mean soil_moisture_0_to_7cm", round(df['soil_moisture_0_to_7cm'].mean(),2), "°C", delta_color="off")

with col7:
  st.metric("Mean soil_moisture_7_to_28cm", round(df['soil_moisture_7_to_28cm'].mean(),2), "°C", delta_color="off")

with col8:
  st.metric("Mean CEC cmol/kg", round(df['cec'].mean(),2), "cmol per kg", delta_color="off")
  st.metric("Mean CEC eqL", round(df['CEC_eqL'].mean(),2), "eqL", delta_color="off")
with col9:
  st.metric("Mean water filled porosity", round(df['water_filled_porosity'].mean(),2), "L porewater / L soil", delta_color="off")



st.header("Mean values from dataset, weighted by crop %")
ph_mean_weighted = (df['ph']*df['is_crop']).sum()/df['is_crop'].sum()
bulk_den_mean_weighted = (df['bulk_den']*df['is_crop']).sum()/df['is_crop'].sum()
precip_mean_weighted = (df['precipitation']*df['is_crop']).sum()/df['is_crop'].sum()
soil_temp_mean_weighted0_7 = (df['soil_temperature_0_to_7cm']*df['is_crop']).sum()/df['is_crop'].sum()
soil_temp_mean_weighted7_28 = (df['soil_temperature_7_to_28cm']*df['is_crop']).sum()/df['is_crop'].sum()
soil_moisture_mean_weighted0_7 = (df['soil_moisture_0_to_7cm']*df['is_crop']).sum()/df['is_crop'].sum()
soil_moisture_mean_weighted7_28 = (df['soil_moisture_7_to_28cm']*df['is_crop']).sum()/df['is_crop'].sum()
cec_weighted = (df['cec']*df['is_crop']).sum()/df['is_crop'].sum()
cec_eqL_weighted = (df['cec_eqL']*df['is_crop']).sum()/df['is_crop'].sum()
water_filled_porosity_weighted = (df['water_filled_porosity']*df['is_crop']).sum()/df['is_crop'].sum()

col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

with col1:
  st.metric("Weighted mean ph", round(ph_mean_weighted,2), "", delta_color="off")

with col2:
  st.metric("Weighted bulk_den", round(bulk_den_mean_weighted,2), "kg dm-3", delta_color="off")

with col3:
  st.metric("Weighted Precip", round(precip_mean_weighted, 0), "mm/yr", delta_color="off")

with col4:
  st.metric("Weighted soil_temperature_0_to_7cm", round(soil_temp_mean_weighted0_7, 2), "°C", delta_color="off")

with col5:
  st.metric("Weighted soil_temperature_7_to_28cm", round(soil_temp_mean_weighted7_28, 2), "°C", delta_color="off")
  
with col6:
  st.metric("Weighted soil_moisture_0_to_7cm", round(soil_moisture_mean_weighted0_7, 2), "m3 m-3", delta_color="off")

with col7:
  st.metric("Weighted soil_moisture_7_to_28cm", round(soil_moisture_mean_weighted7_28, 2), "m3 m-3", delta_color="off")

with col8:
  st.metric("Weighted CEC", round(cec_weighted, 2), "cmol per kg", delta_color="off")
  st.metric("Weighted CEC", round(cec_eqL_weighted,2), "eqL", delta_color="off")
  
with col9:
  st.metric("Weighted water filled porosity", round(water_filled_porosity_weighted, 2), "L porewater / L soil", delta_color="off")
