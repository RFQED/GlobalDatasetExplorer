import pandas as pd
from azure.storage.blob import BlobServiceClient
import io
import os
import streamlit as st
import plotly.express as px
from math import sin, cos, sqrt, atan2, radians, pi
import plotly

st.set_page_config(layout="wide")

mapbox_token='pk.eyJ1IjoicmZxZWQiLCJhIjoiY2t4MHBxZjE4MHU3NzJ2bnl3cmV6bzZodCJ9.qwxACnMntkPpdmBIa1zzug'
px.set_mapbox_access_token(mapbox_token)

connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

def getDist(lat1,lon1,lat2,lon2):
  R = 6373.0
  lat1 = radians(lat1)
  lon1 = radians(lon1)
  lat2 = radians(lat2)
  lon2 = radians(lon2)

  dlon = lon2 - lon1
  dlat = lat2 - lat1

  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))

  return R * c


@st.cache_data(show_spinner="Fetching data from Azure...")
def get_data():
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client("globaldataset")
    blob_client = container_client.get_blob_client("regridded_data_v5.csv")
    csv_content = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(csv_content))
    return df


df = get_data()


col1, col2, col3, col4 = st.columns(4)
with col1:
    point_lat = st.number_input('Point (lat)', value=55.8852, format="%.6f")
with col2:
    point_lon = st.number_input('Point (lon)', value=-3.8819, format="%.6f")
with col3:
    chosen_radius = st.slider("Pick radius from the chosen point (km)", 0, 800, 80)
with col4:
    chosen_per_crop = st.slider("Pick a % cropland to use", 0, 100, 25)
    

#make a bounding box around this lat_lon
df = df[df['latitude'] < point_lat + 10]
df = df[df['latitude'] > point_lat - 10]

df = df[df['longitude'] < point_lon + 10]
df = df[df['longitude'] > point_lon - 10]


#Apply distance function to dataframe
df['dist']=list(map(lambda k: getDist(df.loc[k]['latitude'],df.loc[k]['longitude'], point_lat, point_lon), df.index))
df = df[df['dist'] < chosen_radius]

df = df[df['is_crop'] > chosen_per_crop]

r_earth = 6373.0
lat_low  = point_lat - (chosen_radius / r_earth) * (180 / pi);
lat_high   = point_lat + (chosen_radius / r_earth) * (180 / pi);

lon_low = point_lon - (chosen_radius / r_earth) * (180 / pi) / cos(point_lat * pi/180);
lon_high = point_lon + (chosen_radius / r_earth) * (180 / pi) / cos(point_lat * pi/180);
    
mid_lat = (lat_low + lat_high) / 2
mid_lon = (lon_low + lon_high) / 2

chosen_point_data = [[point_lat, point_lon, 100]]
df_chosen_point = pd.DataFrame(chosen_point_data, columns=['lat', 'lon', 'size'])


colph, colbulkdensity = st.columns(2)

with colph:
    st.write("Topsoil pH (in H2O), -log(H+)")
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='ph', zoom=6.5, center={"lat":mid_lat, "lon":mid_lon}, continous_color_scale=plotly.express.colors.sequential)
    #fig2 = px.scatter_mapbox(df_chosen_point, lat='lat', lon='lon', size='size', opacity=0.9, zoom=8, center={"lat":mid_lat, "lon":mid_lon})
    #fig.add_trace(fig2.data[0])
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0},
                            mapbox = { 'style': "mapbox://styles/rfqed/ckx0prtk02gmq15mty3tlmhpu"},
                            showlegend = False)
    st.plotly_chart(fig, use_container_width=True)
    # Show pH plot

with colbulkdensity:
    # Plot SOIL BULK DENSITY data from soil data  #54.65270979260174, -2.392090808630309
    st.write("Topsoil bulk density, kg/dm-3")
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='bulk_den', zoom=6.5, center={"lat":mid_lat, "lon":mid_lon})
    fig2 = px.scatter_mapbox(df_chosen_point, lat='lat', lon='lon', size='size', opacity=0.9, zoom=8, center={"lat":mid_lat, "lon":mid_lon})
    fig.add_trace(fig2.data[0])
    fig.update_layout( margin={"r":0,"t":0,"l":0,"b":0},
                            mapbox = { 'style': "mapbox://styles/rfqed/ckx0prtk02gmq15mty3tlmhpu"},
                            showlegend = False)
    
    st.plotly_chart(fig, use_container_width=True)
