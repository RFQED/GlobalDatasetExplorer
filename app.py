import pandas as pd
from azure.storage.blob import BlobServiceClient
import io
import os
import streamlit as st

connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

@st.cache_data
def get_data():
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client("globaldataset")
    blob_client = container_client.get_blob_client("regridded_data_v5.csv")
    csv_content = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(csv_content))
    return df


df = get_data()
st.dataframe(df)
