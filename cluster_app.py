import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as kmeans
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

st.header("Report on clustering")

data = pd.read_csv("https://raw.github.com/auliadians/streamlit/master/pelanggar_akb.csv")

#display the dataframe
st.subheader("The processed dataframe")
st.dataframe(data)
#dataframe() function gives an interactive table
df = data.drop(data.columns[0], axis = 1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
#creating and fitting the model
model = kmeans(n_clusters=3, n_init=10).fit_predict(scaled_data)
#visualisation of the clusters
fig_cluster = px.scatter(data_frame=scaled_data,)
#displaying the clusters
st.subheader("Displaying the cluster")
st.plotly_chart(fig_cluster)#for showing plotly figures, we have the function plotly chart and pass our fig_cluster as the parameter
number_of_clusters = st.slider("number of clusters to display", min_value=4, max_value=48, step=2, value=10)