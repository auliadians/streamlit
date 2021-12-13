import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as kmeans
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

st.header("Report on clustering")

data = pd.read_csv("https://raw.github.com/auliadians/streamlit/master/pelanggar_akb.csv")

#display the dataframe
st.subheader("The processed dataframe")
st.dataframe(data)
#dataframe() function gives an interactive table
df = data.drop(data.columns[0], axis = 1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

number_of_clusters = st.slider("number of clusters to display", min_value=2, max_value=6, value=3)
#source_df = number_of_clusters     
#source = source_df
#st.write("This Silhuette Score for ", number_of_clusters  )
#creating and fitting the model
model = kmeans(n_clusters=number_of_clusters, n_init=10).fit_predict(scaled_data)

chart = alt.Chart(model).mark_circle(size=60).encode(
        color=alt.Color('Jalan', scale=alt.Scale(scheme='category20')) ,
        tooltip=['Jalan','Pelanggar Roda 3']
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    ).properties(
        width=700,
        height=500,
    ).interactive()

#visualisation of the clusters
fig_cluster = px.scatter(data_frame=scaled_data,)
#displaying the clusters
st.subheader("Displaying the cluster")
st.plotly_chart(fig_cluster)#for showing plotly figures, we have the function plotly chart and pass our fig_cluster as the parameter