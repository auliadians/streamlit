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

number_of_clusters = st.slider("number of clusters to display", min_value=2, max_value=6, value=2)

# initialising the PCA
pca=PCA(n_components=2) #we're calling the PCA method and we're specifying that we want two components.
#fitting the PCA
principalComponents = pca.fit_transform(scaled_data)#we're fitting the pca method on our dataset
#making a dataframe of the principal components
principalDf = pd.DataFrame(data = principalComponents, columns=['pca1', 'pca2']





