import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as kmeans
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

st.header("Report on clustering")

data = pd.read_csv("https://raw.githubusercontent.com/auliadians/streamlit/main/test.csv", index_col='Unnamed: 0',nrows=1000)# Unnamed: 0 is the index column present in the dataset, if its not explicitly mentioned this way, it will also get displayed
#nrows=1000 is for demo purposes, since the dataset has a large number of rows
#preprocessing the dataset
data = pd.get_dummies(data, columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class','satisfaction'])
data.drop(['id','Gender_Female','Customer Type_disloyal Customer','Type of Travel_Business travel','satisfaction_neutral or dissatisfied'
            , 'Arrival Delay in Minutes'], axis =1 , inplace=True)

#display the dataframe
st.subheader("The processed dataframe")
st.dataframe(data)
#dataframe() function gives an interactive table
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#creating and fitting the model
model = kmeans(n_clusters=2, n_init=25).fit(scaled_data)

# initialising the PCA
pca=PCA(n_components=2) #we're calling the PCA method and we're specifying that we want two components.

#fitting the PCA
principalComponents = pca.fit_transform(scaled_data)#we're fitting the pca method on our dataset

#making a dataframe of the principal components
principalDf = pd.DataFrame(data = principalComponents, columns=['principal_component_1', 'principal_component_2'])# after gettong the output we're creating a dataset

#displaying the pcs dataframe
st.subheader("The dataframe showing the 2 principal components")
st.dataframe(principalDf)

#visualisation of the clusters
fig_cluster = px.scatter(data_frame=principalDf, x='principal_component_1', y='principal_component_2', color=model.labels_)

#displaying the clusters
st.subheader("Displaying the cluster")
st.plotly_chart(fig_cluster)#for showing plotly figures, we have the function plotly chart and pass our fig_cluster as the parameter

# list to store the within sum of squared error(inertia) for the different clusters given the respective cluster size
wss = []

# loop to iterate over the no of clusters and calculate the wss
for i in range(1,11): # 1 to 10
  #kmeans
  fitx = kmeans(n_clusters=i, init='random', n_init=5, random_state=109).fit(scaled_data)
  # appending the value
  wss.append(fitx.inertia_)

#to make a plotly figure, we need a figure variable and an access variable
#making the matplotlib figure

#to make a curve that can be passed to a variable, use subplot
fig, ax = plt.subplots(figsize=(11,8.5))
plt.plot(range(1,11), wss, 'bx-')
plt.xlabel('Number of clusters $k$')
plt.ylabel('Inertia')
plt.title('The Elbow method showing the optimal $k$')

#displaying the elbow curve
st.subheader("The Elbow curve")
st.pyplot(fig)

#Let's add user interaction, where the user provides the scaled data and we give the output
#the element we use to provide this data is text, since if we need to use numbers, streamlit allows only one number at a time

st.subheader("Getting the input from the user")
input=st.text_area("Input your values : ", value = "Enter the predictor values")#text_area is used to take input

try:
    cleaned_input = [float(i.strip()) for i in input.split(", ")]
    output = model.predict(np.array(cleaned_input).reshape(1, 24))

    st.text("The closest cluster to the data is {}".format(output[0]))#text function is used to display
    #in the dashboard to directly input values, use control+enter
except:
    pass

