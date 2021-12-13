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

model = kmeans(n_clusters=2, n_init=10).fit(scaled_data)

# initialising the PCA
pca=PCA(n_components=2) #we're calling the PCA method and we're specifying that we want two components.
#fitting the PCA
principalComponents = pca.fit_transform(scaled_data)#we're fitting the pca method on our dataset
#making a dataframe of the principal components
principalDf = pd.DataFrame(data = principalComponents, columns=['pca1', 'pca2'])

#visualisation of the clusters
fig_cluster = px.scatter(data_frame=principalDf, x='pca1', y='pca2')

#displaying the clusters
st.subheader("Displaying the cluster")
st.plotly_chart(fig_cluster)

# list to store the within sum of squared error(inertia) for the different clusters given the respective cluster size
wss = []

# loop to iterate over the no of clusters and calculate the wss
for i in range(1,7): # 1 to 10
  #kmeans
  fitx = kmeans(n_clusters=i, init='random', n_init=5, random_state=42).fit(scaled_data)
  # appending the value
  wss.append(fitx.inertia_)

#to make a curve that can be passed to a variable, use subplot
fig, ax = plt.subplots(figsize=(11,8.5))
plt.plot(range(1,7), wss, 'bx-')
plt.xlabel('Number of clusters $k$')
plt.ylabel('Inertia')
plt.title('The Elbow method showing the optimal $k$')

#displaying the elbow curve
st.subheader("The Elbow curve")
st.pyplot(fig)

st.subheader("Getting the input from the user")
input=st.text_area("Input your values : ", value = "Enter the predictor values")#text_area is used to take input

try:
    cleaned_input = [float(i.strip()) for i in input.split(", ")]
    output = model.predict(np.array(cleaned_input).reshape(1, 24))

    st.text("The closest cluster to the data is {}".format(output[0]))#text function is used to display
    #in the dashboard to directly input values, use control+enter
except:
    pass

