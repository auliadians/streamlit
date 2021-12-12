import streamlit as st
import pandas as pd
import joblib

model = open("clustering.pkl","rb")
y_kmeans = joblib.load(model)

st.title("Clustering")