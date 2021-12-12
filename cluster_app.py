import streamlit as st
import pandas as pd
import joblib

model = open("kmeans.pkl","rb")
y_kmeans = joblib.load(model)

st.title("Clustering")