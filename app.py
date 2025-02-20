import os
import streamlit as st
import pandas as pd
from pipeline import MLPipeline
from utils import load_data

st.title("AutoML Web App with PyCaret")
st.write("Upload your dataset to find the best model!")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if df.empty:
            st.error("⚠️ The uploaded dataset is empty. Please upload a valid CSV file.")
    else:
        st.write("### 📊 Data Preview")
        st.dataframe(df.head())

        # Initialize ML Pipeline
        pipeline = MLPipeline(data=df)
        
        pipeline.clean_data()

        # Select Target Variable
        target = st.selectbox("Select the Target Column", pipeline.data.columns)
        pipeline.target=target

        # Perform EDA
        if st.button("🔍 Perform EDA"):
            pipeline.perform_eda()

        # Train Models
        model_options = ['All Models', 'lr', 'dt', 'rf', 'xgboost', 'lightgbm','knn','ada','svm','lasso','ridge','gbr','et','catboost']
        selected_models = st.multiselect(" Select Models to Compare", model_options, default=['All Models'])
        pipeline.preprocess_data()
        
        if st.button("🚀 Train Models"):
            if 'All Models' in selected_models:
                selected_models = None
            best_model = pipeline.train_model(selected_models)

            model_path = "models/best_model.pkl"

            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    st.download_button("📥 Download Best Model", f, file_name="best_model.pkl")
            else:
                st.error("⚠️ Model file not found! Train and save the model first.")

