import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.impute import SimpleImputer
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, save_model as cls_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
class MLPipeline:
    def __init__(self, data):
        self.data = data
        self.target = None
        

    def clean_data(self):
        """Handles missing values, drops constant and high-cardinality columns."""
        # Drop constant columns
        constant_columns = [col for col in self.data.columns if self.data[col].nunique(dropna=False) == 1]

        # Drop high-cardinality columns
        high_cardinality_columns = [col for col in self.data.columns if self.data[col].nunique() > 0.95 * len(self.data)]

        # Drop columns with all NaN values
        all_nan_columns = [col for col in self.data.columns if self.data[col].isnull().all()]

        # Drop selected columns
        columns_to_drop = list(set(constant_columns + high_cardinality_columns + all_nan_columns))
        self.data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Handle missing values
        mean_imputer = SimpleImputer(strategy='mean')
        mode_imputer = SimpleImputer(strategy='most_frequent')

        for col in self.data.columns:
            if self.data[col].isna().any():
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    self.data[col] = mean_imputer.fit_transform(self.data[[col]]).ravel()
                else:
                    self.data[col] = mode_imputer.fit_transform(self.data[[col]]).ravel()
        return self.data
    
    def preprocess_data(self):
        """Encodes categorical features and scales numerical features, excluding the target column."""
        
        num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target is None:
            raise ValueError("Target variable must be set before preprocessing.")
        
        # Ensure target is not included in processing
        if self.target in num_cols:
            num_cols.remove(self.target)
        if self.target in cat_cols:
            cat_cols.remove(self.target)

        # Encode categorical variables
        for col in cat_cols:
            if self.data[col].nunique() <= 10:
                one_hot = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                self.data = pd.concat([self.data, one_hot], axis=1)
            else:
                self.data[col] = LabelEncoder().fit_transform(self.data[col].astype(str)) 

        self.data.drop(columns=cat_cols, inplace=True)

        scaler = StandardScaler()
        self.data[num_cols] = scaler.fit_transform(self.data[num_cols])

        return self.data


    def determine_problem_type(self):
        """Determine if the problem is classification or regression."""
        if self.target is None:
            raise ValueError("Target variable is not set.")
        if self.data[self.target].dtype == 'object' or self.data[self.target].nunique() < 10:
            return "classification"
        return "regression"
    
    def perform_eda(self):
        """Generate basic statistics and visualizations."""
        st.subheader("Summary Statistics")
        st.write(self.data.describe())

        st.subheader("Target Variable Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.data[self.target], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(self.data[numerical_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    def train_model(self, models_to_compare=None):
        """Train models and return the best one."""
        st.subheader("Training Models...")

        # Ensure the 'models' directory exists before saving
        os.makedirs("models", exist_ok=True)

        # Debugging: Print dataset info before setup
        st.write("Checking missing values after handling:")
        st.write(self.data.isnull().sum())

        problem_type = self.determine_problem_type()
        try:
            if problem_type == "classification":
                cls_setup(data=self.data, target=self.target, verbose=False)
                best_model = cls_compare(include=models_to_compare)
                cls_save(best_model, "models/best_model")
            else:
                reg_setup(data=self.data, target=self.target, verbose=False)
                best_model = reg_compare(include=models_to_compare)
                reg_save(best_model, "models/best_model")

            st.success(f"Best Model: {best_model}")
            return best_model
        except Exception as e:
            st.error(f"⚠️ Error during training: {e}")
            return None

